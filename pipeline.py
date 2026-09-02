"""
pipeline.py
-----------
DocuParse AI — the 4-step engine.

  [1: Ingestion & Prep] -> [2: Serialization] -> [3: Intent Parsing] -> [4: Structuring]

Unlike a Tesseract-based pipeline, this never runs character-level OCR at
all. The raw image is handed directly to a vision-capable LLM (gpt-4o-mini),
which reads the document the way a human would — using layout, position,
and visual hierarchy to understand that the number next to "Total:" is the
total, not just extracting characters and hoping regex can guess the intent.
"""

import base64
import json
import os
from datetime import datetime
from pathlib import Path

from dateutil import parser as date_parser

try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


SUPPORTED_EXTS = {".pdf", ".jpg", ".jpeg", ".png"}

# Step 3 payload. response_format={"type": "json_object"} (set on the API
# call itself, not here) is what actually forces JSON-only output; this
# prompt defines the schema and the "don't guess" rule inside it.
EXTRACTION_PROMPT = """You are an invoice/receipt data extraction engine.
Look at this document image and extract exactly these fields as a JSON object:

{
  "invoice_number": string or null,
  "vendor_name": string or null,
  "invoice_date": string or null,
  "total_amount": number or null,
  "currency": string or null
}

RULES:
- Use the document's visual layout to understand meaning, not just read characters.
  If a number sits next to a label like "Total", "Total Due", "Amount Due", or
  "Grand Total", that is the total_amount — don't confuse it with a subtotal,
  tax line, or line-item price.
- Do NOT invent, guess, or infer any value not actually visible in the image.
  If a field cannot be found with confidence, return null for it.
- total_amount must be a plain number with no currency symbols or commas.
- invoice_date should be normalized to YYYY-MM-DD if you can confidently
  determine it; otherwise return the raw text you see.
- Return ONLY the JSON object. No explanation, no markdown fences.
"""


# ---------------------------------------------------------------------------
# Step 1: Ingestion & Format Standardization
# ---------------------------------------------------------------------------

def list_supported_files(input_dir: Path):
    """Every .pdf/.jpg/.jpeg/.png in the folder, in a stable order."""
    return sorted(p for p in input_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)


def ensure_image_path(path: Path, tmp_dir: Path) -> Path:
    """
    If the file is already an image, return it unchanged. If it's a PDF,
    rasterize the FIRST page to a JPEG in tmp_dir and return that path.
    (First page only — see README 'Known limitations' for why.)
    """
    if path.suffix.lower() != ".pdf":
        return path

    if convert_from_path is None:
        raise RuntimeError("pdf2image is required for PDF input (needs poppler-utils installed)")

    pages = convert_from_path(str(path), dpi=200, first_page=1, last_page=1)
    if not pages:
        raise RuntimeError(f"Could not rasterize any page from {path.name}")

    out_path = tmp_dir / f"{path.stem}_page1.jpg"
    pages[0].save(out_path, "JPEG")
    return out_path


# ---------------------------------------------------------------------------
# Step 2: Base64 Serialization
# ---------------------------------------------------------------------------

def image_to_data_uri(image_path: Path) -> str:
    """Read an image file and return it as a data:image/...;base64,... URI
    — the format the Vision API expects instead of a local file path."""
    ext = image_path.suffix.lower().lstrip(".")
    mime = "jpeg" if ext == "jpg" else ext
    raw_bytes = image_path.read_bytes()
    b64_str = base64.b64encode(raw_bytes).decode("utf-8")
    return f"data:image/{mime};base64,{b64_str}"


# ---------------------------------------------------------------------------
# Step 3: Intent Parsing & Context Extraction (the LLM node)
# ---------------------------------------------------------------------------

def get_client():
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Run: pip install openai")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set the OPENAI_API_KEY environment variable first.")
    return OpenAI(api_key=api_key)


def call_vision_model(client, data_uri: str, model: str = "gpt-4o-mini") -> str:
    """Sends the image + prompt to the model. temperature=0 and a JSON-object
    response format are the two levers that most directly reduce
    hallucination and free-text drift — see README 'Hallucination control'."""
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": EXTRACTION_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
        max_tokens=500,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Step 4: Data Matrix Tabulation & Cleaning
# ---------------------------------------------------------------------------

def _clean_amount(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        cleaned = str(value).replace(",", "").replace("$", "").replace("₹", "").strip()
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def _normalize_date(value):
    """Uses dateutil's fuzzy parser to handle whatever format the model
    returns (it's instructed to normalize, but this is a code-level safety
    net for cases like "2nd Sept '26" that the model returns unnormalized)."""
    if not value:
        return None
    try:
        parsed = date_parser.parse(str(value), fuzzy=True, dayfirst=False)
        return parsed.strftime("%Y-%m-%d")
    except (ValueError, OverflowError):
        return str(value)  # keep the raw text rather than silently dropping it


def parse_model_response(raw_response: str) -> dict:
    """Turn the model's JSON string into a cleaned dict. Any parsing failure
    becomes an explicit error field — never a silently wrong row."""
    try:
        cleaned = raw_response.replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned)
    except (json.JSONDecodeError, AttributeError):
        return {
            "invoice_number": None, "vendor_name": None, "invoice_date": None,
            "total_amount": None, "currency": None,
            "error": "Model did not return valid JSON",
        }

    return {
        "invoice_number": data.get("invoice_number"),
        "vendor_name": data.get("vendor_name"),
        "invoice_date": _normalize_date(data.get("invoice_date")),
        "total_amount": _clean_amount(data.get("total_amount")),
        "currency": data.get("currency"),
        "error": None,
    }


def process_single_file(client, path: Path, tmp_dir: Path, model: str = "gpt-4o-mini") -> dict:
    """Full pipeline for one file: ingest -> serialize -> extract -> structure."""
    try:
        image_path = ensure_image_path(path, tmp_dir)
        data_uri = image_to_data_uri(image_path)
        raw_response = call_vision_model(client, data_uri, model=model)
        record = parse_model_response(raw_response)
    except Exception as exc:
        record = {
            "invoice_number": None, "vendor_name": None, "invoice_date": None,
            "total_amount": None, "currency": None,
            "error": f"Processing failed: {exc}",
        }
    record["source_file"] = path.name
    record["processed_at"] = datetime.now().isoformat(timespec="seconds")
    return record
