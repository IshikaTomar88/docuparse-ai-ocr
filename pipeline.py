"""
pipeline.py
-----------
Invoice/receipt extraction using Google Gemini's vision capability.

Migrated to the `google-genai` SDK (the current, actively-developed
package) instead of `google.generativeai`, which Google has deprecated —
see https://github.com/google-gemini/deprecated-generative-ai-python and
https://ai.google.dev/gemini-api/docs/migrate. The old package still runs
today, but new features only land in the new SDK and it's the one Google
tells you to be on going forward.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

from dateutil import parser as date_parser
from PIL import Image

try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


SUPPORTED_EXTS = {".pdf", ".jpg", ".jpeg", ".png"}

EXTRACTION_PROMPT = """Analyze this invoice/receipt image and extract exactly
these fields as a JSON object:

{
  "invoice_number": string or null,
  "vendor_name": string or null,
  "invoice_date": string or null,
  "total_amount": number or null,
  "currency": string or null
}

RULES:
- Use the document's visual layout to understand meaning, not just read
  characters. If a number sits next to a label like "Total", "Total Due",
  "Amount Due", or "Grand Total", that is the total_amount — don't confuse
  it with a subtotal, tax line, or line-item price.
- Do NOT invent, guess, or infer any value not actually visible in the
  image. If a field cannot be found with confidence, return null for it.
- total_amount must be a plain number with no currency symbols or commas.
- invoice_date should be normalized to YYYY-MM-DD if you can confidently
  determine it; otherwise return the raw text you see.
- Return ONLY the JSON object. No explanation, no markdown fences.
"""


def get_client():
    """Creates a genai.Client using GEMINI_API_KEY. Named correctly this
    time — the original version of this code stored the Gemini key under
    OPENAI_API_KEY, which only worked because it was self-consistent; it
    would silently break the moment someone also had a real OpenAI key set."""
    if genai is None:
        raise RuntimeError("google-genai package not installed. Run: pip install google-genai")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set the GEMINI_API_KEY environment variable first.")
    return genai.Client(api_key=api_key)


def convert_pdf_page_to_jpeg(pdf_path: Path, output_dir: Path) -> Path:
    """Rasterizes the first page of a PDF to a JPEG.
    Filename is derived from the source file's own name (e.g.
    "invoice_042_render.jpg"), not a fixed "temp_render.jpg" — the
    original version reused one fixed filename across every PDF in a
    batch, which works only by accident today (strictly sequential
    processing) and would silently corrupt results the moment anyone
    parallelizes this loop."""
    if convert_from_path is None:
        raise RuntimeError("pdf2image is required for PDF input (needs poppler-utils installed)")

    pages = convert_from_path(str(pdf_path), dpi=200, first_page=1, last_page=1)
    if not pages:
        raise RuntimeError(f"Could not rasterize any page from {pdf_path.name}")

    jpeg_path = output_dir / f"{pdf_path.stem}_render.jpg"
    pages[0].save(jpeg_path, "JPEG")
    return jpeg_path


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


_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _normalize_date(value):
    """Normalizes a date string to YYYY-MM-DD, as a code-level safety net
    for when the model doesn't comply with the prompt's format instruction.

    Two things verified by direct testing, not assumed:
    1. An ISO string (the exact format the model is asked to return) is
       matched and validated directly, WITHOUT going through dateutil's
       ambiguous-date resolution — testing showed dateutil's dayfirst
       flag can misinterpret an already-correct "2026-09-02" as day=09/
       month=02 ("Feb 9"), corrupting the model's own correct output in
       the common case where it complied with the prompt.
    2. For everything else (the model didn't comply, or returned a
       non-ISO format), fuzzy parsing falls back to dayfirst=True — this
       project's stated target market is Indian clients, where numeric
       dates are conventionally DD/MM/YYYY. This is a real, unavoidable
       ambiguity for a global freelance tool: a bare "09/02/2026" is
       genuinely ambiguous without a locale to anchor it. dayfirst=True
       is the documented, deliberate choice for this market — swap to
       dayfirst=False if you're serving a US-only client base instead.
    """
    if not value:
        return None
    text = str(value).strip()

    if _ISO_DATE_RE.match(text):
        try:
            datetime.strptime(text, "%Y-%m-%d")
            return text
        except ValueError:
            pass  # malformed pseudo-ISO (e.g. month 13) — fall through below

    try:
        parsed = date_parser.parse(text, fuzzy=True, dayfirst=True)
        return parsed.strftime("%Y-%m-%d")
    except (ValueError, OverflowError):
        return text  # keep the raw text rather than silently dropping it


def parse_model_response(raw_text: str) -> dict:
    """Parses the model's response into a cleaned dict. Strips markdown
    fences as a safety net even though response_mime_type='application/json'
    should prevent them, and turns any parse failure into an explicit
    error field rather than a silently wrong row.

    Also guards against valid-JSON-but-wrong-shape responses (a bare
    `null`, a list, a number, a string) — verified by direct testing that
    without this guard, `data.get(...)` raises an uncaught AttributeError
    for any of those cases."""
    def _error(message):
        return {
            "invoice_number": None, "vendor_name": None, "invoice_date": None,
            "total_amount": None, "currency": None,
            "error": message,
        }

    try:
        cleaned = (raw_text or "").replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return _error("Model did not return valid JSON")

    if not isinstance(data, dict):
        return _error(f"Model returned valid JSON but not an object (got {type(data).__name__})")

    return {
        "invoice_number": data.get("invoice_number"),
        "vendor_name": data.get("vendor_name"),
        "invoice_date": _normalize_date(data.get("invoice_date")),
        "total_amount": _clean_amount(data.get("total_amount")),
        "currency": data.get("currency"),
        "error": None,
    }


def process_single_file(client, file_path: Path, tmp_path: Path, model: str = "gemini-3.6-flash") -> dict:
    """Full pipeline for one file. `client` is now actually used (a real
    genai.Client instance) instead of being accepted and then ignored in
    favor of a global module call, as the original version did."""
    record = {
        "source_file": Path(file_path).name,
        "invoice_number": None, "vendor_name": None, "invoice_date": None,
        "total_amount": None, "currency": None, "error": None,
        "processed_at": None,  # set at the end — see below, not here
    }

    file_path = Path(file_path)
    working_image_path = file_path

    if file_path.suffix.lower() == ".pdf":
        try:
            working_image_path = convert_pdf_page_to_jpeg(file_path, tmp_path)
        except Exception as exc:
            record["error"] = f"PDF conversion failed: {exc}"
            record["processed_at"] = datetime.now().isoformat(timespec="seconds")
            return record

    try:
        image = Image.open(working_image_path)
        response = client.models.generate_content(
            model=model,
            contents=[EXTRACTION_PROMPT, image],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )
        parsed = parse_model_response(response.text)
        record.update(parsed)
    except Exception as exc:
        record["error"] = str(exc)

    # Timestamp reflects when processing actually finished, not when the
    # function started — the original version stamped this before any
    # work happened, which is misleading for anything that takes real time
    # (PDF rasterization + a network call).
    record["processed_at"] = datetime.now().isoformat(timespec="seconds")
    return record
