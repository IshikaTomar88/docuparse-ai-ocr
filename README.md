# 🧠 DocuParse AI

Invoice/receipt extraction that reads documents the way a human does —
by *looking* at layout and context, not by extracting characters and
guessing with regex.

**Live demo:** _add your Streamlit Cloud link here after deploying_
**Stack:** Python · GPT-4o-mini Vision · pdf2image · Pandas · Streamlit

---

## Why this is different from an OCR+regex pipeline

Traditional OCR (Tesseract etc.) only reads characters. If an invoice says
`Total: $500`, plain OCR gives you the text `Total: $500` and you're left
writing regex to guess that `500` is the total and not a line-item price,
a subtotal, or a tax amount.

This project skips character-level OCR entirely. The raw invoice image is
sent directly to a vision-capable LLM (`gpt-4o-mini`), which understands
the document's *visual hierarchy* — it can tell that the number next to
`Grand Total` is the one that matters, the same way a person scanning the
page would.

## The 4-step pipeline

```
[1: Ingestion & Prep] → [2: Base64 Serialization] → [3: Intent Parsing (LLM)] → [4: Structuring]
```

**Step 1 — Ingestion & Format Standardization** (`ensure_image_path`)
Looks inside the input folder. `.jpg`/`.png` files pass through untouched.
`.pdf` files have their **first page** rasterized to a JPEG via
`pdf2image` (backed by `poppler`).

**Step 2 — Base64 Serialization** (`image_to_data_uri`)
The vision API can't read a local file path — it needs the image sent as
data. The image is opened in binary mode, base64-encoded, and wrapped in a
`data:image/jpeg;base64,...` URI.

**Step 3 — Intent Parsing & Context Extraction** (`call_vision_model`)
The data URI + a strict instruction prompt go to `gpt-4o-mini` with
`response_format={"type": "json_object"}` and `temperature=0.0`. The model
returns a JSON object with exactly five keys — no conversational fluff.

**Step 4 — Data Matrix Tabulation & Cleaning** (`parse_model_response` +
`process_single_file`)
The JSON string is parsed into a dict, `total_amount` is stripped of
currency symbols/commas and coerced to a float, `invoice_date` is run
through a fuzzy date parser to normalize whatever format the model
returned, `source_file` and `processed_at` are injected, and every row is
combined into a Pandas DataFrame → Excel/CSV.

## Hallucination control

- **`temperature=0.0`** — minimizes creative/random variation in the model's output.
- **`response_format={"type": "json_object"}`** — the API rejects
  non-JSON completions outright; the model can't wander into prose.
- **Explicit "return null, never guess" instruction** in the prompt itself —
  a missing field becomes `null`, not a fabricated plausible-looking value.
- **Every response is parsed, not trusted.** `parse_model_response()` runs
  the raw text through `json.loads()`; if that fails, the row gets an
  `error` field instead of silently passing bad data through.
- **`total_amount` is re-validated in code**, not just trusted from the
  model — `_clean_amount()` rejects anything that can't become a real float.
- **Every row carries an `error` column** in the final export, so a human
  reviewer can spot-check exactly which documents need a second look
  instead of blindly trusting 100% of the output.

## What was actually verified vs. what wasn't (read this before trusting the repo)

Being direct about this rather than letting a working demo imply more than
it proves:

**Verified, with real code run against a real generated PDF invoice:**
- ✅ PDF → first-page rasterization → JPEG (`pdf2image`/`poppler`) — confirmed a real 400×500 PDF produces a real ~49KB JPEG.
- ✅ Image → base64 data URI serialization — confirmed correct `data:image/jpeg;base64,...` format and non-empty payload.
- ✅ JSON response parsing, including realistic edge cases: markdown-fenced JSON, partial `null` fields, and non-JSON garbage responses (each produces the correct row or the correct `error`).
- ✅ Amount cleaning across `$500`, `₹14,500.50`, `1200.00`, raw numbers, `None`, and garbage strings.
- ✅ Date normalization across `"2nd Sept '26"`, `"09/02/2026"`, `"2026-09-02"`, `"Sept 2, 2026"`, and unparseable text.
- ✅ Full pipeline glue (`process_single_file`) end-to-end, with only the actual OpenAI network call stubbed out.

**NOT verified — you should test this yourself before client delivery:**
- ❌ **The actual `gpt-4o-mini` vision call.** This repo was built in a sandbox without network access to `api.openai.com`, so the real extraction accuracy — how well the model actually reads a messy, real-world invoice photo — has not been tested end-to-end. Run it against a handful of your own real invoices before you trust it or pitch it.

## Setup

1. **Get an OpenAI API key** with access to `gpt-4o-mini` (platform.openai.com).

2. **Install poppler** (needed by `pdf2image` for PDF input):
   - macOS: `brew install poppler`
   - Ubuntu/Debian: `sudo apt-get install poppler-utils`
   - Windows: download a poppler build and add it to `PATH`.

3. **Install Python dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Run it:**
   ```bash
   export OPENAI_API_KEY="your-key-here"
   streamlit run app.py
   ```

5. **Or use it headless** (no UI needed — for batch scripts):
   ```python
   from pathlib import Path
   from pipeline import get_client, process_single_file

   client = get_client()
   record = process_single_file(client, Path("invoices/your_invoice.pdf"), Path("/tmp"))
   print(record)
   ```

## Deploy it (free, Streamlit Community Cloud)

1. Push this repo to your own GitHub.
2. [share.streamlit.io](https://share.streamlit.io) → New app → this repo, main file `app.py`.
3. Advanced settings → Secrets:
   ```toml
   OPENAI_API_KEY = "your-key-here"
   ```
4. Deploy — `packages.txt` installs `poppler-utils` automatically.

## Project structure

```
docuparse-ai/
├── app.py            # Streamlit UI
├── pipeline.py         # the 4-step engine (ingestion, serialization, LLM call, structuring)
├── utils.py             # DataFrame -> Excel/CSV export helpers
├── requirements.txt
├── packages.txt          # poppler-utils for Streamlit Cloud
└── invoices/              # drop sample invoices here for local testing
```

## Known limitations

- **First page only** for multi-page PDFs — a second-page continuation or
  a total that only appears on a later page won't be seen.
- **No retry/backoff logic** on the API call — a transient network error
  or rate limit currently surfaces as a per-row `error`, not an automatic
  retry. Worth adding before high-volume production use.
- **Single-document-per-call.** Each file is one API request; there's no
  batching, so a folder of 200 invoices means 200 sequential calls.

## Roadmap / ideas for extending

- Retry logic with exponential backoff on rate limits/transient errors
- Multi-page PDF handling (send all pages, or detect which page has the totals)
- Confidence scores per field, not just null/not-null
- Side-by-side image + extracted-JSON review UI for human QA before export

---

Built as a freelance portfolio project. Feedback and issues welcome.
