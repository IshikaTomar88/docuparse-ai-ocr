# 🧠 DocuParse AI — Gemini Edition

Invoice/receipt extraction using Google Gemini's vision model, reading
documents by layout rather than character-level OCR + regex.

**Live demo:** _add your Streamlit Cloud link here after deploying_
**Stack:** Python · Gemini 3.6 Flash (vision) · pdf2image · Pandas · Streamlit

---

## ⚠️ This is a corrected version — here's exactly what was wrong

This code went through **two review passes**. The first caught six issues;
a second, more adversarial pass — deliberately testing edge cases and
inspecting the actual SDK source rather than trusting the first pass was
complete — found four more that only a real test run surfaced.

**First pass:**

1. **Used `google.generativeai`, which Google has deprecated.**
   Confirmed via Google's own repo:
   [google-gemini/deprecated-generative-ai-python](https://github.com/google-gemini/deprecated-generative-ai-python)
   — "This SDK is now deprecated, use the new unified Google GenAI SDK."
   Fixed: migrated to `google-genai` (`from google import genai`).

2. **Offered `gemini-2.5-flash` as a model choice.** Per Google's own
   deprecation notices, `gemini-2.5-flash` shuts down **October 16, 2026**
   (Gemini Developer API). Fixed: replaced with `gemini-3.5-flash-lite`,
   with `gemini-3.6-flash` (confirmed current, GA) as the default.

3. **Gemini API key stored under the env var name `OPENAI_API_KEY`.**
   Fixed: renamed to `GEMINI_API_KEY` throughout.

4. **The `client` parameter was accepted and then silently ignored** — the
   actual API call used a global module reference instead. Fixed and
   verified with a test that asserts the mocked client **is** called.

5. **PDF pages rendered to a hardcoded filename** (`temp_render.jpg`)
   reused across every file in a batch. Fixed: filename is now derived
   from the source file, verified with a test.

6. **No safety nets on the model's output** — no markdown-fence stripping,
   no currency/comma cleanup, no date normalization. Fixed and verified.

**Second pass — found by deliberately testing edge cases the first pass
didn't try, and by reading the actual `google-genai` SDK source instead of
assuming how it handles images:**

7. **`parse_model_response()` crashed on valid-but-wrong-shape JSON.** If
   the model returned a bare `null`, a list, a number, or a string instead
   of an object, `data.get(...)` raised an uncaught `AttributeError`. This
   didn't crash the *app* (an outer try/except in `process_single_file`
   masked it), but it meant the function didn't do what its own docstring
   claimed — "never a silently wrong row" — it could raise instead of
   returning a structured error. Verified with a direct test of all four
   shapes before and after the fix. Fixed: explicit `isinstance(data, dict)`
   check with its own error message.

8. **The date-normalization fallback was internally inconsistent.** Same
   source format, two different results: `"09/02/2026"` parsed as
   month-first (Sept 2) while `"15/03/2026"` — from the exact same
   DD/MM/YYYY convention — correctly parsed as day-first (March 15),
   purely because 15 can't be a month and dateutil auto-corrected only
   when forced to. Given this project's stated target market (Indian
   clients), fixed by defaulting to `dayfirst=True`, verified consistent
   across a test set.

9. **That date fix, tested on its own, revealed a second, worse bug**:
   `dayfirst=True` also corrupted an already-correct ISO date — the exact
   `"2026-09-02"` format the prompt asks the model to return got
   reinterpreted as day=09/month=02 ("Feb 9"), meaning the "safety net"
   would silently corrupt the model's own correct output in the common
   case where it complied with instructions. Fixed with an ISO-format
   short-circuit: strings already matching `YYYY-MM-DD` are validated and
   passed through directly, never re-parsed through the ambiguous fuzzy
   parser. Verified with a full before/after test matrix, including the
   exact case that broke.

10. **`processed_at` was stamped at the start of processing, not the
    end** — meaning it didn't actually represent when processing finished
    despite the field name. Minor, but fixed to reflect real completion
    time.

11. **`requirements.txt` pinned `google-genai>=0.5`**, a version number
    that predates the SDK's actual API shape used in this code. The real
    installed version verified in testing was `2.22.0`. Fixed to `>=1.0`.

**Also checked and confirmed NOT a bug** (worth stating, since "no bugs"
claims are only credible if you show what you ruled out, not just what you
fixed): PIL images with an alpha channel (RGBA PNGs) were suspected of
possibly crashing Gemini's image upload. Direct inspection of the
`google-genai` SDK's `pil_to_blob()` source showed it defaults to PNG
encoding (which supports alpha) unless the image is already a
JPEG-with-compatible-mode — so RGBA input is handled safely with no fix
needed.

## What was actually verified vs. what wasn't

**Verified, with real test runs (see the exact assertions in the repo's
test history — reproducible with the commands below):**
- ✅ `google-genai` actually installs and exposes `genai.Client` / `types.GenerateContentConfig` as documented.
- ✅ Real PDF → real rasterized JPEG, with the collision bug fixed (filename now unique per source file).
- ✅ `parse_model_response()` correctly handles: clean JSON, markdown-fenced JSON, comma-formatted amounts, non-standard dates, and outright non-JSON garbage.
- ✅ `process_single_file()` end-to-end with a mocked client — confirmed the mock **is** called (proving the client-ignored bug is fixed), confirmed the correct model name is passed, confirmed the final record is correctly structured.

**NOT verified — genuinely can't be, from this environment:**
- ❌ **The actual Gemini API call itself.** This was built in a sandbox without network access to Google's API endpoints, so real-world extraction accuracy on a messy invoice photo — versus the mocked response used in testing — has not been tested. Test against your own real invoices with your own `GEMINI_API_KEY` before client delivery.

## Setup

1. **Get a free Gemini API key** at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

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
   export GEMINI_API_KEY="your-key-here"
   streamlit run app.py
   ```

5. **Reproduce the verification tests above yourself:**
   ```python
   from pathlib import Path
   from unittest.mock import MagicMock
   import pipeline

   fake_client = MagicMock()
   fake_response = MagicMock()
   fake_response.text = '{"invoice_number": "INV-1", "vendor_name": "Test Co", "invoice_date": "2026-01-15", "total_amount": "1,200.00", "currency": "USD"}'
   fake_client.models.generate_content.return_value = fake_response

   record = pipeline.process_single_file(fake_client, Path("invoices/some_invoice.pdf"), Path("/tmp"))
   print(record)
   assert fake_client.models.generate_content.called  # proves client isn't ignored
   ```

## Deploy it (free, Streamlit Community Cloud)

1. Push this repo to your own GitHub.
2. [share.streamlit.io](https://share.streamlit.io) → New app → this repo, main file `app.py`.
3. Advanced settings → Secrets:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
4. Deploy — `packages.txt` installs `poppler-utils` automatically.

## Project structure

```
docuparse-gemini/
├── app.py            # Streamlit UI
├── pipeline.py         # ingestion, PDF rasterization, Gemini call, parsing/cleaning
├── utils.py             # DataFrame -> Excel/CSV export helpers
├── requirements.txt      # google-genai, NOT the deprecated google-generativeai
├── packages.txt            # poppler-utils for Streamlit Cloud
└── invoices/                 # drop sample invoices here for local testing
```

## Known limitations

- **First page only** for multi-page PDFs.
- **No retry/backoff** on the API call — a transient error surfaces as a
  per-row `error`, not an automatic retry.
- **Model availability changes fast.** Gemini model names get deprecated
  on a matter of months, not years (see bug #2 above) — check
  [Google's release notes](https://ai.google.dev/gemini-api/docs/changelog)
  periodically and update `model` choices in `app.py` accordingly.

---

Built as a freelance portfolio project. Feedback and issues welcome.
