"""
app.py
------
Streamlit UI for the Gemini-based invoice extractor.
Run locally:  streamlit run app.py
Deploy free:  push to GitHub, deploy on streamlit.io/cloud, add GEMINI_API_KEY
              as a secret. packages.txt installs poppler-utils automatically.
"""

import os
import tempfile
from pathlib import Path

import streamlit as st

from pipeline import SUPPORTED_EXTS, get_client, process_single_file
from utils import dataframe_to_csv_bytes, dataframe_to_excel_bytes, records_to_dataframe

st.set_page_config(page_title="DocuParse AI (Gemini)", page_icon="🧠", layout="wide")

st.title("🧠 DocuParse AI — Gemini Edition")
st.caption("Vision-LLM invoice extraction — understands document layout, not just characters.")

with st.sidebar:
    st.header("Settings")
    api_key_input = st.text_input(
        "Gemini API Key", type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
        help="Get a free key at aistudio.google.com/apikey",
    )
    # gemini-2.5-flash is NOT offered here: it shuts down Oct 16, 2026
    # (Gemini Developer API) per Google's own deprecation notice — offering
    # it in a dropdown today would mean shipping a client something that
    # stops working in weeks. gemini-3.6-flash is the current GA flash
    # model; gemini-3.5-flash-lite is the cheaper/faster alternative.
    model = st.selectbox("Model", ["gemini-3.6-flash", "gemini-3.5-flash-lite"], index=0)

uploaded_files = st.file_uploader(
    "Upload invoices (PDF, JPG, PNG)",
    type=["pdf", "jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

run = st.button("Extract data", type="primary", disabled=not uploaded_files)

if run:
    if not api_key_input:
        st.error("Please provide a Gemini API key in the sidebar.")
        st.stop()

    os.environ["GEMINI_API_KEY"] = api_key_input
    try:
        client = get_client()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    records = []
    progress = st.progress(0.0, text="Starting...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for i, uploaded in enumerate(uploaded_files):
            file_path = tmp_path / uploaded.name
            file_path.write_bytes(uploaded.getbuffer())
            if file_path.suffix.lower() not in SUPPORTED_EXTS:
                continue

            progress.progress(i / len(uploaded_files), text=f"Processing {uploaded.name}...")
            record = process_single_file(client, file_path, tmp_path, model=model)
            records.append(record)

        progress.progress(1.0, text="Done")

    df = records_to_dataframe(records)
    st.success(f"Extracted {len(df)} document(s).")
    st.dataframe(df, use_container_width=True)

    if (df["error"].notna()).any():
        st.warning(f"{df['error'].notna().sum()} row(s) had an extraction error — check the `error` column.")

    excel_bytes = dataframe_to_excel_bytes(df)
    st.download_button(
        "⬇️ Download as Excel",
        data=excel_bytes,
        file_name="docuparse_output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.download_button(
        "⬇️ Download as CSV",
        data=dataframe_to_csv_bytes(df),
        file_name="docuparse_output.csv",
        mime="text/csv",
    )
else:
    st.info("Upload one or more invoices and click **Extract data** to begin.")
