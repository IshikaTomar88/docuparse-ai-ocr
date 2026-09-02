import os
import tempfile
from pathlib import Path

import streamlit as st

from pipeline import SUPPORTED_EXTS, get_client, process_single_file
from utils import dataframe_to_csv_bytes, dataframe_to_excel_bytes, records_to_dataframe

# Page Configuration Setup
st.set_page_config(page_title="DocuParse AI", page_icon="🧠", layout="wide")

st.title("🧠 DocuParse AI")
st.caption("Vision-LLM invoice extraction — understands document layout, not just characters.")

# Sidebar Configuration Layout
with st.sidebar:
    st.header("Settings")
    api_key_input = st.text_input(
        "OpenAI API Key", type="password",
        value=os.environ.get("OPENAI_API_KEY", ""),
        help="Used only for this session, never saved to disk.",
    )
    model = st.selectbox("Model", ["gpt-4o-mini"], index=0)

# Main File Ingestion Uploader 
uploaded_files = st.file_uploader(
    "Upload invoices (PDF, JPG, PNG)",
    type=["pdf", "jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

# Core Action Trigger Button Node
run = st.button("Extract data", type="primary", disabled=not uploaded_files)

if run:
    if not api_key_input:
        st.error("Please provide an OpenAI API key in the sidebar.")
        st.stop()

    os.environ["OPENAI_API_KEY"] = api_key_input
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

    # Format the array outputs into a structured Pandas tabular layout
    df = records_to_dataframe(records)
    st.success(f"Extracted {len(df)} document(s).")
    st.dataframe(df, use_container_width=True)

    if (df["error"].notna()).any():
        st.warning(f"{df['error'].notna().sum()} row(s) had an extraction error — check the `error` column.")

    # Export & Download Pipeline Nodes
    out_path = Path(tempfile.gettempdir()) / "docuparse_output.xlsx"
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
