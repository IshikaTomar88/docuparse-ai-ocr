"""
utils.py
--------
Export helpers — shared by app.py.
"""

import io
import pandas as pd

EXPORT_COLUMNS = [
    "source_file", "invoice_number", "vendor_name",
    "invoice_date", "total_amount", "currency", "error", "processed_at",
]


def records_to_dataframe(records: list) -> pd.DataFrame:
    return pd.DataFrame(records, columns=EXPORT_COLUMNS)


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Extracted_Invoices")
    return buffer.getvalue()


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")
