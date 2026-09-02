# pipeline.py
import os
import json
import base64
from datetime import datetime
from PIL import Image
from pdf2image import convert_from_path
from google import genai
from google.genai import types

SUPPORTED_EXTS = {".pdf", ".jpg", ".jpeg", ".png"}

def get_client():
    """Initializes the Google GenAI client securely using the sidebar key."""
    api_key = os.environ.get("OPENAI_API_KEY", "") # Keep variable name same so app.py doesn't break
    if not api_key:
        raise ValueError("Gemini API Key is missing. Enter it in the sidebar.")
    return genai.Client(api_key=api_key)

def convert_pdf_page_to_jpeg(pdf_path, output_dir):
    """Converts the first page of a PDF file to a temporary JPEG image."""
    try:
        pages = convert_from_path(pdf_path, dpi=150)
        if pages:
            jpeg_path = os.path.join(output_dir, "temp_render.jpg")
            pages.save(jpeg_path, "JPEG")
            return jpeg_path
    except Exception as e:
        print(f"PDF Conversion Error: {e}")
    return None

def process_single_file(client, file_path, tmp_path, model="gemini-2.0-flash"):
    """
    Core Extraction Node: Sends images to Google Gemini for free 
    and outputs a structured dictionary mapping to your layout columns.
    """
    record = {
        "source_file": os.path.basename(file_path),
        "invoice_number": None,
        "vendor_name": None,
        "invoice_date": None,
        "total_amount": None,
        "currency": None,
        "error": None,
        "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    working_image_path = str(file_path)
    is_pdf = str(file_path).lower().endswith(".pdf")
    
    if is_pdf:
        rendered_jpg = convert_pdf_page_to_jpeg(file_path, tmp_path)
        if not rendered_jpg:
            record["error"] = "Failed to convert PDF page down to JPEG image format."
            return record
        working_image_path = rendered_jpg

    try:
        # Load the image using Pillow for the Gemini SDK
        raw_image = Image.open(working_image_path)
        
        prompt = """
        Analyze this invoice image structure. Extract data points into a valid JSON schema with keys:
        - invoice_number (string or null)
        - vendor_name (string or null)
        - invoice_date (string formatting as YYYY-MM-DD or null)
        - total_amount (float/number or null)
        - currency (string symbol/code like USD, INR or null)
        
        Return ONLY a raw valid JSON object structure. Do not wrap it in markdown block tags like ```json.
        """
        
        # Call Google's multimodal API model with structured JSON constraint
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[raw_image, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0
            ),
        )
        
        extracted_data = json.loads(response.text)
        
        # Merge values over our default record dictionary layout template
        for key in ["invoice_number", "vendor_name", "invoice_date", "total_amount", "currency"]:
            if key in extracted_data:
                record[key] = extracted_data[key]
                
    except Exception as e:
        record["error"] = str(e)
        
    return record
