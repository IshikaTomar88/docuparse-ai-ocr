# pipeline.py
import os
import base64
import json
from datetime import datetime
from openai import OpenAI
from pdf2image import convert_from_path

SUPPORTED_EXTS = {".pdf", ".jpg", ".jpeg", ".png"}

def get_client():
    """Initializes the OpenAI client securely."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OpenAI API Key environment variable is missing.")
    return OpenAI(api_key=api_key)

def convert_pdf_page_to_jpeg(pdf_path, output_dir):
    """Converts the first page of a PDF file to a temporary JPEG string image."""
    try:
        pages = convert_from_path(pdf_path, dpi=150)
        if pages:
            jpeg_path = os.path.join(output_dir, "temp_render.jpg")
            pages[0].save(jpeg_path, "JPEG")
            return jpeg_path
    except Exception as e:
        print(f"PDF Conversion Error: {e}")
    return None

def encode_image_base64(image_path):
    """Encodes a local file into a standard base64 data utility string."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def process_single_file(client, file_path, tmp_path, model="gpt-4o-mini"):
    """
    Core Extraction Node: Sends base64 arrays to the Vision LLM 
    and outputs a structured dictionary tracking our schema.
    """
    # Default return template matching your EXPORT_COLUMNS structure in utils.py
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
    
    working_image = str(file_path)
    is_pdf = str(file_path).lower().endswith(".pdf")
    
    if is_pdf:
        rendered_jpg = convert_pdf_page_to_jpeg(file_path, tmp_path)
        if not rendered_jpg:
            record["error"] = "Failed to convert PDF page down to JPEG image format."
            return record
        working_image = rendered_jpg

    try:
        base64_str = encode_image_base64(working_image)
        
        prompt = """
        Analyze this invoice image structure. Extract data points into a valid JSON schema with keys:
        - invoice_number (string or null)
        - vendor_name (string or null)
        - invoice_date (string formatting as YYYY-MM-DD or null)
        - total_amount (float/number or null)
        - currency (string symbol/code like USD, INR or null)
        
        Return ONLY raw valid JSON structures. Do not write markdown tags or block text code wrapping.
        """
        
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_str}"}}
                    ]
                }
            ],
            temperature=0.0,
            max_tokens=400
        )
        
        extracted_data = json.loads(response.choices.message.content)
        
        # Merge values cleanly over our default record dictionary array template keys
        for key in ["invoice_number", "vendor_name", "invoice_date", "total_amount", "currency"]:
            if key in extracted_data:
                record[key] = extracted_data[key]
                
    except Exception as e:
        record["error"] = str(e)
        
    return record
