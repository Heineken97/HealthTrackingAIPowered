"""
Medical File Processing Utilities
MIT Challenge Submission

Handles PDF and image processing with text extraction
and medical data parsing.
"""

import PyPDF2
import pytesseract
from PIL import Image
import io
import re
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def process_medical_file(file_bytes: bytes) -> Dict[str, Any]:
    """
    Process medical files to extract structured data
    
    Args:
        file_bytes: Raw file content
        
    Returns:
        Dictionary containing:
        - lab_values: Extracted test results
        - clinical_text: Raw medical notes
        - abnormal_flags: Any abnormal values
    """
    try:
        # Try PDF extraction first
        text = extract_pdf_text(file_bytes)
        
        # Fallback to OCR if needed
        if len(text) < 100:
            text = extract_text_with_ocr(file_bytes)
            
        return parse_medical_data(text)
    except Exception as e:
        logger.error(f"File processing failed: {str(e)}")
        return {"error": str(e)}

def extract_pdf_text(file_bytes: bytes) -> str:
    """Extract text from PDF files"""
    text = ""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        logger.warning(f"PDF extraction failed: {str(e)}")
    return text

def extract_text_with_ocr(file_bytes: bytes) -> str:
    """Extract text from images using OCR"""
    try:
        img = Image.open(io.BytesIO(file_bytes))
        return pytesseract.image_to_string(img)
    except Exception as e:
        logger.warning(f"OCR failed: {str(e)}")
        return ""

def parse_medical_data(text: str) -> Dict[str, Any]:
    """Parse medical information from extracted text"""
    results = {
        "lab_values": {},
        "clinical_text": text,
        "abnormal_flags": []
    }
    
    # Lab value patterns
    patterns = {
        "hemoglobin": r"(?i)hemo\w*globin\D*(\d+\.?\d*)",
        "glucose": r"(?i)glucose\D*(\d+\.?\d*)",
        "wbc": r"(?i)white\s*blood\s*cells?\D*(\d+\.?\d*)",
        "creatinine": r"(?i)creatinine\D*(\d+\.?\d*)"
    }
    
    # Extract and validate lab values
    for test, pattern in patterns.items():
        matches = re.finditer(pattern, text)
        values = [float(m.group(1)) for m in matches if m.group(1)]
        
        if values:
            results["lab_values"][test] = values[-1]  # Take most recent
    
    return results