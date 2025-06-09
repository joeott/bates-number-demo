"""
Vision-based OCR module using Gemma 3 models via LM Studio.
Implements OCR functionality for extracting text from scanned documents.
"""

import base64
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
import io
from PIL import Image
import pdf2image
from openai import OpenAI

from .config import (
    LMSTUDIO_HOST,
    VISION_OCR_MODEL,
    PDF_DPI,
    ENABLE_VISION_OCR,
    LLM_PROVIDER
)

logger = logging.getLogger(__name__)


class VisionOCRExtractor:
    """
    Extract text from images using Gemma 3 vision models via LM Studio.
    """
    
    def __init__(self, model: str = None, base_url: str = None):
        """Initialize the Vision OCR extractor."""
        self.model = model or VISION_OCR_MODEL
        self.base_url = base_url or LMSTUDIO_HOST
        self.enabled = ENABLE_VISION_OCR and LLM_PROVIDER == "lmstudio"
        
        if self.enabled:
            try:
                # Initialize OpenAI client pointing to LM Studio
                self.client = OpenAI(
                    base_url=self.base_url,
                    api_key="lm-studio"  # LM Studio doesn't require a real API key
                )
                logger.info(f"Vision OCR initialized with model: {self.model}")
            except Exception as e:
                logger.error(f"Failed to initialize Vision OCR client: {e}")
                self.enabled = False
        else:
            logger.info("Vision OCR is disabled or not using LM Studio")
    
    def encode_image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        try:
            buffer = io.BytesIO()
            # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
            image.save(buffer, format='PNG')
            buffer.seek(0)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            raise
    
    def extract_text_from_image(self, image: Image.Image, page_num: int = 0) -> str:
        """
        Extract text from a single image using Gemma 3 vision model.
        
        Args:
            image: PIL Image object
            page_num: Page number for logging
            
        Returns:
            Extracted text string
        """
        if not self.enabled:
            return ""
        
        try:
            # Encode image to base64
            base64_image = self.encode_image_to_base64(image)
            
            # Craft OCR prompt
            ocr_prompt = """You are an Optical Character Recognition (OCR) assistant.
Extract all text from the provided image.
Preserve line breaks and general layout.
If there are tables, represent them in a structured way.
Focus on accuracy and completeness.
Return ONLY the extracted text, no explanations or metadata."""
            
            # Make API call to LM Studio
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": ocr_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2048,
                temperature=0.1  # Low temperature for accurate OCR
            )
            
            if response.choices and response.choices[0].message:
                extracted_text = response.choices[0].message.content
                logger.info(f"OCR extracted {len(extracted_text)} characters from page {page_num}")
                return extracted_text.strip()
            else:
                logger.warning(f"No text content in OCR response for page {page_num}")
                return ""
                
        except Exception as e:
            logger.error(f"Vision OCR failed for page {page_num}: {e}")
            if "Model does not support images" in str(e):
                logger.error(f"Model {self.model} may not support vision tasks")
                self.enabled = False  # Disable for subsequent calls
            return ""
    
    def extract_text_from_pdf_page(self, pdf_path: Path, page_num: int) -> str:
        """
        Extract text from a specific PDF page using vision OCR.
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number (1-based)
            
        Returns:
            Extracted text string
        """
        if not self.enabled:
            return ""
        
        try:
            # Convert specific PDF page to image
            images = pdf2image.convert_from_path(
                pdf_path,
                dpi=PDF_DPI,
                first_page=page_num,
                last_page=page_num
            )
            
            if images:
                return self.extract_text_from_image(images[0], page_num)
            else:
                logger.warning(f"Failed to convert PDF page {page_num} to image")
                return ""
                
        except Exception as e:
            logger.error(f"Failed to process PDF page {page_num}: {e}")
            return ""
    
    def extract_text_from_pdf(self, pdf_path: Path, max_pages: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Extract text from all pages of a PDF using vision OCR.
        
        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum number of pages to process (None for all)
            
        Returns:
            List of dictionaries with page_num and extracted text
        """
        if not self.enabled:
            return []
        
        results = []
        try:
            # Get total page count
            from pypdf import PdfReader
            with open(pdf_path, 'rb') as f:
                reader = PdfReader(f)
                total_pages = len(reader.pages)
            
            pages_to_process = min(total_pages, max_pages) if max_pages else total_pages
            logger.info(f"Starting vision OCR on {pdf_path.name} ({pages_to_process} pages)")
            
            # Process each page
            for page_num in range(1, pages_to_process + 1):
                text = self.extract_text_from_pdf_page(pdf_path, page_num)
                results.append({
                    'page_num': page_num,
                    'text': text,
                    'char_count': len(text)
                })
                
                # Log progress every 10 pages
                if page_num % 10 == 0:
                    logger.info(f"Processed {page_num}/{pages_to_process} pages")
            
            # Calculate statistics
            total_chars = sum(r['char_count'] for r in results)
            avg_chars = total_chars / len(results) if results else 0
            logger.info(f"Vision OCR complete: {total_chars} total characters, {avg_chars:.0f} avg per page")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {e}")
            return results


# Singleton instance for easy access
_vision_ocr_instance = None

def get_vision_ocr() -> VisionOCRExtractor:
    """Get or create the singleton Vision OCR instance."""
    global _vision_ocr_instance
    if _vision_ocr_instance is None:
        _vision_ocr_instance = VisionOCRExtractor()
    return _vision_ocr_instance