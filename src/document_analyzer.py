"""Document analyzer for detecting scanned vs text-based PDFs and other characteristics."""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from pypdf import PdfReader
import fitz  # PyMuPDF
from PIL import Image
import io
import numpy as np

logger = logging.getLogger(__name__)


class DocumentAnalyzer:
    """Analyzes PDF documents to determine their characteristics."""
    
    def __init__(self):
        self.min_text_ratio = 0.01  # Minimum text-to-page ratio for text-based classification
        self.image_dpi_threshold = 150  # DPI threshold for high-quality scans
    
    def analyze_document(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Analyze a PDF document to determine its characteristics.
        
        Returns:
            Dictionary with analysis results including:
            - is_scanned: Whether the document appears to be scanned
            - has_images: Whether the document contains images
            - text_ratio: Ratio of text content to page size
            - page_count: Number of pages
            - needs_ocr: Whether OCR might be beneficial
            - dominant_type: 'text', 'scanned', 'mixed', or 'image'
        """
        try:
            # Try PyMuPDF first for more robust analysis
            return self._analyze_with_pymupdf(pdf_path)
        except Exception as e:
            logger.warning(f"PyMuPDF analysis failed, falling back to PyPDF: {e}")
            try:
                return self._analyze_with_pypdf(pdf_path)
            except Exception as e2:
                logger.error(f"Document analysis failed: {e2}")
                return self._default_analysis()
    
    def _analyze_with_pymupdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Analyze using PyMuPDF for detailed inspection."""
        doc = fitz.open(str(pdf_path))
        
        total_pages = doc.page_count
        pages_with_text = 0
        pages_with_images = 0
        total_text_length = 0
        total_area = 0
        image_info = []
        
        for page_num in range(total_pages):
            page = doc[page_num]
            
            # Get text
            text = page.get_text()
            text_length = len(text.strip())
            if text_length > 10:  # More than just whitespace
                pages_with_text += 1
            total_text_length += text_length
            
            # Get page area
            rect = page.rect
            page_area = rect.width * rect.height
            total_area += page_area
            
            # Check for images
            image_list = page.get_images()
            if image_list:
                pages_with_images += 1
                
                # Analyze first few images
                for img_index, img in enumerate(image_list[:3]):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        image_info.append({
                            "width": pix.width,
                            "height": pix.height,
                            "dpi": self._estimate_dpi(pix.width, pix.height, rect.width, rect.height)
                        })
                    except:
                        pass
        
        doc.close()
        
        # Calculate metrics
        text_ratio = total_text_length / (total_area / 100) if total_area > 0 else 0
        text_coverage = pages_with_text / total_pages if total_pages > 0 else 0
        image_coverage = pages_with_images / total_pages if total_pages > 0 else 0
        
        # Determine document type
        is_scanned = False
        needs_ocr = False
        dominant_type = "unknown"
        
        if text_coverage < 0.1 and image_coverage > 0.8:
            # Mostly images with little text - likely scanned
            is_scanned = True
            needs_ocr = True
            dominant_type = "scanned"
        elif text_ratio < self.min_text_ratio and pages_with_images > 0:
            # Low text ratio but has images - possibly scanned
            is_scanned = True
            needs_ocr = True
            dominant_type = "scanned"
        elif text_coverage > 0.8 and image_coverage < 0.2:
            # Mostly text - native PDF
            dominant_type = "text"
        elif text_coverage > 0.5 and image_coverage > 0.5:
            # Mixed content
            dominant_type = "mixed"
        elif image_coverage > 0.8:
            # Mostly images
            dominant_type = "image"
            needs_ocr = text_coverage < 0.3
        else:
            dominant_type = "text"
        
        # Check if high-quality scan based on image DPI
        avg_dpi = 0
        if image_info:
            dpis = [img["dpi"] for img in image_info if img["dpi"] > 0]
            if dpis:
                avg_dpi = sum(dpis) / len(dpis)
        
        return {
            "is_scanned": is_scanned,
            "has_images": pages_with_images > 0,
            "text_ratio": text_ratio,
            "text_coverage": text_coverage,
            "image_coverage": image_coverage,
            "page_count": total_pages,
            "pages_with_text": pages_with_text,
            "pages_with_images": pages_with_images,
            "needs_ocr": needs_ocr,
            "dominant_type": dominant_type,
            "avg_image_dpi": avg_dpi,
            "total_text_length": total_text_length
        }
    
    def _analyze_with_pypdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Fallback analysis using PyPDF."""
        reader = PdfReader(str(pdf_path))
        
        total_pages = len(reader.pages)
        pages_with_text = 0
        total_text_length = 0
        
        for page in reader.pages:
            text = page.extract_text()
            text_length = len(text.strip())
            if text_length > 10:
                pages_with_text += 1
            total_text_length += text_length
        
        text_coverage = pages_with_text / total_pages if total_pages > 0 else 0
        
        # Simple heuristic for PyPDF
        is_scanned = text_coverage < 0.1
        needs_ocr = is_scanned
        
        if text_coverage > 0.8:
            dominant_type = "text"
        elif text_coverage < 0.2:
            dominant_type = "scanned"
        else:
            dominant_type = "mixed"
        
        return {
            "is_scanned": is_scanned,
            "has_images": None,  # PyPDF doesn't easily detect images
            "text_ratio": total_text_length / (total_pages * 1000),  # Rough estimate
            "text_coverage": text_coverage,
            "image_coverage": None,
            "page_count": total_pages,
            "pages_with_text": pages_with_text,
            "pages_with_images": None,
            "needs_ocr": needs_ocr,
            "dominant_type": dominant_type,
            "avg_image_dpi": 0,
            "total_text_length": total_text_length
        }
    
    def _estimate_dpi(self, img_width: int, img_height: int, 
                      page_width: float, page_height: float) -> float:
        """Estimate DPI of an image based on its size and page dimensions."""
        if page_width <= 0 or page_height <= 0:
            return 0
        
        # Assume page dimensions are in points (1 point = 1/72 inch)
        page_width_inches = page_width / 72
        page_height_inches = page_height / 72
        
        # Calculate DPI
        dpi_w = img_width / page_width_inches if page_width_inches > 0 else 0
        dpi_h = img_height / page_height_inches if page_height_inches > 0 else 0
        
        return (dpi_w + dpi_h) / 2
    
    def _default_analysis(self) -> Dict[str, Any]:
        """Return default analysis when all methods fail."""
        return {
            "is_scanned": False,
            "has_images": None,
            "text_ratio": 0,
            "text_coverage": 0,
            "image_coverage": None,
            "page_count": 0,
            "pages_with_text": 0,
            "pages_with_images": None,
            "needs_ocr": False,
            "dominant_type": "unknown",
            "avg_image_dpi": 0,
            "total_text_length": 0
        }
    
    def should_use_vision_model(self, analysis: Dict[str, Any]) -> bool:
        """
        Determine if a vision model should be used for this document.
        
        Args:
            analysis: Document analysis results
            
        Returns:
            Whether to use vision model
        """
        # Use vision model for:
        # 1. Scanned documents
        # 2. Documents that need OCR
        # 3. Image-dominant documents
        # 4. Documents with high-res images (good for vision models)
        
        if analysis.get("is_scanned", False):
            return True
        
        if analysis.get("needs_ocr", False):
            return True
        
        if analysis.get("dominant_type") in ["scanned", "image"]:
            return True
        
        # Use vision for high-quality mixed documents
        if (analysis.get("dominant_type") == "mixed" and 
            analysis.get("avg_image_dpi", 0) > self.image_dpi_threshold):
            return True
        
        return False
    
    def get_recommended_models(self, analysis: Dict[str, Any]) -> Dict[str, str]:
        """
        Get recommended models for different tasks based on document analysis.
        
        Returns:
            Dictionary mapping tasks to recommended model types
        """
        recommendations = {}
        
        if self.should_use_vision_model(analysis):
            # Use vision models for visual analysis
            recommendations["categorization"] = "visual"
            recommendations["extraction"] = "visual"
            logger.info(f"Recommending vision model (scanned: {analysis.get('is_scanned')}, "
                       f"type: {analysis.get('dominant_type')})")
        else:
            # Use standard text models
            recommendations["categorization"] = "categorization"
            recommendations["extraction"] = "reasoning"
        
        # Always use standard models for synthesis
        recommendations["synthesis"] = "synthesis"
        
        return recommendations