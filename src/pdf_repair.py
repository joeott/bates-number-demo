"""PDF repair and recovery utilities for handling corrupted or problematic PDFs."""

import logging
from pathlib import Path
from typing import Optional, Tuple
import fitz  # PyMuPDF
from pypdf import PdfReader, PdfWriter
import pikepdf
from io import BytesIO

logger = logging.getLogger(__name__)


class PDFRepairService:
    """Service for repairing and recovering corrupted PDF files."""
    
    def __init__(self):
        self.repair_methods = [
            self._repair_with_pymupdf,
            self._repair_with_pikepdf,
            self._repair_with_pypdf_lenient
        ]
    
    def repair_pdf(self, input_path: Path, output_path: Optional[Path] = None) -> Tuple[bool, Optional[Path]]:
        """
        Attempt to repair a PDF using multiple methods.
        
        Args:
            input_path: Path to the potentially corrupted PDF
            output_path: Optional output path for repaired PDF
            
        Returns:
            Tuple of (success, repaired_path)
        """
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_repaired{input_path.suffix}"
        
        # Try each repair method
        for repair_method in self.repair_methods:
            try:
                success = repair_method(input_path, output_path)
                if success:
                    logger.info(f"Successfully repaired PDF using {repair_method.__name__}")
                    return True, output_path
            except Exception as e:
                logger.debug(f"Repair method {repair_method.__name__} failed: {e}")
                continue
        
        logger.warning(f"All repair methods failed for {input_path}")
        return False, None
    
    def _repair_with_pymupdf(self, input_path: Path, output_path: Path) -> bool:
        """Repair PDF using PyMuPDF (fitz)."""
        try:
            # Open and save with PyMuPDF to fix structure
            doc = fitz.open(str(input_path))
            
            # Clean up the PDF
            doc.clean()
            
            # Save with garbage collection and linearization
            doc.save(
                str(output_path),
                garbage=4,  # Maximum garbage collection
                clean=True,
                deflate=True,
                linear=True
            )
            doc.close()
            
            # Verify the repaired PDF
            test_doc = fitz.open(str(output_path))
            page_count = test_doc.page_count
            test_doc.close()
            
            return page_count > 0
            
        except Exception as e:
            logger.debug(f"PyMuPDF repair failed: {e}")
            return False
    
    def _repair_with_pikepdf(self, input_path: Path, output_path: Path) -> bool:
        """Repair PDF using pikepdf."""
        try:
            # Open with pikepdf (more tolerant of errors)
            with pikepdf.open(str(input_path), allow_overwriting_input=True) as pdf:
                # Remove problematic features
                pdf.remove_links()
                
                # Clean up the PDF
                pdf.save(
                    str(output_path),
                    compress_streams=True,
                    object_stream_mode=pikepdf.ObjectStreamMode.generate,
                    linearize=True
                )
            
            # Verify the repaired PDF
            with pikepdf.open(str(output_path)) as test_pdf:
                return len(test_pdf.pages) > 0
                
        except Exception as e:
            logger.debug(f"pikepdf repair failed: {e}")
            return False
    
    def _repair_with_pypdf_lenient(self, input_path: Path, output_path: Path) -> bool:
        """Repair PDF using PyPDF in lenient mode."""
        try:
            # Open with strict=False for lenient parsing
            reader = PdfReader(str(input_path), strict=False)
            writer = PdfWriter()
            
            # Copy all pages
            for page in reader.pages:
                writer.add_page(page)
            
            # Write the repaired PDF
            with open(output_path, 'wb') as f:
                writer.write(f)
            
            # Verify the repaired PDF
            test_reader = PdfReader(str(output_path))
            return len(test_reader.pages) > 0
            
        except Exception as e:
            logger.debug(f"PyPDF lenient repair failed: {e}")
            return False
    
    def validate_pdf(self, pdf_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Validate if a PDF is properly structured.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Try opening with strict mode
            reader = PdfReader(str(pdf_path), strict=True)
            page_count = len(reader.pages)
            
            if page_count == 0:
                return False, "PDF has no pages"
            
            # Try to access first page
            first_page = reader.pages[0]
            _ = first_page.mediabox
            
            return True, None
            
        except Exception as e:
            return False, str(e)
    
    def extract_text_fallback(self, pdf_path: Path) -> Optional[str]:
        """
        Extract text using multiple methods as fallback.
        Useful when standard extraction fails.
        """
        # Try PyMuPDF first (most robust)
        try:
            doc = fitz.open(str(pdf_path))
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            if text.strip():
                return text
        except Exception as e:
            logger.debug(f"PyMuPDF text extraction failed: {e}")
        
        # Try pikepdf with page iteration
        try:
            with pikepdf.open(str(pdf_path)) as pdf:
                text = ""
                for page in pdf.pages:
                    if '/Contents' in page:
                        # This is a simplified extraction
                        # In practice, you might need more sophisticated parsing
                        pass
                return text if text else None
        except Exception as e:
            logger.debug(f"pikepdf text extraction failed: {e}")
        
        return None


class RobustPDFProcessor:
    """Enhanced PDF processor with automatic repair capabilities."""
    
    def __init__(self, base_processor, repair_service: Optional[PDFRepairService] = None):
        self.base_processor = base_processor
        self.repair_service = repair_service or PDFRepairService()
        self._repair_cache = {}  # Cache repaired PDFs
    
    def process_with_repair(self, input_path: Path, process_func, *args, **kwargs):
        """
        Process a PDF with automatic repair on failure.
        
        Args:
            input_path: Path to the PDF
            process_func: The processing function to call
            *args, **kwargs: Arguments for the processing function
        """
        # First, try normal processing
        try:
            return process_func(input_path, *args, **kwargs)
        except Exception as e:
            logger.warning(f"Initial processing failed for {input_path}: {e}")
        
        # Check if we have a cached repair
        cache_key = str(input_path)
        if cache_key in self._repair_cache:
            repaired_path = self._repair_cache[cache_key]
            logger.info(f"Using cached repaired PDF: {repaired_path}")
            return process_func(repaired_path, *args, **kwargs)
        
        # Try to repair the PDF
        logger.info(f"Attempting to repair {input_path}")
        success, repaired_path = self.repair_service.repair_pdf(input_path)
        
        if success and repaired_path:
            # Cache the repaired path
            self._repair_cache[cache_key] = repaired_path
            
            # Try processing the repaired PDF
            try:
                return process_func(repaired_path, *args, **kwargs)
            except Exception as e:
                logger.error(f"Processing still failed after repair: {e}")
                raise
        else:
            raise ValueError(f"Unable to repair PDF: {input_path}")
    
    def bates_stamp_pdf(self, input_pdf_path: Path, output_pdf_path: Path,
                        start_bates_number: int, bates_prefix: str = "", num_digits: int = 6):
        """Bates stamp with automatic repair."""
        return self.process_with_repair(
            input_pdf_path,
            self.base_processor.bates_stamp_pdf,
            output_pdf_path,
            start_bates_number,
            bates_prefix,
            num_digits
        )
    
    def exhibit_mark_pdf(self, input_pdf_path: Path, output_pdf_path: Path, exhibit_id: str):
        """Exhibit mark with automatic repair."""
        return self.process_with_repair(
            input_pdf_path,
            self.base_processor.exhibit_mark_pdf,
            output_pdf_path,
            exhibit_id
        )