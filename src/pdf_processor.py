import logging
from pathlib import Path
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.colors import black, red
from io import BytesIO

from src.config import (
    DEFAULT_BATES_FONT, DEFAULT_BATES_FONT_SIZE,
    DEFAULT_EXHIBIT_FONT, DEFAULT_EXHIBIT_FONT_SIZE,
    STAMP_MARGIN_POINTS
)

logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self, bates_font=DEFAULT_BATES_FONT, bates_font_size=DEFAULT_BATES_FONT_SIZE,
                 exhibit_font=DEFAULT_EXHIBIT_FONT, exhibit_font_size=DEFAULT_EXHIBIT_FONT_SIZE,
                 stamp_margin=STAMP_MARGIN_POINTS):
        self.bates_font = bates_font
        self.bates_font_size = bates_font_size
        self.exhibit_font = exhibit_font
        self.exhibit_font_size = exhibit_font_size
        self.stamp_margin = stamp_margin

    def _add_stamp_to_page(self, page, stamp_text: str, is_exhibit_stamp: bool, page_width, page_height):
        """
        Creates an overlay with the stamp text and merges it onto the page.
        is_exhibit_stamp: True for exhibit stamp (larger, potentially different font/position), False for Bates.
        """
        packet = BytesIO()
        # Use page dimensions for the canvas
        can = canvas.Canvas(packet, pagesize=(page_width, page_height))

        if is_exhibit_stamp:
            can.setFont(self.exhibit_font, self.exhibit_font_size)
            # Exhibit stamp in lower-right corner, above potential Bates
            # Adjust y_position if both are very close. For now, assume Bates is lower.
            x_position = page_width - self.stamp_margin - can.stringWidth(stamp_text, self.exhibit_font, self.exhibit_font_size)
            y_position = self.stamp_margin + self.bates_font_size + 5  # Slightly above bates
        else: # Bates stamp
            can.setFont(self.bates_font, self.bates_font_size)
            # Bates stamp in lower-right corner
            x_position = page_width - self.stamp_margin - can.stringWidth(stamp_text, self.bates_font, self.bates_font_size)
            y_position = self.stamp_margin

        # Use red color for exhibit stamps, black for Bates
        if is_exhibit_stamp:
            can.setFillColor(red)
        else:
            can.setFillColor(black)
        can.drawString(x_position, y_position, stamp_text)
        can.save()
        packet.seek(0)

        stamp_pdf = PdfReader(packet)
        page.merge_page(stamp_pdf.pages[0])
        return page

    def bates_stamp_pdf(self, input_pdf_path: Path, output_pdf_path: Path,
                        start_bates_number: int, bates_prefix: str = "", num_digits: int = 6) -> tuple[int, int]:
        """
        Adds Bates numbers to each page of a PDF.
        Returns a tuple (start_bates_for_this_doc, end_bates_for_this_doc).
        """
        try:
            reader = PdfReader(input_pdf_path)
            writer = PdfWriter()
            current_bates_number = start_bates_number
            
            first_bates_on_doc = f"{bates_prefix}{current_bates_number:0{num_digits}d}"

            for i, page in enumerate(reader.pages):
                bates_text = f"{bates_prefix}{current_bates_number:0{num_digits}d}"
                page_width = float(page.mediabox.width)
                page_height = float(page.mediabox.height)
                
                # Add Bates stamp
                page = self._add_stamp_to_page(page, bates_text, False, page_width, page_height)
                writer.add_page(page)
                
                if i == 0:
                    first_bates_on_doc = bates_text # Capture the actual first bates for this doc
                
                current_bates_number += 1
            
            last_bates_on_doc = f"{bates_prefix}{current_bates_number - 1:0{num_digits}d}"

            with open(output_pdf_path, "wb") as f:
                writer.write(f)
            
            logger.info(f"Bates stamped '{input_pdf_path.name}' to '{output_pdf_path.name}' (Bates: {first_bates_on_doc}-{last_bates_on_doc})")
            return first_bates_on_doc, last_bates_on_doc, current_bates_number # Return next bates number to use

        except Exception as e:
            logger.error(f"Error Bates stamping '{input_pdf_path.name}': {e}")
            # Propagate the error or handle as needed; here we return None to indicate failure
            return None, None, start_bates_number


    def exhibit_mark_pdf(self, input_pdf_path: Path, output_pdf_path: Path, exhibit_id: str):
        """
        Adds an exhibit mark to each page of a PDF.
        This typically takes a Bates-numbered PDF as input.
        The exhibit mark is placed above the Bates number.
        """
        try:
            reader = PdfReader(input_pdf_path)
            writer = PdfWriter()

            for page in reader.pages:
                page_width = float(page.mediabox.width)
                page_height = float(page.mediabox.height)

                # Add Exhibit stamp
                page = self._add_stamp_to_page(page, exhibit_id, True, page_width, page_height)
                writer.add_page(page)

            with open(output_pdf_path, "wb") as f:
                writer.write(f)
            logger.info(f"Exhibit marked '{input_pdf_path.name}' as '{exhibit_id}' to '{output_pdf_path.name}'")
            return True
        except Exception as e:
            logger.error(f"Error exhibit marking '{input_pdf_path.name}': {e}")
            return False