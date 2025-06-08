Okay, this is an excellent project for a CLE! It combines practical utility with a demonstration of AI's capabilities.

Here's a detailed markdown guide for an agentic coding tool to build this Bates numbering and exhibit marking system.

```markdown
# Project: AI-Powered Bates Numbering & Exhibit Marking Tool

**Objective:** Create a Python-based command-line tool that processes a directory of PDF documents. It will:
1.  Use an OpenAI-compatible LLM to categorize documents based on their filenames (and optionally, first page content).
2.  Bates number all pages of all documents sequentially and save these versions.
3.  Mark each Bates-numbered document as a distinct exhibit (creating a new copy with an exhibit sticker).
4.  Generate a CSV log detailing each exhibit, its category, and its Bates number range.

## 1. Directory Structure

Establish the following directory structure for the project:

```
ai_bates_exhibit_tool/
├── .env                     # For API keys and sensitive configuration
├── .gitignore
├── README.md                # Basic usage instructions
├── requirements.txt         # Python dependencies
├── input_documents/         # User places their source PDFs here (create this, can be empty initially)
│   └── (example: contract.pdf, email_thread.pdf)
├── output/                  # Generated files will go here (tool will create this)
│   ├── bates_numbered/      # All documents with Bates numbers applied
│   ├── exhibits/            # Bates-numbered documents, also marked as exhibits
│   └── exhibit_log.csv      # CSV log of exhibits and Bates numbers
└── src/
    ├── __init__.py
    ├── main.py              # Main script, CLI entry point
    ├── llm_handler.py       # Handles interaction with the LLM for categorization
    ├── pdf_processor.py     # Handles PDF manipulation (Bates, exhibit stamps)
    ├── config.py            # Loads .env, stores configuration constants
    └── utils.py             # Utility functions (e.g., file handling, logging setup)
```

## 2. Setup Files

### 2.1. `.gitignore`

Create a `.gitignore` file to exclude common Python artifacts, environment files, and output directories from version control:

```
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
#  Usually these files are written by a python script from a template
#  before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Jupyter Notebook
.ipynb_checkpoints

# Output directory (if you don't want to commit example outputs)
# output/ # Keep this if you want to commit the structure, or add if you don't want to commit content
input_documents/* # Don't commit user's input documents by default
!input_documents/.gitkeep # Allow committing an empty input_documents folder
output/*
!output/.gitkeep # Allow committing an empty output folder

# IDE
.idea/
.vscode/
```
*Action: Create `.gitignore` with the content above.*

### 2.2. `requirements.txt`

Create a `requirements.txt` file listing necessary Python libraries:

```
openai>=1.0.0         # For interacting with OpenAI compatible LLMs
python-dotenv>=0.19.0 # For loading .env files
pypdf>=3.0.0          # For reading and manipulating PDF pages (PyPDF2 successor)
reportlab>=3.6.0      # For drawing text/stamps on PDFs
Pillow>=9.0.0         # reportlab dependency, good to have explicitly
```
*Action: Create `requirements.txt` with the content above.*

### 2.3. `.env` (Template)

Create a `.env.template` file as a guide for users. The actual `.env` file should be created by the user and not committed.

```
# Rename this file to .env and fill in your details
OPENAI_API_KEY="sk-your_openai_api_key_here"
# Optional: Specify a different OpenAI-compatible API base if not using OpenAI directly
# OPENAI_API_BASE="your_custom_llm_api_base_url"
```
*Action: Create `.env.template` with the content above. The user will manually create `.env` from this.*

### 2.4. `README.md`

Create a basic `README.md`:

```markdown
# AI-Powered Bates Numbering & Exhibit Marking Tool

This tool automates the categorization, Bates numbering, and exhibit marking of PDF documents for legal discovery.

## Features

*   Categorizes documents using an LLM (OpenAI compatible).
*   Sequentially Bates numbers all pages of all documents.
*   Marks each document as a distinct exhibit.
*   Generates a CSV log of exhibits with their Bates numbers and categories.

## Setup

1.  **Clone the repository (if applicable) or create the directory structure.**
2.  **Create a Python virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure API Key:**
    *   Copy `.env.template` to `.env`.
    *   Edit `.env` and add your `OPENAI_API_KEY`.
    *   If using a non-OpenAI LLM provider that's OpenAI API compatible, you might also need to set `OPENAI_API_BASE`.

## Usage

1.  Place your PDF documents into the `input_documents/` directory.
2.  Run the main script:
    ```bash
    python src/main.py
    ```
    Optional arguments:
    ```bash
    python src/main.py --input_dir path/to/your/docs --output_dir path/to/output --llm_model gpt-3.5-turbo --exhibit_prefix "Ex. "
    ```

    *   `--input_dir`: Directory containing input PDFs (defaults to `input_documents/`).
    *   `--output_dir`: Directory where processed files and log will be saved (defaults to `output/`).
    *   `--llm_model`: The LLM model to use for categorization (defaults to `gpt-3.5-turbo`).
    *   `--exhibit_prefix`: Prefix for exhibit marks (e.g., "Exhibit ", "Ex. ", "Plaintiff's Ex. "). Defaults to "Exhibit ".
    *   `--bates_prefix`: Prefix for Bates numbers (e.g., "ABC"). Defaults to empty string.
    *   `--bates_digits`: Number of digits for Bates numbers (e.g., 6 for 000001). Defaults to 6.

All processed files will be in the specified `output/` directory, organized into `bates_numbered/` and `exhibits/`, along with an `exhibit_log.csv`.
```
*Action: Create `README.md` with the content above.*

## 3. Source Code (`src/`)

### 3.1. `src/__init__.py`

*Action: Create an empty `src/__init__.py` file to mark `src` as a Python package.*

### 3.2. `src/config.py`

This file will load environment variables and store application-wide configurations.

*Action: Create `src/config.py` with the following content:*
```python
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- LLM Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE") # Optional, for self-hosted or alternative LLMs

# --- Default Paths ---
# Relative to the project root. main.py will resolve these.
DEFAULT_INPUT_DIR = "input_documents"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_BATES_SUBDIR = "bates_numbered"
DEFAULT_EXHIBITS_SUBDIR = "exhibits"
DEFAULT_CSV_LOG_NAME = "exhibit_log.csv"

# --- PDF Processing Configuration ---
DEFAULT_BATES_FONT = "Helvetica"
DEFAULT_BATES_FONT_SIZE = 8
DEFAULT_EXHIBIT_FONT = "Helvetica-Bold"
DEFAULT_EXHIBIT_FONT_SIZE = 10
STAMP_MARGIN_POINTS = 18  # Approx 0.25 inch from bottom and right edges

# --- LLM Prompting ---
CATEGORIZATION_SYSTEM_PROMPT = """
You are a helpful legal assistant. Your task is to categorize documents based on their filename.
The primary categories are:
- Correspondence (e.g., emails, letters)
- Pleadings (e.g., complaints, answers, motions)
- Discovery (e.g., interrogatories, requests for production, deposition transcripts)
- Contracts & Agreements
- Financial Records (e.g., invoices, bank statements, spreadsheets)
- Internal Memoranda & Reports
- Evidence & Exhibits (e.g., photos, diagrams, specific evidence items)
- Client Documents (e.g., intake forms, client-provided materials)
- Research & Notes
- Miscellaneous

Based *only* on the filename provided by the user, choose the most appropriate single category from the list above.
If the filename is ambiguous or doesn't clearly fit, classify it as 'Miscellaneous'.
Provide only the category name as your response. For example, if the filename is "MSA_final_signed.pdf", your response should be "Contracts & Agreements".
"""

# --- Error Handling & Logging ---
# (Could add logging levels, formats here if using a more complex logging setup via utils.py)

# --- Validation ---
if not OPENAI_API_KEY:
    # This check is more for guiding the developer; main.py will handle user-facing errors.
    print("WARNING: OPENAI_API_KEY is not set in the environment or .env file.")

```

### 3.3. `src/utils.py`

This file will contain helper functions.

*Action: Create `src/utils.py` with the following content:*
```python
import os
import logging
from pathlib import Path

def setup_logging(level=logging.INFO):
    """Configures basic logging."""
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_dir_exists(dir_path: Path):
    """Creates a directory if it doesn't exist."""
    dir_path.mkdir(parents=True, exist_ok=True)

def sanitize_filename(filename: str) -> str:
    """Removes or replaces characters that are problematic in filenames."""
    # Basic sanitization, can be expanded
    return "".join(c if c.isalnum() or c in ('.', '_', '-') else '_' for c in filename)

```

### 3.4. `src/llm_handler.py`

This file will manage interactions with the LLM for document categorization.

*Action: Create `src/llm_handler.py` with the following content:*
```python
import logging
from openai import OpenAI
from src.config import OPENAI_API_KEY, OPENAI_API_BASE, CATEGORIZATION_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class LLMCategorizer:
    def __init__(self, api_key: str, api_base: str = None, model: str = "gpt-3.5-turbo"):
        if not api_key:
            raise ValueError("OpenAI API key is required for LLM categorization.")
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = model

    def categorize_document(self, filename: str) -> str:
        """
        Categorizes a document based on its filename using an LLM.
        """
        try:
            logger.info(f"Attempting to categorize: {filename} using model {self.model}")
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": CATEGORIZATION_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Filename: {filename}"}
                ],
                temperature=0.2, # Low temperature for more deterministic category output
                max_tokens=50
            )
            category = completion.choices[0].message.content.strip()
            logger.info(f"LLM categorized '{filename}' as: {category}")
            
            # Basic validation of category format (e.g., not too long, no strange chars)
            # For simplicity, we assume the LLM follows instructions well.
            # A more robust solution might check against a predefined list of categories.
            if not category or len(category) > 50: # Arbitrary length limit
                logger.warning(f"Unexpected category format for '{filename}': '{category}'. Defaulting to Miscellaneous.")
                return "Miscellaneous"
            return category

        except Exception as e:
            logger.error(f"Error during LLM categorization for '{filename}': {e}")
            return "Miscellaneous" # Fallback category on error
```

### 3.5. `src/pdf_processor.py`

This file will handle all PDF manipulation tasks: Bates stamping and exhibit marking.

*Action: Create `src/pdf_processor.py` with the following content:*
```python
import logging
from pathlib import Path
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import black
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
            y_position = self.stamp_margin + (self.bates_font_size + 5 if not is_exhibit_stamp else 0) # Slightly above bates
        else: # Bates stamp
            can.setFont(self.bates_font, self.bates_font_size)
            # Bates stamp in lower-right corner
            x_position = page_width - self.stamp_margin - can.stringWidth(stamp_text, self.bates_font, self.bates_font_size)
            y_position = self.stamp_margin

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

```

### 3.6. `src/main.py`

This is the main script that orchestrates the entire process and provides the CLI.

*Action: Create `src/main.py` with the following content:*
```python
import argparse
import csv
import logging
from pathlib import Path
import sys

# Ensure src directory is in Python path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src import config
from src.llm_handler import LLMCategorizer
from src.pdf_processor import PDFProcessor
from src.utils import setup_logging, ensure_dir_exists, sanitize_filename

logger = logging.getLogger(__name__)

def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="AI-Powered Bates Numbering & Exhibit Marking Tool")
    parser.add_argument("--input_dir", type=Path, default=project_root / config.DEFAULT_INPUT_DIR,
                        help="Directory containing input PDF documents.")
    parser.add_argument("--output_dir", type=Path, default=project_root / config.DEFAULT_OUTPUT_DIR,
                        help="Directory to save processed files and log.")
    parser.add_argument("--llm_model", type=str, default="gpt-3.5-turbo",
                        help="OpenAI model to use for categorization (e.g., gpt-3.5-turbo, gpt-4).")
    parser.add_argument("--exhibit_prefix", type=str, default="Exhibit ",
                        help="Prefix for exhibit marks (e.g., 'Ex. ', 'Plaintiff Ex. ').")
    parser.add_argument("--bates_prefix", type=str, default="",
                        help="Prefix for Bates numbers (e.g., 'ABC').")
    parser.add_argument("--bates_digits", type=int, default=6,
                        help="Number of digits for Bates numbers (e.g., 6 for '000001').")
    
    args = parser.parse_args()

    if not config.OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY is not configured. Please set it in your .env file or environment variables.")
        sys.exit(1)

    if not args.input_dir.is_dir():
        logger.error(f"Input directory '{args.input_dir}' not found.")
        sys.exit(1)

    # Create output directories
    bates_output_dir = args.output_dir / config.DEFAULT_BATES_SUBDIR
    exhibits_output_dir = args.output_dir / config.DEFAULT_EXHIBITS_SUBDIR
    ensure_dir_exists(args.output_dir)
    ensure_dir_exists(bates_output_dir)
    ensure_dir_exists(exhibits_output_dir)

    # Initialize components
    try:
        llm_categorizer = LLMCategorizer(
            api_key=config.OPENAI_API_KEY,
            api_base=config.OPENAI_API_BASE, # Will be None if not set in .env
            model=args.llm_model
        )
    except ValueError as e:
        logger.error(f"Failed to initialize LLM Categorizer: {e}")
        sys.exit(1)
        
    pdf_processor = PDFProcessor()

    # --- Main Processing Logic ---
    documents_to_process = sorted([f for f in args.input_dir.glob("*.pdf") if f.is_file()])
    if not documents_to_process:
        logger.warning(f"No PDF files found in '{args.input_dir}'. Exiting.")
        sys.exit(0)

    logger.info(f"Found {len(documents_to_process)} PDF documents in '{args.input_dir}'.")

    exhibit_log_data = []
    current_bates_counter = 1
    current_exhibit_number = 1 # Could also be letters, or configurable

    for doc_path in documents_to_process:
        logger.info(f"Processing document: {doc_path.name}")

        # 1. Categorize using LLM
        category = llm_categorizer.categorize_document(doc_path.name)

        # 2. Bates Number
        sanitized_filename_base = sanitize_filename(doc_path.stem)
        bates_numbered_pdf_name = f"{sanitized_filename_base}_BATES.pdf"
        bates_numbered_output_path = bates_output_dir / bates_numbered_pdf_name
        
        bates_start, bates_end, next_bates_counter = pdf_processor.bates_stamp_pdf(
            input_pdf_path=doc_path,
            output_pdf_path=bates_numbered_output_path,
            start_bates_number=current_bates_counter,
            bates_prefix=args.bates_prefix,
            num_digits=args.bates_digits
        )

        if bates_start is None: # Error during Bates stamping
            logger.error(f"Skipping exhibit marking for {doc_path.name} due to Bates stamping error.")
            continue # Skip to next document
        
        current_bates_counter = next_bates_counter

        # 3. Mark as Exhibit (using the Bates-numbered file as source)
        exhibit_id_str = f"{args.exhibit_prefix.strip()} {current_exhibit_number}"
        exhibit_marked_pdf_name = f"{sanitized_filename_base}_EXH_{current_exhibit_number}.pdf"
        exhibit_marked_output_path = exhibits_output_dir / exhibit_marked_pdf_name

        success_exhibit_marking = pdf_processor.exhibit_mark_pdf(
            input_pdf_path=bates_numbered_output_path, # Source is the Bates-numbered PDF
            output_pdf_path=exhibit_marked_output_path,
            exhibit_id=exhibit_id_str
        )
        
        if not success_exhibit_marking:
            logger.error(f"Failed to mark exhibit for {bates_numbered_output_path.name}. It will not be added to the log.")
            continue

        # 4. Record for CSV Log
        exhibit_log_data.append({
            "Exhibit ID": exhibit_id_str,
            "Original Filename": doc_path.name,
            "Category": category,
            "Bates Start": bates_start,
            "Bates End": bates_end,
            "Bates Numbered File": bates_numbered_pdf_name,
            "Exhibit Marked File": exhibit_marked_pdf_name,
        })
        current_exhibit_number += 1

    # 5. Output CSV Log
    if exhibit_log_data:
        csv_output_path = args.output_dir / config.DEFAULT_CSV_LOG_NAME
        try:
            with open(csv_output_path, "w", newline="", encoding="utf-8") as csvfile:
                fieldnames = [
                    "Exhibit ID", "Original Filename", "Category", 
                    "Bates Start", "Bates End", 
                    "Bates Numbered File", "Exhibit Marked File"
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(exhibit_log_data)
            logger.info(f"Exhibit log successfully written to '{csv_output_path}'.")
        except IOError as e:
            logger.error(f"Error writing CSV log: {e}")
    else:
        logger.warning("No documents were successfully processed to create an exhibit log.")

    logger.info("Processing complete.")

if __name__ == "__main__":
    main()
```

## 4. Agent Instructions Summary

1.  Create the full directory structure as specified in Section 1.
2.  Populate `.gitignore`, `requirements.txt`, `.env.template`, and `README.md` as per Section 2.
3.  Create `src/__init__.py` (empty).
4.  Implement `src/config.py` as per Section 3.2.
5.  Implement `src/utils.py` as per Section 3.3.
6.  Implement `src/llm_handler.py` as per Section 3.4. Pay close attention to the `OpenAI` client initialization and the chat completion API call structure.
7.  Implement `src/pdf_processor.py` as per Section 3.5. This involves:
    *   Using `pypdf` for reading and writing PDF structures.
    *   Using `reportlab` to create stamp overlays.
    *   Careful calculation of stamp positions in the `_add_stamp_to_page` method, ensuring exhibit stamps are placed distinctly (e.g., above Bates if both are in the same corner). The current implementation places both in the bottom-right, with the exhibit stamp textually above Bates if needed (by adjusting `y_position` logic based on `is_exhibit_stamp`).
8.  Implement `src/main.py` as per Section 3.6. This script should:
    *   Use `argparse` for command-line arguments.
    *   Initialize and use `LLMCategorizer` and `PDFProcessor`.
    *   Iterate through PDF files in the input directory.
    *   Orchestrate the categorization, Bates numbering, and exhibit marking workflow for each file.
    *   Maintain a running Bates counter and exhibit counter.
    *   Collect data for and write the `exhibit_log.csv`.
    *   Include robust error handling and logging.

## 5. Testing and Refinement Notes for the Agent

*   **API Key**: Ensure the `OPENAI_API_KEY` (and `OPENAI_API_BASE` if applicable) are correctly loaded from `.env` and used by `LLMCategorizer`.
*   **PDF Libraries**: `pypdf` and `reportlab` can have complex interactions. Test with various PDF types if possible. The current stamping method (merging an overlay) is generally robust.
*   **Stamp Placement**: The exact coordinates for stamps in `_add_stamp_to_page` might need slight adjustments depending on typical page margins and desired aesthetics. The `STAMP_MARGIN_POINTS` constant provides a starting point. The logic tries to place the exhibit stamp slightly above the Bates number if both are in the bottom right.
*   **Error Handling**: The `main.py` script includes `try-except` blocks for major operations. Ensure individual file processing errors don't halt the entire batch.
*   **LLM Prompt**: The `CATEGORIZATION_SYSTEM_PROMPT` in `config.py` is crucial. It's designed for zero-shot categorization based on filename. For better accuracy, one might consider extracting the first page of text from the PDF and including it in the prompt, but this adds complexity (text extraction libraries, larger prompts). For this version, filename-based is simpler.
*   **Sequential Numbering**: Ensure Bates numbers increment correctly across all pages of all documents. The `current_bates_counter` in `main.py` and its update logic via `pdf_processor.bates_stamp_pdf` is key.
*   **Exhibit IDs**: The current implementation uses simple numerical exhibit IDs (e.g., "Exhibit 1", "Exhibit 2"). This could be made more flexible (e.g., letters, user-defined scheme).
*   **File Paths**: Using `pathlib.Path` throughout is recommended for robust path manipulation across operating systems.

This detailed guide should provide a clear roadmap for an agentic coding tool to construct the application. Good luck with your CLE presentation!
```