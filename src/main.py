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
from datetime import datetime

logger = logging.getLogger(__name__)

def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="AI-Powered Bates Numbering & Exhibit Marking Tool")
    parser.add_argument("--input_dir", type=Path, default=project_root / config.DEFAULT_INPUT_DIR,
                        help="Directory containing input PDF documents.")
    parser.add_argument("--output_dir", type=Path, default=project_root / config.DEFAULT_OUTPUT_DIR,
                        help="Directory to save processed files and log.")
    parser.add_argument("--llm_model", type=str, default=None,
                        help="LLM model to use (overrides environment configuration).")
    parser.add_argument("--exhibit_prefix", type=str, default="Exhibit ",
                        help="Prefix for exhibit marks (e.g., 'Ex. ', 'Plaintiff Ex. ').")
    parser.add_argument("--bates_prefix", type=str, default="",
                        help="Prefix for Bates numbers (e.g., 'ABC').")
    parser.add_argument("--bates_digits", type=int, default=6,
                        help="Number of digits for Bates numbers (e.g., 6 for '000001').")
    
    args = parser.parse_args()

    # Validate configuration based on provider
    if config.LLM_PROVIDER == "openai" and not config.OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY is not configured. Please set it in your .env file or environment variables.")
        sys.exit(1)
    elif config.LLM_PROVIDER == "ollama":
        logger.info("Using Ollama for local LLM processing.")

    if not args.input_dir.is_dir():
        logger.error(f"Input directory '{args.input_dir}' not found.")
        sys.exit(1)

    # Create output directories
    bates_output_dir = args.output_dir / config.DEFAULT_BATES_SUBDIR
    exhibits_output_dir = args.output_dir / config.DEFAULT_EXHIBITS_SUBDIR
    ensure_dir_exists(args.output_dir)
    ensure_dir_exists(bates_output_dir)
    ensure_dir_exists(exhibits_output_dir)
    
    # Create category subdirectories
    categories = ["Pleading", "Medical Record", "Bill", "Correspondence", "Photo", "Video", "Documentary Evidence", "Uncategorized"]
    category_dirs = {}
    for category in categories:
        category_dir = exhibits_output_dir / sanitize_filename(category.lower().replace(" ", "_"))
        ensure_dir_exists(category_dir)
        category_dirs[category] = category_dir

    # Initialize components
    try:
        llm_categorizer = LLMCategorizer()  # Provider selected from environment
    except ValueError as e:
        logger.error(f"Failed to initialize LLM Categorizer: {e}")
        sys.exit(1)
        
    pdf_processor = PDFProcessor()
    
    # Initialize vector processor if enabled
    vector_processor = None
    if config.ENABLE_VECTOR_SEARCH:
        try:
            from src.vector_processor import VectorProcessor
            vector_processor = VectorProcessor(use_vision=False)  # Start with text extraction
            logger.info("Vector search enabled - documents will be indexed for semantic search")
        except Exception as e:
            logger.warning(f"Failed to initialize vector processor: {e}")
            logger.warning("Continuing without vector search capability")
    
    # Initialize PostgreSQL storage if enabled
    postgres_storage = None
    if config.ENABLE_POSTGRES_STORAGE:
        try:
            from src.db_storage import PostgresStorage
            postgres_storage = PostgresStorage(
                connection_string=config.POSTGRES_CONNECTION,
                pool_size=config.POSTGRES_POOL_SIZE
            )
            logger.info("PostgreSQL storage enabled - document text will be stored in database")
        except Exception as e:
            logger.warning(f"Failed to initialize PostgreSQL storage: {e}")
            logger.warning("Continuing without PostgreSQL storage capability")

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

        # 1. Categorize, summarize, and generate descriptive filename using LLM
        category = llm_categorizer.categorize_document(doc_path.name)
        summary = llm_categorizer.summarize_document(doc_path.name)
        descriptive_name = llm_categorizer.generate_descriptive_filename(doc_path.name)

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
        # Use AI-generated descriptive filename
        sanitized_descriptive = sanitize_filename(descriptive_name)
        exhibit_marked_pdf_name = f"Exhibit {current_exhibit_number} - {sanitized_descriptive}.pdf"
        # Place exhibit in category subfolder
        exhibit_marked_output_path = category_dirs[category] / exhibit_marked_pdf_name

        success_exhibit_marking = pdf_processor.exhibit_mark_pdf(
            input_pdf_path=bates_numbered_output_path, # Source is the Bates-numbered PDF
            output_pdf_path=exhibit_marked_output_path,
            exhibit_id=exhibit_id_str
        )
        
        if not success_exhibit_marking:
            logger.error(f"Failed to mark exhibit for {bates_numbered_output_path.name}. It will not be added to the log.")
            continue

        # 4. Process for vector search if enabled
        full_text = ""
        page_texts = []
        if vector_processor and success_exhibit_marking:
            try:
                metadata = {
                    "filename": exhibit_marked_pdf_name,
                    "category": category,
                    "exhibit_number": current_exhibit_number,
                    "bates_start": bates_start,
                    "bates_end": bates_end,
                    "summary": summary,
                    "processed_date": datetime.now().isoformat()
                }
                chunk_ids, full_text, page_texts = vector_processor.process_document(
                    str(exhibit_marked_output_path),
                    metadata
                )
                logger.info(f"Created {len(chunk_ids)} vector chunks for {exhibit_marked_pdf_name}")
            except Exception as e:
                logger.error(f"Vector processing failed for {exhibit_marked_pdf_name}: {e}")
                # Continue processing other documents
        
        # 5. Store in PostgreSQL if enabled
        if postgres_storage and success_exhibit_marking:
            try:
                # If we don't have text from vector processing, extract it
                if not full_text and not vector_processor:
                    try:
                        from src.vector_processor import TextExtractor
                        extractor = TextExtractor(use_vision=False)
                        extracted_pages = extractor.extract_text_from_pdf(str(exhibit_marked_output_path))
                        
                        page_texts = []
                        full_text_parts = []
                        for page_data in extracted_pages:
                            if 'raw_text' in page_data['content']:
                                page_text = page_data['content']['raw_text']
                                page_texts.append(page_text)
                                if page_text.strip():
                                    full_text_parts.append(page_text)
                        full_text = "\n\n".join(full_text_parts)
                    except Exception as e:
                        logger.error(f"Failed to extract text for PostgreSQL storage: {e}")
                
                if full_text:
                    document_id = postgres_storage.store_document_text(
                        exhibit_id=current_exhibit_number,
                        original_filename=doc_path.name,
                        exhibit_filename=exhibit_marked_pdf_name,
                        bates_start=bates_start,
                        bates_end=bates_end,
                        category=category,
                        full_text=full_text,
                        page_texts=page_texts if config.STORE_PAGE_LEVEL_TEXT else None
                    )
                    logger.info(f"Stored document {exhibit_marked_pdf_name} in PostgreSQL with ID: {document_id}")
            except Exception as e:
                logger.error(f"PostgreSQL storage failed for {exhibit_marked_pdf_name}: {e}")
                # Continue processing other documents

        # 5. Record for CSV Log
        exhibit_log_data.append({
            "Exhibit ID": exhibit_id_str,
            "Original Filename": doc_path.name,
            "Final Filename": exhibit_marked_pdf_name,
            "Category": category,
            "Summary": summary,
            "Bates Start": bates_start,
            "Bates End": bates_end,
            "Bates Numbered File": bates_numbered_pdf_name,
        })
        current_exhibit_number += 1

    # 5. Output CSV Log
    if exhibit_log_data:
        csv_output_path = args.output_dir / config.DEFAULT_CSV_LOG_NAME
        try:
            with open(csv_output_path, "w", newline="", encoding="utf-8") as csvfile:
                fieldnames = [
                    "Exhibit ID", "Original Filename", "Final Filename", "Category", "Summary",
                    "Bates Start", "Bates End", 
                    "Bates Numbered File"
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(exhibit_log_data)
            logger.info(f"Exhibit log successfully written to '{csv_output_path}'.")
        except IOError as e:
            logger.error(f"Error writing CSV log: {e}")
    else:
        logger.warning("No documents were successfully processed to create an exhibit log.")

    # Log vector store statistics if enabled
    if vector_processor:
        try:
            stats = vector_processor.get_stats()
            logger.info(f"Vector store statistics: {stats}")
        except Exception as e:
            logger.warning(f"Could not retrieve vector store stats: {e}")
    
    # Log PostgreSQL statistics if enabled
    if postgres_storage:
        try:
            stats = postgres_storage.get_statistics()
            logger.info(f"PostgreSQL storage statistics: {stats}")
        except Exception as e:
            logger.warning(f"Could not retrieve PostgreSQL stats: {e}")
        finally:
            postgres_storage.close()

    logger.info("Processing complete.")

if __name__ == "__main__":
    main()