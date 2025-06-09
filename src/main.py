import argparse
import logging
from pathlib import Path
import sys
from datetime import datetime

# Ensure src directory is in Python path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src import config
from src.llm_handler import LLMCategorizer
from src.pdf_processor import PDFProcessor
from src.pdf_repair import RobustPDFProcessor
from src.vector_processor import VectorProcessor
from src.db_storage import PostgresStorage
from src.document_orchestrator import DocumentOrchestrator, ProcessingResult
from src.batch_processor import BatchProcessor, AdaptiveBatchProcessor
from src.cache_manager import ModelResultCache, CachedLLMHandler
from src.document_analyzer import DocumentAnalyzer
from src.utils import setup_logging, ensure_dir_exists

logger = logging.getLogger(__name__)


def main():
    """Main entry point using LangChain orchestration."""
    setup_logging()
    
    # Log LangSmith configuration if enabled
    if config.LANGSMITH_TRACING:
        logger.info(f"LangSmith tracing enabled for project: {config.LANGSMITH_PROJECT}")
        logger.info(f"LangSmith endpoint: {config.LANGSMITH_ENDPOINT}")
    else:
        logger.debug("LangSmith tracing disabled")

    parser = argparse.ArgumentParser(
        description="AI-Powered Bates Numbering & Exhibit Marking Tool"
    )
    parser.add_argument(
        "--input_dir", 
        type=Path, 
        default=project_root / config.DEFAULT_INPUT_DIR,
        help="Directory containing input PDF documents."
    )
    parser.add_argument(
        "--output_dir", 
        type=Path, 
        default=project_root / config.DEFAULT_OUTPUT_DIR,
        help="Directory to save processed files and log."
    )
    parser.add_argument(
        "--llm_model", 
        type=str, 
        default=None,
        help="LLM model to use (overrides environment configuration)."
    )
    parser.add_argument(
        "--exhibit_prefix", 
        type=str, 
        default="Exhibit ",
        help="Prefix for exhibit marks (e.g., 'Ex. ', 'Plaintiff Ex. ')."
    )
    parser.add_argument(
        "--bates_prefix", 
        type=str, 
        default="",
        help="Prefix for Bates numbers (e.g., 'ABC')."
    )
    parser.add_argument(
        "--bates_digits", 
        type=int, 
        default=6,
        help="Number of digits for Bates numbers (e.g., 6 for '000001')."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Process documents in batches (0 for sequential processing)."
    )
    parser.add_argument(
        "--use_optimized_batch",
        action="store_true",
        help="Use optimized batch processing with parallel categorization."
    )
    parser.add_argument(
        "--adaptive_batch",
        action="store_true",
        help="Use adaptive batch sizing based on performance."
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Maximum concurrent workers for batch processing."
    )
    
    args = parser.parse_args()

    # Validate configuration
    if config.LLM_PROVIDER == "openai" and not config.OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY is not configured. Please set it in your .env file.")
        sys.exit(1)
    elif config.LLM_PROVIDER == "ollama":
        logger.info("Using Ollama for local LLM processing.")

    if not args.input_dir.is_dir():
        logger.error(f"Input directory '{args.input_dir}' not found.")
        sys.exit(1)

    # Initialize components
    try:
        base_llm_categorizer = LLMCategorizer()
    except ValueError as e:
        logger.error(f"Failed to initialize LLM Categorizer: {e}")
        sys.exit(1)
    
    # Initialize caching if enabled
    if config.ENABLE_MODEL_CACHE:
        cache_dir = Path(config.CACHE_DIR) if config.ENABLE_DISK_CACHE else None
        model_cache = ModelResultCache(
            cache_dir=cache_dir,
            enable_disk_cache=config.ENABLE_DISK_CACHE,
            memory_cache_size=config.MEMORY_CACHE_SIZE
        )
        llm_categorizer = CachedLLMHandler(base_llm_categorizer, model_cache)
        logger.info("Model result caching enabled")
        if config.ENABLE_DISK_CACHE:
            logger.info(f"Disk cache enabled at: {cache_dir}")
    else:
        llm_categorizer = base_llm_categorizer
        model_cache = None
        
    base_pdf_processor = PDFProcessor()
    pdf_processor = RobustPDFProcessor(base_pdf_processor)
    
    # Initialize optional components
    vector_processor = None
    if config.ENABLE_VECTOR_SEARCH:
        try:
            vector_processor = VectorProcessor()
            logger.info("Vector search enabled - documents will be indexed for semantic search")
        except Exception as e:
            logger.warning(f"Failed to initialize vector processor: {e}")
            logger.warning("Continuing without vector search capability")
    
    postgres_storage = None
    if config.ENABLE_POSTGRES_STORAGE:
        try:
            postgres_storage = PostgresStorage(
                connection_string=config.POSTGRES_CONNECTION,
                pool_size=config.POSTGRES_POOL_SIZE
            )
            logger.info("PostgreSQL storage enabled - document text will be stored in database")
        except Exception as e:
            logger.warning(f"Failed to initialize PostgreSQL storage: {e}")
            logger.warning("Continuing without PostgreSQL storage")
    
    # Initialize orchestrator
    orchestrator = DocumentOrchestrator(
        llm_categorizer=llm_categorizer,
        pdf_processor=pdf_processor,
        vector_processor=vector_processor,
        postgres_storage=postgres_storage,
        output_dir=args.output_dir,
        exhibit_prefix=args.exhibit_prefix,
        bates_prefix=args.bates_prefix,
        bates_digits=args.bates_digits
    )
    
    # Add document analyzer for intelligent model selection
    if config.ENABLE_MULTI_MODEL and config.LLM_PROVIDER == "lmstudio":
        orchestrator.document_analyzer = DocumentAnalyzer()
        logger.info("Document analyzer enabled for intelligent model selection")
    
    # Get documents to process (case-insensitive PDF detection)
    pdf_files = list(args.input_dir.glob("*.pdf")) + list(args.input_dir.glob("*.PDF"))
    documents_to_process = sorted(
        pdf_files,
        key=lambda x: x.name.lower()
    )
    total_documents = len(documents_to_process)
    
    if total_documents == 0:
        logger.warning(f"No PDF documents found in '{args.input_dir}'.")
        sys.exit(0)
    
    logger.info(f"Found {total_documents} PDF documents in '{args.input_dir}'.")
    
    # Process documents
    start_time = datetime.now()
    
    # Create base run configuration for LangSmith tracing
    base_run_config = {
        "metadata": {
            "processing_mode": "batch" if args.batch_size > 0 and total_documents > args.batch_size else "sequential",
            "total_documents": total_documents,
            "output_directory": str(args.output_dir),
            "bates_prefix": args.bates_prefix,
            "exhibit_prefix": args.exhibit_prefix,
            "timestamp": datetime.now().isoformat()
        },
        "tags": [
            "bates_numbering",
            "legal_document_processing",
            f"llm_{config.LLM_PROVIDER}",
            f"total_docs_{total_documents}"
        ]
    }
    
    if args.use_optimized_batch or args.adaptive_batch:
        # Use optimized batch processor
        if args.adaptive_batch:
            logger.info("Using adaptive batch processing")
            batch_processor = AdaptiveBatchProcessor(
                orchestrator=orchestrator,
                max_workers=args.max_workers,
                batch_size=args.batch_size or 10,
                enable_prefetch=True
            )
        else:
            logger.info(f"Using optimized batch processing (batch size: {args.batch_size or 10})")
            batch_processor = BatchProcessor(
                orchestrator=orchestrator,
                max_workers=args.max_workers,
                batch_size=args.batch_size or 10,
                enable_prefetch=True
            )
        
        # Process all documents
        batch_result = batch_processor.process_documents(
            documents_to_process,
            start_bates=1,
            start_exhibit=1
        )
        
        logger.info(f"Batch processing complete:")
        logger.info(f"  Successful: {batch_result.successful}")
        logger.info(f"  Failed: {batch_result.failed}")
        logger.info(f"  Total time: {batch_result.total_time:.2f}s")
        logger.info(f"  Avg per doc: {batch_result.avg_time_per_doc:.2f}s")
        
        if batch_result.errors:
            logger.warning("Failed documents:")
            for doc_path, error in batch_result.errors:
                logger.warning(f"  {doc_path.name}: {error}")
        
        # Convert to standard results format for CSV writing
        all_results = []
        for result in batch_result.results:
            if result.get("success"):
                all_results.append(ProcessingResult(
                    success=True,
                    exhibit_id=result["exhibit_id"],
                    original_filename=result["path"].name,
                    final_filename=result.get("exhibit_path", Path()).name,
                    category=result["category"],
                    summary=result["summary"],
                    bates_range=result["bates_range"],
                    exhibit_path=result.get("exhibit_path"),
                    vector_chunks=result.get("vector_chunks"),
                    postgres_stored=result.get("postgres_stored", False)
                ))
    
    elif args.batch_size > 0 and total_documents > args.batch_size:
        # Standard batch processing
        logger.info(f"Processing documents in batches of {args.batch_size}")
        all_results = []
        
        # Add batch-specific metadata
        batch_run_config = base_run_config.copy()
        batch_run_config["metadata"]["batch_size"] = args.batch_size
        batch_run_config["tags"].append("batched_processing")
        
        for i in range(0, total_documents, args.batch_size):
            batch = documents_to_process[i:i + args.batch_size]
            # Track counters from previous batch
            if i == 0:
                batch_start = 1
                exhibit_start = 1
            else:
                # Get the last successful result to determine counters
                # The orchestrator returns the next counters after processing
                batch_start = orchestrator._next_bates if hasattr(orchestrator, '_next_bates') else 1
                exhibit_start = orchestrator._next_exhibit if hasattr(orchestrator, '_next_exhibit') else 1
            
            batch_num = i//args.batch_size + 1
            total_batches = (total_documents + args.batch_size - 1)//args.batch_size
            logger.info(f"Processing batch {batch_num}/{total_batches}")
            
            # Update run config with batch-specific info
            current_batch_config = batch_run_config.copy()
            current_batch_config["metadata"].update({
                "batch_number": batch_num,
                "total_batches": total_batches,
                "batch_documents": len(batch)
            })
            current_batch_config["tags"].append(f"batch_{batch_num}_of_{total_batches}")
            current_batch_config["run_name"] = f"BatesProcessing-Batch{batch_num}"
            
            batch_results = orchestrator.process_batch(
                batch, 
                starting_bates=batch_start,
                starting_exhibit=exhibit_start,
                run_config_template=current_batch_config
            )
            all_results.extend(batch_results)
    else:
        # Sequential processing (default)
        sequential_run_config = base_run_config.copy()
        sequential_run_config["tags"].append("sequential_processing")
        sequential_run_config["run_name"] = "BatesProcessing-Sequential"
        
        all_results = orchestrator.process_batch(
            documents_to_process,
            starting_bates=1,
            starting_exhibit=1,
            run_config_template=sequential_run_config
        )
    
    # Calculate statistics
    successful = sum(1 for r in all_results if r.success)
    failed = sum(1 for r in all_results if not r.success)
    
    # Generate CSV log
    csv_output_path = args.output_dir / "exhibit_log.csv"
    orchestrator.generate_csv_log(all_results, csv_output_path)
    logger.info(f"Exhibit log saved to: {csv_output_path}")
    
    # Display summary
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Total Documents: {total_documents}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Processing Time: {processing_time:.2f} seconds")
    print(f"Average Time per Document: {processing_time/total_documents:.2f} seconds")
    
    # Show category breakdown
    category_counts = {}
    for result in all_results:
        if result.success and result.category:
            category_counts[result.category] = category_counts.get(result.category, 0) + 1
    
    if category_counts:
        print("\nDocuments by Category:")
        for category, count in sorted(category_counts.items()):
            print(f"  {category}: {count}")
    
    # Show additional statistics
    if vector_processor:
        stats = orchestrator.get_stats()
        print(f"\nVector Store Statistics:")
        print(f"  Total Chunks: {stats.get('vector_chunks', 0)}")
        print(f"  Categories: {len(stats.get('vector_categories', []))}")
    
    if postgres_storage:
        pg_stats = postgres_storage.get_statistics()
        print(f"\nPostgreSQL Statistics:")
        print(f"  Documents: {pg_stats['overall']['total_documents']}")
        print(f"  Total Pages: {pg_stats['overall']['total_pages']}")
        print(f"  Total Text Size: {pg_stats['overall']['total_chars']:,} characters")
    
    if model_cache:
        cache_stats = model_cache.get_stats()
        print(f"\nCache Statistics:")
        print(f"  Memory Cache Size: {cache_stats['size']}/{cache_stats['max_size']}")
        print(f"  Cache Hits: {cache_stats['hits']}")
        print(f"  Cache Misses: {cache_stats['misses']}")
        print(f"  Hit Rate: {cache_stats['hit_rate']:.1%}")
        if cache_stats.get('disk_cache_enabled'):
            print(f"  Disk Cache Files: {cache_stats.get('disk_cache_files', 0)}")
    
    # Show failures if any
    if failed > 0:
        print("\nFailed Documents:")
        for result in all_results:
            if not result.success:
                print(f"  - {result.original_filename}: {result.error}")
    
    # Clean up
    if postgres_storage:
        postgres_storage.close()
    
    print("\n" + "="*60)
    print("All processing complete!")
    
    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()