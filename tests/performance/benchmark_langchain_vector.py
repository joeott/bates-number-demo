#!/usr/bin/env python3
"""
Performance benchmark comparing old vs new LangChain vector processor implementation.
"""

import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.vector_processor import VectorProcessor
from src.vector_processor_backup import VectorProcessor as OldVectorProcessor


def benchmark_processor(processor_class, test_pdf: Path, label: str, use_old_api=False):
    """Benchmark a vector processor implementation."""
    print(f"\n{'='*50}")
    print(f"Benchmarking: {label}")
    print(f"{'='*50}")
    
    # Initialize processor
    start_init = time.time()
    processor = processor_class()
    init_time = time.time() - start_init
    print(f"Initialization time: {init_time:.3f}s")
    
    # Process document
    start_process = time.time()
    if use_old_api:
        # Old API expects path and metadata dict
        metadata = {
            "exhibit_number": 1,
            "category": "Test Document",
            "bates_start": "000001",
            "bates_end": "000001",
            "filename": test_pdf.name
        }
        chunk_ids, full_text, page_texts = processor.process_document(
            str(test_pdf),
            metadata
        )
    else:
        # New API expects individual parameters
        chunk_ids, full_text, page_texts = processor.process_document(
            test_pdf,
            exhibit_number=1,
            category="Test Document",
            bates_start="000001",
            bates_end="000001"
        )
    process_time = time.time() - start_process
    
    # Results
    print(f"Processing time: {process_time:.3f}s")
    print(f"Chunks created: {len(chunk_ids)}")
    print(f"Full text length: {len(full_text)} chars")
    print(f"Pages extracted: {len(page_texts)}")
    print(f"Avg time per chunk: {process_time/len(chunk_ids):.3f}s")
    
    return {
        "init_time": init_time,
        "process_time": process_time,
        "chunks": len(chunk_ids),
        "text_length": len(full_text),
        "pages": len(page_texts)
    }


def main():
    """Run performance comparison."""
    # Test file
    test_pdf = project_root / "test_input" / "CPACharge.pdf"
    
    if not test_pdf.exists():
        print(f"Error: Test file not found: {test_pdf}")
        sys.exit(1)
    
    print(f"Test file: {test_pdf.name}")
    
    # Benchmark old implementation
    old_results = benchmark_processor(
        OldVectorProcessor,
        test_pdf,
        "Old Implementation (Custom Components)",
        use_old_api=True
    )
    
    # Benchmark new implementation
    new_results = benchmark_processor(
        VectorProcessor,
        test_pdf,
        "New Implementation (LangChain)",
        use_old_api=False
    )
    
    # Comparison
    print(f"\n{'='*50}")
    print("Performance Comparison")
    print(f"{'='*50}")
    
    init_diff = new_results["init_time"] - old_results["init_time"]
    init_pct = (init_diff / old_results["init_time"]) * 100
    print(f"Initialization: {init_diff:+.3f}s ({init_pct:+.1f}%)")
    
    process_diff = new_results["process_time"] - old_results["process_time"]
    process_pct = (process_diff / old_results["process_time"]) * 100
    print(f"Processing: {process_diff:+.3f}s ({process_pct:+.1f}%)")
    
    print(f"\nChunks: {old_results['chunks']} -> {new_results['chunks']}")
    print(f"Text extracted: {old_results['text_length']} -> {new_results['text_length']} chars")
    
    # Overall assessment
    print(f"\n{'='*50}")
    print("Summary")
    print(f"{'='*50}")
    
    if process_pct < -10:
        print("✅ New implementation is significantly faster!")
    elif process_pct > 10:
        print("⚠️  New implementation is slower, but provides better standardization")
    else:
        print("✅ Performance is comparable, with improved maintainability")
    
    print("\nBenefits of LangChain implementation:")
    print("- Standard interfaces for embeddings and vector stores")
    print("- Better error handling and retry logic")
    print("- Easier to extend and maintain")
    print("- Access to LangChain ecosystem")


if __name__ == "__main__":
    main()