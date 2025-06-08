#!/usr/bin/env python3
"""
Performance benchmark for vector search implementation.
Measures the processing time increase with vector indexing enabled.
"""

import time
import sys
import tempfile
import shutil
from pathlib import Path
import subprocess
import os

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def run_benchmark(enable_vector: bool, input_dir: Path, output_dir: Path) -> float:
    """Run the main.py script and measure execution time."""
    env = os.environ.copy()
    env['ENABLE_VECTOR_SEARCH'] = 'true' if enable_vector else 'false'
    
    start_time = time.time()
    
    result = subprocess.run([
        sys.executable,
        'src/main.py',
        '--input_dir', str(input_dir),
        '--output_dir', str(output_dir)
    ], env=env, capture_output=True, text=True)
    
    end_time = time.time()
    
    if result.returncode != 0:
        print(f"Error running benchmark: {result.stderr}")
        return -1
    
    return end_time - start_time


def main():
    """Run performance benchmarks."""
    print("=== Vector Search Performance Benchmark ===\n")
    
    # Check if we have test PDFs
    project_root = Path(__file__).resolve().parent.parent.parent
    test_input_dir = project_root / "test_performance_subset"
    
    if not test_input_dir.exists() or not list(test_input_dir.glob("*.pdf")):
        print("WARNING: No test PDFs found in test_performance_subset/")
        print("Using test_input/ instead")
        test_input_dir = project_root / "test_input"
        
        if not test_input_dir.exists() or not list(test_input_dir.glob("*.pdf")):
            print("ERROR: No PDF files found to benchmark")
            return 1
    
    pdf_count = len(list(test_input_dir.glob("*.pdf")))
    print(f"Found {pdf_count} PDF files to process\n")
    
    # Create temporary output directories
    temp_base = tempfile.mkdtemp()
    output_without_vector = Path(temp_base) / "without_vector"
    output_with_vector = Path(temp_base) / "with_vector"
    
    try:
        # Benchmark without vector search
        print("Running benchmark WITHOUT vector search...")
        time_without = run_benchmark(False, test_input_dir, output_without_vector)
        if time_without < 0:
            print("Failed to run benchmark without vector search")
            return 1
        print(f"Time without vector search: {time_without:.2f} seconds")
        
        # Clear any existing vector store
        vector_store_path = output_with_vector / "vector_store"
        if vector_store_path.exists():
            shutil.rmtree(vector_store_path)
        
        # Benchmark with vector search
        print("\nRunning benchmark WITH vector search...")
        time_with = run_benchmark(True, test_input_dir, output_with_vector)
        if time_with < 0:
            print("Failed to run benchmark with vector search")
            return 1
        print(f"Time with vector search: {time_with:.2f} seconds")
        
        # Calculate performance impact
        time_increase = time_with - time_without
        percent_increase = (time_increase / time_without) * 100
        
        print("\n=== Results ===")
        print(f"Documents processed: {pdf_count}")
        print(f"Time without vector: {time_without:.2f}s")
        print(f"Time with vector: {time_with:.2f}s")
        print(f"Time increase: {time_increase:.2f}s ({percent_increase:.1f}%)")
        
        # Check vector store was created
        if vector_store_path.exists():
            print(f"\nVector store created at: {vector_store_path}")
            # Try to get chunk count
            try:
                import chromadb
                client = chromadb.PersistentClient(path=str(vector_store_path))
                collection = client.get_collection("legal_documents")
                chunk_count = collection.count()
                print(f"Total chunks indexed: {chunk_count}")
                print(f"Average chunks per document: {chunk_count/pdf_count:.1f}")
            except Exception as e:
                print(f"Could not read chunk count: {e}")
        
        # Performance verdict
        print("\n=== Performance Verdict ===")
        if percent_increase < 20:
            print(f"✅ PASS: Processing time increased by {percent_increase:.1f}% (< 20% target)")
        else:
            print(f"❌ FAIL: Processing time increased by {percent_increase:.1f}% (> 20% target)")
        
        # Additional metrics
        if time_without > 0:
            docs_per_second_without = pdf_count / time_without
            docs_per_second_with = pdf_count / time_with
            print(f"\nThroughput without vector: {docs_per_second_without:.2f} docs/second")
            print(f"Throughput with vector: {docs_per_second_with:.2f} docs/second")
        
        return 0 if percent_increase < 20 else 1
        
    finally:
        # Cleanup
        if Path(temp_base).exists():
            shutil.rmtree(temp_base)
        print("\nBenchmark complete.")


if __name__ == "__main__":
    sys.exit(main())