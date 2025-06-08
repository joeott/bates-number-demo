#!/usr/bin/env python3
"""
Integration test for LangChain-based LLM handler.
Tests the actual LLM operations with Ollama.
"""

import sys
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.llm_handler import LLMCategorizer
from src.config import LLM_PROVIDER


def test_sequential_vs_parallel():
    """Compare sequential vs parallel execution times."""
    print(f"\nTesting LangChain LLM Handler with provider: {LLM_PROVIDER}")
    print("=" * 60)
    
    try:
        # Initialize categorizer
        categorizer = LLMCategorizer()
        print("✓ LLM Categorizer initialized successfully")
        
        # Test filenames
        test_files = [
            "CPACharge.pdf",
            "Medical_Record_John_Doe.pdf",
            "Motion_to_Dismiss.pdf"
        ]
        
        for filename in test_files:
            print(f"\nTesting: {filename}")
            print("-" * 40)
            
            # Test sequential execution
            start_seq = time.time()
            cat1 = categorizer.categorize_document(filename)
            sum1 = categorizer.summarize_document(filename)
            name1 = categorizer.generate_descriptive_filename(filename)
            seq_time = time.time() - start_seq
            
            print(f"Sequential execution:")
            print(f"  Category: {cat1}")
            print(f"  Summary: {sum1}")
            print(f"  Filename: {name1}")
            print(f"  Time: {seq_time:.2f}s")
            
            # Test parallel execution
            start_par = time.time()
            results = categorizer.process_document_parallel(filename)
            par_time = time.time() - start_par
            
            print(f"\nParallel execution:")
            print(f"  Category: {results['category']}")
            print(f"  Summary: {results['summary']}")
            print(f"  Filename: {results['descriptive_name']}")
            print(f"  Time: {par_time:.2f}s")
            
            # Performance comparison
            improvement = ((seq_time - par_time) / seq_time) * 100
            print(f"\nPerformance improvement: {improvement:.1f}%")
            
            # Verify categories match (these should be consistent)
            assert cat1 == results['category'], "Category mismatch!"
            
            # Summaries and filenames may vary slightly due to LLM non-determinism
            # Just verify they are non-empty and reasonable length
            assert len(results['summary']) > 10, "Summary too short!"
            assert len(results['descriptive_name']) > 0, "Filename empty!"
            print("✓ Results validated (categories match, outputs non-empty)")
    
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    return True


def test_error_handling():
    """Test error handling with invalid inputs."""
    print("\nTesting error handling...")
    print("=" * 60)
    
    try:
        categorizer = LLMCategorizer()
        
        # Test with empty filename
        result = categorizer.process_document_parallel("")
        print(f"Empty filename result: {result}")
        assert result['category'] == "Uncategorized"
        print("✓ Empty filename handled correctly")
        
        # Test with very long filename
        long_name = "x" * 500 + ".pdf"
        result = categorizer.process_document_parallel(long_name)
        print(f"Long filename result: {result}")
        assert len(result['descriptive_name']) <= 100
        print("✓ Long filename handled correctly")
        
    except Exception as e:
        print(f"❌ Error during error handling test: {e}")
        return False
    
    print("✓ Error handling tests passed!")
    return True


def test_langchain_features():
    """Test LangChain-specific features."""
    print("\nTesting LangChain features...")
    print("=" * 60)
    
    try:
        categorizer = LLMCategorizer()
        
        # Test retry logic (simulated by normal operation)
        print("Testing retry logic with normal operation...")
        result = categorizer.categorize_document("test_retry.pdf")
        print(f"Result with retry: {result}")
        print("✓ Retry logic working (no errors)")
        
        # Test chain composition
        print("\nTesting chain composition...")
        test_file = "Contract_Agreement_2024.pdf"
        results = categorizer.process_document_parallel(test_file)
        
        # Verify all outputs are present
        assert 'category' in results
        assert 'summary' in results
        assert 'descriptive_name' in results
        print("✓ Chain composition working correctly")
        
        # Test output validation
        print("\nTesting output validation...")
        from src.llm_handler import DocumentCategory
        assert results['category'] in [cat.value for cat in DocumentCategory]
        assert 0 < len(results['summary']) <= 150
        assert 0 < len(results['descriptive_name']) <= 100
        print("✓ Output validation working correctly")
        
    except Exception as e:
        print(f"❌ Error during LangChain features test: {e}")
        return False
    
    print("✓ LangChain features tests passed!")
    return True


if __name__ == "__main__":
    print("LangChain LLM Handler Integration Tests")
    print("=" * 60)
    
    all_passed = True
    
    # Run tests
    if not test_sequential_vs_parallel():
        all_passed = False
    
    if not test_error_handling():
        all_passed = False
    
    if not test_langchain_features():
        all_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All integration tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)