#!/usr/bin/env python3
"""
Test script for multi-model LM Studio integration.
Tests both backward compatibility and multi-model functionality.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.llm_handler import LLMCategorizer
from src.config import ENABLE_MULTI_MODEL, LMSTUDIO_MODEL_MAPPING, LLM_PROVIDER


def test_backward_compatibility():
    """Test that single-model mode still works."""
    print("\n" + "=" * 60)
    print("Testing Backward Compatibility (Single Model)")
    print("=" * 60)
    
    # Force single model mode
    os.environ["ENABLE_MULTI_MODEL"] = "false"
    
    try:
        categorizer = LLMCategorizer(provider="lmstudio")
        print("✓ Single model initialization successful")
        
        # Test categorization
        test_filename = "motion_for_summary_judgment_2024.pdf"
        category = categorizer.categorize_document(test_filename)
        print(f"✓ Categorization: '{test_filename}' -> '{category}'")
        
        # Test parallel processing
        results = categorizer.process_document_parallel(test_filename)
        print(f"✓ Parallel processing completed")
        print(f"  - Category: {results['category']}")
        print(f"  - Summary: {results['summary'][:50]}...")
        print(f"  - Descriptive name: {results['descriptive_name']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in backward compatibility test: {e}")
        return False
    finally:
        # Reset environment
        os.environ.pop("ENABLE_MULTI_MODEL", None)


def test_multi_model():
    """Test multi-model functionality."""
    print("\n" + "=" * 60)
    print("Testing Multi-Model Pipeline")
    print("=" * 60)
    
    # Show current configuration
    print("\nCurrent model mapping:")
    for task, model in LMSTUDIO_MODEL_MAPPING.items():
        if model and model != "none":
            print(f"  {task}: {model}")
    
    print(f"\nMulti-model enabled: {ENABLE_MULTI_MODEL}")
    
    try:
        categorizer = LLMCategorizer(provider="lmstudio")
        
        if categorizer.multi_model_enabled:
            print("✓ Multi-model initialization successful")
            print(f"✓ Loaded {len(categorizer.models)} models:")
            for task, model in categorizer.models.items():
                print(f"  - {task}: {model}")
        else:
            print("⚠ Multi-model not enabled (need multiple different models configured)")
            return False
        
        # Test document processing
        test_cases = [
            "medical_records_patient_john_doe.pdf",
            "invoice_legal_services_12345.pdf",
            "motion_to_dismiss_defendant.pdf",
            "deposition_transcript_smith_2024.pdf"
        ]
        
        print("\nTesting document processing with multiple models:")
        for filename in test_cases:
            try:
                results = categorizer.process_document_parallel(filename)
                print(f"\n✓ Processed: {filename}")
                print(f"  - Category: {results['category']}")
                print(f"  - Summary: {results['summary'][:60]}...")
                print(f"  - Descriptive: {results['descriptive_name']}")
            except Exception as e:
                print(f"\n✗ Error processing {filename}: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in multi-model test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_discovery():
    """Test model discovery and mapping."""
    print("\n" + "=" * 60)
    print("Testing Model Discovery Service")
    print("=" * 60)
    
    try:
        from src.model_discovery import ModelDiscoveryService
        
        service = ModelDiscoveryService()
        profiles = service.discover_and_profile_models()
        
        if profiles:
            print(f"✓ Discovered {len(profiles)} models")
            
            # Show task mapping
            mapping = service.generate_task_mapping()
            print("\n✓ Generated task mapping:")
            for task, model in mapping.items():
                print(f"  {task}: {model}")
            
            # Show memory usage
            total_memory = sum(p.estimated_memory_gb for p in profiles.values())
            print(f"\n✓ Total estimated memory: {total_memory:.1f}GB")
            
            return True
        else:
            print("✗ No models discovered")
            return False
            
    except Exception as e:
        print(f"✗ Error in model discovery test: {e}")
        return False


def main():
    """Run all tests."""
    print("Multi-Model LM Studio Integration Test")
    print("=" * 60)
    print(f"Provider: {LLM_PROVIDER}")
    
    if LLM_PROVIDER != "lmstudio":
        print("\n⚠ LLM_PROVIDER is not set to 'lmstudio'")
        print("Please set LLM_PROVIDER=lmstudio in your .env file")
        return
    
    # Run tests
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Model Discovery
    if test_model_discovery():
        tests_passed += 1
    
    # Test 2: Backward Compatibility
    if test_backward_compatibility():
        tests_passed += 1
    
    # Test 3: Multi-Model Pipeline
    if test_multi_model():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("\n✓ All tests passed! Multi-model system is working correctly.")
    else:
        print(f"\n⚠ {total_tests - tests_passed} test(s) failed.")
    
    print("\nTo use multi-model pipeline:")
    print("1. Ensure multiple models are loaded in LM Studio")
    print("2. Update .env with the configuration shown above")
    print("3. Set ENABLE_MULTI_MODEL=true (or auto)")
    print("4. Run your normal document processing")


if __name__ == "__main__":
    main()