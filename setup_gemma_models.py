#!/usr/bin/env python3
"""
Setup script to configure Gemma vision models for multi-model pipeline.
Updates .env file with the correct model assignments.
"""

import os
from pathlib import Path

def update_env_for_gemma():
    """Update .env file with Gemma model configuration."""
    
    # Gemma model assignments based on size and capability
    gemma_config = {
        'LMSTUDIO_VISUAL_MODEL': 'google/gemma-3-4b',           # Fast visual analysis
        'LMSTUDIO_REASONING_MODEL': 'google/gemma-3-12b',       # Balanced reasoning
        'LMSTUDIO_CATEGORIZATION_MODEL': 'google/gemma-3-4b',   # Fast categorization
        'LMSTUDIO_SYNTHESIS_MODEL': 'google/gemma-3-27b',       # Deep synthesis
        'LMSTUDIO_EMBEDDING_MODEL': 'text-embedding-snowflake-arctic-embed-l-v2.0',
        'ENABLE_MULTI_MODEL': 'true'
    }
    
    env_path = Path('.env')
    
    # Read existing .env
    if env_path.exists():
        with open(env_path, 'r') as f:
            lines = f.readlines()
    else:
        print("No .env file found. Creating from template...")
        template_path = Path('.env.template')
        if template_path.exists():
            with open(template_path, 'r') as f:
                lines = f.readlines()
        else:
            lines = []
    
    # Update or add Gemma configurations
    updated_lines = []
    updated_keys = set()
    
    for line in lines:
        # Check if this line contains one of our keys
        for key, value in gemma_config.items():
            if line.strip().startswith(f'{key}='):
                updated_lines.append(f'{key}={value}\n')
                updated_keys.add(key)
                break
        else:
            # Keep the original line if not updating
            updated_lines.append(line)
    
    # Add any missing keys
    for key, value in gemma_config.items():
        if key not in updated_keys:
            updated_lines.append(f'{key}={value}\n')
    
    # Write back to .env
    with open(env_path, 'w') as f:
        f.writelines(updated_lines)
    
    print("✅ Updated .env with Gemma model configuration:")
    print("\nModel Assignments:")
    print(f"  Visual Analysis: {gemma_config['LMSTUDIO_VISUAL_MODEL']} (3.03 GB)")
    print(f"  Reasoning: {gemma_config['LMSTUDIO_REASONING_MODEL']} (8.07 GB)")
    print(f"  Categorization: {gemma_config['LMSTUDIO_CATEGORIZATION_MODEL']} (3.03 GB)")
    print(f"  Synthesis: {gemma_config['LMSTUDIO_SYNTHESIS_MODEL']} (16.87 GB)")
    print(f"  Embeddings: {gemma_config['LMSTUDIO_EMBEDDING_MODEL']}")
    print(f"\n  Multi-Model: {gemma_config['ENABLE_MULTI_MODEL']}")
    
    print("\n📝 Next steps:")
    print("1. Ensure all Gemma models are loaded in LM Studio")
    print("2. Start LM Studio server (port 1234)")
    print("3. Run: python src/main.py --input_dir 'input_documents/Recamier v. YMCA'")
    print("\nFor testing specific documents, see context_44_multi_model_test_flow.md")

def verify_lmstudio_connection():
    """Verify LM Studio is running and models are available."""
    try:
        import requests
        response = requests.get("http://localhost:1234/v1/models", timeout=5)
        if response.status_code == 200:
            models = response.json().get('data', [])
            print(f"\n✅ LM Studio connected. Found {len(models)} models:")
            for model in models:
                print(f"   - {model.get('id', 'unknown')}")
            return True
        else:
            print("\n❌ LM Studio server returned error")
            return False
    except Exception as e:
        print(f"\n❌ Cannot connect to LM Studio: {e}")
        print("   Please start LM Studio and load the Gemma models")
        return False

if __name__ == "__main__":
    print("Gemma Model Configuration Setup")
    print("=" * 50)
    
    # Update configuration
    update_env_for_gemma()
    
    # Verify connection
    print("\nChecking LM Studio connection...")
    verify_lmstudio_connection()