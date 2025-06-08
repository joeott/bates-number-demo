# Context 3: Local Model Implementation with Ollama

## Overview
This document describes the implementation plan for adding local model support to the Bates Numbering & Exhibit Marking Tool using Ollama, providing an alternative to OpenAI for organizations that require on-premises processing or want to avoid external API calls.

## Implementation Goals
1. **Dual Model Support**: Allow users to choose between OpenAI (cloud) or Ollama (local) via environment configuration
2. **Minimal Complexity**: Keep implementation simple for novice users with minimal additional scripts
3. **Seamless Switching**: Users can switch between providers by changing a single environment variable
4. **Self-Contained**: Include everything needed for local model operation

## Architecture Design

### Provider Selection Flow
```
.env configuration
    ↓
LLM_PROVIDER = "openai" or "ollama"
    ↓
LLMCategorizer class initialization
    ↓
Provider-specific implementation
    ├── OpenAI: Uses existing OpenAI client
    └── Ollama: Uses ollama Python library
```

## Implementation Steps

### 1. Update Requirements
Add Ollama to `requirements.txt`:
```
ollama>=0.1.0         # For local LLM support
```

### 2. Update Environment Configuration
Modify `.env.template` to include:
```
# LLM Provider Selection
LLM_PROVIDER="openai"  # Options: "openai" or "ollama"

# OpenAI Configuration (if using LLM_PROVIDER="openai")
OPENAI_API_KEY="sk-your_openai_api_key_here"
# OPENAI_API_BASE="your_custom_llm_api_base_url"
OPENAI_MODEL="gpt-4o-mini-2024-07-18"

# Ollama Configuration (if using LLM_PROVIDER="ollama")
OLLAMA_MODEL="llama3.2:3b"  # Or any installed Ollama model
OLLAMA_HOST="http://localhost:11434"  # Default Ollama endpoint
```

### 3. Update Configuration Module (`src/config.py`)
Add new configuration variables:
```python
# --- LLM Provider Configuration ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini-2024-07-18")

# Validate provider selection
if LLM_PROVIDER not in ["openai", "ollama"]:
    print(f"WARNING: Invalid LLM_PROVIDER '{LLM_PROVIDER}'. Defaulting to 'openai'.")
    LLM_PROVIDER = "openai"
```

### 4. Refactor LLM Handler (`src/llm_handler.py`)
Create a unified interface that supports both providers:

```python
import logging
from abc import ABC, abstractmethod
from openai import OpenAI
import ollama
from src.config import (
    OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_MODEL,
    OLLAMA_MODEL, OLLAMA_HOST, LLM_PROVIDER,
    CATEGORIZATION_SYSTEM_PROMPT, SUMMARIZATION_SYSTEM_PROMPT, 
    FILENAME_GENERATION_PROMPT
)

logger = logging.getLogger(__name__)

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 50) -> str:
        pass

class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str, api_base: str = None, model: str = "gpt-4o-mini-2024-07-18"):
        if not api_key:
            raise ValueError("OpenAI API key is required")
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = model
    
    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 50) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return completion.choices[0].message.content.strip()

class OllamaProvider(BaseLLMProvider):
    def __init__(self, model: str = "llama3.2:3b", host: str = "http://localhost:11434"):
        self.model = model
        self.client = ollama.Client(host=host)
        # Verify Ollama is running and model is available
        try:
            self.client.list()
        except Exception as e:
            raise ValueError(f"Cannot connect to Ollama at {host}. Ensure Ollama is running: {e}")
    
    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 50) -> str:
        # Combine prompts for Ollama format
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        response = self.client.generate(
            model=self.model,
            prompt=full_prompt,
            options={
                "temperature": temperature,
                "num_predict": max_tokens
            }
        )
        return response['response'].strip()

class LLMCategorizer:
    def __init__(self, provider: str = None):
        # Use configured provider if not specified
        provider = provider or LLM_PROVIDER
        
        logger.info(f"Initializing LLM with provider: {provider}")
        
        if provider == "openai":
            self.provider = OpenAIProvider(
                api_key=OPENAI_API_KEY,
                api_base=OPENAI_API_BASE,
                model=OPENAI_MODEL
            )
        elif provider == "ollama":
            self.provider = OllamaProvider(
                model=OLLAMA_MODEL,
                host=OLLAMA_HOST
            )
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
    
    def _call_llm(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        """Unified method to call the LLM provider"""
        try:
            return self.provider.complete(system_prompt, user_prompt, temperature)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def categorize_document(self, filename: str) -> str:
        """Categorizes a document based on its filename using an LLM."""
        try:
            logger.info(f"Attempting to categorize: {filename}")
            category = self._call_llm(
                CATEGORIZATION_SYSTEM_PROMPT,
                f"Filename: {filename}"
            )
            logger.info(f"LLM categorized '{filename}' as: {category}")
            
            # Validate category
            valid_categories = ["Pleading", "Medical Record", "Bill", "Correspondence", 
                              "Photo", "Video", "Documentary Evidence", "Uncategorized"]
            if category not in valid_categories:
                logger.warning(f"Invalid category '{category}' for '{filename}'. Defaulting to Uncategorized.")
                return "Uncategorized"
            return category
        except Exception as e:
            logger.error(f"Error during categorization for '{filename}': {e}")
            return "Uncategorized"
    
    def summarize_document(self, filename: str) -> str:
        """Creates a one-sentence summary of a document based on its filename."""
        try:
            logger.info(f"Attempting to summarize: {filename}")
            summary = self._call_llm(
                SUMMARIZATION_SYSTEM_PROMPT,
                f"Create a one-sentence summary for this document filename: {filename}",
                temperature=0.3
            )
            logger.info(f"LLM summary for '{filename}': {summary}")
            
            if not summary or len(summary) > 150:
                logger.warning(f"Summary too long or empty for '{filename}'. Using default.")
                return f"Document titled '{filename}'"
            return summary
        except Exception as e:
            logger.error(f"Error during summarization for '{filename}': {e}")
            return f"Document titled '{filename}'"
    
    def generate_descriptive_filename(self, filename: str) -> str:
        """Generates a descriptive filename for an exhibit."""
        try:
            logger.info(f"Attempting to generate descriptive filename for: {filename}")
            descriptive_name = self._call_llm(
                FILENAME_GENERATION_PROMPT,
                f"Generate a descriptive filename for: {filename}"
            )
            logger.info(f"Generated filename for '{filename}': {descriptive_name}")
            
            if not descriptive_name or len(descriptive_name) > 100:
                logger.warning(f"Generated filename too long or empty for '{filename}'. Using default.")
                return "Document"
            return descriptive_name
        except Exception as e:
            logger.error(f"Error during filename generation for '{filename}': {e}")
            return "Document"
```

### 5. Update Main Script
Modify `src/main.py` initialization to remove the model parameter:
```python
# Initialize components
try:
    llm_categorizer = LLMCategorizer()  # Provider selected from environment
except ValueError as e:
    logger.error(f"Failed to initialize LLM Categorizer: {e}")
    sys.exit(1)
```

### 6. Add Ollama Setup Script
Create `setup_ollama.py` for easy installation:
```python
#!/usr/bin/env python3
"""
Setup script for Ollama local model support.
This script helps users install and configure Ollama for the Bates numbering tool.
"""
import os
import sys
import subprocess
import platform

def check_ollama_installed():
    """Check if Ollama is installed"""
    try:
        subprocess.run(["ollama", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_ollama():
    """Provide instructions for installing Ollama"""
    system = platform.system().lower()
    
    print("\n🚀 Ollama Installation Instructions\n")
    
    if system == "darwin":  # macOS
        print("For macOS, run:")
        print("  curl -fsSL https://ollama.com/install.sh | sh")
    elif system == "linux":
        print("For Linux, run:")
        print("  curl -fsSL https://ollama.com/install.sh | sh")
    elif system == "windows":
        print("For Windows:")
        print("  1. Download from https://ollama.com/download/windows")
        print("  2. Run the installer")
    else:
        print("Please visit https://ollama.com for installation instructions")
    
    print("\nAfter installation, run this script again.")
    return False

def pull_model(model_name="llama3.2:3b"):
    """Pull the specified Ollama model"""
    print(f"\n📥 Pulling model: {model_name}")
    print("This may take a few minutes...")
    
    try:
        subprocess.run(["ollama", "pull", model_name], check=True)
        print(f"✅ Successfully pulled {model_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to pull model: {e}")
        return False

def test_ollama():
    """Test Ollama with a simple prompt"""
    print("\n🧪 Testing Ollama...")
    
    try:
        result = subprocess.run(
            ["ollama", "run", "llama3.2:3b", "Say 'Hello, legal world!' in 5 words or less"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✅ Ollama test successful: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ollama test failed: {e}")
        return False

def update_env_file():
    """Update .env file to use Ollama"""
    env_path = ".env"
    
    if not os.path.exists(env_path):
        print("\n⚠️  No .env file found. Creating from template...")
        if os.path.exists(".env.template"):
            with open(".env.template", "r") as template:
                content = template.read()
            with open(env_path, "w") as env_file:
                env_file.write(content)
        else:
            print("❌ No .env.template found. Please create .env manually.")
            return False
    
    print("\n📝 To use Ollama, update your .env file:")
    print("   LLM_PROVIDER=ollama")
    print("   OLLAMA_MODEL=llama3.2:3b")
    print("\n✅ Ollama setup complete!")
    return True

def main():
    print("🎯 Ollama Setup for Bates Numbering Tool")
    print("=" * 40)
    
    # Check if Ollama is installed
    if not check_ollama_installed():
        print("\n❌ Ollama is not installed.")
        if not install_ollama():
            sys.exit(1)
    else:
        print("\n✅ Ollama is installed")
    
    # Pull the default model
    if not pull_model():
        sys.exit(1)
    
    # Test Ollama
    if not test_ollama():
        print("\n⚠️  Ollama test failed, but you can still proceed.")
    
    # Update environment configuration
    update_env_file()

if __name__ == "__main__":
    main()
```

### 7. Update README
Add section for local model usage:
```markdown
## Using Local Models with Ollama

For organizations that prefer to run models locally without external API calls:

### Quick Setup
1. Run the Ollama setup script:
   ```bash
   python setup_ollama.py
   ```

2. Update your `.env` file:
   ```
   LLM_PROVIDER=ollama
   OLLAMA_MODEL=llama3.2:3b
   ```

3. Run the tool as normal:
   ```bash
   python src/main.py
   ```

### Manual Setup
1. Install Ollama from https://ollama.com
2. Pull a model: `ollama pull llama3.2:3b`
3. Update `.env` to set `LLM_PROVIDER=ollama`

### Supported Models
- `llama3.2:3b` (Default, 2GB) - Fast and efficient
- `llama3.2:1b` (1GB) - Smaller, faster
- `mistral:7b` (4GB) - More capable
- Any model from https://ollama.com/library
```

## Testing Strategy

### 1. Provider Switching Test
```bash
# Test with OpenAI
LLM_PROVIDER=openai python src/main.py

# Test with Ollama
LLM_PROVIDER=ollama python src/main.py
```

### 2. Error Handling Test
- Ollama not running
- Invalid model name
- Network issues
- Missing API keys

### 3. Performance Comparison
Track processing time for both providers on same document set.

## Benefits of This Implementation

1. **Minimal Changes**: Only 3 files modified (config.py, llm_handler.py, requirements.txt)
2. **User-Friendly**: Single environment variable switches providers
3. **Backward Compatible**: Existing OpenAI setup continues to work
4. **Self-Contained**: Setup script handles Ollama installation
5. **Flexible**: Easy to add more providers in the future

## Deployment Considerations

### For IT Teams
- Ollama runs completely offline after model download
- No data leaves the organization
- Models stored in `~/.ollama/models/`
- Resource requirements: 4-8GB RAM depending on model

### For End Users
- Run `setup_ollama.py` once
- Change one line in `.env` file
- Everything else works the same

## Future Enhancements

1. **Model Selection UI**: Add command-line option to select model
2. **Performance Metrics**: Log processing times for optimization
3. **Model Management**: Script to download/remove models
4. **Custom Prompts**: Provider-specific prompt optimization

## Implementation Timeline

1. **Phase 1** (2 hours): Core implementation
   - Update config.py
   - Refactor llm_handler.py
   - Update requirements.txt

2. **Phase 2** (1 hour): Setup automation
   - Create setup_ollama.py
   - Update documentation

3. **Phase 3** (1 hour): Testing
   - Test both providers
   - Verify error handling
   - Performance benchmarking

Total estimated time: 4 hours

## Success Criteria

1. Users can switch between OpenAI and Ollama with one config change
2. Both providers produce comparable categorization results
3. Setup process requires no technical expertise
4. Error messages are clear and actionable
5. Performance is acceptable (< 2 seconds per document)