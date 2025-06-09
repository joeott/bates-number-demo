# Context 40: LM Studio + MLX Implementation Plan

## Executive Summary

This implementation plan adds MLX model support through LM Studio's OpenAI-compatible endpoints while preserving ALL existing functionality. The approach leverages LM Studio's ability to serve MLX models via standard OpenAI API, requiring minimal code changes.

## Architecture Overview

### Current State
- **Providers**: OpenAI API, Ollama
- **Framework**: LangChain with structured outputs
- **Pattern**: Provider abstraction in LLMCategorizer

### Target State
- **Providers**: OpenAI API, Ollama, LM Studio (serving MLX models)
- **Framework**: Same LangChain structure
- **Pattern**: Extend provider abstraction to include LM Studio

## Key Insight: LM Studio as OpenAI-Compatible Server

LM Studio provides OpenAI-compatible endpoints at `http://localhost:1234/v1`, allowing us to:
1. Use existing OpenAI code paths
2. Serve MLX-optimized models
3. Maintain all current functionality
4. Add provider flexibility

## Implementation Requirements

### 1. Dependencies (Minimal)
```bash
# No new dependencies required for basic integration!
# LM Studio acts as OpenAI-compatible server

# Optional: For advanced features
pip install lmstudio-sdk  # Only if using WebSocket features
```

### 2. LM Studio Setup
```bash
# 1. Download LM Studio from https://lmstudio.ai
# 2. Install MLX models through LM Studio UI:
#    - mlx-community/Qwen2.5-7B-Instruct-4bit
#    - mlx-community/Qwen2.5-VL-3B-Instruct-4bit
#    - mlx-community/Mistral-7B-Instruct-v0.3-4bit
# 3. Start server on port 1234 (default)
```

### 3. Configuration Changes

#### Update .env.template
```bash
# Existing providers
LLM_PROVIDER=openai  # Options: openai, ollama, lmstudio
OPENAI_API_KEY=your-api-key
OLLAMA_HOST=http://localhost:11434

# New LM Studio configuration
LMSTUDIO_HOST=http://localhost:1234/v1
LMSTUDIO_MODEL=mlx-community/Qwen2.5-7B-Instruct-4bit
LMSTUDIO_VISION_MODEL=mlx-community/Qwen2.5-VL-3B-Instruct-4bit
LMSTUDIO_EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5-GGUF

# Optional: Advanced LM Studio features
LMSTUDIO_MAX_TOKENS=2048
LMSTUDIO_CONTEXT_LENGTH=32768
```

## Code Implementation

### Phase 1: Extend Configuration (src/config.py)

```python
# Add to src/config.py (around line 13, after OLLAMA_HOST)

# LM Studio Configuration (OpenAI-compatible)
LMSTUDIO_HOST = os.getenv("LMSTUDIO_HOST", "http://localhost:1234/v1")
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "mlx-community/Qwen2.5-7B-Instruct-4bit")
LMSTUDIO_VISION_MODEL = os.getenv("LMSTUDIO_VISION_MODEL", "mlx-community/Qwen2.5-VL-3B-Instruct-4bit")
LMSTUDIO_EMBEDDING_MODEL = os.getenv("LMSTUDIO_EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5-GGUF")
LMSTUDIO_MAX_TOKENS = int(os.getenv("LMSTUDIO_MAX_TOKENS", "2048"))
LMSTUDIO_CONTEXT_LENGTH = int(os.getenv("LMSTUDIO_CONTEXT_LENGTH", "32768"))

# Update validation (line 16)
if LLM_PROVIDER not in ["openai", "ollama", "lmstudio"]:
    print(f"WARNING: Invalid LLM_PROVIDER '{LLM_PROVIDER}'. Defaulting to 'openai'.")
    LLM_PROVIDER = "openai"
```

### Phase 2: Extend LLM Handler (src/llm_handler.py)

```python
# Modify _init_llm method (starting at line 117)
def _init_llm(self, provider: str) -> BaseChatModel:
    """Initialize the appropriate LLM based on provider."""
    if provider == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key is required")
        
        return ChatOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_API_BASE,
            model=OPENAI_MODEL,
            temperature=0.2,
            max_tokens=50
        )
    elif provider == "ollama":
        # Existing Ollama code...
        try:
            import ollama
            client = ollama.Client(host=OLLAMA_HOST)
            client.list()  # Test connection
        except Exception as e:
            raise ValueError(f"Cannot connect to Ollama at {OLLAMA_HOST}. Ensure Ollama is running: {e}")
        
        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_HOST,
            temperature=0.2,
            num_predict=50
        )
    elif provider == "lmstudio":
        # LM Studio uses OpenAI-compatible API
        # Import at top: from src.config import LMSTUDIO_HOST, LMSTUDIO_MODEL, LMSTUDIO_MAX_TOKENS
        
        # Verify LM Studio is running
        try:
            import requests
            response = requests.get(f"{LMSTUDIO_HOST.replace('/v1', '')}/api/health", timeout=5)
            if response.status_code != 200:
                raise ValueError("LM Studio server not healthy")
        except Exception as e:
            raise ValueError(f"Cannot connect to LM Studio at {LMSTUDIO_HOST}. Ensure LM Studio is running: {e}")
        
        # Use ChatOpenAI with LM Studio endpoint
        return ChatOpenAI(
            api_key="lm-studio",  # LM Studio doesn't require API key
            base_url=LMSTUDIO_HOST,
            model=LMSTUDIO_MODEL,
            temperature=0.2,
            max_tokens=LMSTUDIO_MAX_TOKENS,
            model_kwargs={
                "max_tokens": LMSTUDIO_MAX_TOKENS,
                "temperature": 0.2
            }
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
```

### Phase 3: Extend Vector Search for MLX Embeddings

```python
# Create new file: src/lmstudio_embeddings.py

from typing import List
from langchain_core.embeddings import Embeddings
import requests
import logging

logger = logging.getLogger(__name__)

class LMStudioEmbeddings(Embeddings):
    """LM Studio embeddings using OpenAI-compatible API."""
    
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer lm-studio"  # Dummy token
        }
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        embeddings = []
        for text in texts:
            embedding = self._get_embedding(text)
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self._get_embedding(text)
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding from LM Studio."""
        try:
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers=self.headers,
                json={
                    "model": self.model,
                    "input": text
                }
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise
```

### Phase 4: Update Vector Search Integration

```python
# Modify src/vector_search.py to support LM Studio embeddings
# Add to imports:
from src.config import LLM_PROVIDER, LMSTUDIO_HOST, LMSTUDIO_EMBEDDING_MODEL
from src.lmstudio_embeddings import LMStudioEmbeddings

# Update get_embeddings function (around line 168):
def get_embeddings():
    """Get the appropriate embeddings based on provider."""
    if LLM_PROVIDER == "lmstudio":
        return LMStudioEmbeddings(
            base_url=LMSTUDIO_HOST,
            model=LMSTUDIO_EMBEDDING_MODEL
        )
    elif ENABLE_POSTGRES_STORAGE and ENABLE_CUSTOM_EMBEDDINGS:
        # Existing custom embeddings code...
    else:
        # Existing Ollama embeddings code...
```

### Phase 5: Add Vision Support for MLX Models

```python
# Extend src/pdf_processor.py for vision capabilities
# This is optional - only if using vision models

def process_with_vision(self, pdf_path: Path) -> Dict:
    """Process PDF with vision model if provider supports it."""
    if LLM_PROVIDER != "lmstudio":
        return None
    
    try:
        # Convert PDF page to image
        images = convert_from_path(pdf_path, first_page=1, last_page=1)
        if not images:
            return None
        
        # Save temporary image
        temp_image = Path("/tmp/temp_page.png")
        images[0].save(temp_image)
        
        # Use vision model through LM Studio
        import base64
        with open(temp_image, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode()
        
        response = requests.post(
            f"{LMSTUDIO_HOST}/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": LMSTUDIO_VISION_MODEL,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this document"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                    ]
                }],
                "max_tokens": 500
            }
        )
        
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Vision processing error: {e}")
        return None
```

## Testing Strategy

### 1. Basic Integration Test
```python
# tests/test_lmstudio_integration.py
import pytest
from src.llm_handler import LLMCategorizer
from src.config import LMSTUDIO_HOST

def test_lmstudio_connection():
    """Test LM Studio server is reachable."""
    import requests
    try:
        response = requests.get(f"{LMSTUDIO_HOST.replace('/v1', '')}/api/health")
        assert response.status_code == 200
    except:
        pytest.skip("LM Studio not running")

def test_lmstudio_categorization():
    """Test document categorization with LM Studio."""
    categorizer = LLMCategorizer(provider="lmstudio")
    
    # Test cases
    test_files = {
        "medical_report_2024.pdf": "Medical Record",
        "invoice_123.pdf": "Bill",
        "complaint_filed.pdf": "Pleading"
    }
    
    for filename, expected_category in test_files.items():
        result = categorizer.categorize_document(filename)
        # Allow some flexibility in categorization
        assert result in ["Medical Record", "Bill", "Pleading", "Documentary Evidence", expected_category]
```

### 2. Performance Comparison
```python
# scripts/benchmark_providers.py
import time
from src.llm_handler import LLMCategorizer

def benchmark_provider(provider: str, test_files: list):
    """Benchmark a provider."""
    categorizer = LLMCategorizer(provider=provider)
    
    start_time = time.time()
    results = []
    
    for filename in test_files:
        result = categorizer.process_document_parallel(filename)
        results.append(result)
    
    elapsed = time.time() - start_time
    
    return {
        "provider": provider,
        "files_processed": len(test_files),
        "total_time": elapsed,
        "avg_time_per_file": elapsed / len(test_files),
        "results": results
    }

# Run benchmarks
test_files = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]  # Add real test files
for provider in ["openai", "ollama", "lmstudio"]:
    try:
        stats = benchmark_provider(provider, test_files)
        print(f"\n{provider.upper()} Performance:")
        print(f"  Total time: {stats['total_time']:.2f}s")
        print(f"  Avg per file: {stats['avg_time_per_file']:.2f}s")
    except Exception as e:
        print(f"\n{provider.upper()} Error: {e}")
```

## Deployment Steps

### 1. Development Environment
```bash
# 1. Update configuration
cp .env.template .env
# Edit .env to add LM Studio settings

# 2. Start LM Studio
# Launch LM Studio app
# Load desired MLX model
# Start server (default port 1234)

# 3. Test connection
python -c "import requests; print(requests.get('http://localhost:1234/api/health').json())"

# 4. Run with LM Studio
export LLM_PROVIDER=lmstudio
python src/main.py
```

### 2. Production Deployment
```bash
# 1. Install LM Studio on production machine
# 2. Configure systemd service for LM Studio
# 3. Load models and configure auto-start
# 4. Update application config
# 5. Deploy application
```

## Model Recommendations for M4 Max

### Primary Models (via LM Studio)
1. **Text Generation**: `mlx-community/Qwen2.5-7B-Instruct-4bit`
   - Excellent performance on M4 Max
   - 32K context window
   - ~5GB memory

2. **Vision**: `mlx-community/Qwen2.5-VL-3B-Instruct-4bit`
   - Handles document images
   - Lightweight for vision tasks
   - ~4GB memory

3. **Embeddings**: `nomic-ai/nomic-embed-text-v1.5-GGUF`
   - Fast local embeddings
   - 768 dimensions
   - ~1GB memory

### Advanced Options
1. **Long Context**: `mlx-community/Qwen2.5-7B-Instruct-1M-4bit`
   - For multi-document analysis
   - Up to 250K practical context

2. **Complex Reasoning**: `mlx-community/Mistral-7B-Instruct-v0.3-4bit`
   - Alternative model
   - Good for fallback

## Advantages of This Approach

1. **Zero Breaking Changes**: All existing code continues to work
2. **Simple Integration**: LM Studio acts as drop-in OpenAI replacement
3. **Performance**: MLX models optimized for Apple Silicon
4. **Flexibility**: Easy to switch between providers
5. **Local Processing**: No data leaves the machine
6. **Cost Effective**: No API costs for local models

## Rollback Plan

If issues arise:
1. Change `LLM_PROVIDER` back to `openai` or `ollama`
2. No code changes required
3. All functionality preserved

## Monitoring and Debugging

### 1. LM Studio Logs
```bash
# Check LM Studio console for:
- Model loading status
- Request/response logs
- Performance metrics
```

### 2. Application Logs
```python
# Add to logging configuration
if LLM_PROVIDER == "lmstudio":
    logger.info(f"Using LM Studio at {LMSTUDIO_HOST}")
    logger.info(f"Model: {LMSTUDIO_MODEL}")
```

### 3. Health Checks
```python
# Add to src/utils.py
def check_lmstudio_health():
    """Check if LM Studio is healthy."""
    try:
        import requests
        response = requests.get(f"{LMSTUDIO_HOST.replace('/v1', '')}/api/health")
        return response.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}
```

## Conclusion

This implementation adds MLX model support through LM Studio with minimal changes:
- 2 file modifications (config.py, llm_handler.py)
- 1 optional new file (lmstudio_embeddings.py)
- No breaking changes
- Full backward compatibility
- Leverages existing OpenAI code paths

The approach prioritizes simplicity and functionality while enabling powerful MLX models on your M4 Max hardware.