# Context 59: Dual Embedding Models Implementation

## Date: 2025-06-10

### Overview
This document proposes an implementation strategy for using two distinct embedding models served by LM Studio:
1. **text-embedding-snowflake-arctic-embed-l-v2.0** - For generating document and query embeddings
2. **text-embedding-bge-reranker-v2-m3** - For reranking search results

Both models are served through the `/v1/embeddings` endpoint, requiring careful differentiation in the implementation.

### Current Issues
- The system is failing with "400 Bad Request" errors when trying to use embeddings
- Error message: `'input' field must be a string or an array of strings`
- The embeddings handler needs to properly format requests for LM Studio

### Understanding BGE Reranker vs Traditional Embeddings

#### Traditional Embeddings (Snowflake Arctic)
- **Purpose**: Convert text into dense vector representations
- **Usage**: Store in vector database, perform similarity search
- **Input**: Single text or array of texts
- **Output**: Array of float vectors

#### BGE Reranker Model
- **Purpose**: Score relevance between query-document pairs
- **Usage**: Rerank initial search results for better relevance
- **Input**: Query-document pairs
- **Output**: Relevance scores

### Proposed Implementation

#### 1. Enhanced Embeddings Handler

```python
# src/embeddings_handler.py

import logging
from typing import List, Optional, Union, Tuple
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
import requests
import json

logger = logging.getLogger(__name__)


class LMStudioEmbeddings(Embeddings):
    """Custom embeddings class for LM Studio that handles both embedding and reranking models."""
    
    def __init__(self, base_url: str, model: str, is_reranker: bool = False):
        self.base_url = base_url
        self.model = model
        self.is_reranker = is_reranker
        self.headers = {
            "Content-Type": "application/json"
        }
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        if self.is_reranker:
            raise ValueError("Reranker models cannot be used for document embedding")
        
        # LM Studio expects proper format
        payload = {
            "model": self.model,
            "input": texts  # Array of strings for batch processing
        }
        
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers=self.headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise ValueError(f"Embedding request failed: {response.status_code} - {response.text}")
        
        data = response.json()
        return [embedding["embedding"] for embedding in data["data"]]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        if self.is_reranker:
            raise ValueError("Reranker models cannot be used for query embedding")
        
        # For single query, still send as string (not array)
        payload = {
            "model": self.model,
            "input": text  # Single string
        }
        
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers=self.headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise ValueError(f"Embedding request failed: {response.status_code} - {response.text}")
        
        data = response.json()
        return data["data"][0]["embedding"]


class LMStudioReranker:
    """Reranker using BGE reranker model through LM Studio embeddings endpoint."""
    
    def __init__(self, base_url: str, model: str = "text-embedding-bge-reranker-v2-m3"):
        self.base_url = base_url
        self.model = model
        self.headers = {
            "Content-Type": "application/json"
        }
    
    def rerank(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[Tuple[int, float]]:
        """
        Rerank documents based on relevance to query.
        
        Returns: List of (index, score) tuples sorted by score descending
        """
        # BGE reranker expects query-document pairs
        # Format: [[query, doc1], [query, doc2], ...]
        pairs = [[query, doc] for doc in documents]
        
        payload = {
            "model": self.model,
            "input": pairs  # Array of query-document pairs
        }
        
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers=self.headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise ValueError(f"Reranking request failed: {response.status_code} - {response.text}")
        
        data = response.json()
        
        # Extract scores from embeddings (BGE reranker returns scores as embeddings)
        scores = []
        for i, embedding_data in enumerate(data["data"]):
            # BGE reranker typically returns a single score value
            score = embedding_data["embedding"][0] if isinstance(embedding_data["embedding"], list) else embedding_data["embedding"]
            scores.append((i, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        if top_k:
            scores = scores[:top_k]
        
        return scores


class UnifiedEmbeddings:
    """
    Unified embeddings handler that automatically selects the appropriate
    embeddings provider based on configuration.
    """
    
    def __init__(self, provider: Optional[str] = None):
        """
        Initialize embeddings with the specified or configured provider.
        
        Args:
            provider: Override the default provider from config
        """
        from src.config import (
            LLM_PROVIDER, OPENAI_API_KEY, OPENAI_API_BASE,
            OLLAMA_HOST, EMBEDDING_MODEL,
            LMSTUDIO_HOST, LMSTUDIO_EMBEDDING_MODEL
        )
        
        self.provider = provider or LLM_PROVIDER
        self.embeddings = self._init_embeddings()
        logger.info(f"Initialized embeddings with provider: {self.provider}")
    
    def _init_embeddings(self) -> Embeddings:
        """Initialize the appropriate embeddings based on provider."""
        if self.provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            
            if not OPENAI_API_KEY:
                raise ValueError("OpenAI API key is required for OpenAI embeddings")
            
            return OpenAIEmbeddings(
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_API_BASE
            )
            
        elif self.provider == "ollama":
            # Only import if using Ollama
            try:
                from langchain_ollama import OllamaEmbeddings
            except ImportError:
                raise ImportError(
                    "langchain-ollama is required for Ollama embeddings. "
                    "Install with: pip install langchain-ollama"
                )
            
            # Verify Ollama is running
            try:
                import ollama
                client = ollama.Client(host=OLLAMA_HOST)
                client.list()  # Test connection
            except Exception as e:
                raise ValueError(f"Cannot connect to Ollama at {OLLAMA_HOST}. Ensure Ollama is running: {e}")
            
            return OllamaEmbeddings(
                model=EMBEDDING_MODEL,
                base_url=OLLAMA_HOST
            )
            
        elif self.provider == "lmstudio":
            from src.config import LMSTUDIO_HOST, LMSTUDIO_EMBEDDING_MODEL
            
            # Use custom LMStudioEmbeddings that properly formats requests
            return LMStudioEmbeddings(
                base_url=LMSTUDIO_HOST.replace('/v1', ''),  # Remove /v1 suffix
                model=LMSTUDIO_EMBEDDING_MODEL,
                is_reranker=False
            )
            
        else:
            raise ValueError(f"Unknown embeddings provider: {self.provider}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        return self.embeddings.embed_query(text)
    
    @property
    def model(self) -> Embeddings:
        """Get the underlying embeddings model."""
        return self.embeddings


def get_embeddings(provider: Optional[str] = None) -> Embeddings:
    """
    Factory function to get embeddings instance.
    
    Args:
        provider: Optional provider override
        
    Returns:
        Configured embeddings instance
    """
    unified = UnifiedEmbeddings(provider)
    return unified.model


def get_reranker() -> Optional[LMStudioReranker]:
    """
    Factory function to get reranker instance if using LM Studio.
    
    Returns:
        LMStudioReranker instance or None if not using LM Studio
    """
    from src.config import LLM_PROVIDER, LMSTUDIO_HOST
    
    if LLM_PROVIDER == "lmstudio":
        return LMStudioReranker(
            base_url=LMSTUDIO_HOST.replace('/v1', ''),
            model="text-embedding-bge-reranker-v2-m3"
        )
    return None
```

#### 2. Updated Hybrid Search Implementation

```python
# In src/retrieval_agent/agent_tools.py

def perform_hybrid_search_with_reranking(
    query_text: str,
    k_results: int = 5,
    fetch_k: int = 15,
    metadata_filters: Optional[Dict[str, Any]] = None
) -> List[Dict]:
    """
    Perform hybrid search with reranking using BGE reranker.
    
    Args:
        query_text: The search query
        k_results: Final number of results to return
        fetch_k: Number of results to fetch before reranking
        metadata_filters: Optional metadata filters
    
    Returns:
        List of reranked search results
    """
    from src.embeddings_handler import get_reranker
    from src.hybrid_search import HybridSearcher
    
    # Initialize hybrid searcher
    searcher = HybridSearcher()
    
    # Perform initial search fetching more results for reranking
    initial_results = searcher.search(
        query=query_text,
        k=fetch_k,  # Fetch more for reranking
        metadata_filter=metadata_filters,
        search_method="hybrid"
    )
    
    if not initial_results:
        return []
    
    # Get reranker if available
    reranker = get_reranker()
    if reranker and len(initial_results) > k_results:
        # Extract documents for reranking
        documents = [r["chunk_text"] for r in initial_results]
        
        # Rerank documents
        reranked_indices = reranker.rerank(
            query=query_text,
            documents=documents,
            top_k=k_results
        )
        
        # Return reranked results
        reranked_results = []
        for idx, score in reranked_indices:
            result = initial_results[idx].copy()
            result["rerank_score"] = score
            reranked_results.append(result)
        
        return reranked_results
    
    # If no reranker or too few results, return as is
    return initial_results[:k_results]
```

### Configuration Updates

```bash
# .env file
LLM_PROVIDER=lmstudio
LMSTUDIO_HOST=http://localhost:1234/v1
LMSTUDIO_EMBEDDING_MODEL=text-embedding-snowflake-arctic-embed-l-v2.0
LMSTUDIO_RERANKER_MODEL=text-embedding-bge-reranker-v2-m3
```

### Implementation Notes

1. **Model Differentiation**: The implementation differentiates between embedding and reranking models by:
   - Using separate classes for each purpose
   - Storing model names in configuration
   - Using different input formats (single/array for embeddings vs pairs for reranking)

2. **Error Handling**: The current 400 errors are likely due to:
   - Incorrect input format (not sending as proper string/array)
   - Missing model specification in the request
   - Incorrect endpoint URL format

3. **Reranking Strategy**:
   - Fetch more results initially (e.g., 15-20)
   - Use BGE reranker to score relevance
   - Return top-k based on reranking scores

4. **Fallback Behavior**:
   - If reranker is unavailable, fall back to standard search
   - If using providers other than LM Studio, skip reranking

### Testing Strategy

1. Test embeddings generation:
   ```python
   embeddings = get_embeddings()
   vector = embeddings.embed_query("test query")
   assert len(vector) > 0
   ```

2. Test reranking:
   ```python
   reranker = get_reranker()
   if reranker:
       scores = reranker.rerank(
           "test query",
           ["doc1", "doc2", "doc3"],
           top_k=2
       )
       assert len(scores) == 2
   ```

3. Integration test with search pipeline

### Next Steps

1. Implement the enhanced embeddings handler
2. Update hybrid search to use reranking
3. Test with both models loaded in LM Studio
4. Monitor performance and adjust fetch_k parameter
5. Add logging for debugging model selection

This implementation ensures proper differentiation between the two embedding models while maintaining compatibility with the existing system architecture.