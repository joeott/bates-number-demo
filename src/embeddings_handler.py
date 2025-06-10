"""
Unified embeddings handler that respects LLM_PROVIDER configuration.
Provides a consistent interface for embeddings across different providers.
"""

import logging
import requests
import json
from typing import List, Optional, Union, Tuple
from langchain_core.embeddings import Embeddings

# Configuration imports
from src.config import (
    LLM_PROVIDER, OPENAI_API_KEY, OPENAI_API_BASE,
    OLLAMA_HOST, EMBEDDING_MODEL,
    LMSTUDIO_HOST, LMSTUDIO_EMBEDDING_MODEL
)

logger = logging.getLogger(__name__)


class LMStudioEmbeddings(Embeddings):
    """Custom embeddings class for LM Studio that handles proper request formatting."""
    
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip('/v1')  # Ensure no /v1 suffix
        self.model = model
        self.headers = {
            "Content-Type": "application/json"
        }
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        # LM Studio expects proper format
        payload = {
            "model": self.model,
            "input": texts  # Array of strings for batch processing
        }
        
        response = requests.post(
            f"{self.base_url}/v1/embeddings",
            headers=self.headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise ValueError(f"Embedding request failed: {response.status_code} - {response.text}")
        
        data = response.json()
        return [embedding["embedding"] for embedding in data["data"]]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        # For single query, still send as string (not array)
        payload = {
            "model": self.model,
            "input": text  # Single string
        }
        
        response = requests.post(
            f"{self.base_url}/v1/embeddings",
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
        self.base_url = base_url.rstrip('/v1')
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
            f"{self.base_url}/v1/embeddings",
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
            # Use custom LMStudioEmbeddings that properly formats requests
            return LMStudioEmbeddings(
                base_url=LMSTUDIO_HOST.replace('/v1', ''),  # Remove /v1 suffix
                model=LMSTUDIO_EMBEDDING_MODEL
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