"""
LM Studio embeddings using OpenAI-compatible API.

This module provides a LangChain-compatible embeddings class for LM Studio,
allowing the use of local embedding models through LM Studio's server.
"""

from typing import List
from langchain_core.embeddings import Embeddings
import requests
import logging

logger = logging.getLogger(__name__)


class LMStudioEmbeddings(Embeddings):
    """
    LM Studio embeddings implementation using OpenAI-compatible API.
    
    This class provides embeddings functionality for documents and queries
    using models served by LM Studio.
    """
    
    def __init__(self, base_url: str, model: str, timeout: int = 30):
        """
        Initialize LM Studio embeddings.
        
        Args:
            base_url: The base URL for LM Studio API (e.g., "http://localhost:1234/v1")
            model: The model ID to use for embeddings
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer lm-studio"  # Dummy token for LM Studio
        }
        
        # Verify the model is available
        self._verify_model()
    
    def _verify_model(self):
        """Verify that the specified model is available in LM Studio."""
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            models = response.json().get("data", [])
            model_ids = [m.get("id") for m in models]
            
            if self.model not in model_ids:
                logger.warning(
                    f"Model '{self.model}' not found in LM Studio. "
                    f"Available models: {model_ids}"
                )
        except Exception as e:
            logger.warning(f"Could not verify model availability: {e}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed search documents.
        
        Args:
            texts: List of text documents to embed
            
        Returns:
            List of embeddings, one for each text
        """
        embeddings = []
        
        # Process texts in batches for efficiency
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self._get_embeddings(batch)
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed query text.
        
        Args:
            text: The query text to embed
            
        Returns:
            Embedding for the query
        """
        embeddings = self._get_embeddings([text])
        return embeddings[0] if embeddings else []
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings from LM Studio for a batch of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        if not texts:
            return []
        
        try:
            # LM Studio expects either a single string or list of strings
            input_data = texts[0] if len(texts) == 1 else texts
            
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers=self.headers,
                json={
                    "model": self.model,
                    "input": input_data,
                    "encoding_format": "float"  # Optional parameter
                },
                timeout=self.timeout
            )
            
            response.raise_for_status()
            
            data = response.json()
            embeddings = []
            
            # Extract embeddings from response
            for item in data.get("data", []):
                embedding = item.get("embedding", [])
                if embedding:
                    embeddings.append(embedding)
                else:
                    logger.warning("Empty embedding received from LM Studio")
                    embeddings.append([])
            
            # Ensure we have the right number of embeddings
            while len(embeddings) < len(texts):
                logger.warning("Fewer embeddings than texts received")
                embeddings.append([])
            
            return embeddings[:len(texts)]
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout getting embeddings from LM Studio")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting embeddings from LM Studio: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    logger.error(f"LM Studio error detail: {error_detail}")
                except:
                    logger.error(f"LM Studio error response: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting embeddings: {e}")
            raise
    
    @property
    def embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Embedding dimension (attempts to determine dynamically)
        """
        try:
            # Get a sample embedding to determine dimension
            sample_embedding = self.embed_query("test")
            return len(sample_embedding)
        except Exception:
            # Common embedding dimensions as fallback
            if "nomic" in self.model.lower():
                return 768
            elif "bge" in self.model.lower():
                return 1024
            else:
                return 768  # Default fallback


def test_lmstudio_embeddings():
    """Test function for LM Studio embeddings."""
    from src.config import LMSTUDIO_HOST, LMSTUDIO_EMBEDDING_MODEL
    
    print("Testing LM Studio Embeddings...")
    print(f"Host: {LMSTUDIO_HOST}")
    print(f"Model: {LMSTUDIO_EMBEDDING_MODEL}")
    
    try:
        # Initialize embeddings
        embeddings = LMStudioEmbeddings(
            base_url=LMSTUDIO_HOST,
            model=LMSTUDIO_EMBEDDING_MODEL
        )
        
        # Test single query
        test_text = "This is a test legal document about contract breach."
        print(f"\nTesting single query: '{test_text}'")
        
        query_embedding = embeddings.embed_query(test_text)
        print(f"Embedding dimension: {len(query_embedding)}")
        print(f"First 5 values: {query_embedding[:5]}")
        
        # Test multiple documents
        test_docs = [
            "Medical report from Dr. Smith",
            "Invoice for legal services",
            "Defendant's motion to dismiss"
        ]
        print(f"\nTesting document batch: {len(test_docs)} documents")
        
        doc_embeddings = embeddings.embed_documents(test_docs)
        print(f"Number of embeddings: {len(doc_embeddings)}")
        for i, emb in enumerate(doc_embeddings):
            print(f"  Doc {i+1} dimension: {len(emb)}")
        
        print("\n✓ LM Studio embeddings test successful!")
        return True
        
    except Exception as e:
        print(f"\n✗ LM Studio embeddings test failed: {e}")
        return False


if __name__ == "__main__":
    # Run test when module is executed directly
    test_lmstudio_embeddings()