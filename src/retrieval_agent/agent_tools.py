# File: src/retrieval_agent/agent_tools.py
"""
LangChain tools for the Iterative Retrieval Agent.

This module defines the tools the agent can use, primarily wrapping
existing functionality from the main document processing pipeline.
"""

from typing import List, Dict, Optional, Any
import logging
from pathlib import Path

from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain.retrievers import ContextualCompressionRetriever
# Note: LLMChainExtractor and LLMChainFilter may need to be imported differently
# or implemented manually for compression functionality
import requests

# Import your existing VectorSearcher and its config
# Assuming vector_search.py has been refactored to use Langchain components
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))  # Add project root to path

from src.vector_search import VectorSearcher
from src.config import VECTOR_STORE_PATH, POSTGRES_CONNECTION, POSTGRES_POOL_SIZE, ENABLE_POSTGRES_STORAGE  # For default path
from src.hybrid_search import HybridSearcher, SearchMethod

# Import agent-specific config
from .agent_config import NUM_RESULTS_PER_SUB_QUERY
from . import agent_config

logger = logging.getLogger(__name__)

# Global instance or factory function for VectorSearcher to manage resources
# For simplicity in a CLI context, creating it per call is okay,
# but for an API, you'd manage its lifecycle.
_vector_searcher_instance = None
_hybrid_searcher_instance = None

def get_vector_searcher():
    """Get or create a singleton VectorSearcher instance."""
    global _vector_searcher_instance
    if _vector_searcher_instance is None:
        # Ensure VECTOR_STORE_PATH is correctly resolved
        # VECTOR_STORE_PATH should be a Path object from config
        vector_store_path = str(VECTOR_STORE_PATH)
        logger.info(f"Initializing VectorSearcher with path: {vector_store_path}")
        _vector_searcher_instance = VectorSearcher(vector_store_path=vector_store_path)
    return _vector_searcher_instance

def get_hybrid_searcher():
    """Get or create a singleton HybridSearcher instance."""
    global _hybrid_searcher_instance
    if _hybrid_searcher_instance is None:
        # Initialize HybridSearcher with PostgreSQL config if available
        postgres_config = None
        if ENABLE_POSTGRES_STORAGE:
            postgres_config = {
                'connection_string': POSTGRES_CONNECTION,
                'pool_size': POSTGRES_POOL_SIZE
            }
            logger.info("Initializing HybridSearcher with PostgreSQL support")
        else:
            logger.info("Initializing HybridSearcher with vector search only")
        
        _hybrid_searcher_instance = HybridSearcher(
            vector_store_path=str(VECTOR_STORE_PATH),
            postgres_config=postgres_config
        )
        logger.info("HybridSearcher initialized.")
    return _hybrid_searcher_instance


class LMStudioReranker(BaseDocumentCompressor):
    """LM Studio-based reranker implementing BaseDocumentCompressor interface.
    
    This implementation uses embeddings similarity for reranking since BGE reranker
    models in LM Studio are typically accessed through the embeddings API.
    """
    
    def __init__(self, base_url: str, model: str, top_n: int = 5, timeout: int = 30):
        """Initialize the LM Studio reranker.
        
        Args:
            base_url: LM Studio API URL (e.g., "http://localhost:1234/v1")
            model: Model name/ID in LM Studio (e.g., "gpustack/bge-reranker-v2-m3-GGUF")
            top_n: Number of top documents to return
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.top_n = top_n
        self.timeout = timeout
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer lm-studio"
        }
        logger.info(f"LMStudioReranker initialized with model: {model}")
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from LM Studio for reranking."""
        try:
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers=self.headers,
                json={
                    "model": self.model,
                    "input": texts,
                    "encoding_format": "float"
                },
                timeout=self.timeout
            )
            
            response.raise_for_status()
            data = response.json()
            
            embeddings = []
            for item in data.get("data", []):
                embedding = item.get("embedding", [])
                embeddings.append(embedding)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error getting embeddings for reranking: {e}")
            return []
    
    def _compute_similarity_scores(self, query: str, documents: List[str]) -> List[float]:
        """Compute similarity scores between query and documents.
        
        BGE reranker models are designed to encode query-document pairs together
        for cross-attention, so we concatenate them with a separator.
        """
        # For BGE reranker, we encode query-document pairs together
        pairs = [f"{query} [SEP] {doc}" for doc in documents]
        
        # Get embeddings for all pairs
        pair_embeddings = self._get_embeddings(pairs)
        
        if not pair_embeddings:
            return [0.0] * len(documents)
        
        # For reranker models, the embedding itself represents the relevance score
        # We'll use the first dimension or magnitude as the score
        scores = []
        for embedding in pair_embeddings:
            if embedding:
                # Use the magnitude of the embedding vector as score
                # Alternatively, could use the first dimension if model outputs that way
                import numpy as np
                score = np.linalg.norm(embedding)
                scores.append(float(score))
            else:
                scores.append(0.0)
        
        # Normalize scores to 0-1 range
        if scores:
            min_score = min(scores)
            max_score = max(scores)
            if max_score > min_score:
                scores = [(s - min_score) / (max_score - min_score) for s in scores]
        
        return scores
    
    def compress_documents(
        self,
        documents: List[Document],
        query: str,
        callbacks: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """Rerank documents using LM Studio model."""
        if not documents:
            return documents
        
        try:
            # Extract text content from documents
            doc_texts = [doc.page_content for doc in documents]
            
            # Get reranking scores
            scores = self._compute_similarity_scores(query, doc_texts)
            
            # Sort documents by score
            doc_scores = list(zip(documents, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Take top N and add scores to metadata
            reranked_docs = []
            for doc, score in doc_scores[:self.top_n]:
                # Create new document with updated metadata
                new_metadata = doc.metadata.copy()
                new_metadata["rerank_score"] = float(score)
                new_metadata["relevance_score"] = float(score)
                
                reranked_doc = Document(
                    page_content=doc.page_content,
                    metadata=new_metadata
                )
                reranked_docs.append(reranked_doc)
            
            logger.info(f"Reranked {len(documents)} documents, returning top {len(reranked_docs)}")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            return documents[:self.top_n] if len(documents) > self.top_n else documents


# Global reranker instance
_reranker_instance = None

# Global compressor instances
_llm_extractor_instance = None
_llm_filter_instance = None

def get_reranker():
    """Get or create a singleton reranker instance."""
    global _reranker_instance
    if _reranker_instance is None and agent_config.ENABLE_RERANKING:
        try:
            # Import config for LM Studio settings
            from src.config import LMSTUDIO_HOST
            
            _reranker_instance = LMStudioReranker(
                base_url=LMSTUDIO_HOST,
                model=agent_config.RERANKER_MODEL,
                top_n=agent_config.RERANKER_TOP_N
            )
        except Exception as e:
            logger.warning(f"Failed to initialize reranker: {e}")
    return _reranker_instance


def get_llm_extractor():
    """Get or create a singleton LLM extractor instance."""
    global _llm_extractor_instance
    if _llm_extractor_instance is None and agent_config.ENABLE_CONTEXTUAL_COMPRESSION:
        try:
            # Temporarily disabled - LLMChainExtractor import needs to be fixed
            logger.warning("LLM extractor temporarily disabled - import needs fixing")
            return None
        except Exception as e:
            logger.warning(f"Failed to initialize LLM extractor: {e}")
    return _llm_extractor_instance


def compress_results(results: List[Dict], query: str, max_results: Optional[int] = None) -> List[Dict]:
    """Apply contextual compression to search results.
    
    Args:
        results: List of search results to compress
        query: The original query for context
        max_results: Maximum number of results to return (if None, use config)
        
    Returns:
        List of compressed results
    """
    if not agent_config.ENABLE_CONTEXTUAL_COMPRESSION or not results:
        return results
    
    extractor = get_llm_extractor()
    if not extractor:
        return results
    
    try:
        # Convert results to Documents for compression
        docs = [Document(
            page_content=r["text"],
            metadata=r.get("metadata", {})
        ) for r in results]
        
        # Apply compression
        compressed_docs = extractor.compress_documents(
            documents=docs,
            query=query,
            callbacks=None
        )
        
        # Convert back to result format
        compressed_results = []
        for i, doc in enumerate(compressed_docs):
            # Find the original result to preserve all fields
            for orig in results:
                if i < len(results) and orig["text"][:100] in doc.page_content or doc.page_content in orig["text"]:
                    result = orig.copy()
                    result["text"] = doc.page_content  # Use compressed text
                    result["compressed"] = True
                    compressed_results.append(result)
                    break
        
        # If we couldn't match compressed docs back to originals, use what we have
        if not compressed_results:
            for doc in compressed_docs:
                compressed_results.append({
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance": doc.metadata.get("relevance_score", 1.0),
                    "compressed": True
                })
        
        # Limit results if specified
        if max_results:
            compressed_results = compressed_results[:max_results]
        
        logger.info(f"Compressed {len(results)} results to {len(compressed_results)} results")
        return compressed_results
        
    except Exception as e:
        logger.error(f"Error during contextual compression: {e}")
        return results

@tool
def perform_vector_search(
    query_text: str, 
    k_results: int = NUM_RESULTS_PER_SUB_QUERY, 
    metadata_filters: Optional[Dict[str, Any]] = None
) -> List[Dict]:
    """
    Searches the legal document vector store for passages relevant to the query_text.
    
    Args:
        query_text: The text to search for.
        k_results: The number of results to return.
        metadata_filters: A dictionary of metadata to filter by (e.g., {"category": "Pleading", "exhibit_number": 1}).
    
    Returns:
        A list of search results, each with 'text', 'metadata', and 'relevance'.
    """
    try:
        searcher = get_vector_searcher()
        
        # Adapt filter keys if your VectorSearcher.search expects different names
        # The refactored VectorSearcher expects: category, exhibit_number
        # So, ensure metadata_filters aligns with that or adapt here.
        
        formatted_filters = {}
        if metadata_filters:
            if "category" in metadata_filters:
                formatted_filters["category"] = metadata_filters["category"]
            if "exhibit_number" in metadata_filters:
                try:
                    formatted_filters["exhibit_number"] = int(metadata_filters["exhibit_number"])
                except (ValueError, TypeError):
                    # Handle cases where exhibit_number might not be a valid int from LLM
                    logger.warning(f"Invalid exhibit_number filter: {metadata_filters.get('exhibit_number')}")
            # Add more filter mappings as needed
            
            # Log the filters being applied
            logger.info(f"Applying filters: {formatted_filters}")
        
        # Execute the search
        search_results = searcher.search(
            query=query_text,
            n_results=k_results,
            # Pass filters directly if VectorSearcher.search() supports **kwargs for them
            # or pass them as specific arguments if defined.
            # Based on the vector_search.py, it takes category and exhibit_number as named args
            category=formatted_filters.get("category"),
            exhibit_number=formatted_filters.get("exhibit_number")
        )
        
        # Ensure output is serializable and contains what the agent expects
        # The current VectorSearcher.search returns list of dicts with expected fields
        logger.info(f"Vector search returned {len(search_results)} results")
        
        # Validate and clean the results
        cleaned_results = []
        for result in search_results:
            if isinstance(result, dict) and "text" in result:
                cleaned_results.append({
                    "text": result.get("text", ""),
                    "metadata": result.get("metadata", {}),
                    "relevance": result.get("relevance", 0.0),
                    # Include other fields that might be useful
                    "filename": result.get("filename", "Unknown"),
                    "category": result.get("category", "Uncategorized"),
                    "exhibit_number": result.get("exhibit_number"),
                    "bates_start": result.get("bates_start"),
                    "bates_end": result.get("bates_end"),
                    "page": result.get("page_number")
                })
        
        # Apply reranking if enabled and we have enough results
        if agent_config.ENABLE_RERANKING and len(cleaned_results) > agent_config.RERANKER_TOP_N:
            reranker = get_reranker()
            if reranker:
                logger.info(f"Applying reranking to {len(cleaned_results)} vector search results")
                
                # Convert to Document format for reranker
                docs = [Document(
                    page_content=r["text"], 
                    metadata=r.get("metadata", {})
                ) for r in cleaned_results]
                
                # Apply reranking
                reranked_docs = reranker.compress_documents(
                    documents=docs,
                    query=query_text,
                    callbacks=None
                )
                
                # Convert back to expected format
                reranked_results = []
                for doc in reranked_docs:
                    # Find original result to preserve all fields
                    for orig in cleaned_results:
                        if orig["text"] == doc.page_content:
                            result = orig.copy()
                            result["relevance"] = doc.metadata.get("relevance_score", 1.0)
                            result["rerank_score"] = doc.metadata.get("rerank_score", 1.0)
                            reranked_results.append(result)
                            break
                
                logger.info(f"Reranking complete, returning top {len(reranked_results)} results")
                cleaned_results = reranked_results
        
        # Apply contextual compression if enabled
        if agent_config.ENABLE_CONTEXTUAL_COMPRESSION:
            compressed_results = compress_results(cleaned_results, query_text)
            return compressed_results
        
        return cleaned_results
        
    except Exception as e:
        logger.error(f"Error in perform_vector_search: {str(e)}", exc_info=True)
        # Return empty results rather than failing completely
        return []

@tool
def perform_hybrid_search(
    query_text: str,
    k_results: int = NUM_RESULTS_PER_SUB_QUERY,
    metadata_filters: Optional[Dict[str, Any]] = None,
    search_method: str = "hybrid"
) -> List[Dict]:
    """
    Performs hybrid search combining vector and PostgreSQL full-text search.
    
    Args:
        query_text: The text to search for.
        k_results: The number of results to return.
        metadata_filters: A dictionary of metadata to filter by (e.g., {"category": "Pleading", "exhibit_number": 1}).
        search_method: Search method to use ("vector", "postgres", or "hybrid").
    
    Returns:
        A list of search results, each with 'text', 'metadata', and 'relevance'.
    """
    try:
        searcher = get_hybrid_searcher()
        
        # Convert search method string to enum
        method_map = {
            "vector": SearchMethod.VECTOR,
            "postgres": SearchMethod.POSTGRES,
            "hybrid": SearchMethod.HYBRID
        }
        method = method_map.get(search_method.lower(), SearchMethod.HYBRID)
        
        # Log search parameters
        logger.info(f"Performing {search_method} search with query: '{query_text}'")
        if metadata_filters:
            logger.info(f"Applying filters: {metadata_filters}")
        
        # Execute the search
        search_results = searcher.search(
            query=query_text,
            method=method,
            limit=k_results,
            filters=metadata_filters
        )
        
        # Convert SearchResult objects to dictionaries
        cleaned_results = []
        for result in search_results:
            cleaned_result = {
                "text": result.text,
                "metadata": {
                    "document_id": result.document_id,
                    "filename": result.filename,
                    "category": result.category,
                    "exhibit_number": result.exhibit_number,
                    "bates_range": result.bates_range,
                    "page": result.page,
                    "source": result.source
                },
                "relevance": result.score,
                "filename": result.filename,
                "category": result.category,
                "exhibit_number": result.exhibit_number,
                "bates_start": result.bates_range.split('-')[0] if '-' in result.bates_range else result.bates_range,
                "bates_end": result.bates_range.split('-')[1] if '-' in result.bates_range else result.bates_range,
                "page": result.page
            }
            
            # Add source-specific scores if available
            if result.vector_score is not None:
                cleaned_result["vector_score"] = result.vector_score
            if result.postgres_score is not None:
                cleaned_result["postgres_score"] = result.postgres_score
                
            cleaned_results.append(cleaned_result)
        
        logger.info(f"Hybrid search ('{search_method}') returned {len(cleaned_results)} results")
        
        # Apply reranking if enabled
        if agent_config.ENABLE_RERANKING and len(cleaned_results) > agent_config.RERANKER_TOP_N:
            reranker = get_reranker()
            if reranker:
                logger.info(f"Applying reranking to {len(cleaned_results)} results")
                
                # Convert to Document format for reranker
                docs = [Document(
                    page_content=r["text"], 
                    metadata=r.get("metadata", {})
                ) for r in cleaned_results]
                
                # Apply reranking
                reranked_docs = reranker.compress_documents(
                    documents=docs,
                    query=query_text,
                    callbacks=None
                )
                
                # Convert back to expected format
                reranked_results = []
                for doc in reranked_docs:
                    # Find original result to preserve all fields
                    for orig in cleaned_results:
                        if orig["text"] == doc.page_content:
                            result = orig.copy()
                            result["relevance"] = doc.metadata.get("relevance_score", 1.0)
                            result["rerank_score"] = doc.metadata.get("rerank_score", 1.0)
                            reranked_results.append(result)
                            break
                
                logger.info(f"Reranking complete, returning top {len(reranked_results)} results")
                cleaned_results = reranked_results
        
        # Apply contextual compression if enabled
        if agent_config.ENABLE_CONTEXTUAL_COMPRESSION:
            compressed_results = compress_results(cleaned_results, query_text)
            return compressed_results
        
        return cleaned_results
        
    except Exception as e:
        logger.error(f"Error in perform_hybrid_search: {str(e)}", exc_info=True)
        # Fall back to vector search if hybrid search fails
        logger.info("Falling back to vector search")
        return perform_vector_search(query_text, k_results, metadata_filters)

# --- Future Tools ---

# @tool
# def perform_postgres_search(
#     keywords: List[str], 
#     category_filter: Optional[str] = None,
#     limit: int = 10
# ) -> List[Dict]:
#     """
#     Performs a keyword search against the PostgreSQL database.
#     
#     Args:
#         keywords: List of keywords to search for.
#         category_filter: Optional category to filter results.
#         limit: Maximum number of results to return.
#     
#     Returns:
#         A list of matching documents with metadata.
#     """
#     # Implementation would use src.db_storage.PostgresStorage
#     # This is a placeholder for future functionality
#     pass

# @tool
# def get_document_by_exhibit_number(exhibit_number: int) -> Optional[Dict]:
#     """
#     Retrieves a specific document by its exhibit number.
#     
#     Args:
#         exhibit_number: The exhibit number to look up.
#     
#     Returns:
#         Document metadata and content if found, None otherwise.
#     """
#     # Implementation would query PostgreSQL directly
#     # This is a placeholder for future functionality
#     pass

# --- Tool Registry ---
# This can be used to dynamically select tools based on configuration

AVAILABLE_TOOLS = {
    "vector_search": perform_vector_search,
    "hybrid_search": perform_hybrid_search,
    # "postgres_search": perform_postgres_search,  # Uncomment when implemented
    # "exhibit_lookup": get_document_by_exhibit_number,  # Uncomment when implemented
}

def get_agent_tools(enable_sql: bool = False, enable_hybrid: bool = True) -> List:
    """
    Get the list of tools available to the agent based on configuration.
    
    Args:
        enable_sql: Whether to include SQL-based search tools.
        enable_hybrid: Whether to include hybrid search tool.
    
    Returns:
        List of tool functions.
    """
    tools = [perform_vector_search]
    
    # Add hybrid search if enabled
    if enable_hybrid:
        tools.append(perform_hybrid_search)
    
    # Add SQL tools if enabled
    # if enable_sql and "postgres_search" in AVAILABLE_TOOLS:
    #     tools.append(AVAILABLE_TOOLS["postgres_search"])
    
    return tools