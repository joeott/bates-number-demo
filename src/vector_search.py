"""
Vector search module for querying legal documents using LangChain.
Provides semantic search capabilities across the document corpus.
"""

import logging
from typing import List, Dict, Optional
from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from src.config import (
    VECTOR_STORE_PATH, EMBEDDING_MODEL, OLLAMA_HOST,
    LLM_PROVIDER, LMSTUDIO_HOST, LMSTUDIO_EMBEDDING_MODEL
)
from src.lmstudio_embeddings import LMStudioEmbeddings

logger = logging.getLogger(__name__)


class VectorSearcher:
    """
    Semantic search interface for legal documents using LangChain.
    """
    
    def __init__(self, vector_store_path: str = VECTOR_STORE_PATH):
        self.vector_store_path = Path(vector_store_path)
        
        if not self.vector_store_path.exists():
            raise ValueError(f"Vector store not found at {self.vector_store_path}")
        
        # Initialize LangChain components with provider-specific embeddings
        if LLM_PROVIDER == "lmstudio":
            self.embeddings = LMStudioEmbeddings(
                base_url=LMSTUDIO_HOST,
                model=LMSTUDIO_EMBEDDING_MODEL
            )
        else:
            # Default to Ollama embeddings
            self.embeddings = OllamaEmbeddings(
                model=EMBEDDING_MODEL,
                base_url=OLLAMA_HOST
            )
        
        self.vector_store = Chroma(
            collection_name="legal_documents",
            embedding_function=self.embeddings,
            persist_directory=str(self.vector_store_path)
        )
        
        logger.info(f"VectorSearcher initialized with LangChain components")
        logger.info(f"Vector store: {self.vector_store_path}")
    
    def search(
        self,
        query: str,
        n_results: int = 10,
        category: Optional[str] = None,
        exhibit_number: Optional[int] = None
    ) -> List[Dict]:
        """
        Search documents using semantic similarity.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            category: Filter by document category
            exhibit_number: Filter by exhibit number
            
        Returns:
            List of search results with metadata
        """
        try:
            # Build filter criteria
            filter_dict = {}
            if category:
                filter_dict["category"] = category
            if exhibit_number is not None:
                filter_dict["exhibit_number"] = exhibit_number
            
            # Use LangChain's similarity search
            docs = self.vector_store.similarity_search_with_relevance_scores(
                query,
                k=n_results,
                filter=filter_dict if filter_dict else None
            )
            
            # Format results
            results = []
            for doc, score in docs:
                result = {
                    "text": doc.page_content,
                    "relevance": score,
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "category": doc.metadata.get("category", "Unknown"),
                    "exhibit_number": doc.metadata.get("exhibit_number", 0),
                    "page": doc.metadata.get("page", 0),
                    "bates_start": doc.metadata.get("bates_start", "N/A"),
                    "bates_end": doc.metadata.get("bates_end", "N/A"),
                    "summary": doc.metadata.get("summary", "")
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def search_by_bates_range(self, start: int, end: int) -> List[Dict]:
        """
        Search documents by Bates number range.
        """
        try:
            # Get all documents and filter by Bates range
            all_docs = self.vector_store.get()
            
            results = []
            seen_exhibits = set()
            
            for i, metadata in enumerate(all_docs.get("metadatas", [])):
                if not metadata:
                    continue
                    
                # Check if document falls within Bates range
                doc_start = int(metadata.get("bates_start", "0"))
                doc_end = int(metadata.get("bates_end", "0"))
                exhibit_num = metadata.get("exhibit_number")
                
                if (doc_start <= end and doc_end >= start and 
                    exhibit_num not in seen_exhibits):
                    seen_exhibits.add(exhibit_num)
                    
                    # Get the document content
                    doc_text = all_docs["documents"][i] if i < len(all_docs.get("documents", [])) else ""
                    
                    result = {
                        "text": doc_text,
                        "relevance": 1.0,  # Exact match
                        "filename": metadata.get("filename", "Unknown"),
                        "category": metadata.get("category", "Unknown"),
                        "exhibit_number": exhibit_num,
                        "page": metadata.get("page", 0),
                        "bates_start": doc_start,
                        "bates_end": doc_end,
                        "summary": metadata.get("summary", "")
                    }
                    results.append(result)
            
            return sorted(results, key=lambda x: x["bates_start"])
            
        except Exception as e:
            logger.error(f"Bates range search failed: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        try:
            collection = self.vector_store._collection
            count = collection.count()
            
            # Get unique categories and exhibits
            results = collection.get(include=["metadatas"])
            
            categories = set()
            exhibits = set()
            
            if results and results["metadatas"]:
                for metadata in results["metadatas"]:
                    if metadata.get("category"):
                        categories.add(metadata["category"])
                    if metadata.get("exhibit_number"):
                        exhibits.add(metadata["exhibit_number"])
            
            return {
                "total_chunks": count,
                "categories": sorted(list(categories)),
                "num_categories": len(categories),
                "num_exhibits": len(exhibits)
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                "total_chunks": 0,
                "categories": [],
                "num_categories": 0,
                "num_exhibits": 0
            }
    
    def get_categories(self) -> List[str]:
        """Get list of unique categories in the vector store."""
        stats = self.get_stats()
        return stats.get("categories", [])