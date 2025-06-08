"""
Vector search module for querying legal documents.
Provides semantic search capabilities across the document corpus.
"""

import logging
from typing import List, Dict, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings

from src.config import VECTOR_STORE_PATH, EMBEDDING_MODEL
from src.vector_processor import QwenEmbedder

logger = logging.getLogger(__name__)


class VectorSearcher:
    """
    Semantic search interface for legal documents.
    """
    
    def __init__(self, vector_store_path: str = VECTOR_STORE_PATH):
        self.vector_store_path = Path(vector_store_path)
        
        if not self.vector_store_path.exists():
            raise ValueError(f"Vector store not found at {self.vector_store_path}")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.vector_store_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get collection
        try:
            self.collection = self.client.get_collection("legal_documents")
            logger.info(f"Connected to vector store with {self.collection.count()} chunks")
        except Exception as e:
            raise ValueError(f"Failed to access vector collection: {e}")
        
        # Initialize embedder
        self.embedder = QwenEmbedder(model=EMBEDDING_MODEL)
    
    def search(self, 
               query: str, 
               n_results: int = 10,
               category: Optional[str] = None,
               exhibit_number: Optional[int] = None) -> List[Dict]:
        """
        Perform semantic search across documents.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            category: Filter by document category
            exhibit_number: Filter by specific exhibit number
            
        Returns:
            List of search results with metadata
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        # Generate query embedding
        try:
            query_embedding = self.embedder.embed_text(query)
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise
        
        # Build where clause for filtering
        where_clause = {}
        if category:
            where_clause["category"] = category
        if exhibit_number is not None:
            where_clause["exhibit_number"] = exhibit_number
        
        # Perform search
        try:
            if where_clause:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=where_clause,
                    include=["documents", "metadatas", "distances"]
                )
            else:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    include=["documents", "metadatas", "distances"]
                )
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
        
        return self._format_results(results)
    
    def search_by_bates_range(self, bates_start: int, bates_end: int) -> List[Dict]:
        """
        Search for documents within a Bates number range.
        """
        where_clause = {
            "$and": [
                {"bates_start": {"$lte": bates_end}},
                {"bates_end": {"$gte": bates_start}}
            ]
        }
        
        try:
            # Use get instead of query for metadata-only search
            results = self.collection.get(
                where=where_clause,
                include=["documents", "metadatas"],
                limit=100  # Get all matching documents
            )
            
            # Format results to match query output structure
            formatted_results = {
                'ids': [results['ids']],
                'documents': [results['documents']],
                'metadatas': [results['metadatas']]
            }
        except Exception as e:
            logger.error(f"Bates range search failed: {e}")
            raise
        
        return self._format_results(formatted_results, include_score=False)
    
    def get_categories(self) -> List[str]:
        """Get list of unique categories in the collection."""
        # ChromaDB doesn't have a direct way to get unique values
        # So we'll get a sample and extract unique categories
        try:
            sample = self.collection.get(limit=1000, include=["metadatas"])
            categories = set()
            for metadata in sample['metadatas']:
                if metadata and 'category' in metadata:
                    categories.add(metadata['category'])
            return sorted(list(categories))
        except Exception as e:
            logger.error(f"Failed to get categories: {e}")
            return []
    
    def _format_results(self, results: Dict, include_score: bool = True) -> List[Dict]:
        """Format ChromaDB results into a consistent structure."""
        formatted_results = []
        
        if not results or not results.get('ids'):
            return formatted_results
        
        # ChromaDB returns results in nested lists
        ids = results['ids'][0] if results['ids'] else []
        documents = results['documents'][0] if results['documents'] else []
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        distances = results.get('distances', [[]])[0] if include_score else []
        
        for i in range(len(ids)):
            result = {
                'id': ids[i],
                'text': documents[i] if i < len(documents) else '',
                'metadata': metadatas[i] if i < len(metadatas) else {}
            }
            
            # Add relevance score if available (convert distance to similarity)
            if include_score and i < len(distances):
                # Cosine distance to similarity score
                result['relevance'] = 1.0 - distances[i]
            
            # Extract key metadata fields
            metadata = result['metadata']
            result['filename'] = metadata.get('filename', 'Unknown')
            result['category'] = metadata.get('category', 'Uncategorized')
            result['exhibit_number'] = metadata.get('exhibit_number', 0)
            result['bates_start'] = metadata.get('bates_start', 0)
            result['bates_end'] = metadata.get('bates_end', 0)
            result['page'] = metadata.get('page', 0)
            result['summary'] = metadata.get('summary', '')
            
            formatted_results.append(result)
        
        return formatted_results
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        try:
            count = self.collection.count()
            categories = self.get_categories()
            
            return {
                'total_chunks': count,
                'categories': categories,
                'num_categories': len(categories),
                'collection_name': self.collection.name
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {'error': str(e)}