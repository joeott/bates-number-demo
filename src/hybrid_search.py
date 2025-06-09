"""
Hybrid search module combining vector and PostgreSQL search capabilities.
Implements result fusion and ranking algorithms for optimal search results.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from src.vector_search import VectorSearcher
from src.db_storage import PostgresStorage
from src.config import VECTOR_STORE_PATH, POSTGRES_CONNECTION

logger = logging.getLogger(__name__)


class SearchMethod(Enum):
    """Available search methods for hybrid search."""
    VECTOR = "vector"
    POSTGRES = "postgres"
    HYBRID = "hybrid"


@dataclass
class SearchResult:
    """Unified search result structure."""
    document_id: str
    filename: str
    category: str
    exhibit_number: int
    bates_range: str
    text: str
    score: float
    page: Optional[int] = None
    source: str = "hybrid"
    vector_score: Optional[float] = None
    postgres_score: Optional[float] = None


class HybridSearcher:
    """
    Hybrid search implementation combining vector and PostgreSQL search.
    Provides unified interface with result fusion and ranking.
    """
    
    def __init__(
        self, 
        vector_store_path: str = VECTOR_STORE_PATH,
        postgres_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize hybrid searcher with vector and PostgreSQL components."""
        self.vector_searcher = None
        self.postgres_searcher = None
        
        # Initialize vector searcher if available
        try:
            if Path(vector_store_path).exists():
                self.vector_searcher = VectorSearcher(vector_store_path)
                logger.info("Vector search initialized")
            else:
                logger.warning(f"Vector store not found at {vector_store_path}")
        except Exception as e:
            logger.warning(f"Failed to initialize vector search: {e}")
        
        # Initialize PostgreSQL searcher if available
        if postgres_config:
            try:
                self.postgres_searcher = PostgresStorage(
                    connection_string=postgres_config['connection_string'],
                    pool_size=postgres_config.get('pool_size', 5)
                )
                logger.info("PostgreSQL search initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize PostgreSQL search: {e}")
        else:
            logger.info("PostgreSQL config not provided, running vector-only")
    
    def search(
        self,
        query: str,
        method: SearchMethod = SearchMethod.HYBRID,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining vector and PostgreSQL results.
        
        Args:
            query: Search query string
            method: Search method to use (vector, postgres, or hybrid)
            limit: Maximum number of results to return
            filters: Dictionary of filters (category, exhibit_number, etc.)
            
        Returns:
            List of unified search results
        """
        filters = filters or {}
        category = filters.get('category')
        exhibit_number = filters.get('exhibit_number')
        bates_start = filters.get('bates_start')
        bates_end = filters.get('bates_end')
        
        if method == SearchMethod.VECTOR:
            return self._vector_search(query, limit, category, exhibit_number, bates_start, bates_end)
        elif method == SearchMethod.POSTGRES:
            return self._postgres_search(query, limit, category, exhibit_number, bates_start, bates_end)
        else:  # HYBRID
            return self._hybrid_search(query, limit, category, exhibit_number, bates_start, bates_end)
    
    def _vector_search(
        self,
        query: str,
        limit: int,
        category: Optional[str] = None,
        exhibit_number: Optional[int] = None,
        bates_start: Optional[str] = None,
        bates_end: Optional[str] = None
    ) -> List[SearchResult]:
        """Perform vector-only search."""
        if not self.vector_searcher:
            logger.warning("Vector search not available")
            return []
        
        try:
            # Build metadata filter
            where_clause = self._build_vector_filter(category, exhibit_number, bates_start, bates_end)
            
            # Perform vector search
            vector_results = self.vector_searcher.search(
                query=query,
                n_results=limit,
                category=category,
                exhibit_number=exhibit_number
            )
            
            # Convert to unified format
            unified_results = []
            for result in vector_results:
                unified_result = SearchResult(
                    document_id=f"{result.get('filename', 'Unknown')}_{result.get('page', 0)}",
                    filename=result.get("filename", "Unknown"),
                    category=result.get("category", "Unknown"),
                    exhibit_number=result.get("exhibit_number", 0),
                    bates_range=f"{result.get('bates_start', '')}-{result.get('bates_end', '')}",
                    text=result.get("text", ""),
                    score=result.get("relevance", 0.0),
                    page=result.get("page"),
                    source="vector",
                    vector_score=result.get("relevance", 0.0)
                )
                unified_results.append(unified_result)
                
            return unified_results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def _postgres_search(
        self,
        query: str,
        limit: int,
        category: Optional[str] = None,
        exhibit_number: Optional[int] = None,
        bates_start: Optional[str] = None,
        bates_end: Optional[str] = None
    ) -> List[SearchResult]:
        """Perform PostgreSQL-only search."""
        if not self.postgres_searcher:
            logger.warning("PostgreSQL search not available")
            return []
        
        try:
            # Perform PostgreSQL search
            postgres_results = self.postgres_searcher.search_text(
                query=query,
                limit=limit
            )
            
            # Filter results if needed
            if category:
                postgres_results = [r for r in postgres_results if r.get('category') == category]
            if exhibit_number is not None:
                postgres_results = [r for r in postgres_results if r.get('exhibit_id') == exhibit_number]
            
            # Convert to unified format
            unified_results = []
            for result in postgres_results:
                unified_result = SearchResult(
                    document_id=f"{result.get('exhibit_filename', 'Unknown')}_{result.get('exhibit_id', 0)}",
                    filename=result.get("exhibit_filename", "Unknown"),
                    category=result.get("category", "Unknown"),
                    exhibit_number=result.get("exhibit_id", 0),
                    bates_range=f"{result.get('bates_start', '')}-{result.get('bates_end', '')}",
                    text=result.get("excerpt", "").replace('<b>', '').replace('</b>', ''),
                    score=result.get("rank", 0.0),
                    source="postgres",
                    postgres_score=result.get("rank", 0.0)
                )
                unified_results.append(unified_result)
                
            return unified_results
            
        except Exception as e:
            logger.error(f"PostgreSQL search failed: {e}")
            return []
    
    def _hybrid_search(
        self,
        query: str,
        limit: int,
        category: Optional[str] = None,
        exhibit_number: Optional[int] = None,
        bates_start: Optional[str] = None,
        bates_end: Optional[str] = None
    ) -> List[SearchResult]:
        """Perform hybrid search combining vector and PostgreSQL results."""
        # Get results from both sources
        vector_results = self._vector_search(query, limit * 2, category, exhibit_number, bates_start, bates_end)
        postgres_results = self._postgres_search(query, limit * 2, category, exhibit_number, bates_start, bates_end)
        
        # If only one source is available, return those results
        if not vector_results and postgres_results:
            return postgres_results[:limit]
        elif vector_results and not postgres_results:
            return vector_results[:limit]
        elif not vector_results and not postgres_results:
            return []
        
        # Fuse results using reciprocal rank fusion
        fused_results = self._reciprocal_rank_fusion(vector_results, postgres_results)
        
        # Sort by combined score and return top results
        fused_results.sort(key=lambda x: x.score, reverse=True)
        return fused_results[:limit]
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: List[SearchResult],
        postgres_results: List[SearchResult],
        k: int = 60
    ) -> List[SearchResult]:
        """
        Implement Reciprocal Rank Fusion (RRF) algorithm.
        
        Args:
            vector_results: Results from vector search
            postgres_results: Results from PostgreSQL search
            k: RRF parameter (typically 60)
            
        Returns:
            Fused and ranked results
        """
        # Create document identifier for deduplication
        def doc_id(result: SearchResult) -> str:
            return f"{result.exhibit_number}_{result.filename}"
        
        # Build rank maps
        vector_ranks = {doc_id(result): i + 1 for i, result in enumerate(vector_results)}
        postgres_ranks = {doc_id(result): i + 1 for i, result in enumerate(postgres_results)}
        
        # Combine all unique documents
        all_docs = {}
        
        # Add vector results
        for result in vector_results:
            doc_key = doc_id(result)
            all_docs[doc_key] = result
        
        # Add PostgreSQL results (merge with existing if duplicate)
        for result in postgres_results:
            doc_key = doc_id(result)
            if doc_key in all_docs:
                # Merge scores for existing document
                existing = all_docs[doc_key]
                existing.postgres_score = result.postgres_score
                if not existing.vector_score:
                    existing.vector_score = 0.0
            else:
                # Add new document
                result.vector_score = 0.0  # No vector score for this document
                all_docs[doc_key] = result
        
        # Calculate RRF scores
        fused_results = []
        for doc_key, result in all_docs.items():
            rrf_score = 0.0
            
            # Add vector contribution
            if doc_key in vector_ranks:
                rrf_score += 1.0 / (k + vector_ranks[doc_key])
            
            # Add PostgreSQL contribution
            if doc_key in postgres_ranks:
                rrf_score += 1.0 / (k + postgres_ranks[doc_key])
            
            # Update result with fused score
            result.score = rrf_score
            result.source = "hybrid"
            fused_results.append(result)
        
        return fused_results
    
    def _build_vector_filter(
        self,
        category: Optional[str] = None,
        exhibit_number: Optional[int] = None,
        bates_start: Optional[str] = None,
        bates_end: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Build metadata filter for vector search."""
        where_clause = {}
        
        if category:
            where_clause["category"] = category
        
        if exhibit_number:
            where_clause["exhibit_number"] = exhibit_number
        
        # Note: Bates range filtering is more complex and might need special handling
        if bates_start or bates_end:
            # For now, we'll skip Bates range filtering in vector search
            # This could be enhanced to support range queries
            pass
        
        return where_clause if where_clause else None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get search statistics from both sources."""
        stats = {}
        
        # Get vector statistics
        if self.vector_searcher:
            try:
                vector_stats = self.vector_searcher.get_stats()
                stats["vector"] = vector_stats
            except Exception as e:
                logger.warning(f"Failed to get vector statistics: {e}")
        
        # Get PostgreSQL statistics
        if self.postgres_searcher:
            try:
                postgres_stats = self.postgres_searcher.get_statistics()
                stats["postgres"] = postgres_stats
            except Exception as e:
                logger.warning(f"Failed to get PostgreSQL statistics: {e}")
        
        return stats
    
    def is_available(self) -> Tuple[bool, bool, bool]:
        """
        Check availability of search methods.
        
        Returns:
            Tuple of (vector_available, postgres_available, hybrid_available)
        """
        vector_available = self.vector_searcher is not None
        postgres_available = self.postgres_searcher is not None
        hybrid_available = vector_available and postgres_available
        
        return vector_available, postgres_available, hybrid_available
    
    def close(self):
        """Close connections to search backends."""
        if self.postgres_searcher:
            try:
                self.postgres_searcher.close()
            except Exception as e:
                logger.warning(f"Error closing PostgreSQL connection: {e}")
        # Vector searcher doesn't need explicit closing