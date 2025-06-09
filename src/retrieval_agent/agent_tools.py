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

# Import your existing VectorSearcher and its config
# Assuming vector_search.py has been refactored to use Langchain components
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))  # Add project root to path

from src.vector_search import VectorSearcher
from src.config import VECTOR_STORE_PATH  # For default path

# Import agent-specific config
from .agent_config import NUM_RESULTS_PER_SUB_QUERY

logger = logging.getLogger(__name__)

# Global instance or factory function for VectorSearcher to manage resources
# For simplicity in a CLI context, creating it per call is okay,
# but for an API, you'd manage its lifecycle.
_vector_searcher_instance = None

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
        
        return cleaned_results
        
    except Exception as e:
        logger.error(f"Error in perform_vector_search: {str(e)}", exc_info=True)
        # Return empty results rather than failing completely
        return []

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
    # "postgres_search": perform_postgres_search,  # Uncomment when implemented
    # "exhibit_lookup": get_document_by_exhibit_number,  # Uncomment when implemented
}

def get_agent_tools(enable_sql: bool = False) -> List:
    """
    Get the list of tools available to the agent based on configuration.
    
    Args:
        enable_sql: Whether to include SQL-based search tools.
    
    Returns:
        List of tool functions.
    """
    tools = [perform_vector_search]
    
    # Add SQL tools if enabled
    # if enable_sql and "postgres_search" in AVAILABLE_TOOLS:
    #     tools.append(AVAILABLE_TOOLS["postgres_search"])
    
    return tools