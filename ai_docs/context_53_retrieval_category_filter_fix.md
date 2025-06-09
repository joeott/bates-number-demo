# Context 53: Fixing Retrieval Agent Category Filtering

## Problem Statement

The retrieval agent is currently using category filters as hard constraints, which prevents it from finding relevant documents. The agent should instead use categories to inform its search strategy while still performing broad vector searches that can find relevant content regardless of category.

## Current Category Filtering Locations

### 1. **agent_prompts.py - Query Understanding Prompt**
**Location**: Lines in QUERY_UNDERSTANDING_HUMAN_MESSAGE
```python
"filters": {{
    "category": "Medical Records",  # Hard filter applied
    "exhibit_number": "5"
}}
```

**Issue**: The prompt encourages the LLM to generate category filters that are applied as hard constraints.

### 2. **agent_tools.py - perform_vector_search Function**
**Location**: Lines 80-95
```python
def perform_vector_search(query: str, num_results: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
    # ...
    if filters:
        # Convert filters to metadata filter format
        metadata_filters = {}
        if "category" in filters:
            metadata_filters["category"] = filters["category"]  # Hard filter!
```

**Issue**: Category filters are directly applied to ChromaDB queries, excluding all documents not matching the exact category.

### 3. **main_retriever.py - Query Execution**
**Location**: Lines around 296-310
```python
# Execute vector search with filters
search_results = agent_tools.perform_vector_search(
    query=keyword_query,
    num_results=agent_config.NUM_RESULTS_PER_SUB_QUERY,
    filters=sub_query_obj.filters  # Filters passed directly!
)
```

**Issue**: Filters from query analysis are passed directly to search without modification.

## Recommended Changes

### 1. **Modify agent_prompts.py**

Change the query understanding prompt to treat categories as hints, not filters:

```python
QUERY_UNDERSTANDING_HUMAN_MESSAGE = """Given the user's legal query, analyze and decompose it into sub-queries.

User Query: {query}

Return a JSON with your analysis:
{{
    "main_intent": "Brief description of what the user wants",
    "sub_queries": [
        {{
            "query_text": "Specific aspect to search for",
            "keywords": ["key", "terms"],
            "category_hints": ["Medical Record", "Pleading"],  # Changed from filters
            "relevance_boost": {{"category": "Medical Record", "weight": 1.5}}  # New field
        }}
    ],
    "analysis_approach": "How you plan to find relevant information"
}}

Important: category_hints are suggestions to help rank results, NOT filters to exclude documents.
"""
```

### 2. **Update output_parsers.py**

Modify the SubQuery model to support hints instead of filters:

```python
class SubQuery(BaseModel):
    """Represents a decomposed sub-query with search hints."""
    query_text: str = Field(description="The sub-query text")
    keywords: List[str] = Field(description="Key terms to search for")
    category_hints: Optional[List[str]] = Field(
        default=None, 
        description="Categories that might contain relevant information"
    )
    relevance_boost: Optional[Dict[str, float]] = Field(
        default=None,
        description="Categories to boost in ranking"
    )
    # Remove or deprecate 'filters' field
```

### 3. **Modify agent_tools.py**

Update the vector search function to use categories for ranking, not filtering:

```python
def perform_vector_search(
    query: str, 
    num_results: int = 5, 
    category_hints: Optional[List[str]] = None,
    relevance_boost: Optional[Dict[str, float]] = None
) -> List[Dict]:
    """
    Perform vector search with optional category-based relevance boosting.
    
    Args:
        query: Search query text
        num_results: Number of results to return
        category_hints: Categories to prioritize (not filter)
        relevance_boost: Category weights for re-ranking
    """
    try:
        searcher = VectorSearcher()
        
        # First, do a broad search without filters
        initial_results = searcher.search(
            query=query,
            n=num_results * 3,  # Get more results for re-ranking
            filter_metadata=None  # No hard filters!
        )
        
        # Re-rank results based on category hints
        if category_hints or relevance_boost:
            ranked_results = _rerank_by_category(
                initial_results, 
                category_hints, 
                relevance_boost
            )
            return ranked_results[:num_results]
        
        return initial_results[:num_results]
        
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return []

def _rerank_by_category(
    results: List[Dict], 
    category_hints: Optional[List[str]], 
    relevance_boost: Optional[Dict[str, float]]
) -> List[Dict]:
    """Re-rank results based on category preferences."""
    for result in results:
        original_score = result.get("relevance", 0.0)
        category = result.get("metadata", {}).get("category", "")
        
        # Apply boost if category matches hints
        boost_factor = 1.0
        if category_hints and category in category_hints:
            boost_factor *= 1.2  # 20% boost for hint match
        
        if relevance_boost and category in relevance_boost:
            boost_factor *= relevance_boost[category]
        
        result["adjusted_relevance"] = original_score * boost_factor
    
    # Sort by adjusted relevance
    return sorted(results, key=lambda x: x.get("adjusted_relevance", 0), reverse=True)
```

### 4. **Update main_retriever.py**

Modify the search execution to use the new approach:

```python
# Around line 296, replace the search call
search_results = agent_tools.perform_vector_search(
    query=keyword_query,
    num_results=agent_config.NUM_RESULTS_PER_SUB_QUERY,
    category_hints=sub_query_obj.category_hints,  # Use hints
    relevance_boost=sub_query_obj.relevance_boost  # Use boost weights
)
```

### 5. **Add Iterative Query Evolution**

Add logic to evolve queries based on results:

```python
def _evolve_query_based_on_results(
    self, 
    original_query: str, 
    current_results: List[Dict],
    iteration: int
) -> str:
    """Evolve the query based on what we've found so far."""
    
    if not current_results:
        # No results - broaden the query
        evolution_prompt = f"""
        The query '{original_query}' returned no results.
        Generate a broader, more general version of this query.
        """
    else:
        # Have results - refine based on patterns
        categories_found = [r.get("metadata", {}).get("category") for r in current_results]
        evolution_prompt = f"""
        Original query: '{original_query}'
        Found documents in categories: {set(categories_found)}
        
        Generate a refined query that explores related aspects not yet covered.
        """
    
    evolved_query = self.llm.invoke(evolution_prompt).content
    return evolved_query
```

## Implementation Priority

### Phase 1: Remove Hard Filters (Critical)
1. Modify `perform_vector_search` to never use category as a hard filter
2. Update prompts to use "category_hints" instead of "filters"
3. Test with the YMCA negligence query

### Phase 2: Add Smart Re-ranking
1. Implement `_rerank_by_category` function
2. Add relevance boosting based on category hints
3. Allow dynamic weight adjustment

### Phase 3: Query Evolution
1. Track which categories have been explored
2. Evolve queries to explore new aspects
3. Learn from successful/unsuccessful iterations

## Verification Test

After implementation, test with:
```bash
python -m src.retrieval_agent.cli \
  "Find all evidence of YMCA negligence causing injury to Nikita" \
  --debug
```

Expected behavior:
1. Initial broad search returns documents from multiple categories
2. System identifies Medical Record and Pleading categories as relevant
3. Subsequent iterations explore these categories more deeply
4. No documents are excluded due to category mismatch
5. Final synthesis includes facts from all relevant documents

## Configuration Updates

Add to `agent_config.py`:
```python
# Category Handling
USE_CATEGORY_FILTERING = False  # Disable hard filtering
CATEGORY_BOOST_FACTOR = 1.2    # Boost for category hint matches
MAX_INITIAL_RESULTS = 15       # Get more results for re-ranking
ENABLE_QUERY_EVOLUTION = True  # Allow queries to evolve
```

## Error Prevention

Common mistakes to avoid:
1. Don't use `filter_metadata` parameter in ChromaDB calls
2. Don't require exact category matches
3. Don't trust LLM-generated category names (they may not match exactly)
4. Always fallback to broad search if filtered search returns nothing

## Conclusion

By removing hard category filters and implementing intelligent re-ranking, the retrieval agent will be more robust and capable of finding relevant information regardless of how documents are categorized. The system will use categories as hints to improve relevance ranking while maintaining the ability to discover unexpected connections across document types.