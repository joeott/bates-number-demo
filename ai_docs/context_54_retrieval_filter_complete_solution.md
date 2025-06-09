# Context 54: Complete Solution for Retrieval Filter Issue

## Problem Analysis

The retrieval agent's category filtering happens in this flow:
1. **Query Understanding** (main_retriever.py:240) → LLM generates `potential_filters` including category
2. **Filter Assignment** (main_retriever.py:260) → `current_filters = query_analysis.potential_filters`
3. **Search Execution** (main_retriever.py:284) → Filters passed to `perform_vector_search`
4. **Hard Filtering** (agent_tools.py:90-91) → Category filter applied directly to ChromaDB

This prevents the agent from finding relevant documents that don't match the exact category.

## Complete Solution

### 1. **Modify output_parsers.py**

Add new fields to support category hints instead of filters:

```python
# In output_parsers.py, modify the QueryAnalysis class:

class QueryAnalysis(BaseModel):
    """Represents the agent's understanding of the user's query."""
    main_intent: str = Field(description="The primary goal of the query")
    sub_queries: List[str] = Field(description="Decomposed sub-queries to explore")
    search_keywords: List[str] = Field(description="Key terms to search for")
    
    # Replace potential_filters with:
    category_hints: Optional[List[str]] = Field(
        default=None,
        description="Categories that might contain relevant information"
    )
    category_weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Relative importance of each category (1.0 = normal, 2.0 = double weight)"
    )
    
    # Keep for backward compatibility but deprecated
    potential_filters: Optional[Dict] = Field(
        default=None,
        description="[DEPRECATED] Use category_hints instead"
    )
```

### 2. **Update agent_prompts.py**

Change the query understanding prompt:

```python
QUERY_UNDERSTANDING_HUMAN_MESSAGE = """Analyze this legal query and decompose it for iterative search.

User Query: {original_query}

Instructions:
1. Identify the main intent
2. Break down into specific sub-queries
3. Extract key search terms
4. Suggest relevant document categories (these are hints, not filters!)

Return JSON:
{{
    "main_intent": "What the user wants to find",
    "sub_queries": ["First aspect to search", "Second aspect to search"],
    "search_keywords": ["key", "terms", "to", "search"],
    "category_hints": ["Medical Record", "Pleading", "Correspondence"],
    "category_weights": {{"Medical Record": 1.5, "Pleading": 1.2}}
}}

IMPORTANT: 
- category_hints are suggestions to help find relevant documents
- The search will NOT exclude documents from other categories
- Use exact category names: Pleading, Medical Record, Bill, Correspondence, Photo, Video, Documentary Evidence, Uncategorized
"""
```

### 3. **Modify main_retriever.py**

Update the retrieval logic to use hints instead of filters:

```python
# Around line 260, replace filter assignment:
# OLD: current_filters = query_analysis.potential_filters
# NEW:
current_category_hints = query_analysis.category_hints or []
current_category_weights = query_analysis.category_weights or {}

# Remove the current_filters variable entirely

# Around line 279-289, update the search call:
logger.info(f"Executing vector search with category hints: {current_category_hints}")
try:
    # First, do a broad search without filters
    broad_results = agent_tools.perform_vector_search.invoke({
        "query_text": sub_query,
        "k_results": agent_config.NUM_RESULTS_PER_SUB_QUERY * 3,  # Get extra for re-ranking
        "metadata_filters": None  # NO FILTERS!
    }, config=run_config)
    
    # Re-rank results based on category hints
    search_results = self._rerank_by_categories(
        broad_results, 
        current_category_hints, 
        current_category_weights
    )[:agent_config.NUM_RESULTS_PER_SUB_QUERY]
    
    logger.info(f"Retrieved {len(search_results)} chunks after re-ranking.")
```

Add the re-ranking method:

```python
def _rerank_by_categories(
    self, 
    results: List[Dict], 
    category_hints: List[str], 
    category_weights: Dict[str, float]
) -> List[Dict]:
    """Re-rank search results based on category preferences."""
    if not category_hints and not category_weights:
        return results
    
    for result in results:
        original_score = result.get("relevance", 0.0)
        doc_category = result.get("metadata", {}).get("category", "")
        
        # Calculate boost factor
        boost = 1.0
        
        # Boost if in hint list
        if category_hints and doc_category in category_hints:
            boost *= 1.2
        
        # Apply specific weight if provided
        if category_weights and doc_category in category_weights:
            boost *= category_weights[doc_category]
        
        # Store both original and adjusted scores
        result["original_relevance"] = original_score
        result["adjusted_relevance"] = original_score * boost
        result["category_boost"] = boost
    
    # Sort by adjusted relevance
    return sorted(results, key=lambda x: x.get("adjusted_relevance", 0), reverse=True)
```

### 4. **Update Iteration Decision Logic**

Modify the iteration decision to evolve search strategy:

```python
# In the iteration decision section (around line 340), add:
# Track which categories we've found useful
categories_explored = {}
for fact in accumulated_facts:
    cat = fact.get("source_metadata", {}).get("category", "Unknown")
    categories_explored[cat] = categories_explored.get(cat, 0) + 1

# Pass this to the iteration decision
decision_input = {
    "original_query": original_query,
    "retrieved_facts_summary": facts_summary,
    "executed_queries_list": "\n".join(f"- {q}" for q in executed_queries),
    "current_iteration": current_iteration,
    "categories_explored": str(categories_explored)  # Add this
}
```

### 5. **Add Query Evolution Strategy**

Update the iteration decision prompt in agent_prompts.py:

```python
ITERATION_DECISION_HUMAN_MESSAGE = """Based on the search progress, decide next steps.

Original Query: {original_query}

Executed Queries:
{executed_queries_list}

Categories Explored: {categories_explored}

Retrieved Facts:
{retrieved_facts_summary}

Current Iteration: {current_iteration}/{max_iterations}

Instructions:
1. If we found relevant facts in certain categories, explore those more deeply
2. If we found nothing, broaden the search (remove category hints)
3. Try alternative phrasings if initial queries failed

Return JSON:
{{
    "continue_iteration": true/false,
    "next_sub_queries": ["refined query 1", "broader query 2"],
    "category_focus": ["categories to prioritize"],
    "reasoning": "why this approach"
}}
"""
```

### 6. **Emergency Fallback**

Add a fallback mechanism when no results are found:

```python
# In main_retriever.py, after search_results = []
if not search_results and current_iteration < self.max_iterations:
    logger.warning("No results found. Attempting broader search without hints.")
    
    # Try again with no category hints
    fallback_results = agent_tools.perform_vector_search.invoke({
        "query_text": sub_query,
        "k_results": agent_config.NUM_RESULTS_PER_SUB_QUERY * 2,
        "metadata_filters": None
    }, config=run_config)
    
    if fallback_results:
        logger.info(f"Fallback search found {len(fallback_results)} results.")
        search_results = fallback_results
```

## Testing Strategy

### Test 1: Verify No Hard Filtering
```python
# Add logging to agent_tools.py perform_vector_search:
if metadata_filters:
    logger.warning("DEPRECATED: Hard filters passed to vector search. Ignoring.")
    # Don't use them!
```

### Test 2: Complex Query
```bash
python -m src.retrieval_agent.cli \
  "Find evidence of YMCA negligence causing Nikita's injury" \
  --debug 2>&1 | grep -E "(category|filter|hint)"
```

Expected output:
- "Executing vector search with category hints: ['Pleading', 'Medical Record']"
- "Retrieved X chunks after re-ranking"
- NO "Applying filters:" messages

### Test 3: Verify Broad Search
Monitor that searches return results from multiple categories, not just the hinted ones.

## Configuration

Add to agent_config.py:
```python
# Search Strategy
ENABLE_CATEGORY_HINTS = True
CATEGORY_HINT_BOOST = 1.2
CATEGORY_WEIGHT_MULTIPLIER = 1.0
FALLBACK_ON_EMPTY_RESULTS = True
BROADEN_SEARCH_MULTIPLIER = 3  # Get 3x results for re-ranking
```

## Verification Checklist

- [ ] No hard category filters in perform_vector_search
- [ ] Query understanding generates hints, not filters
- [ ] Re-ranking boosts relevant categories without excluding others
- [ ] Fallback search when no results found
- [ ] Iteration decisions adapt based on found categories
- [ ] Complex queries find documents across multiple categories

## Expected Behavior After Fix

For the query "Find evidence of YMCA negligence causing Nikita's injury":

1. **Iteration 1**: Broad search finds documents mentioning YMCA, injury, Nikita
2. **Categories Found**: Pleading, Medical Record, Correspondence
3. **Iteration 2**: Focus on promising categories but still search broadly
4. **Iteration 3**: Synthesize findings from multiple document types
5. **Result**: Comprehensive answer drawing from all relevant documents

The key change is that the system will find relevant information regardless of category, using categories only to prioritize and rank results.