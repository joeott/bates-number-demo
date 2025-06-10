# Context 56: LangChain Enhancement Implementation Plan

## Executive Summary

This document provides a phased implementation plan for enhancing the retrieval agent with advanced LangChain features, including hybrid search, re-ranking, contextual compression, and sophisticated query understanding. All changes will be made to existing scripts without creating new files unless explicitly authorized.

**Important Note**: This plan has been updated based on the LangChain reference repository to ensure correct import paths, class inheritance, and method signatures.

## Phase 1: Hybrid Search Integration (Priority: High)

### 1.1 Modify `src/retrieval_agent/agent_tools.py`

**Changes Required:**
1. Add imports for HybridSearcher (lines 1-30)
2. Create `get_hybrid_searcher()` function (lines 32-42)
3. Add `perform_hybrid_search` tool (lines 44-101)
4. Update tool registry if exists

**Specific Code Changes:**
```python
# Line 12: Add after existing imports
from src.hybrid_search import HybridSearcher, SearchMethod
from src.config import POSTGRES_CONNECTION, POSTGRES_POOL_SIZE, ENABLE_POSTGRES_STORAGE

# Line 32: Add after get_vector_searcher()
_hybrid_searcher_instance = None

def get_hybrid_searcher():
    global _hybrid_searcher_instance
    if _hybrid_searcher_instance is None:
        # Option 1: Use custom HybridSearcher
        _hybrid_searcher_instance = HybridSearcher(
            vector_store_path=str(VECTOR_STORE_PATH),
            postgres_config={
                'connection_string': POSTGRES_CONNECTION,
                'pool_size': POSTGRES_POOL_SIZE
            } if ENABLE_POSTGRES_STORAGE else None
        )
        
        # Option 2: Use LangChain's EnsembleRetriever
        # from langchain.retrievers import EnsembleRetriever
        # vector_retriever = get_vector_searcher().as_retriever()
        # keyword_retriever = get_postgres_searcher().as_retriever()
        # _hybrid_searcher_instance = EnsembleRetriever(
        #     retrievers=[vector_retriever, keyword_retriever],
        #     weights=agent_config.ENSEMBLE_WEIGHTS,
        #     id_key=agent_config.ENSEMBLE_ID_KEY
        # )
        
        logger.info("HybridSearcher initialized.")
    return _hybrid_searcher_instance

# Line 44: Add new tool function
from langchain_core.tools import tool
from langchain.retrievers import EnsembleRetriever  # For hybrid approach

@tool
def perform_hybrid_search(
    query_text: str,
    k_results: int = NUM_RESULTS_PER_SUB_QUERY,
    metadata_filters: Optional[Dict[str, Any]] = None,
    search_method: str = "hybrid"
) -> List[Dict]:
    # Implementation as shown in context_55
```

### 1.2 Modify `src/retrieval_agent/main_retriever.py`

**Changes Required:**
1. Update search tool selection logic (around line 284)
2. Add configuration for search method preference

**Specific Code Changes:**
```python
# Line 284: Replace existing search call
# Determine search method based on query or config
search_method = "hybrid"  # Can be made dynamic
if agent_config.ENABLE_HYBRID_SEARCH:
    tool_input = {
        "query_text": sub_query,
        "k_results": agent_config.NUM_RESULTS_PER_SUB_QUERY,
        "metadata_filters": current_filters,
        "search_method": search_method
    }
    search_results = agent_tools.perform_hybrid_search.invoke(
        tool_input,
        config=run_config
    )
else:
    # Fallback to existing vector search
    search_results = agent_tools.perform_vector_search.invoke(...)
```

### 1.3 Modify `src/retrieval_agent/agent_config.py`

**Changes Required:**
1. Add hybrid search configuration parameters

**Specific Code Changes:**
```python
# Line 55: Add after existing configs
# Hybrid Search Configuration
ENABLE_HYBRID_SEARCH = True
DEFAULT_SEARCH_METHOD = "hybrid"  # "vector", "postgres", or "hybrid"
HYBRID_WEIGHT_VECTOR = 0.7
HYBRID_WEIGHT_KEYWORD = 0.3
# For EnsembleRetriever
ENSEMBLE_WEIGHTS = [0.7, 0.3]  # Vector weight, keyword weight
ENSEMBLE_ID_KEY = "doc_id"  # For document deduplication
```

## Phase 2: Advanced Re-ranking (Priority: High)

### 2.1 Modify `src/retrieval_agent/agent_tools.py`

**Changes Required:**
1. Add cross-encoder imports and initialization
2. Integrate re-ranking into search functions

**Specific Code Changes:**
```python
# Line 15: Add imports (corrected based on LangChain reference)
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
# Note: CrossEncoderReranker may need custom implementation inheriting BaseDocumentCompressor

# Line 35: Add initialization
_reranker_model = None

def get_reranker():
    global _reranker_model
    if _reranker_model is None and agent_config.ENABLE_RERANKING:
        model_name = agent_config.RERANKER_MODEL
        # Initialize cross-encoder model
        hf_cross_encoder = HuggingFaceCrossEncoder(model_name=model_name)
        
        # Create custom reranker implementing BaseDocumentCompressor
        # Note: CrossEncoderReranker needs to inherit from BaseDocumentCompressor
        # and implement compress_documents(documents, query, callbacks=None)
        _reranker_model = CrossEncoderReranker(
            model=hf_cross_encoder, 
            top_n=agent_config.RERANKER_TOP_N
        )
    return _reranker_model

# Inside perform_hybrid_search, after getting results:
if agent_config.ENABLE_RERANKING and len(search_results) > agent_config.RERANKER_TOP_N:
    reranker = get_reranker()
    if reranker:
        # Convert results to Document format for reranker
        from langchain_core.documents import Document
        docs = [Document(page_content=r["text"], metadata=r.get("metadata", {})) for r in search_results]
        
        # Reranker expects documents and query parameters
        reranked_docs = reranker.compress_documents(
            documents=docs, 
            query=query_text,
            callbacks=None  # Or pass run_config.get("callbacks")
        )
        
        # Convert back to expected format, preserving all fields
        search_results = []
        for doc in reranked_docs:
            result = {
                "text": doc.page_content,
                "metadata": doc.metadata,
                "relevance": doc.metadata.get("relevance_score", 1.0)
            }
            search_results.append(result)
```

### 2.2 Modify `src/retrieval_agent/agent_config.py`

**Changes Required:**
1. Add re-ranking configuration

**Specific Code Changes:**
```python
# Line 60: Add re-ranking config
# Re-ranking Configuration
ENABLE_RERANKING = True
RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANKER_TOP_N = 5
PRE_RERANK_MULTIPLIER = 3  # Get 3x results before re-ranking
```

## Phase 3: Contextual Compression (Priority: Medium)

### 3.1 Modify `src/retrieval_agent/agent_tools.py`

**Changes Required:**
1. Add LLM-based compression capabilities
2. Create compression functions

**Specific Code Changes:**
```python
# Line 20: Add imports (corrected based on LangChain reference)
from langchain_community.document_compressors import LLMChainExtractor, LLMChainFilter
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

# Line 40: Add compression initialization
_llm_extractor = None

def get_llm_extractor():
    global _llm_extractor
    if _llm_extractor is None and agent_config.ENABLE_CONTEXTUAL_COMPRESSION:
        # Use existing LLM instance
        from src.llm_handler import LLMHandler
        llm_handler = LLMHandler()
        # LLMChainExtractor requires specific initialization
        _llm_extractor = LLMChainExtractor.from_llm(llm_handler.llm)
        # Alternative: Use LLMChainFilter for yes/no filtering
        # _llm_filter = LLMChainFilter.from_llm(llm_handler.llm)
    return _llm_extractor

# Add compression option to search functions
def compress_results(results: List[Dict], query: str) -> List[Dict]:
    if not agent_config.ENABLE_CONTEXTUAL_COMPRESSION:
        return results
    
    extractor = get_llm_extractor()
    if not extractor:
        return results
    
    # Convert to Documents, compress, convert back
    docs = [Document(page_content=r["text"], metadata=r.get("metadata", {})) for r in results]
    # Call compress_documents with proper parameters
    compressed_docs = extractor.compress_documents(
        documents=docs,
        query=query,
        callbacks=None  # Or pass callback manager
    )
    
    compressed_results = []
    for doc in compressed_docs:
        result = {
            "text": doc.page_content,
            "metadata": doc.metadata,
            "relevance": doc.metadata.get("relevance_score", 1.0)
        }
        compressed_results.append(result)
    
    return compressed_results
```

### 3.2 Modify `src/retrieval_agent/agent_config.py`

**Changes Required:**
1. Add compression configuration

**Specific Code Changes:**
```python
# Line 65: Add compression config
# Contextual Compression Configuration
ENABLE_CONTEXTUAL_COMPRESSION = True
COMPRESSION_METHOD = "extract"  # "extract" or "filter"
MAX_COMPRESSED_LENGTH = 500  # Characters
```

## Phase 4: Query Understanding Enhancements (Priority: Medium)

**Note**: For Self-Query Retriever, use `langchain.retrievers.self_query.base.SelfQueryRetriever` with proper `AttributeInfo` definitions.

### 4.1 Modify `src/retrieval_agent/agent_prompts.py`

**Changes Required:**
1. Enhance query understanding prompt for structured filters
2. Add HyDE prompt template

**Specific Code Changes:**
```python
# Update QUERY_UNDERSTANDING_SYSTEM_MESSAGE
QUERY_UNDERSTANDING_SYSTEM_MESSAGE = """You are a legal document analysis expert. Analyze queries to:
1. Identify main intent and sub-queries
2. Extract structured filters with operators
3. Suggest document types and metadata filters
4. Identify if hypothetical document generation would help

Available metadata fields:
- category: Document category (exact match from: {categories})
- exhibit_number: Integer exhibit number
- date_range: Date filtering (use ISO format)
- document_type: "scanned", "text", "mixed"
- is_ocr_processed: Boolean
"""

# Add HyDE template
HYDE_PROMPT_TEMPLATE = """Write a detailed legal document excerpt that would perfectly answer this question: {question}

The document should:
1. Use appropriate legal terminology
2. Include specific details that would be found in real documents
3. Be written in the style of {document_type}

Document excerpt:"""
```

### 4.2 Modify `src/retrieval_agent/output_parsers.py`

**Changes Required:**
1. Enhance QueryAnalysis model for structured filters

**Specific Code Changes:**
```python
# Update QueryAnalysis class
class QueryAnalysis(BaseModel):
    """Enhanced query analysis with structured filters."""
    main_intent: str = Field(description="The primary goal")
    sub_queries: List[str] = Field(description="Decomposed sub-queries")
    search_keywords: List[str] = Field(description="Key terms")
    
    # Enhanced filter support
    structured_filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured filters with operators (eq, gt, lt, contains)"
    )
    
    # HyDE suggestion
    use_hypothetical_document: bool = Field(
        default=False,
        description="Whether to use HyDE for this query"
    )
    hypothetical_document_type: Optional[str] = Field(
        default=None,
        description="Type of document to generate (pleading, medical record, etc.)"
    )
```

## Phase 5: Iteration Logic Improvements (Priority: Low)

### 5.1 Modify `src/retrieval_agent/main_retriever.py`

**Changes Required:**
1. Add information gain calculation
2. Implement query evolution tracking

**Specific Code Changes:**
```python
# Line 300: After processing chunks, calculate information gain
def calculate_information_gain(new_facts: List[Dict], existing_facts: List[Dict]) -> float:
    """Calculate how much new information was gained."""
    if not new_facts:
        return 0.0
    
    # Simple metric: ratio of new unique statements
    existing_statements = {f.get("extracted_statement", "") for f in existing_facts}
    new_statements = {f.get("extracted_statement", "") for f in new_facts}
    unique_new = new_statements - existing_statements
    
    return len(unique_new) / len(new_statements) if new_statements else 0.0

# Use in iteration decision
info_gain = calculate_information_gain(newly_extracted_facts, accumulated_facts)
decision_input["information_gain"] = info_gain
```

### 5.2 Modify `src/retrieval_agent/agent_prompts.py`

**Changes Required:**
1. Update iteration decision prompt

**Specific Code Changes:**
```python
# Update ITERATION_DECISION_HUMAN_MESSAGE
ITERATION_DECISION_HUMAN_MESSAGE = """Analyze search progress and decide next steps.

Query: {original_query}
Executed: {executed_queries_list}
Information Gain: {information_gain:.2%}
Categories Found: {categories_explored}

Current Facts:
{retrieved_facts_summary}

Iteration: {current_iteration}/{max_iterations}

Decision criteria:
1. If information gain < 20%, try different approach
2. If key aspects unanswered, generate specific queries
3. If diminishing returns, stop iteration

Return JSON:
{{
    "continue_iteration": boolean,
    "next_sub_queries": ["specific gap-filling queries"],
    "search_strategy": "vector|hybrid|keyword",
    "reasoning": "explanation"
}}"""
```

## Verification Criteria

### Phase 1 Verification: Hybrid Search
1. **Functional Test**:
   ```bash
   python -m src.retrieval_agent.cli "Find contract terms" --debug | grep "Hybrid search"
   ```
   - Expected: Log shows "Hybrid search ('hybrid') returned X results"
   - Both vector and keyword results should appear

2. **Performance Test**:
   - Hybrid search should return more relevant results than vector-only
   - Response time should be < 2x vector-only search

3. **Integration Test**:
   - PostgreSQL results should merge correctly with vector results
   - Scores should be normalized appropriately

### Phase 2 Verification: Re-ranking
1. **Quality Test**:
   ```bash
   python -m src.retrieval_agent.cli "YMCA liability" --save-results
   ```
   - Top 5 results should be more relevant than without re-ranking
   - Compare relevance scores before/after

2. **Performance Test**:
   - Re-ranking overhead should be < 1 second for 15 documents
   - Memory usage should remain stable

### Phase 3 Verification: Contextual Compression
1. **Output Test**:
   - Compressed chunks should be 50-70% shorter
   - Key information should be preserved
   - No hallucinations in compressed text

2. **Relevance Test**:
   - Compressed results should maintain relevance to query
   - Facts extracted should remain accurate

### Phase 4 Verification: Query Understanding
1. **Filter Test**:
   ```bash
   python -m src.retrieval_agent.cli "Medical records from 2023" --debug
   ```
   - Should generate date range filter
   - Should identify "Medical Record" category

2. **HyDE Test**:
   - When enabled, should generate hypothetical documents
   - Search with HyDE should find different/better results

### Phase 5 Verification: Iteration Logic
1. **Information Gain Test**:
   - Low gain iterations should trigger strategy changes
   - High gain should continue with refined queries

2. **Convergence Test**:
   - System should stop when no new information found
   - Should not repeat identical queries

## Implementation Schedule

### Week 1: Phase 1 (Hybrid Search)
- Day 1-2: Implement agent_tools.py changes
- Day 3: Update main_retriever.py
- Day 4: Testing and debugging
- Day 5: Performance optimization

### Week 2: Phase 2 (Re-ranking)
- Day 1-2: Integrate cross-encoder
- Day 3: Update search functions
- Day 4-5: Testing and tuning

### Week 3: Phase 3 (Contextual Compression)
- Day 1-2: Implement compression functions
- Day 3: Integration testing
- Day 4-5: Performance optimization

### Week 4: Phase 4-5 (Query Understanding & Iteration)
- Day 1-2: Enhanced prompts
- Day 3: Parser updates
- Day 4-5: Full system testing

## Risk Mitigation

1. **Performance Degradation**:
   - Add feature flags for each enhancement
   - Implement caching for expensive operations
   - Monitor latency at each step

2. **Memory Issues**:
   - Implement singleton patterns for models
   - Clear large objects after use
   - Add memory monitoring

3. **Integration Conflicts**:
   - Test each phase independently
   - Maintain backward compatibility
   - Add comprehensive logging

## Success Metrics

1. **Retrieval Quality**: 30% improvement in relevance scores
2. **Response Time**: < 10 seconds for complex queries
3. **Coverage**: Find relevant documents missed by current system
4. **Accuracy**: 95% of compressed/extracted facts remain accurate
5. **Efficiency**: Reduce iterations needed by 40%

## Conclusion

This phased approach allows incremental improvements while maintaining system stability. Each phase builds on the previous, with clear verification criteria ensuring quality at each step. No new scripts are created; all enhancements are integrated into existing files.

## Key LangChain Implementation Notes

Based on the reference repository:

1. **Imports**: Use `langchain_core` for base classes and interfaces
2. **Document Handling**: Always use `langchain_core.documents.Document`
3. **Callbacks**: Pass `CallbackManagerForRetrieverRun` through retriever chains
4. **Compressors**: Must implement `BaseDocumentCompressor` interface
5. **Retrievers**: Must inherit from `BaseRetriever` and implement `_get_relevant_documents`
6. **Async Support**: Implement `_aget_relevant_documents` for async operations
7. **Pydantic**: Use `ConfigDict` with `arbitrary_types_allowed=True`
8. **Store Pattern**: Use `BaseStore[str, Document]` for document storage
9. **Ensemble Pattern**: Use Reciprocal Rank Fusion (RRF) for result merging
10. **Parent-Child**: Use `MultiVectorRetriever` as base for parent document retrieval

## Additional Implementation Details

### Proper Async Support Pattern
```python
from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun

class CustomRetriever(BaseRetriever):
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # Sync implementation
        pass
    
    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        # Async implementation
        pass
```

### Proper Document Compressor Pattern
```python
from langchain_core.documents import BaseDocumentCompressor

class CustomReranker(BaseDocumentCompressor):
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        # Implementation
        pass
```

### Multi-Vector Retriever Setup
```python
# For storing multiple representations per document
from langchain.retrievers import MultiVectorRetriever
from langchain_core.stores import InMemoryStore

# During indexing:
docstore = InMemoryStore()
vectorstore = Chroma(collection_name="multi_vector", embedding_function=embeddings)

retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    id_key="doc_id",
    search_kwargs={"k": 5}
)
```

## Pydantic Model & RDS Compliance Requirements

### Critical Compliance Guidelines

This section emphasizes the importance of maintaining simple, database-compatible Pydantic models throughout the enhancement implementation. **ALL modifications must respect existing schema constraints**.

### 1. **Existing Pydantic Models to Preserve**

The following models are already in production and must NOT be modified in ways that break compatibility:

```python
# From output_parsers.py - DO NOT break these interfaces
class QueryAnalysisResult(BaseModel):
    main_intent: str
    sub_queries: List[str] = Field(default_factory=list)
    search_keywords: List[str] = Field(default_factory=list)
    potential_filters: Optional[Dict[str, Any]] = None
    analysis_notes: str

class ExtractedFact(BaseModel):
    is_relevant: bool
    extracted_statement: Optional[str] = None
    relevance_score_assessment: str  # Must be: High, Medium, Low, or Irrelevant
    reasoning_for_relevance: Optional[str] = None

class IterationDecisionOutput(BaseModel):
    continue_iteration: bool
    next_sub_query: Optional[str] = None
    next_keywords: Optional[List[str]] = Field(default_factory=list)
    next_filters: Optional[Dict[str, Any]] = None
    reasoning: str
```

### 2. **RDS Schema Constraints**

The PostgreSQL schema has fixed column types that must be respected:

```sql
-- From db_storage.py - Fixed schema
CREATE TABLE document_texts (
    exhibit_id INTEGER NOT NULL UNIQUE,
    original_filename VARCHAR(500) NOT NULL,
    bates_start VARCHAR(20) NOT NULL,
    bates_end VARCHAR(20) NOT NULL,
    category VARCHAR(100),
    full_text TEXT,
    page_count INTEGER,
    char_count INTEGER
);
```

**Compliance Rules**:
- Exhibit IDs must remain integers
- Filenames limited to 500 characters
- Bates numbers limited to 20 characters
- Categories must match predefined list
- No complex JSON types in main columns

### 3. **Model Enhancement Guidelines**

When adding enhancements, follow these rules:

#### ✅ ALLOWED Enhancements:
```python
# Adding optional fields
class EnhancedQueryAnalysis(QueryAnalysisResult):
    # Inherits all base fields
    use_hypothetical_document: bool = False
    search_strategy: str = "vector"  # Simple string enum
    
# Adding helper methods
class ExtractedFactWithHelpers(ExtractedFact):
    def is_high_relevance(self) -> bool:
        return self.relevance_score_assessment == "High"
```

#### ❌ PROHIBITED Changes:
```python
# DON'T change field types
class BadQueryAnalysis(BaseModel):
    main_intent: List[str]  # NO! Was string
    sub_queries: str  # NO! Was List[str]
    
# DON'T add complex nested structures
class OverlyComplexFact(BaseModel):
    nested_metadata: Dict[str, Dict[str, List[Any]]]  # NO!
    
# DON'T break RDS compatibility
class IncompatibleResult(BaseModel):
    exhibit_id: str  # NO! Must be int for RDS
    bates_range: Tuple[str, str]  # NO! Must be simple strings
```

### 4. **Validation Requirements**

All enhanced models must:

1. **Maintain backward compatibility**
   ```python
   # Old code must still work
   result = QueryAnalysisResult(main_intent="find X", analysis_notes="simple")
   ```

2. **Use simple types for RDS fields**
   ```python
   # Good - maps directly to PostgreSQL
   exhibit_id: int
   category: str
   
   # Bad - requires complex serialization
   exhibit_data: Dict[str, Any]
   category_tree: List[Dict[str, str]]
   ```

3. **Validate against existing constraints**
   ```python
   # Add validators that respect DB constraints
   @validator('category')
   def validate_category(cls, v):
       allowed = ["Pleading", "Medical Record", "Bill", ...]
       if v not in allowed:
           raise ValueError(f"Category must be one of: {allowed}")
       return v
   ```

### 5. **Implementation Checklist**

Before implementing any enhancement:

- [ ] Does it modify existing Pydantic field types? (If yes, STOP)
- [ ] Does it add complex nested structures? (Keep it simple)
- [ ] Can it be stored directly in PostgreSQL? (Must be yes)
- [ ] Will existing code continue to work? (Must be yes)
- [ ] Are all new fields optional with defaults? (Preferred)

### 6. **Testing Requirements**

All Pydantic model changes must include:

```python
# Test backward compatibility
def test_model_backward_compatibility():
    # Old format must still parse
    old_data = {"main_intent": "test", "analysis_notes": "note"}
    result = QueryAnalysisResult(**old_data)
    assert result.main_intent == "test"

# Test RDS compatibility
def test_rds_serialization():
    # Must serialize to simple types
    fact = ExtractedFact(is_relevant=True, relevance_score_assessment="High")
    serialized = fact.dict()
    # All values must be JSON-serializable and RDS-compatible
    assert all(isinstance(v, (str, int, float, bool, type(None))) 
              for v in serialized.values())
```

### 7. **Summary**

The enhancement implementation MUST:
- ✅ Preserve all existing Pydantic model interfaces
- ✅ Respect PostgreSQL column type constraints
- ✅ Keep new fields optional with sensible defaults
- ✅ Use simple, serializable types only
- ✅ Maintain full backward compatibility
- ❌ NOT modify existing field types or names
- ❌ NOT add complex nested structures
- ❌ NOT break RDS foreign key relationships

**Remember**: The legal document processing system is in production. Any model changes that break existing functionality or database compatibility will cause system failures. When in doubt, extend rather than modify.