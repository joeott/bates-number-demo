# Context 51: Retrieval Agent Architecture & LM Studio Integration

## Overview

The retrieval agent is an advanced iterative search system built on LangChain Expression Language (LCEL) that performs multi-step document analysis to answer complex legal queries. It's designed to work seamlessly with the existing LM Studio configuration and processed document corpus.

## Architecture Components

### 1. Core Agent (`main_retriever.py`)
- **Class**: `IterativeRetrieverAgent`
- **Purpose**: Orchestrates multi-step retrieval and analysis
- **Process Flow**:
  1. Query Understanding & Decomposition
  2. Iterative Vector Search with Refinement
  3. Fact Extraction & Relevance Assessment
  4. Dynamic Iteration Decision Making
  5. Final Answer Synthesis with Citations

### 2. Configuration (`agent_config.py`)
```python
# Key Configuration Parameters
MAX_ITERATIONS = 3                    # Search-analysis cycles
AGENT_LLM_TEMPERATURE = 0.1          # Low for predictability
NUM_RESULTS_PER_SUB_QUERY = 5        # Chunks per search
CONTEXT_WINDOW_SIZE_FOR_SYNTHESIS = 8000  # Token limit
MAX_CHUNK_LENGTH = 2000              # Character limit per chunk
```

### 3. Tool System (`agent_tools.py`)
- **Primary Tool**: `perform_vector_search`
  - Executes semantic search via ChromaDB
  - Supports metadata filtering (category, exhibit number)
  - Returns relevance-scored chunks with full metadata
  
- **Future Tools** (Defined but not implemented):
  - `perform_postgres_search`: Full-text keyword search
  - `get_document_by_exhibit_number`: Direct exhibit retrieval

### 4. Prompt Engineering (`agent_prompts.py`)
Specialized prompts for legal domain:
- Query decomposition with legal context awareness
- Fact extraction focused on contractual/legal elements
- Iteration decisions based on information sufficiency
- Synthesis with proper legal citations

### 5. Output Parsing (`output_parsers.py`)
Pydantic models for structured outputs:
- `QueryAnalysis`: Sub-queries with keywords and filters
- `ExtractedFact`: Relevant statements with confidence scores
- `IterationDecision`: Continue/stop with reasoning
- `SynthesizedAnswer`: Comprehensive response with citations

## LM Studio Integration

### Model Configuration
The agent automatically detects and uses LM Studio when `LLM_PROVIDER=lmstudio`:

```python
# From current .env configuration
LLM_PROVIDER=lmstudio
LMSTUDIO_HOST=http://localhost:1234/v1
LMSTUDIO_MODEL=google/gemma-3-4b
LMSTUDIO_EMBEDDING_MODEL=text-embedding-snowflake-arctic-embed-l-v2.0
```

### Multi-Model Pipeline Support
When `ENABLE_MULTI_MODEL=true`, the agent can leverage different models for different tasks:
- **Visual Analysis**: `google/gemma-3-12b`
- **Reasoning**: `google/gemma-3-12b`
- **Categorization**: `google/gemma-3-4b`
- **Synthesis**: `google/gemma-3-27b`

### Model Initialization
The agent creates a ChatOllama instance that automatically handles:
- Ollama native models (when using Ollama provider)
- LM Studio models via OpenAI-compatible API
- Temperature control for predictable agent behavior

## CLI Usage (`cli.py`)

### Basic Usage
```bash
# Simple query
python -m src.retrieval_agent.cli "What is the contract price?"

# Complex multi-aspect query
python -m src.retrieval_agent.cli "Find all evidence of breach of contract"

# With custom iterations
python -m src.retrieval_agent.cli --max-iterations 5 "Identify negligence elements"

# Save results
python -m src.retrieval_agent.cli --save-results "What damages were claimed?"
```

### Features
- LangSmith tracing integration (auto-detected from env)
- Debug logging support (`--debug`)
- Result persistence (`--save-results`)
- Custom iteration limits (`--max-iterations`)

## Integration with Existing System

### Vector Store Connection
- Uses the same ChromaDB instance created during document processing
- Accesses embeddings at `output/vector_store`
- Leverages existing metadata (categories, exhibit numbers, Bates ranges)

### Document Metadata
The agent expects and uses:
- `filename`: Original document name
- `exhibit_number`: Assigned exhibit number
- `category`: Document category (Pleading, Medical Record, etc.)
- `bates_start`/`bates_end`: Bates number range
- `page`: Page number within document

### LangChain Components
- **LCEL Chains**: Composable chains for each processing step
- **ChatPromptTemplate**: Structured prompt management
- **PydanticOutputParser**: Type-safe LLM outputs
- **Vector Store Integration**: Direct ChromaDB access

## Performance Optimizations

1. **Chunk Filtering**: 
   - Initial relevance score filtering
   - Maximum chunk length enforcement
   - Metadata-based pre-filtering

2. **Iterative Refinement**:
   - Learns from previous iterations
   - Avoids redundant searches
   - Builds cumulative knowledge

3. **Caching Potential**:
   - Compatible with existing cache infrastructure
   - Can leverage cached embeddings
   - Potential for result caching

## Legal Domain Specialization

### Document Categories
Pre-configured awareness of:
- Pleading
- Medical Record
- Bill
- Correspondence
- Photo/Video
- Documentary Evidence
- Uncategorized

### Query Types Supported
- Contract analysis ("Find breach of contract evidence")
- Negligence investigation ("Identify elements of negligence")
- Damage assessment ("What damages were claimed?")
- Fact finding ("When did the incident occur?")
- Document discovery ("Show all medical records mentioning...")

### Citation Format
Generates proper legal citations:
```
[Filename: contract.pdf, Exhibit: 5, Bates: 000123-000145, Page: 12]
```

## Future Enhancements

### Planned Features
1. **PostgreSQL Integration**: 
   - Hybrid search combining vector + keyword
   - SQL-based filtering for complex queries

2. **Cross-Encoder Re-ranking**:
   - Local re-ranking model support
   - Improved relevance scoring

3. **Multi-Modal Support**:
   - Integration with vision models for scanned documents
   - OCR result incorporation

4. **Advanced Filtering**:
   - Date range queries
   - Multi-field boolean filters
   - Proximity searches

## Testing with Current Setup

### Prerequisites
1. Ensure LM Studio is running with models loaded
2. Verify processed documents exist in `output/`
3. Check vector store at `output/vector_store/`

### Test Queries
```bash
# Test on Recamier v. YMCA documents
python -m src.retrieval_agent.cli "What was the YMCA membership agreement about?"
python -m src.retrieval_agent.cli "Find all medical records for Nikita Wors"
python -m src.retrieval_agent.cli "What evidence exists of negligence?"
```

### Expected Behavior
- Agent will decompose complex queries
- Execute 1-3 iterations of search
- Extract relevant facts with sources
- Synthesize comprehensive answer with citations

## Troubleshooting

### Common Issues
1. **Model Not Found**: Ensure LM Studio has the configured model loaded
2. **Empty Results**: Check if documents were properly vectorized
3. **Slow Performance**: Consider using smaller models or fewer iterations
4. **Memory Issues**: Reduce chunk size or context window

### Debug Mode
Enable detailed logging:
```bash
python -m src.retrieval_agent.cli --debug "your query"
```

This will show:
- Query decomposition details
- Each search iteration
- Fact extraction process
- Decision reasoning
- Final synthesis steps

## Conclusion

The retrieval agent provides a production-ready system for complex legal document analysis, fully integrated with the existing LM Studio setup and processed document corpus. It combines the power of iterative search, LLM reasoning, and proper legal citation formatting to deliver comprehensive answers to complex queries.