# Context 29: Synthetic Implementation Plan for Iterative Retriever Module

## Executive Summary

After deep analysis of the retriever concepts, requirements, and draft implementation, this synthetic plan provides a streamlined, modular approach to building the iterative retriever agent. The plan emphasizes simplicity, durability, repeatability, and maintainability while avoiding unnecessary complexity.

## Core Design Principles

1. **Non-Interference**: Zero modifications to existing functional pipeline
2. **Modular Architecture**: Isolated components with clear interfaces
3. **Production-Ready**: Built for real legal discovery workflows
4. **Testable**: Comprehensive verification through production data
5. **Traceable**: Full LangSmith integration for debugging and optimization
6. **Extensible**: Framework for future legal concept specialization

## Implementation Strategy

### Phase 1: Foundation Infrastructure (Priority: Critical)

**Directory Structure Creation**
```
src/retrieval_agent/
├── __init__.py
├── agent_config.py      # Configuration constants and parameters
├── agent_prompts.py     # LLM prompt templates for each step
├── output_parsers.py    # Pydantic models for structured LLM outputs
├── agent_tools.py       # Langchain tools (vector search, future SQL search)
├── main_retriever.py    # Core iterative agent logic with LCEL chains
└── cli.py              # Command-line interface for testing and verification
```

**Key Implementation Requirements:**
- All scripts must be self-contained within `src/retrieval_agent/`
- No modifications to existing `src/` files except optional import additions
- Must leverage existing `VectorSearcher` from `src/vector_search.py`
- Must use existing `LLMHandler` architecture for consistency

### Phase 2: Core Component Implementation

#### 2.1 Configuration Module (`agent_config.py`)
**Purpose**: Centralized configuration for agent behavior
**Key Constants**:
- `MAX_ITERATIONS = 3` (prevents infinite loops)
- `NUM_RESULTS_PER_SUB_QUERY = 5` (balance quality vs performance)
- `MIN_RELEVANCE_SCORE_FOR_FACTS = 0.6` (noise filtering)
- `CONTEXT_WINDOW_SIZE_FOR_SYNTHESIS = 8000` (LLM context limits)
- `ENABLE_LLM_RE_RANKING = False` (future enhancement flag)

#### 2.2 Prompt Engineering (`agent_prompts.py`)
**Purpose**: Legal-domain-optimized prompts for each cognitive step
**Core Prompts**:
1. **Query Understanding**: Decomposes complex legal queries into searchable components
2. **Fact Extraction**: Analyzes chunks for relevance and extracts precise legal facts
3. **Iteration Decision**: Evaluates completeness and formulates next search strategy
4. **Final Synthesis**: Produces coherent answers with proper legal citations

**Design Principles**:
- Prompts must emphasize precision over creativity (low temperature)
- Must enforce structured JSON outputs for reliability
- Must include explicit citation requirements for legal compliance
- Must handle edge cases (no results, ambiguous queries)

#### 2.3 Output Parsers (`output_parsers.py`)
**Purpose**: Structured data models for LLM outputs
**Core Models**:
- `QueryAnalysisResult`: Sub-queries, keywords, potential filters
- `ExtractedFact`: Relevance assessment, extracted statements, reasoning
- `IterationDecisionOutput`: Continue/stop decision, next query parameters
- `SynthesizedAnswer`: Final answer with citation tracking

#### 2.4 Tool Integration (`agent_tools.py`)
**Purpose**: Langchain tools wrapping existing functionality
**Primary Tool**: `perform_vector_search`
- Wraps existing `VectorSearcher.search()` method
- Handles filter mapping and parameter validation
- Returns serializable results for LCEL chains
- Manages searcher instance lifecycle for performance

#### 2.5 Core Agent Logic (`main_retriever.py`)
**Purpose**: Orchestrates iterative retrieval using LCEL chains
**Architecture**:
```python
class IterativeRetrieverAgent:
    def __init__(self, max_iterations=3):
        # Initialize LLM instances and LCEL sub-chains
        self.query_understanding_chain = prompt | llm | parser
        self.fact_extraction_chain = prompt | llm | parser
        self.iteration_decision_chain = prompt | llm | parser
        self.final_synthesis_chain = prompt | llm | parser
    
    def invoke(self, query: str) -> str:
        # Main iterative loop with state management
```

**Iterative Flow**:
1. **Query Understanding**: Analyze and decompose user query
2. **Search Execution**: Process sub-queries through vector search
3. **Fact Extraction**: Analyze chunks and extract relevant facts
4. **Iteration Decision**: Evaluate completeness and plan next iteration
5. **Final Synthesis**: Generate comprehensive answer with citations

#### 2.6 Command-Line Interface (`cli.py`)
**Purpose**: Production testing and verification interface
**Features**:
- Direct query execution against production vector store
- LangSmith trace integration for debugging
- Configurable iteration limits for testing
- Comprehensive error handling and logging

### Phase 3: Verification and Testing Framework

#### 3.1 Verification Criteria Categories

**A. System Functionality** (VC1.x)
- Successful invocation without crashes
- Proper iteration limit adherence
- Graceful handling of no-result scenarios
- Observable steps through LangSmith traces

**B. Query Understanding** (VC2.x)
- Logical sub-query generation for complex queries
- Accurate keyword and entity extraction
- Appropriate metadata filter suggestions

**C. Iterative Search** (VC3.x)
- Dynamic query evolution across iterations
- Correct filter application in vector searches
- Accurate relevance assessment at chunk level
- Precise fact extraction from relevant chunks

**D. Recursive Reasoning** (VC4.x)
- Contextual refinement based on accumulated facts
- Gap identification and filling strategies
- Drill-down capability for detail exploration
- Cross-iteration evidence synthesis

**E. Answer Generation** (VC5.x)
- Coherent, grammatically correct responses
- Factual accuracy based on retrieved evidence
- Proper citation format and accuracy
- Comprehensive incorporation of relevant facts

**F. Legal Concept Customization** (VC6.x)
- Adaptability to concept-specific prompts
- Concept-relevant retrieval performance

#### 3.2 Testing Phases

**Phase 1: Basic Functionality**
- Simple fact retrieval queries
- No-result handling
- Metadata filter utilization

**Phase 2: Multi-Aspect Queries**
- Two-part legal questions
- Vague queries requiring refinement
- Complex fault-determination scenarios

**Phase 3: Legal Concept Specialization**
- Breach of contract element identification
- Notice requirement verification
- Damages calculation support

**Phase 4: Stress Testing**
- Ambiguous query handling
- Non-existent entity queries
- Edge case robustness

#### 3.3 Test Data and Methodology

**Test Corpus**: Production documents from `input_documents/`
**Test Interface**: `python src/retrieval_agent/cli.py "query"`
**Documentation**: All results logged in sequential `context_X.md` files
**Trace Analysis**: LangSmith traces for each test execution

## Implementation Timeline and Verification Steps

### Step 1: Infrastructure Setup
1. Create directory structure
2. Implement `agent_config.py` with core constants
3. **Verification**: Import test successful

### Step 2: Prompt Engineering
1. Implement `agent_prompts.py` with all templates
2. Implement `output_parsers.py` with Pydantic models
3. **Verification**: Prompt template compilation successful

### Step 3: Tool Integration
1. Implement `agent_tools.py` with vector search tool
2. Test tool independently with simple queries
3. **Verification**: Vector search tool returns expected results

### Step 4: Core Agent Logic
1. Implement `main_retriever.py` with LCEL chains
2. Test individual chains before full integration
3. **Verification**: Each sub-chain processes inputs correctly

### Step 5: CLI and Testing
1. Implement `cli.py` for production testing
2. Execute Phase 1 test cases
3. **Verification**: System completes basic queries without errors

### Step 6: Iterative Testing and Refinement
1. Execute Phases 2-4 test cases systematically
2. Document all results in `context_X.md` files
3. **Verification**: All verification criteria met or documented

## Success Metrics

### Quantitative Metrics
- **Completion Rate**: 95%+ queries complete without unhandled errors
- **Iteration Efficiency**: Average 2.5 iterations per complex query
- **Fact Extraction Accuracy**: 90%+ relevant facts correctly identified
- **Citation Accuracy**: 100% proper source attribution

### Qualitative Metrics
- **Answer Coherence**: Human-readable, legally sound responses
- **Evidence Completeness**: Comprehensive coverage of query aspects
- **Legal Utility**: Answers suitable for discovery and case preparation
- **Trace Clarity**: Clear step-by-step reasoning visible in LangSmith

## Risk Mitigation Strategies

### Technical Risks
- **LLM Hallucination**: Strict fact-based prompting, citation requirements
- **Context Overflow**: Dynamic context management, fact summarization
- **Tool Failures**: Comprehensive error handling, graceful degradation
- **Performance Issues**: Configurable limits, async opportunities

### Operational Risks
- **Prompt Brittleness**: Modular prompt design for easy iteration
- **Model Dependencies**: Consistent use of existing LLM infrastructure
- **Integration Complexity**: Minimal external dependencies, clear interfaces

## Future Enhancement Opportunities

### Near-Term (Post-Implementation)
- **SQL Database Integration**: PostgreSQL keyword search tool
- **Re-ranking Models**: Local cross-encoder for relevance improvement
- **Concept Libraries**: Pre-built prompts for common legal concepts

### Long-Term (Future Phases)
- **Multi-Modal Support**: Document image and table processing
- **Federated Search**: Cross-case evidence discovery
- **Interactive Refinement**: User feedback integration
- **Automated Citation**: Bluebook and legal citation formatting

## Conclusion

This synthetic plan provides a clear, modular path to implementing a production-ready iterative retriever agent. By emphasizing simplicity, thorough testing, and minimal system interference, the implementation will deliver reliable legal document analysis capabilities while maintaining the existing system's stability and performance.

The verification framework ensures systematic validation of all capabilities, while the modular design supports future enhancements and specialization for specific legal concepts. Success depends on disciplined implementation of each phase with comprehensive testing against production data.