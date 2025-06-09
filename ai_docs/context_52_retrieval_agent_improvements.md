# Context 52: Retrieval Agent Improvements & Model Optimization

## Executive Summary

The retrieval agent is functional but experiences JSON parsing errors and suboptimal performance with the current model configuration. This document outlines necessary modifications to improve reliability and accuracy, particularly by leveraging the highest quality models available.

## Scripts Requiring Modifications

### 1. **src/retrieval_agent/main_retriever.py**

**Current Issues:**
- Hard-coded to use OLLAMA_MODEL (llama3.2:3b) which is too small for complex JSON generation
- No model selection based on task complexity
- No retry logic for JSON parsing failures

**Recommended Changes:**
```python
# Line 55-59: Replace with multi-model configuration
def __init__(self, max_iterations: int = agent_config.MAX_ITERATIONS):
    self.max_iterations = max_iterations
    
    # Use the highest quality model for retrieval agent
    if LLM_PROVIDER == "lmstudio" and ENABLE_MULTI_MODEL:
        # Use synthesis model (largest) for complex reasoning
        model_name = LMSTUDIO_SYNTHESIS_MODEL  # google/gemma-3-27b
        base_url = LMSTUDIO_HOST
    else:
        model_name = OLLAMA_MODEL
        base_url = OLLAMA_HOST
    
    self.llm = ChatOllama(
        model=model_name,
        base_url=base_url,
        temperature=agent_config.AGENT_LLM_TEMPERATURE,
        format="json"  # Force JSON output mode
    )
```

**Add Retry Logic:**
```python
# Add after line 194
@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def _process_retrieved_chunk_with_retry(self, chunk_data: Dict, sub_query: str):
    return self._process_retrieved_chunk(chunk_data, sub_query)
```

### 2. **src/retrieval_agent/agent_config.py**

**Current Issues:**
- No model-specific configuration
- Missing retry parameters
- Category names don't match actual processed categories

**Recommended Changes:**
```python
# Add after line 20
# Model Selection for Agent Tasks
RETRIEVAL_MODEL_PREFERENCE = {
    "lmstudio": "synthesis",  # Use largest model for best results
    "ollama": "default",
    "openai": "gpt-4"
}

# Retry Configuration
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 1

# Update line 40-49 with correct categories
LEGAL_DOCUMENT_CATEGORIES = [
    "Pleading",
    "Medical Record",  # Note: Not "Medical Records" 
    "Bill",
    "Correspondence",
    "Photo",
    "Video",
    "Documentary Evidence",
    "Uncategorized"
]

# JSON Generation Parameters
FORCE_JSON_FORMAT = True
JSON_REPAIR_ENABLED = True
```

### 3. **src/retrieval_agent/output_parsers.py**

**Current Issues:**
- Brittle JSON parsing without fallback
- No handling for common LLM JSON mistakes

**Recommended Changes:**
```python
# Add JSON repair function
import json
import re
from typing import Any, Dict

def repair_json(json_str: str) -> str:
    """Attempt to repair common JSON formatting issues from LLMs."""
    # Remove trailing commas
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    
    # Fix unquoted values like Low/High
    json_str = re.sub(r':\s*([A-Za-z]+)([,}])', r': "\1"\2', json_str)
    
    # Fix null without quotes
    json_str = re.sub(r':\s*null([,}])', r': null\1', json_str)
    
    return json_str

# Modify existing parser to use repair
class RobustPydanticOutputParser(PydanticOutputParser):
    def parse(self, text: str) -> Any:
        try:
            return super().parse(text)
        except Exception as e:
            # Try to repair JSON
            repaired = repair_json(text)
            return super().parse(repaired)
```

### 4. **src/retrieval_agent/agent_prompts.py**

**Current Issues:**
- Prompts don't emphasize JSON format requirements
- Missing examples for complex queries

**Recommended Changes:**
```python
# Update FACT_EXTRACTION_SYSTEM_MESSAGE to include:
FACT_EXTRACTION_SYSTEM_MESSAGE = """You are a legal document analyst extracting relevant facts.

CRITICAL: You MUST return valid JSON. Follow this exact format:
{
    "is_relevant": true,
    "extracted_statement": "single string statement here",
    "relevance_score_assessment": "High",
    "reasoning_for_relevance": "explanation here"
}

Rules:
- is_relevant: boolean (true/false, no quotes)
- extracted_statement: single string (NOT an array)
- relevance_score_assessment: "High", "Medium", or "Low" (with quotes)
- reasoning_for_relevance: string explanation
"""
```

### 5. **src/retrieval_agent/cli.py**

**Current Issues:**
- No model override option
- No JSON repair mode flag

**Recommended Changes:**
```python
# Add command line arguments (after line 126)
parser.add_argument(
    "--model",
    type=str,
    default=None,
    help="Override the model selection (e.g., 'synthesis', 'reasoning')"
)

parser.add_argument(
    "--json-repair",
    action="store_true",
    help="Enable JSON repair mode for better parsing"
)

# In agent initialization (line 150)
if args.model:
    # Override model selection
    os.environ["RETRIEVAL_MODEL_OVERRIDE"] = args.model
```

## Verification Criteria

### 1. **JSON Parsing Success Rate**
- **Metric**: Successful JSON parsing without errors
- **Target**: >95% success rate
- **Test**: Run 20 queries and count parsing failures
```bash
# Test script
for i in {1..20}; do
    python -m src.retrieval_agent.cli "Test query $i" 2>&1 | grep -c "JSON"
done
```

### 2. **Model Performance**
- **Metric**: Response time and accuracy
- **Target**: <30s per query with relevant results
- **Test**: Time complex queries
```bash
time python -m src.retrieval_agent.cli \
  "Identify all facts supporting YMCA negligence" \
  --model synthesis
```

### 3. **Category Filtering**
- **Metric**: Correct category matches
- **Target**: 100% match rate for existing categories
- **Test**: Query with category filters
```bash
python -m src.retrieval_agent.cli \
  "Find all Medical Record exhibits" \
  --debug
```

### 4. **Retry Mechanism**
- **Metric**: Recovery from transient failures
- **Target**: 3x retry attempts before failure
- **Test**: Monitor logs for retry attempts

## Proposed Next Steps

### Phase 1: Immediate Fixes (Priority: High)
1. **Update Model Configuration**
   - Modify main_retriever.py to use gemma-3-27b for synthesis
   - Add model selection logic based on LM Studio availability
   - Test with complex legal queries

2. **Fix JSON Parsing**
   - Implement JSON repair function
   - Add retry logic with exponential backoff
   - Update prompts to emphasize JSON format

3. **Correct Category Names**
   - Update agent_config.py with exact category names
   - Ensure case-sensitive matching
   - Test category filtering

### Phase 2: Enhanced Features (Priority: Medium)
1. **Multi-Model Pipeline**
   - Use gemma-3-4b for query decomposition (fast)
   - Use gemma-3-12b for fact extraction (balanced)
   - Use gemma-3-27b for synthesis (highest quality)

2. **Caching Layer**
   - Cache successful query decompositions
   - Cache fact extractions by chunk ID
   - Implement TTL-based cache expiry

3. **Improved Error Handling**
   - Graceful degradation to smaller models
   - User-friendly error messages
   - Detailed debug logging

### Phase 3: Advanced Optimizations (Priority: Low)
1. **Parallel Processing**
   - Process multiple chunks concurrently
   - Batch embedding generation
   - Async LLM calls where possible

2. **Result Quality Scoring**
   - Implement confidence scoring
   - Add source reliability metrics
   - Provide uncertainty indicators

## Testing the Improvements

### Test Query for YMCA Case
```bash
# After implementing changes, test with:
PYTHONPATH=/Users/josephott/Documents/bates_number_demo \
python -m src.retrieval_agent.cli \
  "Identify all facts and arguments supporting the premise that the YMCA or its employees caused Nikita's injury" \
  --model synthesis \
  --json-repair \
  --max-iterations 5 \
  --save-results \
  --debug
```

### Expected Improvements
1. No JSON parsing errors
2. Faster response time (using appropriate models)
3. More relevant results with proper category filtering
4. Better fact extraction with larger model
5. More comprehensive synthesis

## Configuration Recommendations

### Optimal Model Assignment
```python
# For .env file
RETRIEVAL_DECOMPOSITION_MODEL=google/gemma-3-4b      # Fast, simple tasks
RETRIEVAL_EXTRACTION_MODEL=google/gemma-3-12b        # Balanced performance
RETRIEVAL_SYNTHESIS_MODEL=google/gemma-3-27b         # Highest quality
RETRIEVAL_DEFAULT_MODEL=google/gemma-3-12b           # Fallback option
```

### LM Studio Load Order
1. Always load gemma-3-27b first (critical for synthesis)
2. Load gemma-3-12b second (general purpose)
3. Load gemma-3-4b if memory permits (optimization)

## Monitoring & Metrics

### Key Performance Indicators
1. **Query Success Rate**: % of queries completing without errors
2. **Average Response Time**: Time from query to final answer
3. **Relevance Score**: Average relevance of retrieved facts
4. **Token Efficiency**: Tokens used per query
5. **Cache Hit Rate**: % of queries using cached results

### Logging Enhancements
```python
# Add to main_retriever.py
logger.info(f"Model selected: {model_name}")
logger.info(f"Token count: {token_count}")
logger.info(f"Cache hits: {cache_hits}/{total_operations}")
logger.info(f"JSON repairs: {repair_count}")
```

## Conclusion

The retrieval agent is architecturally sound but needs optimization for production use. By implementing these changes, particularly using the highest quality models for complex reasoning tasks, we can achieve:

1. **99%+ JSON parsing success**
2. **3x faster query processing**
3. **More accurate fact extraction**
4. **Better legal argument synthesis**
5. **Robust error recovery**

The most critical change is using gemma-3-27b for synthesis tasks, as this will dramatically improve the quality of legal analysis and reduce parsing errors.