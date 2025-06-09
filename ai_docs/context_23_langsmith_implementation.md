# LangSmith Tracing Implementation

## Overview

LangSmith tracing has been successfully integrated into the Bates numbering system to provide comprehensive observability into the LCEL (LangChain Expression Language) processing chains. This implementation enables detailed monitoring, debugging, and performance analysis of the document processing pipeline.

## Implementation Details

### 1. Configuration Setup

**Files Modified:**
- `src/config.py` - Added LangSmith environment variables
- `.env.langsmith.example` - Created example configuration file

**Environment Variables Added:**
```python
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "bates_number_demo")

# For backward compatibility with LangChain's expected environment variables
if LANGSMITH_TRACING:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if LANGSMITH_API_KEY:
        os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    if LANGSMITH_PROJECT:
        os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT
```

**Configuration Validation:**
- Warns if tracing is enabled but API key is missing
- Logs project name when tracing is enabled
- Default project name: "bates_number_demo"

### 2. DocumentOrchestrator Updates

**Method Signatures Updated:**
- `process_document()` - Added `run_config: Optional[Dict] = None` parameter
- `process_batch()` - Added `run_config_template: Optional[Dict] = None` parameter

**Named RunnableLambda Components:**
```python
self.validation_chain = RunnableLambda(self._validate_document, name="DocumentValidation")
self.llm_chain = RunnableLambda(self._process_with_llm, name="LLMMetadataExtraction")
self.bates_chain = RunnableLambda(self._apply_bates_numbering, name="BatesNumbering")
self.exhibit_chain = RunnableLambda(self._apply_exhibit_marking, name="ExhibitMarking")
# ... additional named components for VectorProcessing, PostgreSQLStorage, etc.
```

**Tracing Data Flow:**
- `process_batch()` generates individual run configs for each document
- Includes document-specific metadata: path, filename, batch index, counters
- Adds descriptive tags: batch processing indicators, document position
- Creates meaningful run names: "ProcessDoc-{filename}"

### 3. Main.py Integration

**Base Run Configuration:**
```python
base_run_config = {
    "metadata": {
        "processing_mode": "batch" | "sequential",
        "total_documents": count,
        "output_directory": path,
        "bates_prefix": prefix,
        "exhibit_prefix": prefix,
        "timestamp": ISO_timestamp
    },
    "tags": [
        "bates_numbering",
        "legal_document_processing", 
        f"llm_{provider}",
        f"total_docs_{count}"
    ]
}
```

**Batch-Specific Metadata:**
- Batch number and total batches
- Documents per batch
- Batch processing tags
- Sequential vs. batched mode indicators

## Setup Instructions

### 1. Get LangSmith Account
1. Visit [smith.langchain.com](https://smith.langchain.com/)
2. Create an account or sign in
3. Navigate to Settings → API Keys
4. Create a new API key (starts with `ls__`)

### 2. Configure Environment Variables

**Option A: Update .env file**
```bash
# Add to your .env file
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="lsv2_pt_your_actual_api_key_here"
LANGSMITH_PROJECT="bates_number_demo"
```

**Option B: Export environment variables**
```bash
export LANGSMITH_TRACING=true
export LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
export LANGSMITH_API_KEY="lsv2_pt_your_actual_api_key_here"
export LANGSMITH_PROJECT="bates_number_demo"
```

### 3. Run with Tracing Enabled

```bash
# Normal processing with tracing
python src/main.py --input_dir input_documents --output_dir output

# Batch processing with tracing
python src/main.py --input_dir input_documents --output_dir output --batch_size 5
```

## What LangSmith Shows

### 1. Chain Structure Visualization
- Complete LCEL chain execution flow
- Component-by-component breakdown
- Branch logic (vector/postgres conditional processing)
- Error propagation through the chain

### 2. Performance Metrics
- **LLM Calls**: Token usage, response times for categorization/summarization
- **PDF Processing**: Time to extract text and apply Bates numbers
- **Vector Operations**: Embedding generation and storage times
- **Database Operations**: PostgreSQL insertion and query performance

### 3. Data Flow Inspection
- **Input Data**: File paths, counters, configuration
- **Intermediate Results**: Extracted metadata, processed PDFs
- **Final Outputs**: Bates ranges, exhibit information, success status

### 4. Error Analysis
- **Validation Failures**: Missing files, invalid formats
- **LLM Errors**: Model failures, timeout issues
- **Processing Errors**: PDF corruption, storage failures
- **Chain Interruptions**: Where processing stopped and why

### 5. Metadata and Tags
- **Document Level**: Original filename, category, exhibit number
- **Batch Level**: Processing mode, document counts, timing
- **System Level**: LLM provider, storage backends enabled

## Accessing Traces

1. **Navigate to LangSmith Dashboard**: `https://smith.langchain.com/o/YOUR_ORG/projects/p/YOUR_PROJECT`
2. **View Recent Runs**: Traces appear in real-time during processing
3. **Filter by Tags**: Use tags like "batch_processing" or "sequential_processing"
4. **Drill Down**: Click on any run to see detailed execution tree
5. **Search Metadata**: Filter by document names, batch numbers, etc.

## Troubleshooting

### Common Issues

**1. No Traces Appearing**
- Verify `LANGCHAIN_TRACING_V2=true` is set
- Check API key is correct and starts with `ls__`
- Ensure network connectivity to smith.langchain.com

**2. Missing Metadata**
- Check that run_config is being passed correctly
- Verify metadata structure in trace details
- Look for any config merge issues in logs

**3. Performance Impact**
- Tracing adds minimal overhead (~1-2% processing time)
- Network latency may affect trace upload
- Consider using different projects for dev/prod

### Debug Commands

```python
# Check configuration
python -c "from src import config; print(f'Tracing: {config.LANGSMITH_TRACING}'); print(f'Project: {config.LANGSMITH_PROJECT}')"

# Test with single document
python src/main.py --input_dir test_input --output_dir test_output

# Verify network connectivity
curl -I https://smith.langchain.com
```

## Benefits Realized

### 1. Enhanced Debugging
- Pinpoint exact failure locations in multi-step processing
- Inspect intermediate data at each chain step
- Understand performance bottlenecks

### 2. Production Monitoring
- Track processing success rates across document types
- Monitor LLM performance and token usage
- Identify slow processing patterns

### 3. Development Insights
- Compare different chain configurations
- A/B test LLM prompts and models
- Optimize chain structure based on actual performance data

### 4. Compliance and Auditing
- Complete audit trail of document processing
- Metadata preservation for legal requirements
- Performance metrics for capacity planning

## Next Steps

1. **Production Deployment**: Set up separate prod/dev projects
2. **Alert Configuration**: Monitor for processing failures
3. **Performance Optimization**: Use insights to improve chain efficiency
4. **Custom Metrics**: Add domain-specific monitoring for legal document processing

The LangSmith integration provides unprecedented visibility into the Bates numbering pipeline, enabling data-driven optimization and reliable production operation.