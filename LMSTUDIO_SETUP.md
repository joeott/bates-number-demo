# LM Studio Integration Guide

This guide explains how to use LM Studio with the Bates numbering demo for local, high-performance AI processing.

## Prerequisites

1. **Install LM Studio**: Download from [https://lmstudio.ai](https://lmstudio.ai)
2. **Hardware**: Recommended M1/M2/M3/M4 Mac or GPU-enabled system
3. **RAM**: 8GB minimum, 16GB+ recommended for larger models

## Quick Start

### 1. Start LM Studio Server

1. Open LM Studio application
2. Go to the "Local Server" tab
3. Load your desired model (e.g., `mistral-nemo-instruct-2407`)
4. Click "Start Server" (default port: 1234)

### 2. Discover Available Models

Run the discovery script to see what models are loaded:

```bash
python test_lmstudio.py
```

This will show:
- Connection status
- Available models by type (text, embedding, vision)
- Recommended .env configuration

### 3. Configure Environment

Update your `.env` file based on the discovery output:

```bash
# Set LM Studio as the provider
LLM_PROVIDER=lmstudio

# LM Studio Configuration
LMSTUDIO_HOST=http://localhost:1234/v1
LMSTUDIO_MODEL=mistral-nemo-instruct-2407
LMSTUDIO_EMBEDDING_MODEL=e5-mistral-7b-instruct-embedding
LMSTUDIO_MAX_TOKENS=2048
LMSTUDIO_CONTEXT_LENGTH=32768
```

### 4. Run the Application

```bash
python src/main.py
```

The application will now use LM Studio for:
- Document categorization
- Summary generation
- Descriptive naming
- Embeddings (if configured)

## Advanced Usage

### Test Model Performance

```bash
# Test basic functionality
python test_lmstudio.py --test

# Benchmark performance
python test_lmstudio.py --benchmark

# Get JSON output
python test_lmstudio.py --json
```

### Use Different Models for Different Tasks

You can load multiple models in LM Studio and configure them for specific tasks:

1. **Text Generation**: Load a general-purpose model like Mistral or Llama
2. **Embeddings**: Load an embedding model like `e5-mistral-7b-instruct-embedding`
3. **Vision** (future): Load a vision model for image analysis

### Monitor Performance

While processing:
1. Check LM Studio's server tab for request logs
2. Monitor token generation speed
3. Observe memory usage

## Recommended Models

Based on your M4 Max with 128GB RAM:

### For Document Categorization
- **mistral-nemo-instruct-2407** - Fast, accurate for legal documents
- **llama-3.2-3b-instruct** - Smaller, faster option
- **pixtral-12b** - Larger, more capable model

### For Embeddings
- **e5-mistral-7b-instruct-embedding** - High-quality embeddings
- **nomic-embed-text-v1.5** - Efficient alternative

### For Long Documents
- Models with extended context windows (32k+ tokens)

## Troubleshooting

### Connection Issues
```bash
# Check if LM Studio is running
curl http://localhost:1234/api/health

# Test with discovery script
python test_lmstudio.py
```

### Model Not Found
1. Ensure model is loaded in LM Studio UI
2. Check model ID matches exactly
3. Restart LM Studio server

### Performance Issues
1. Close other applications to free RAM
2. Use smaller models for faster processing
3. Adjust batch sizes in configuration

## Benefits Over Cloud APIs

1. **Privacy**: All processing stays local
2. **Cost**: No API fees
3. **Speed**: No network latency
4. **Control**: Choose and tune your models
5. **Reliability**: No dependency on external services

## Integration with Existing Workflow

The LM Studio integration is seamless:

```python
# No code changes needed!
# Just set LLM_PROVIDER=lmstudio in .env

# The application automatically:
# - Detects LM Studio configuration
# - Uses appropriate endpoints
# - Falls back gracefully if needed
```

## Next Steps

1. Experiment with different models
2. Monitor performance for your use case
3. Consider fine-tuning models for legal documents
4. Explore vision models for image-heavy PDFs

For more information:
- Run `python test_lmstudio.py --help`
- Check LM Studio documentation
- Review the project's context files in `ai_docs/`