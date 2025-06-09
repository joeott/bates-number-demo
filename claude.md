# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI-powered legal document processing tool that automates Bates numbering and exhibit marking for discovery documents. The system categorizes PDFs using LLMs, applies sequential Bates stamps, marks exhibits, and organizes documents by category while maintaining a detailed CSV log.

## Common Development Commands

### Setup and Environment
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.template .env
# Edit .env with API keys and provider settings
```

### Running the Application
```bash
# Basic usage
python src/main.py

# With custom directories
python src/main.py --input_dir path/to/docs --output_dir path/to/output

# With custom exhibit/Bates configuration
python src/main.py --exhibit_prefix "Ex. " --bates_prefix "ABC" --bates_digits 6

# Using local Ollama models
python setup_ollama.py  # One-time setup
# Set LLM_PROVIDER=ollama in .env
python src/main.py
```

### Testing
```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# Run with coverage
pytest --cov=src tests/
```

## Architecture and Code Structure

### Core Components
- **src/main.py**: Entry point, orchestrates the entire document processing workflow
- **src/config.py**: Environment configuration management, handles OpenAI/Ollama provider switching
- **src/llm_handler.py**: Unified LLM interface supporting both OpenAI API and local Ollama models
- **src/pdf_processor.py**: PDF manipulation including Bates stamping and exhibit marking using PyPDF/ReportLab
- **src/utils.py**: Shared utilities for logging and directory management
- **src/vector_processor.py**: LangChain-based vector embeddings and ChromaDB management
- **src/db_storage.py**: PostgreSQL integration for full-text search
- **src/search_cli.py**: Command-line interface for hybrid search functionality
- **src/document_orchestrator.py**: Production-ready orchestration with caching and batch processing
- **src/vision_ocr.py**: Vision model OCR extraction for scanned/image PDFs

### Key Design Patterns
1. **Provider Abstraction**: LLMHandler class abstracts OpenAI/Ollama/LM Studio differences, allowing seamless switching via config
2. **Sequential Processing**: Documents processed in sorted order to ensure consistent Bates numbering
3. **Dual Output Strategy**: Generates both Bates-only and full exhibit versions for flexibility
4. **Category-Based Organization**: Auto-creates subdirectories based on LLM classification
5. **LangChain Integration**: Fully refactored to use LangChain for LLMs, embeddings, and vector stores
6. **Hybrid Search**: Combines vector embeddings with PostgreSQL full-text search for comprehensive retrieval
7. **Production Orchestration**: Document processing with caching, parallel processing, and performance monitoring

### Document Categories
- Pleading, Medical Record, Bill, Correspondence, Photo, Video, Documentary Evidence, Uncategorized

### Output Structure
```
output/
├── bates_numbered/       # PDFs with Bates stamps only
├── exhibits/             # PDFs with both Bates and exhibit stamps
│   ├── pleading/
│   ├── medical_record/
│   └── [other categories]/
└── exhibit_log.csv      # Complete tracking with Bates ranges
```

## CODEBASE MAINTENANCE DIRECTIVE

### FILE CREATION RESTRICTIONS
- NEVER create test_*.py files in root directory or scripts/ directory
- NEVER create temporary debug files in production locations
- NEVER create one-off experimental scripts
- ALL tests must go in organized tests/ structure

### TEST ORGANIZATION REQUIREMENTS
- Unit tests: tests/unit/ (isolated component testing)
- Integration tests: tests/integration/ (multi-component interactions)
- E2E tests: tests/e2e/ (full pipeline scenarios)
- Use pytest framework exclusively
- All tests must have clear docstrings explaining purpose

### CORE SCRIPT PROTECTION
- src/ directory contains ONLY production code
- Modifications to core scripts must be minimal and well-documented
- New functionality added through configuration, not new files
- Core modules: main.py, config.py, llm_handler.py, pdf_processor.py, utils.py, vector_processor.py, db_storage.py, search_cli.py, document_orchestrator.py

### DEBUGGING PROTOCOL
- For debugging: use existing tests in tests/ structure
- For exploration: create temporary files with explicit deletion plan
- For verification: add to existing test suites, don't create new files
- Document debugging findings in ai_docs/ context files

### WHEN TO CREATE NEW FILES
- Only when implementing new core functionality
- Only when approved through architectural review
- Only when no existing file can be extended
- Must follow established naming conventions

### CLEANUP RESPONSIBILITY
- Always clean up temporary files
- Archive obsolete code instead of leaving in place
- Consolidate duplicate functionality
- Maintain documentation of changes

### ERROR RESPONSE
If you find yourself creating test_*.py files outside tests/ structure, STOP and:
1. Explain why existing test structure doesn't meet needs
2. Propose proper location in tests/ hierarchy
3. Get approval before proceeding

REMEMBER: This is production code serving legal document processing. Maintain discipline and organization at all times.

## Memory Management

- Actively manage your memory and context using /ai_docs/
- Store plans, concepts, notes and results in dedicated documentation files
- Reference existing context files:
  - context_1_implementation_plan.md - Original implementation blueprint
  - context_2_verification_criteria.md - Testing and verification standards
  - context_3_local_model_implementation.md - Ollama integration details
  - context_18_langchain_refactor_complete.md - LangChain refactoring documentation
  - context_38_manifest_implementation_complete.md - Manifest system for processing tracking
  - context_40_lmstudio_mlx_implementation.md - LM Studio MLX model support
  - context_50_vision_ocr_implementation_complete.md - Vision OCR capabilities
- Plan using deep thought and emphasize the production of reliable, pragmatic and fit for purpose code

## Project Memory

- This project is a robust local implementation of a RAG (Retrieval-Augmented Generation) process
- ALL data, embeddings, etc. are saved in appropriate local storage:
  - ChromaDB: `vector_store/` directory for embeddings
  - PostgreSQL: Optional database for full-text search
  - Cache: `cache/` directory for LLM response caching
  - Manifests: JSON files tracking processing state

## LM Studio Support

- Supports Apple Silicon MLX models via LM Studio's OpenAI-compatible API
- Run `python test_lmstudio.py` to discover available models
- Configure in .env with `LLM_PROVIDER=lmstudio`
- See LMSTUDIO_SETUP.md for detailed configuration

## Search and Retrieval

The system provides hybrid search capabilities:
```bash
# Semantic vector search
python src/search_cli.py "legal motion"

# Full-text PostgreSQL search
python src/search_cli.py "exact phrase" --search-engine postgres

# Compare both engines
python src/search_cli.py "discovery documents" --search-engine both
```

## Performance Optimization

- Batch processing with configurable batch sizes
- LLM response caching to avoid redundant API calls
- Parallel PDF processing when possible
- Manifest system for resumable processing
- Vision OCR only for PDFs with low text extraction confidence