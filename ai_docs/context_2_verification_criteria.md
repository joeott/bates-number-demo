# Document Processing Pipeline Verification Criteria

## Overview
This document provides comprehensive verification criteria for the document processing pipeline, including specific function references, expected behaviors, and validation points.

## Pipeline Architecture

### Main Entry Point
- **File**: `src/main.py`
- **Function**: `main()` (Line 22-288)
- **Purpose**: Orchestrates the entire document processing workflow

### Core Components
1. **LLM Handler**: Document categorization and metadata extraction
2. **PDF Processor**: Bates numbering and exhibit marking
3. **Vector Processor**: Text extraction and embedding creation
4. **Database Storage**: PostgreSQL text storage
5. **Document Orchestrator**: LCEL chain coordination

## Detailed Pipeline Flow

### 1. Initialization Phase

#### Component Initialization (main.py)
- **Lines 94-97**: Initialize LLMCategorizer
  - **Success**: Logger shows "Initializing LLM with provider: {provider}"
  - **Failure**: ValueError with message "Failed to initialize LLM Categorizer"
  - **Validation**: Check LLM provider configuration (openai/ollama)

- **Lines 102-110**: Initialize VectorProcessor (optional)
  - **Success**: "Vector search enabled - documents will be indexed for semantic search"
  - **Failure**: Warning "Failed to initialize vector processor"
  - **Non-critical**: Pipeline continues without vector search

- **Lines 111-122**: Initialize PostgresStorage (optional)
  - **Success**: "PostgreSQL storage enabled - document text will be stored in database"
  - **Failure**: Warning "Failed to initialize PostgreSQL storage"
  - **Non-critical**: Pipeline continues without database storage

- **Lines 124-133**: Initialize DocumentOrchestrator
  - **Required**: All core components must be passed
  - **Success**: No specific log, but directories created

### 2. Document Discovery
- **Lines 136-141**: Find PDF files
  - **Success**: "Found {n} PDF documents in '{dir}'"
  - **Failure**: "No PDF documents found" (exits with code 0)
  - **Validation**: Case-insensitive PDF detection (*.pdf, *.PDF)

### 3. Processing Execution

#### Batch vs Sequential Processing (main.py)
- **Lines 170-224**: Processing logic
  - **Batch Mode**: When `batch_size > 0` and `total_documents > batch_size`
    - Log: "Processing documents in batches of {size}"
    - Log: "Processing batch {n}/{total}"
  - **Sequential Mode**: Default processing
    - Uses single batch with all documents

### 4. Document Orchestrator Pipeline (document_orchestrator.py)

#### LCEL Chain Construction (Lines 132-173)
The processing chain consists of:

1. **Validation Chain** (`_validate_document`, Line 175-189)
   - **Input**: DocumentInput with file_path, bates_counter, exhibit_counter
   - **Success**: Returns dict with "success": True
   - **Failure**: ValueError if file not found or not PDF
   - **Log**: "Validating document: {filename}"

2. **LLM Chain** (`_process_with_llm`, Line 191-210)
   - **Purpose**: Extract category, summary, and descriptive name
   - **Success**: Returns DocumentMetadata object
   - **Log**: "LLM processing: {filename}"
   - **Validation**: All three fields must be populated

3. **Bates Chain** (`_apply_bates_numbering`, Line 212-242)
   - **Purpose**: Apply Bates stamps to PDF
   - **Success**: Returns BatesResult with start/end numbers
   - **Log**: "Bates numbered: {start}-{end}"
   - **Validation**: Next counter must be > current counter

4. **Exhibit Chain** (`_apply_exhibit_marking`, Line 244-282)
   - **Purpose**: Apply exhibit stamps and organize by category
   - **Success**: Returns ExhibitResult
   - **Log**: "Exhibit marked: {filename}"
   - **Creates**: Category subdirectory in exhibits folder

5. **Vector Branch** (`_process_vectors`, Line 284-309)
   - **Conditional**: Only if vector_processor exists and success=True
   - **Success**: Sets vector_chunks count
   - **Log**: "Created {n} vector chunks"
   - **Non-critical**: Errors logged but processing continues

6. **PostgreSQL Branch** (`_store_in_postgres`, Line 311-348)
   - **Conditional**: Only if postgres_storage exists and success=True
   - **Success**: Sets postgres_stored=True
   - **Log**: "Stored in PostgreSQL with ID: {id}"
   - **Non-critical**: Errors logged but processing continues

7. **Finalize Result** (`_finalize_result`, Line 350-381)
   - **Purpose**: Create ProcessingResult object
   - **Always runs**: Even on failure
   - **Returns**: ProcessingResult with success flag

#### Safe Processing Wrapper (Line 383-398)
- **Function**: `_safe_process`
- **Purpose**: Catch any unhandled exceptions
- **On Error**: Returns ProcessingResult with success=False and error message

### 5. LLM Processing Details (llm_handler.py)

#### LLM Initialization (Lines 84-116)
- **OpenAI Provider** (Lines 119-129):
  - Requires OPENAI_API_KEY
  - Uses ChatOpenAI with temperature=0.2, max_tokens=50
  
- **Ollama Provider** (Lines 130-144):
  - Verifies Ollama connection with client.list()
  - Uses ChatOllama with temperature=0.2, num_predict=50

#### Parallel Processing Chain (Lines 175-179)
- **Function**: `process_document_parallel` (Line 262-283)
- **Executes**: All three operations simultaneously
  - Categorization
  - Summarization 
  - Filename generation
- **Success**: Returns dict with all three results
- **Failure**: Returns defaults (Uncategorized, generic summary, "Document")

### 6. PDF Processing (pdf_processor.py)

#### Bates Stamping (Lines 60-98)
- **Function**: `bates_stamp_pdf`
- **Returns**: Tuple (start_bates, end_bates, next_counter)
- **Success Log**: "Bates stamped '{input}' to '{output}' (Bates: {start}-{end})"
- **Stamp Position**: Lower-right corner (Line 44-45)
- **Font**: Configurable, default from config

#### Exhibit Marking (Lines 101-125)
- **Function**: `exhibit_mark_pdf`
- **Returns**: Boolean success
- **Success Log**: "Exhibit marked '{input}' as '{id}' to '{output}'"
- **Stamp Position**: Above Bates stamp (Line 40)
- **Color**: Red for exhibits (Line 49)

### 7. Vector Processing (vector_processor.py)

#### Document Processing (Lines 125-195)
- **Function**: `process_document`
- **Steps**:
  1. Load PDF with PDFToLangChainLoader (Lines 144-150)
  2. Split into chunks with RecursiveCharacterTextSplitter (Line 168)
  3. Add to vector store (Line 181)
- **Logs**:
  - "Processing document: {name}"
  - "Processed {name}: Pages: {n}, Chunks: {m}, Time: {t}s"
- **Returns**: (chunk_ids, full_text, page_texts)

### 8. Database Storage (db_storage.py)

#### Document Storage (Lines 139-217)
- **Function**: `store_document_text`
- **Database Operations**:
  1. Upsert document record (Lines 168-192)
  2. Store individual pages if provided (Lines 195-214)
- **Success Log**: "Stored document {exhibit_id} with {n} pages"
- **Returns**: Document ID from database

### 9. Final Output Generation

#### CSV Log Generation (Lines 561-584 in document_orchestrator.py)
- **Function**: `generate_csv_log`
- **Creates**: exhibit_log.csv with columns:
  - Exhibit ID
  - Original Filename
  - Final Filename
  - Category
  - Summary
  - Bates Range
  - Status
  - Vector Chunks
  - PostgreSQL Stored
  - Error (if any)

#### Summary Statistics (main.py Lines 236-279)
- **Displays**:
  - Total documents processed
  - Success/failure counts
  - Processing time and average
  - Category breakdown
  - Vector store statistics (if enabled)
  - PostgreSQL statistics (if enabled)
  - Failed document list with errors

## Verification Checkpoints

### Pre-Processing Validation
1. ✓ Input directory exists
2. ✓ PDF files found
3. ✓ LLM provider configured
4. ✓ Output directories created

### Per-Document Validation
1. ✓ File exists and is PDF
2. ✓ LLM returns valid category
3. ✓ Bates numbers applied sequentially
4. ✓ Exhibit number incremented
5. ✓ Output files created in correct directories
6. ✓ Category subdirectory created

### Post-Processing Validation
1. ✓ All documents have results (success or failure)
2. ✓ CSV log contains all documents
3. ✓ Statistics match processed count
4. ✓ Exit code reflects success (0) or failure (1)

### Error Handling Validation
1. ✓ Component initialization failures are non-fatal (except LLM)
2. ✓ Individual document failures don't stop pipeline
3. ✓ Errors are captured in ProcessingResult
4. ✓ Failed documents listed in final output

## Expected Log Patterns

### Successful Document Processing
```
INFO - Validating document: example.pdf
INFO - LLM processing: example.pdf
INFO - LLM categorized 'example.pdf' as: Medical Record
INFO - LLM summary for 'example.pdf': Medical examination report for patient
INFO - Generated filename for 'example.pdf': Patient_Medical_Examination
INFO - Bates numbered: TEST000001-TEST000005
INFO - Exhibit marked: Exhibit 1 - Patient_Medical_Examination.pdf
INFO - Processing vectors...
INFO - Created 12 vector chunks
INFO - Stored in PostgreSQL with ID: 1
```

### Failed Document Processing
```
ERROR - Processing failed for example.pdf: [error message]
```

### Component Initialization
```
INFO - Initializing LLM with provider: openai
INFO - Vector search enabled - documents will be indexed for semantic search
INFO - PostgreSQL storage enabled - document text will be stored in database
INFO - VectorProcessor initialized with LangChain components
INFO - PostgreSQL connection pool initialized (size: 5)
INFO - PostgreSQL tables created/verified successfully
```

## Performance Metrics

### Expected Processing Times
- LLM Operations: 0.5-2s per document
- PDF Processing: 0.1-0.5s per page
- Vector Embedding: 0.5-1s per document
- Database Storage: 0.1-0.3s per document

### Resource Usage
- Memory: ~100-500MB depending on PDF size
- CPU: Moderate during PDF processing
- Network: API calls for LLM operations
- Disk I/O: Read source, write outputs

## Troubleshooting Guide

### Common Issues
1. **LLM Initialization Failure**
   - Check API keys in .env
   - Verify Ollama is running (if using local)
   - Check network connectivity

2. **Vector Processing Failure**
   - Verify Ollama embedding model is available
   - Check disk space for vector store

3. **PostgreSQL Connection Failure**
   - Verify connection string
   - Check database server is running
   - Verify credentials and permissions

4. **PDF Processing Errors**
   - Check PDF is not corrupted
   - Verify sufficient disk space
   - Check file permissions

### Debug Mode
Enable debug logging by setting log level in config:
```python
logging.getLogger().setLevel(logging.DEBUG)
```

This will show:
- Detailed LCEL chain execution
- LangSmith tracing (if enabled)
- Database query details
- Vector processing statistics