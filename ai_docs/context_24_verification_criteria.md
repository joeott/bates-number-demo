# Document Processing Pipeline Verification Criteria

## Overview

This document provides comprehensive verification criteria for each step of the legal document processing pipeline. It defines success conditions, expected log output, responsible functions, and validation points for quality assurance and troubleshooting.

## Processing Pipeline Flow

```
Input Validation → Document Discovery → LCEL Chain Execution → Output Generation → Finalization
      ↓                    ↓                    ↓                     ↓              ↓
Configuration      File Enumeration    DocumentOrchestrator    File Generation   CSV Logging
  & Setup           & Validation       LCEL Chain Processing    & Storage        & Statistics
```

## Detailed Verification Criteria

### 1. System Initialization & Configuration

**Responsible Components:**
- `src/main.py`: `main()` function (lines 22-62)
- `src/config.py`: Configuration loading and validation

**Success Criteria:**
- All environment variables loaded correctly
- LLM provider configuration validated
- LangSmith tracing properly initialized (if enabled)
- Output directories created successfully

**Expected Log Output:**
```
INFO: LangSmith tracing enabled for project: bates_number_demo
INFO: LangSmith endpoint: https://api.smith.langchain.com
INFO: Using LLM provider: ollama
DEBUG: LangSmith tracing disabled (if tracing disabled)
```

**Validation Points:**
- `config.LLM_PROVIDER` in ["openai", "ollama"]
- `config.LANGSMITH_TRACING` boolean properly set
- Required API keys present if providers enabled
- Output directory path writable

**Failure Indicators:**
```
WARNING: OPENAI_API_KEY is not set in the environment or .env file.
WARNING: LANGSMITH_TRACING is enabled but LANGSMITH_API_KEY is not set.
ERROR: Output directory not writable: {path}
```

### 2. Document Discovery & Enumeration

**Responsible Components:**
- `src/main.py`: Document discovery logic (lines 120-147)
- `src/utils.py`: File validation utilities

**Success Criteria:**
- All PDF files in input directory discovered
- File accessibility verified
- Document count logged
- File paths correctly resolved

**Expected Log Output:**
```
INFO: Found {n} PDF documents in {input_dir}
INFO: Processing {n} documents total
INFO: Processing documents in batches of {batch_size} (if batching enabled)
INFO: Starting document processing...
```

**Validation Points:**
- Only PDF files included in processing list
- All discovered files exist and are readable
- File paths are absolute and valid
- No empty file list (unless input directory empty)

**Failure Indicators:**
```
ERROR: No PDF documents found in {input_dir}
ERROR: Input directory does not exist: {input_dir}
WARNING: File not accessible: {file_path}
```

### 3. DocumentOrchestrator Initialization

**Responsible Components:**
- `src/document_orchestrator.py`: `__init__()` method (lines 95-130)
- Component initialization: LLM, PDF, Vector, PostgreSQL

**Success Criteria:**
- All required components initialized successfully
- LCEL chain built without errors
- Component availability properly detected
- Chain structure validated

**Expected Log Output:**
```
INFO: DocumentOrchestrator initialized successfully
INFO: LLM categorizer ready
INFO: PDF processor initialized
INFO: Vector processor initialized (if enabled)
INFO: PostgreSQL storage initialized (if enabled)
```

**Validation Points:**
- `self.llm_categorizer` not None
- `self.pdf_processor` not None
- `self.processing_chain` properly constructed
- Conditional components (vector/postgres) initialized based on config

**Failure Indicators:**
```
ERROR: Failed to initialize LLM categorizer: {error}
ERROR: Failed to initialize PDF processor: {error}
WARNING: Vector processing disabled: {reason}
WARNING: PostgreSQL storage disabled: {reason}
```

### 4. LCEL Chain Execution - Document Validation

**Responsible Components:**
- `src/document_orchestrator.py`: `_validate_document()` (lines 175-189)

**Success Criteria:**
- File existence confirmed
- PDF format validated
- File accessibility verified
- Document input structure created

**Expected Log Output:**
```
DEBUG: Validating document: {filename}
DEBUG: Document validation successful: {filename}
```

**Validation Points:**
- `doc_input.file_path.exists()` returns True
- File is readable
- DocumentInput model created successfully
- File path is absolute

**Failure Indicators:**
```
ERROR: File not found: {file_path}
ERROR: Unable to read file: {file_path}
ValueError: File not found: {file_path}
```

### 5. LCEL Chain Execution - LLM Metadata Extraction

**Responsible Components:**
- `src/document_orchestrator.py`: `_process_with_llm()` (lines 191-210)
- `src/llm_handler.py`: `process_document_parallel()` (lines 262-283)

**Success Criteria:**
- Document category determined correctly
- Summary generated within character limits
- Descriptive name created
- All LLM calls completed successfully

**Expected Log Output:**
```
DEBUG: Processing document with LLM: {filename}
INFO: Document categorized as: {category}
DEBUG: Generated summary: {summary}
DEBUG: Generated descriptive name: {descriptive_name}
```

**Validation Points:**
- Category in valid list: ["Pleading", "Medical Record", "Bill", "Correspondence", "Photo", "Video", "Documentary Evidence", "Uncategorized"]
- Summary length < 100 characters
- Descriptive name length reasonable (3-50 characters)
- No empty responses from LLM

**Failure Indicators:**
```
ERROR: LLM categorization failed: {error}
ERROR: LLM summarization failed: {error}
ERROR: LLM filename generation failed: {error}
WARNING: LLM returned empty response for {task}
```

### 6. LCEL Chain Execution - Bates Numbering

**Responsible Components:**
- `src/document_orchestrator.py`: `_apply_bates_numbering()` (lines 212-242)
- `src/pdf_processor.py`: `bates_stamp_pdf()` (lines 60-98)

**Success Criteria:**
- PDF successfully opened and processed
- Bates numbers applied to all pages
- Output file created in correct location
- Bates range properly calculated and formatted

**Expected Log Output:**
```
DEBUG: Applying Bates numbering: {filename}
INFO: Bates numbering applied: {bates_range}
DEBUG: Bates numbered file created: {output_path}
```

**Validation Points:**
- Output file exists and is valid PDF
- Page count matches original document
- Bates numbers sequential and correctly formatted
- Bates range format: "PREFIX{start:06d}-PREFIX{end:06d}"

**Failure Indicators:**
```
ERROR: Failed to apply Bates numbering: {error}
ERROR: PDF processing failed: {error}
ERROR: Output file creation failed: {path}
WARNING: Page count mismatch in Bates numbering
```

### 7. LCEL Chain Execution - Exhibit Marking

**Responsible Components:**
- `src/document_orchestrator.py`: `_apply_exhibit_marking()` (lines 244-282)
- `src/pdf_processor.py`: `exhibit_mark_pdf()` (lines 101-125)

**Success Criteria:**
- Category-specific folder created
- Exhibit number applied to document
- File copied to correct exhibit folder
- Exhibit filename properly formatted

**Expected Log Output:**
```
DEBUG: Applying exhibit marking: {filename}
INFO: Exhibit marked as: Exhibit {number}
DEBUG: File copied to: {exhibit_path}
INFO: Category folder: {category}
```

**Validation Points:**
- Exhibit folder exists: `{output_dir}/exhibits/{category_folder}/`
- Exhibit file exists in correct location
- Exhibit number is sequential integer
- File copy successful (same file size)

**Failure Indicators:**
```
ERROR: Failed to create exhibit folder: {path}
ERROR: Exhibit marking failed: {error}
ERROR: File copy failed: {source} -> {destination}
WARNING: Exhibit folder already exists: {path}
```

### 8. LCEL Chain Execution - Vector Processing (Conditional)

**Responsible Components:**
- `src/document_orchestrator.py`: `_process_vectors()` (lines 284-309)
- `src/vector_processor.py`: `process_document()` (lines 125-195)

**Success Criteria:**
- Text extracted from PDF successfully
- Document chunked appropriately
- Embeddings generated for all chunks
- Chunks stored in vector database

**Expected Log Output:**
```
DEBUG: Processing vectors for: {filename}
INFO: Extracted {page_count} pages of text
INFO: Created {chunk_count} text chunks
INFO: Generated embeddings for {chunk_count} chunks
INFO: Stored {chunk_count} chunks in vector store
```

**Validation Points:**
- `chunk_count > 0` (document not empty)
- All chunks have embeddings
- Vector store size increased by chunk count
- No embedding generation failures

**Failure Indicators:**
```
ERROR: Text extraction failed: {error}
ERROR: Chunking failed: {error}
ERROR: Embedding generation failed: {error}
ERROR: Vector store insertion failed: {error}
WARNING: No text extracted from document: {filename}
```

### 9. LCEL Chain Execution - PostgreSQL Storage (Conditional)

**Responsible Components:**
- `src/document_orchestrator.py`: `_store_in_postgres()` (lines 311-348)
- `src/db_storage.py`: `store_document_text()` (lines 139-217)

**Success Criteria:**
- Database connection established
- Document metadata inserted successfully
- Page-level text stored (if enabled)
- Transaction committed properly

**Expected Log Output:**
```
DEBUG: Storing in PostgreSQL: {filename}
INFO: Document metadata stored in database
INFO: Stored {page_count} pages of text content
DEBUG: Database transaction committed
```

**Validation Points:**
- Database connection active
- Document record exists in database
- Page records created (if page-level storage enabled)
- All required fields populated

**Failure Indicators:**
```
ERROR: Database connection failed: {error}
ERROR: Document insertion failed: {error}
ERROR: Page text storage failed: {error}
ERROR: Database transaction rollback: {error}
```

### 10. LCEL Chain Execution - Result Finalization

**Responsible Components:**
- `src/document_orchestrator.py`: `_finalize_result()` (lines 350-381)

**Success Criteria:**
- ProcessingResult object created successfully
- All required fields populated
- Success status determined correctly
- Error information captured (if applicable)

**Expected Log Output:**
```
DEBUG: Finalizing processing result: {filename}
INFO: Document processing completed successfully: {filename}
INFO: Processing result: {success_status}
```

**Validation Points:**
- `result.success` boolean correctly set
- `result.bates_range` properly formatted
- `result.exhibit_number` is valid integer
- Error messages populated for failures

**Failure Indicators:**
```
ERROR: Result finalization failed: {error}
WARNING: Incomplete processing result for: {filename}
```

### 11. Batch Processing Coordination

**Responsible Components:**
- `src/document_orchestrator.py`: `process_batch()` (lines 496-559)
- `src/main.py`: Batch coordination logic (lines 169-223)

**Success Criteria:**
- All documents in batch processed
- Counter state maintained correctly
- LangSmith tracing configured per document
- Batch statistics calculated accurately

**Expected Log Output:**
```
INFO: Processing batch {batch_num}/{total_batches}
INFO: Processing document {doc_num}/{total_docs}: {filename}
INFO: Batch completed: {success_count} successful, {failed_count} failed
```

**Validation Points:**
- Sequential counter increments maintained
- No counter conflicts between documents
- All documents attempted for processing
- Batch completion statistics accurate

**Failure Indicators:**
```
ERROR: Batch processing interrupted: {error}
WARNING: Counter state inconsistency detected
ERROR: Document processing failed in batch: {filename}
```

### 12. Output Generation & CSV Logging

**Responsible Components:**
- `src/document_orchestrator.py`: `generate_csv_log()` (lines 561-592)
- `src/main.py`: Statistics calculation (lines 225-240)

**Success Criteria:**
- CSV log file created successfully
- All processed documents logged
- Statistics calculated correctly
- Processing time recorded

**Expected Log Output:**
```
INFO: Generated CSV log: {csv_path}
INFO: Processing completed in {duration}
INFO: Statistics: {successful} successful, {failed} failed
INFO: Overall success rate: {percentage}%
```

**Validation Points:**
- CSV file exists and is readable
- Row count matches processed document count
- All required columns present and populated
- No missing data for successful documents

**Failure Indicators:**
```
ERROR: CSV log generation failed: {error}
ERROR: Unable to write CSV file: {path}
WARNING: Missing data in CSV log for document: {filename}
```

## LangSmith Tracing Verification

### Tracing Configuration

**Success Criteria:**
- LangSmith environment variables properly set
- Trace data successfully uploaded
- Run metadata correctly populated
- Chain structure visible in traces

**Expected Trace Elements:**
- Run name: "ProcessDoc-{filename}" or "BatesProcessing-{mode}"
- Tags: ["bates_numbering", "legal_document_processing", "batch_processing"]
- Metadata: document_path, batch_index, counters, timestamps
- Chain steps: DocumentValidation → LLMMetadataExtraction → BatesNumbering → ExhibitMarking → VectorProcessing → PostgreSQLStorage → FinalizeResult

**Verification Points:**
- All chain steps appear in trace
- LLM calls tracked with token usage
- Error traces show failure points
- Performance metrics captured

## Performance Benchmarks

### Expected Processing Times (per document)

| Component | Small Doc (<5 pages) | Medium Doc (5-20 pages) | Large Doc (>20 pages) |
|-----------|---------------------|-------------------------|------------------------|
| LLM Processing | 2-5 seconds | 3-8 seconds | 5-15 seconds |
| PDF Processing | 1-3 seconds | 2-8 seconds | 5-20 seconds |
| Vector Processing | 3-10 seconds | 10-30 seconds | 30-120 seconds |
| PostgreSQL Storage | <1 second | 1-3 seconds | 2-5 seconds |
| **Total Pipeline** | **10-30 seconds** | **20-60 seconds** | **60-300 seconds** |

### Resource Usage Expectations

- **Memory**: 200-800MB per document (depending on size and vector processing)
- **CPU**: Moderate usage during LLM calls and vector processing
- **Disk**: Input size × 2-3 (original + Bates numbered + exhibit copies)
- **Network**: LLM API calls and LangSmith trace uploads

## Error Classification & Troubleshooting

### Critical Errors (Processing Stops)
- Configuration errors (missing API keys, invalid paths)
- Component initialization failures
- LCEL chain construction errors

### Document-Level Errors (Skip Document, Continue Batch)
- File access errors (permissions, corruption)
- LLM processing failures
- PDF processing errors

### Warning Conditions (Log but Continue)
- Optional component unavailability (vector/postgres)
- Performance degradation
- Partial feature failures

## Testing Verification Commands

```bash
# 1. Configuration Check
python -c "from src import config; print(f'LLM: {config.LLM_PROVIDER}'); print(f'Tracing: {config.LANGSMITH_TRACING}')"

# 2. Component Initialization Test
python -c "from src.document_orchestrator import DocumentOrchestrator; from pathlib import Path; do = DocumentOrchestrator(Path('output'), 'TEST', 'Exhibit ')"

# 3. Single Document Test
python src/main.py --input_dir test_input --output_dir test_output

# 4. Batch Processing Test
python src/main.py --input_dir input_documents --output_dir output --batch_size 3

# 5. Integration Test
python -m pytest tests/integration/ -v

# 6. Vector Search Test (if enabled)
python src/search_cli.py "test query" --search-engine vector

# 7. PostgreSQL Test (if enabled)
python src/search_cli.py "test query" --search-engine postgres

# 8. Hybrid Search Test
python src/search_cli.py "test query" --search-engine hybrid
```

## Success Metrics Summary

### Pipeline Success Indicators
- ✅ All documents processed without critical errors
- ✅ Bates numbers applied sequentially without gaps
- ✅ Exhibit numbering consistent and sequential
- ✅ CSV log generated with complete data
- ✅ LangSmith traces show complete chain execution
- ✅ Output files properly organized by category
- ✅ Processing time within expected benchmarks

### Quality Assurance Checklist
- [ ] Configuration validated and logged
- [ ] All input documents discovered
- [ ] LLM categorization accuracy verified
- [ ] Bates numbering sequential and correct
- [ ] Exhibit organization by category accurate
- [ ] Vector search functional (if enabled)
- [ ] PostgreSQL storage complete (if enabled)
- [ ] CSV log data integrity confirmed
- [ ] LangSmith traces complete and accurate
- [ ] Processing statistics match expectations

This verification criteria document enables comprehensive monitoring and validation of the legal document processing pipeline, ensuring reliable operation and quality output for production use.