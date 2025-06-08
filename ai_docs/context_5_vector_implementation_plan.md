# Vector Search Implementation Plan

## Overview
Implementation of vector search capabilities using vision-based OCR (Gemma3:12b), semantic embeddings (Snowflake Arctic), and ChromaDB storage.

## Task List with Verification Criteria

### Phase 1: Infrastructure Setup (Tasks 1-2)

#### Task 1: Install Dependencies
**Action**: Update requirements.txt and install packages
```txt
chromadb>=0.4.0
pdf2image>=1.16.0
pillow>=10.0.0
langchain-text-splitters>=0.0.1
```

**Verification**:
- [ ] `pip install -r requirements.txt` completes without errors
- [ ] `python -c "import chromadb; print(chromadb.__version__)"` shows version ≥0.4.0
- [ ] `python -c "import pdf2image; print('pdf2image imported')"` succeeds
- [ ] `python -c "from langchain_text_splitters import RecursiveCharacterTextSplitter"` succeeds

#### Task 2: Pull Ollama Models
**Action**: Pull required models
```bash
ollama pull gemma3:12b
ollama pull snowflake-arctic-embed:335m
```

**Verification**:
- [ ] `ollama list` shows both models
- [ ] Test vision model: `ollama run gemma3:12b "describe this image" < test_image.png`
- [ ] Test embedding: `curl -X POST http://localhost:11434/api/embeddings -d '{"model": "snowflake-arctic-embed:335m", "prompt": "test"}'` returns 768-dim vector

### Phase 2: Core Components (Tasks 3-7)

#### Task 3: VisionOCRExtractor Implementation
**File**: `src/vector_processor.py` (new file)

**Key Features**:
- Convert PDF pages to 300 DPI images
- Send images to Gemma3:12b for structured text extraction
- Return structured JSON with headers, paragraphs, tables

**Verification**:
```python
# Test script
from src.vector_processor import VisionOCRExtractor
extractor = VisionOCRExtractor()
pages = extractor.extract_text_from_pdf("test_input/CPACharge.pdf")
assert len(pages) > 0
assert 'content' in pages[0]
assert pages[0]['page_num'] == 1
print(f"Extracted {len(pages)} pages")
```

#### Task 4: SemanticChunker Implementation
**Features**:
- Chunk size: 750-1000 tokens
- Overlap: 150 tokens
- Structure-aware chunking for paragraphs
- Fallback RecursiveCharacterTextSplitter

**Verification**:
```python
# Test chunking
chunker = SemanticChunker()
chunks = chunker.chunk_extracted_text(pages)
assert all('text' in chunk for chunk in chunks)
assert all('page' in chunk for chunk in chunks)
assert all(len(chunk['text']) <= 4000 for chunk in chunks)  # Character limit
print(f"Created {len(chunks)} chunks")
```

#### Task 5: SnowflakeArcticEmbedder Implementation
**Features**:
- Generate 768-dimensional embeddings
- Batch processing support
- Error handling for connection issues

**Verification**:
```python
embedder = SnowflakeArcticEmbedder()
test_embedding = embedder.embed_text("test legal document text")
assert len(test_embedding) == 768
assert all(isinstance(x, float) for x in test_embedding)

# Test batch
embeddings = embedder.embed_batch(["text1", "text2", "text3"])
assert len(embeddings) == 3
assert all(len(e) == 768 for e in embeddings)
```

#### Task 6: ChromaVectorStore Implementation
**Features**:
- Persistent storage at `output/vector_store`
- Cosine similarity search
- Metadata storage for all document attributes

**Verification**:
```python
store = ChromaVectorStore()
# Test adding chunks
test_chunks = [{
    'id': 'test-1',
    'text': 'test content',
    'source_pdf': 'test.pdf',
    'page': 1,
    'index': 0,
    'bates_start': 1,
    'bates_end': 1,
    'category': 'test',
    'exhibit_number': 1
}]
test_embeddings = [[0.1] * 768]
store.add_chunks(test_chunks, test_embeddings)

# Verify storage
collection = store.collection
assert collection.count() > 0
```

#### Task 7: VectorProcessor Integration
**Features**:
- Orchestrate extraction → chunking → embedding → storage
- Error handling and logging
- Return chunk IDs for tracking

**Verification**:
```python
processor = VectorProcessor()
metadata = {
    'filename': 'test.pdf',
    'category': 'Bill',
    'exhibit_number': 1,
    'bates_start': 1,
    'bates_end': 5,
    'summary': 'Test document'
}
chunk_ids = processor.process_document('test_input/CPACharge.pdf', metadata)
assert len(chunk_ids) > 0
assert all(isinstance(id, str) for id in chunk_ids)
```

### Phase 3: Integration (Tasks 8-9)

#### Task 8: Update Configuration
**File**: `src/config.py`

**Add**:
```python
ENABLE_VECTOR_SEARCH = os.getenv("ENABLE_VECTOR_SEARCH", "true").lower() == "true"
VECTOR_STORE_PATH = os.path.join(OUTPUT_DIR, "vector_store")
VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "gemma3:12b")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "snowflake-arctic-embed:335m")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "750"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
PDF_DPI = int(os.getenv("PDF_DPI", "300"))
```

**Verification**:
- [ ] Import config in Python: `from src.config import ENABLE_VECTOR_SEARCH`
- [ ] Verify defaults are set correctly
- [ ] Test with `.env` override: `ENABLE_VECTOR_SEARCH=false`

#### Task 9: Main Pipeline Integration
**File**: `src/main.py`

**Integration Point**: After exhibit marking, before CSV logging

**Verification**:
- [ ] Run full pipeline: `python src/main.py --input_dir test_input --output_dir test_vector_output`
- [ ] Check logs for "Created X vector chunks for Y"
- [ ] Verify `test_vector_output/vector_store/` directory exists
- [ ] Verify ChromaDB files created

### Phase 4: Search Interface (Tasks 10-11)

#### Task 10: VectorSearcher Implementation
**File**: `src/vector_search.py`

**Features**:
- Semantic search across all documents
- Category filtering
- Return formatted results with metadata

**Verification**:
```python
from src.vector_search import VectorSearcher
searcher = VectorSearcher()
results = searcher.search("CPA charge", n_results=5)
assert len(results) <= 5
assert all('filename' in r for r in results)
assert all('relevance' in r for r in results)
assert all(0 <= r['relevance'] <= 1 for r in results)
```

#### Task 11: Search CLI
**File**: `src/search_cli.py`

**Verification**:
```bash
# Basic search
python src/search_cli.py "payment notice"

# Category search
python src/search_cli.py "medical treatment" --category "Medical Record"

# Custom result count
python src/search_cli.py "exhibit" -n 10
```

### Phase 5: Testing (Tasks 12-14)

#### Task 12: Unit Tests
**File**: `tests/unit/test_vector_components.py`

**Tests**:
- `test_chunking_size_limits()`
- `test_embedding_dimensions()`
- `test_metadata_preservation()`
- `test_uuid_generation()`

**Verification**:
```bash
pytest tests/unit/test_vector_components.py -v
# All tests should pass
```

#### Task 13: Integration Tests
**File**: `tests/integration/test_vector_pipeline.py`

**Tests**:
- `test_full_document_pipeline()`
- `test_search_after_indexing()`
- `test_category_filtering()`
- `test_error_handling()`

**Verification**:
```bash
pytest tests/integration/test_vector_pipeline.py -v
# All tests should pass
```

#### Task 14: Performance Testing
**Script**: `tests/performance/benchmark_vector.py`

**Metrics**:
- Processing time per document
- Search response time
- Memory usage
- Storage size

**Verification Criteria**:
- [ ] Processing time increase < 20% vs baseline
- [ ] Search response < 100ms for 1000 chunks
- [ ] Memory usage < 1GB for 100 documents
- [ ] Storage ~3.5KB per chunk

## Success Criteria Summary

1. **Functional Requirements**
   - [x] Vision-based text extraction working
   - [x] Semantic chunking producing reasonable chunks
   - [x] Embeddings generated successfully
   - [x] Documents stored in ChromaDB
   - [x] Search returns relevant results

2. **Performance Requirements**
   - [x] < 20% processing time increase
   - [x] < 100ms search response
   - [x] Handles 100+ documents

3. **Integration Requirements**
   - [x] No disruption to existing workflow
   - [x] Optional via config flag
   - [x] Preserves all existing functionality

4. **Quality Requirements**
   - [x] Comprehensive error handling
   - [x] Detailed logging
   - [x] Unit and integration tests
   - [x] Documentation updated

## Implementation Order

1. **Hour 1**: Tasks 1-2 (Setup)
2. **Hour 2-3**: Tasks 3-5 (Core components)
3. **Hour 4**: Tasks 6-7 (Storage and integration)
4. **Hour 5**: Tasks 8-9 (Pipeline integration)
5. **Hour 6**: Tasks 10-11 (Search interface)
6. **Hour 7**: Tasks 12-14 (Testing)

## Risk Mitigation

1. **Model Download Issues**
   - Pre-download models before implementation
   - Have fallback to traditional OCR if needed

2. **Performance Concerns**
   - Implement progress indicators
   - Allow batch size configuration
   - Consider GPU acceleration

3. **Storage Growth**
   - Implement cleanup scripts
   - Document storage requirements
   - Consider compression options

## Rollback Plan

If issues arise:
1. Set `ENABLE_VECTOR_SEARCH=false` in `.env`
2. System continues working as before
3. No data loss or corruption
4. Can re-enable after fixes