"""
Vector processing module for legal document search.
Handles text extraction, chunking, embedding generation, and storage.
"""

import uuid
import logging
import time
from typing import List, Dict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pypdf
import ollama
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    VECTOR_STORE_PATH,
    EMBEDDING_MODEL,
    VISION_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

logger = logging.getLogger(__name__)


class TextExtractor:
    """
    Extract text from PDFs using multiple methods.
    Falls back to PyPDF if vision model is not available.
    """
    
    def __init__(self, use_vision: bool = False):
        self.use_vision = use_vision
        if self.use_vision:
            try:
                self.client = ollama.Client()
                # Test if vision model is available
                self.client.show(VISION_MODEL)
                logger.info(f"Using vision model {VISION_MODEL} for text extraction")
            except Exception as e:
                logger.warning(f"Vision model {VISION_MODEL} not available, falling back to PyPDF: {e}")
                self.use_vision = False
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract text from PDF pages"""
        if self.use_vision:
            return self._extract_with_vision(pdf_path)
        else:
            return self._extract_with_pypdf(pdf_path)
    
    def _extract_with_pypdf(self, pdf_path: str) -> List[Dict]:
        """Extract text using PyPDF"""
        extracted_pages = []
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    extracted_pages.append({
                        'page_num': page_num,
                        'content': {'raw_text': text},
                        'extraction_method': 'pypdf'
                    })
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            raise
        
        return extracted_pages
    
    def _extract_with_vision(self, pdf_path: str) -> List[Dict]:
        """Extract text using vision model (future implementation)"""
        # This would use the vision model when available
        # For now, fall back to PyPDF
        logger.info("Vision extraction not yet implemented, using PyPDF")
        return self._extract_with_pypdf(pdf_path)


class SemanticChunker:
    """
    Intelligent chunking for legal documents.
    Preserves semantic boundaries and legal context.
    """
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Configure text splitter for legal documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n\n",  # Multiple line breaks (section boundaries)
                "\n\n",    # Paragraph boundaries
                ".\n",     # Sentence with line break
                ". ",      # Sentence boundaries
                ";\n",     # Semicolon with line break
                "; ",      # Semicolon boundaries
                ",\n",     # Comma with line break
                ", ",      # Comma boundaries
                "\n",      # Line boundaries
                " ",       # Word boundaries
                ""         # Character boundaries
            ]
        )
    
    def chunk_extracted_text(self, extracted_pages: List[Dict], metadata: Dict) -> List[Dict]:
        """Create semantic chunks from extracted pages"""
        chunks = []
        chunk_index = 0
        
        for page_data in extracted_pages:
            page_num = page_data['page_num']
            content = page_data['content']
            
            # Extract text based on structure
            if 'raw_text' in content:
                page_text = content['raw_text']
                if not page_text or len(page_text.strip()) < 10:
                    continue
                
                # Split text into chunks
                text_chunks = self.text_splitter.split_text(page_text)
                
                for text_chunk in text_chunks:
                    if len(text_chunk.strip()) > 20:  # Skip very short chunks
                        chunk_id = str(uuid.uuid4())
                        chunks.append({
                            'id': chunk_id,
                            'text': text_chunk,
                            'page': page_num,
                            'index': chunk_index,
                            'extraction_method': page_data.get('extraction_method', 'unknown'),
                            **metadata
                        })
                        chunk_index += 1
        
        logger.info(f"Created {len(chunks)} chunks from {len(extracted_pages)} pages")
        return chunks


class QwenEmbedder:
    """
    Generate embeddings using Qwen3-Embedding model via Ollama.
    """
    
    def __init__(self, model: str = EMBEDDING_MODEL):
        self.model = model
        self.client = ollama.Client()
        
        # Get embedding dimension by testing
        try:
            test_response = self.client.embeddings(
                model=self.model,
                prompt="test"
            )
            self.dimension = len(test_response['embedding'])
            logger.info(f"Using {self.model} with {self.dimension} dimensions")
        except Exception as e:
            logger.error(f"Failed to initialize embedder: {e}")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        try:
            response = self.client.embeddings(
                model=self.model,
                prompt=text
            )
            return response['embedding']
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def embed_batch(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """Batch embed multiple texts for efficiency with parallel processing"""
        if not texts:
            return []
        
        embeddings = [None] * len(texts)
        
        # Use thread pool for parallel embedding
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all embedding tasks
            future_to_index = {}
            for i, text in enumerate(texts):
                future = executor.submit(self._embed_with_retry, text)
                future_to_index[future] = i
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    embeddings[index] = future.result()
                    completed += 1
                    
                    # Log progress
                    if completed % 10 == 0 or completed == len(texts):
                        logger.info(f"Embedded {completed}/{len(texts)} texts")
                        
                except Exception as e:
                    logger.error(f"Failed to embed text at index {index}: {e}")
                    embeddings[index] = [0.0] * self.dimension
        
        return embeddings
    
    def _embed_with_retry(self, text: str, max_retries: int = 3) -> List[float]:
        """Embed text with retry logic"""
        for attempt in range(max_retries):
            try:
                return self.embed_text(text)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Embedding attempt {attempt + 1} failed, retrying: {e}")
                time.sleep(0.5 * (attempt + 1))  # Exponential backoff


class ChromaVectorStore:
    """
    ChromaDB vector storage for legal documents.
    Optimized for semantic search with metadata filtering.
    """
    
    def __init__(self, path: str = VECTOR_STORE_PATH):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create or get collection
        collection_name = "legal_documents"
        try:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:construction_ef": 200,
                    "hnsw:M": 16
                }
            )
            logger.info(f"Created new collection: {collection_name}")
        except Exception:
            self.collection = self.client.get_collection(collection_name)
            logger.info(f"Using existing collection: {collection_name}")
    
    def add_chunks(self, chunks: List[Dict], embeddings: List[List[float]]):
        """Store chunks with their embeddings and metadata"""
        if not chunks or not embeddings:
            logger.warning("No chunks or embeddings to add")
            return
        
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) count mismatch")
        
        # Prepare data for ChromaDB
        ids = [chunk['id'] for chunk in chunks]
        documents = [chunk['text'] for chunk in chunks]
        metadatas = []
        
        for chunk in chunks:
            metadata = {
                'source_pdf': str(chunk.get('source_pdf', '')),
                'filename': str(chunk.get('filename', '')),
                'page': int(chunk.get('page', 0)),
                'chunk_index': int(chunk.get('index', 0)),
                'bates_start': int(chunk.get('bates_start', 0)),
                'bates_end': int(chunk.get('bates_end', 0)),
                'category': str(chunk.get('category', 'uncategorized')),
                'exhibit_number': int(chunk.get('exhibit_number', 0)),
                'extraction_method': str(chunk.get('extraction_method', 'unknown')),
                'processed_date': str(chunk.get('processed_date', ''))
            }
            # Add summary if available
            if 'summary' in chunk:
                metadata['summary'] = str(chunk['summary'])[:500]  # Limit summary length
            
            metadatas.append(metadata)
        
        # Add to collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"Added {len(chunks)} chunks to vector store")
        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {e}")
            raise
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the vector store"""
        try:
            count = self.collection.count()
            return {
                'total_chunks': count,
                'collection_name': self.collection.name,
                'path': str(self.path)
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'error': str(e)}


class VectorProcessor:
    """
    Main class orchestrating the vector processing pipeline.
    """
    
    def __init__(self, use_vision: bool = False):
        # Validate models and disk space before initialization
        self._validate_setup()
        
        # Initialize components
        self.text_extractor = TextExtractor(use_vision=use_vision)
        self.chunker = SemanticChunker()
        self.embedder = QwenEmbedder()
        self.vector_store = ChromaVectorStore()
    
    def _validate_setup(self):
        """Validate models and disk space before processing."""
        # Check embedding model availability
        try:
            client = ollama.Client()
            client.show(EMBEDDING_MODEL)
            logger.info(f"Embedding model {EMBEDDING_MODEL} validated")
        except Exception as e:
            raise ValueError(f"Required embedding model {EMBEDDING_MODEL} not available: {e}")
        
        # Check disk space
        vector_path = Path(VECTOR_STORE_PATH)
        vector_path.parent.mkdir(parents=True, exist_ok=True)
        
        import shutil
        stat = shutil.disk_usage(vector_path.parent)
        free_gb = stat.free / (1024 ** 3)
        
        if free_gb < 1.0:  # Require at least 1GB free
            raise ValueError(f"Insufficient disk space: {free_gb:.2f}GB free (minimum 1GB required)")
        
        logger.info(f"Disk space check passed: {free_gb:.2f}GB available")
    
    def process_document(self, pdf_path: str, metadata: Dict) -> tuple[List[str], str, List[str]]:
        """
        Process a PDF document through the full vector pipeline.
        Returns tuple of (chunk_ids, full_text, page_texts).
        """
        start_time = time.time()
        doc_name = Path(pdf_path).name
        
        try:
            logger.info(f"[{doc_name}] Starting vector processing")
            
            # Step 1: Extract text from PDF with retry
            logger.info(f"[{doc_name}] Extracting text...")
            extracted_pages = None
            for attempt in range(3):
                try:
                    extracted_pages = self.text_extractor.extract_text_from_pdf(pdf_path)
                    break
                except Exception as e:
                    logger.warning(f"[{doc_name}] Text extraction attempt {attempt + 1} failed: {e}")
                    if attempt == 2:
                        logger.error(f"[{doc_name}] All extraction attempts failed")
                        return [], "", []
                    time.sleep(1)
            
            if not extracted_pages:
                logger.warning(f"[{doc_name}] No text extracted")
                return [], "", []
            
            logger.info(f"[{doc_name}] Extracted text from {len(extracted_pages)} pages")
            
            # Prepare full text and page texts for return
            page_texts = []
            full_text_parts = []
            
            for page_data in extracted_pages:
                if 'raw_text' in page_data['content']:
                    page_text = page_data['content']['raw_text']
                    page_texts.append(page_text)
                    if page_text.strip():
                        full_text_parts.append(page_text)
            
            full_text = "\n\n".join(full_text_parts)
            
            # Add source_pdf to metadata
            metadata['source_pdf'] = pdf_path
            
            # Step 2: Create semantic chunks
            logger.info(f"[{doc_name}] Creating chunks...")
            chunks = self.chunker.chunk_extracted_text(extracted_pages, metadata)
            if not chunks:
                logger.warning(f"[{doc_name}] No chunks created")
                return [], full_text, page_texts
            
            logger.info(f"[{doc_name}] Created {len(chunks)} chunks")
            
            # Step 3: Generate embeddings with improved batching
            logger.info(f"[{doc_name}] Generating embeddings...")
            chunk_texts = [chunk['text'] for chunk in chunks]
            embeddings = self.embedder.embed_batch(chunk_texts)
            
            # Step 4: Store in ChromaDB with retry
            logger.info(f"[{doc_name}] Storing in vector database...")
            for attempt in range(3):
                try:
                    self.vector_store.add_chunks(chunks, embeddings)
                    break
                except Exception as e:
                    logger.warning(f"[{doc_name}] Storage attempt {attempt + 1} failed: {e}")
                    if attempt == 2:
                        logger.error(f"[{doc_name}] All storage attempts failed")
                        return [], full_text, page_texts
                    time.sleep(1)
            
            # Return chunk IDs and text data
            chunk_ids = [chunk['id'] for chunk in chunks]
            elapsed = time.time() - start_time
            logger.info(f"[{doc_name}] Successfully processed {len(chunk_ids)} chunks in {elapsed:.2f}s")
            
            return chunk_ids, full_text, page_texts
            
        except Exception as e:
            logger.error(f"[{doc_name}] Processing failed: {str(e)}")
            # Don't raise - allow pipeline to continue with other documents
            return [], "", []
    
    def get_stats(self) -> Dict:
        """Get processing statistics"""
        return self.vector_store.get_collection_stats()