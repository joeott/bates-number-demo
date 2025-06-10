"""
Vector processing module using LangChain for document processing and embeddings.
Refactored to use LangChain components for better standardization and maintainability.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# LangChain imports
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders.base import BaseLoader

# Legacy imports for backward compatibility
import pypdf
import chromadb
from chromadb.config import Settings

from .config import (
    VECTOR_STORE_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MAX_CHUNKS_PER_DOCUMENT,
    EMBEDDING_BATCH_SIZE,
    ENABLE_VISION_OCR,
    VISION_OCR_MODEL
)
from .embeddings_handler import get_embeddings
from .vision_ocr import get_vision_ocr

logger = logging.getLogger(__name__)


class PDFToLangChainLoader(BaseLoader):
    """
    Custom loader that converts PDF to LangChain Documents.
    Maintains compatibility with existing text extraction logic.
    """
    
    def __init__(self, file_path: str, enable_vision_ocr: bool = False, vision_model: str = None):
        self.file_path = file_path
        self.enable_vision_ocr = enable_vision_ocr
        self.vision_model = vision_model
        self.vision_ocr = get_vision_ocr() if enable_vision_ocr else None
    
    def load(self) -> List[Document]:
        """Load PDF and return list of LangChain Documents (one per page)."""
        documents = []
        
        try:
            with open(self.file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                for page_num in range(total_pages):
                    page = pdf_reader.pages[page_num]
                    
                    # Extract text using PyPDF
                    text = page.extract_text()
                    
                    # If text is too short and vision OCR is enabled, try vision model
                    if self.enable_vision_ocr and self.vision_ocr and self.vision_ocr.enabled and len(text.strip()) < 100:
                        logger.info(f"Page {page_num + 1} has minimal text ({len(text.strip())} chars), attempting vision OCR...")
                        
                        # Use vision OCR to extract text
                        vision_text = self.vision_ocr.extract_text_from_pdf_page(
                            Path(self.file_path), 
                            page_num + 1  # Vision OCR uses 1-based page numbers
                        )
                        
                        # Use vision text if it's significantly better
                        if vision_text and len(vision_text.strip()) > len(text.strip()):
                            logger.info(f"Vision OCR extracted {len(vision_text)} characters (vs {len(text.strip())} from PyPDF)")
                            text = vision_text
                        else:
                            logger.info(f"Vision OCR did not improve extraction ({len(vision_text)} chars)")
                    
                    # Create LangChain Document
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": self.file_path,
                            "page": page_num + 1,
                            "total_pages": total_pages
                        }
                    )
                    documents.append(doc)
                    
        except Exception as e:
            logger.error(f"Error loading PDF {self.file_path}: {e}")
            raise
            
        return documents


class VectorProcessor:
    """
    Processes documents into vector embeddings using LangChain components.
    """
    
    def __init__(self, vector_store_path: str = VECTOR_STORE_PATH):
        """Initialize the vector processor with LangChain components."""
        self.vector_store_path = Path(vector_store_path)
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings using unified handler
        self.embeddings = get_embeddings()
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            keep_separator=True
        )
        
        # Initialize or load existing Chroma vector store
        self.vector_store = Chroma(
            collection_name="legal_documents",
            embedding_function=self.embeddings,
            persist_directory=str(self.vector_store_path)
        )
        
        logger.info(f"VectorProcessor initialized with LangChain components")
        logger.info(f"Vector store path: {self.vector_store_path}")
        logger.info(f"Embedding model: {EMBEDDING_MODEL}")
    
    def process_document(
        self,
        pdf_path: Path,
        exhibit_number: int,
        category: str,
        bates_start: str,
        bates_end: str,
        collection_name: str = "legal_documents"
    ) -> Tuple[List[str], str, List[str]]:
        """
        Process a PDF document into vector embeddings using LangChain pipeline.
        
        Returns:
            Tuple of (chunk_ids, full_text, page_texts)
        """
        start_time = time.time()
        logger.info(f"Processing document: {pdf_path.name}")
        
        try:
            # Step 1: Load PDF into LangChain Documents
            loader = PDFToLangChainLoader(
                str(pdf_path),
                enable_vision_ocr=ENABLE_VISION_OCR,
                vision_model=VISION_OCR_MODEL
            )
            pages = loader.load()
            
            # Extract full text and page texts for PostgreSQL storage
            full_text = "\n\n".join([doc.page_content for doc in pages])
            page_texts = [doc.page_content for doc in pages]
            
            # Add document-level metadata to all pages
            for doc in pages:
                doc.metadata.update({
                    "exhibit_number": exhibit_number,
                    "category": category,
                    "bates_start": bates_start,
                    "bates_end": bates_end,
                    "filename": pdf_path.name,
                    "collection": collection_name
                })
            
            # Step 2: Split documents into chunks
            chunks = self.text_splitter.split_documents(pages)
            
            # Limit chunks if necessary
            if len(chunks) > MAX_CHUNKS_PER_DOCUMENT:
                logger.warning(f"Document has {len(chunks)} chunks, limiting to {MAX_CHUNKS_PER_DOCUMENT}")
                chunks = chunks[:MAX_CHUNKS_PER_DOCUMENT]
            
            # Add chunk-specific metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["total_chunks"] = len(chunks)
            
            # Step 3: Add chunks to vector store (embeddings happen automatically)
            chunk_ids = self.vector_store.add_documents(chunks)
            
            # Log processing stats
            elapsed_time = time.time() - start_time
            logger.info(f"Processed {pdf_path.name}:")
            logger.info(f"  - Pages: {len(pages)}")
            logger.info(f"  - Chunks: {len(chunks)}")
            logger.info(f"  - Time: {elapsed_time:.2f}s")
            logger.info(f"  - Chunks/sec: {len(chunks)/elapsed_time:.2f}")
            
            return chunk_ids, full_text, page_texts
            
        except Exception as e:
            logger.error(f"Error processing document {pdf_path}: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        try:
            # Get collection stats from Chroma
            collection = self.vector_store._collection
            count = collection.count()
            
            # Get unique categories and exhibits
            results = collection.get(include=["metadatas"])
            
            categories = set()
            exhibits = set()
            
            if results and results["metadatas"]:
                for metadata in results["metadatas"]:
                    if metadata.get("category"):
                        categories.add(metadata["category"])
                    if metadata.get("exhibit_number"):
                        exhibits.add(metadata["exhibit_number"])
            
            return {
                "total_chunks": count,
                "categories": sorted(list(categories)),
                "num_categories": len(categories),
                "num_exhibits": len(exhibits)
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                "total_chunks": 0,
                "categories": [],
                "num_categories": 0,
                "num_exhibits": 0
            }


# Backward compatibility function
def process_document(
    pdf_path: Path,
    vector_store,  # This parameter is kept for compatibility but not used
    exhibit_number: int,
    category: str,
    bates_start: str,
    bates_end: str,
    collection_name: str = "legal_documents"
) -> Tuple[List[str], str, List[str]]:
    """
    Backward compatibility wrapper for the process_document function.
    Creates a VectorProcessor instance and calls its process_document method.
    """
    processor = VectorProcessor()
    return processor.process_document(
        pdf_path, exhibit_number, category, bates_start, bates_end, collection_name
    )