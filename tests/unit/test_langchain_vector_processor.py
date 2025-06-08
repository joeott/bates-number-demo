"""
Unit tests for the LangChain-based vector processor components.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile

from langchain.docstore.document import Document

# Import components to test
from src.vector_processor import (
    PDFToLangChainLoader,
    VectorProcessor,
    process_document
)


class TestPDFToLangChainLoader:
    """Test the PDF to LangChain Document loader."""
    
    def test_loader_initialization(self):
        """Test loader initializes correctly."""
        loader = PDFToLangChainLoader("/path/to/test.pdf")
        assert loader.file_path == "/path/to/test.pdf"
        assert loader.enable_vision_ocr is False
        assert loader.vision_model is None
        assert loader.ollama_client is None
    
    def test_loader_with_vision_ocr(self):
        """Test loader initialization with vision OCR enabled."""
        with patch('src.vector_processor.ollama.Client') as mock_client:
            loader = PDFToLangChainLoader(
                "/path/to/test.pdf",
                enable_vision_ocr=True,
                vision_model="llava"
            )
            assert loader.enable_vision_ocr is True
            assert loader.vision_model == "llava"
            mock_client.assert_called_once()
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.vector_processor.pypdf.PdfReader')
    def test_load_pdf_documents(self, mock_pdf_reader, mock_file):
        """Test loading PDF and converting to LangChain documents."""
        # Mock PDF reader
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "Page 1 content"
        mock_page2 = Mock()
        mock_page2.extract_text.return_value = "Page 2 content"
        
        mock_reader_instance = Mock()
        mock_reader_instance.pages = [mock_page1, mock_page2]
        mock_pdf_reader.return_value = mock_reader_instance
        
        # Test loading
        loader = PDFToLangChainLoader("/path/to/test.pdf")
        documents = loader.load()
        
        # Verify results
        assert len(documents) == 2
        assert isinstance(documents[0], Document)
        assert isinstance(documents[1], Document)
        
        # Check document content
        assert documents[0].page_content == "Page 1 content"
        assert documents[0].metadata["page"] == 1
        assert documents[0].metadata["total_pages"] == 2
        assert documents[0].metadata["source"] == "/path/to/test.pdf"
        
        assert documents[1].page_content == "Page 2 content"
        assert documents[1].metadata["page"] == 2


class TestVectorProcessor:
    """Test the VectorProcessor class."""
    
    @patch('src.vector_processor.OllamaEmbeddings')
    @patch('src.vector_processor.Chroma')
    def test_processor_initialization(self, mock_chroma, mock_embeddings):
        """Test VectorProcessor initializes with LangChain components."""
        processor = VectorProcessor("/test/vector/store")
        
        # Verify embeddings initialized
        mock_embeddings.assert_called_once()
        
        # Verify Chroma initialized
        mock_chroma.assert_called_once()
        call_args = mock_chroma.call_args
        assert call_args.kwargs["collection_name"] == "legal_documents"
        assert call_args.kwargs["persist_directory"] == "/test/vector/store"
    
    @patch('src.vector_processor.PDFToLangChainLoader')
    @patch('src.vector_processor.OllamaEmbeddings')
    @patch('src.vector_processor.Chroma')
    def test_process_document_flow(self, mock_chroma, mock_embeddings, mock_loader):
        """Test the complete document processing flow."""
        # Mock loader
        mock_doc1 = Document(
            page_content="Test page 1",
            metadata={"page": 1}
        )
        mock_doc2 = Document(
            page_content="Test page 2",
            metadata={"page": 2}
        )
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = [mock_doc1, mock_doc2]
        mock_loader.return_value = mock_loader_instance
        
        # Mock vector store
        mock_vector_store = Mock()
        mock_vector_store.add_documents.return_value = ["id1", "id2", "id3"]
        mock_chroma.return_value = mock_vector_store
        
        # Create processor
        processor = VectorProcessor()
        processor.text_splitter = Mock()
        processor.text_splitter.split_documents.return_value = [
            Document(page_content="Chunk 1", metadata={}),
            Document(page_content="Chunk 2", metadata={}),
            Document(page_content="Chunk 3", metadata={})
        ]
        
        # Process document
        chunk_ids, full_text, page_texts = processor.process_document(
            Path("/test/doc.pdf"),
            exhibit_number=1,
            category="Test Category",
            bates_start="000001",
            bates_end="000002"
        )
        
        # Verify results
        assert chunk_ids == ["id1", "id2", "id3"]
        assert full_text == "Test page 1\n\nTest page 2"
        assert page_texts == ["Test page 1", "Test page 2"]
        
        # Verify metadata was added
        mock_vector_store.add_documents.assert_called_once()
        added_docs = mock_vector_store.add_documents.call_args[0][0]
        assert len(added_docs) == 3
        for doc in added_docs:
            assert doc.metadata["exhibit_number"] == 1
            assert doc.metadata["category"] == "Test Category"
            assert doc.metadata["bates_start"] == "000001"
            assert doc.metadata["bates_end"] == "000002"
    
    @patch('src.vector_processor.OllamaEmbeddings')
    @patch('src.vector_processor.Chroma')
    def test_get_stats(self, mock_chroma, mock_embeddings):
        """Test getting statistics from the vector store."""
        # Mock collection
        mock_collection = Mock()
        mock_collection.count.return_value = 100
        mock_collection.get.return_value = {
            "metadatas": [
                {"category": "Pleading", "exhibit_number": 1},
                {"category": "Bill", "exhibit_number": 2},
                {"category": "Pleading", "exhibit_number": 3},
            ]
        }
        
        mock_vector_store = Mock()
        mock_vector_store._collection = mock_collection
        mock_chroma.return_value = mock_vector_store
        
        # Get stats
        processor = VectorProcessor()
        stats = processor.get_stats()
        
        # Verify results
        assert stats["total_chunks"] == 100
        assert stats["num_categories"] == 2
        assert stats["num_exhibits"] == 3
        assert set(stats["categories"]) == {"Bill", "Pleading"}


class TestBackwardCompatibility:
    """Test backward compatibility wrapper."""
    
    @patch('src.vector_processor.VectorProcessor')
    def test_process_document_wrapper(self, mock_processor_class):
        """Test the backward compatibility wrapper function."""
        # Mock processor instance
        mock_processor = Mock()
        mock_processor.process_document.return_value = (["id1"], "full text", ["page1"])
        mock_processor_class.return_value = mock_processor
        
        # Call wrapper function
        result = process_document(
            Path("/test/doc.pdf"),
            None,  # vector_store parameter (ignored)
            exhibit_number=1,
            category="Test",
            bates_start="000001",
            bates_end="000001"
        )
        
        # Verify
        assert result == (["id1"], "full text", ["page1"])
        mock_processor.process_document.assert_called_once_with(
            Path("/test/doc.pdf"), 1, "Test", "000001", "000001", "legal_documents"
        )


def mock_open(mock=None, read_data=''):
    """Helper to create a mock for open()."""
    import io
    from unittest.mock import MagicMock
    
    mock = MagicMock(spec=io.IOBase)
    handle = MagicMock(spec=io.IOBase)
    handle.__enter__.return_value = handle
    handle.read.return_value = read_data
    mock.return_value = handle
    return mock