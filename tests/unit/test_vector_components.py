"""
Unit tests for vector search components.
Tests chunking, embeddings, and metadata preservation.
"""

import pytest
import uuid
from unittest.mock import Mock, patch, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.vector_processor import (
    TextExtractor,
    SemanticChunker,
    QwenEmbedder,
    ChromaVectorStore,
    VectorProcessor
)


class TestSemanticChunker:
    """Test the document chunking functionality."""
    
    def test_chunking_size_limits(self):
        """Verify chunk size and overlap settings are respected."""
        chunk_size = 500
        chunk_overlap = 100
        chunker = SemanticChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # Create test text that should result in multiple chunks
        test_text = "This is a test sentence. " * 50  # ~1250 characters
        
        extracted_pages = [{
            'page_num': 1,
            'content': {'raw_text': test_text},
            'extraction_method': 'test'
        }]
        
        metadata = {'test': 'metadata'}
        chunks = chunker.chunk_extracted_text(extracted_pages, metadata)
        
        # Verify we got multiple chunks
        assert len(chunks) > 1, "Should create multiple chunks from long text"
        
        # Verify chunk properties
        for chunk in chunks:
            assert 'id' in chunk
            assert 'text' in chunk
            assert 'page' in chunk
            assert 'index' in chunk
            assert len(chunk['text']) <= chunk_size + 100  # Allow some flexibility
            
        # Verify metadata is preserved
        for chunk in chunks:
            assert chunk.get('test') == 'metadata'
    
    def test_empty_page_handling(self):
        """Test handling of empty or very short pages."""
        chunker = SemanticChunker()
        
        extracted_pages = [
            {
                'page_num': 1,
                'content': {'raw_text': ''},  # Empty page
                'extraction_method': 'test'
            },
            {
                'page_num': 2,
                'content': {'raw_text': '   \n\n   '},  # Whitespace only
                'extraction_method': 'test'
            },
            {
                'page_num': 3,
                'content': {'raw_text': 'Short'},  # Very short text
                'extraction_method': 'test'
            }
        ]
        
        chunks = chunker.chunk_extracted_text(extracted_pages, {})
        
        # Should skip empty pages but might include very short text
        assert len(chunks) <= 1
    
    def test_chunk_indexing(self):
        """Test that chunks are properly indexed."""
        chunker = SemanticChunker(chunk_size=100, chunk_overlap=20)
        
        # Create pages with enough text for multiple chunks
        extracted_pages = [
            {
                'page_num': 1,
                'content': {'raw_text': 'First page. ' * 20},
                'extraction_method': 'test'
            },
            {
                'page_num': 2,
                'content': {'raw_text': 'Second page. ' * 20},
                'extraction_method': 'test'
            }
        ]
        
        chunks = chunker.chunk_extracted_text(extracted_pages, {})
        
        # Verify sequential indexing
        indices = [chunk['index'] for chunk in chunks]
        assert indices == list(range(len(chunks)))
        
        # Verify page numbers are preserved
        page_1_chunks = [c for c in chunks if c['page'] == 1]
        page_2_chunks = [c for c in chunks if c['page'] == 2]
        assert len(page_1_chunks) > 0
        assert len(page_2_chunks) > 0


class TestQwenEmbedder:
    """Test the embedding generation functionality."""
    
    @patch('ollama.Client')
    def test_embedding_dimensions(self, mock_client_class):
        """Verify embeddings have correct dimensions."""
        # Mock the Ollama client
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock embedding response
        test_embedding = [0.1] * 2048  # Simulated embedding
        mock_client.embeddings.return_value = {'embedding': test_embedding}
        
        embedder = QwenEmbedder()
        
        # Test single embedding
        result = embedder.embed_text("test text")
        
        assert len(result) == 2048
        assert all(isinstance(x, float) for x in result)
        assert embedder.dimension == 2048
    
    @patch('ollama.Client')
    def test_batch_embedding(self, mock_client_class):
        """Test batch embedding functionality."""
        # Mock the Ollama client
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock embedding responses
        test_embedding = [0.1] * 2048
        mock_client.embeddings.return_value = {'embedding': test_embedding}
        
        embedder = QwenEmbedder()
        
        # Test batch
        texts = ["text1", "text2", "text3"]
        results = embedder.embed_batch(texts, batch_size=2)
        
        assert len(results) == 3
        assert all(len(emb) == 2048 for emb in results)
        
        # Verify embeddings were requested for each text
        assert mock_client.embeddings.call_count >= 4  # 1 for init + 3 for texts
    
    @patch('ollama.Client')
    def test_embedding_error_handling(self, mock_client_class):
        """Test error handling in embedding generation."""
        # Mock the Ollama client
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # First call succeeds for initialization
        mock_client.embeddings.side_effect = [
            {'embedding': [0.1] * 2048},  # Init call
            Exception("Embedding failed")  # Error on actual use
        ]
        
        embedder = QwenEmbedder()
        
        # Should raise exception
        with pytest.raises(Exception):
            embedder.embed_text("test text")


class TestChromaVectorStore:
    """Test the vector storage functionality."""
    
    @patch('chromadb.PersistentClient')
    def test_metadata_preservation(self, mock_client_class):
        """Verify all metadata is properly stored."""
        # Mock ChromaDB client and collection
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        store = ChromaVectorStore("/tmp/test_store")
        
        # Test chunks with full metadata
        chunks = [{
            'id': str(uuid.uuid4()),
            'text': 'test content',
            'source_pdf': 'test.pdf',
            'filename': 'test_exhibit.pdf',
            'page': 1,
            'index': 0,
            'bates_start': 100,
            'bates_end': 110,
            'category': 'Bill',
            'exhibit_number': 5,
            'extraction_method': 'pypdf',
            'processed_date': '2024-01-01T00:00:00',
            'summary': 'Test summary'
        }]
        
        embeddings = [[0.1] * 768]
        
        store.add_chunks(chunks, embeddings)
        
        # Verify add was called with correct metadata
        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args[1]
        
        assert len(call_args['metadatas']) == 1
        metadata = call_args['metadatas'][0]
        
        # Check all metadata fields are preserved
        assert metadata['source_pdf'] == 'test.pdf'
        assert metadata['filename'] == 'test_exhibit.pdf'
        assert metadata['page'] == 1
        assert metadata['chunk_index'] == 0
        assert metadata['bates_start'] == 100
        assert metadata['bates_end'] == 110
        assert metadata['category'] == 'Bill'
        assert metadata['exhibit_number'] == 5
        assert metadata['extraction_method'] == 'pypdf'
        assert 'summary' in metadata
    
    @patch('chromadb.PersistentClient')
    def test_uuid_generation(self, mock_client_class):
        """Test that valid UUIDs are generated for chunks."""
        # Mock ChromaDB
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        store = ChromaVectorStore("/tmp/test_store")
        
        # Create chunks with UUIDs
        chunks = []
        for i in range(5):
            chunks.append({
                'id': str(uuid.uuid4()),
                'text': f'chunk {i}',
                'page': 1,
                'index': i
            })
        
        embeddings = [[0.1] * 768] * 5
        
        store.add_chunks(chunks, embeddings)
        
        # Verify IDs are valid UUIDs
        call_args = mock_collection.add.call_args[1]
        ids = call_args['ids']
        
        assert len(ids) == 5
        for chunk_id in ids:
            # Should be valid UUID string
            uuid.UUID(chunk_id)  # Will raise if invalid
    
    @patch('chromadb.PersistentClient')
    def test_empty_chunk_handling(self, mock_client_class):
        """Test handling of empty chunks."""
        # Mock ChromaDB
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        store = ChromaVectorStore("/tmp/test_store")
        
        # Test with empty lists
        store.add_chunks([], [])
        
        # Should not call add
        mock_collection.add.assert_not_called()
        
        # Test with mismatched lengths (should raise)
        with pytest.raises(ValueError):
            store.add_chunks([{'id': '1', 'text': 'test'}], [])


class TestTextExtractor:
    """Test text extraction functionality."""
    
    def test_pypdf_extraction(self, tmp_path):
        """Test PyPDF text extraction fallback."""
        extractor = TextExtractor(use_vision=False)
        
        # Create a simple test PDF
        from pypdf import PdfWriter, PdfReader
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        import io
        
        # Create PDF with reportlab
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        c.drawString(100, 750, "Test PDF Content")
        c.drawString(100, 700, "Page 1 of the test document")
        c.showPage()
        c.drawString(100, 750, "Second page content")
        c.showPage()
        c.save()
        
        # Write to file
        pdf_path = tmp_path / "test.pdf"
        with open(pdf_path, 'wb') as f:
            f.write(buffer.getvalue())
        
        # Extract text
        pages = extractor.extract_text_from_pdf(str(pdf_path))
        
        assert len(pages) == 2
        assert pages[0]['page_num'] == 1
        assert pages[1]['page_num'] == 2
        assert 'Test PDF Content' in pages[0]['content']['raw_text']
        assert 'Second page' in pages[1]['content']['raw_text']
        assert pages[0]['extraction_method'] == 'pypdf'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])