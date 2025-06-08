"""
Integration tests for the vector search pipeline.
Tests the full document processing and search workflow.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import io

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.vector_processor import VectorProcessor
from src.vector_search import VectorSearcher
from pypdf import PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter


class TestVectorPipeline:
    """Test the complete vector processing pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_pdf(self, temp_dir):
        """Create a sample PDF for testing."""
        pdf_path = temp_dir / "test_document.pdf"
        
        # Create PDF with reportlab
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        
        # Page 1
        c.drawString(100, 750, "Medical Report for John Doe")
        c.drawString(100, 700, "Patient: John Doe")
        c.drawString(100, 650, "Date: January 15, 2024")
        c.drawString(100, 600, "Diagnosis: The patient presented with acute symptoms.")
        c.drawString(100, 550, "Treatment plan includes medication and physical therapy.")
        c.showPage()
        
        # Page 2
        c.drawString(100, 750, "Follow-up Notes")
        c.drawString(100, 700, "Patient showed improvement after initial treatment.")
        c.drawString(100, 650, "Recommend continuing current medication regimen.")
        c.showPage()
        
        c.save()
        
        # Write to file
        with open(pdf_path, 'wb') as f:
            f.write(buffer.getvalue())
        
        return pdf_path
    
    @patch('ollama.Client')
    @patch('chromadb.PersistentClient')
    def test_full_document_pipeline(self, mock_chroma_client, mock_ollama_client, temp_dir, sample_pdf):
        """Test processing a document through the full pipeline."""
        # Mock Ollama embeddings
        mock_ollama = MagicMock()
        mock_ollama.embeddings.return_value = {'embedding': [0.1] * 2048}
        mock_ollama_client.return_value = mock_ollama
        
        # Mock ChromaDB
        mock_chroma = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_chroma.create_collection.return_value = mock_collection
        mock_chroma.get_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_chroma
        
        # Initialize processor with temp directory
        processor = VectorProcessor(use_vision=False)
        processor.vector_store.path = temp_dir / "vector_store"
        
        # Process document
        metadata = {
            'filename': 'test_medical_report.pdf',
            'category': 'Medical Record',
            'exhibit_number': 1,
            'bates_start': 1,
            'bates_end': 2,
            'summary': 'Medical report for John Doe'
        }
        
        chunk_ids = processor.process_document(str(sample_pdf), metadata)
        
        # Verify chunks were created
        assert len(chunk_ids) > 0
        
        # Verify add was called
        mock_collection.add.assert_called()
        
        # Check the chunks that were added
        call_args = mock_collection.add.call_args[1]
        assert len(call_args['ids']) == len(chunk_ids)
        assert len(call_args['documents']) == len(chunk_ids)
        assert len(call_args['embeddings']) == len(chunk_ids)
        assert len(call_args['metadatas']) == len(chunk_ids)
        
        # Verify metadata
        for metadata_item in call_args['metadatas']:
            assert metadata_item['category'] == 'Medical Record'
            assert metadata_item['exhibit_number'] == 1
    
    @patch('ollama.Client')
    @patch('chromadb.PersistentClient')
    def test_search_after_indexing(self, mock_chroma_client, mock_ollama_client, temp_dir):
        """Test searching for documents after indexing."""
        # Mock Ollama embeddings
        mock_ollama = MagicMock()
        query_embedding = [0.2] * 2048
        mock_ollama.embeddings.return_value = {'embedding': query_embedding}
        mock_ollama_client.return_value = mock_ollama
        
        # Mock ChromaDB with search results
        mock_chroma = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 10
        
        # Mock search results
        mock_collection.query.return_value = {
            'ids': [['chunk1', 'chunk2']],
            'documents': [['Medical report text...', 'Patient diagnosis...']],
            'metadatas': [[
                {
                    'filename': 'medical_report.pdf',
                    'category': 'Medical Record',
                    'exhibit_number': 1,
                    'bates_start': 1,
                    'bates_end': 5,
                    'page': 1,
                    'summary': 'Medical report'
                },
                {
                    'filename': 'medical_report.pdf',
                    'category': 'Medical Record',
                    'exhibit_number': 1,
                    'bates_start': 1,
                    'bates_end': 5,
                    'page': 2,
                    'summary': 'Medical report'
                }
            ]],
            'distances': [[0.1, 0.2]]
        }
        
        mock_chroma.get_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_chroma
        
        # Initialize searcher
        searcher = VectorSearcher(str(temp_dir / "vector_store"))
        
        # Perform search
        results = searcher.search("patient diagnosis", n_results=5)
        
        # Verify results
        assert len(results) == 2
        assert results[0]['category'] == 'Medical Record'
        assert results[0]['relevance'] > results[1]['relevance']  # Lower distance = higher relevance
        assert 'Medical report text' in results[0]['text']
    
    @patch('ollama.Client')
    @patch('chromadb.PersistentClient')
    def test_category_filtering(self, mock_chroma_client, mock_ollama_client, temp_dir):
        """Test searching with category filters."""
        # Mock Ollama
        mock_ollama = MagicMock()
        mock_ollama.embeddings.return_value = {'embedding': [0.1] * 2048}
        mock_ollama_client.return_value = mock_ollama
        
        # Mock ChromaDB
        mock_chroma = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 50
        
        # Mock filtered results
        mock_collection.query.return_value = {
            'ids': [['bill1']],
            'documents': [['Invoice for services...']],
            'metadatas': [[{
                'filename': 'invoice.pdf',
                'category': 'Bill',
                'exhibit_number': 3,
                'bates_start': 10,
                'bates_end': 12,
                'page': 1
            }]],
            'distances': [[0.15]]
        }
        
        mock_chroma.get_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_chroma
        
        # Initialize searcher
        searcher = VectorSearcher(str(temp_dir / "vector_store"))
        
        # Search with category filter
        results = searcher.search("invoice payment", category="Bill")
        
        # Verify query was called with where clause
        mock_collection.query.assert_called_once()
        call_args = mock_collection.query.call_args[1]
        assert 'where' in call_args
        assert call_args['where']['category'] == 'Bill'
        
        # Verify results
        assert len(results) == 1
        assert results[0]['category'] == 'Bill'
    
    @patch('chromadb.PersistentClient')
    def test_bates_range_search(self, mock_chroma_client, temp_dir):
        """Test searching by Bates number range."""
        # Mock ChromaDB
        mock_chroma = MagicMock()
        mock_collection = MagicMock()
        
        # Mock results in Bates range
        mock_collection.query.return_value = {
            'ids': [['doc1', 'doc2']],
            'documents': [['Document 1 text', 'Document 2 text']],
            'metadatas': [[
                {
                    'filename': 'doc1.pdf',
                    'category': 'Pleading',
                    'bates_start': 50,
                    'bates_end': 75,
                    'exhibit_number': 5
                },
                {
                    'filename': 'doc2.pdf',
                    'category': 'Correspondence',
                    'bates_start': 80,
                    'bates_end': 90,
                    'exhibit_number': 6
                }
            ]]
        }
        
        mock_chroma.get_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_chroma
        
        # Initialize searcher  
        searcher = VectorSearcher(str(temp_dir / "vector_store"))
        
        # Search by Bates range
        results = searcher.search_by_bates_range(60, 85)
        
        # Verify query was called with proper where clause
        mock_collection.query.assert_called_once()
        call_args = mock_collection.query.call_args[1]
        assert 'where' in call_args
        assert '$and' in call_args['where']
        
        # Verify results
        assert len(results) == 2
        assert results[0]['bates_start'] == 50
        assert results[1]['bates_start'] == 80
    
    @patch('ollama.Client')
    @patch('chromadb.PersistentClient')
    def test_error_handling(self, mock_chroma_client, mock_ollama_client, temp_dir, sample_pdf):
        """Test error handling in the pipeline."""
        # Mock Ollama to fail
        mock_ollama = MagicMock()
        mock_ollama.embeddings.side_effect = [
            {'embedding': [0.1] * 2048},  # Success for init
            Exception("Embedding service unavailable")  # Fail during processing
        ]
        mock_ollama_client.return_value = mock_ollama
        
        # Mock ChromaDB
        mock_chroma = MagicMock()
        mock_collection = MagicMock()
        mock_chroma.create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_chroma
        
        # Initialize processor
        processor = VectorProcessor(use_vision=False)
        
        # Process should raise exception
        with pytest.raises(Exception) as exc_info:
            processor.process_document(str(sample_pdf), {})
        
        assert "Error processing" in str(exc_info.value)
    
    def test_vector_processor_stats(self, temp_dir):
        """Test getting statistics from vector processor."""
        with patch('chromadb.PersistentClient') as mock_chroma_client:
            # Mock ChromaDB
            mock_chroma = MagicMock()
            mock_collection = MagicMock()
            mock_collection.count.return_value = 42
            mock_collection.name = "legal_documents"
            mock_chroma.create_collection.return_value = mock_collection
            mock_chroma_client.return_value = mock_chroma
            
            with patch('ollama.Client') as mock_ollama_client:
                # Mock Ollama
                mock_ollama = MagicMock()
                mock_ollama.embeddings.return_value = {'embedding': [0.1] * 2048}
                mock_ollama_client.return_value = mock_ollama
                
                # Initialize processor
                processor = VectorProcessor(use_vision=False)
                
                # Get stats
                stats = processor.get_stats()
                
                assert stats['total_chunks'] == 42
                assert stats['collection_name'] == "legal_documents"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])