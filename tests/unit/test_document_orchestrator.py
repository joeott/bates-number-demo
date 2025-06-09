"""
Unit tests for the document orchestrator using LangChain LCEL chains.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import tempfile

from src.document_orchestrator import (
    DocumentOrchestrator,
    DocumentInput,
    DocumentMetadata,
    BatesResult,
    ExhibitResult,
    ProcessingResult
)


class TestPydanticModels:
    """Test Pydantic models used in orchestration."""
    
    def test_document_input(self):
        """Test DocumentInput model."""
        doc_input = DocumentInput(
            file_path=Path("/test/doc.pdf"),
            bates_counter=100,
            exhibit_counter=5
        )
        assert doc_input.file_path == Path("/test/doc.pdf")
        assert doc_input.bates_counter == 100
        assert doc_input.exhibit_counter == 5
    
    def test_processing_result(self):
        """Test ProcessingResult model."""
        result = ProcessingResult(
            success=True,
            exhibit_id="Exhibit 1",
            original_filename="test.pdf",
            final_filename="Exhibit 1 - Test Document.pdf",
            category="Pleading",
            summary="Test summary",
            bates_range="000001-000010",
            vector_chunks=5,
            postgres_stored=True
        )
        assert result.success is True
        assert result.exhibit_id == "Exhibit 1"
        assert result.vector_chunks == 5


class TestDocumentOrchestrator:
    """Test the DocumentOrchestrator class."""
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components for testing."""
        llm_categorizer = Mock()
        llm_categorizer.process_document_parallel.return_value = {
            "category": "Pleading",
            "summary": "Test document summary",
            "descriptive_name": "Test Document"
        }
        
        pdf_processor = Mock()
        pdf_processor.bates_stamp_pdf.return_value = ("000001", "000005", 6)
        pdf_processor.add_exhibit_mark.return_value = True
        
        vector_processor = Mock()
        vector_processor.process_document.return_value = (["id1", "id2"], "full text", ["page1", "page2"])
        vector_processor.get_stats.return_value = {"total_chunks": 100}
        
        postgres_storage = Mock()
        postgres_storage.store_document_text.return_value = 1
        
        return {
            "llm_categorizer": llm_categorizer,
            "pdf_processor": pdf_processor,
            "vector_processor": vector_processor,
            "postgres_storage": postgres_storage
        }
    
    def test_orchestrator_initialization(self, mock_components):
        """Test orchestrator initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            orchestrator = DocumentOrchestrator(
                llm_categorizer=mock_components["llm_categorizer"],
                pdf_processor=mock_components["pdf_processor"],
                vector_processor=mock_components["vector_processor"],
                postgres_storage=mock_components["postgres_storage"],
                output_dir=Path(temp_dir)
            )
            
            assert orchestrator.llm_categorizer is not None
            assert orchestrator.pdf_processor is not None
            assert orchestrator.vector_processor is not None
            assert orchestrator.postgres_storage is not None
            assert orchestrator.bates_output_dir.exists()
            assert orchestrator.exhibits_output_dir.exists()
    
    def test_chain_building(self, mock_components):
        """Test that LCEL chains are built correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            orchestrator = DocumentOrchestrator(
                llm_categorizer=mock_components["llm_categorizer"],
                pdf_processor=mock_components["pdf_processor"],
                output_dir=Path(temp_dir)
            )
            
            # Verify chains are created
            assert hasattr(orchestrator, 'validation_chain')
            assert hasattr(orchestrator, 'llm_chain')
            assert hasattr(orchestrator, 'bates_chain')
            assert hasattr(orchestrator, 'exhibit_chain')
            assert hasattr(orchestrator, 'vector_branch')
            assert hasattr(orchestrator, 'postgres_branch')
            assert hasattr(orchestrator, 'processing_chain')
            assert hasattr(orchestrator, 'safe_processing_chain')
    
    def test_document_validation(self, mock_components):
        """Test document validation chain."""
        with tempfile.TemporaryDirectory() as temp_dir:
            orchestrator = DocumentOrchestrator(
                llm_categorizer=mock_components["llm_categorizer"],
                pdf_processor=mock_components["pdf_processor"],
                output_dir=Path(temp_dir)
            )
            
            # Valid document
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                valid_input = {
                    "file_path": Path(tmp.name),
                    "bates_counter": 1,
                    "exhibit_counter": 1
                }
                result = orchestrator._validate_document(valid_input)
                assert result["success"] is True
                assert isinstance(result["input"], DocumentInput)
            
            # Invalid document (non-existent)
            invalid_input = {
                "file_path": Path("/nonexistent/doc.pdf"),
                "bates_counter": 1,
                "exhibit_counter": 1
            }
            with pytest.raises(ValueError, match="File not found"):
                orchestrator._validate_document(invalid_input)
    
    def test_llm_processing(self, mock_components):
        """Test LLM processing chain."""
        with tempfile.TemporaryDirectory() as temp_dir:
            orchestrator = DocumentOrchestrator(
                llm_categorizer=mock_components["llm_categorizer"],
                pdf_processor=mock_components["pdf_processor"],
                output_dir=Path(temp_dir)
            )
            
            data = {
                "input": DocumentInput(
                    file_path=Path("test.pdf"),
                    bates_counter=1,
                    exhibit_counter=1
                ),
                "success": True
            }
            
            result = orchestrator._process_with_llm(data)
            
            assert "metadata" in result
            assert isinstance(result["metadata"], DocumentMetadata)
            assert result["metadata"].category == "Pleading"
            assert result["metadata"].summary == "Test document summary"
    
    def test_process_document_success(self, mock_components):
        """Test successful document processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test PDF
            test_pdf = Path(temp_dir) / "test.pdf"
            test_pdf.write_text("dummy pdf content")
            
            orchestrator = DocumentOrchestrator(
                llm_categorizer=mock_components["llm_categorizer"],
                pdf_processor=mock_components["pdf_processor"],
                vector_processor=mock_components["vector_processor"],
                postgres_storage=mock_components["postgres_storage"],
                output_dir=Path(temp_dir)
            )
            
            result, next_bates, next_exhibit = orchestrator.process_document(
                test_pdf, 
                bates_counter=1, 
                exhibit_counter=1
            )
            
            # Verify result
            assert isinstance(result, ProcessingResult)
            assert result.success is True
            assert result.exhibit_id == "Exhibit 1"
            assert result.category == "Pleading"
            assert result.bates_range == "000001-000005"
            
            # Verify counters updated
            assert next_bates == 6  # From mock: 5 pages processed
            assert next_exhibit == 2  # Next exhibit number
    
    def test_process_batch(self, mock_components):
        """Test batch processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test PDFs
            test_pdfs = []
            for i in range(3):
                test_pdf = Path(temp_dir) / f"test{i}.pdf"
                test_pdf.write_text("dummy pdf content")
                test_pdfs.append(test_pdf)
            
            orchestrator = DocumentOrchestrator(
                llm_categorizer=mock_components["llm_categorizer"],
                pdf_processor=mock_components["pdf_processor"],
                output_dir=Path(temp_dir)
            )
            
            # Mock different page counts for each document
            mock_components["pdf_processor"].bates_stamp_pdf.side_effect = [
                ("000001", "000005", 6),    # 5 pages
                ("000006", "000008", 9),    # 3 pages
                ("000009", "000015", 16)    # 7 pages
            ]
            
            results = orchestrator.process_batch(test_pdfs)
            
            assert len(results) == 3
            assert all(isinstance(r, ProcessingResult) for r in results)
            assert orchestrator._next_bates == 16
            assert orchestrator._next_exhibit == 4
    
    def test_error_handling(self, mock_components):
        """Test error handling in processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            orchestrator = DocumentOrchestrator(
                llm_categorizer=mock_components["llm_categorizer"],
                pdf_processor=mock_components["pdf_processor"],
                output_dir=Path(temp_dir)
            )
            
            # Process non-existent file
            result, next_bates, next_exhibit = orchestrator.process_document(
                Path("/nonexistent/file.pdf"),
                bates_counter=1,
                exhibit_counter=1
            )
            
            assert result.success is False
            assert result.error is not None
            assert next_bates == 1  # Unchanged on error
            assert next_exhibit == 1  # Unchanged on error
    
    def test_conditional_branches(self, mock_components):
        """Test conditional vector and postgres branches."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test without optional components
            orchestrator_minimal = DocumentOrchestrator(
                llm_categorizer=mock_components["llm_categorizer"],
                pdf_processor=mock_components["pdf_processor"],
                vector_processor=None,
                postgres_storage=None,
                output_dir=Path(temp_dir)
            )
            
            # The branches should pass through without processing
            data = {"success": True, "exhibit_result": Mock()}
            
            # Vector branch should pass through
            result = orchestrator_minimal.vector_branch.invoke(data)
            assert "vector_chunks" not in result
            
            # Postgres branch should pass through  
            result = orchestrator_minimal.postgres_branch.invoke(data)
            assert "postgres_stored" not in result


class TestCSVGeneration:
    """Test CSV log generation."""
    
    def test_generate_csv_log(self, tmp_path):
        """Test CSV log generation."""
        orchestrator = DocumentOrchestrator(
            llm_categorizer=Mock(),
            pdf_processor=Mock(),
            output_dir=tmp_path
        )
        
        results = [
            ProcessingResult(
                success=True,
                exhibit_id="Exhibit 1",
                original_filename="doc1.pdf",
                final_filename="Exhibit 1 - Document One.pdf",
                category="Pleading",
                summary="First document",
                bates_range="000001-000005",
                vector_chunks=3,
                postgres_stored=True
            ),
            ProcessingResult(
                success=False,
                exhibit_id="",
                original_filename="doc2.pdf",
                final_filename="",
                category="",
                summary="",
                bates_range="",
                error="Processing failed"
            )
        ]
        
        csv_path = tmp_path / "test_log.csv"
        orchestrator.generate_csv_log(results, csv_path)
        
        assert csv_path.exists()
        
        # Read and verify CSV content
        import csv
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        assert len(rows) == 2
        assert rows[0]['Status'] == 'Success'
        assert rows[0]['Exhibit ID'] == 'Exhibit 1'
        assert rows[1]['Status'] == 'Failed'
        assert rows[1]['Error'] == 'Processing failed'