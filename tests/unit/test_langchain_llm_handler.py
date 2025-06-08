"""
Unit tests for the LangChain-based LLM handler components.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict

from src.llm_handler import (
    LLMCategorizer,
    DocumentCategory,
    CategoryOutput,
    SummaryOutput,
    FilenameOutput,
    get_llm_categorizer
)


class TestDocumentCategory:
    """Test the DocumentCategory enum."""
    
    def test_all_categories_defined(self):
        """Test that all expected categories are defined."""
        expected_categories = [
            "Pleading", "Medical Record", "Bill", "Correspondence",
            "Photo", "Video", "Documentary Evidence", "Uncategorized"
        ]
        
        actual_categories = [cat.value for cat in DocumentCategory]
        assert set(expected_categories) == set(actual_categories)


class TestPydanticModels:
    """Test the Pydantic models for structured output."""
    
    def test_category_output_validation(self):
        """Test CategoryOutput validation."""
        # Valid category
        output = CategoryOutput(category=DocumentCategory.PLEADING)
        assert output.category == DocumentCategory.PLEADING
        
        # String conversion
        output = CategoryOutput(category="Bill")
        assert output.category == DocumentCategory.BILL
        
        # Invalid category defaults to uncategorized
        output = CategoryOutput(category="InvalidCategory")
        assert output.category == DocumentCategory.UNCATEGORIZED
    
    def test_summary_output_validation(self):
        """Test SummaryOutput validation."""
        output = SummaryOutput(summary="This is a test summary.")
        assert output.summary == "This is a test summary."
    
    def test_filename_output_validation(self):
        """Test FilenameOutput validation."""
        output = FilenameOutput(filename="Test_Document")
        assert output.filename == "Test_Document"


class TestLLMCategorizer:
    """Test the LLMCategorizer class."""
    
    @patch('src.llm_handler.ChatOllama')
    @patch('src.llm_handler.ollama.Client')
    def test_init_ollama_provider(self, mock_ollama_client, mock_chat_ollama):
        """Test initialization with Ollama provider."""
        # Mock Ollama client
        mock_client_instance = Mock()
        mock_client_instance.list.return_value = []
        mock_ollama_client.return_value = mock_client_instance
        
        # Initialize categorizer
        with patch('src.llm_handler.LLM_PROVIDER', 'ollama'):
            categorizer = LLMCategorizer()
        
        # Verify Ollama was initialized
        mock_chat_ollama.assert_called_once()
        assert categorizer.llm is not None
    
    @patch('src.llm_handler.ChatOpenAI')
    def test_init_openai_provider(self, mock_chat_openai):
        """Test initialization with OpenAI provider."""
        with patch('src.llm_handler.OPENAI_API_KEY', 'test-key'):
            with patch('src.llm_handler.LLM_PROVIDER', 'openai'):
                categorizer = LLMCategorizer()
        
        # Verify OpenAI was initialized
        mock_chat_openai.assert_called_once()
        assert categorizer.llm is not None
    
    def test_init_invalid_provider(self):
        """Test initialization with invalid provider."""
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            LLMCategorizer(provider="invalid")
    
    def test_parse_category(self):
        """Test category parsing."""
        categorizer = LLMCategorizer.__new__(LLMCategorizer)
        
        # Valid categories
        assert categorizer._parse_category("Pleading") == "Pleading"
        assert categorizer._parse_category("medical record") == "Medical Record"
        assert categorizer._parse_category("BILL") == "Bill"
        
        # Invalid category
        assert categorizer._parse_category("Invalid") == "Uncategorized"
    
    def test_validate_summary(self):
        """Test summary validation."""
        categorizer = LLMCategorizer.__new__(LLMCategorizer)
        
        # Valid summary
        assert categorizer._validate_summary("This is a valid summary.") == "This is a valid summary."
        
        # Empty summary
        assert categorizer._validate_summary("") == "Document summary not available"
        
        # Too long summary
        long_summary = "x" * 200
        assert categorizer._validate_summary(long_summary) == "Document summary not available"
    
    def test_validate_filename(self):
        """Test filename validation."""
        categorizer = LLMCategorizer.__new__(LLMCategorizer)
        
        # Valid filename
        assert categorizer._validate_filename("Valid_Filename") == "Valid_Filename"
        
        # Filename with invalid characters
        assert categorizer._validate_filename("File/Name:With*Invalid|Chars") == "FileNameWithInvalidChars"
        
        # Empty filename
        assert categorizer._validate_filename("") == "Document"
    
    @patch('src.llm_handler.ChatOllama')
    @patch('src.llm_handler.ollama.Client')
    def test_categorize_document(self, mock_ollama_client, mock_chat_ollama):
        """Test document categorization."""
        # Setup mocks
        mock_client_instance = Mock()
        mock_client_instance.list.return_value = []
        mock_ollama_client.return_value = mock_client_instance
        
        mock_llm = Mock()
        mock_chat_ollama.return_value = mock_llm
        
        # Create mock chain that returns a category
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Pleading"
        
        with patch('src.llm_handler.LLM_PROVIDER', 'ollama'):
            categorizer = LLMCategorizer()
            categorizer.categorization_chain = mock_chain
        
        # Test categorization
        result = categorizer.categorize_document("test_document.pdf")
        assert result == "Pleading"
        mock_chain.invoke.assert_called_with({"filename": "test_document.pdf"})
    
    @patch('src.llm_handler.ChatOllama')
    @patch('src.llm_handler.ollama.Client')
    def test_process_document_parallel(self, mock_ollama_client, mock_chat_ollama):
        """Test parallel document processing."""
        # Setup mocks
        mock_client_instance = Mock()
        mock_client_instance.list.return_value = []
        mock_ollama_client.return_value = mock_client_instance
        
        mock_llm = Mock()
        mock_chat_ollama.return_value = mock_llm
        
        # Create mock parallel chain
        mock_parallel_chain = MagicMock()
        mock_parallel_chain.invoke.return_value = {
            "category": "Bill",
            "summary": "This is a billing document.",
            "descriptive_name": "Invoice_Document"
        }
        
        with patch('src.llm_handler.LLM_PROVIDER', 'ollama'):
            categorizer = LLMCategorizer()
            categorizer.parallel_chain = mock_parallel_chain
        
        # Test parallel processing
        result = categorizer.process_document_parallel("invoice.pdf")
        
        assert result["category"] == "Bill"
        assert result["summary"] == "This is a billing document."
        assert result["descriptive_name"] == "Invoice_Document"
        
        mock_parallel_chain.invoke.assert_called_with({"filename": "invoice.pdf"})
    
    @patch('src.llm_handler.ChatOllama')
    @patch('src.llm_handler.ollama.Client')
    def test_error_handling(self, mock_ollama_client, mock_chat_ollama):
        """Test error handling in LLM operations."""
        # Setup mocks
        mock_client_instance = Mock()
        mock_client_instance.list.return_value = []
        mock_ollama_client.return_value = mock_client_instance
        
        mock_llm = Mock()
        mock_chat_ollama.return_value = mock_llm
        
        with patch('src.llm_handler.LLM_PROVIDER', 'ollama'):
            categorizer = LLMCategorizer()
        
        # Mock chain that raises an error
        categorizer.categorization_chain = Mock(side_effect=Exception("LLM Error"))
        
        # Test error handling in categorization
        result = categorizer.categorize_document("test.pdf")
        assert result == "Uncategorized"
        
        # Test error handling in parallel processing
        categorizer.parallel_chain = Mock(side_effect=Exception("LLM Error"))
        result = categorizer.process_document_parallel("test.pdf")
        
        assert result["category"] == "Uncategorized"
        assert result["summary"] == "Document titled 'test.pdf'"
        assert result["descriptive_name"] == "Document"


class TestBackwardCompatibility:
    """Test backward compatibility features."""
    
    @patch('src.llm_handler.LLMCategorizer')
    def test_get_llm_categorizer(self, mock_categorizer_class):
        """Test the factory function."""
        mock_instance = Mock()
        mock_categorizer_class.return_value = mock_instance
        
        result = get_llm_categorizer("openai")
        
        mock_categorizer_class.assert_called_once_with("openai")
        assert result == mock_instance