"""
LLM handler module using LangChain for document categorization, summarization, and naming.
Refactored to use LangChain components for better standardization and parallel execution.
"""

import logging
from typing import Dict, Optional, List
from enum import Enum

# Pydantic for structured output
from pydantic import BaseModel, Field, field_validator

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

# Configuration imports
from src.config import (
    OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_MODEL,
    OLLAMA_MODEL, OLLAMA_HOST, LLM_PROVIDER,
    CATEGORIZATION_SYSTEM_PROMPT, SUMMARIZATION_SYSTEM_PROMPT, 
    FILENAME_GENERATION_PROMPT
)

logger = logging.getLogger(__name__)


# Pydantic models for structured output
class DocumentCategory(str, Enum):
    """Valid document categories."""
    PLEADING = "Pleading"
    MEDICAL_RECORD = "Medical Record"
    BILL = "Bill"
    CORRESPONDENCE = "Correspondence"
    PHOTO = "Photo"
    VIDEO = "Video"
    DOCUMENTARY_EVIDENCE = "Documentary Evidence"
    UNCATEGORIZED = "Uncategorized"


class CategoryOutput(BaseModel):
    """Structured output for document categorization."""
    category: DocumentCategory = Field(
        description="The category of the document based on its filename"
    )
    
    @field_validator('category', mode='before')
    @classmethod
    def validate_category(cls, v):
        """Ensure the category is valid."""
        if isinstance(v, str):
            # Try to match the string to enum value
            for cat in DocumentCategory:
                if cat.value.lower() == v.lower():
                    return cat
        return DocumentCategory.UNCATEGORIZED


class SummaryOutput(BaseModel):
    """Structured output for document summary."""
    summary: str = Field(
        description="A one-sentence summary of the document",
        max_length=150
    )


class FilenameOutput(BaseModel):
    """Structured output for descriptive filename."""
    filename: str = Field(
        description="A descriptive filename for the document",
        max_length=100
    )


class LLMCategorizer:
    """
    Handles document categorization, summarization, and filename generation using LangChain.
    """
    
    def __init__(self, provider: str = None):
        """Initialize the LLM categorizer with specified provider."""
        # Use configured provider if not specified
        provider = provider or LLM_PROVIDER
        
        logger.info(f"Initializing LLM with provider: {provider}")
        
        # Initialize the appropriate LLM
        self.llm = self._init_llm(provider)
        
        # Create prompt templates
        self.categorization_prompt = ChatPromptTemplate.from_messages([
            ("system", CATEGORIZATION_SYSTEM_PROMPT),
            ("user", "Filename: {filename}")
        ])
        
        self.summarization_prompt = ChatPromptTemplate.from_messages([
            ("system", SUMMARIZATION_SYSTEM_PROMPT),
            ("user", "Create a one-sentence summary for this document filename: {filename}")
        ])
        
        self.filename_prompt = ChatPromptTemplate.from_messages([
            ("system", FILENAME_GENERATION_PROMPT),
            ("user", "Generate a descriptive filename for: {filename}")
        ])
        
        # Create output parsers
        self.category_parser = PydanticOutputParser(pydantic_object=CategoryOutput)
        self.str_parser = StrOutputParser()
        
        # Build the LCEL chains
        self._build_chains()
    
    def _init_llm(self, provider: str) -> BaseChatModel:
        """Initialize the appropriate LLM based on provider."""
        if provider == "openai":
            if not OPENAI_API_KEY:
                raise ValueError("OpenAI API key is required")
            
            return ChatOpenAI(
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_API_BASE,
                model=OPENAI_MODEL,
                temperature=0.2,
                max_tokens=50
            )
        elif provider == "ollama":
            # Verify Ollama is running
            try:
                import ollama
                client = ollama.Client(host=OLLAMA_HOST)
                client.list()  # Test connection
            except Exception as e:
                raise ValueError(f"Cannot connect to Ollama at {OLLAMA_HOST}. Ensure Ollama is running: {e}")
            
            return ChatOllama(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_HOST,
                temperature=0.2,
                num_predict=50
            )
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
    
    def _build_chains(self):
        """Build the LCEL chains for each operation."""
        # Categorization chain with structured output
        self.categorization_chain = (
            self.categorization_prompt 
            | self.llm 
            | self.str_parser
            | self._parse_category
        )
        
        # Summarization chain
        self.summarization_chain = (
            self.summarization_prompt 
            | self.llm.with_config({"temperature": 0.3})
            | self.str_parser
            | self._validate_summary
        )
        
        # Filename generation chain
        self.filename_chain = (
            self.filename_prompt 
            | self.llm 
            | self.str_parser
            | self._validate_filename
        )
        
        # Parallel execution chain for all three operations
        self.parallel_chain = RunnableParallel(
            category=self.categorization_chain,
            summary=self.summarization_chain,
            descriptive_name=self.filename_chain
        )
    
    def _parse_category(self, category_str: str) -> str:
        """Parse and validate category string."""
        try:
            # Clean the output
            category_str = category_str.strip()
            
            # Try to match to valid category
            for cat in DocumentCategory:
                if cat.value.lower() == category_str.lower():
                    return cat.value
            
            logger.warning(f"Invalid category '{category_str}'. Defaulting to Uncategorized.")
            return DocumentCategory.UNCATEGORIZED.value
        except Exception as e:
            logger.error(f"Error parsing category: {e}")
            return DocumentCategory.UNCATEGORIZED.value
    
    def _validate_summary(self, summary: str) -> str:
        """Validate and clean summary."""
        summary = summary.strip()
        if not summary or len(summary) > 150:
            return "Document summary not available"
        return summary
    
    def _validate_filename(self, filename: str) -> str:
        """Validate and clean filename."""
        filename = filename.strip()
        # Remove any invalid filename characters
        invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        for char in invalid_chars:
            filename = filename.replace(char, '')
        
        if not filename or len(filename) > 100:
            return "Document"
        return filename
    
    def categorize_document(self, filename: str) -> str:
        """Categorizes a document based on its filename using an LLM."""
        try:
            logger.info(f"Attempting to categorize: {filename}")
            
            # Use the categorization chain
            category = self.categorization_chain.invoke({"filename": filename})
            
            logger.info(f"LLM categorized '{filename}' as: {category}")
            return category
            
        except Exception as e:
            logger.error(f"Error during categorization for '{filename}': {e}")
            return DocumentCategory.UNCATEGORIZED.value
    
    def summarize_document(self, filename: str) -> str:
        """Creates a one-sentence summary of a document based on its filename."""
        try:
            logger.info(f"Attempting to summarize: {filename}")
            
            # Use the summarization chain
            summary = self.summarization_chain.invoke({"filename": filename})
            
            logger.info(f"LLM summary for '{filename}': {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"Error during summarization for '{filename}': {e}")
            return f"Document titled '{filename}'"
    
    def generate_descriptive_filename(self, filename: str) -> str:
        """Generates a descriptive filename for an exhibit."""
        try:
            logger.info(f"Attempting to generate descriptive filename for: {filename}")
            
            # Use the filename generation chain
            descriptive_name = self.filename_chain.invoke({"filename": filename})
            
            logger.info(f"Generated filename for '{filename}': {descriptive_name}")
            return descriptive_name
            
        except Exception as e:
            logger.error(f"Error during filename generation for '{filename}': {e}")
            return "Document"
    
    def process_document_parallel(self, filename: str) -> Dict[str, str]:
        """
        Process all three operations (categorize, summarize, generate filename) in parallel.
        Returns a dictionary with all results.
        """
        try:
            logger.info(f"Processing document '{filename}' with parallel execution")
            
            # Execute all three operations in parallel
            results = self.parallel_chain.invoke({"filename": filename})
            
            logger.info(f"Parallel processing complete for '{filename}'")
            return results
            
        except Exception as e:
            logger.error(f"Error during parallel processing for '{filename}': {e}")
            # Return defaults
            return {
                "category": DocumentCategory.UNCATEGORIZED.value,
                "summary": f"Document titled '{filename}'",
                "descriptive_name": "Document"
            }


# Backward compatibility: maintain the same interface
def get_llm_categorizer(provider: str = None) -> LLMCategorizer:
    """Factory function to get an LLM categorizer instance."""
    return LLMCategorizer(provider)