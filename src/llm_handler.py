"""
LLM handler module using LangChain for document categorization, summarization, and naming.
Refactored to use LangChain components for better standardization and parallel execution.
"""

import logging
from typing import Dict, Optional, List, Any
from enum import Enum
from pathlib import Path

# Pydantic for structured output
from pydantic import BaseModel, Field, field_validator

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

# Configuration imports
from src.config import (
    OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_MODEL,
    OLLAMA_MODEL, OLLAMA_HOST, LLM_PROVIDER,
    CATEGORIZATION_SYSTEM_PROMPT, SUMMARIZATION_SYSTEM_PROMPT, 
    FILENAME_GENERATION_PROMPT,
    LMSTUDIO_HOST, LMSTUDIO_MODEL, LMSTUDIO_MAX_TOKENS,
    LMSTUDIO_MODEL_MAPPING, ENABLE_MULTI_MODEL
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
        self.provider = provider
        
        logger.info(f"Initializing LLM with provider: {provider}")
        
        # Check if multi-model is enabled for LM Studio
        self.multi_model_enabled = (
            provider == "lmstudio" and 
            ENABLE_MULTI_MODEL and 
            len(set(LMSTUDIO_MODEL_MAPPING.values()) - {"", "none"}) > 1
        )
        
        if self.multi_model_enabled:
            logger.info("Multi-model pipeline enabled")
            # Initialize multiple models for different tasks
            self.models = self._init_multi_models()
            # Set default model for backward compatibility
            self.llm = self.models.get("categorization", list(self.models.values())[0])
        else:
            # Initialize single model (existing behavior)
            self.llm = self._init_llm(provider)
            self.models = {"default": self.llm}
        
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
        elif provider == "lmstudio":
            # LM Studio uses OpenAI-compatible API
            # Verify LM Studio is running
            try:
                import requests
                # Extract base URL without /v1
                base_url = LMSTUDIO_HOST.replace('/v1', '')
                response = requests.get(f"{base_url}/api/health", timeout=5)
                if response.status_code != 200:
                    raise ValueError("LM Studio server not healthy")
            except Exception as e:
                raise ValueError(f"Cannot connect to LM Studio at {LMSTUDIO_HOST}. Ensure LM Studio is running: {e}")
            
            # Use ChatOpenAI with LM Studio endpoint
            return ChatOpenAI(
                api_key="lm-studio",  # LM Studio doesn't require API key
                base_url=LMSTUDIO_HOST,
                model=LMSTUDIO_MODEL,
                temperature=0.2,
                max_tokens=LMSTUDIO_MAX_TOKENS
            )
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
    
    def _init_multi_models(self) -> Dict[str, BaseChatModel]:
        """Initialize task-specific models for LM Studio."""
        models = {}
        
        # Only initialize models that are configured and not "none"
        for task, model_id in LMSTUDIO_MODEL_MAPPING.items():
            if model_id and model_id != "none":
                logger.info(f"Initializing {task} model: {model_id}")
                try:
                    # Each task gets its own model instance
                    models[task] = ChatOpenAI(
                        api_key="lm-studio",
                        base_url=LMSTUDIO_HOST,
                        model=model_id,
                        temperature=0.2,
                        max_tokens=LMSTUDIO_MAX_TOKENS
                    )
                except Exception as e:
                    logger.error(f"Failed to initialize {task} model {model_id}: {e}")
                    # Skip this model but continue with others
                    continue
        
        if not models:
            raise ValueError("No models could be initialized for multi-model pipeline")
            
        logger.info(f"Successfully initialized {len(models)} models")
        return models
    
    def _get_model_for_task(self, task: str = "default") -> BaseChatModel:
        """Get appropriate model for a specific task."""
        if self.multi_model_enabled and task in self.models:
            return self.models[task]
        elif "default" in self.models:
            return self.models["default"]
        else:
            # Fallback to any available model
            return list(self.models.values())[0]
    
    def _build_chains(self):
        """Build the LCEL chains for each operation."""
        # Get appropriate models for each task
        categorization_model = self._get_model_for_task("categorization")
        synthesis_model = self._get_model_for_task("synthesis")
        
        # Categorization chain with structured output
        self.categorization_chain = (
            self.categorization_prompt 
            | categorization_model 
            | self.str_parser
            | self._parse_category
        )
        
        # Summarization chain (use synthesis model if available)
        self.summarization_chain = (
            self.summarization_prompt 
            | synthesis_model.with_config({"temperature": 0.3})
            | self.str_parser
            | self._validate_summary
        )
        
        # Filename generation chain (use synthesis model if available)
        self.filename_chain = (
            self.filename_prompt 
            | synthesis_model 
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
    
    def categorize_with_model(self, pdf_path: Path, model_task: str = "categorization") -> Dict[str, Any]:
        """
        Categorize document using a specific model task.
        
        Args:
            pdf_path: Path to the PDF document
            model_task: Task type for model selection ('visual', 'categorization', etc.)
            
        Returns:
            Categorization results
        """
        if not self.multi_model_enabled:
            # Fallback to standard categorization
            return self.categorize_document(pdf_path)
        
        # Get the appropriate model
        if model_task == "visual" and "visual" in self.models:
            model = self.models["visual"]
            logger.info(f"Using visual model for {pdf_path.name}")
        else:
            model = self._get_model_for_task(model_task)
        
        # Build a custom chain with the selected model
        custom_chain = (
            {"filename": RunnablePassthrough()}
            | RunnableParallel(
                category=(
                    self.categorization_prompt 
                    | model 
                    | self.str_parser
                    | self._parse_category
                ),
                summary=(
                    self.summarization_prompt 
                    | model.with_config({"temperature": 0.3})
                    | self.str_parser
                    | self._validate_summary
                ),
                descriptive_name=(
                    self.filename_prompt 
                    | model.with_config({"temperature": 0.5})
                    | self.str_parser
                    | self._clean_descriptive_name
                )
            )
        )
        
        try:
            filename = pdf_path.name
            results = custom_chain.invoke({"filename": filename})
            logger.info(f"Categorized '{filename}' with {model_task} model")
            return results
        except Exception as e:
            logger.error(f"Error categorizing with {model_task} model: {e}")
            # Fallback to standard categorization
            return self.categorize_document(pdf_path)


# Backward compatibility: maintain the same interface
def get_llm_categorizer(provider: str = None) -> LLMCategorizer:
    """Factory function to get an LLM categorizer instance."""
    return LLMCategorizer(provider)