import logging
from abc import ABC, abstractmethod
from openai import OpenAI
import ollama
from src.config import (
    OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_MODEL,
    OLLAMA_MODEL, OLLAMA_HOST, LLM_PROVIDER,
    CATEGORIZATION_SYSTEM_PROMPT, SUMMARIZATION_SYSTEM_PROMPT, 
    FILENAME_GENERATION_PROMPT
)

logger = logging.getLogger(__name__)

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 50) -> str:
        pass

class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str, api_base: str = None, model: str = "gpt-4o-mini-2024-07-18"):
        if not api_key:
            raise ValueError("OpenAI API key is required")
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = model
    
    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 50) -> str:
        # Build request parameters - some models don't support all parameters
        params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }
        
        # Only add optional parameters for models that support them
        try:
            completion = self.client.chat.completions.create(
                **params,
                temperature=temperature,
                max_completion_tokens=max_tokens
            )
        except Exception as e:
            if "temperature" in str(e) or "max_completion_tokens" in str(e):
                # Fallback for models that don't support these parameters
                completion = self.client.chat.completions.create(**params)
            else:
                raise e
        
        return completion.choices[0].message.content.strip()

class OllamaProvider(BaseLLMProvider):
    def __init__(self, model: str = "llama3.2:3b", host: str = "http://localhost:11434"):
        self.model = model
        self.client = ollama.Client(host=host)
        # Verify Ollama is running and model is available
        try:
            self.client.list()
        except Exception as e:
            raise ValueError(f"Cannot connect to Ollama at {host}. Ensure Ollama is running: {e}")
    
    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 50) -> str:
        # Combine prompts for Ollama format
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        response = self.client.generate(
            model=self.model,
            prompt=full_prompt,
            options={
                "temperature": temperature,
                "num_predict": max_tokens
            }
        )
        return response['response'].strip()

class LLMCategorizer:
    def __init__(self, provider: str = None):
        # Use configured provider if not specified
        provider = provider or LLM_PROVIDER
        
        logger.info(f"Initializing LLM with provider: {provider}")
        
        if provider == "openai":
            self.provider = OpenAIProvider(
                api_key=OPENAI_API_KEY,
                api_base=OPENAI_API_BASE,
                model=OPENAI_MODEL
            )
        elif provider == "ollama":
            self.provider = OllamaProvider(
                model=OLLAMA_MODEL,
                host=OLLAMA_HOST
            )
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
    
    def _call_llm(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        """Unified method to call the LLM provider"""
        try:
            return self.provider.complete(system_prompt, user_prompt, temperature)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def categorize_document(self, filename: str) -> str:
        """Categorizes a document based on its filename using an LLM."""
        try:
            logger.info(f"Attempting to categorize: {filename}")
            category = self._call_llm(
                CATEGORIZATION_SYSTEM_PROMPT,
                f"Filename: {filename}"
            )
            logger.info(f"LLM categorized '{filename}' as: {category}")
            
            # Validate category
            valid_categories = ["Pleading", "Medical Record", "Bill", "Correspondence", 
                              "Photo", "Video", "Documentary Evidence", "Uncategorized"]
            if category not in valid_categories:
                logger.warning(f"Invalid category '{category}' for '{filename}'. Defaulting to Uncategorized.")
                return "Uncategorized"
            return category
        except Exception as e:
            logger.error(f"Error during categorization for '{filename}': {e}")
            return "Uncategorized"
    
    def summarize_document(self, filename: str) -> str:
        """Creates a one-sentence summary of a document based on its filename."""
        try:
            logger.info(f"Attempting to summarize: {filename}")
            summary = self._call_llm(
                SUMMARIZATION_SYSTEM_PROMPT,
                f"Create a one-sentence summary for this document filename: {filename}",
                temperature=0.3
            )
            logger.info(f"LLM summary for '{filename}': {summary}")
            
            if not summary or len(summary) > 150:
                logger.warning(f"Summary too long or empty for '{filename}'. Using default.")
                return f"Document titled '{filename}'"
            return summary
        except Exception as e:
            logger.error(f"Error during summarization for '{filename}': {e}")
            return f"Document titled '{filename}'"
    
    def generate_descriptive_filename(self, filename: str) -> str:
        """Generates a descriptive filename for an exhibit."""
        try:
            logger.info(f"Attempting to generate descriptive filename for: {filename}")
            descriptive_name = self._call_llm(
                FILENAME_GENERATION_PROMPT,
                f"Generate a descriptive filename for: {filename}"
            )
            logger.info(f"Generated filename for '{filename}': {descriptive_name}")
            
            if not descriptive_name or len(descriptive_name) > 100:
                logger.warning(f"Generated filename too long or empty for '{filename}'. Using default.")
                return "Document"
            return descriptive_name
        except Exception as e:
            logger.error(f"Error during filename generation for '{filename}': {e}")
            return "Document"