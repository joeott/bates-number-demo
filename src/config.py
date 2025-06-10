import os
from pathlib import Path
from dotenv import load_dotenv


# Find the project root by looking for .env file
current_dir = Path(__file__).parent
project_root = current_dir.parent


# Load environment variables from .env file in project root
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    # Fallback to default behavior
    load_dotenv()

# --- LLM Provider Configuration ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE") # Optional, for self-hosted or alternative LLMs
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini-2024-07-18")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# LM Studio Configuration (OpenAI-compatible)
LMSTUDIO_HOST = os.getenv("LMSTUDIO_HOST", "http://localhost:1234/v1")
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "mlx-community/Qwen2.5-7B-Instruct-4bit")
LMSTUDIO_VISION_MODEL = os.getenv("LMSTUDIO_VISION_MODEL", "mlx-community/Qwen2.5-VL-3B-Instruct-4bit")
LMSTUDIO_EMBEDDING_MODEL = os.getenv("LMSTUDIO_EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5-GGUF")
LMSTUDIO_MAX_TOKENS = int(os.getenv("LMSTUDIO_MAX_TOKENS", "2048"))
LMSTUDIO_CONTEXT_LENGTH = int(os.getenv("LMSTUDIO_CONTEXT_LENGTH", "32768"))

# Multi-Model Configuration for LM Studio
# Allows using different models for different tasks in the pipeline
LMSTUDIO_VISUAL_MODEL = os.getenv("LMSTUDIO_VISUAL_MODEL", "pixtral-12b")
LMSTUDIO_REASONING_MODEL = os.getenv("LMSTUDIO_REASONING_MODEL", "mathstral-7b-v0.1")
LMSTUDIO_CATEGORIZATION_MODEL = os.getenv("LMSTUDIO_CATEGORIZATION_MODEL", LMSTUDIO_MODEL)  # Default to base model
LMSTUDIO_SYNTHESIS_MODEL = os.getenv("LMSTUDIO_SYNTHESIS_MODEL", "llama-4-scout-17b-16e-mlx-text")
# Embedding model already defined above as LMSTUDIO_EMBEDDING_MODEL

# Task mapping for multi-model pipeline
LMSTUDIO_MODEL_MAPPING = {
    "visual": LMSTUDIO_VISUAL_MODEL,
    "reasoning": LMSTUDIO_REASONING_MODEL,
    "categorization": LMSTUDIO_CATEGORIZATION_MODEL,
    "synthesis": LMSTUDIO_SYNTHESIS_MODEL,
    "embedding": LMSTUDIO_EMBEDDING_MODEL
}

# Enable multi-model pipeline (auto-enabled if multiple models are configured)
ENABLE_MULTI_MODEL = os.getenv("ENABLE_MULTI_MODEL", "auto").lower()
if ENABLE_MULTI_MODEL == "auto":
    # Check if multiple different models are configured
    unique_models = set(LMSTUDIO_MODEL_MAPPING.values())
    unique_models.discard("")  # Remove empty strings
    unique_models.discard("none")  # Remove "none" values
    ENABLE_MULTI_MODEL = len(unique_models) > 1
else:
    ENABLE_MULTI_MODEL = ENABLE_MULTI_MODEL == "true"

# Validate provider selection
if LLM_PROVIDER not in ["openai", "ollama", "lmstudio"]:
    print(f"WARNING: Invalid LLM_PROVIDER '{LLM_PROVIDER}'. Defaulting to 'openai'.")
    LLM_PROVIDER = "openai"

# --- Default Paths ---
# Relative to the project root. main.py will resolve these.
DEFAULT_INPUT_DIR = "input_documents"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_BATES_SUBDIR = "bates_numbered"
DEFAULT_EXHIBITS_SUBDIR = "exhibits"
DEFAULT_CSV_LOG_NAME = "exhibit_log.csv"

# --- PDF Processing Configuration ---
DEFAULT_BATES_FONT = "Helvetica"
DEFAULT_BATES_FONT_SIZE = 8
DEFAULT_EXHIBIT_FONT = "Helvetica-Bold"
DEFAULT_EXHIBIT_FONT_SIZE = 10
STAMP_MARGIN_POINTS = 18  # Approx 0.25 inch from bottom and right edges

# --- LLM Prompting ---
CATEGORIZATION_SYSTEM_PROMPT = """
You are a helpful legal assistant. Your task is to categorize documents based on their filename.
The categories are:
- Pleading (complaints, answers, motions, legal filings)
- Medical Record (medical reports, hospital records, treatment notes)
- Bill (invoices, receipts, financial charges, statements)
- Correspondence (emails, texts, letters, any written communication)
- Photo (images, photographs, screenshots)
- Video (video files, recordings)
- Documentary Evidence (contracts, reports, forms, or any other documentary evidence not fitting above categories)
- Uncategorized (when the category cannot be determined)

Based *only* on the filename provided by the user, choose the most appropriate single category from the list above.
Provide only the category name as your response. For example, if the filename is "invoice_2024.pdf", your response should be "Bill".
"""

SUMMARIZATION_SYSTEM_PROMPT = """
You are a legal assistant tasked with creating a one-sentence summary of a document based solely on its filename.
The summary should be concise, descriptive, and suitable for a legal exhibit log.
Focus on what the document likely contains based on its filename.
Keep the summary under 100 characters if possible.
"""

FILENAME_GENERATION_PROMPT = """
You are a legal assistant tasked with generating a descriptive filename for an exhibit based on the original filename.

Create a clear description that:
- Identifies the document type (e.g., Letter, Email, Invoice, Report, Motion, Medical Record, etc.)
- Includes key parties when identifiable (e.g., "to Smith", "from Jones Hospital", "re Johnson case")
- Keeps it concise (3-7 words typically)
- Uses proper case

Examples:
- "invoice_2024_01_15_smith.pdf" → "Invoice from Smith"
- "Letter_to_Johnson_re_settlement.pdf" → "Letter to Johnson re Settlement"
- "Medical_Report_Doe_Jan2024.pdf" → "Medical Report for Doe"
- "DEF_Initial_Production_2024_01_29.pdf" → "Defendant Initial Production"
- "Busey_Bank_Statement_July_2024.pdf" → "Bank Statement from Busey Bank"

Respond with ONLY the descriptive name. Do not include "Exhibit", exhibit numbers, or dates.
"""

# --- Error Handling & Logging ---
# (Could add logging levels, formats here if using a more complex logging setup via utils.py)

# --- Vector Search Configuration ---
ENABLE_VECTOR_SEARCH = os.getenv("ENABLE_VECTOR_SEARCH", "true").lower() == "true"
VECTOR_STORE_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "vector_store")
VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "qwen2.5-vision:7b")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "hf.co/Casual-Autopsy/snowflake-arctic-embed-l-v2.0-gguf:F32")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "750"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
PDF_DPI = int(os.getenv("PDF_DPI", "300"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
MAX_CHUNKS_PER_DOCUMENT = int(os.getenv("MAX_CHUNKS_PER_DOCUMENT", "100"))
ENABLE_VISION_OCR = os.getenv("ENABLE_VISION_OCR", "true").lower() == "true"
VISION_OCR_MODEL = os.getenv("VISION_OCR_MODEL", "google/gemma-3-12b")

# --- PostgreSQL Storage Configuration ---
ENABLE_POSTGRES_STORAGE = os.getenv("ENABLE_POSTGRES_STORAGE", "false").lower() == "true"
POSTGRES_CONNECTION = os.getenv(
    "POSTGRES_CONNECTION",
    "postgresql://user:password@localhost:5432/bates_documents"
)
POSTGRES_POOL_SIZE = int(os.getenv("POSTGRES_POOL_SIZE", "5"))
STORE_PAGE_LEVEL_TEXT = os.getenv("STORE_PAGE_LEVEL_TEXT", "true").lower() == "true"

# --- Caching Configuration ---
ENABLE_MODEL_CACHE = os.getenv("ENABLE_MODEL_CACHE", "true").lower() == "true"
ENABLE_DISK_CACHE = os.getenv("ENABLE_DISK_CACHE", "false").lower() == "true"
CACHE_DIR = os.getenv("CACHE_DIR", "cache")
MEMORY_CACHE_SIZE = int(os.getenv("MEMORY_CACHE_SIZE", "1000"))
CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", "24"))

# --- LangSmith Tracing Configuration ---
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "bates_number_demo")

# For backward compatibility with LangChain's expected environment variables
if LANGSMITH_TRACING:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if LANGSMITH_API_KEY:
        os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    if LANGSMITH_PROJECT:
        os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT

# --- Validation ---
if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
    # This check is more for guiding the developer; main.py will handle user-facing errors.
    print("WARNING: OPENAI_API_KEY is not set in the environment or .env file.")

if LANGSMITH_TRACING and not LANGSMITH_API_KEY:
    print("WARNING: LANGSMITH_TRACING is enabled but LANGSMITH_API_KEY is not set.")

if LANGSMITH_TRACING:
    print(f"INFO: LangSmith tracing enabled for project: {LANGSMITH_PROJECT}")
    print(f"INFO: LangSmith endpoint: {LANGSMITH_ENDPOINT}")