# File: src/retrieval_agent/agent_config.py
"""
Configuration settings for the Iterative Retrieval Agent.

This module centralizes all configuration parameters specific to the retrieval agent,
allowing for easy tuning and modification without changing core logic.
"""

# --- Iteration Control ---
MAX_ITERATIONS = 3  # Default maximum number of search-analysis-refinement cycles
MIN_RELEVANCE_SCORE_FOR_FACTS = 0.0  # Minimum relevance score from vector search to consider a chunk (disabled for now)

# --- LLM Configuration for Agent Tasks ---
# Agent tasks often benefit from lower temperature for predictability
AGENT_LLM_TEMPERATURE = 0.1

# You can use the global OLLAMA_MODEL from src.config or define specific ones here
# For example, a more capable model for analysis/synthesis vs. a quicker one for decomposition.
# If using global, ensure src.config is imported.
# For now, let's assume we use the global OLLAMA_MODEL for simplicity.

# --- Search Parameters ---
NUM_RESULTS_PER_SUB_QUERY = 5  # Number of documents to retrieve per vector search
CONTEXT_WINDOW_SIZE_FOR_SYNTHESIS = 8000  # Approx token limit for context sent to final synthesis

# --- Re-ranking ---
# If you implement LLM-based re-ranking, config for it might go here
ENABLE_LLM_RE_RANKING = False
NUM_CHUNKS_TO_RE_RANK = 10  # How many top chunks to re-rank using LLM

# --- Logging & Debugging ---
LOG_INTERMEDIATE_STEPS = True  # For verbose logging during agent execution

# --- Tool Specific ---
# Tool-specific configurations can be added here as needed
# For now, VectorSearcher uses src.config

# --- Legal Domain Specific ---
# Categories that the agent should be aware of for filtering
LEGAL_DOCUMENT_CATEGORIES = [
    "Pleading",
    "Medical Record", 
    "Bill",
    "Correspondence",
    "Photo",
    "Video",
    "Documentary Evidence",
    "Uncategorized"
]

# --- Performance Tuning ---
# Maximum length for any single retrieved chunk to process
MAX_CHUNK_LENGTH = 2000  # Characters

# --- Future Enhancement Flags ---
ENABLE_SQL_SEARCH = False  # When True, enables PostgreSQL keyword search tool
ENABLE_CROSS_ENCODER_RERANKING = False  # For future local re-ranking model