Okay, here are the exact steps to implement every required script for the iterative retriever module, keeping in mind the directive to modify only these new files within the `src/retrieval_agent/` directory and to leverage the existing, functioning codebase.

**REMINDER AGENT: YOU ARE IMPLEMENTING *NEW* FILES WITHIN THE `src/retrieval_agent/` DIRECTORY. DO NOT MODIFY OTHER EXISTING SCRIPTS UNLESS EXPLICITLY STATED (e.g., adding an import in `src/__init__.py`). THE GOAL IS TO BUILD A NEW, ISOLATED MODULE THAT USES THE OUTPUTS OF THE EXISTING PIPELINE.**

---

**Step 0: Create Directory Structure**

Ensure the following directory and empty `__init__.py` files are created if they don't exist:

1.  Create directory: `project_root/src/retrieval_agent/`
2.  Create file: `project_root/src/retrieval_agent/__init__.py` (can be empty for now, or you can expose key classes later).

---

**Step 1: Implement `src/retrieval_agent/agent_config.py`**

This file will hold configurations specific to the iterative retriever.

```python
# File: src/retrieval_agent/agent_config.py

# --- Iteration Control ---
MAX_ITERATIONS = 3  # Default maximum number of search-analysis-refinement cycles
MIN_RELEVANCE_SCORE_FOR_FACTS = 0.6 # Minimum relevance score from vector search to consider a chunk

# --- LLM Configuration for Agent Tasks ---
# You can use the global OLLAMA_MODEL from src.config or define specific ones here
# For example, a more capable model for analysis/synthesis vs. a quicker one for decomposition.
# If using global, ensure src.config is imported.
# For now, let's assume we use the global OLLAMA_MODEL for simplicity.
# from src.config import OLLAMA_MODEL, OLLAMA_HOST

# --- Search Parameters ---
NUM_RESULTS_PER_SUB_QUERY = 5  # Number of documents to retrieve per vector search
CONTEXT_WINDOW_SIZE_FOR_SYNTHESIS = 8000 # Approx token limit for context sent to final synthesis

# --- Re-ranking ---
# If you implement LLM-based re-ranking, config for it might go here
ENABLE_LLM_RE_RANKING = False
NUM_CHUNKS_TO_RE_RANK = 10 # How many top chunks to re-rank using LLM

# --- Logging & Debugging ---
LOG_INTERMEDIATE_STEPS = True # For verbose logging during agent execution

# --- Tool Specific ---
# (No specific tool configs for now, VectorSearcher uses src.config)

```

---

**Step 2: Implement `src/retrieval_agent/agent_prompts.py`**

This file will contain all the prompt templates used by the agent's LLM calls.

```python
# File: src/retrieval_agent/agent_prompts.py

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# --- 1. Query Understanding & Decomposition Prompt ---
# Input: original_query, conversation_history (optional)
QUERY_UNDERSTANDING_SYSTEM_MESSAGE = """
You are an expert legal research assistant. Your task is to analyze a user's query about legal documents and break it down into a strategic search plan.
Focus on identifying:
1.  The core intent and information need.
2.  Key legal concepts, entities (people, organizations, specific document types mentioned), dates, or other critical terms.
3.  If the query is complex or multi-faceted, decompose it into 1 to 3 specific sub-queries that can be searched independently.
4.  Suggest any relevant keywords that should be emphasized in a search.
5.  Suggest potential metadata filters if applicable (e.g., document category like "Pleading", "Correspondence"; specific exhibit numbers if mentioned).

Respond ONLY with a JSON object matching the following Pydantic schema:
{json_schema}
"""

QUERY_UNDERSTANDING_HUMAN_MESSAGE = """
Original User Query:
{original_query}

{conversation_history_block}
Analyze this query and provide a search plan.
"""

# conversation_history_block will be formatted if history exists, e.g., "Previous turns:\n{history_string}"

# --- 2. Vector Search Query Formulation Prompt (if needed, often direct from sub-query) ---
# Sometimes the sub-query itself is good enough. If refinement is needed:
SEARCH_QUERY_REFINEMENT_SYSTEM_MESSAGE = """
You are an AI assistant that refines a sub-query for optimal vector search performance.
Given the sub-query and any relevant keywords, produce a concise and effective search string.
Focus on including terms most likely to match relevant passages in legal documents.
"""
SEARCH_QUERY_REFINEMENT_HUMAN_MESSAGE = """
Original Sub-Query: {sub_query}
Keywords: {keywords}

Refine this into an optimal vector search query string. Respond with only the query string.
"""

# --- 3. Retrieval Analysis & Fact Extraction Prompt ---
# Input: sub_query, retrieved_chunk_text
FACT_EXTRACTION_SYSTEM_MESSAGE = """
You are a meticulous legal analyst. Review the provided text chunk and determine its relevance to the given sub-query.
If relevant, extract the precise sentence(s) or fact(s) that directly address or support the sub-query.
Assess the relevance on a scale of High, Medium, Low, or Irrelevant.

Respond ONLY with a JSON object matching the following Pydantic schema:
{json_schema}
"""

FACT_EXTRACTION_HUMAN_MESSAGE = """
Sub-Query:
{sub_query}

Retrieved Text Chunk:
---
{retrieved_chunk_text}
---

Analyze the chunk for relevance and extract pertinent facts.
"""

# --- 4. Iteration Decision Prompt ---
# Input: original_query, retrieved_facts_summary, executed_queries, current_iteration, max_iterations
ITERATION_DECISION_SYSTEM_MESSAGE = """
You are a strategic legal research manager. Based on the original query, the facts retrieved so far, and the queries already executed:
1.  Determine if the current set of facts is sufficient to comprehensively answer the original query.
2.  If not, and if maximum iterations ({max_iterations}) have not been reached, decide if another search iteration is warranted.
3.  If another iteration is needed, formulate the next sub-query to explore unaddressed aspects, clarify points, or find complementary evidence. Also suggest any keywords or filters.
Your goal is to build a complete evidence base.

Respond ONLY with a JSON object matching the following Pydantic schema:
{json_schema}
"""

ITERATION_DECISION_HUMAN_MESSAGE = """
Original Query:
{original_query}

Summary of Facts Retrieved So Far:
{retrieved_facts_summary}

Queries Already Executed:
{executed_queries_list}

Current Iteration: {current_iteration} / {max_iterations}

Based on this, decide the next step and, if applicable, the next search parameters.
"""

# --- 5. Final Synthesis Prompt ---
# Input: original_query, all_relevant_facts (list of strings with source metadata)
FINAL_SYNTHESIS_SYSTEM_MESSAGE = """
You are a legal writing expert. Your task is to synthesize a clear, concise, and comprehensive answer to the original user query based *only* on the provided, retrieved facts.
Structure your answer logically.
For every piece of information or assertion you make, you MUST cite its source using the provided metadata (e.g., "[Filename: X.pdf, Exhibit: Y, Bates: SSSS-EEEE, Page: P]").
If the provided facts are insufficient to answer the query, state that clearly. Do not introduce outside information.
"""

FINAL_SYNTHESIS_HUMAN_MESSAGE = """
Original User Query:
{original_query}

Relevant Retrieved Facts (with sources):
---
{formatted_relevant_facts}
---

Synthesize an answer to the original query based *only* on these facts, citing each source.
"""

```

---

**Step 3: Implement `src/retrieval_agent/output_parsers.py`**

This file defines Pydantic models for structured LLM outputs and their Langchain parsers.

```python
# File: src/retrieval_agent/output_parsers.py

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser

# --- For Query Understanding (Step 1) ---
class QueryAnalysisResult(BaseModel):
    main_intent: str = Field(description="The core information the user is seeking.")
    sub_queries: List[str] = Field(description="Specific sub-queries to search for. Can be empty if original query is simple.")
    search_keywords: List[str] = Field(description="Key terms to emphasize in searches.")
    potential_filters: Optional[Dict[str, Any]] = Field(default=None, description="Suggested metadata filters (e.g., {'category': 'Pleading'}).")
    analysis_notes: str = Field(description="Brief notes on the search strategy.")

query_analysis_parser = PydanticOutputParser(pydantic_object=QueryAnalysisResult)

# --- For Fact Extraction (Step 2.C) ---
class ExtractedFact(BaseModel):
    is_relevant: bool = Field(description="Whether the chunk is relevant to the sub-query.")
    extracted_statement: Optional[str] = Field(default=None, description="The precise sentence(s) or fact(s) extracted if relevant.")
    relevance_score_assessment: str = Field(description="Assessed relevance: High, Medium, Low, or Irrelevant.") # Could be enum
    reasoning_for_relevance: Optional[str] = Field(default=None, description="Brief reasoning for the relevance assessment.")
    # We will add source_metadata programmatically after this LLM call

extracted_fact_parser = PydanticOutputParser(pydantic_object=ExtractedFact)

# --- For Iteration Decision (Step 3) ---
class IterationDecisionOutput(BaseModel):
    continue_iteration: bool = Field(description="Whether to perform another search iteration.")
    next_sub_query: Optional[str] = Field(default=None, description="The next sub-query to execute, if continuing.")
    next_keywords: Optional[List[str]] = Field(default_factory=list, description="Keywords for the next sub-query.")
    next_filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters for the next sub-query.")
    reasoning: str = Field(description="Reasoning for the decision and the next step formulation.")

iteration_decision_parser = PydanticOutputParser(pydantic_object=IterationDecisionOutput)

# --- For Final Answer (Step 4) ---
# StrOutputParser is often sufficient, but a Pydantic model can be used for more structure if needed.
class SynthesizedAnswer(BaseModel):
    answer_text: str = Field(description="The synthesized answer to the user's query.")
    cited_sources_summary: List[str] = Field(description="A list of unique sources cited in the answer.")

# synthesized_answer_parser = PydanticOutputParser(pydantic_object=SynthesizedAnswer) # Optional
```

---

**Step 4: Implement `src/retrieval_agent/agent_tools.py`**

This file defines the tools the agent can use, primarily for vector search.

```python
# File: src/retrieval_agent/agent_tools.py

from typing import List, Dict, Optional, Any
from langchain_core.tools import tool

# Import your existing VectorSearcher and its config
# Assuming vector_search.py has been refactored to use Langchain components as per Phase 2
from src.vector_search import VectorSearcher
from src.config import VECTOR_STORE_PATH # For default path

# Import agent-specific config if needed
from .agent_config import NUM_RESULTS_PER_SUB_QUERY

# Global instance or factory function for VectorSearcher to manage resources
# For simplicity in a CLI context, creating it per call is okay,
# but for an API, you'd manage its lifecycle.
_vector_searcher_instance = None

def get_vector_searcher():
    global _vector_searcher_instance
    if _vector_searcher_instance is None:
        # Ensure VECTOR_STORE_PATH is correctly resolved if it's relative
        # from project_root / config.DEFAULT_OUTPUT_DIR / "vector_store"
        # For now, assume it's an absolute path or correctly relative from where script runs
        _vector_searcher_instance = VectorSearcher(vector_store_path=str(VECTOR_STORE_PATH))
    return _vector_searcher_instance

@tool
def perform_vector_search(query_text: str, k_results: int = NUM_RESULTS_PER_SUB_QUERY, metadata_filters: Optional[Dict[str, Any]] = None) -> List[Dict]:
    """
    Searches the legal document vector store for passages relevant to the query_text.
    'query_text': The text to search for.
    'k_results': The number of results to return.
    'metadata_filters': A dictionary of metadata to filter by (e.g., {"category": "Pleading", "exhibit_number": 1}).
    Returns a list of search results, each with 'text', 'metadata', and 'relevance'.
    """
    searcher = get_vector_searcher()
    
    # Adapt filter keys if your VectorSearcher.search expects different names
    # The refactored VectorSearcher in the prompt expects: category, exhibit_number
    # So, ensure metadata_filters aligns with that or adapt here.
    # Example: if agent produces {"custom_filter_key": val}, map to {"exhibit_number": val}
    
    formatted_filters = {}
    if metadata_filters:
        if "category" in metadata_filters:
            formatted_filters["category"] = metadata_filters["category"]
        if "exhibit_number" in metadata_filters:
             try:
                formatted_filters["exhibit_number"] = int(metadata_filters["exhibit_number"])
             except ValueError:
                # Handle cases where exhibit_number might not be a valid int from LLM
                pass 
        # Add more filter mappings as needed

    search_results = searcher.search(
        query=query_text,
        n_results=k_results,
        # Pass filters directly if VectorSearcher.search() supports **kwargs for them
        # or pass them as specific arguments if defined.
        # Based on your vector_search.py, it takes category and exhibit_number as named args
        category=formatted_filters.get("category"),
        exhibit_number=formatted_filters.get("exhibit_number")
    )
    # Ensure output is serializable and contains what the agent expects
    # Your current VectorSearcher.search returns list of dicts:
    # {"text": ..., "relevance": ..., "filename": ..., "category": ..., ...} which is good.
    return search_results

# Potential future tool:
# @tool
# def postgres_keyword_search(keywords: List[str], category_filter: Optional[str] = None) -> List[Dict]:
#     """Performs a keyword search against the PostgreSQL database."""
#     # ... implementation using src.db_storage.PostgresStorage ...
#     pass

```

---

**Step 5: Implement `src/retrieval_agent/main_retriever.py`**

This is the core of the iterative retriever. It will use LCEL to chain the steps.

```python
# File: src/retrieval_agent/main_retriever.py

import logging
from typing import Dict, List, Any, Optional

from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama # Or your preferred LLM wrapper

# Local imports
from . import agent_config, agent_prompts, output_parsers, agent_tools
from src.config import OLLAMA_MODEL, OLLAMA_HOST # Or use agent_config for models
from src.llm_handler import LLMCategorizer # Could reuse or define agent-specific LLM instances

logger = logging.getLogger(__name__)

class IterativeRetrieverAgent:
    def __init__(self, max_iterations: int = agent_config.MAX_ITERATIONS):
        self.max_iterations = max_iterations
        
        # Initialize LLM (can be shared or specific for agent tasks)
        # For simplicity, using a new ChatOllama instance here.
        # You could also pass an LLM instance from llm_handler.
        self.llm = ChatOllama(
            model=OLLAMA_MODEL, # Or a model from agent_config
            base_url=OLLAMA_HOST,
            temperature=0.1 # Agent tasks often benefit from low temperature for predictability
        )

        # --- Define LCEL Sub-Chains ---

        # 1. Query Understanding Chain
        _query_understanding_prompt = agent_prompts.ChatPromptTemplate.from_messages([
            ("system", agent_prompts.QUERY_UNDERSTANDING_SYSTEM_MESSAGE.format(json_schema=output_parsers.QueryAnalysisResult.model_json_schema())),
            ("human", agent_prompts.QUERY_UNDERSTANDING_HUMAN_MESSAGE)
        ])
        self.query_understanding_chain = (
            _query_understanding_prompt 
            | self.llm 
            | output_parsers.query_analysis_parser
        ).with_config({"run_name": "QueryUnderstanding"})

        # 2. Fact Extraction & Relevance Chain (per chunk)
        _fact_extraction_prompt = agent_prompts.ChatPromptTemplate.from_messages([
            ("system", agent_prompts.FACT_EXTRACTION_SYSTEM_MESSAGE.format(json_schema=output_parsers.ExtractedFact.model_json_schema())),
            ("human", agent_prompts.FACT_EXTRACTION_HUMAN_MESSAGE)
        ])
        self.fact_extraction_chain = (
            _fact_extraction_prompt
            | self.llm
            | output_parsers.extracted_fact_parser
        ).with_config({"run_name": "FactExtractionPerChunk"})

        # 3. Iteration Decision Chain
        _iteration_decision_prompt = agent_prompts.ChatPromptTemplate.from_messages([
            ("system", agent_prompts.ITERATION_DECISION_SYSTEM_MESSAGE.format(max_iterations=self.max_iterations)),
            ("human", agent_prompts.ITERATION_DECISION_HUMAN_MESSAGE)
        ])
        self.iteration_decision_chain = (
            _iteration_decision_prompt
            | self.llm
            | output_parsers.iteration_decision_parser
        ).with_config({"run_name": "IterationDecision"})
        
        # 4. Final Synthesis Chain
        _final_synthesis_prompt = agent_prompts.ChatPromptTemplate.from_template(
            agent_prompts.FINAL_SYNTHESIS_SYSTEM_MESSAGE + "\n\n" + agent_prompts.FINAL_SYNTHESIS_HUMAN_MESSAGE
        )
        self.final_synthesis_chain = (
            _final_synthesis_prompt
            | self.llm
            | StrOutputParser() # Or a Pydantic parser for SynthesizedAnswer
        ).with_config({"run_name": "FinalSynthesis"})

    def _format_retrieved_facts_for_llm(self, facts: List[Dict]) -> str:
        if not facts:
            return "No relevant facts retrieved yet."
        
        formatted = []
        for i, fact_item in enumerate(facts):
            # fact_item is expected to be what ExtractedFact (plus source) produces
            source_info = fact_item.get("source_metadata", {})
            filename = source_info.get("filename", "Unknown")
            exhibit = source_info.get("exhibit_number", "N/A")
            bates_start = source_info.get("bates_start", "N/A")
            bates_end = source_info.get("bates_end", "N/A")
            page = source_info.get("page", "N/A")
            
            cite = f"[Filename: {filename}, Exhibit: {exhibit}, Bates: {bates_start}-{bates_end}, Page: {page}]"
            formatted.append(f"Fact {i+1}: {fact_item.get('extracted_statement', 'N/A')}\nSource: {cite}\nRelevance: {fact_item.get('relevance_score_assessment', 'N/A')}")
        return "\n\n".join(formatted)

    def _process_retrieved_chunk(self, chunk_data: Dict, sub_query: str) -> Optional[Dict]:
        """Processes a single retrieved chunk using fact_extraction_chain."""
        try:
            # chunk_data is one item from vector_search_tool output: {"text": ..., "metadata": ..., "relevance": ...}
            if not chunk_data or not chunk_data.get("text"):
                return None

            # Filter by initial relevance score from vector DB
            if chunk_data.get("relevance", 0) < agent_config.MIN_RELEVANCE_SCORE_FOR_FACTS:
                 logger.info(f"Chunk skipped due to low vector relevance: {chunk_data.get('relevance')}")
                 return None

            extracted_fact_obj: output_parsers.ExtractedFact = self.fact_extraction_chain.invoke({
                "sub_query": sub_query,
                "retrieved_chunk_text": chunk_data["text"]
            })

            if extracted_fact_obj.is_relevant and extracted_fact_obj.extracted_statement:
                return {
                    "extracted_statement": extracted_fact_obj.extracted_statement,
                    "relevance_score_assessment": extracted_fact_obj.relevance_score_assessment,
                    "source_metadata": chunk_data["metadata"], # Carry over metadata from vector search
                    "original_chunk_text": chunk_data["text"] # For context if needed
                }
        except Exception as e:
            logger.error(f"Error processing chunk: {e}", exc_info=True)
        return None

    def invoke(self, original_query: str, run_config: Optional[Dict] = None) -> str:
        """
        Main invocation method for the iterative retriever.
        Manages the iterative loop.
        """
        if run_config is None: # For LangSmith tracing
            run_config = {"metadata": {"original_query": original_query}, "name": f"IterativeRetrieve-{original_query[:30]}"}
        
        current_iteration = 0
        accumulated_facts: List[Dict] = [] # List of fact dicts (from _process_retrieved_chunk)
        executed_queries: List[str] = []
        
        # Initial Query Understanding
        logger.info(f"Iteration {current_iteration}: Understanding original query.")
        query_analysis: output_parsers.QueryAnalysisResult = self.query_understanding_chain.invoke(
            {"original_query": original_query, "conversation_history_block": ""} # Add history if implementing
        , config=run_config)
        
        logger.info(f"Query Analysis: {query_analysis.model_dump_json(indent=2)}")
        
        # Prepare initial sub-queries or use main intent
        sub_queries_to_process = query_analysis.sub_queries
        if not sub_queries_to_process and query_analysis.main_intent:
            sub_queries_to_process = [query_analysis.main_intent] # Fallback to main intent
        
        # Use keywords and filters from initial analysis for the first round of sub-queries
        current_keywords = query_analysis.search_keywords
        current_filters = query_analysis.potential_filters

        while current_iteration < self.max_iterations:
            current_iteration += 1
            logger.info(f"--- Starting Iteration {current_iteration} ---")

            if not sub_queries_to_process:
                logger.info("No more sub-queries to process in this iteration. Moving to decision.")
                # We might still go to iteration decision to see if we are done overall
            
            # Process current batch of sub-queries
            for sub_query_idx, sub_query in enumerate(sub_queries_to_process):
                if not sub_query: continue
                logger.info(f"Processing Sub-query {sub_query_idx+1}/{len(sub_queries_to_process)}: '{sub_query}'")
                executed_queries.append(sub_query)

                # A. Formulate search (can be simple passthrough or an LLM call if refinement is needed)
                # For now, use sub_query directly. Keywords/filters from query_analysis or iteration_decision.
                vector_query_text = sub_query # Or refine with SEARCH_QUERY_REFINEMENT_CHAIN
                
                # B. Execute Vector Search Tool
                logger.info(f"Executing vector search for: '{vector_query_text}' with filters: {current_filters}")
                try:
                    tool_input = {
                        "query_text": vector_query_text, 
                        "k_results": agent_config.NUM_RESULTS_PER_SUB_QUERY,
                        "metadata_filters": current_filters
                    }
                    search_results: List[Dict] = agent_tools.perform_vector_search.invoke(tool_input, config=run_config)
                    logger.info(f"Retrieved {len(search_results)} chunks.")
                except Exception as e:
                    logger.error(f"Vector search tool failed for sub_query '{sub_query}': {e}", exc_info=True)
                    search_results = []

                # C. Retrieval Analysis & Fact Extraction for each chunk
                newly_extracted_facts_for_sub_query = []
                for chunk in search_results:
                    processed_chunk = self._process_retrieved_chunk(chunk, sub_query)
                    if processed_chunk:
                        newly_extracted_facts_for_sub_query.append(processed_chunk)
                
                if newly_extracted_facts_for_sub_query:
                    logger.info(f"Extracted {len(newly_extracted_facts_for_sub_query)} relevant facts for sub-query '{sub_query}'.")
                    accumulated_facts.extend(newly_extracted_facts_for_sub_query)
                else:
                    logger.info(f"No new relevant facts extracted for sub-query '{sub_query}'.")
            
            # Clear sub_queries for next iteration decision
            sub_queries_to_process = []

            # D. Iteration Decision
            if current_iteration >= self.max_iterations:
                logger.info("Max iterations reached. Moving to synthesis.")
                break

            logger.info("Making iteration decision...")
            facts_summary_for_decision = self._format_retrieved_facts_for_llm(accumulated_facts)
            
            decision_input = {
                "original_query": original_query,
                "retrieved_facts_summary": facts_summary_for_decision[:agent_config.CONTEXT_WINDOW_SIZE_FOR_SYNTHESIS//2], # Avoid overly long prompts
                "executed_queries_list": "\n".join(f"- {q}" for q in executed_queries),
                "current_iteration": current_iteration,
                "max_iterations": self.max_iterations
            }
            iteration_decision: output_parsers.IterationDecisionOutput = self.iteration_decision_chain.invoke(decision_input, config=run_config)
            logger.info(f"Iteration Decision: {iteration_decision.model_dump_json(indent=2)}")

            if iteration_decision.continue_iteration and iteration_decision.next_sub_query:
                sub_queries_to_process = [iteration_decision.next_sub_query] # Assuming one refined sub_query for next step for now
                # Could be extended to handle multiple next_sub_queries from LLM
                current_keywords = iteration_decision.next_keywords
                current_filters = iteration_decision.next_filters
                logger.info(f"Continuing to next iteration with sub-query: {iteration_decision.next_sub_query}")
            else:
                logger.info("Decision to stop iteration. Moving to synthesis.")
                break
        
        # --- Final Synthesis ---
        logger.info("Synthesizing final answer...")
        if not accumulated_facts:
            logger.warning("No relevant facts accumulated for synthesis.")
            return "After iterative searching, no relevant facts were found to answer the query."

        formatted_facts_for_synthesis = self._format_retrieved_facts_for_llm(accumulated_facts)
        
        # Ensure context for synthesis is not too large
        if len(formatted_facts_for_synthesis) > agent_config.CONTEXT_WINDOW_SIZE_FOR_SYNTHESIS:
            logger.warning(f"Accumulated facts context is too large ({len(formatted_facts_for_synthesis)} chars), truncating for synthesis.")
            # A more sophisticated truncation or summarization of facts might be needed here.
            # For now, simple truncation.
            formatted_facts_for_synthesis = formatted_facts_for_synthesis[:agent_config.CONTEXT_WINDOW_SIZE_FOR_SYNTHESIS]

        synthesis_input = {
            "original_query": original_query,
            "formatted_relevant_facts": formatted_facts_for_synthesis
        }
        final_answer = self.final_synthesis_chain.invoke(synthesis_input, config=run_config)
        
        logger.info("Iterative retrieval complete.")
        return final_answer

```

---

**Step 6: Implement `src/retrieval_agent/cli.py` (for testing)**

This is a simple command-line interface to run the `IterativeRetrieverAgent`.

```python
# File: src/retrieval_agent/cli.py

import argparse
import logging
import sys
from pathlib import Path

# Ensure src directory is in Python path for imports
project_root = Path(__file__).resolve().parent.parent.parent # Adjust if cli.py is in src/retrieval_agent/
sys.path.insert(0, str(project_root))

from src.retrieval_agent.main_retriever import IterativeRetrieverAgent
from src.utils import setup_logging
# Import LangSmith config if needed for explicit setup, though env vars are preferred
# from src.config import LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT (example)
# import os # if setting env vars here

logger = logging.getLogger(__name__)

def main():
    setup_logging(level=logging.INFO) # Set to DEBUG for more verbose Langchain logs

    # --- LangSmith Configuration (Example of explicit setup if not relying purely on .env) ---
    # Ensure these are set in your .env file or environment for LangSmith to work
    # os.environ["LANGCHAIN_TRACING_V2"] = "true"
    # os.environ["LANGCHAIN_API_KEY"] = "YOUR_LANGSMITH_API_KEY"
    # os.environ["LANGCHAIN_PROJECT"] = "BatesNumbering-RetrievalAgent" # Choose your project name
    # logger.info(f"LangSmith Tracing V2: {os.getenv('LANGCHAIN_TRACING_V2')}")
    # logger.info(f"LangSmith Project: {os.getenv('LANGCHAIN_PROJECT')}")
    # --- End LangSmith Configuration ---


    parser = argparse.ArgumentParser(description="Iterative Retriever Agent CLI")
    parser.add_argument("query", type=str, help="The legal query to process.")
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=None, # Will use agent_config default
        help="Maximum number of retrieval iterations."
    )
    
    args = parser.parse_args()

    logger.info(f"Initializing IterativeRetrieverAgent...")
    if args.max_iterations:
        agent = IterativeRetrieverAgent(max_iterations=args.max_iterations)
    else:
        agent = IterativeRetrieverAgent() # Uses default from agent_config
    
    logger.info(f"Invoking agent with query: '{args.query}'")
    
    # For LangSmith, define run-specific metadata (optional but good)
    run_config = {
        "metadata": {
            "user_query": args.query,
            "cli_invocation": True
        },
        "tags": ["cli_test", "iterative_retrieval"],
        "name": f"CLIQuery-{args.query[:50].replace(' ', '_')}" # A descriptive run name
    }

    try:
        answer = agent.invoke(args.query, run_config=run_config)
        
        print("\n" + "="*20 + " Synthesized Answer " + "="*20)
        print(answer)
        print("="*60)

    except Exception as e:
        logger.error(f"An error occurred during agent invocation: {e}", exc_info=True)
        print(f"\nERROR: {e}")

if __name__ == "__main__":
    main()
```

---

**Final Steps:**

1.  **Review Imports:** Double-check all imports in each new file. Ensure they correctly reference other parts of your `src` module or Langchain libraries.
2.  **Test Incrementally:**
    *   After implementing `agent_tools.py`, you can test the `perform_vector_search` tool independently with a simple query.
    *   Once `main_retriever.py` has its sub-chains defined, you can test them individually (e.g., `agent.query_understanding_chain.invoke(...)`) before running the full `agent.invoke()`.
3.  **LangSmith Setup:** Ensure your LangSmith environment variables (`LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`) are correctly set in your `.env` file or shell environment *before* running `src/retrieval_agent/cli.py`.
4.  **Run with Production Data:** Use the `src/retrieval_agent/cli.py` to execute queries against your vector store populated from `/Users/josephott/Documents/bates_number_demo/input_documents`.
5.  **Document `context_X.md`:** Log your implementation steps, any challenges, and the results of your tests using the criteria and plan outlined previously.

This detailed step-by-step guide should provide a solid foundation for implementing the iterative retriever. The `main_retriever.py` and its `IterativeRetrieverAgent.invoke` method will be the most complex due to the iterative logic, but breaking it down into LCEL sub-chains for each distinct cognitive step should make it manageable.