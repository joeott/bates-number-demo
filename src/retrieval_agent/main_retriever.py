# File: src/retrieval_agent/main_retriever.py
"""
Core iterative retrieval agent implementation using LangChain Expression Language (LCEL).

This module orchestrates the multi-step retrieval process, managing iterations
and state to build comprehensive answers to complex legal queries.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys

from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableBranch
from langchain_core.output_parsers import StrOutputParser
try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama  # Fallback to community version

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Local imports
from . import agent_config, agent_prompts, output_parsers, agent_tools
from src.config import OLLAMA_MODEL, OLLAMA_HOST  # Or use agent_config for models
from src.llm_handler import LLMCategorizer  # Could reuse or define agent-specific LLM instances

logger = logging.getLogger(__name__)

class IterativeRetrieverAgent:
    """
    An agent that iteratively searches and analyzes legal documents to answer complex queries.
    
    The agent follows this process:
    1. Understand and decompose the query
    2. Execute targeted searches
    3. Extract and analyze relevant facts
    4. Decide if more information is needed
    5. Synthesize a comprehensive answer
    """
    
    def __init__(self, max_iterations: int = agent_config.MAX_ITERATIONS):
        """
        Initialize the iterative retriever agent.
        
        Args:
            max_iterations: Maximum number of search-analysis cycles to perform.
        """
        self.max_iterations = max_iterations
        
        # Initialize LLM (can be shared or specific for agent tasks)
        # For simplicity, using a new ChatOllama instance here.
        # You could also pass an LLM instance from llm_handler.
        self.llm = ChatOllama(
            model=OLLAMA_MODEL,  # Or a model from agent_config
            base_url=OLLAMA_HOST,
            temperature=agent_config.AGENT_LLM_TEMPERATURE  # Low temperature for predictability
        )
        
        logger.info(f"Initialized IterativeRetrieverAgent with model: {OLLAMA_MODEL}, max_iterations: {max_iterations}")

        # --- Define LCEL Sub-Chains ---
        
        # 1. Query Understanding Chain
        self._init_query_understanding_chain()
        
        # 2. Fact Extraction & Relevance Chain (per chunk)
        self._init_fact_extraction_chain()
        
        # 3. Iteration Decision Chain
        self._init_iteration_decision_chain()
        
        # 4. Final Synthesis Chain
        self._init_final_synthesis_chain()
    
    def _init_query_understanding_chain(self):
        """Initialize the query understanding chain."""
        # Create the prompt
        _query_understanding_prompt = agent_prompts.ChatPromptTemplate.from_messages([
            ("system", agent_prompts.QUERY_UNDERSTANDING_SYSTEM_MESSAGE),
            ("human", agent_prompts.QUERY_UNDERSTANDING_HUMAN_MESSAGE)
        ])
        
        self.query_understanding_chain = (
            _query_understanding_prompt 
            | self.llm 
            | output_parsers.query_analysis_parser
        ).with_config({"run_name": "QueryUnderstanding"})
    
    def _init_fact_extraction_chain(self):
        """Initialize the fact extraction chain."""
        # Create the prompt
        _fact_extraction_prompt = agent_prompts.ChatPromptTemplate.from_messages([
            ("system", agent_prompts.FACT_EXTRACTION_SYSTEM_MESSAGE),
            ("human", agent_prompts.FACT_EXTRACTION_HUMAN_MESSAGE)
        ])
        
        self.fact_extraction_chain = (
            _fact_extraction_prompt
            | self.llm
            | output_parsers.extracted_fact_parser
        ).with_config({"run_name": "FactExtractionPerChunk"})
    
    def _init_iteration_decision_chain(self):
        """Initialize the iteration decision chain."""
        # Create the prompt with max_iterations
        _iteration_decision_prompt = agent_prompts.ChatPromptTemplate.from_messages([
            ("system", agent_prompts.ITERATION_DECISION_SYSTEM_MESSAGE),
            ("human", agent_prompts.ITERATION_DECISION_HUMAN_MESSAGE)
        ])
        _iteration_decision_prompt = _iteration_decision_prompt.partial(
            max_iterations=self.max_iterations
        )
        
        self.iteration_decision_chain = (
            _iteration_decision_prompt
            | self.llm
            | output_parsers.iteration_decision_parser
        ).with_config({"run_name": "IterationDecision"})
    
    def _init_final_synthesis_chain(self):
        """Initialize the final synthesis chain."""
        _final_synthesis_prompt = agent_prompts.ChatPromptTemplate.from_template(
            agent_prompts.FINAL_SYNTHESIS_SYSTEM_MESSAGE + "\n\n" + agent_prompts.FINAL_SYNTHESIS_HUMAN_MESSAGE
        )
        
        self.final_synthesis_chain = (
            _final_synthesis_prompt
            | self.llm
            | StrOutputParser()  # Or use synthesized_answer_parser for structured output
        ).with_config({"run_name": "FinalSynthesis"})
    
    def _format_retrieved_facts_for_llm(self, facts: List[Dict]) -> str:
        """
        Format retrieved facts for presentation to the LLM.
        
        Args:
            facts: List of fact dictionaries with extracted statements and metadata.
        
        Returns:
            Formatted string representation of the facts.
        """
        if not facts:
            return "No relevant facts retrieved yet."
        
        formatted = []
        for i, fact_item in enumerate(facts):
            # fact_item is expected to have extracted_statement and source_metadata
            source_info = fact_item.get("source_metadata", {})
            filename = source_info.get("filename", "Unknown")
            exhibit = source_info.get("exhibit_number", "N/A")
            bates_start = source_info.get("bates_start", "N/A")
            bates_end = source_info.get("bates_end", "N/A")
            page = source_info.get("page", "N/A")
            
            cite = f"[Filename: {filename}, Exhibit: {exhibit}, Bates: {bates_start}-{bates_end}, Page: {page}]"
            formatted.append(
                f"Fact {i+1}: {fact_item.get('extracted_statement', 'N/A')}\n"
                f"Source: {cite}\n"
                f"Relevance: {fact_item.get('relevance_score_assessment', 'N/A')}"
            )
        
        return "\n\n".join(formatted)
    
    def _process_retrieved_chunk(self, chunk_data: Dict, sub_query: str) -> Optional[Dict]:
        """
        Process a single retrieved chunk using the fact extraction chain.
        
        Args:
            chunk_data: Dictionary containing chunk text and metadata from vector search.
            sub_query: The sub-query this chunk is being evaluated against.
        
        Returns:
            Dictionary with extracted fact and metadata if relevant, None otherwise.
        """
        try:
            # chunk_data is one item from vector_search_tool output
            if not chunk_data or not chunk_data.get("text"):
                return None
            
            # Filter by initial relevance score from vector DB
            if chunk_data.get("relevance", 0) < agent_config.MIN_RELEVANCE_SCORE_FOR_FACTS:
                logger.debug(f"Chunk skipped due to low vector relevance: {chunk_data.get('relevance')}")
                return None
            
            # Truncate chunk if too long
            chunk_text = chunk_data["text"]
            if len(chunk_text) > agent_config.MAX_CHUNK_LENGTH:
                logger.debug(f"Truncating chunk from {len(chunk_text)} to {agent_config.MAX_CHUNK_LENGTH} chars")
                chunk_text = chunk_text[:agent_config.MAX_CHUNK_LENGTH] + "..."
            
            # Extract facts using LLM
            extracted_fact_obj: output_parsers.ExtractedFact = self.fact_extraction_chain.invoke({
                "sub_query": sub_query,
                "retrieved_chunk_text": chunk_text
            })
            
            # Only return if relevant and has extracted statement
            if extracted_fact_obj.is_relevant and extracted_fact_obj.extracted_statement:
                return {
                    "extracted_statement": extracted_fact_obj.extracted_statement,
                    "relevance_score_assessment": extracted_fact_obj.relevance_score_assessment,
                    "source_metadata": chunk_data.get("metadata", {}),  # Carry over metadata
                    "original_chunk_text": chunk_data["text"]  # Keep original for context
                }
        except Exception as e:
            logger.error(f"Error processing chunk: {e}", exc_info=True)
        
        return None
    
    def invoke(self, original_query: str, run_config: Optional[Dict] = None) -> str:
        """
        Main invocation method for the iterative retriever.
        
        Args:
            original_query: The user's legal query to answer.
            run_config: Optional configuration for LangSmith tracing.
        
        Returns:
            Synthesized answer with citations.
        """
        if run_config is None:  # For LangSmith tracing
            run_config = {
                "metadata": {"original_query": original_query},
                "tags": ["iterative_retrieval"],
                "run_name": f"IterativeRetrieve-{original_query[:30]}"
            }
        
        current_iteration = 0
        accumulated_facts: List[Dict] = []  # List of fact dicts
        executed_queries: List[str] = []
        
        # Initial Query Understanding
        logger.info(f"Starting iterative retrieval for query: {original_query}")
        logger.info(f"Iteration {current_iteration}: Understanding original query.")
        
        try:
            query_analysis: output_parsers.QueryAnalysisResult = self.query_understanding_chain.invoke({
                "original_query": original_query,
                "conversation_history_block": ""  # Add history if implementing multi-turn
            }, config=run_config)
            
            logger.info(f"Query Analysis Complete - Main Intent: {query_analysis.main_intent}")
            logger.debug(f"Sub-queries: {query_analysis.sub_queries}")
            logger.debug(f"Keywords: {query_analysis.search_keywords}")
            logger.debug(f"Filters: {query_analysis.potential_filters}")
            
        except Exception as e:
            logger.error(f"Error in query understanding: {e}", exc_info=True)
            return "I encountered an error understanding your query. Please try rephrasing it."
        
        # Prepare initial sub-queries or use main intent
        sub_queries_to_process = query_analysis.sub_queries
        if not sub_queries_to_process and query_analysis.main_intent:
            sub_queries_to_process = [query_analysis.main_intent]  # Fallback to main intent
        
        # Use keywords and filters from initial analysis
        current_keywords = query_analysis.search_keywords
        current_filters = query_analysis.potential_filters
        
        # Main iteration loop
        while current_iteration < self.max_iterations:
            current_iteration += 1
            logger.info(f"--- Starting Iteration {current_iteration}/{self.max_iterations} ---")
            
            if not sub_queries_to_process:
                logger.info("No more sub-queries to process. Moving to iteration decision.")
            
            # Process current batch of sub-queries
            for sub_query_idx, sub_query in enumerate(sub_queries_to_process):
                if not sub_query:
                    continue
                    
                logger.info(f"Processing Sub-query {sub_query_idx+1}/{len(sub_queries_to_process)}: '{sub_query}'")
                executed_queries.append(sub_query)
                
                # Execute Vector Search Tool
                logger.info(f"Executing vector search with filters: {current_filters}")
                try:
                    tool_input = {
                        "query_text": sub_query,
                        "k_results": agent_config.NUM_RESULTS_PER_SUB_QUERY,
                        "metadata_filters": current_filters
                    }
                    search_results: List[Dict] = agent_tools.perform_vector_search.invoke(
                        tool_input, 
                        config=run_config
                    )
                    logger.info(f"Retrieved {len(search_results)} chunks.")
                except Exception as e:
                    logger.error(f"Vector search failed for sub_query '{sub_query}': {e}", exc_info=True)
                    search_results = []
                
                # Process each chunk and extract facts
                newly_extracted_facts = []
                for chunk in search_results:
                    processed_chunk = self._process_retrieved_chunk(chunk, sub_query)
                    if processed_chunk:
                        newly_extracted_facts.append(processed_chunk)
                
                if newly_extracted_facts:
                    logger.info(f"Extracted {len(newly_extracted_facts)} relevant facts.")
                    accumulated_facts.extend(newly_extracted_facts)
                else:
                    logger.info(f"No relevant facts extracted for sub-query '{sub_query}'.")
            
            # Clear sub_queries for next iteration
            sub_queries_to_process = []
            
            # Check if we've reached max iterations
            if current_iteration >= self.max_iterations:
                logger.info("Maximum iterations reached. Moving to synthesis.")
                break
            
            # Iteration Decision
            logger.info("Making iteration decision...")
            facts_summary = self._format_retrieved_facts_for_llm(accumulated_facts)
            
            # Truncate facts summary if too long for decision prompt
            max_decision_context = agent_config.CONTEXT_WINDOW_SIZE_FOR_SYNTHESIS // 2
            if len(facts_summary) > max_decision_context:
                facts_summary = facts_summary[:max_decision_context] + "\n...[truncated]"
            
            try:
                decision_input = {
                    "original_query": original_query,
                    "retrieved_facts_summary": facts_summary,
                    "executed_queries_list": "\n".join(f"- {q}" for q in executed_queries),
                    "current_iteration": current_iteration,
                    "max_iterations": self.max_iterations
                }
                
                iteration_decision: output_parsers.IterationDecisionOutput = self.iteration_decision_chain.invoke(
                    decision_input, 
                    config=run_config
                )
                
                logger.info(f"Iteration Decision: Continue={iteration_decision.continue_iteration}")
                logger.debug(f"Reasoning: {iteration_decision.reasoning}")
                
                # Validate and act on decision
                if iteration_decision.continue_iteration and iteration_decision.next_sub_query:
                    sub_queries_to_process = [iteration_decision.next_sub_query]
                    current_keywords = iteration_decision.next_keywords or current_keywords
                    current_filters = iteration_decision.next_filters or current_filters
                    logger.info(f"Continuing with: {iteration_decision.next_sub_query}")
                else:
                    logger.info("Decision to stop iteration. Moving to synthesis.")
                    break
                    
            except Exception as e:
                logger.error(f"Error in iteration decision: {e}", exc_info=True)
                logger.info("Error in decision process. Moving to synthesis with current facts.")
                break
        
        # --- Final Synthesis ---
        logger.info("Synthesizing final answer...")
        
        if not accumulated_facts:
            logger.warning("No relevant facts accumulated for synthesis.")
            return (
                f"After searching through the legal documents, I was unable to find relevant information "
                f"to answer your query: \"{original_query}\". This could mean the information is not "
                f"present in the available documents, or my search strategy needs refinement."
            )
        
        # Format facts for synthesis
        formatted_facts = self._format_retrieved_facts_for_llm(accumulated_facts)
        
        # Ensure context isn't too large
        if len(formatted_facts) > agent_config.CONTEXT_WINDOW_SIZE_FOR_SYNTHESIS:
            logger.warning(f"Facts context too large ({len(formatted_facts)} chars), truncating.")
            formatted_facts = formatted_facts[:agent_config.CONTEXT_WINDOW_SIZE_FOR_SYNTHESIS]
        
        try:
            synthesis_input = {
                "original_query": original_query,
                "formatted_relevant_facts": formatted_facts
            }
            final_answer = self.final_synthesis_chain.invoke(synthesis_input, config=run_config)
            
            logger.info("Iterative retrieval complete.")
            return final_answer
            
        except Exception as e:
            logger.error(f"Error in final synthesis: {e}", exc_info=True)
            return (
                f"I found {len(accumulated_facts)} relevant facts but encountered an error "
                f"synthesizing the final answer. The facts have been logged for manual review."
            )