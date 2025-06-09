# File: src/retrieval_agent/agent_prompts.py
"""
Prompt templates for the Iterative Retrieval Agent.

This module contains all the prompt templates used by the agent's LLM calls,
specifically designed for legal document analysis and retrieval.
"""

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# --- 1. Query Understanding & Decomposition Prompt ---
# Input: original_query, conversation_history (optional)
QUERY_UNDERSTANDING_SYSTEM_MESSAGE = """
You are an expert legal research assistant. Your task is to analyze a user's query about legal documents and break it down into a strategic search plan.
Focus on identifying:
1. The core intent and information need.
2. Key legal concepts, entities (people, organizations, specific document types mentioned), dates, or other critical terms.
3. If the query is complex or multi-faceted, decompose it into 1 to 3 specific sub-queries that can be searched independently.
4. Suggest any relevant keywords that should be emphasized in a search.
5. Suggest potential metadata filters if applicable (e.g., document category like "Pleading", "Correspondence"; specific exhibit numbers if mentioned).

Respond ONLY with a JSON object with the following fields:
- main_intent: (string) The core information the user is seeking
- sub_queries: (array of strings) Specific sub-queries to search for, can be empty if query is simple
- search_keywords: (array of strings) Key terms to emphasize in searches
- potential_filters: (object or null) Suggested metadata filters like {{"category": "Pleading"}}
- analysis_notes: (string) Brief notes on the search strategy

Example response:
{{
  "main_intent": "Find contract price information",
  "sub_queries": ["contract price Brentwood Glass", "Glass Installation Contract amount"],
  "search_keywords": ["contract", "price", "Brentwood", "Glass"],
  "potential_filters": {{"category": "Pleading"}},
  "analysis_notes": "Searching for contract price specifically related to Brentwood Glass case"
}}
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

Respond ONLY with a JSON object with the following fields:
- is_relevant: (boolean) Whether the chunk is relevant to the sub-query
- extracted_statement: (string or null) The precise sentence(s) or fact(s) extracted if relevant
- relevance_score_assessment: (string) High, Medium, Low, or Irrelevant
- reasoning_for_relevance: (string or null) Brief reasoning for the relevance assessment

Example response:
{{
  "is_relevant": true,
  "extracted_statement": "The contract price was $150,000.",
  "relevance_score_assessment": "High",
  "reasoning_for_relevance": "Directly states the contract price requested in the query"
}}
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
1. Determine if the current set of facts is sufficient to comprehensively answer the original query.
2. If not, and if maximum iterations ({{max_iterations}}) have not been reached, decide if another search iteration is warranted.
3. If another iteration is needed, formulate the next sub-query to explore unaddressed aspects, clarify points, or find complementary evidence. Also suggest any keywords or filters.
Your goal is to build a complete evidence base.

Respond ONLY with a JSON object with the following fields:
- continue_iteration: (boolean) Whether to perform another search iteration
- next_sub_query: (string or null) The next sub-query to execute, if continuing
- next_keywords: (array of strings) Keywords for the next sub-query
- next_filters: (object or null) Metadata filters for the next sub-query
- reasoning: (string) Reasoning for the decision and next step formulation

Example response:
{{
  "continue_iteration": true,
  "next_sub_query": "damages claimed Brentwood Glass",
  "next_keywords": ["damages", "claimed", "amount"],
  "next_filters": {{"category": "Pleading"}},
  "reasoning": "Found contract price but need to search for damages to complete the financial picture"
}}
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

# --- 6. Legal Concept-Specific Prompts (for future customization) ---
# These can be swapped in based on specific legal concepts being investigated

BREACH_OF_CONTRACT_SYSTEM_MESSAGE = """
You are a legal research assistant specializing in contract law. When analyzing queries and documents, 
focus on identifying elements of breach of contract:
1. Existence of a valid contract (offer, acceptance, consideration)
2. Performance by the plaintiff
3. Breach by the defendant
4. Damages resulting from the breach

Structure your analysis and search strategy around these elements.
"""

NEGLIGENCE_SYSTEM_MESSAGE = """
You are a legal research assistant specializing in tort law. When analyzing queries and documents,
focus on identifying elements of negligence:
1. Duty of care owed
2. Breach of that duty
3. Causation (both factual and proximate)
4. Damages

Structure your analysis and search strategy around these elements.
"""

# --- Helper Functions to Create Prompt Templates ---

def get_query_understanding_prompt():
    """Returns the query understanding chat prompt template."""
    return ChatPromptTemplate.from_messages([
        ("system", QUERY_UNDERSTANDING_SYSTEM_MESSAGE),
        ("human", QUERY_UNDERSTANDING_HUMAN_MESSAGE)
    ])

def get_fact_extraction_prompt():
    """Returns the fact extraction chat prompt template."""
    return ChatPromptTemplate.from_messages([
        ("system", FACT_EXTRACTION_SYSTEM_MESSAGE),
        ("human", FACT_EXTRACTION_HUMAN_MESSAGE)
    ])

def get_iteration_decision_prompt(max_iterations: int):
    """Returns the iteration decision chat prompt template with max_iterations filled."""
    system_message = ITERATION_DECISION_SYSTEM_MESSAGE.format(
        max_iterations=max_iterations,
        json_schema="{json_schema}"  # This will be filled by the actual parser
    )
    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", ITERATION_DECISION_HUMAN_MESSAGE)
    ])

def get_final_synthesis_prompt():
    """Returns the final synthesis chat prompt template."""
    return ChatPromptTemplate.from_template(
        FINAL_SYNTHESIS_SYSTEM_MESSAGE + "\n\n" + FINAL_SYNTHESIS_HUMAN_MESSAGE
    )