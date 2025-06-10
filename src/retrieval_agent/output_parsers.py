# File: src/retrieval_agent/output_parsers.py
"""
Pydantic models and output parsers for structured LLM outputs.

This module defines the data models that ensure reliable, structured outputs
from the LLMs at each step of the iterative retrieval process.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser

# --- For Query Understanding (Step 1) ---
class QueryAnalysisResult(BaseModel):
    """Structured output from the query understanding step."""
    main_intent: str = Field(
        description="The core information the user is seeking."
    )
    sub_queries: List[str] = Field(
        default_factory=list,
        description="Specific sub-queries to search for. Can be empty if original query is simple."
    )
    search_keywords: List[str] = Field(
        default_factory=list,
        description="Key terms to emphasize in searches."
    )
    potential_filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Suggested metadata filters (e.g., {'category': 'Pleading'})."
    )
    analysis_notes: str = Field(
        description="Brief notes on the search strategy."
    )
    # Enhanced fields for advanced query understanding
    structured_filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Advanced filters with operators (e.g., {'exhibit_number': {'gte': 5, 'lte': 10}})."
    )
    use_hypothetical_document: bool = Field(
        default=False,
        description="Whether to use HyDE for this query."
    )
    hypothetical_document_type: Optional[str] = Field(
        default=None,
        description="Type of document to generate for HyDE (e.g., 'pleading', 'medical record')."
    )

# Create the parser instance
query_analysis_parser = PydanticOutputParser(pydantic_object=QueryAnalysisResult)

# --- For Fact Extraction (Step 2.C) ---
class ExtractedFact(BaseModel):
    """Structured output from the fact extraction step for a single chunk."""
    is_relevant: bool = Field(
        description="Whether the chunk is relevant to the sub-query."
    )
    extracted_statement: Optional[str] = Field(
        default=None,
        description="The precise sentence(s) or fact(s) extracted if relevant."
    )
    relevance_score_assessment: str = Field(
        description="Assessed relevance: High, Medium, Low, or Irrelevant."
    )
    reasoning_for_relevance: Optional[str] = Field(
        default=None,
        description="Brief reasoning for the relevance assessment."
    )
    # Note: source_metadata will be added programmatically after this LLM call

# Create the parser instance
extracted_fact_parser = PydanticOutputParser(pydantic_object=ExtractedFact)

# --- For Iteration Decision (Step 3) ---
class IterationDecisionOutput(BaseModel):
    """Structured output from the iteration decision step."""
    continue_iteration: bool = Field(
        description="Whether to perform another search iteration."
    )
    next_sub_query: Optional[str] = Field(
        default=None,
        description="The next sub-query to execute, if continuing."
    )
    next_keywords: Optional[List[str]] = Field(
        default_factory=list,
        description="Keywords for the next sub-query."
    )
    next_filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata filters for the next sub-query."
    )
    reasoning: str = Field(
        description="Reasoning for the decision and the next step formulation."
    )
    # Enhanced fields for better iteration control (optional, with defaults)
    search_strategy: Optional[str] = Field(
        default="hybrid",
        description="Search strategy to use: 'vector', 'postgres', or 'hybrid'."
    )
    explore_different_category: Optional[str] = Field(
        default=None,
        description="Specific category to explore if switching focus."
    )

# Create the parser instance
iteration_decision_parser = PydanticOutputParser(pydantic_object=IterationDecisionOutput)

# --- For Final Answer (Step 4) ---
# For the final synthesis, we often use StrOutputParser for flexibility,
# but we can also define a structured model if needed.
class SynthesizedAnswer(BaseModel):
    """Structured output for the final synthesized answer (optional)."""
    answer_text: str = Field(
        description="The synthesized answer to the user's query."
    )
    cited_sources_summary: List[str] = Field(
        default_factory=list,
        description="A list of unique sources cited in the answer."
    )

# Create the parser instance (optional - often just use StrOutputParser for synthesis)
synthesized_answer_parser = PydanticOutputParser(pydantic_object=SynthesizedAnswer)

# --- Additional Helper Classes ---
class RelevanceScore:
    """Enum-like class for relevance scoring."""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    IRRELEVANT = "Irrelevant"
    
    @classmethod
    def is_relevant(cls, score: str) -> bool:
        """Check if a relevance score indicates the chunk is relevant."""
        return score in [cls.HIGH, cls.MEDIUM, cls.LOW]

class FactWithSource(BaseModel):
    """Complete fact with source metadata included."""
    extracted_statement: str = Field(
        description="The extracted fact or statement."
    )
    relevance_score_assessment: str = Field(
        description="Assessed relevance level."
    )
    source_metadata: Dict[str, Any] = Field(
        description="Complete source metadata including filename, exhibit number, Bates range, etc."
    )
    original_chunk_text: Optional[str] = Field(
        default=None,
        description="The original chunk text for context."
    )

# --- Parser Format Instructions ---
def get_parser_format_instructions(parser: PydanticOutputParser) -> str:
    """Get format instructions for a parser to include in prompts."""
    return parser.get_format_instructions()

# --- Validation Helpers ---
def validate_query_analysis(result: QueryAnalysisResult) -> bool:
    """Validate that query analysis produced actionable output."""
    return bool(result.main_intent) and (
        bool(result.sub_queries) or 
        bool(result.search_keywords) or
        bool(result.main_intent)
    )

def validate_iteration_decision(decision: IterationDecisionOutput) -> bool:
    """Validate that iteration decision is internally consistent."""
    if decision.continue_iteration:
        return bool(decision.next_sub_query)
    return True  # If not continuing, no next_sub_query needed