\
Given the directive **NOT TO MESS WITH WHAT IS ALREADY FUNCTIONING** and to consider future microservice deployment, we'll design the retriever implementation as a distinct module that *consumes* the outputs of the current pipeline (the vector store and potentially the PostgreSQL database) but doesn't alter the ingestion logic.

## Iterative Retriever Implementation Plan

**Core Idea:** Create a multi-step "Retriever Agent" or a complex LCEL chain that iteratively refines its understanding of a user query, performs multiple targeted vector searches, analyzes results, and synthesizes an answer.

**Directory Structure Consideration:**

To keep things clean and separate, while allowing interaction:

```
project_root/
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- db_storage.py             # Existing
|   |-- document_orchestrator.py  # Existing (for ingestion)
|   |-- llm_handler.py            # Existing (could be used by retriever too)
|   |-- pdf_processor.py          # Existing
|   |-- utils.py                  # Existing
|   |-- vector_processor.py       # Existing (for ingestion)
|   |-- vector_search.py          # Existing (for basic search, can be leveraged/extended)
|   |
|   |-- retrieval_agent/          # NEW MODULE for the advanced retriever
|   |   |-- __init__.py
|   |   |-- agent_config.py       # Configuration specific to the retrieval agent
|   |   |-- agent_prompts.py      # Prompts for query analysis, sub-query generation, synthesis
|   |   |-- agent_tools.py        # Langchain tools (e.g., custom vector search tool)
|   |   |-- main_retriever.py     # Core logic for the iterative retrieval agent/chain
|   |   |-- output_parsers.py     # Parsers for structured LLM outputs during retrieval
|   |
|   |-- search_cli.py             # Existing (could be augmented or a new one created for the agent)
|
|-- input_documents/
|-- output/
|   |-- vector_store/
|   |-- ...
|-- main.py                     # Existing (for ingestion)
|-- .env
|-- ...
```

This structure:
1.  Isolates the new retrieval logic in `src/retrieval_agent/`.
2.  Allows the retrieval agent to import and use `src.vector_search.VectorSearcher` (or a refactored version) and potentially `src.llm_handler.LLMCategorizer` if needed for its internal LLM calls.
3.  Facilitates future deployment as the `retrieval_agent` could become its own microservice, interacting with the vector store (and potentially Postgres) which might also be independent services or accessed via APIs.

---

**Langchain Concepts to Leverage for the Retriever Implementation:**

1.  **LCEL (Langchain Expression Language):** The backbone for defining the iterative flow.
2.  **Chat Models (`ChatOllama`, `ChatOpenAI` from `src.llm_handler` or new instances):** For all reasoning steps: query analysis, sub-query generation, result analysis, synthesis.
3.  **Prompt Templates (`ChatPromptTemplate`):** Crucial for guiding the LLMs at each step. These will be defined in `retrieval_agent/agent_prompts.py`.
4.  **Output Parsers (`PydanticOutputParser`, `JsonOutputParser`, custom parsers):** To get structured output from LLMs (e.g., a list of sub-queries, analysis of search results, parameters for vector search). Defined in `retrieval_agent/output_parsers.py`.
5.  **Tools (`@tool` decorator, `BaseTool`):**
    *   A custom `VectorSearchTool` will be essential. This tool will wrap your existing `VectorSearcher.search()` method (or a more advanced version of it). It will accept parameters like the query string, number of results, and *critically, dynamic filter conditions* derived by the agent.
    *   Potentially a `PostgresSearchTool` if direct keyword or metadata search against the SQL DB is also desired as part of the iterative process.
6.  **Retrievers (as part of the `VectorSearchTool`):** The `VectorSearchTool` will internally use `Chroma` as a retriever, configured with `OllamaEmbeddings`, similar to `src.vector_search.py`.
7.  **AgentExecutor (Optional, but likely useful for complex iteration):**
    *   If the iteration involves dynamic decision-making about *which* tool to use next or complex looping based on LLM analysis, an `AgentExecutor` (with a ReAct-style agent or an OpenAI Functions/Tools agent if using OpenAI models, or a custom agent loop for Ollama) could manage the process.
    *   Alternatively, a more complex LCEL chain with `RunnableBranch` and custom `RunnableLambda`s can implement the iterative loop.
8.  **`RunnableConfigurableFields` / `RunnablePassthrough.assign` / State Management:** To pass information and context between iterations.
9.  **Re-ranking:**
    *   Could be a `RunnableLambda` that takes initial retrieval results and the original query, then uses an LLM (or a specialized re-ranking model like Cohere ReRank if you ever move beyond fully local) to score them for relevance.
    *   Local re-ranking might involve an LLM call per retrieved document, which can be slow. A simpler approach might be to use a cross-encoder model locally if one is available and performant. For a pure LLM approach, a prompt would ask the LLM to rate the relevance of a chunk to the query.

---

**Conceptual Flow for the Iterative Retriever Agent:**

User Query: *"Identify all facts supporting the argument that the Defendant was at fault in causing the crash"*

**(Defined in `retrieval_agent/main_retriever.py` using LCEL)**

```
Initial State: 
{
  "original_query": "User Query",
  "conversation_history": [], // For potential multi-turn interaction
  "retrieved_facts": [],
  "search_queries_executed": [],
  "max_iterations": 3 // Configurable
}

Iteration Loop (e.g., using a custom LCEL loop or managed by AgentExecutor):
```

**Step 1: Query Understanding & Decomposition (LLM)**
    *   **Input:** `original_query`, `conversation_history` (if any)
    *   **Prompt (`agent_prompts.py`):** "Analyze the query. Identify key entities, concepts, and the core information need. If complex, break it down into 2-3 sub-queries or aspects to search for. Specify search parameters (e.g., keywords, potential date ranges if inferable, document categories to prioritize if applicable like 'Pleading', 'Correspondence')."
    *   **LLM:** `ChatOllama` (from `src.llm_handler` or a new instance tailored for retrieval agent tasks).
    *   **Output Parser (`output_parsers.py`):** A Pydantic model for `QueryAnalysisResult` containing:
        *   `main_intent: str`
        *   `sub_queries: List[str]`
        *   `search_keywords: List[str]`
        *   `potential_filters: Dict[str, Any]` (e.g., `{"category": "Pleading"}`)
        *   `analysis_notes: str`
    *   **LCEL:** `query_understanding_chain = prompt | llm | QueryAnalysisOutputParser()`

**Step 2: Iterative Search & Context Building**
    *   For each `sub_query` (or the refined main query) generated in Step 1:
        *   **A. Formulate Vector Search (`RunnableLambda` or LLM)**
            *   **Input:** `sub_query`, `search_keywords`, `potential_filters`, `retrieved_facts` (from previous iterations, to avoid redundancy or to guide negative searches).
            *   **Logic/Prompt:** "Given the sub-query and existing context, formulate an optimal vector search query string and any specific metadata filters (e.g., category, exhibit_number) for the `VectorSearchTool`."
            *   **Output:** `{"vector_query": str, "vector_filters": Dict}`

        *   **B. Execute Vector Search (Custom `VectorSearchTool` in `agent_tools.py`)**
            *   **Tool Input:** `vector_query`, `vector_filters`, `num_results` (configurable).
            *   **Tool Logic:**
                *   Instantiates `VectorSearcher` from `src.vector_search`.
                *   Calls `vector_searcher.search(query=vector_query, filter=vector_filters, n_results=...)`.
            *   **Tool Output:** `List[Dict]` (raw search results, each dict containing `text`, `metadata`, `relevance`).

        *   **C. Retrieval Analysis & Re-ranking (LLM or custom `RunnableLambda`)**
            *   **Input:** `sub_query`, `raw_search_results` from Step 2B.
            *   **Prompt/Logic:** "For each retrieved chunk, assess its direct relevance to the sub-query: '[sub_query]'. Assign a relevance score (e.g., High, Medium, Low) and extract the specific sentence(s) or fact(s) that are most pertinent. Discard irrelevant chunks."
                *   *Local Re-ranking:* If using an LLM, this could be slow. A simpler first pass: filter by the `relevance` score from ChromaDB. Then, for the top N, do a more detailed LLM-based assessment.
                *   *Advanced:* Use a local cross-encoder model if available.
            *   **Output Parser:** List of `{"chunk_text": str, "extracted_fact": str, "source_metadata": Dict, "assessed_relevance": str}`.

        *   **D. Update Context (`RunnableLambda`)**
            *   Add `extracted_facts` to the main `retrieved_facts` list in the loop's state.
            *   Add the `sub_query` to `search_queries_executed`.

**Step 3: Iteration Decision & Next Query Formulation (LLM)**
    *   **Input:** `original_query`, current `retrieved_facts`, `search_queries_executed`, `analysis_notes` from Step 1, `max_iterations`, current iteration number.
    *   **Prompt:** "Review the original query and the facts retrieved so far.
        1.  Are the retrieved facts sufficient to answer the original query?
        2.  Are there any unexplored aspects or missing links?
        3.  Based on the analysis, should another search iteration be performed?
        4.  If yes, formulate the next `sub_query` and any refined `search_keywords` or `potential_filters`. Aim to find new, complementary information or to clarify existing points. If no further iteration is needed or max iterations reached, indicate completion."
    *   **Output Parser:** `{"continue_iteration": bool, "next_sub_query": Optional[str], "next_keywords": Optional[List[str]], "next_filters": Optional[Dict], "reasoning": str}`.
    *   If `continue_iteration` is true and `max_iterations` not reached, loop back to Step 2 with the new query parameters.

**Step 4: Synthesis & Answer Generation (LLM)**
    *   **Input:** `original_query`, all `retrieved_facts` (with sources).
    *   **Prompt:** "Based on the following retrieved facts, synthesize a comprehensive answer to the query: '[original_query]'. Present the answer clearly. For each fact or statement in your answer, cite the source document (filename, exhibit number, Bates range, page number) from the provided metadata. If no relevant facts were found, state that."
    *   **LLM:** `ChatOllama`.
    *   **Output Parser:** `StrOutputParser()` or a Pydantic model for a structured answer with citations.
    *   **LCEL:** `synthesis_chain = prompt | llm | AnswerOutputParser()`

---

**Implementation in `retrieval_agent/`:**

*   **`agent_config.py`:**
    *   `MAX_ITERATIONS`
    *   `NUM_RESULTS_PER_SEARCH`
    *   LLM models to use for different agent steps (if different from global config).
*   **`agent_prompts.py`:**
    *   `QUERY_DECOMPOSITION_PROMPT`
    *   `VECTOR_SEARCH_FORMULATION_PROMPT`
    *   `RESULT_ANALYSIS_RELEVANCE_PROMPT`
    *   `ITERATION_DECISION_PROMPT`
    *   `FINAL_SYNTHESIS_PROMPT`
*   **`agent_tools.py`:**
    ```python
    from langchain_core.tools import tool
    from src.vector_search import VectorSearcher # Assuming VectorSearcher is robust
    from src.config import VECTOR_STORE_PATH # Or from agent_config

    @tool
    def vector_search_tool(query: str, num_results: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        """
        Performs a vector search on the legal documents.
        Filters can be provided as a dictionary, e.g., {"category": "Pleading", "exhibit_number": 2}.
        """
        searcher = VectorSearcher(vector_store_path=VECTOR_STORE_PATH) # Path from config
        # The VectorSearcher.search method needs to be robust and accept these params.
        # The current VectorSearcher.search in vector_search.py looks compatible.
        results = searcher.search(query=query, n_results=num_results, **(filters or {}))
        # Return a serializable version of results, possibly just the text and metadata
        return [{"text": r.get("text"), "metadata": r.get("metadata"), "relevance": r.get("relevance")} for r in results]
    ```
*   **`output_parsers.py`:** Pydantic models for `QueryAnalysisResult`, `FactItem`, `IterationDecisionOutput`, `FinalAnswer`.
*   **`main_retriever.py`:**
    *   Class `IterativeRetrieverAgent` or a set of LCEL chains.
    *   Method `invoke(query: str)` that orchestrates the loop.
    *   This will be the most complex part, defining the LCEL for the loop and steps.
        *   A `RunnableLambda` could manage the loop state and call sub-chains for each step.
        *   Or, a more traditional Python loop calling LCEL chains for each step. Given the complexity of managing state across iterations, a Python loop calling LCEL sub-chains might be more straightforward to implement initially than a pure, deeply nested LCEL-only loop.

**Interaction with Existing Codebase:**

*   The `VectorSearchTool` directly uses `src.vector_search.VectorSearcher`. This is the primary interaction point with the existing search functionality.
*   The LLM instances for the agent can be new instances of `ChatOllama` (or `ChatOpenAI`) configured via `src.config` or `retrieval_agent.agent_config`.
*   No modification to `main.py` (ingestion) or `document_orchestrator.py` is needed.

**Future API Endpoints:**

*   The ingestion pipeline (`main.py` or `DocumentOrchestrator`) could expose an API endpoint (e.g., `/process_document`) to add new documents.
*   The `IterativeRetrieverAgent` in `retrieval_agent/main_retriever.py` could expose an API endpoint (e.g., `/query_agent`) that takes a user query and returns the synthesized answer.

This plan provides a structured way to build the advanced retriever while respecting the existing, functional ingestion pipeline. The key will be carefully designing the prompts and the logic within the iterative loop.