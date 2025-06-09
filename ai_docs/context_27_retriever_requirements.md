Okay, this is crucial for ensuring the advanced iterative retriever meets its sophisticated goals. We need a robust testing plan that uses the actual production scripts and data, and verification criteria that measure its ability to perform multi-level, recursive reasoning.

**REMINDER AGENT: ALL TESTING AND VERIFICATION *MUST* BE PERFORMED BY RUNNING THE MODIFIED PRODUCTION SCRIPTS (`src/retrieval_agent/main_retriever.py` or a CLI that uses it) WITH PRODUCTION DATA. NO DEDICATED TEST SCRIPTS. ALL OBSERVATIONS, RESULTS, AND VERIFICATIONS ARE TO BE LOGGED IN THE SEQUENTIALLY NUMBERED `context_X.md` FILE IN `/ai_docs/`. CONSULT `/Users/josephott/Documents/bates_number_demo/resources/langchain/` IF NEEDED.**

---

## Iterative Retriever: Verification Criteria & Testing Plan

**Global Goal for Verification:**
The iterative retriever implementation should demonstrate the capacity to:
1.  **Deconstruct complex legal queries** into searchable components.
2.  **Iteratively search a vector store** (and potentially a relational DB via tools) using dynamically formulated queries and filters based on ongoing analysis.
3.  **Analyze and re-rank** retrieved document chunks for relevance to the immediate query and the overall proposition.
4.  **Synthesize retrieved information** into a coherent answer that directly addresses the user's query, supporting its conclusions with citations to the evidence.
5.  **Exhibit recursive reasoning:** The system should show evidence of refining its search strategy and understanding based on the results of previous iterations, effectively "drilling down" or "broadening out" its search as needed to gather comprehensive support for or against a legal proposition or concept. This implies an ability to identify gaps in evidence and attempt to fill them.
6.  **Customizability for Legal Concepts:** The framework should be flexible enough (primarily through prompt engineering and potentially tool definition) to be tailored to identify evidence related to specific, well-defined legal concepts (e.g., "negligence," "breach of duty," "hearsay," "consideration in contract").

---

**A. General System Functionality Verification Criteria:**

1.  **VC1.1: Successful Invocation:** The iterative retriever can be invoked with a user query through its designated interface (e.g., a method in `IterativeRetrieverAgent` or a new CLI).
2.  **VC1.2: Completion without Unhandled Errors:** For a range of queries, the retriever completes its iterations and produces an answer without crashing or throwing unhandled Python exceptions.
3.  **VC1.3: Observable Iterations (via LangSmith):** If LangSmith is integrated, each step of the conceptual flow (Query Understanding, Vector Search, Retrieval Analysis, Iteration Decision, Synthesis) should be visible as distinct operations in the trace.
4.  **VC1.4: Adherence to Max Iterations:** The retriever respects the `max_iterations` configuration.
5.  **VC1.5: Graceful Handling of No Results:** If vector searches yield no relevant results after reasonable attempts, the system provides a "no relevant facts found" (or similar) response rather than hallucinating or erroring.

---

**B. Query Understanding & Decomposition Verification Criteria:**

1.  **VC2.1: Sub-Query Generation:** For complex queries, the Query Understanding step (Step 1 in the conceptual flow) generates logical and relevant sub-queries or identifiable aspects for focused searching. (Verify via LangSmith trace or by logging the output of this step).
2.  **VC2.2: Keyword/Entity Extraction:** The Query Understanding step correctly identifies key legal terms, entities (names, organizations), and concepts from the original query to guide initial searches.
3.  **VC2.3: Filter Identification (Potential):** The Query Understanding step can (where appropriate) suggest potential metadata filters (e.g., "category: Pleading") based on the query's nature.

---

**C. Iterative Search & Retrieval Analysis Verification Criteria:**

1.  **VC3.1: Dynamic Query Formulation:** Subsequent search queries (vector_query in Step 2.A) should demonstrably evolve or differ from previous queries based on the analysis of prior results or remaining aspects of the original query. (Verify via LangSmith or logs).
2.  **VC3.2: Dynamic Filter Application:** If the agent logic decides to use metadata filters, these should be correctly applied in the `VectorSearchTool` calls.
3.  **VC3.3: Relevance Assessment (Chunk Level):** The Retrieval Analysis step (Step 2.C) should assign relevance scores/assessments to individual chunks that align with a human evaluator's judgment for a given sub-query.
4.  **VC3.4: Fact Extraction:** Pertinent facts or sentences should be accurately extracted from relevant chunks.
5.  **VC3.5: Re-ranking Effectiveness (If Implemented):** If a distinct re-ranking step is implemented, it should demonstrably improve the ordering of retrieved chunks, bringing more relevant chunks to the top.

---

**D. Recursive Reasoning & Iteration Logic Verification Criteria:**

1.  **VC4.1: Contextual Refinement:** The Iteration Decision step (Step 3) should show evidence (via LangSmith trace of its reasoning output, or logs) that its decision to continue/stop and its formulation of the *next* query are influenced by the `retrieved_facts` so far.
2.  **VC4.2: Gap Identification (Implicit/Explicit):** The system should attempt to search for different aspects of the query across iterations, implicitly trying to fill gaps. (e.g., if query is "facts for X and Y", it might search for X then Y).
3.  **VC4.3: "Drill-Down" Capability:** Given an initial broad query and some initial results, can a follow-up (either as a new query or through agent's internal iteration if smart enough) focus on a specific detail mentioned in the initial results and find more supporting evidence for that detail?
4.  **VC4.4: "Connecting the Dots":** For queries requiring multiple pieces of evidence that are not co-located, the retriever should be able to gather these disparate pieces across iterations and synthesize them.

---

**E. Synthesis & Answer Generation Verification Criteria:**

1.  **VC5.1: Coherent Answer:** The final synthesized answer (Step 4) is grammatically correct, understandable, and directly addresses the original user query.
2.  **VC5.2: Factual Accuracy (Based on Retrieved Chunks):** The statements in the synthesized answer are factually supported by the `retrieved_facts` used as input to the synthesis step.
3.  **VC5.3: Correct Citations:** All facts or assertions in the answer are accurately cited with source document metadata (filename, exhibit #, Bates range, page #) as present in the retrieved chunks' metadata.
4.  **VC5.4: Completeness (Relative to Retrieved Facts):** The answer incorporates all or most of the highly relevant `retrieved_facts` without significant omission of pertinent supporting details found during retrieval.

---

**F. Customizability for Legal Concepts Verification Criteria:**

1.  **VC6.1: Adaptability to Concept-Specific Prompts:** The system (primarily Query Understanding and Synthesis prompts) can be modified to specifically target evidence related to a pre-defined legal concept (e.g., "elements of negligence").
2.  **VC6.2: Concept-Relevant Retrieval:** When prompted to find evidence for a specific legal concept, the retrieved chunks and synthesized answer should predominantly contain information relevant to that concept.

---

## Detailed Testing Plan (Using Production Scripts & Data)

**Test Data:** `/Users/josephott/Documents/bates_number_demo/input_documents` (after being processed by the ingestion pipeline (`main.py`) to populate the vector store and PostgreSQL).
**Test Script:** The primary interface to the iterative retriever (e.g., `python src/retrieval_agent/cli.py "user query"` if a CLI is built, or by calling the main invocation method of `IterativeRetrieverAgent` from a simple Python script *within the existing structure*).

**Test Case Design Philosophy:**
Queries should be designed to test specific verification criteria, especially those related to iterative refinement and legal concept deduction. Start simple and increase complexity.

**Phase 1: Basic Functionality & Single Iteration Tests**

*   **Test Case 1.1: Simple Fact Retrieval**
    *   **Query:** "What is Zaid Aday's business address according to the Alrashedi deposition?" (Assuming "Alrashedi depo" is identifiable and its content is in the vector store).
    *   **Expected:** Correct address cited from the Alrashedi deposition (document_id: 2 from the dump).
    *   **Verify:** VC1.1, VC1.2, VC5.1, VC5.2, VC5.3. (LangSmith: VC1.3, single effective iteration).
*   **Test Case 1.2: Query with No Obvious Answer**
    *   **Query:** "Find evidence of aliens mentioned in the Sixth Amended Petition."
    *   **Expected:** Graceful response indicating no relevant facts found.
    *   **Verify:** VC1.5.
*   **Test Case 1.3: Query Requiring Metadata Filter (if Query Understanding supports it early)**
    *   **Query:** "What motions were filed by Brentwood Glass?" (This might require understanding to filter for `category: Pleading` and keyword "Brentwood Glass").
    *   **Expected:** List of motions or relevant pleadings.
    *   **Verify:** VC2.3 (if filter identified), VC3.2, VC5.1-VC5.3.

**Phase 2: Multi-Aspect & Iterative Refinement Tests**

*   **Test Case 2.1: Two-Part Query**
    *   **Query:** "What was the contract price in the Glass Installation Contract and what was Change Order No. 1 amount in the Brentwood Glass case?" (Referring to Exhibit 4 from dump).
    *   **Expected:** Two distinct facts retrieved, potentially in two sub-queries/iterations, synthesized correctly with citations.
    *   **Verify:** VC2.1, VC3.1, VC4.1, VC4.4, VC5.1-VC5.4. (LangSmith: multiple iterations if sub-queries are generated and processed sequentially).
*   **Test Case 2.2: Vague Initial Query Leading to Refinement**
    *   **Query:** "Tell me about Pal's Glass financial issues."
    *   **Expected:** Initial broad search. Subsequent iterations (if designed for this) might pick up on specific amounts owed or judgments from initial results and then search for more details about *those*. This tests VC4.1, VC4.3.
    *   **Verification:** Requires careful examination of LangSmith traces to see if search queries evolve.
*   **Test Case 2.3: Original User Query Example**
    *   **Query:** "Identify all facts supporting the argument that the Defendant (Pal's Glass) was at fault in causing financial loss to Brentwood Glass."
    *   **Expected:** Multiple facts from "Pal’s Glass’ Breaches of Contact" and "Amount Due" sections of the "Sixth Amended Petition" (Exhibit 4), potentially retrieved over several iterations focusing on different aspects of "fault" or "loss."
    *   **Verify:** VC2.1, VC3.1, VC3.3, VC3.4, VC4.1, VC4.2, VC4.4, VC5.1-VC5.4.

**Phase 3: Legal Concept Deduction Tests**

*   **Setup:** Modify/Create specific prompts in `retrieval_agent/agent_prompts.py` tailored for a chosen legal concept. For instance, for "Breach of Contract (Pal's Glass against Brentwood Glass)":
    *   Query Understanding prompt might be guided to look for elements like: "existence of contract," "plaintiff's performance," "defendant's breach," "damages."
*   **Test Case 3.1: Find Evidence for "Breach of Contract"**
    *   **Query (with specialized prompts active):** "Gather all evidence showing Pal's Glass breached its contract with Brentwood Glass."
    *   **Expected:** System should identify sections in Exhibit 4 (Sixth Amended Petition) like:
        *   "Pal’s Glass’ Breaches of Contact" (paragraph 47 onwards).
        *   Failure to pay `X` amount (paragraphs 46, 52, 53).
        *   Failure to issue change orders (paragraph 47).
    *   The iterations should focus on these elements. Synthesis should tie them to "breach."
    *   **Verify:** VC6.1, VC6.2, and relevant criteria from B, C, D, E.
*   **Test Case 3.2: Find Evidence for a Different Legal Concept**
    *   **Setup:** Define prompts for "Notice Requirement for Mechanic's Lien (Brentwood Glass)."
    *   **Query:** "Find all facts demonstrating Brentwood Glass met the notice requirements for its mechanic's lien."
    *   **Expected:** Evidence from Exhibit 4 around paragraphs 65-68 (Notice served, filed).
    *   **Verify:** VC6.1, VC6.2.

**Phase 4: Stress & Edge Case Tests**

*   **Test Case 4.1: Highly Ambiguous Query**
    *   **Query:** "What's the main problem in these documents?"
    *   **Expected:** The agent might struggle but should ideally try to identify dominant themes or ask for clarification if its internal analysis leads to high uncertainty. At minimum, it shouldn't crash.
    *   **Verify:** Robustness, VC1.2, potentially quality of its internal `analysis_notes`.
*   **Test Case 4.2: Query Referencing Non-Existent Entities/Facts**
    *   **Query:** "Find statements made by John Doe in the Alrashedi deposition regarding the Toyota Camry."
    *   **Expected:** Correctly states John Doe is not mentioned or no such statements found.
    *   **Verify:** VC1.5, VC5.2 (by correctly stating absence).

**Testing Procedure for Each Test Case:**

1.  **Formulate Query:** Based on the test case.
2.  **Set Up (If Needed):** Ensure any specific prompts for legal concept testing are active. Ensure LangSmith environment variables are set.
3.  **Execute:** Run the query through the iterative retriever's main interface.
    *   `python src/retrieval_agent/cli.py "YOUR_QUERY_HERE"` (or equivalent invocation).
4.  **Observe Output:** Record the final synthesized answer.
5.  **Inspect LangSmith Trace (Crucial):**
    *   Verify each step of the conceptual flow.
    *   Examine LLM inputs/outputs at each stage (query understanding, iteration decision, synthesis).
    *   Check the actual vector search queries and filters used.
    *   Analyze the retrieved chunks before and after any re-ranking/analysis.
6.  **Verify Against Criteria:** Check off the specific VCs targeted by the test case.
7.  **Document in `context_X.md`:**
    *   Test Case ID & Query.
    *   Expected Outcome.
    *   Actual Outcome.
    *   LangSmith Trace URL (if applicable).
    *   Verification Criteria Met/Not Met (with brief justification).
    *   Any errors, unexpected behavior, or observations.

By following this plan, you can systematically assess the capabilities and robustness of your iterative retriever, focusing on its ability to perform complex, reasoned information retrieval from your legal document corpus.