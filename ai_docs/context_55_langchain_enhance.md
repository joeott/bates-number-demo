Based on the comprehensive search of Langchain documentation and related materials, here is a list of proposed updates to your scripts, focusing on the most impactful enhancements for your iterative retrieval agent.
I. Core Retrieval & Re-ranking Enhancements
Integrate Hybrid Search (Leveraging hybrid_search.py)
Script: src/retrieval_agent/agent_tools.py
Lines/Approach:
Modify get_vector_searcher or add get_hybrid_searcher:
Currently, get_vector_searcher initializes VectorSearcher. You'll need a similar mechanism for HybridSearcher from src.hybrid_search.py.
Consider if HybridSearcher should be the default or an alternative.
Create a new tool perform_hybrid_search:
# In agent_tools.py
from src.hybrid_search import HybridSearcher, SearchMethod # Add this import

_hybrid_searcher_instance = None

def get_hybrid_searcher():
    global _hybrid_searcher_instance
    if _hybrid_searcher_instance is None:
        # Initialize HybridSearcher, potentially using configs from src.config
        # or agent_config.py for vector store path and PG connection
        _hybrid_searcher_instance = HybridSearcher(
            vector_store_path=str(VECTOR_STORE_PATH),
            postgres_config={ # Assuming PG details are in src.config
                'connection_string': POSTGRES_CONNECTION,
                'pool_size': POSTGRES_POOL_SIZE
            } if ENABLE_POSTGRES_STORAGE else None
        )
        logger.info("HybridSearcher initialized.")
    return _hybrid_searcher_instance

@tool
def perform_hybrid_search(
    query_text: str,
    k_results: int = NUM_RESULTS_PER_SUB_QUERY,
    metadata_filters: Optional[Dict[str, Any]] = None,
    search_method: str = "hybrid" # Allow "vector", "postgres", or "hybrid"
) -> List[Dict]:
    """
    Searches documents using a hybrid approach (vector + keyword) or specific methods.
    Args:
        query_text: The text to search for.
        k_results: The number of results to return.
        metadata_filters: A dictionary of metadata to filter by.
        search_method: 'vector', 'postgres', or 'hybrid'.
    Returns:
        A list of search results.
    """
    try:
        searcher = get_hybrid_searcher()
        method_enum = SearchMethod.HYBRID
        if search_method == "vector":
            method_enum = SearchMethod.VECTOR
        elif search_method == "postgres":
            method_enum = SearchMethod.POSTGRES

        # HybridSearcher.search expects filters directly
        # Ensure metadata_filters keys match what HybridSearcher expects or adapt them
        # e.g., category, exhibit_number (as per hybrid_search.py)
        
        search_results = searcher.search(
            query=query_text,
            limit=k_results,
            filters=metadata_filters,
            method=method_enum
        )
        
        # Convert SearchResult objects to dicts for broader compatibility if needed
        # or ensure the agent handles SearchResult objects.
        # For now, let's assume agent can handle SearchResult objects or they are dict-like
        # If conversion is needed:
        # cleaned_results = [result.__dict__ for result in search_results]
        
        # The SearchResult dataclass in hybrid_search.py already provides a good structure.
        # Let's ensure the output is a list of dictionaries as the tool is typed.
        output_results = []
        for res in search_results:
            output_results.append({
                "document_id": res.document_id,
                "filename": res.filename,
                "category": res.category,
                "exhibit_number": res.exhibit_number,
                "bates_range": res.bates_range,
                "text": res.text, # This is the excerpt or relevant chunk
                "score": res.score,
                "page": res.page,
                "source": res.source, # 'vector', 'postgres', 'hybrid'
                "vector_score": res.vector_score,
                "postgres_score": res.postgres_score,
                "metadata": { # Reconstruct a metadata dict for consistency with vector_search tool
                    "filename": res.filename,
                    "category": res.category,
                    "exhibit_number": res.exhibit_number,
                    "bates_start": res.bates_range.split('-')[0] if res.bates_range and '-' in res.bates_range else res.bates_range,
                    "bates_end": res.bates_range.split('-')[1] if res.bates_range and '-' in res.bates_range else None,
                    "page": res.page
                }
            })
        logger.info(f"Hybrid search ('{search_method}') returned {len(output_results)} results for query: '{query_text}'")
        return output_results
    except Exception as e:
        logger.error(f"Error in perform_hybrid_search: {str(e)}", exc_info=True)
        return []
Use code with caution.
Python
Update AVAILABLE_TOOLS and get_agent_tools in agent_tools.py: Add perform_hybrid_search and potentially make it the default or selectable.
Script: src/retrieval_agent/main_retriever.py
Lines/Approach:
Modify the section where agent_tools.perform_vector_search.invoke is called (around line 284 in the context).
The agent should decide which search tool to use (vector, keyword, or hybrid) based on the query analysis or configuration. This decision could be part of the iteration_decision_chain.
# In IterativeRetrieverAgent.invoke, around line 284:
# Decision logic to select search_tool_to_use (e.g., agent_tools.perform_hybrid_search)
# and search_method_for_tool ("vector", "postgres", "hybrid")

# Example using the new hybrid tool:
tool_input = {
    "query_text": sub_query,
    "k_results": agent_config.NUM_RESULTS_PER_SUB_QUERY,
    "metadata_filters": current_filters, # Ensure current_filters is compatible
    "search_method": "hybrid" # or make this dynamic
}
search_results: List[Dict] = agent_tools.perform_hybrid_search.invoke(
    tool_input,
    config=run_config
)
Use code with caution.
Python
Documentation: Langchain docs on hybrid search[1][2] mention that there isn't a single unified way and it's vectorstore-dependent. Your HybridSearcher already abstracts this. The BAAI/bge-m3 paper[3] and MongoDB Atlas guide[4] provide context on why hybrid search is useful.
Advanced Re-ranking (Cross-Encoder)
Script: src/retrieval_agent/agent_tools.py (or a new rerankers.py module)
Lines/Approach:
Import ContextualCompressionRetriever and a cross-encoder model (e.g., HuggingFaceCrossEncoder or CohereRerank if API key is available).
# In agent_tools.py or a new rerankers.py
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker # Or CohereRerank
from langchain_community.cross_encoders import HuggingFaceCrossEncoder # If using local HF model

# Initialize the cross-encoder model (once)
# model_name = "BAAI/bge-reranker-base" # Example
# hf_cross_encoder = HuggingFaceCrossEncoder(model_name=model_name)
# compressor = CrossEncoderReranker(model=hf_cross_encoder, top_n=3) # top_n is how many docs to return after reranking

# OR for Cohere:
# from langchain_cohere import CohereRerank # Ensure langchain-cohere is installed
# compressor = CohereRerank(cohere_api_key="YOUR_COHERE_KEY", top_n=3)
Use code with caution.
Python
The perform_vector_search (or perform_hybrid_search) tool could internally use this. The base retriever (e.g., vector store's as_retriever()) would be wrapped by ContextualCompressionRetriever.
# Inside perform_vector_search or similar tool, after getting the base retriever from vector_store
# base_retriever = vector_store.as_retriever(search_kwargs={"k": k_results_before_rerank}) # Get more docs initially
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=compressor, # compressor initialized above
#     base_retriever=base_retriever
# )
# reranked_docs = compression_retriever.invoke(query_text) 
# search_results = [{"text": doc.page_content, "metadata": doc.metadata, "relevance": getattr(doc, 'relevance_score', 1.0)} for doc in reranked_docs]
Use code with caution.
Python
Script: src/retrieval_agent/main_retriever.py
Lines/Approach: The change would primarily be in the tool. The agent would receive already re-ranked results. You'd adjust NUM_RESULTS_PER_SUB_QUERY in agent_config.py to reflect the top_n from the reranker.
Documentation: Langchain docs on CohereRerank[2] and general cross-encoder reranking[1][3][5] show how to set this up. The key is wrapping a base retriever.
Contextual Compression Retriever Pattern
Script: src/retrieval_agent/agent_tools.py
Lines/Approach: Similar to re-ranking, but using LLMChainExtractor or LLMChainFilter as the base_compressor.
# In agent_tools.py or a new compressors.py
from langchain.retrievers.document_compressors import LLMChainExtractor, LLMChainFilter
from langchain_openai import OpenAI # Or your chosen LLM for compression

# llm_for_compression = OpenAI(temperature=0) # Or ChatOllama etc.
# extractor_compressor = LLMChainExtractor.from_llm(llm_for_compression)
# filter_compressor = LLMChainFilter.from_llm(llm_for_compression)

# Then, in the search tool:
# base_retriever = vector_store.as_retriever(...)
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=extractor_compressor, # or filter_compressor
#     base_retriever=base_retriever
# )
# compressed_docs = compression_retriever.invoke(query_text)
Use code with caution.
Python
Documentation: Langchain blog[1] and how-to guides[2][4][5] provide clear examples of ContextualCompressionRetriever with LLMChainExtractor and LLMChainFilter.
Self-Querying Retriever Capabilities
Script: src/retrieval_agent/agent_prompts.py
Lines/Approach:
Modify QUERY_UNDERSTANDING_SYSTEM_MESSAGE to explicitly ask the LLM to generate structured filter objects based on the user query and attribute descriptions.
Script: src/retrieval_agent/output_parsers.py
Lines/Approach:
Ensure QueryAnalysisResult.potential_filters can hold complex structured queries (e.g., with and, or, comparators like eq, gt, lt).
Script: src/retrieval_agent/agent_tools.py
Lines/Approach:
The perform_vector_search (or hybrid) tool needs to be able to parse these structured filters and pass them to the underlying vector store's search method if supported (e.g., ChromaDB supports where clauses with operators).
Alternatively, implement SelfQueryRetriever from Langchain. This requires defining AttributeType for your metadata and using SelfQueryRetriever.from_llm(...).
# In agent_tools.py, potentially as a new tool or integrated into existing ones
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
# from langchain_openai import OpenAI # LLM for query construction

# metadata_field_info = [
#     AttributeInfo(name="genre", description="The genre of the movie", type="string"),
#     AttributeInfo(name="year", description="The year the movie was released", type="integer"),
#     AttributeInfo(name="category", description="Document category (e.g., Pleading, Medical Record)", type="string"),
#     AttributeInfo(name="exhibit_number", description="The exhibit number of the document", type="integer"),
# ]
# document_content_description = "Content of legal documents"
# llm_for_self_query = OpenAI(temperature=0) # Or ChatOllama
# self_query_retriever = SelfQueryRetriever.from_llm(
#     llm_for_self_query,
#     vector_store, # Your Chroma instance
#     document_content_description,
#     metadata_field_info,
# )
# # Then use self_query_retriever.invoke(query_text)
Use code with caution.
Python
Documentation: Langchain's Self-Querying guide[1][2][3][4][5] is very detailed, explaining the need for AttributeType, document_content_description, and how the LLM chain constructs structured queries.
Parent Document Retriever Pattern
Script: src/vector_processor.py
Lines/Approach:
During indexing, you need to store parent documents (or larger chunks) in a docstore (e.g., InMemoryStore, LocalFileStore) and index smaller child chunks in the vectorstore. The ParentDocumentRetriever links them.
Modify VectorProcessor.process_document to implement this logic. It will involve creating child splits from parent documents and managing IDs to link them.
Script: src/retrieval_agent/agent_tools.py
Lines/Approach:
The search tool would use ParentDocumentRetriever instead of directly using the vector store's similarity_search.
# In agent_tools.py
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore # Or LocalFileStore for persistence
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Assuming vectorstore is your Chroma instance, and embeddings are defined
# store = InMemoryStore() # This would need to be populated during indexing
# child_splitter = RecursiveCharacterTextSplitter(chunk_size=400) 
# parent_retriever = ParentDocumentRetriever(
#     vectorstore=vectorstore, # Chroma with child chunks
#     docstore=store,         # Store with parent documents
#     child_splitter=child_splitter,
#     # parent_splitter can also be defined if you want to retrieve specific parent chunks
# )
# # Then use parent_retriever.invoke(query_text)
Use code with caution.
Python
Documentation: Langchain's ParentDocumentRetriever examples[1][2][3][4][5] show how to set up the vectorstore for child chunks and the docstore for parent documents.
Multi-Vector Retriever (for HyDE, Summaries, etc.)
Script: src/vector_processor.py
Lines/Approach:
During indexing in process_document:
Generate summaries of documents/chunks using an LLM.
Generate hypothetical questions for documents/chunks using an LLM.
Store these alternative representations (summaries, hypothetical questions) in the vector store, linked to the original document ID.
Script: src/retrieval_agent/agent_tools.py
Lines/Approach:
Use MultiVectorRetriever in the search tool. This retriever allows you to define how different vector representations are queried and how results are retrieved (e.g., retrieve original document based on a match with its summary embedding).
# In agent_tools.py
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore # Or another docstore

# Assuming vectorstore stores multiple embeddings per doc (e.g., original, summary, hypothetical_q)
# And id_key links these embeddings to the original doc ID
# docstore stores the actual parent documents

# multi_vector_retriever = MultiVectorRetriever(
#     vectorstore=vectorstore, 
#     docstore=docstore, # To retrieve the full document content
#     id_key="doc_id" # Metadata key that links embeddings to original docs in docstore
# )
# Then use multi_vector_retriever.invoke(query_text)
Use code with caution.
Python
Documentation: Langchain's MultiVectorRetriever examples[1][2][3][4] explain how to store multiple vectors per document (e.g., for chunks, summaries, hypothetical questions) and retrieve the original document.
II. Query Understanding & Expansion Enhancements
HyDE (Hypothetical Document Embeddings)
Script: src/retrieval_agent/main_retriever.py
Lines/Approach:
Before calling the search tool in the main iteration loop (around line 284), add a step to generate a hypothetical document.
# In IterativeRetrieverAgent.invoke, before search tool call
# from langchain.chains import LLMChain # If using older style
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnableSequence

# hyde_prompt_template = "Please write a passage to answer the question: {question}"
# hyde_prompt = PromptTemplate.from_template(hyde_prompt_template)

# generate_hyde_doc_chain = hyde_prompt | self.llm | StrOutputParser()
# hypothetical_document = generate_hyde_doc_chain.invoke({"question": sub_query})

# # The search tool would then need to be adapted to optionally take this hypothetical_document,
# # embed it, and use that embedding for search.
# # Or, use Langchain's HydeRetriever directly.
Use code with caution.
Python
Alternatively, wrap your base retriever with HydeRetriever.
Script: src/retrieval_agent/agent_tools.py
Lines/Approach:
If not using HydeRetriever directly, the search tool (perform_vector_search or perform_hybrid_search) would need modification. It would take an optional hypothetical_document_text. If provided, it would embed this text and use the resulting embedding for the similarity search instead of embedding the query_text.
# In agent_tools.py, inside perform_vector_search or perform_hybrid_search
# if hypothetical_document_text:
#     query_embedding = self.embeddings.embed_query(hypothetical_document_text)
#     # Perform vector search using this precomputed query_embedding
# else:
#     # Standard search by embedding query_text
Use code with caution.
Python
Documentation: Langchain.js has a HydeRetriever[3]. The core idea is to generate a document with an LLM and then embed that[1][2][4][5]. HypotheticalDocumentEmbedder can also be used.
III. Iteration Logic & Synthesis Enhancements
More Sophisticated Iteration Decision Logic (Information Gain, Specific Gaps)
Script: src/retrieval_agent/agent_prompts.py
Lines/Approach:
Modify ITERATION_DECISION_SYSTEM_MESSAGE to ask the LLM to:
Assess if new and significant information was found in the last iteration.
Identify specific information gaps remaining relative to the original query.
Suggest next_sub_query to fill these specific gaps.
Script: src/retrieval_agent/main_retriever.py
Lines/Approach:
In the main loop, before the iteration decision call, compare newly_extracted_facts with accumulated_facts to crudely estimate information gain. Pass this observation to the iteration_decision_chain.
If iteration_decision.next_sub_query is very similar to an already executed_queries, consider broadening or stopping.
IV. Integration with Document Processing & Metadata
Leverage DocumentAnalyzer Output
Script: src/vector_processor.py
Lines/Approach:
In VectorProcessor.process_document (or the underlying PDFToLangChainLoader), invoke DocumentAnalyzer for each PDF.
Add the analysis results (e.g., dominant_type, is_scanned, needs_ocr, avg_image_dpi) to the metadata of each Document object and thus to the metadata of the chunks stored in ChromaDB.
# In PDFToLangChainLoader.load() or VectorProcessor.process_document()
# analyzer = DocumentAnalyzer() # Assuming DocumentAnalyzer is in scope
# analysis_results = analyzer.analyze_document(Path(self.file_path))
# ...
# for doc in pages:
#     doc.metadata.update(analysis_results) # Add all analysis fields
Use code with caution.
Python
Script: src/retrieval_agent/agent_tools.py
Lines/Approach:
If using SelfQueryRetriever (Enhancement #4), define these new metadata fields in AttributeInfo so the agent can filter on them (e.g., "find scanned documents related to X").
The standard metadata filter in perform_vector_search can also use these new keys if the query understanding step generates them.
Script: src/retrieval_agent/agent_prompts.py
Lines/Approach:
Update QUERY_UNDERSTANDING_SYSTEM_MESSAGE to inform the LLM about these new available metadata fields (e.g., "You can filter by document_type: 'scanned', 'text', 'mixed'").
This is a substantial list. Implementing all would be a significant effort. I'd recommend prioritizing based on the biggest pain points in your current retrieval quality. Hybrid Search, Advanced Re-ranking, Contextual Compression, and Self-Querying are often good starting points for major improvements.