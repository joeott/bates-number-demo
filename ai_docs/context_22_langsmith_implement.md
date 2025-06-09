
---

## Implementing LangSmith Tracing

LangSmith provides invaluable observability into your LCEL chains, showing exactly what's happening at each step, including inputs, outputs, timings, and any errors.

**Prerequisites:**

1.  **Sign up for LangSmith:** Go to [smith.langchain.com](https://smith.langchain.com/) and create an account if you haven't already.
2.  **Get API Key:** Create an API key from your LangSmith settings.
3.  **Set Environment Variables:** LangSmith tracing is typically enabled by setting these environment variables:
    *   `LANGCHAIN_TRACING_V2="true"`
    *   `LANGCHAIN_API_KEY="ls__your_api_key_here"`
    *   `LANGCHAIN_PROJECT="your_project_name"` (e.g., "BatesNumberingProd" or "BatesNumberingDev") - This helps organize runs in the LangSmith UI.

You can set these in your `.env` file (which `python-dotenv` loads) or directly in your shell environment before running the script.

**Where to Add/Modify Code:**

Usually, **no direct code changes are *required* in your LCEL chains to enable basic tracing** if the environment variables are set. Langchain automatically picks them up.

However, you can enhance tracing with more metadata:

1.  **`main.py` (or wherever your main orchestration happens):**
    *   When you invoke your main chain, you can add metadata and tags to the run.

    ```python
    # In main.py, before processing documents

    # (Ensure environment variables for LangSmith are set)
    # import os
    # print(f"LANGCHAIN_TRACING_V2: {os.getenv('LANGCHAIN_TRACING_V2')}")
    # print(f"LANGCHAIN_PROJECT: {os.getenv('LANGCHAIN_PROJECT')}")


    # ... inside the loop in main() where orchestrator.process_document is called
    # or before orchestrator.process_batch

    # For process_document in the loop:
    # results = []
    # for doc_path in documents_to_process:
    #     run_name = f"Process-{doc_path.name}" # Example run name
    #     result, bates_counter, exhibit_counter = orchestrator.process_document(
    #         doc_path, 
    #         bates_counter, 
    #         exhibit_counter,
    #         # LangChain will pick up tracing automatically if env vars are set.
    #         # For more explicit control or adding metadata to a specific invocation:
    #         # config={"metadata": {"document_path": str(doc_path)}, "tags": ["single_doc_processing"], "run_name": run_name}
    #         # This config would be passed to the .invoke() call on an LCEL chain.
    #         # Since orchestrator.process_document wraps the chain, you'd need to pass
    #         # this config dict into process_document and then to self.safe_processing_chain.invoke(input_data, config=run_config)
    #     )
    #     results.append(result)

    # For process_batch:
    # If orchestrator.process_batch internally calls .invoke() on the main chain for each item,
    # you'd modify DocumentOrchestrator.process_document or process_batch
    # to pass a config dict to the .invoke() call.

    # Modify DocumentOrchestrator.process_document to accept and use config
    # In DocumentOrchestrator.py:
    # def process_document(
    #     self, 
    #     file_path: Path, 
    #     bates_counter: int, 
    #     exhibit_counter: int,
    #     run_config: Optional[Dict] = None # New parameter
    # ) -> Tuple[ProcessingResult, int, int]:
    #     # ...
    #     # When invoking the chain:
    #     result = self.safe_processing_chain.invoke(input_data, config=run_config)
    #     # ...

    # Then in main.py:
    # for file_path in file_paths: # in orchestrator.process_batch or main.py loop
    #     current_run_config = {
    #         "metadata": {"document_path": str(file_path), "original_filename": file_path.name},
    #         "tags": ["batch_item", f"category_target:{category_from_llm_or_input}"], # add dynamic tags
    #         "name": f"ProcessDoc-{file_path.stem}" # LangSmith run name
    #     }
    #     result, bates_counter, exhibit_counter = self.process_document(
    #         file_path, bates_counter, exhibit_counter, run_config=current_run_config
    #     )
    #     results.append(result)
    ```

2.  **Naming Your Runnables (Optional but good for clarity in LangSmith):**
    When defining complex chains, especially with `RunnableLambda`, giving them names can make the LangSmith trace much easier to read.
    ```python
    # In DocumentOrchestrator._build_chain()
    self.validation_chain = RunnableLambda(self._validate_document, name="DocumentValidation")
    self.llm_chain = RunnableLambda(self._process_with_llm, name="LLMMetadataExtraction")
    # ... and so on for other RunnableLambdas
    ```
    Langchain >=0.1.17 allows `name` kwarg for `RunnableLambda`. For older versions, you might wrap it:
    ```python
    # from langchain_core.runnables import RunnableConfig
    # from langchain_core.runnables.utils import get_config_list
    # def named_lambda(name, func):
    #     def wrapper(input, config: RunnableConfig, **kwargs):
    #         # Ensure the name is set for this part of the trace
    #         configs = get_config_list(config, 1)
    #         configs[0]["run_name"] = name 
    #         return func(input, config=configs[0], **kwargs) # Pass modified config
    #     return RunnableLambda(wrapper)
    # self.validation_chain = named_lambda("DocumentValidation", self._validate_document)
    ```
    However, the direct `name=` kwarg is much cleaner if your Langchain version supports it.

**How to Test LangSmith Integration:**

1.  **Set the environment variables** mentioned above (`LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`).
2.  **Run your `main.py` script** as usual with production data:
    `python main.py --input_dir /Users/josephott/Documents/bates_number_demo/input_documents --output_dir /ai_docs/output_langsmith_test`
3.  **Go to your LangSmith project URL:** (e.g., `https://smith.langchain.com/o/YOUR_ORG/projects/p/YOUR_PROJECT_ID_OR_NAME`)
4.  You should see new runs appearing in the "Traces" section. Click on a run to see the detailed trace of your LCEL chain execution.
    *   Inspect the inputs/outputs of each step.
    *   Check timings.
    *   Look for any errors.
    *   Verify metadata and tags if you added them.

**What LangSmith Will Show You:**

*   **Overall Chain Structure:** A visual representation of how your `DocumentOrchestrator.processing_chain` executed.
*   **Individual Component Performance:**
    *   Time taken by `LLMCategorizer` calls (and the underlying LLM calls for categorize, summarize, name).
    *   Time taken by `PDFProcessor` calls.
    *   Time taken by `VectorProcessor` (loading, chunking, embedding via `Chroma.add_documents`).
    *   Time taken by `PostgresStorage` calls.
*   **Data Flow:** The exact input and output (Pydantic models, strings, etc.) for each `RunnableLambda` and built-in component. This is incredibly useful for debugging.
*   **Errors:** If any part of the chain throws an unhandled exception, it will be highlighted.
*   **LLM Interactions:** For `ChatOllama` calls, you'll see the exact prompt sent to the model and the raw response received, token counts, etc.

By integrating LangSmith, you'll gain deep insights into your pipeline's behavior, making it much easier to debug, optimize, and monitor. This is especially valuable given the multi-step nature of your document processing.

Remember to create your `context_X.md` file after this phase, detailing the LangSmith setup and your observations from the traces.