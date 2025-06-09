"""
Document processing orchestrator using LangChain LCEL chains.
Provides a declarative pipeline for the complete document processing workflow.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import csv

# LangChain imports
from langchain_core.runnables import (
    RunnableSequence, 
    RunnableParallel, 
    RunnableLambda,
    RunnablePassthrough,
    RunnableBranch
)
from langchain_core.output_parsers import BaseOutputParser
from pydantic import BaseModel, Field

# Local imports
from src.llm_handler import LLMCategorizer
from src.pdf_processor import PDFProcessor
from src.vector_processor import VectorProcessor
from src.db_storage import PostgresStorage
from src.utils import sanitize_filename, ensure_dir_exists
from src import config

logger = logging.getLogger(__name__)


# Pydantic models for chain data flow
class DocumentInput(BaseModel):
    """Input for document processing chain."""
    file_path: Path
    bates_counter: int
    exhibit_counter: int
    
    class Config:
        arbitrary_types_allowed = True


class DocumentMetadata(BaseModel):
    """Metadata extracted from LLM processing."""
    category: str
    summary: str
    descriptive_name: str
    file_path: Path
    
    class Config:
        arbitrary_types_allowed = True


class BatesResult(BaseModel):
    """Result from Bates numbering."""
    bates_start: str
    bates_end: str
    next_counter: int
    bates_pdf_path: Path
    metadata: DocumentMetadata
    
    class Config:
        arbitrary_types_allowed = True


class ExhibitResult(BaseModel):
    """Result from exhibit marking."""
    exhibit_number: int
    exhibit_pdf_path: Path
    exhibit_pdf_name: str
    bates_info: BatesResult
    
    class Config:
        arbitrary_types_allowed = True


class ProcessingResult(BaseModel):
    """Complete processing result for a document."""
    success: bool
    exhibit_id: str
    original_filename: str
    final_filename: str
    category: str
    summary: str
    bates_range: str
    exhibit_path: Optional[Path] = None
    error: Optional[str] = None
    vector_chunks: Optional[int] = None
    postgres_stored: bool = False
    
    class Config:
        arbitrary_types_allowed = True


class DocumentOrchestrator:
    """
    Orchestrates document processing using LangChain LCEL chains.
    """
    
    def __init__(
        self,
        llm_categorizer: LLMCategorizer,
        pdf_processor: PDFProcessor,
        vector_processor: Optional[VectorProcessor] = None,
        postgres_storage: Optional[PostgresStorage] = None,
        output_dir: Path = None,
        exhibit_prefix: str = "Exhibit ",
        bates_prefix: str = "",
        bates_digits: int = 6
    ):
        """Initialize the orchestrator with required components."""
        self.llm_categorizer = llm_categorizer
        self.pdf_processor = pdf_processor
        self.vector_processor = vector_processor
        self.postgres_storage = postgres_storage
        self.output_dir = output_dir or Path(config.DEFAULT_OUTPUT_DIR)
        self.exhibit_prefix = exhibit_prefix
        self.bates_prefix = bates_prefix
        self.bates_digits = bates_digits
        self.document_analyzer = None  # Will be set later if needed
        
        # Create output directories
        self.bates_output_dir = self.output_dir / config.DEFAULT_BATES_SUBDIR
        self.exhibits_output_dir = self.output_dir / config.DEFAULT_EXHIBITS_SUBDIR
        ensure_dir_exists(self.bates_output_dir)
        ensure_dir_exists(self.exhibits_output_dir)
        
        # Build the processing chain
        self._build_chain()
    
    def _build_chain(self):
        """Build the LCEL processing chain."""
        
        # Document validation
        self.validation_chain = RunnableLambda(self._validate_document, name="DocumentValidation")
        
        # LLM processing chain
        self.llm_chain = RunnableLambda(self._process_with_llm, name="LLMMetadataExtraction")
        
        # Bates numbering chain
        self.bates_chain = RunnableLambda(self._apply_bates_numbering, name="BatesNumbering")
        
        # Exhibit marking chain
        self.exhibit_chain = RunnableLambda(self._apply_exhibit_marking, name="ExhibitMarking")
        
        # Vector processing branch (conditional)
        self.vector_branch = RunnableBranch(
            (lambda x: self.vector_processor is not None and x.get("success", False), 
             RunnableLambda(self._process_vectors, name="VectorProcessing")),
            RunnablePassthrough()
        )
        
        # PostgreSQL storage branch (conditional)
        self.postgres_branch = RunnableBranch(
            (lambda x: self.postgres_storage is not None and x.get("success", False),
             RunnableLambda(self._store_in_postgres, name="PostgreSQLStorage")),
            RunnablePassthrough()
        )
        
        # Complete processing chain
        self.processing_chain = (
            self.validation_chain
            | self.llm_chain
            | self.bates_chain
            | self.exhibit_chain
            | self.vector_branch
            | self.postgres_branch
            | RunnableLambda(self._finalize_result, name="FinalizeResult")
        )
        
        # Error handling wrapper
        self.safe_processing_chain = RunnableLambda(self._safe_process, name="SafeDocumentProcessor")
    
    def _validate_document(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the input document."""
        doc_input = DocumentInput(**input_data)
        
        if not doc_input.file_path.exists():
            raise ValueError(f"File not found: {doc_input.file_path}")
        
        if not doc_input.file_path.suffix.lower() == '.pdf':
            raise ValueError(f"Not a PDF file: {doc_input.file_path}")
        
        logger.info(f"Validating document: {doc_input.file_path.name}")
        return {
            "input": doc_input,
            "success": True
        }
    
    def _process_with_llm(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process document with LLM for categorization and metadata."""
        doc_input = data["input"]
        
        logger.info(f"LLM processing: {doc_input.file_path.name}")
        
        # Use parallel LLM processing
        llm_results = self.llm_categorizer.process_document_parallel(
            doc_input.file_path.name
        )
        
        metadata = DocumentMetadata(
            category=llm_results["category"],
            summary=llm_results["summary"],
            descriptive_name=llm_results["descriptive_name"],
            file_path=doc_input.file_path
        )
        
        data["metadata"] = metadata
        return data
    
    def _apply_bates_numbering(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Bates numbering to the document."""
        doc_input = data["input"]
        metadata = data["metadata"]
        
        # Generate Bates output path
        sanitized_filename = sanitize_filename(doc_input.file_path.stem)
        bates_pdf_name = f"{sanitized_filename}_BATES.pdf"
        bates_output_path = self.bates_output_dir / bates_pdf_name
        
        # Apply Bates stamping
        bates_start, bates_end, next_counter = self.pdf_processor.bates_stamp_pdf(
            input_pdf_path=doc_input.file_path,
            output_pdf_path=bates_output_path,
            start_bates_number=doc_input.bates_counter,
            bates_prefix=self.bates_prefix,
            num_digits=self.bates_digits
        )
        
        bates_result = BatesResult(
            bates_start=bates_start,
            bates_end=bates_end,
            next_counter=next_counter,
            bates_pdf_path=bates_output_path,
            metadata=metadata
        )
        
        data["bates_result"] = bates_result
        data["next_bates"] = next_counter
        logger.info(f"Bates numbered: {bates_start}-{bates_end}")
        return data
    
    def _apply_exhibit_marking(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply exhibit marking to the document."""
        doc_input = data["input"]
        bates_result = data["bates_result"]
        
        # Create category directory
        category_dir = self.exhibits_output_dir / sanitize_filename(
            bates_result.metadata.category.lower().replace(" ", "_")
        )
        ensure_dir_exists(category_dir)
        
        # Generate exhibit filename
        sanitized_name = sanitize_filename(bates_result.metadata.descriptive_name)
        exhibit_pdf_name = f"{self.exhibit_prefix}{doc_input.exhibit_counter} - {sanitized_name}.pdf"
        exhibit_output_path = category_dir / exhibit_pdf_name
        
        # Apply exhibit marking
        exhibit_id = f"{self.exhibit_prefix}{doc_input.exhibit_counter}"
        success = self.pdf_processor.exhibit_mark_pdf(
            input_pdf_path=bates_result.bates_pdf_path,
            output_pdf_path=exhibit_output_path,
            exhibit_id=exhibit_id
        )
        
        if success:
            exhibit_result = ExhibitResult(
                exhibit_number=doc_input.exhibit_counter,
                exhibit_pdf_path=exhibit_output_path,
                exhibit_pdf_name=exhibit_pdf_name,
                bates_info=bates_result
            )
            data["exhibit_result"] = exhibit_result
            data["next_exhibit"] = doc_input.exhibit_counter + 1
            logger.info(f"Exhibit marked: {exhibit_pdf_name}")
        else:
            data["success"] = False
            data["error"] = "Failed to mark exhibit"
            
        return data
    
    def _process_vectors(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process document for vector search."""
        exhibit_result = data.get("exhibit_result")
        if not exhibit_result:
            return data
            
        try:
            logger.info("Processing vectors...")
            chunk_ids, full_text, page_texts = self.vector_processor.process_document(
                exhibit_result.exhibit_pdf_path,
                exhibit_number=exhibit_result.exhibit_number,
                category=exhibit_result.bates_info.metadata.category,
                bates_start=exhibit_result.bates_info.bates_start,
                bates_end=exhibit_result.bates_info.bates_end
            )
            
            data["vector_chunks"] = len(chunk_ids)
            data["full_text"] = full_text
            data["page_texts"] = page_texts
            logger.info(f"Created {len(chunk_ids)} vector chunks")
            
        except Exception as e:
            logger.error(f"Vector processing failed: {e}")
            # Non-fatal error, continue processing
            
        return data
    
    def _store_in_postgres(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Store document text in PostgreSQL."""
        exhibit_result = data.get("exhibit_result")
        if not exhibit_result:
            return data
            
        try:
            # Get text from vector processing or extract it
            full_text = data.get("full_text", "")
            page_texts = data.get("page_texts", [])
            
            if not full_text and not self.vector_processor:
                # Extract text if we didn't do vector processing
                from src.vector_processor import PDFToLangChainLoader
                loader = PDFToLangChainLoader(str(exhibit_result.exhibit_pdf_path))
                documents = loader.load()
                page_texts = [doc.page_content for doc in documents]
                full_text = "\n\n".join(page_texts)
            
            if full_text:
                document_id = self.postgres_storage.store_document_text(
                    exhibit_id=exhibit_result.exhibit_number,
                    original_filename=data["input"].file_path.name,
                    exhibit_filename=exhibit_result.exhibit_pdf_name,
                    bates_start=exhibit_result.bates_info.bates_start,
                    bates_end=exhibit_result.bates_info.bates_end,
                    category=exhibit_result.bates_info.metadata.category,
                    full_text=full_text,
                    page_texts=page_texts if config.STORE_PAGE_LEVEL_TEXT else None
                )
                data["postgres_stored"] = True
                logger.info(f"Stored in PostgreSQL with ID: {document_id}")
                
        except Exception as e:
            logger.error(f"PostgreSQL storage failed: {e}")
            # Non-fatal error, continue processing
            
        return data
    
    def _finalize_result(self, data: Dict[str, Any]) -> ProcessingResult:
        """Finalize the processing result."""
        doc_input = data.get("input")
        exhibit_result = data.get("exhibit_result")
        bates_result = data.get("bates_result")
        
        if exhibit_result and bates_result:
            result = ProcessingResult(
                success=True,
                exhibit_id=f"{self.exhibit_prefix}{exhibit_result.exhibit_number}",
                original_filename=doc_input.file_path.name,
                final_filename=exhibit_result.exhibit_pdf_name,
                category=bates_result.metadata.category,
                summary=bates_result.metadata.summary,
                bates_range=f"{bates_result.bates_start}-{bates_result.bates_end}",
                exhibit_path=exhibit_result.exhibit_pdf_path,
                vector_chunks=data.get("vector_chunks"),
                postgres_stored=data.get("postgres_stored", False)
            )
        else:
            result = ProcessingResult(
                success=False,
                exhibit_id="",
                original_filename=doc_input.file_path.name if doc_input else "Unknown",
                final_filename="",
                category="",
                summary="",
                bates_range="",
                error=data.get("error", "Processing failed")
            )
            
        return result
    
    def _safe_process(self, input_data: Dict[str, Any]) -> ProcessingResult:
        """Safely process a document with error handling."""
        try:
            return self.processing_chain.invoke(input_data)
        except Exception as e:
            logger.error(f"Processing failed for {input_data.get('file_path', 'unknown')}: {e}")
            return ProcessingResult(
                success=False,
                exhibit_id="",
                original_filename=str(input_data.get("file_path", "Unknown")),
                final_filename="",
                category="",
                summary="",
                bates_range="",
                error=str(e)
            )
    
    def process_document(
        self, 
        file_path: Path, 
        bates_counter: int, 
        exhibit_counter: int,
        run_config: Optional[Dict] = None
    ) -> Tuple[ProcessingResult, int, int]:
        """
        Process a single document through the pipeline.
        
        Args:
            file_path: Path to the document to process
            bates_counter: Current Bates counter value
            exhibit_counter: Current exhibit counter value
            run_config: Optional LangChain config for tracing (metadata, tags, run_name)
        
        Returns:
            Tuple of (result, next_bates_counter, next_exhibit_counter)
        """
        input_data = {
            "file_path": file_path,
            "bates_counter": bates_counter,
            "exhibit_counter": exhibit_counter
        }
        
        # Process the document using the safe processing chain
        result = self.safe_processing_chain.invoke(input_data, config=run_config)
        
        # Calculate next counters based on processing results
        if result.success:
            # Calculate next Bates counter based on the pages processed
            # Extract the ending Bates number from the result
            bates_range = result.bates_range
            if bates_range and "-" in bates_range:
                # Extract numeric part from ending Bates (e.g., "TEST000005" -> 5)
                end_bates = bates_range.split("-")[1]
                # Remove prefix to get the number
                bates_num_str = end_bates.replace(self.bates_prefix, "").lstrip("0") or "0"
                try:
                    next_bates = int(bates_num_str) + 1
                except ValueError:
                    next_bates = bates_counter + 1
            else:
                next_bates = bates_counter + 1
                
            # Next exhibit counter
            next_exhibit = exhibit_counter + 1
        else:
            # On failure, don't advance counters
            next_bates = bates_counter
            next_exhibit = exhibit_counter
            
        return result, next_bates, next_exhibit
    
    def _process_with_state_capture(self, chain, input_data):
        """Process with state capture for counter tracking."""
        try:
            # Modify chains to capture state
            original_bates = self.bates_chain
            original_exhibit = self.exhibit_chain
            
            self.bates_chain = RunnableLambda(
                lambda x: self._capture_bates_state(original_bates, x)
            )
            self.exhibit_chain = RunnableLambda(
                lambda x: self._capture_exhibit_state(original_exhibit, x)
            )
            
            # Rebuild the chain
            self._build_chain()
            
            # Process
            result = self.processing_chain.invoke(input_data)
            
            # Restore original chains
            self.bates_chain = original_bates
            self.exhibit_chain = original_exhibit
            
            return result
        except Exception as e:
            return self._safe_process(input_data)
    
    def _capture_bates_state(self, original_chain, data):
        """Capture Bates counter state."""
        result = original_chain.invoke(data)
        if "next_bates" in result:
            self._chain_state["next_bates"] = result["next_bates"]
        return result
    
    def _capture_exhibit_state(self, original_chain, data):
        """Capture exhibit counter state."""
        result = original_chain.invoke(data)
        if "next_exhibit" in result:
            self._chain_state["next_exhibit"] = result["next_exhibit"]
        return result
    
    def process_single_document(
        self, 
        pdf_path: Path, 
        current_bates_number: int,
        current_exhibit_number: int,
        categorization_result: Optional[Dict[str, Any]] = None,
        cached_text: Optional[str] = None,
        cached_metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single document with optional cached data.
        
        Args:
            pdf_path: Path to the PDF document
            current_bates_number: Starting Bates number
            current_exhibit_number: Current exhibit number
            categorization_result: Pre-computed categorization result
            cached_text: Pre-extracted text (for vector processing)
            cached_metadata: Pre-computed metadata
            
        Returns:
            Processing result dictionary
        """
        try:
            # Prepare input data
            input_data = {
                "file_path": pdf_path,
                "bates_counter": current_bates_number,
                "exhibit_counter": current_exhibit_number
            }
            
            # If we have cached categorization, inject it into the process
            if categorization_result:
                # Create a modified chain that skips LLM categorization
                cached_chain = self._create_cached_processing_chain(
                    categorization_result, cached_text, cached_metadata
                )
                result = cached_chain.invoke(input_data)
            else:
                # Use normal processing chain
                result = self.safe_processing_chain.invoke(input_data)
            
            # Extract counter information
            if result.success:
                bates_range = result.bates_range
                if bates_range and "-" in bates_range:
                    end_bates = bates_range.split("-")[1]
                    bates_num_str = end_bates.replace(self.bates_prefix, "").lstrip("0") or "0"
                    try:
                        next_bates = int(bates_num_str) + 1
                    except ValueError:
                        next_bates = current_bates_number + 1
                else:
                    next_bates = current_bates_number + 1
                
                return {
                    "success": True,
                    "exhibit_id": result.exhibit_id,
                    "exhibit_number": current_exhibit_number,
                    "category": result.category,
                    "summary": result.summary,
                    "bates_range": result.bates_range,
                    "next_bates": next_bates,
                    "exhibit_path": result.exhibit_path,
                    "vector_chunks": result.vector_chunks,
                    "postgres_stored": result.postgres_stored
                }
            else:
                return {
                    "success": False,
                    "error": result.error or "Unknown error",
                    "next_bates": current_bates_number,
                    "exhibit_number": current_exhibit_number
                }
                
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "next_bates": current_bates_number,
                "exhibit_number": current_exhibit_number
            }
    
    def _create_cached_processing_chain(
        self, 
        categorization_result: Dict[str, Any],
        cached_text: Optional[str] = None,
        cached_metadata: Optional[Dict[str, Any]] = None
    ):
        """Create a processing chain that uses cached categorization."""
        # Create a lambda that injects the cached results
        def inject_cached_results(data):
            doc_input = DocumentInput(**data)
            data["input"] = doc_input
            
            # Inject cached categorization
            metadata = DocumentMetadata(
                category=categorization_result.get("category", "Uncategorized"),
                summary=categorization_result.get("summary", ""),
                descriptive_name=categorization_result.get("descriptive_name", doc_input.file_path.stem),
                file_path=doc_input.file_path
            )
            data["metadata"] = metadata
            
            # Inject cached text if available
            if cached_text:
                data["cached_text"] = cached_text
            if cached_metadata:
                data["cached_metadata"] = cached_metadata
                
            return data
        
        # Build chain that skips LLM categorization
        return (
            RunnableLambda(inject_cached_results)
            | self.bates_chain
            | self.exhibit_chain
            | self.vector_chain
            | self.postgres_chain
            | RunnableLambda(self._finalize_result)
        )
    
    def process_batch(
        self, 
        file_paths: List[Path], 
        starting_bates: int = 1,
        starting_exhibit: int = 1,
        run_config_template: Optional[Dict] = None
    ) -> List[ProcessingResult]:
        """
        Process multiple documents in batch.
        
        Args:
            file_paths: List of document paths to process
            starting_bates: Starting Bates number
            starting_exhibit: Starting exhibit number  
            run_config_template: Optional base config for LangSmith tracing
        
        Uses RunnableParallel for concurrent processing where possible.
        """
        results = []
        bates_counter = starting_bates
        exhibit_counter = starting_exhibit
        
        # Process documents sequentially (due to counter dependencies)
        # Future enhancement: pre-calculate counters for true parallel processing
        for i, file_path in enumerate(file_paths):
            # Create run-specific config with metadata and tags
            if run_config_template or config.LANGSMITH_TRACING:
                current_run_config = {
                    "metadata": {
                        "document_path": str(file_path),
                        "original_filename": file_path.name,
                        "batch_index": i + 1,
                        "total_documents": len(file_paths),
                        "starting_bates": bates_counter,
                        "starting_exhibit": exhibit_counter
                    },
                    "tags": [
                        "batch_processing",
                        f"doc_{i+1}_of_{len(file_paths)}",
                        f"bates_start_{bates_counter}",
                        f"exhibit_{exhibit_counter}"
                    ],
                    "run_name": f"ProcessDoc-{file_path.stem}"
                }
                
                # Merge with template if provided
                if run_config_template:
                    current_run_config["metadata"].update(run_config_template.get("metadata", {}))
                    current_run_config["tags"].extend(run_config_template.get("tags", []))
                    if "run_name" in run_config_template:
                        current_run_config["run_name"] = f"{run_config_template['run_name']}-{file_path.stem}"
            else:
                current_run_config = None
            
            result, bates_counter, exhibit_counter = self.process_document(
                file_path, bates_counter, exhibit_counter, run_config=current_run_config
            )
            results.append(result)
        
        # Store final counters for external access
        self._next_bates = bates_counter
        self._next_exhibit = exhibit_counter
            
        return results
    
    def generate_csv_log(self, results: List[ProcessingResult], output_path: Path):
        """Generate CSV log of processing results."""
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'Exhibit ID', 'Original Filename', 'Final Filename',
                'Category', 'Summary', 'Bates Range', 'Status',
                'Vector Chunks', 'PostgreSQL Stored', 'Error'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                writer.writerow({
                    'Exhibit ID': result.exhibit_id,
                    'Original Filename': result.original_filename,
                    'Final Filename': result.final_filename,
                    'Category': result.category,
                    'Summary': result.summary,
                    'Bates Range': result.bates_range,
                    'Status': 'Success' if result.success else 'Failed',
                    'Vector Chunks': result.vector_chunks or '',
                    'PostgreSQL Stored': 'Yes' if result.postgres_stored else 'No',
                    'Error': result.error or ''
                })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "categories": {}
        }
        
        if self.vector_processor:
            vector_stats = self.vector_processor.get_stats()
            stats["vector_chunks"] = vector_stats.get("total_chunks", 0)
            stats["vector_categories"] = vector_stats.get("categories", [])
            
        return stats