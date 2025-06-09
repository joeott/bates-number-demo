"""Batch processing optimization for improved throughput."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import time
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Result of processing a batch of documents."""
    successful: int
    failed: int
    total_time: float
    avg_time_per_doc: float
    errors: List[Tuple[Path, str]]
    results: List[Dict[str, Any]]


class BatchProcessor:
    """Optimized batch processing for document pipeline."""
    
    def __init__(self, 
                 orchestrator,
                 max_workers: int = 4,
                 batch_size: int = 10,
                 enable_prefetch: bool = True):
        """
        Initialize batch processor.
        
        Args:
            orchestrator: DocumentOrchestrator instance
            max_workers: Maximum concurrent processing threads
            batch_size: Documents per batch
            enable_prefetch: Whether to prefetch next batch while processing
        """
        self.orchestrator = orchestrator
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.enable_prefetch = enable_prefetch
        self._prefetch_cache = {}
    
    def process_documents(self, documents: List[Path], 
                         start_bates: int = 1,
                         start_exhibit: int = 1) -> BatchResult:
        """
        Process documents in optimized batches.
        
        Args:
            documents: List of document paths
            start_bates: Starting Bates number
            start_exhibit: Starting exhibit number
            
        Returns:
            BatchResult with processing summary
        """
        start_time = time.time()
        total_docs = len(documents)
        successful = 0
        failed = 0
        errors = []
        all_results = []
        
        # Split into batches
        batches = [documents[i:i + self.batch_size] 
                  for i in range(0, total_docs, self.batch_size)]
        
        logger.info(f"Processing {total_docs} documents in {len(batches)} batches")
        
        current_bates = start_bates
        current_exhibit = start_exhibit
        
        for batch_idx, batch in enumerate(batches):
            batch_num = batch_idx + 1
            logger.info(f"Processing batch {batch_num}/{len(batches)} ({len(batch)} documents)")
            
            # Process batch with parallel categorization
            batch_results = self._process_batch_parallel(
                batch, current_bates, current_exhibit
            )
            
            # Update counters and collect results
            for result in batch_results:
                if result.get("success"):
                    successful += 1
                    all_results.append(result)
                    # Update counters based on successful processing
                    if "end_bates" in result:
                        current_bates = result["next_bates"]
                    if "exhibit_number" in result:
                        current_exhibit = result["exhibit_number"] + 1
                else:
                    failed += 1
                    errors.append((result["path"], result.get("error", "Unknown error")))
            
            # Prefetch next batch if enabled
            if self.enable_prefetch and batch_idx < len(batches) - 1:
                self._prefetch_batch(batches[batch_idx + 1])
        
        total_time = time.time() - start_time
        avg_time = total_time / total_docs if total_docs > 0 else 0
        
        return BatchResult(
            successful=successful,
            failed=failed,
            total_time=total_time,
            avg_time_per_doc=avg_time,
            errors=errors,
            results=all_results
        )
    
    def _process_batch_parallel(self, batch: List[Path], 
                               start_bates: int, 
                               start_exhibit: int) -> List[Dict[str, Any]]:
        """Process a batch with parallel categorization."""
        results = []
        
        # Phase 1: Parallel categorization and text extraction
        categorization_results = self._parallel_categorize(batch)
        
        # Phase 2: Sequential Bates numbering (must be sequential)
        current_bates = start_bates
        current_exhibit = start_exhibit
        
        for idx, (doc_path, cat_result) in enumerate(zip(batch, categorization_results)):
            if cat_result.get("error"):
                results.append({
                    "path": doc_path,
                    "success": False,
                    "error": cat_result["error"]
                })
                continue
            
            # Process document with categorization result
            try:
                # Use cached data if available
                cached_data = self._prefetch_cache.get(str(doc_path), {})
                
                result = self.orchestrator.process_single_document(
                    pdf_path=doc_path,
                    current_bates_number=current_bates,
                    current_exhibit_number=current_exhibit,
                    categorization_result=cat_result.get("category"),
                    cached_text=cached_data.get("text"),
                    cached_metadata=cached_data.get("metadata")
                )
                
                if result:
                    result["path"] = doc_path
                    result["success"] = True
                    results.append(result)
                    
                    # Update counters
                    if "next_bates" in result:
                        current_bates = result["next_bates"]
                    current_exhibit += 1
                else:
                    results.append({
                        "path": doc_path,
                        "success": False,
                        "error": "Processing returned no result"
                    })
                    
            except Exception as e:
                logger.error(f"Error processing {doc_path}: {e}")
                results.append({
                    "path": doc_path,
                    "success": False,
                    "error": str(e)
                })
        
        # Clear prefetch cache for processed documents
        for doc_path in batch:
            self._prefetch_cache.pop(str(doc_path), None)
        
        return results
    
    def _parallel_categorize(self, documents: List[Path]) -> List[Dict[str, Any]]:
        """Categorize multiple documents in parallel."""
        results = [None] * len(documents)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all categorization tasks
            future_to_idx = {
                executor.submit(self._categorize_document, doc): idx
                for idx, doc in enumerate(documents)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Categorization failed: {e}")
                    results[idx] = {"error": str(e)}
        
        return results
    
    def _categorize_document(self, doc_path: Path) -> Dict[str, Any]:
        """Categorize a single document with intelligent model selection."""
        try:
            # Analyze document if analyzer is available
            analysis = None
            model_task = "categorization"
            
            if hasattr(self.orchestrator, 'document_analyzer') and self.orchestrator.document_analyzer:
                analysis = self.orchestrator.document_analyzer.analyze_document(doc_path)
                recommendations = self.orchestrator.document_analyzer.get_recommended_models(analysis)
                model_task = recommendations.get("categorization", "categorization")
                logger.debug(f"Document analysis for {doc_path.name}: "
                           f"type={analysis.get('dominant_type')}, "
                           f"recommended_model={model_task}")
            
            # Check if LLM handler supports multi-model
            if hasattr(self.orchestrator.llm_categorizer, 'categorize_with_model'):
                # Use recommended model based on analysis
                category = self.orchestrator.llm_categorizer.categorize_with_model(
                    doc_path, 
                    model_task=model_task
                )
            else:
                # Fallback to standard categorization
                category = self.orchestrator.llm_categorizer.categorize_document(doc_path)
            
            result = {"category": category}
            if analysis:
                result["document_analysis"] = analysis
                
            return result
        except Exception as e:
            return {"error": str(e)}
    
    def _prefetch_batch(self, documents: List[Path]):
        """Prefetch text and metadata for next batch."""
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(self._prefetch_document, doc): doc
                for doc in documents
            }
            
            for future in as_completed(futures):
                doc = futures[future]
                try:
                    self._prefetch_cache[str(doc)] = future.result()
                except Exception as e:
                    logger.debug(f"Prefetch failed for {doc}: {e}")
    
    def _prefetch_document(self, doc_path: Path) -> Dict[str, Any]:
        """Prefetch document text and metadata."""
        result = {}
        
        try:
            # Extract text if vector processor is available
            if self.orchestrator.vector_processor:
                text = self.orchestrator.vector_processor.extract_text(doc_path)
                result["text"] = text
            
            # Extract basic metadata
            result["metadata"] = {
                "filename": doc_path.name,
                "size": doc_path.stat().st_size,
                "modified": datetime.fromtimestamp(doc_path.stat().st_mtime)
            }
        except Exception as e:
            logger.debug(f"Prefetch error for {doc_path}: {e}")
        
        return result


class AdaptiveBatchProcessor(BatchProcessor):
    """Batch processor that adapts batch size based on performance."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.performance_history = []
        self.min_batch_size = 5
        self.max_batch_size = 50
    
    def process_documents(self, documents: List[Path], 
                         start_bates: int = 1,
                         start_exhibit: int = 1) -> BatchResult:
        """Process with adaptive batch sizing."""
        # Start with configured batch size
        current_batch_size = self.batch_size
        
        start_time = time.time()
        total_docs = len(documents)
        successful = 0
        failed = 0
        errors = []
        all_results = []
        
        current_bates = start_bates
        current_exhibit = start_exhibit
        
        idx = 0
        batch_num = 0
        
        while idx < total_docs:
            # Get next batch with current size
            batch_end = min(idx + current_batch_size, total_docs)
            batch = documents[idx:batch_end]
            batch_num += 1
            
            logger.info(f"Processing adaptive batch {batch_num} "
                       f"(size: {len(batch)}, total: {idx}/{total_docs})")
            
            # Process batch
            batch_start_time = time.time()
            batch_results = self._process_batch_parallel(
                batch, current_bates, current_exhibit
            )
            batch_time = time.time() - batch_start_time
            
            # Analyze performance
            batch_successful = sum(1 for r in batch_results if r.get("success"))
            batch_failed = len(batch_results) - batch_successful
            
            successful += batch_successful
            failed += batch_failed
            
            # Update counters and collect results
            for result in batch_results:
                if result.get("success"):
                    all_results.append(result)
                    if "end_bates" in result:
                        current_bates = result["next_bates"]
                    if "exhibit_number" in result:
                        current_exhibit = result["exhibit_number"] + 1
                else:
                    errors.append((result["path"], result.get("error", "Unknown error")))
            
            # Adapt batch size based on performance
            current_batch_size = self._adapt_batch_size(
                current_batch_size,
                len(batch),
                batch_time,
                batch_successful / len(batch) if batch else 0
            )
            
            idx = batch_end
        
        total_time = time.time() - start_time
        avg_time = total_time / total_docs if total_docs > 0 else 0
        
        return BatchResult(
            successful=successful,
            failed=failed,
            total_time=total_time,
            avg_time_per_doc=avg_time,
            errors=errors,
            results=all_results
        )
    
    def _adapt_batch_size(self, current_size: int, batch_size: int, 
                         batch_time: float, success_rate: float) -> int:
        """Adapt batch size based on performance metrics."""
        # Calculate throughput
        throughput = batch_size / batch_time if batch_time > 0 else 0
        
        # Record performance
        self.performance_history.append({
            "size": batch_size,
            "throughput": throughput,
            "success_rate": success_rate,
            "time": batch_time
        })
        
        # Keep only recent history
        if len(self.performance_history) > 10:
            self.performance_history.pop(0)
        
        # Adapt based on performance
        if success_rate < 0.8:  # High failure rate
            # Reduce batch size
            new_size = max(self.min_batch_size, int(current_size * 0.8))
        elif throughput > 2.0 and success_rate > 0.95:  # Good performance
            # Increase batch size
            new_size = min(self.max_batch_size, int(current_size * 1.2))
        else:
            # Keep current size
            new_size = current_size
        
        if new_size != current_size:
            logger.info(f"Adapting batch size: {current_size} -> {new_size} "
                       f"(throughput: {throughput:.2f}, success: {success_rate:.1%})")
        
        return new_size