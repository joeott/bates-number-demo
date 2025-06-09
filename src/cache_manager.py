"""Caching system for model results to improve performance on similar documents."""

import logging
import hashlib
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
import threading
from collections import OrderedDict

logger = logging.getLogger(__name__)


class DocumentCache:
    """In-memory cache for document processing results."""
    
    def __init__(self, max_size: int = 1000, ttl_hours: int = 24):
        """
        Initialize document cache.
        
        Args:
            max_size: Maximum number of cached entries
            ttl_hours: Time-to-live for cache entries in hours
        """
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0
        }
    
    def _generate_key(self, content_hash: str, model_task: str) -> str:
        """Generate cache key from content hash and model task."""
        return f"{content_hash}:{model_task}"
    
    def get(self, content_hash: str, model_task: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached result.
        
        Args:
            content_hash: Hash of document content
            model_task: Model task type (categorization, visual, etc.)
            
        Returns:
            Cached result or None
        """
        key = self._generate_key(content_hash, model_task)
        
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                # Check if entry is still valid
                if datetime.now() - entry["timestamp"] < self.ttl:
                    # Move to end (LRU)
                    self.cache.move_to_end(key)
                    self.stats["hits"] += 1
                    logger.debug(f"Cache hit for {key}")
                    return entry["data"]
                else:
                    # Expired entry
                    del self.cache[key]
            
            self.stats["misses"] += 1
            return None
    
    def set(self, content_hash: str, model_task: str, data: Dict[str, Any]):
        """
        Store result in cache.
        
        Args:
            content_hash: Hash of document content
            model_task: Model task type
            data: Result data to cache
        """
        key = self._generate_key(content_hash, model_task)
        
        with self.lock:
            # Check if we need to evict
            if len(self.cache) >= self.max_size:
                # Remove oldest entry (LRU)
                self.cache.popitem(last=False)
                self.stats["evictions"] += 1
            
            self.cache[key] = {
                "data": data,
                "timestamp": datetime.now()
            }
            logger.debug(f"Cached result for {key}")
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "evictions": self.stats["evictions"],
                "hit_rate": hit_rate,
                "total_requests": total_requests
            }


class ContentHasher:
    """Service for generating content hashes for documents."""
    
    @staticmethod
    def hash_pdf_content(pdf_path: Path, include_metadata: bool = False) -> str:
        """
        Generate hash of PDF content.
        
        Args:
            pdf_path: Path to PDF file
            include_metadata: Whether to include file metadata in hash
            
        Returns:
            SHA256 hash of content
        """
        hasher = hashlib.sha256()
        
        # Hash file content
        with open(pdf_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        
        # Optionally include metadata
        if include_metadata:
            stat = pdf_path.stat()
            metadata = f"{pdf_path.name}:{stat.st_size}:{stat.st_mtime}"
            hasher.update(metadata.encode())
        
        return hasher.hexdigest()
    
    @staticmethod
    def hash_text_content(text: str) -> str:
        """Generate hash of text content."""
        return hashlib.sha256(text.encode()).hexdigest()


class ModelResultCache:
    """High-level caching service for model results."""
    
    def __init__(self, cache_dir: Optional[Path] = None, 
                 enable_disk_cache: bool = False,
                 memory_cache_size: int = 1000):
        """
        Initialize model result cache.
        
        Args:
            cache_dir: Directory for disk-based cache
            enable_disk_cache: Whether to enable disk caching
            memory_cache_size: Size of in-memory cache
        """
        self.memory_cache = DocumentCache(max_size=memory_cache_size)
        self.hasher = ContentHasher()
        self.enable_disk_cache = enable_disk_cache
        
        if enable_disk_cache and cache_dir:
            self.cache_dir = cache_dir
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None
    
    def get_categorization(self, pdf_path: Path) -> Optional[Dict[str, Any]]:
        """Get cached categorization result."""
        content_hash = self.hasher.hash_pdf_content(pdf_path)
        
        # Check memory cache first
        result = self.memory_cache.get(content_hash, "categorization")
        if result:
            return result
        
        # Check disk cache if enabled
        if self.enable_disk_cache and self.cache_dir:
            result = self._load_from_disk(content_hash, "categorization")
            if result:
                # Store in memory cache for faster access
                self.memory_cache.set(content_hash, "categorization", result)
                return result
        
        return None
    
    def set_categorization(self, pdf_path: Path, result: Dict[str, Any]):
        """Cache categorization result."""
        content_hash = self.hasher.hash_pdf_content(pdf_path)
        
        # Store in memory cache
        self.memory_cache.set(content_hash, "categorization", result)
        
        # Store on disk if enabled
        if self.enable_disk_cache and self.cache_dir:
            self._save_to_disk(content_hash, "categorization", result)
    
    def get_model_result(self, pdf_path: Path, model_task: str) -> Optional[Dict[str, Any]]:
        """Get cached result for any model task."""
        content_hash = self.hasher.hash_pdf_content(pdf_path)
        
        # Check memory cache
        result = self.memory_cache.get(content_hash, model_task)
        if result:
            return result
        
        # Check disk cache
        if self.enable_disk_cache and self.cache_dir:
            result = self._load_from_disk(content_hash, model_task)
            if result:
                self.memory_cache.set(content_hash, model_task, result)
                return result
        
        return None
    
    def set_model_result(self, pdf_path: Path, model_task: str, result: Dict[str, Any]):
        """Cache result for any model task."""
        content_hash = self.hasher.hash_pdf_content(pdf_path)
        
        # Store in memory
        self.memory_cache.set(content_hash, model_task, result)
        
        # Store on disk if enabled
        if self.enable_disk_cache and self.cache_dir:
            self._save_to_disk(content_hash, model_task, result)
    
    def _get_disk_cache_path(self, content_hash: str, model_task: str) -> Path:
        """Get path for disk cache file."""
        # Use subdirectories to avoid too many files in one directory
        subdir = content_hash[:2]
        filename = f"{content_hash}_{model_task}.pkl"
        return self.cache_dir / subdir / filename
    
    def _save_to_disk(self, content_hash: str, model_task: str, data: Dict[str, Any]):
        """Save result to disk cache."""
        try:
            cache_path = self._get_disk_cache_path(content_hash, model_task)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            cache_entry = {
                "data": data,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_entry, f)
                
            logger.debug(f"Saved to disk cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save to disk cache: {e}")
    
    def _load_from_disk(self, content_hash: str, model_task: str) -> Optional[Dict[str, Any]]:
        """Load result from disk cache."""
        try:
            cache_path = self._get_disk_cache_path(content_hash, model_task)
            if not cache_path.exists():
                return None
            
            with open(cache_path, 'rb') as f:
                cache_entry = pickle.load(f)
            
            # Check if entry is still valid (24 hour TTL)
            timestamp = datetime.fromisoformat(cache_entry["timestamp"])
            if datetime.now() - timestamp > timedelta(hours=24):
                # Expired, remove file
                cache_path.unlink()
                return None
            
            logger.debug(f"Loaded from disk cache: {cache_path}")
            return cache_entry["data"]
            
        except Exception as e:
            logger.warning(f"Failed to load from disk cache: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self.memory_cache.get_stats()
        
        if self.enable_disk_cache and self.cache_dir:
            # Count disk cache files
            cache_files = list(self.cache_dir.rglob("*.pkl"))
            stats["disk_cache_files"] = len(cache_files)
            stats["disk_cache_enabled"] = True
        else:
            stats["disk_cache_enabled"] = False
        
        return stats
    
    def clear_memory_cache(self):
        """Clear in-memory cache."""
        self.memory_cache.clear()
    
    def clear_disk_cache(self):
        """Clear disk cache."""
        if self.cache_dir and self.cache_dir.exists():
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Disk cache cleared")


class CachedLLMHandler:
    """Wrapper for LLMHandler that adds caching."""
    
    def __init__(self, llm_handler, cache: ModelResultCache):
        self.llm_handler = llm_handler
        self.cache = cache
    
    def categorize_document(self, pdf_path: Path) -> Dict[str, Any]:
        """Categorize document with caching."""
        # Check cache first
        cached_result = self.cache.get_categorization(pdf_path)
        if cached_result:
            logger.info(f"Using cached categorization for {pdf_path.name}")
            return cached_result
        
        # Not in cache, run categorization
        result = self.llm_handler.categorize_document(pdf_path)
        
        # Cache the result
        self.cache.set_categorization(pdf_path, result)
        
        return result
    
    def categorize_with_model(self, pdf_path: Path, model_task: str) -> Dict[str, Any]:
        """Categorize with specific model and caching."""
        # Check cache
        cached_result = self.cache.get_model_result(pdf_path, model_task)
        if cached_result:
            logger.info(f"Using cached {model_task} result for {pdf_path.name}")
            return cached_result
        
        # Run model
        if hasattr(self.llm_handler, 'categorize_with_model'):
            result = self.llm_handler.categorize_with_model(pdf_path, model_task)
        else:
            result = self.llm_handler.categorize_document(pdf_path)
        
        # Cache result
        self.cache.set_model_result(pdf_path, model_task, result)
        
        return result
    
    def process_document_parallel(self, filename: str) -> Dict[str, Any]:
        """Process document in parallel with caching."""
        # For caching, we need the actual file path, but we only have the filename
        # So we'll cache based on filename alone (not ideal but works)
        cache_key = f"parallel:{filename}"
        
        # Check memory cache using filename as key
        if hasattr(self.cache.memory_cache, 'get'):
            cached = self.cache.memory_cache.get(cache_key, "parallel")
            if cached:
                logger.info(f"Using cached parallel results for {filename}")
                return cached
        
        # Not in cache, run parallel processing
        result = self.llm_handler.process_document_parallel(filename)
        
        # Cache the result using filename
        if hasattr(self.cache.memory_cache, 'set'):
            self.cache.memory_cache.set(cache_key, "parallel", result)
        
        return result
    
    def __getattr__(self, name):
        """Forward any undefined attributes to the wrapped handler."""
        return getattr(self.llm_handler, name)