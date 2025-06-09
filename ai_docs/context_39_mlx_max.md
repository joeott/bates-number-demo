# Context 39: MLX Models for M4 Max Legal Document Processing

## Executive Summary

This document outlines the optimal MLX model selection for legal document processing on MacBook Pro M4 Max with 128GB RAM. While Ollama doesn't currently support MLX backend directly, we provide a hybrid approach leveraging both frameworks for maximum performance.

## Hardware Capabilities

### M4 Max Specifications
- **Unified Memory**: 128GB (no CPU/GPU transfer overhead)
- **Neural Engine**: 16-core for ML acceleration
- **GPU Cores**: 40-core for parallel processing
- **Performance**: ~65 tokens/second for 8B models
- **Advantage**: Can run multiple large models simultaneously

## Model Architecture Overview

### Current State: Ollama + MLX
- **Ollama**: Uses GGUF format (llama.cpp backend)
- **MLX**: Apple's optimized framework for Apple Silicon
- **Performance**: Comparable speed, MLX loads 3x faster
- **Integration**: Pending official Ollama MLX support

## Recommended Model Stack

### 1. Document Understanding & Vision
```yaml
Primary:
  model: "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
  purpose: "PDF analysis, layout understanding"
  context: 32768 tokens
  memory: ~4GB
  features:
    - Vision-language understanding
    - Document structure recognition
    - Table/form extraction

Secondary:
  model: "mlx-community/llava-v1.6-mistral-7b-4bit"
  purpose: "Complex visual analysis"
  context: 4096 tokens
  memory: ~5GB
  features:
    - Handwritten text recognition
    - Image-based evidence analysis
```

### 2. Text Generation & Reasoning
```yaml
Primary:
  model: "mlx-community/Qwen2.5-7B-Instruct-4bit"
  purpose: "Legal document categorization, analysis"
  context: 131072 tokens (practical: ~50k)
  memory: ~5GB
  features:
    - Strong reasoning capabilities
    - Multi-language support
    - Efficient 4-bit quantization

Advanced:
  model: "mlx-community/Qwen2.5-14B-Instruct-4bit"
  purpose: "Complex legal reasoning"
  context: 131072 tokens
  memory: ~10GB
  features:
    - Superior reasoning for complex cases
    - Better context retention

Long Context:
  model: "mlx-community/Qwen2.5-7B-Instruct-1M-4bit"
  purpose: "Multi-document analysis"
  context: 1048576 tokens (practical: ~250k)
  memory: ~8GB
  features:
    - Analyze entire case files
    - Cross-reference multiple documents
```

### 3. Embeddings & RAG
```yaml
Primary:
  model: "mlx-embeddings/bge-m3"
  purpose: "Document embeddings for vector search"
  dimensions: 1024
  memory: ~2GB
  features:
    - Multi-lingual support
    - Dense + sparse retrieval
    - Colbert reranking

Legal-Specific:
  model: "mlx-embeddings/legal-bert-base"
  purpose: "Legal terminology understanding"
  dimensions: 768
  memory: ~1GB
  features:
    - Pre-trained on legal corpus
    - Better legal concept matching

Modern Alternative:
  model: "mlx-embeddings/ModernBERT-base"
  purpose: "State-of-the-art embeddings"
  dimensions: 768
  memory: ~1GB
  features:
    - Latest BERT improvements
    - Faster inference
```

### 4. Specialized Models
```yaml
Mistral-Based:
  model: "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
  purpose: "General reasoning, fallback"
  context: 32768 tokens
  memory: ~5GB

Code/Structure:
  model: "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit"
  purpose: "Structured data extraction"
  context: 163840 tokens
  memory: ~3GB

Phi-3 Vision:
  model: "mlx-community/Phi-3.5-vision-instruct-4bit"
  purpose: "Lightweight vision tasks"
  context: 128000 tokens
  memory: ~4GB
```

## Implementation Architecture

### 1. Hybrid Framework Approach
```python
# src/mlx_model_manager.py

import mlx_lm
from ollama import Client
from typing import Optional, Dict, Union

class HybridModelManager:
    """Manage both Ollama and MLX models"""
    
    def __init__(self):
        self.ollama_client = Client()
        self.mlx_models: Dict[str, any] = {}
        self.model_mapping = {
            # Map tasks to optimal models
            "categorization": {
                "framework": "mlx",
                "model": "mlx-community/Qwen2.5-7B-Instruct-4bit"
            },
            "vision_analysis": {
                "framework": "mlx",
                "model": "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
            },
            "embeddings": {
                "framework": "mlx",
                "model": "mlx-embeddings/bge-m3"
            },
            "complex_reasoning": {
                "framework": "mlx",
                "model": "mlx-community/Qwen2.5-14B-Instruct-4bit"
            },
            "long_context": {
                "framework": "mlx",
                "model": "mlx-community/Qwen2.5-7B-Instruct-1M-4bit"
            }
        }
    
    def load_mlx_model(self, model_name: str):
        """Load MLX model with caching"""
        if model_name not in self.mlx_models:
            import mlx_lm
            model, tokenizer = mlx_lm.load(model_name)
            self.mlx_models[model_name] = {
                "model": model,
                "tokenizer": tokenizer
            }
        return self.mlx_models[model_name]
    
    def generate(self, task: str, prompt: str, **kwargs):
        """Route to appropriate model/framework"""
        config = self.model_mapping.get(task)
        
        if config["framework"] == "mlx":
            return self._mlx_generate(config["model"], prompt, **kwargs)
        else:
            return self._ollama_generate(config["model"], prompt, **kwargs)
    
    def _mlx_generate(self, model_name: str, prompt: str, max_tokens=2048):
        """Generate using MLX"""
        model_data = self.load_mlx_model(model_name)
        return mlx_lm.generate(
            model_data["model"],
            model_data["tokenizer"],
            prompt=prompt,
            max_tokens=max_tokens
        )
```

### 2. Document Processing Pipeline
```python
# src/mlx_document_processor.py

class MLXDocumentProcessor:
    """Optimized document processing for M4 Max"""
    
    def __init__(self):
        self.manager = HybridModelManager()
        self.batch_size = 20  # Leverage 128GB RAM
    
    def process_legal_document(self, pdf_path: Path) -> Dict:
        """Full pipeline with MLX models"""
        
        # 1. Vision analysis for layout
        if self._has_complex_layout(pdf_path):
            layout_analysis = self.manager.generate(
                task="vision_analysis",
                prompt=f"Analyze document layout: {pdf_path}",
                image_path=pdf_path
            )
        
        # 2. Text extraction and categorization
        text = self._extract_text(pdf_path)
        category = self.manager.generate(
            task="categorization",
            prompt=self._build_categorization_prompt(text)
        )
        
        # 3. Generate embeddings
        embeddings = self.manager.generate(
            task="embeddings",
            prompt=text,
            return_embeddings=True
        )
        
        # 4. Complex analysis if needed
        if self._requires_deep_analysis(category):
            analysis = self.manager.generate(
                task="complex_reasoning",
                prompt=self._build_analysis_prompt(text, category)
            )
        
        return {
            "category": category,
            "embeddings": embeddings,
            "analysis": analysis if 'analysis' in locals() else None,
            "layout": layout_analysis if 'layout_analysis' in locals() else None
        }
```

### 3. Memory-Optimized Batch Processing
```python
# src/mlx_batch_processor.py

class MLXBatchProcessor:
    """Leverage 128GB RAM for parallel processing"""
    
    def __init__(self, max_parallel_models: int = 4):
        self.max_parallel = max_parallel_models
        self.loaded_models = {}
    
    def process_batch(self, documents: List[Path]) -> List[Dict]:
        """Process multiple documents in parallel"""
        
        # Pre-load models based on document types
        self._preload_models_for_batch(documents)
        
        # Process in optimized batches
        results = []
        for batch in self._create_batches(documents):
            batch_results = self._process_parallel(batch)
            results.extend(batch_results)
        
        return results
    
    def _preload_models_for_batch(self, documents: List[Path]):
        """Intelligently preload models based on document analysis"""
        # Analyze first few documents to determine needed models
        sample_size = min(5, len(documents))
        sample_docs = documents[:sample_size]
        
        required_models = set()
        for doc in sample_docs:
            if self._is_image_heavy(doc):
                required_models.add("mlx-community/Qwen2.5-VL-3B-Instruct-4bit")
            if self._is_long_document(doc):
                required_models.add("mlx-community/Qwen2.5-7B-Instruct-1M-4bit")
            else:
                required_models.add("mlx-community/Qwen2.5-7B-Instruct-4bit")
        
        # Load models
        for model in required_models:
            self._load_model(model)
```

## Performance Optimization

### 1. Model Loading Strategy
```python
# Fastest loading times with MLX
LOAD_TIMES = {
    "mlx": {
        "7B-4bit": "~10 seconds",
        "14B-4bit": "~18 seconds",
        "vision-3B": "~6 seconds"
    },
    "ollama": {
        "7B-4bit": "~30 seconds",
        "14B-4bit": "~45 seconds"
    }
}
```

### 2. Memory Management
```python
# With 128GB RAM, optimal configuration:
MAX_LOADED_MODELS = {
    "simultaneous_7B_models": 15,  # ~75GB
    "simultaneous_14B_models": 8,   # ~80GB
    "mixed_deployment": {
        "14B_models": 4,   # ~40GB
        "7B_models": 6,    # ~30GB
        "3B_models": 10,   # ~20GB
        "embeddings": 5,   # ~10GB
        "buffer": "28GB"   # OS and processing
    }
}
```

### 3. Throughput Optimization
```python
THROUGHPUT_ESTIMATES = {
    "single_model": {
        "7B": "65 tokens/sec",
        "14B": "45 tokens/sec",
        "3B": "120 tokens/sec"
    },
    "parallel_processing": {
        "documents_per_minute": 25,
        "with_vision": 15,
        "with_embeddings": 30
    }
}
```

## Migration Path

### Phase 1: Immediate Implementation
1. Install MLX and required packages
2. Download recommended models
3. Implement hybrid manager
4. Test with sample documents

### Phase 2: Optimization
1. Profile memory usage
2. Optimize batch sizes
3. Implement model caching
4. Fine-tune for legal documents

### Phase 3: Full Integration
1. Replace Ollama calls where beneficial
2. Implement fallback mechanisms
3. Add monitoring and metrics
4. Document model performance

## Setup Commands

```bash
# Install MLX
pip install mlx mlx-lm

# Install embedding support
pip install mlx-embeddings

# Download models (using MLX CLI)
mlx_lm.convert --hf-repo mlx-community/Qwen2.5-7B-Instruct-4bit
mlx_lm.convert --hf-repo mlx-community/Qwen2.5-VL-3B-Instruct-4bit
mlx_lm.convert --hf-repo mlx-community/Qwen2.5-14B-Instruct-4bit

# Or use Python
python -c "import mlx_lm; mlx_lm.load('mlx-community/Qwen2.5-7B-Instruct-4bit')"
```

## Configuration File

```yaml
# mlx_config.yaml
models:
  categorization:
    model: "mlx-community/Qwen2.5-7B-Instruct-4bit"
    max_tokens: 2048
    temperature: 0.1
    
  vision:
    model: "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
    max_tokens: 4096
    
  reasoning:
    model: "mlx-community/Qwen2.5-14B-Instruct-4bit"
    max_tokens: 8192
    temperature: 0.3
    
  embeddings:
    model: "mlx-embeddings/bge-m3"
    batch_size: 32
    
performance:
  max_parallel_models: 4
  model_cache_size: 10
  batch_size: 20
  
memory:
  max_allocation_gb: 100
  reserve_system_gb: 28
```

## Monitoring and Metrics

```python
# src/mlx_monitor.py
class MLXMonitor:
    """Monitor MLX model performance"""
    
    def get_metrics(self):
        return {
            "memory_usage": self._get_memory_usage(),
            "tokens_per_second": self._calculate_throughput(),
            "model_load_times": self._get_load_times(),
            "active_models": self._list_active_models()
        }
```

## Key Advantages for Legal Processing

1. **Speed**: 3x faster model loading
2. **Memory**: Unified architecture eliminates transfers
3. **Parallelism**: Process multiple documents simultaneously
4. **Quality**: Latest models with strong reasoning
5. **Flexibility**: Mix model sizes based on task complexity

## Conclusion

The M4 Max with 128GB RAM is ideally suited for running multiple MLX models simultaneously. The recommended stack provides:
- State-of-the-art document understanding
- Efficient memory usage
- Fast inference speeds
- Flexibility for various document types

While waiting for official Ollama MLX support, the hybrid approach maximizes performance while maintaining compatibility with existing infrastructure.