# Context 5: MIT License Model Migration Plan

## Overview
This document outlines the migration from proprietary models to MIT-licensed models for the legal document processing system. All models listed are open-source with MIT or similar permissive licenses.

## Model Selection

### Text Generation & Reasoning Models
1. **Mistral 7B** (Apache 2.0)
   - Variants: `mistral:7b-instruct-v0.2-q4_0` (int4), `mistral:7b-instruct-v0.2-fp16` (float16)
   - Use case: General text processing, document categorization
   - Memory: ~4GB (int4), ~14GB (fp16)

2. **Mixtral 8x7B** (Apache 2.0)
   - Model: `mixtral:8x7b-instruct-v0.1-q4_0`
   - Use case: Complex reasoning tasks, multi-document synthesis
   - Memory: ~26GB (int4)
   - Note: Mixture of Experts architecture for better performance

3. **Hermes 2 Pro** (Apache 2.0)
   - Model: `adrienbrault/nous-hermes2pro:Q4_K_M`
   - Use case: Advanced reasoning, legal document analysis
   - Memory: ~4GB
   - Note: Fine-tuned Mistral with strong reasoning capabilities

4. **OpenChat 3.5** (Apache 2.0)
   - Model: `openchat:7b-v3.5-q4_0`
   - Use case: Interactive dialogue, user queries
   - Memory: ~4GB

5. **Nous Hermes 2** (Apache 2.0)
   - Model: `teknium/openhermes-2.5-mistral-7b:Q4_K_M`
   - Use case: General-purpose tasks, fallback model
   - Memory: ~4GB

### Vision + Text Models (VLMs)
1. **LLaVA-v1.6 Mistral** (Apache 2.0)
   - Model: `llava:13b-v1.6-mistral-q4_0`
   - Use case: PDF image analysis, handwritten notes
   - Memory: ~8GB

2. **BakLLaVA** (Apache 2.0)
   - Model: `bakllava:7b-v1-q4_0`
   - Use case: Lightweight vision tasks
   - Memory: ~4GB
   - Note: Mistral backbone for consistency

3. **MiniGemma** (Apache 2.0)
   - Model: `gemma:2b`
   - Use case: Quick image pre-screening
   - Memory: ~1.5GB

### Embedding Model
1. **Nomic Embed Text v1** (Apache 2.0)
   - Model: `nomic-embed-text:v1`
   - Use case: All vector embeddings for RAG
   - Memory: ~275MB
   - Dimensions: 768

## Implementation Steps

### Phase 1: Environment Preparation

1. **Update Ollama Installation**
   ```bash
   # Ensure latest Ollama version
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Verify installation
   ollama --version
   ```

2. **Create Model Configuration File**
   ```python
   # src/model_config.py
   MIT_MODELS = {
       "text_generation": {
           "primary": "mistral:7b-instruct-v0.2-q4_0",
           "advanced": "mixtral:8x7b-instruct-v0.1-q4_0",
           "reasoning": "adrienbrault/nous-hermes2pro:Q4_K_M",
           "dialogue": "openchat:7b-v3.5-q4_0",
           "fallback": "teknium/openhermes-2.5-mistral-7b:Q4_K_M"
       },
       "vision": {
           "primary": "llava:13b-v1.6-mistral-q4_0",
           "lightweight": "bakllava:7b-v1-q4_0",
           "quick": "gemma:2b"
       },
       "embedding": {
           "default": "nomic-embed-text:v1"
       }
   }
   
   MODEL_REQUIREMENTS = {
       "document_categorization": "text_generation.primary",
       "exhibit_marking": "text_generation.primary",
       "complex_analysis": "text_generation.advanced",
       "legal_reasoning": "text_generation.reasoning",
       "image_analysis": "vision.primary",
       "vector_embedding": "embedding.default"
   }
   ```

### Phase 2: Model Download and Setup

1. **Create Setup Script**
   ```python
   # setup_mit_models.py
   import subprocess
   import json
   from pathlib import Path
   from typing import Dict, List
   
   def pull_model(model_name: str) -> bool:
       """Pull a model using Ollama CLI"""
       try:
           print(f"Pulling {model_name}...")
           result = subprocess.run(
               ["ollama", "pull", model_name],
               capture_output=True,
               text=True
           )
           return result.returncode == 0
       except Exception as e:
           print(f"Error pulling {model_name}: {e}")
           return False
   
   def verify_model(model_name: str) -> Dict:
       """Verify model is available and get info"""
       try:
           result = subprocess.run(
               ["ollama", "show", model_name, "--modelfile"],
               capture_output=True,
               text=True
           )
           if result.returncode == 0:
               return {
                   "available": True,
                   "info": result.stdout
               }
       except:
           pass
       return {"available": False}
   
   def setup_all_models():
       """Pull all required MIT-licensed models"""
       from src.model_config import MIT_MODELS
       
       results = {}
       for category, models in MIT_MODELS.items():
           results[category] = {}
           for purpose, model_name in models.items():
               success = pull_model(model_name)
               results[category][purpose] = {
                   "model": model_name,
                   "downloaded": success,
                   "verified": verify_model(model_name)["available"]
               }
       
       # Save results
       with open("model_setup_results.json", "w") as f:
           json.dump(results, f, indent=2)
       
       return results
   ```

2. **Run Model Setup**
   ```bash
   # Download all models
   python setup_mit_models.py
   
   # This will download approximately 60GB of models
   # Ensure sufficient disk space
   ```

### Phase 3: Update LLM Handler

1. **Modify LLMCategorizer for Model Selection**
   ```python
   # src/llm_handler.py updates
   
   from src.model_config import MIT_MODELS, MODEL_REQUIREMENTS
   
   class LLMCategorizer:
       def __init__(self, provider: str = "ollama", api_key: Optional[str] = None):
           self.provider = provider
           self.api_key = api_key
           
           if provider == "ollama":
               # Use MIT-licensed model based on task
               self.model_mapping = {
                   "categorization": self._get_model_for_requirement("document_categorization"),
                   "analysis": self._get_model_for_requirement("complex_analysis"),
                   "reasoning": self._get_model_for_requirement("legal_reasoning"),
                   "embedding": self._get_model_for_requirement("vector_embedding")
               }
       
       def _get_model_for_requirement(self, requirement: str) -> str:
           """Get appropriate MIT model for requirement"""
           model_path = MODEL_REQUIREMENTS.get(requirement)
           if not model_path:
               return MIT_MODELS["text_generation"]["primary"]
           
           category, purpose = model_path.split(".")
           return MIT_MODELS.get(category, {}).get(purpose, MIT_MODELS["text_generation"]["primary"])
       
       def categorize_document(self, text: str, filename: str = "") -> str:
           """Use MIT model for categorization"""
           if self.provider == "ollama":
               model = self.model_mapping["categorization"]
               # Rest of implementation...
   ```

2. **Update Embedding Handler**
   ```python
   # src/embedding_handler.py (new file)
   
   from langchain_community.embeddings import OllamaEmbeddings
   from src.model_config import MIT_MODELS
   
   class MITEmbeddingHandler:
       def __init__(self):
           self.model = MIT_MODELS["embedding"]["default"]
           self.embeddings = OllamaEmbeddings(
               model=self.model,
               base_url="http://localhost:11434"
           )
       
       def embed_documents(self, texts: List[str]) -> List[List[float]]:
           """Embed documents using MIT-licensed model"""
           return self.embeddings.embed_documents(texts)
       
       def embed_query(self, text: str) -> List[float]:
           """Embed query using MIT-licensed model"""
           return self.embeddings.embed_query(text)
   ```

### Phase 4: Update Vector Search

1. **Modify Vector Store Configuration**
   ```python
   # src/vector_search.py updates
   
   from src.embedding_handler import MITEmbeddingHandler
   
   def initialize_vector_store(persist_directory: str = "./chroma_db"):
       """Initialize with MIT-licensed embedding model"""
       embedding_handler = MITEmbeddingHandler()
       
       vectorstore = Chroma(
           collection_name="legal_documents",
           embedding_function=embedding_handler.embeddings,
           persist_directory=persist_directory
       )
       return vectorstore
   ```

### Phase 5: Testing Framework

1. **Create Model Test Suite**
   ```python
   # tests/test_mit_models.py
   
   import pytest
   from src.model_config import MIT_MODELS
   from src.llm_handler import LLMCategorizer
   from src.embedding_handler import MITEmbeddingHandler
   
   class TestMITModels:
       def test_all_models_available(self):
           """Verify all MIT models are downloaded"""
           import subprocess
           
           for category, models in MIT_MODELS.items():
               for purpose, model_name in models.items():
                   result = subprocess.run(
                       ["ollama", "show", model_name],
                       capture_output=True
                   )
                   assert result.returncode == 0, f"Model {model_name} not available"
       
       def test_categorization_with_mit_model(self):
           """Test document categorization with MIT model"""
           categorizer = LLMCategorizer(provider="ollama")
           
           test_text = "This is a medical record from Dr. Smith regarding patient care."
           category = categorizer.categorize_document(test_text)
           
           assert category in ["Medical Record", "Uncategorized"]
       
       def test_embedding_generation(self):
           """Test embedding generation with nomic model"""
           handler = MITEmbeddingHandler()
           
           test_texts = ["Legal document", "Medical record"]
           embeddings = handler.embed_documents(test_texts)
           
           assert len(embeddings) == 2
           assert len(embeddings[0]) == 768  # Nomic embed dimension
   ```

### Phase 6: Performance Optimization

1. **Model Loading Strategy**
   ```python
   # src/model_manager.py
   
   import subprocess
   from typing import Optional, Dict
   from datetime import datetime, timedelta
   
   class ModelManager:
       """Manage MIT model lifecycle"""
       
       def __init__(self):
           self.loaded_models: Dict[str, datetime] = {}
           self.model_timeout = timedelta(minutes=30)
       
       def load_model(self, model_name: str) -> bool:
           """Ensure model is loaded in Ollama"""
           try:
               # Check if already loaded
               result = subprocess.run(
                   ["ollama", "ps"],
                   capture_output=True,
                   text=True
               )
               
               if model_name in result.stdout:
                   self.loaded_models[model_name] = datetime.now()
                   return True
               
               # Load model
               subprocess.run(
                   ["ollama", "run", model_name, "hello"],
                   capture_output=True
               )
               self.loaded_models[model_name] = datetime.now()
               return True
           except:
               return False
       
       def unload_inactive_models(self):
           """Unload models not used recently"""
           current_time = datetime.now()
           for model, load_time in list(self.loaded_models.items()):
               if current_time - load_time > self.model_timeout:
                   self._unload_model(model)
                   del self.loaded_models[model]
   ```

2. **Batch Processing Optimization**
   ```python
   # Update src/pdf_processor.py
   
   def process_pdfs_with_mit_models(input_dir: Path, output_dir: Path):
       """Process PDFs using MIT-licensed models"""
       from src.model_manager import ModelManager
       
       manager = ModelManager()
       
       # Pre-load required models
       manager.load_model(MIT_MODELS["text_generation"]["primary"])
       manager.load_model(MIT_MODELS["embedding"]["default"])
       
       # Process files...
   ```

### Phase 7: Configuration Updates

1. **Update .env.template**
   ```bash
   # Model Configuration
   LLM_PROVIDER=ollama
   OLLAMA_BASE_URL=http://localhost:11434
   
   # MIT Model Selection
   TEXT_MODEL=mistral:7b-instruct-v0.2-q4_0
   VISION_MODEL=llava:13b-v1.6-mistral-q4_0
   EMBEDDING_MODEL=nomic-embed-text:v1
   
   # Performance Settings
   MODEL_TIMEOUT_MINUTES=30
   BATCH_SIZE=10
   ```

2. **Update Requirements**
   ```txt
   # requirements.txt additions
   ollama>=0.1.7
   langchain-community>=0.0.10
   ```

## Verification Checklist

- [ ] All MIT models downloaded successfully
- [ ] Model manager handles loading/unloading
- [ ] Categorization works with Mistral models
- [ ] Embeddings generate with nomic-embed-text
- [ ] Vision models process PDF images
- [ ] Performance acceptable for batch processing
- [ ] Memory usage within limits
- [ ] No proprietary model dependencies remain

## Migration Timeline

1. **Day 1**: Download all models, update configuration
2. **Day 2**: Update LLM handlers and embedding system
3. **Day 3**: Test with sample documents
4. **Day 4**: Full system testing and optimization
5. **Day 5**: Production deployment

## Rollback Plan

If issues arise:
1. Keep original OpenAI configuration in .env.backup
2. Use provider switching in LLMCategorizer
3. Maintain parallel processing paths
4. Document any model-specific quirks

## Notes

- Total disk space required: ~60GB
- RAM requirements: 16GB minimum, 32GB recommended
- GPU optional but improves performance
- All models are quantized (int4) for efficiency
- Mixtral requires more resources but handles complex tasks better