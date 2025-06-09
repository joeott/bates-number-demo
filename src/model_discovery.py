"""
Model discovery and mapping service for multi-model pipeline.

This module handles the discovery of available LM Studio models and
maps them to specific tasks based on their characteristics.
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from test_lmstudio import LMStudioDiscovery

logger = logging.getLogger(__name__)


class ModelTask(Enum):
    """Tasks that models can be assigned to."""
    VISUAL_ANALYSIS = "visual"
    REASONING = "reasoning"
    CATEGORIZATION = "categorization"
    SYNTHESIS = "synthesis"
    EMBEDDING = "embedding"


@dataclass
class ModelProfile:
    """Profile of a discovered model with its capabilities."""
    model_id: str
    model_type: str  # text, embedding, vision, etc.
    suggested_tasks: List[ModelTask]
    context_length: Optional[int] = None
    estimated_memory_gb: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "suggested_tasks": [task.value for task in self.suggested_tasks],
            "context_length": self.context_length,
            "estimated_memory_gb": self.estimated_memory_gb
        }


class ModelDiscoveryService:
    """Service for discovering and mapping LM Studio models to tasks."""
    
    def __init__(self, base_url: str = "http://localhost:1234"):
        self.discovery = LMStudioDiscovery(base_url)
        self.model_profiles: Dict[str, ModelProfile] = {}
        
    def discover_and_profile_models(self) -> Dict[str, ModelProfile]:
        """Discover available models and create profiles."""
        try:
            # Check connection first
            connected, message = self.discovery.check_connection()
            if not connected:
                logger.error(f"Cannot connect to LM Studio: {message}")
                return {}
            
            # Discover models
            models = self.discovery.discover_models()
            
            # Profile each model
            for model in models:
                profile = self._create_model_profile(model)
                self.model_profiles[model["id"]] = profile
                
            return self.model_profiles
            
        except Exception as e:
            logger.error(f"Error discovering models: {e}")
            return {}
    
    def _create_model_profile(self, model_info: Dict) -> ModelProfile:
        """Create a profile for a discovered model."""
        model_id = model_info["id"]
        model_type = model_info["type"]
        
        # Determine suggested tasks based on model characteristics
        suggested_tasks = self._suggest_tasks_for_model(model_id, model_type)
        
        # Estimate context length and memory requirements
        context_length = self._estimate_context_length(model_id)
        memory_gb = self._estimate_memory_usage(model_id)
        
        return ModelProfile(
            model_id=model_id,
            model_type=model_type,
            suggested_tasks=suggested_tasks,
            context_length=context_length,
            estimated_memory_gb=memory_gb
        )
    
    def _suggest_tasks_for_model(self, model_id: str, model_type: str) -> List[ModelTask]:
        """Suggest appropriate tasks for a model based on its characteristics."""
        model_lower = model_id.lower()
        tasks = []
        
        # Vision models
        if model_type == "vision" or "pixtral" in model_lower:
            tasks.append(ModelTask.VISUAL_ANALYSIS)
            
        # Reasoning models
        if "mathstral" in model_lower or "reasoning" in model_lower:
            tasks.extend([ModelTask.REASONING])
            
        # Instruction-following models
        if "instruct" in model_lower or "nemo" in model_lower:
            tasks.extend([ModelTask.CATEGORIZATION])
            
        # Large models for synthesis
        if any(size in model_lower for size in ["13b", "17b", "20b", "30b", "70b"]):
            tasks.append(ModelTask.SYNTHESIS)
        elif "llama-4-scout" in model_lower:
            tasks.append(ModelTask.SYNTHESIS)
            
        # Embedding models
        if model_type == "embedding":
            tasks.append(ModelTask.EMBEDDING)
            
        # Default fallback for text models
        if not tasks and model_type == "text":
            tasks.append(ModelTask.CATEGORIZATION)
            
        return tasks
    
    def _estimate_context_length(self, model_id: str) -> int:
        """Estimate context length based on model name."""
        model_lower = model_id.lower()
        
        # Look for explicit context indicators
        if "1m" in model_lower:
            return 1048576
        elif "256k" in model_lower:
            return 262144
        elif "128k" in model_lower or "nemo" in model_lower:
            return 131072
        elif "32k" in model_lower:
            return 32768
        elif "16k" in model_lower:
            return 16384
        elif "8k" in model_lower:
            return 8192
        else:
            # Default based on model family
            if "llama-3" in model_lower:
                return 8192
            elif "mistral" in model_lower:
                return 32768
            else:
                return 4096
    
    def _estimate_memory_usage(self, model_id: str) -> float:
        """Estimate memory usage in GB based on model size."""
        model_lower = model_id.lower()
        
        # Extract parameter count if present
        if "70b" in model_lower:
            return 40.0
        elif "30b" in model_lower:
            return 20.0
        elif "20b" in model_lower:
            return 15.0
        elif "17b" in model_lower:
            return 12.0
        elif "13b" in model_lower:
            return 10.0
        elif "12b" in model_lower:
            return 9.0
        elif "7b" in model_lower:
            return 5.0
        elif "3b" in model_lower:
            return 3.0
        elif "1b" in model_lower:
            return 1.5
        else:
            # Default estimates
            if "embedding" in model_lower:
                return 2.0
            else:
                return 5.0
    
    def generate_task_mapping(self) -> Dict[str, str]:
        """Generate optimal task-to-model mapping based on discovered models."""
        if not self.model_profiles:
            self.discover_and_profile_models()
            
        mapping = {}
        
        # For each task, find the best available model
        for task in ModelTask:
            best_model = self._find_best_model_for_task(task)
            if best_model:
                mapping[task.value] = best_model
                
        return mapping
    
    def _find_best_model_for_task(self, task: ModelTask) -> Optional[str]:
        """Find the best available model for a specific task."""
        candidates = []
        
        for model_id, profile in self.model_profiles.items():
            if task in profile.suggested_tasks:
                candidates.append((model_id, profile))
        
        if not candidates:
            return None
            
        # Sort by preference (can be customized)
        if task == ModelTask.VISUAL_ANALYSIS:
            # Prefer models with "pixtral" in name
            candidates.sort(key=lambda x: ("pixtral" in x[0].lower(), -x[1].estimated_memory_gb))
        elif task == ModelTask.REASONING:
            # Prefer mathstral
            candidates.sort(key=lambda x: ("mathstral" in x[0].lower(), -x[1].estimated_memory_gb))
        elif task == ModelTask.SYNTHESIS:
            # Prefer larger models
            candidates.sort(key=lambda x: -x[1].estimated_memory_gb)
        else:
            # Default: prefer smaller models for efficiency
            candidates.sort(key=lambda x: x[1].estimated_memory_gb)
            
        return candidates[0][0]
    
    def generate_env_config(self, mapping: Dict[str, str]) -> str:
        """Generate environment configuration for discovered models."""
        config_lines = [
            "# Multi-Model LM Studio Configuration",
            "# Generated by model discovery service",
            ""
        ]
        
        # Add model mappings
        for task, model_id in mapping.items():
            env_var = f"LMSTUDIO_{task.upper()}_MODEL"
            config_lines.append(f"{env_var}={model_id}")
            
        # Add metadata comments
        config_lines.extend([
            "",
            "# Model Profiles:",
        ])
        
        for model_id in mapping.values():
            if model_id in self.model_profiles:
                profile = self.model_profiles[model_id]
                config_lines.append(f"# {model_id}:")
                config_lines.append(f"#   Type: {profile.model_type}")
                config_lines.append(f"#   Context: {profile.context_length} tokens")
                config_lines.append(f"#   Memory: ~{profile.estimated_memory_gb}GB")
                
        return "\n".join(config_lines)
    
    def save_discovery_results(self, output_path: str = "model_discovery_results.json"):
        """Save discovery results to a JSON file."""
        results = {
            "discovery_timestamp": str(Path(output_path).stat().st_mtime if Path(output_path).exists() else ""),
            "model_profiles": {
                model_id: profile.to_dict() 
                for model_id, profile in self.model_profiles.items()
            },
            "task_mapping": self.generate_task_mapping(),
            "total_models": len(self.model_profiles),
            "total_memory_gb": sum(p.estimated_memory_gb for p in self.model_profiles.values())
        }
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Discovery results saved to {output_path}")


def main():
    """Run model discovery and generate configuration."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Discover and map LM Studio models")
    parser.add_argument("--url", default="http://localhost:1234", help="LM Studio base URL")
    parser.add_argument("--save", action="store_true", help="Save discovery results to file")
    parser.add_argument("--env", action="store_true", help="Generate .env configuration")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run discovery
    service = ModelDiscoveryService(args.url)
    profiles = service.discover_and_profile_models()
    
    if not profiles:
        print("No models discovered. Is LM Studio running?")
        return
        
    # Display results
    print(f"\nDiscovered {len(profiles)} models:")
    for model_id, profile in profiles.items():
        print(f"\n{model_id}:")
        print(f"  Type: {profile.model_type}")
        print(f"  Suggested tasks: {[t.value for t in profile.suggested_tasks]}")
        print(f"  Context: {profile.context_length} tokens")
        print(f"  Memory: ~{profile.estimated_memory_gb}GB")
    
    # Generate task mapping
    mapping = service.generate_task_mapping()
    print("\nOptimal task mapping:")
    for task, model in mapping.items():
        print(f"  {task}: {model}")
    
    # Save results if requested
    if args.save:
        service.save_discovery_results()
        
    # Generate env config if requested
    if args.env:
        print("\n" + "=" * 60)
        print("Environment Configuration:")
        print("=" * 60)
        print(service.generate_env_config(mapping))


if __name__ == "__main__":
    main()