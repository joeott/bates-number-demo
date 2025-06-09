#!/usr/bin/env python3
"""
LM Studio Model Discovery and Testing Utility

This script helps discover available models in LM Studio and test connections.
It provides information needed to configure .env variables for the bates_number_demo project.

Usage:
    python test_lmstudio.py              # Basic discovery
    python test_lmstudio.py --test       # Test model loading and generation
    python test_lmstudio.py --benchmark  # Benchmark performance
"""

import sys
import json
import time
import argparse
from typing import Dict, List, Optional, Tuple
import requests
from pathlib import Path

# Try to import lmstudio SDK (optional - for advanced features)
try:
    import lmstudio as lms
    HAS_LMSTUDIO_SDK = True
except ImportError:
    HAS_LMSTUDIO_SDK = False
    print("Note: lmstudio SDK not installed. Using OpenAI-compatible API only.")
    print("To install: pip install lmstudio-sdk")
    print()


class LMStudioDiscovery:
    """Discover and test LM Studio models."""
    
    def __init__(self, base_url: str = "http://localhost:1234"):
        self.base_url = base_url
        self.api_base = f"{base_url}/v1"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer lm-studio"  # Dummy token
        }
        
    def check_connection(self) -> Tuple[bool, str]:
        """Check if LM Studio is running and accessible."""
        try:
            # Try health endpoint
            response = requests.get(f"{self.base_url}/api/health", timeout=5)
            if response.status_code == 200:
                return True, "LM Studio server is healthy"
                
            # Try models endpoint as fallback
            response = requests.get(f"{self.api_base}/models", timeout=5)
            if response.status_code == 200:
                return True, "LM Studio server is accessible (via OpenAI API)"
                
            return False, f"Server returned status {response.status_code}"
            
        except requests.exceptions.ConnectionError:
            return False, "Cannot connect to LM Studio. Is it running?"
        except requests.exceptions.Timeout:
            return False, "Connection timeout. Check if LM Studio is responding."
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"
    
    def discover_models(self) -> List[Dict]:
        """Discover available models via OpenAI-compatible API."""
        try:
            response = requests.get(f"{self.api_base}/models", headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            models = []
            for model in data.get("data", []):
                model_info = {
                    "id": model.get("id"),
                    "object": model.get("object"),
                    "created": model.get("created"),
                    "owned_by": model.get("owned_by", "lmstudio"),
                    "type": self._infer_model_type(model.get("id", ""))
                }
                models.append(model_info)
            
            return models
            
        except Exception as e:
            print(f"Error discovering models: {e}")
            return []
    
    def _infer_model_type(self, model_id: str) -> str:
        """Infer model type from model ID."""
        model_lower = model_id.lower()
        
        # Vision models
        if any(v in model_lower for v in ["vision", "vl", "llava", "phi-3.5-vision"]):
            return "vision"
        
        # Embedding models
        if any(e in model_lower for e in ["embed", "bge", "nomic-embed", "gte"]):
            return "embedding"
        
        # Code models
        if any(c in model_lower for c in ["code", "codestral", "deepseek-coder"]):
            return "code"
        
        # Long context models
        if "1m" in model_lower or "128k" in model_lower or "256k" in model_lower:
            return "long-context"
        
        # Default to text generation
        return "text"
    
    def test_model(self, model_id: str, model_type: str = "text") -> Dict:
        """Test a specific model with appropriate prompts."""
        print(f"\nTesting model: {model_id} (type: {model_type})")
        
        if model_type == "embedding":
            return self._test_embedding_model(model_id)
        elif model_type == "vision":
            return self._test_vision_model(model_id)
        else:
            return self._test_text_model(model_id)
    
    def _test_text_model(self, model_id: str) -> Dict:
        """Test text generation model."""
        test_prompt = "Categorize this document: 'Plaintiff's Motion for Summary Judgment filed on behalf of John Doe v. ABC Corp'"
        
        try:
            start_time = time.time()
            
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=self.headers,
                json={
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": "You are a legal document categorizer. Respond with only the category."},
                        {"role": "user", "content": test_prompt}
                    ],
                    "max_tokens": 50,
                    "temperature": 0.1
                }
            )
            
            response.raise_for_status()
            result = response.json()
            
            elapsed = time.time() - start_time
            
            return {
                "success": True,
                "model": model_id,
                "response": result["choices"][0]["message"]["content"],
                "tokens": result.get("usage", {}),
                "time": elapsed,
                "tokens_per_second": result.get("usage", {}).get("completion_tokens", 0) / elapsed if elapsed > 0 else 0
            }
            
        except Exception as e:
            return {
                "success": False,
                "model": model_id,
                "error": str(e)
            }
    
    def _test_embedding_model(self, model_id: str) -> Dict:
        """Test embedding model."""
        test_text = "This is a legal document about contract breach."
        
        try:
            start_time = time.time()
            
            response = requests.post(
                f"{self.api_base}/embeddings",
                headers=self.headers,
                json={
                    "model": model_id,
                    "input": test_text
                }
            )
            
            response.raise_for_status()
            result = response.json()
            
            elapsed = time.time() - start_time
            embedding = result["data"][0]["embedding"]
            
            return {
                "success": True,
                "model": model_id,
                "embedding_dim": len(embedding),
                "time": elapsed,
                "sample": embedding[:5]  # First 5 dimensions
            }
            
        except Exception as e:
            return {
                "success": False,
                "model": model_id,
                "error": str(e)
            }
    
    def _test_vision_model(self, model_id: str) -> Dict:
        """Test vision model (placeholder - would need actual image)."""
        return {
            "success": False,
            "model": model_id,
            "error": "Vision model testing not implemented in basic mode"
        }
    
    def benchmark_models(self, models: List[Dict]) -> List[Dict]:
        """Benchmark performance of discovered models."""
        results = []
        
        # Legal document categorization prompts
        test_cases = [
            "Medical records from Dr. Smith dated January 15, 2024",
            "Invoice #12345 for legal services rendered in the amount of $5,000",
            "Defendant's Motion to Dismiss for Failure to State a Claim",
            "Email correspondence between plaintiff and defendant regarding contract terms",
            "Police report filed on March 1, 2024 regarding incident at 123 Main St"
        ]
        
        for model in models:
            if model["type"] not in ["text", "code", "long-context"]:
                continue
                
            print(f"\nBenchmarking: {model['id']}")
            
            total_time = 0
            total_tokens = 0
            successes = 0
            
            for prompt in test_cases:
                result = self._test_text_model(model["id"])
                if result["success"]:
                    total_time += result["time"]
                    total_tokens += result.get("tokens", {}).get("completion_tokens", 0)
                    successes += 1
            
            if successes > 0:
                results.append({
                    "model": model["id"],
                    "type": model["type"],
                    "avg_time": total_time / successes,
                    "avg_tokens_per_sec": total_tokens / total_time if total_time > 0 else 0,
                    "success_rate": successes / len(test_cases)
                })
        
        return results
    
    def generate_env_config(self, models: List[Dict]) -> str:
        """Generate .env configuration based on discovered models."""
        # Find best models for each purpose
        text_models = [m for m in models if m["type"] in ["text", "long-context"]]
        vision_models = [m for m in models if m["type"] == "vision"]
        embedding_models = [m for m in models if m["type"] == "embedding"]
        
        # Map discovered models to tasks based on their characteristics
        model_mapping = self._map_models_to_tasks(models)
        
        # Base configuration
        config_lines = [
            "# LM Studio Configuration",
            f"LMSTUDIO_HOST={self.api_base}",
            ""
        ]
        
        # Single model configuration (backward compatible)
        default_model = model_mapping.get("categorization", 
                                        text_models[0]["id"] if text_models else "none")
        config_lines.extend([
            "# Default model (for backward compatibility)",
            f"LMSTUDIO_MODEL={default_model}",
            f"LMSTUDIO_VISION_MODEL={model_mapping.get('visual', 'none')}",
            f"LMSTUDIO_EMBEDDING_MODEL={model_mapping.get('embedding', 'none')}",
            ""
        ])
        
        # Multi-model configuration
        config_lines.extend([
            "# Multi-Model Pipeline Configuration",
            f"LMSTUDIO_VISUAL_MODEL={model_mapping.get('visual', 'none')}",
            f"LMSTUDIO_REASONING_MODEL={model_mapping.get('reasoning', 'none')}",
            f"LMSTUDIO_CATEGORIZATION_MODEL={model_mapping.get('categorization', default_model)}",
            f"LMSTUDIO_SYNTHESIS_MODEL={model_mapping.get('synthesis', 'none')}",
            "# LMSTUDIO_EMBEDDING_MODEL already defined above",
            "",
            "# Enable multi-model pipeline",
            "ENABLE_MULTI_MODEL=auto  # auto-enables if multiple models configured",
            "",
            "# Performance settings",
            "LMSTUDIO_MAX_TOKENS=2048",
            "LMSTUDIO_CONTEXT_LENGTH=32768"
        ])
        
        return "\n".join(config_lines)
    
    def _map_models_to_tasks(self, models: List[Dict]) -> Dict[str, str]:
        """Map discovered models to specific tasks based on their characteristics."""
        mapping = {}
        
        for model in models:
            model_id = model["id"]
            model_type = model["type"]
            model_lower = model_id.lower()
            
            # Visual analysis
            if "pixtral" in model_lower or model_type == "vision":
                if "visual" not in mapping:
                    mapping["visual"] = model_id
                    
            # Reasoning/math models
            if "mathstral" in model_lower:
                if "reasoning" not in mapping:
                    mapping["reasoning"] = model_id
                    
            # Instruction-following models for categorization
            if ("nemo" in model_lower or "instruct" in model_lower) and model_type != "embedding":
                if "categorization" not in mapping:
                    mapping["categorization"] = model_id
                    
            # Large models for synthesis
            if "llama-4-scout" in model_lower or "17b" in model_lower:
                if "synthesis" not in mapping:
                    mapping["synthesis"] = model_id
                    
            # Embedding models
            if model_type == "embedding":
                if "embedding" not in mapping:
                    mapping["embedding"] = model_id
        
        # Fill in any missing mappings with available models
        text_models = [m for m in models if m["type"] in ["text", "long-context"]]
        if text_models:
            default_text = text_models[0]["id"]
            if "categorization" not in mapping:
                mapping["categorization"] = default_text
            if "synthesis" not in mapping and len(text_models) > 1:
                # Use a different model for synthesis if available
                mapping["synthesis"] = text_models[-1]["id"]
        
        return mapping
    
    def generate_report(self) -> None:
        """Generate a comprehensive report of LM Studio setup."""
        print("=" * 60)
        print("LM Studio Discovery Report")
        print("=" * 60)
        
        # Check connection
        connected, message = self.check_connection()
        print(f"\nConnection Status: {'✓' if connected else '✗'} {message}")
        
        if not connected:
            print("\nPlease ensure LM Studio is running and the server is started.")
            print("Default port is 1234. You can start the server from the LM Studio UI.")
            return
        
        # Discover models
        models = self.discover_models()
        print(f"\nDiscovered Models: {len(models)}")
        
        if models:
            # Group by type
            by_type = {}
            for model in models:
                model_type = model["type"]
                if model_type not in by_type:
                    by_type[model_type] = []
                by_type[model_type].append(model)
            
            # Display by type
            for model_type, type_models in by_type.items():
                print(f"\n{model_type.upper()} Models ({len(type_models)}):")
                for model in type_models:
                    print(f"  - {model['id']}")
            
            # Generate configuration
            print("\n" + "=" * 60)
            print("Recommended .env Configuration:")
            print("=" * 60)
            print(self.generate_env_config(models))
            
            # SDK availability
            print("\n" + "=" * 60)
            print("SDK Status:")
            print("=" * 60)
            if HAS_LMSTUDIO_SDK:
                print("✓ LM Studio SDK is installed")
                print("  Advanced features available: model management, TTL, etc.")
            else:
                print("✗ LM Studio SDK not installed")
                print("  Using OpenAI-compatible API only")
                print("  To enable advanced features: pip install lmstudio-sdk")


def main():
    parser = argparse.ArgumentParser(description="LM Studio Model Discovery and Testing")
    parser.add_argument("--url", default="http://localhost:1234", help="LM Studio base URL")
    parser.add_argument("--test", action="store_true", help="Test model functionality")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark model performance")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    
    args = parser.parse_args()
    
    discovery = LMStudioDiscovery(args.url)
    
    if args.json:
        # JSON output mode
        connected, message = discovery.check_connection()
        result = {
            "connected": connected,
            "message": message,
            "models": discovery.discover_models() if connected else []
        }
        print(json.dumps(result, indent=2))
        
    elif args.benchmark:
        # Benchmark mode
        connected, _ = discovery.check_connection()
        if connected:
            models = discovery.discover_models()
            results = discovery.benchmark_models(models)
            
            print("\n" + "=" * 60)
            print("Benchmark Results:")
            print("=" * 60)
            for result in sorted(results, key=lambda x: x["avg_tokens_per_sec"], reverse=True):
                print(f"\nModel: {result['model']}")
                print(f"  Type: {result['type']}")
                print(f"  Avg Time: {result['avg_time']:.2f}s")
                print(f"  Tokens/sec: {result['avg_tokens_per_sec']:.1f}")
                print(f"  Success Rate: {result['success_rate']*100:.0f}%")
        else:
            print("Cannot connect to LM Studio")
            
    elif args.test:
        # Test mode
        connected, _ = discovery.check_connection()
        if connected:
            models = discovery.discover_models()
            for model in models[:3]:  # Test first 3 models
                result = discovery.test_model(model["id"], model["type"])
                print(f"\nTest Result: {json.dumps(result, indent=2)}")
        else:
            print("Cannot connect to LM Studio")
            
    else:
        # Default: generate report
        discovery.generate_report()


if __name__ == "__main__":
    main()