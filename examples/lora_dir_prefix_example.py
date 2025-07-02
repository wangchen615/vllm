#!/usr/bin/env python3
"""
Example script demonstrating the new LoRA directory prefix functionality.

This script shows how to use the --lora-dir-prefix argument to automatically
discover and load multiple LoRA adapters from a shared directory prefix.

Usage:
    python examples/lora_dir_prefix_example.py

Directory structure expected:
    /path/to/loras/
    ├── sql_adapter/
    │   ├── adapter_config.json
    │   └── adapter_model.safetensors
    ├── code_adapter/
    │   ├── adapter_config.json
    │   └── adapter_model.safetensors
    └── math_adapter/
        ├── adapter_config.json
        └── adapter_model.safetensors
"""

import os
import tempfile
import json
from pathlib import Path

from vllm.lora.utils import discover_lora_adapters_from_prefix


def create_example_lora_structure():
    """Create an example LoRA directory structure for demonstration."""
    with tempfile.TemporaryDirectory() as temp_dir:
        base_path = Path(temp_dir) / "loras"
        base_path.mkdir()
        
        # Create multiple LoRA adapters
        adapters = ["sql_adapter", "code_adapter", "math_adapter"]
        
        for adapter_name in adapters:
            adapter_path = base_path / adapter_name
            adapter_path.mkdir()
            
            # Create adapter_config.json
            config = {
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
                "bias": "none",
                "task_type": "CAUSAL_LM"
            }
            
            with open(adapter_path / "adapter_config.json", "w") as f:
                json.dump(config, f, indent=2)
            
            # Create empty adapter_model.safetensors (for demonstration)
            (adapter_path / "adapter_model.safetensors").touch()
            
            print(f"Created LoRA adapter: {adapter_path}")
        
        return str(base_path)


def demonstrate_discovery():
    """Demonstrate the LoRA discovery functionality."""
    print("=== LoRA Directory Prefix Discovery Demo ===\n")
    
    # Create example directory structure
    lora_prefix = create_example_lora_structure()
    print(f"Created LoRA directory prefix: {lora_prefix}\n")
    
    # Discover LoRA adapters
    try:
        discovered_adapters = discover_lora_adapters_from_prefix(lora_prefix)
        
        print(f"Discovered {len(discovered_adapters)} LoRA adapters:")
        for name, path in discovered_adapters.items():
            print(f"  - {name}: {path}")
        
        print(f"\nDirectory structure:")
        for root, dirs, files in os.walk(lora_prefix):
            level = root.replace(lora_prefix, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
        
    except Exception as e:
        print(f"Error discovering LoRA adapters: {e}")


def show_usage_examples():
    """Show usage examples for the new functionality."""
    print("\n=== Usage Examples ===\n")
    
    print("1. Using --lora-dir-prefix with vLLM server:")
    print("   python -m vllm.entrypoints.openai.api_server \\")
    print("       --model meta-llama/Llama-2-7b-chat-hf \\")
    print("       --enable-lora \\")
    print("       --lora-dir-prefix /path/to/loras \\")
    print("       --max-loras 10")
    print()
    
    print("2. Combining with existing --lora-modules:")
    print("   python -m vllm.entrypoints.openai.api_server \\")
    print("       --model meta-llama/Llama-2-7b-chat-hf \\")
    print("       --enable-lora \\")
    print("       --lora-dir-prefix /path/to/loras \\")
    print("       --lora-modules custom_adapter=/path/to/custom \\")
    print("       --max-loras 10")
    print()
    
    print("3. With periodic discovery enabled:")
    print("   python -m vllm.entrypoints.openai.api_server \\")
    print("       --model meta-llama/Llama-2-7b-chat-hf \\")
    print("       --enable-lora \\")
    print("       --lora-dir-prefix /path/to/loras \\")
    print("       --lora-discovery-interval 300 \\")  # 5 minutes
    print("       --max-loras 10")
    print()
    
    print("4. Manual refresh via REST API (when periodic discovery is disabled):")
    print("   curl -X POST http://localhost:8000/v1/refresh_lora_discovery \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '{}'")
    print()
    print("   Response example:")
    print("   {")
    print('     "message": "Successfully refreshed LoRA discovery. Found 5 total adapters, 2 newly discovered.",')
    print('     "discovered_adapters": ["new_adapter1", "new_adapter2"],')
    print('     "total_adapters": 5')
    print("   }")
    print()
    
    print("5. Using discovered LoRA adapters in API calls:")
    print("   curl -X POST http://localhost:8000/v1/chat/completions \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '{")
    print('          "model": "sql_adapter",')
    print('          "messages": [{"role": "user", "content": "Generate SQL for..."}]')
    print("        }'")
    print()
    
    print("6. List available models (including discovered LoRA adapters):")
    print("   curl http://localhost:8000/v1/models")
    print()
    print("   Response will include all discovered LoRA adapters as separate models.")


if __name__ == "__main__":
    demonstrate_discovery()
    show_usage_examples() 