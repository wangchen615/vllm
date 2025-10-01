#!/usr/bin/env python3
import json
import os
import random

def modify_adapter_config(lora_dir, copy_id):
    """Modify adapter config to make it unique"""
    config_path = os.path.join(lora_dir, "adapter_config.json")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Modify parameters to make each copy unique
    config["lora_alpha"] = 1 + copy_id
    config["lora_dropout"] = round(0.05 + (copy_id * 0.001), 6)
    config["r"] = 8 + (copy_id % 4)  # Vary rank between 8-11
    config["revision"] = f"test_{copy_id}"
    config["unique_id"] = f"test_{copy_id}_{random.randint(100000, 999999)}"
    config["description"] = f"Scalability test LoRA adapter {copy_id}"
    config["metadata"] = {
        "copy_id": copy_id,
        "test_scenario": "scalability_testing",
        "created_by": "modify_lora_configs.py",
        "performance_tier": "tier_" + str((copy_id - 1) % 3 + 1)
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def main():
    base_dir = "/models/loras"
    
    # Modify all test copies (4-20)
    for i in range(2, 21):
        lora_dir = os.path.join(base_dir, f"sql-lora-test-{i}")
        if os.path.exists(lora_dir):
            modify_adapter_config(lora_dir, i)
            print(f"Modified config for sql-lora-test-{i}")
        else:
            print(f"Directory sql-lora-test-{i} not found")

if __name__ == "__main__":
    main()
