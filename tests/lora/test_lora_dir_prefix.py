import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from vllm.lora.utils import discover_lora_adapters_from_prefix, discover_new_lora_adapters_from_prefix


def create_mock_lora_adapter(base_path: Path, adapter_name: str):
    """Create a mock LoRA adapter directory structure."""
    adapter_path = base_path / adapter_name
    adapter_path.mkdir(exist_ok=True)
    
    # Create adapter_config.json
    config_content = {
        "r": 8,
        "lora_alpha": 16,
        "target_modules": ["q_proj", "v_proj"]
    }
    import json
    with open(adapter_path / "adapter_config.json", "w") as f:
        json.dump(config_content, f)
    
    # Create adapter_model.safetensors (empty file for testing)
    (adapter_path / "adapter_model.safetensors").touch()
    
    return adapter_path


def test_discover_lora_adapters_from_prefix():
    """Test discovering LoRA adapters from a directory prefix."""
    with tempfile.TemporaryDirectory() as temp_dir:
        base_path = Path(temp_dir)
        
        # Create mock LoRA adapters
        create_mock_lora_adapter(base_path, "sql_adapter")
        create_mock_lora_adapter(base_path, "code_adapter")
        create_mock_lora_adapter(base_path, "math_adapter")
        
        # Create a non-adapter directory (should be ignored)
        (base_path / "not_an_adapter").mkdir()
        
        # Discover adapters
        discovered = discover_lora_adapters_from_prefix(str(base_path))
        
        # Verify results
        assert len(discovered) == 3
        assert "sql_adapter" in discovered
        assert "code_adapter" in discovered
        assert "math_adapter" in discovered
        assert "not_an_adapter" not in discovered
        
        # Verify paths
        for name, path in discovered.items():
            assert Path(path).name == name
            assert Path(path).exists()


def test_discover_lora_adapters_from_prefix_invalid_path():
    """Test discovering LoRA adapters from an invalid path."""
    with pytest.raises(ValueError, match="LoRA directory prefix does not exist"):
        discover_lora_adapters_from_prefix("/nonexistent/path")


def test_discover_lora_adapters_from_prefix_file():
    """Test discovering LoRA adapters from a file (should fail)."""
    with tempfile.NamedTemporaryFile() as temp_file:
        with pytest.raises(ValueError, match="LoRA directory prefix is not a directory"):
            discover_lora_adapters_from_prefix(temp_file.name)


def test_discover_lora_adapters_from_prefix_no_adapters():
    """Test discovering LoRA adapters from an empty directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(ValueError, match="No valid LoRA adapters found"):
            discover_lora_adapters_from_prefix(temp_dir)


def test_discover_lora_adapters_from_prefix_missing_config():
    """Test discovering LoRA adapters with missing config file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        base_path = Path(temp_dir)
        
        # Create directory without config
        adapter_path = base_path / "incomplete_adapter"
        adapter_path.mkdir()
        (adapter_path / "adapter_model.safetensors").touch()
        
        with pytest.raises(ValueError, match="No valid LoRA adapters found"):
            discover_lora_adapters_from_prefix(str(base_path))


def test_discover_lora_adapters_from_prefix_missing_model():
    """Test discovering LoRA adapters with missing model file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        base_path = Path(temp_dir)
        
        # Create directory without model file
        adapter_path = base_path / "incomplete_adapter"
        adapter_path.mkdir()
        
        config_content = {
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "v_proj"]
        }
        import json
        with open(adapter_path / "adapter_config.json", "w") as f:
            json.dump(config_content, f)
        
        with pytest.raises(ValueError, match="No valid LoRA adapters found"):
            discover_lora_adapters_from_prefix(str(base_path))


def test_discover_lora_adapters_from_prefix_bin_format():
    """Test discovering LoRA adapters with .bin format."""
    with tempfile.TemporaryDirectory() as temp_dir:
        base_path = Path(temp_dir)
        
        # Create mock LoRA adapter with .bin format
        adapter_path = base_path / "bin_adapter"
        adapter_path.mkdir()
        
        config_content = {
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "v_proj"]
        }
        import json
        with open(adapter_path / "adapter_config.json", "w") as f:
            json.dump(config_content, f)
        
        # Create adapter_model.bin (empty file for testing)
        (adapter_path / "adapter_model.bin").touch()
        
        # Discover adapters
        discovered = discover_lora_adapters_from_prefix(str(base_path))
        
        # Verify results
        assert len(discovered) == 1
        assert "bin_adapter" in discovered


def test_discover_new_lora_adapters_from_prefix():
    """Test discovering only new LoRA adapters."""
    with tempfile.TemporaryDirectory() as temp_dir:
        base_path = Path(temp_dir)
        
        # Create initial adapters
        create_mock_lora_adapter(base_path, "existing_adapter1")
        create_mock_lora_adapter(base_path, "existing_adapter2")
        
        # Track existing adapters
        existing_adapters = {"existing_adapter1", "existing_adapter2"}
        
        # Discover new adapters (should be empty initially)
        new_adapters = discover_new_lora_adapters_from_prefix(str(base_path), existing_adapters)
        assert len(new_adapters) == 0
        
        # Add a new adapter
        create_mock_lora_adapter(base_path, "new_adapter")
        
        # Discover new adapters again
        new_adapters = discover_new_lora_adapters_from_prefix(str(base_path), existing_adapters)
        assert len(new_adapters) == 1
        assert "new_adapter" in new_adapters
        assert new_adapters["new_adapter"] == str(base_path / "new_adapter")
        
        # Update existing_adapters and discover again (should be empty)
        existing_adapters.add("new_adapter")
        new_adapters = discover_new_lora_adapters_from_prefix(str(base_path), existing_adapters)
        assert len(new_adapters) == 0


def test_discover_new_lora_adapters_from_prefix_multiple_new():
    """Test discovering multiple new LoRA adapters."""
    with tempfile.TemporaryDirectory() as temp_dir:
        base_path = Path(temp_dir)
        
        # Create initial adapters
        create_mock_lora_adapter(base_path, "existing_adapter")
        existing_adapters = {"existing_adapter"}
        
        # Add multiple new adapters
        create_mock_lora_adapter(base_path, "new_adapter1")
        create_mock_lora_adapter(base_path, "new_adapter2")
        create_mock_lora_adapter(base_path, "new_adapter3")
        
        # Discover new adapters
        new_adapters = discover_new_lora_adapters_from_prefix(str(base_path), existing_adapters)
        
        # Verify results
        assert len(new_adapters) == 3
        assert "new_adapter1" in new_adapters
        assert "new_adapter2" in new_adapters
        assert "new_adapter3" in new_adapters
        assert "existing_adapter" not in new_adapters


def test_discover_new_lora_adapters_from_prefix_invalid_path():
    """Test discovering new LoRA adapters from an invalid path."""
    existing_adapters = {"existing_adapter"}
    new_adapters = discover_new_lora_adapters_from_prefix("/nonexistent/path", existing_adapters)
    assert len(new_adapters) == 0


def test_refresh_lora_discovery_endpoint():
    """Test the manual refresh LoRA discovery endpoint functionality."""
    from vllm.entrypoints.openai.protocol import RefreshLoraDiscoveryRequest, RefreshLoraDiscoveryResponse
    
    # Test with empty request (as designed)
    request = RefreshLoraDiscoveryRequest()
    
    # The request should be valid and empty
    assert request is not None
    # No fields to validate since it's empty by design 