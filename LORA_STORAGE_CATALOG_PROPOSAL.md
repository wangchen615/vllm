# LoRA Storage Catalog Feature Proposal

## Overview

This proposal introduces a new storage-based LoRA catalog feature for vLLM that allows loading LoRA adapters directly from a storage path (local disk, S3 mount, etc.) without requiring them to be pre-loaded into CPU memory. The feature provides on-demand loading with LRU caching for both GPU and CPU memory management.

## Problem Statement

### Current Limitations

1. **Memory-Only Catalog**: The existing LoRA system only supports in-memory catalogs where all LoRA adapters must be pre-loaded into CPU memory
2. **Limited Scalability**: Large numbers of LoRA adapters consume significant CPU memory even when not in use
3. **Manual Management**: Users must manually register each LoRA adapter before use
4. **No Dynamic Discovery**: No automatic discovery of available LoRA adapters from storage

### Use Cases

1. **Large LoRA Collections**: Organizations with hundreds or thousands of LoRA adapters
2. **Dynamic Environments**: Kubernetes pods with mounted storage containing LoRA adapters
3. **Cost Optimization**: Reduce memory usage by loading adapters only when needed
4. **Storage-Based Workflows**: Direct integration with S3, NFS, or other storage systems

## Design Overview

### Architecture

The storage catalog feature introduces a new `StorageCatalogLoRAModelManager` that:

1. **Scans Storage Directory**: Automatically discovers LoRA adapters in a specified path
2. **On-Demand Loading**: Loads adapters from storage only when requested
3. **LRU Caching**: Manages both GPU and CPU memory with LRU eviction
4. **API Compatibility**: Maintains full compatibility with existing LoRA APIs

### Key Components

```
┌─────────────────────────────────────────────────────────────┐
│                    vLLM Engine                             │
├─────────────────────────────────────────────────────────────┤
│  WorkerLoRAManager                                        │
│  ┌─────────────────┐  ┌─────────────────────────────────┐ │
│  │ Memory Catalog  │  │      Storage Catalog           │ │
│  │ (Existing)      │  │  ┌─────────────────────────┐   │ │
│  │                 │  │  │ StorageCatalogLoRA      │   │ │
│  │ LRUCacheLoRA    │  │  │ ModelManager            │   │ │
│  │ ModelManager    │  │  │                         │   │ │
│  └─────────────────┘  │  │ • Scan storage path     │   │ │
│                       │  │ • Load on demand        │   │ │
│                       │  │ • LRU cache management  │   │ │
│                       │  │ • Name↔ID mapping       │   │ │
│                       │  └─────────────────────────┘   │ │
│                       └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Mutual Exclusivity**: Storage catalog and memory catalog are mutually exclusive
2. **API Compatibility**: All existing LoRA APIs work unchanged
3. **Storage Immutability**: vLLM treats storage as immutable once started
4. **Deterministic Mapping**: Consistent name-to-ID mapping for storage adapters
5. **External Change Handling**: Support for storage catalog changes by external processes

## Implementation Details

### Configuration

New configuration parameters in `LoRAConfig`:

```python
@dataclass
class LoRAConfig:
    catalog_type: str = "memory"  # "memory" or "storage"
    catalog_path: Optional[str] = None  # Path to storage directory
```

### Command Line Arguments

```bash
# Enable storage catalog
--lora-catalog-type storage
--lora-catalog-path /path/to/lora/adapters

# Example with S3 mount
--lora-catalog-type storage
--lora-catalog-path /mnt/s3/lora-adapters
```

### Storage Directory Structure

```
/path/to/lora/adapters/
├── adapter_1/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── new_embeddings.safetensors (optional)
├── adapter_2/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
└── adapter_3/
    ├── adapter_config.json
    └── adapter_model.bin
```

### ID Management Strategy

The storage catalog uses the global incremental ID system for cache management:

1. **Global Counter**: Uses `get_lora_id()` for consistent ID generation
2. **Name Mapping**: Maintains `name → id` and `id → name` mappings
3. **Cache Management**: LRU caches use integer IDs for performance
4. **API Compatibility**: All public APIs use integer IDs

### Caching Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                    Storage Catalog                        │
├─────────────────────────────────────────────────────────────┤
│  Storage Path: /path/to/adapters/                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │ adapter_1/  │  │ adapter_2/  │  │ adapter_3/  │      │
│  └─────────────┘  └─────────────┘  └─────────────┘      │
├─────────────────────────────────────────────────────────────┤
│  CPU Cache (LRU)                                         │
│  ┌─────────────┐  ┌─────────────┐                        │
│  │ ID: 1       │  │ ID: 2       │                        │
│  │ Name: a1    │  │ Name: a2    │                        │
│  └─────────────┘  └─────────────┘                        │
├─────────────────────────────────────────────────────────────┤
│  GPU Cache (LRU)                                         │
│  ┌─────────────┐                                          │
│  │ ID: 1       │                                          │
│  │ Name: a1    │                                          │
│  └─────────────┘                                          │
└─────────────────────────────────────────────────────────────┘
```

## API Changes

### New Configuration Options

```python
# Engine configuration
engine_args = EngineArgs(
    lora_catalog_type="storage",
    lora_catalog_path="/path/to/adapters",
    # ... other args
)
```

### New Manager Methods

```python
# Refresh storage catalog (for external changes)
manager.refresh_storage_catalog()

# Get adapter name from ID (debugging)
name = manager.get_adapter_name(adapter_id)
```

### Existing APIs (Unchanged)

```python
# All existing APIs work unchanged
manager.list_adapters()
manager.activate_adapter(adapter_id)
manager.add_adapter(lora_model)
manager.remove_adapter(adapter_id)
```

## Usage Examples

### Basic Usage

```python
from vllm import LLMEngine

# Start engine with storage catalog
engine = LLMEngine.from_args(
    model="meta-llama/Llama-2-7b-hf",
    lora_catalog_type="storage",
    lora_catalog_path="/path/to/lora/adapters",
    max_lora_rank=16,
    max_loras=4,
    max_cpu_loras=8,
)

# Generate with LoRA adapter
output = engine.generate(
    "Hello, how are you?",
    lora_request=LoRARequest(
        lora_name="my_adapter",  # Directory name in storage
        lora_int_id=1,           # Assigned ID
        lora_path="/path/to/lora/adapters/my_adapter"
    )
)
```

### S3 Mount Example

```bash
# Mount S3 bucket
s3fs my-bucket /mnt/s3

# Start vLLM with S3 storage
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --lora-catalog-type storage \
    --lora-catalog-path /mnt/s3/lora-adapters \
    --max-lora-rank 16 \
    --max-loras 4 \
    --max-cpu-loras 8
```

### Kubernetes Example

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: vllm-lora-server
spec:
  containers:
  - name: vllm
    image: vllm/vllm-openai
    command:
    - python
    - -m
    - vllm.entrypoints.openai.api_server
    args:
    - --model
    - meta-llama/Llama-2-7b-hf
    - --lora-catalog-type
    - storage
    - --lora-catalog-path
    - /mnt/lora-adapters
    - --max-lora-rank
    - "16"
    - --max-loras
    - "4"
    - --max-cpu-loras
    - "8"
    volumeMounts:
    - name: lora-storage
      mountPath: /mnt/lora-adapters
  volumes:
  - name: lora-storage
    persistentVolumeClaim:
      claimName: lora-pvc
```

## Testing Guide

### Prerequisites

1. **vLLM Installation**: Install vLLM with LoRA support
2. **Test LoRA Adapters**: Prepare some LoRA adapters in the expected format
3. **Storage Access**: Ensure read access to the storage path

### Test Setup

#### 1. Create Test LoRA Adapters

```bash
# Create test directory structure
mkdir -p /tmp/test-lora-adapters/{adapter_1,adapter_2,adapter_3}

# Copy your LoRA adapters to the test directory
cp -r /path/to/your/lora/adapter_1/* /tmp/test-lora-adapters/adapter_1/
cp -r /path/to/your/lora/adapter_2/* /tmp/test-lora-adapters/adapter_2/
cp -r /path/to/your/lora/adapter_3/* /tmp/test-lora-adapters/adapter_3/
```

#### 2. Verify Directory Structure

```bash
# Check that each adapter has required files
ls -la /tmp/test-lora-adapters/adapter_1/
# Should contain: adapter_config.json, adapter_model.safetensors

ls -la /tmp/test-lora-adapters/adapter_2/
# Should contain: adapter_config.json, adapter_model.safetensors

ls -la /tmp/test-lora-adapters/adapter_3/
# Should contain: adapter_config.json, adapter_model.safetensors
```

### Test Scenarios

#### 1. Basic Storage Catalog Test

```python
# test_storage_catalog.py
import asyncio
from vllm import LLMEngine, SamplingParams
from vllm.lora.request import LoRARequest

async def test_basic_storage_catalog():
    # Initialize engine with storage catalog
    engine = LLMEngine.from_args(
        model="meta-llama/Llama-2-7b-hf",
        lora_catalog_type="storage",
        lora_catalog_path="/tmp/test-lora-adapters",
        max_lora_rank=16,
        max_loras=4,
        max_cpu_loras=8,
    )
    
    # List available adapters
    adapters = engine.llm_engine.worker_manager.lora_manager.list_adapters()
    print(f"Available adapters: {adapters}")
    
    # Test generation with adapter
    sampling_params = SamplingParams(temperature=0.7, max_tokens=50)
    
    # Use adapter_1
    outputs = await engine.generate(
        "Hello, how are you?",
        sampling_params,
        lora_request=LoRARequest(
            lora_name="adapter_1",
            lora_int_id=1,
            lora_path="/tmp/test-lora-adapters/adapter_1"
        )
    )
    
    print(f"Output: {outputs[0].outputs[0].text}")
    
    # Test with adapter_2
    outputs = await engine.generate(
        "What is machine learning?",
        sampling_params,
        lora_request=LoRARequest(
            lora_name="adapter_2",
            lora_int_id=2,
            lora_path="/tmp/test-lora-adapters/adapter_2"
        )
    )
    
    print(f"Output: {outputs[0].outputs[0].text}")

if __name__ == "__main__":
    asyncio.run(test_basic_storage_catalog())
```

#### 2. Cache Management Test

```python
# test_cache_management.py
import asyncio
from vllm import LLMEngine, SamplingParams
from vllm.lora.request import LoRARequest

async def test_cache_management():
    engine = LLMEngine.from_args(
        model="meta-llama/Llama-2-7b-hf",
        lora_catalog_type="storage",
        lora_catalog_path="/tmp/test-lora-adapters",
        max_lora_rank=16,
        max_loras=2,  # Small limit to test eviction
        max_cpu_loras=3,
    )
    
    manager = engine.llm_engine.worker_manager.lora_manager
    
    # Test cache behavior
    print("Initial adapters:", manager.list_adapters())
    
    # Load multiple adapters to test LRU
    for i in range(4):
        adapter_name = f"adapter_{(i % 3) + 1}"
        lora_request = LoRARequest(
            lora_name=adapter_name,
            lora_int_id=i+1,
            lora_path=f"/tmp/test-lora-adapters/{adapter_name}"
        )
        
        # This should trigger cache eviction
        await engine.generate(
            f"Test {i+1}",
            SamplingParams(max_tokens=10),
            lora_request=lora_request
        )
        
        print(f"After loading {adapter_name}:", len(manager.list_adapters()))

if __name__ == "__main__":
    asyncio.run(test_cache_management())
```

#### 3. External Change Test

```python
# test_external_changes.py
import os
import shutil
import asyncio
from vllm import LLMEngine, SamplingParams
from vllm.lora.request import LoRARequest

async def test_external_changes():
    engine = LLMEngine.from_args(
        model="meta-llama/Llama-2-7b-hf",
        lora_catalog_type="storage",
        lora_catalog_path="/tmp/test-lora-adapters",
        max_lora_rank=16,
        max_loras=4,
        max_cpu_loras=8,
    )
    
    manager = engine.llm_engine.worker_manager.lora_manager
    
    # Initial state
    print("Initial adapters:", manager.list_adapters())
    
    # Simulate external addition
    shutil.copytree(
        "/path/to/new/adapter",
        "/tmp/test-lora-adapters/adapter_4"
    )
    
    # Refresh catalog
    manager.refresh_storage_catalog()
    
    print("After refresh:", manager.list_adapters())
    
    # Simulate external removal
    shutil.rmtree("/tmp/test-lora-adapters/adapter_1")
    
    # Refresh catalog
    manager.refresh_storage_catalog()
    
    print("After removal:", manager.list_adapters())

if __name__ == "__main__":
    asyncio.run(test_external_changes())
```

### Performance Testing

#### 1. Load Testing

```python
# test_performance.py
import asyncio
import time
from vllm import LLMEngine, SamplingParams
from vllm.lora.request import LoRARequest

async def test_performance():
    engine = LLMEngine.from_args(
        model="meta-llama/Llama-2-7b-hf",
        lora_catalog_type="storage",
        lora_catalog_path="/tmp/test-lora-adapters",
        max_lora_rank=16,
        max_loras=4,
        max_cpu_loras=8,
    )
    
    # Test loading time
    start_time = time.time()
    
    for i in range(10):
        adapter_name = f"adapter_{(i % 3) + 1}"
        lora_request = LoRARequest(
            lora_name=adapter_name,
            lora_int_id=i+1,
            lora_path=f"/tmp/test-lora-adapters/{adapter_name}"
        )
        
        await engine.generate(
            f"Test {i+1}",
            SamplingParams(max_tokens=10),
            lora_request=lora_request
        )
    
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(test_performance())
```

### Validation Checklist

- [ ] Storage catalog scans correctly
- [ ] LoRA adapters load from storage
- [ ] LRU cache eviction works
- [ ] GPU and CPU cache management
- [ ] API compatibility maintained
- [ ] External changes handled
- [ ] Performance acceptable
- [ ] Error handling works
- [ ] Memory usage reasonable

## Migration Guide

### From Memory Catalog to Storage Catalog

1. **Prepare Storage**: Organize LoRA adapters in the expected directory structure
2. **Update Configuration**: Change `lora_catalog_type` from `"memory"` to `"storage"`
3. **Specify Path**: Set `lora_catalog_path` to your storage location
4. **Test Gradually**: Start with a subset of adapters to verify functionality

### Configuration Changes

```python
# Before (Memory Catalog)
engine_args = EngineArgs(
    model="meta-llama/Llama-2-7b-hf",
    max_lora_rank=16,
    max_loras=4,
    max_cpu_loras=8,
    # No catalog configuration needed
)

# After (Storage Catalog)
engine_args = EngineArgs(
    model="meta-llama/Llama-2-7b-hf",
    max_lora_rank=16,
    max_loras=4,
    max_cpu_loras=8,
    lora_catalog_type="storage",
    lora_catalog_path="/path/to/lora/adapters",
)
```

## Troubleshooting

### Common Issues

1. **Storage Path Not Found**
   ```
   Error: catalog_path (/path/to/adapters) does not exist
   ```
   **Solution**: Verify the storage path exists and is accessible

2. **Invalid LoRA Directory**
   ```
   Warning: Directory /path/to/adapters/invalid_adapter is not a valid LoRA directory
   ```
   **Solution**: Ensure each adapter directory contains `adapter_config.json` and `adapter_model.safetensors`

3. **Cache Eviction Issues**
   ```
   Error: LoRA adapter not found in storage or cache
   ```
   **Solution**: Check cache limits and ensure adapters are properly loaded

4. **Performance Issues**
   - **Slow Loading**: Consider increasing `max_cpu_loras` for more CPU caching
   - **Memory Issues**: Reduce `max_cpu_loras` or `max_loras`
   - **Storage I/O**: Use faster storage or local caching

### Debug Commands

```python
# Check available adapters
manager = engine.llm_engine.worker_manager.lora_manager
print("Available adapters:", manager.list_adapters())

# Check cache status
print("GPU cache size:", len(manager._active_adapters))
print("CPU cache size:", len(manager._cpu_adapters))

# Get adapter name from ID
name = manager.get_adapter_name(adapter_id)
print(f"Adapter {adapter_id} -> {name}")

# Refresh storage catalog
manager.refresh_storage_catalog()
```

## Future Enhancements

### Potential Improvements

1. **Parallel Loading**: Load multiple adapters concurrently
2. **Compression**: Support compressed LoRA adapters
3. **Remote Storage**: Direct S3/HTTP loading without mounting
4. **Caching Strategy**: Configurable cache policies
5. **Monitoring**: Metrics for cache hit rates and loading times
6. **Validation**: Automatic LoRA adapter validation
7. **Hot Reloading**: Automatic catalog refresh on changes

### API Extensions

```python
# Future API possibilities
manager.set_cache_policy("aggressive")  # Keep more in CPU
manager.set_cache_policy("conservative")  # Minimize memory usage

manager.prefetch_adapter("adapter_name")  # Preload adapter
manager.pin_adapter_by_name("adapter_name")  # Pin by name

# Batch operations
manager.load_multiple_adapters(["adapter_1", "adapter_2", "adapter_3"])
```

## Conclusion

The LoRA storage catalog feature provides a scalable, efficient solution for managing large collections of LoRA adapters. By leveraging storage-based discovery with intelligent caching, it enables organizations to deploy hundreds or thousands of LoRA adapters without the memory overhead of pre-loading all adapters.

The implementation maintains full API compatibility while introducing new capabilities for dynamic adapter management. The feature is designed to be production-ready with proper error handling, performance optimization, and support for external storage changes.

This proposal represents a significant enhancement to vLLM's LoRA capabilities, enabling new use cases and deployment patterns for large-scale LoRA inference. 