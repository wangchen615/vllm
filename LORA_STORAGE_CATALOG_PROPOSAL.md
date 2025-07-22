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

The storage catalog uses a two-tier caching system:

1. **GPU Cache**: Contains adapters currently active in GPU memory
2. **CPU Cache**: Contains adapters that were evicted from GPU cache for faster reloading

**Key Rules:**
- An adapter can be in either GPU cache OR CPU cache, but never both
- When an adapter is activated, it moves from CPU cache to GPU cache
- When an adapter is evicted from GPU cache, it moves to CPU cache
- When CPU cache is full, oldest adapters are removed entirely

```
┌─────────────────────────────────────────────────────────────┐
│                    Storage Catalog                        │
├─────────────────────────────────────────────────────────────┤
│  Storage Path: /path/to/adapters/                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │ adapter_1/  │  │ adapter_2/  │  │ adapter_3/  │      │
│  └─────────────┘  └─────────────┘  └─────────────┘      │
├─────────────────────────────────────────────────────────────┤
│  CPU Cache (LRU) - Evicted from GPU                      │
│  ┌─────────────┐  ┌─────────────┐                        │
│  │ ID: 2       │  │ ID: 3       │                        │
│  │ Name: a2    │  │ Name: a3    │                        │
│  └─────────────┘  └─────────────┘                        │
├─────────────────────────────────────────────────────────────┤
│  GPU Cache (LRU) - Currently Active                      │
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

## Testing

### Test Scenarios

1. **Basic Functionality**
   - Storage directory scanning
   - Adapter discovery and loading
   - Name-to-ID mapping
   - Basic generation with different adapters

2. **Cache Management**
   - LRU cache eviction behavior
   - GPU and CPU cache capacity limits
   - Adapter movement between caches

3. **External Changes**
   - Adding new adapters to storage
   - Removing adapters from storage
   - Catalog refresh functionality

4. **Performance**
   - Load time measurements
   - Cache hit performance
   - Memory usage simulation

5. **Error Handling**
   - Invalid storage paths
   - Invalid adapter directories
   - Missing adapter files

### Test Commands

```bash
# Run all tests
python tests/lora/test_lora_storage_catalog.py --test all

# Run specific test scenarios
python tests/lora/test_lora_storage_catalog.py --test basic
python tests/lora/test_lora_storage_catalog.py --test cache
python tests/lora/test_lora_storage_catalog.py --test external-changes
python tests/lora/test_lora_storage_catalog.py --test performance
python tests/lora/test_lora_storage_catalog.py --test error

# Clean up after testing
python tests/lora/test_lora_storage_catalog.py --test all --cleanup
```

### Test Environment

The testing script creates a temporary test environment with:

- **Test Directory**: `/tmp/test-lora-adapters` (default)
- **Sample Adapters**: `adapter_1`, `adapter_2`, `adapter_3`
- **Adapter Structure**: Standard PEFT LoRA format
- **Configuration**: Simulated vLLM engine with storage catalog

### Sample Adapter Structure
```
/tmp/test-lora-adapters/
├── adapter_1/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── adapter_2/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
└── adapter_3/
    ├── adapter_config.json
    └── adapter_model.safetensors
```

### Expected Test Output

#### Basic Functionality Test
```
🧪 Testing Basic Storage Catalog Functionality
==================================================
1. Testing adapter discovery...
   ✓ Found 3 adapters: ['1', '2', '3']
2. Testing adapter name mapping...
   ✓ Adapter 1 -> adapter_1
   ✓ Adapter 2 -> adapter_2
   ✓ Adapter 3 -> adapter_3
3. Testing generation with different adapters...
   ✓ Generated with adapter_1: 'Hello, how are you?...'
   ✓ Generated with adapter_2: 'What is machine learning?...'
   ✓ Generated with adapter_3: 'Explain quantum computing...'
✓ Basic functionality test completed successfully
```

#### Cache Management Test
```
🧪 Testing Cache Management
==================================================
1. Testing cache eviction behavior...
   Loading adapter_1 (iteration 1)...
   Loading adapter_2 (iteration 2)...
   Loading adapter_3 (iteration 3)...
     → Cache full, evicting oldest adapter
   Loading adapter_1 (iteration 4)...
     → Reusing adapter_1 from cache
2. Testing cache capacity limits...
   ✓ GPU cache limit: 2
   ✓ CPU cache limit: 3
✓ Cache management test completed successfully
```

#### External Changes Test
```
🧪 Testing External Storage Changes
==================================================
1. Checking initial adapter list...
   ✓ Initial adapters: ['1', '2', '3']
2. Simulating addition of new adapter...
   ✓ Created new adapter_4
3. Refreshing storage catalog...
   ✓ Storage catalog refreshed
4. Simulating removal of adapter_1...
   ✓ Removed adapter_1
5. Refreshing storage catalog again...
   ✓ Storage catalog refreshed
✓ External changes test completed successfully
```

#### Performance Test
```
🧪 Testing Performance
==================================================
1. Measuring load times...
   ✓ Loaded 10 adapters in 0.50 seconds
   ✓ Average load time: 0.050 seconds per adapter
2. Testing cache hit performance...
   ✓ Cache hits in 0.10 seconds
   ✓ Cache hit time: 0.010 seconds per adapter
3. Simulating memory usage...
   ✓ GPU cache: 2 adapters (active)
   ✓ CPU cache: 3 adapters (standby)
   ✓ Storage: 4 adapters (discovered)
✓ Performance test completed successfully
```

#### Error Handling Test
```
🧪 Testing Error Handling
==================================================
1. Testing invalid storage path...
   ✓ Correctly detected invalid path
2. Testing invalid adapter directory...
   ✓ Created invalid adapter directory
3. Testing missing adapter files...
   ✓ Created incomplete adapter
✓ Error handling test completed successfully
```

### Advanced Testing

#### Custom Test Directory
```bash
python tests/lora/test_lora_storage_catalog.py --test all --test-dir /path/to/custom/test/dir
```

#### Integration with Real vLLM
To test with actual vLLM engine, modify the `initialize_engine` method in the test script:

```python
async def initialize_engine(self, max_loras: int = 4, max_cpu_loras: int = 8):
    """Initialize the vLLM engine with storage catalog configuration."""
    self.engine = LLMEngine.from_args(
        model="meta-llama/Llama-2-7b-hf",
        lora_catalog_type="storage",
        lora_catalog_path=self.test_dir,
        max_lora_rank=16,
        max_loras=max_loras,
        max_cpu_loras=max_cpu_loras,
    )
    
    self.manager = self.engine.llm_engine.worker_manager.lora_manager
```

### Validation Checklist

After running tests, verify:

- [ ] All test scenarios pass
- [ ] No error messages in output
- [ ] Test directory is cleaned up (if using `--cleanup`)
- [ ] Performance metrics are reasonable
- [ ] Cache behavior is correct
- [ ] External changes are handled properly

### Troubleshooting Tests

#### Common Issues

1. **Permission Denied**
   ```bash
   # Ensure write permissions to test directory
   chmod 755 /tmp/test-lora-adapters
   ```

2. **Import Errors**
   ```bash
   # Install vLLM with LoRA support
   pip install vllm[lora]
   ```

3. **Test Directory Already Exists**
   ```bash
   # Clean up existing test directory
   rm -rf /tmp/test-lora-adapters
   ```

#### Debug Mode
Add debug prints to the test script:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Test Coverage

The testing suite covers:

- ✅ **Storage Discovery**: Directory scanning and adapter detection
- ✅ **Cache Management**: LRU eviction and capacity limits
- ✅ **API Compatibility**: All existing LoRA APIs work unchanged
- ✅ **Error Handling**: Invalid paths, directories, and files
- ✅ **Performance**: Load times and cache hit rates
- ✅ **External Changes**: Adding/removing adapters from storage
- ✅ **Memory Management**: GPU and CPU cache behavior
- ✅ **Configuration**: Command-line argument handling

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