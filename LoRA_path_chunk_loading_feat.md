# LoRA Directory Prefix Discovery Features

## Overview

This document describes the implementation of automatic LoRA adapter discovery and management features for vLLM. These features allow vLLM to automatically discover, load, and manage multiple LoRA adapters from a shared directory prefix, eliminating the need for manual configuration of individual adapter paths.

## Features

### 1. Directory Prefix Discovery

**Description**: Automatically discover and load all LoRA adapters from subdirectories under a shared path prefix.

**CLI Argument**: `--lora-dir-prefix <path>`

**Example**:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --enable-lora \
    --lora-dir-prefix /path/to/loras \
    --max-loras 10
```

**Expected Directory Structure**:
```
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
```

**Supported File Formats**:
- `adapter_model.safetensors` (preferred)
- `adapter_model.bin` (fallback)

### 2. Periodic Discovery

**Description**: Continuously monitor the directory for new adapters and register them automatically without server restart.

**CLI Argument**: `--lora-discovery-interval <seconds>`

**Example**:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --enable-lora \
    --lora-dir-prefix /path/to/loras \
    --lora-discovery-interval 300 \
    --max-loras 10
```

**Features**:
- Background task that periodically scans the directory
- Only registers newly discovered adapters
- Continues running even if individual discovery operations fail
- Logs discovery activities for monitoring

### 3. Manual Refresh API

**Description**: REST API endpoint to manually trigger LoRA discovery when periodic discovery is disabled.

**Endpoint**: `POST /v1/refresh_lora_discovery`

**Request**:
```json
{}
```

**Response**:
```json
{
  "message": "Successfully refreshed LoRA discovery. Found 5 total adapters, 2 newly discovered.",
  "discovered_adapters": ["new_adapter1", "new_adapter2"],
  "total_adapters": 5
}
```

**Usage**:
```bash
curl -X POST http://localhost:8000/v1/refresh_lora_discovery \
     -H 'Content-Type: application/json' \
     -d '{}'
```

### 4. Backward Compatibility

**Description**: Works seamlessly with existing LoRA configuration methods.

**Combined Usage**:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --enable-lora \
    --lora-dir-prefix /path/to/loras \
    --lora-modules custom_adapter=/path/to/custom \
    --max-loras 10
```

## Implementation Details

### Core Functions

#### `discover_lora_adapters_from_prefix(dir_prefix: str) -> Dict[str, str]`

Discovers all LoRA adapters from a directory prefix.

**Parameters**:
- `dir_prefix`: The shared directory prefix containing LoRA adapters

**Returns**:
- Dictionary mapping LoRA names to their full paths

**Validation**:
- Checks for valid `adapter_config.json` files
- Supports both `.safetensors` and `.bin` model files
- Validates adapter configuration structure

#### `discover_new_lora_adapters_from_prefix(dir_prefix: str, existing_adapters: Set[str]) -> Dict[str, str]`

Discovers only new LoRA adapters that haven't been registered yet.

**Parameters**:
- `dir_prefix`: The shared directory prefix containing LoRA adapters
- `existing_adapters`: Set of already registered adapter names

**Returns**:
- Dictionary mapping new LoRA names to their full paths

### Architecture Components

#### 1. Engine Arguments (`vllm/engine/arg_utils.py`)

**New Arguments**:
- `lora_dir_prefix: Optional[str] = None`
- `lora_discovery_interval: Optional[int] = None`

**CLI Integration**:
- Added to `EngineArgs.add_cli_args()`
- Proper help text and validation

#### 2. LoRA Utils (`vllm/lora/utils.py`)

**New Functions**:
- `discover_lora_adapters_from_prefix()`
- `discover_new_lora_adapters_from_prefix()`

**Features**:
- Robust error handling
- Support for multiple file formats
- Validation of adapter configurations

#### 3. Serving Engine (`vllm/entrypoints/openai/serving_engine.py`)

**New Methods**:
- `refresh_lora_discovery()`: Manual refresh functionality
- `load_lora_adapter()`: Load a specific LoRA adapter
- `unload_lora_adapter()`: Unload a specific LoRA adapter

**Enhanced Constructor**:
- Added `lora_dir_prefix` parameter
- Added LoRA initialization with `AtomicCounter` and request tracking
- Passed to all serving instances

**New Imports**:
- `LoadLoraAdapterRequest`, `UnloadLoraAdapterRequest` from protocol
- `RefreshLoraDiscoveryRequest`, `RefreshLoraDiscoveryResponse` from protocol
- `AtomicCounter` from `vllm.utils`
- `discover_lora_adapters_from_prefix` from `vllm.lora.utils`

#### 4. API Server (`vllm/entrypoints/openai/api_server.py`)

**New Endpoint**:
- `POST /v1/refresh_lora_discovery`

**Background Task**:
- `periodic_lora_discovery_task()`: Periodic discovery implementation

**Enhanced Initialization**:
- Startup discovery in `init_app_state()`
- Background task creation for periodic discovery
- `lora_dir_prefix` parameter passed to serving instances

#### 5. Protocol (`vllm/entrypoints/openai/protocol.py`)

**New Request/Response Classes**:
- `RefreshLoraDiscoveryRequest`: Empty request for manual refresh
- `RefreshLoraDiscoveryResponse`: Structured response with discovery results

**Updated Imports**:
- Added `List` to typing imports for response fields

### Serving Instance Updates

All serving instances updated to accept `lora_dir_prefix` parameter:

- `OpenAIServingChat`
- `OpenAIServingCompletion`
- `OpenAIServingEmbedding`
- `OpenAIServingTokenization`

**Constructor Changes**:
- Added `lora_dir_prefix: Optional[str] = None` parameter
- Passed parameter to parent constructor calls

## Usage Examples

### Basic Discovery

```bash
# Start server with directory prefix discovery
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --enable-lora \
    --lora-dir-prefix /path/to/loras \
    --max-loras 10
```

### With Periodic Discovery

```bash
# Check for new adapters every 5 minutes
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --enable-lora \
    --lora-dir-prefix /path/to/loras \
    --lora-discovery-interval 300 \
    --max-loras 10
```

### Manual Refresh

```bash
# Trigger manual discovery
curl -X POST http://localhost:8000/v1/refresh_lora_discovery \
     -H 'Content-Type: application/json' \
     -d '{}'
```

### Using Discovered Adapters

```bash
# Use discovered adapters in API calls
curl -X POST http://localhost:8000/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{
       "model": "sql_adapter",
       "messages": [{"role": "user", "content": "Generate SQL for..."}]
     }'
```

### List Available Models

```bash
# List all available models including discovered LoRA adapters
curl http://localhost:8000/v1/models
```

## Benefits

### 1. Simplified Management
- No need to manually specify each adapter path
- Automatic discovery of all adapters in a directory
- Easy to add/remove adapters by adding/removing directories

### 2. Dynamic Updates
- Add new adapters without server restart
- Periodic discovery automatically registers new adapters
- Manual refresh for immediate discovery

### 3. Flexibility
- Works with or without periodic discovery
- Can be combined with existing `--lora-modules` argument
- Supports multiple file formats

### 4. Safety and Security
- Uses configured directory prefix only
- Prevents unauthorized access to arbitrary paths
- Robust error handling and logging

### 5. Performance
- Only loads adapters when first requested
- Efficient discovery with minimal overhead
- Background processing for periodic discovery

## Error Handling

### Discovery Errors
- Invalid directory paths
- Missing adapter configuration files
- Corrupted adapter files
- Permission issues

### API Errors
- Directory prefix not configured
- Discovery operation failures
- Invalid adapter configurations

### Recovery Mechanisms
- Graceful degradation on individual adapter failures
- Continued operation despite discovery errors
- Detailed error logging for debugging

## Configuration Options

### Required Configuration
- `--enable-lora`: Enable LoRA support
- `--lora-dir-prefix`: Directory prefix for discovery

### Optional Configuration
- `--lora-discovery-interval`: Periodic discovery interval (seconds)
- `--max-loras`: Maximum number of LoRA adapters
- `--lora-modules`: Additional manually specified adapters

### Environment Variables
- `VLLM_ALLOW_RUNTIME_LORA_UPDATING`: Enable dynamic LoRA loading

## Testing

### Unit Tests
- `tests/lora/test_lora_dir_prefix.py`: Comprehensive test suite
- Discovery function testing
- Error handling validation
- API endpoint testing

### Integration Tests
- End-to-end discovery workflow
- Periodic discovery functionality
- Manual refresh API testing

## Future Enhancements

### Potential Improvements
1. **File System Monitoring**: Use inotify/fsevents for real-time discovery
2. **Adapter Validation**: Enhanced validation of adapter configurations
3. **Discovery Filters**: Pattern-based filtering of discovered adapters
4. **Metrics**: Prometheus metrics for discovery operations
5. **Caching**: Intelligent caching of discovery results

### Advanced Features
1. **Multi-Directory Support**: Support for multiple directory prefixes
2. **Adapter Dependencies**: Handle adapter dependencies and loading order
3. **Version Management**: Support for adapter versioning
4. **Remote Discovery**: Support for remote/network-based discovery

## Migration Guide

### From Manual Configuration
1. **Before**: Individual `--lora-modules` arguments
2. **After**: Single `--lora-dir-prefix` argument
3. **Benefits**: Simplified configuration, automatic discovery

### Gradual Migration
1. Start with `--lora-dir-prefix` for new adapters
2. Keep existing `--lora-modules` for critical adapters
3. Gradually move adapters to the directory structure
4. Remove manual configuration once migration is complete

## Troubleshooting

### Common Issues
1. **No adapters discovered**: Check directory structure and permissions
2. **Discovery not working**: Verify `--lora-dir-prefix` configuration
3. **Periodic discovery failing**: Check logs for error details
4. **API errors**: Ensure `VLLM_ALLOW_RUNTIME_LORA_UPDATING` is set

### Debug Information
- Detailed logging of discovery operations
- Error messages with context
- API response details for manual refresh
- Background task status information

## Implementation Notes

### Class Naming Convention
- All LoRA-related classes use consistent naming: `LoadLoraAdapterRequest`, `UnloadLoraAdapterRequest`, etc.
- Protocol classes follow the established naming pattern in the codebase

### Import Dependencies
- Added `List` to typing imports in protocol.py for response fields
- All new imports are properly organized and follow existing patterns

### Constructor Parameter Handling
- All serving instances consistently handle the `lora_dir_prefix` parameter
- Parameter is properly passed through the inheritance chain
- Default value of `None` maintains backward compatibility

## Conclusion

The LoRA directory prefix discovery features provide a comprehensive solution for managing multiple LoRA adapters in vLLM. These features simplify configuration, enable dynamic updates, and provide flexibility for different deployment scenarios while maintaining backward compatibility and robust error handling.

The implementation successfully integrates with the existing vLLM architecture, maintaining consistency with established patterns and providing a seamless user experience for LoRA adapter management. 