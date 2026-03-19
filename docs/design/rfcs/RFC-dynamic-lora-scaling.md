# RFC: Dynamic LoRA Slot Scaling for vLLM

**RFC Number:** TBD
**Status:** Draft
**Authors:** (your name)
**Created:** 2026-03-19
**Target vLLM version:** v0.9+

---

## Summary

This RFC proposes a mechanism to dynamically scale the number of concurrent LoRA adapter slots
(`max_loras`) in a running vLLM instance without requiring a server restart. The feature is
delivered as a standalone vLLM plugin (`vllm-lora-scaler`) that exposes a new REST endpoint
`POST /v1/scale_max_loras`.

---

## Motivation

### Problem

vLLM allocates GPU memory for LoRA adapters at startup based on the `--max-loras` flag. Each
LoRA-enabled linear layer pre-allocates stacked weight tensors of shape:

```
lora_a: [max_loras, 1, max_lora_rank, hidden_dim]
lora_b: [max_loras, 1, output_dim,    max_lora_rank]
```

Once the server starts, `max_loras` is immutable. Operators who need to serve more concurrent
LoRA tenants — for example, to handle burst traffic or onboard new customers — must:

1. Stop serving traffic
2. Restart vLLM with a larger `--max-loras`
3. Wait for model weights to reload (minutes on large models)

This causes unnecessary downtime. Conversely, operators who over-provision `max_loras` waste
GPU memory that could be used for KV cache, reducing throughput.

### Use Cases

- **Traffic bursts**: Scale from 4 → 16 LoRA slots when demand spikes, back to 4 afterwards
- **Cost optimization**: Start with fewer slots, scale up only as new tenants are onboarded
- **A/B testing**: Dynamically add slots for experimentation without full restart
- **Multi-tenant SaaS**: Fine-grained resource control per deployment without downtime

---

## Design

### High-Level Approach

The feature is implemented as a **vLLM general plugin** using only public/semi-public vLLM
extension points:

1. A new REST endpoint `POST /v1/scale_max_loras` attached via FastAPI router
2. The endpoint uses the existing `EngineClient.pause_generation()` and
   `EngineClient.collective_rpc()` methods (already on the public protocol) to coordinate
   a pause-reallocate-resume cycle
3. Worker-level reallocation logic is injected into `GPUWorker` via Python `__bases__`
   manipulation (the same mechanism vLLM documents for `worker_extension_cls`)
4. LoRA manager and layer classes are extended via monkey-patching at plugin load time

### Why a Plugin, Not a Core Change

vLLM's plugin system (`vllm.general_plugins` entry point group) is designed for exactly this
kind of operator-level extension. The required `pause_generation`, `resume_generation`, and
`collective_rpc` are all part of the public `EngineClient` abstract interface. No new engine
API surface is needed.

This approach:
- Ships independently of the vLLM release cycle
- Does not require changes to reviewed core code
- Can be tested and iterated without vLLM CI involvement
- Can be upstreamed as a proper core feature once battle-tested

### What Gets Reallocated

When `scale_max_loras(N)` is called, the following structures are resized on every worker:

| Structure | Location | Change |
|-----------|----------|--------|
| `lora_a_stacked`, `lora_b_stacked` | Each LoRA layer module | Freed and reallocated with first dim = N |
| `lora_index_to_id` | `LoRAModelManager` | List resized to length N |
| `token_mapping_meta`, `prompt_mapping_meta` | `PunicaWrapperGPU` | `LoRAKernelMeta` rebuilt at size N |
| `_active_adapters` LRU cache | `LRUCacheLoRAModelManager` | New `LoRALRUCache(N, ...)` |
| `lora_config.max_loras` | `LoRAConfig` | Integer field set to N |

### Operation Sequence

```
1. Client sends POST /v1/scale_max_loras {"new_max_loras": N}
2. Plugin handler calls engine_client.pause_generation(mode="abort")
   → All in-flight requests are aborted; new requests are queued
3. Plugin handler calls engine_client.collective_rpc("reallocate_lora_weights", (N,))
   → Broadcast to all TP/PP workers simultaneously
   → Each worker: evict all LoRAs, free old tensors, allocate new tensors at size N
4. Plugin handler updates app.state.vllm_config.lora_config.max_loras = N
5. Plugin handler calls engine_client.resume_generation()
   → Queued requests begin processing; LoRAs reload on-demand via LRU manager
6. 200 OK returned to client
```

### Scale-Down Safety

On scale-down, all currently loaded LoRA adapters are evicted before the tensors are freed.
This uses the existing `remove_all_adapters()` path, which is the same safe mechanism used
by vLLM's sleep/wake feature. Adapters are reloaded on demand when new requests arrive.

Scale-down to fewer slots than currently *loaded* adapters is safe because eviction happens
before tensor resize — there are no dangling slot indices.

### Memory Behavior

**Scale-up** (`new_max_loras > current`):
- Additional GPU memory is allocated proportional to the number of new slots and model size
- Old tensors are freed before new ones are allocated (via `None` assignment + `empty_cache()`)
- If the GPU cannot fit the new allocation, a CUDA OOM is raised; the handler returns 500

**Scale-down** (`new_max_loras < current`):
- GPU memory is freed; KV cache and other allocations benefit from the headroom

### Tensor-Parallel and Pipeline-Parallel Behavior

`collective_rpc` broadcasts to all workers simultaneously. Each worker independently
reallocates its own shard of the LoRA weight tensors. Because all workers use the same
`max_loras` and their model-parallel shards are deterministic from `tp_rank`/`pp_rank`,
no cross-worker synchronization is needed beyond what `collective_rpc` already provides.

---

## API

### Endpoint

```
POST /v1/scale_max_loras
```

**Request body:**
```json
{
  "new_max_loras": 8
}
```

**Response:**
- `200 OK` — scaling completed successfully
- `400 Bad Request` — `new_max_loras < 1` or LoRA not enabled
- `500 Internal Server Error` — worker reallocation failed (e.g., OOM)

**Prerequisites (same as existing LoRA dynamic loading):**
- `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1`
- `VLLM_PLUGINS=lora_scaler`
- Server started with `--enable-lora`

### Example

```bash
# Scale up from 2 to 8 LoRA slots
curl -X POST http://localhost:8000/v1/scale_max_loras \
     -H "Content-Type: application/json" \
     -d '{"new_max_loras": 8}'

# Scale back down
curl -X POST http://localhost:8000/v1/scale_max_loras \
     -H "Content-Type: application/json" \
     -d '{"new_max_loras": 2}'
```

---

## Plugin Architecture

### Package Layout

```
vllm-lora-scaler/
├── setup.py                         # entry_point: vllm.general_plugins
├── vllm_lora_scaler/
│   ├── __init__.py                  # register() — runs at vLLM import time
│   ├── patches.py                   # monkey-patches for LoRA classes
│   ├── worker_extension.py          # WorkerLoRAScalerMixin
│   └── api_router.py                # FastAPI route handler
```

### Installation

```bash
pip install vllm-lora-scaler
```

The `register()` function in `vllm_lora_scaler/__init__.py` is called automatically by
vLLM's plugin loader when `VLLM_PLUGINS=lora_scaler` is set.

### Extension Points Used

| vLLM Extension Point | Where Defined | How Plugin Uses It |
|---|---|---|
| `vllm.general_plugins` entry point | `vllm/plugins/__init__.py` | Plugin's `register()` called at startup |
| `GPUWorker.__bases__` mutation | `vllm/v1/worker/worker_base.py:248` | Injects `reallocate_lora_weights` method |
| `collective_rpc(method_name, args)` | `vllm/engine/protocol.py:214` | Broadcasts realloc to all workers |
| `pause_generation(mode)` | `vllm/engine/protocol.py:171` | Drains in-flight requests |
| `resume_generation()` | `vllm/engine/protocol.py:194` | Restarts scheduling |
| `app.state.engine_client` | `vllm/entrypoints/openai/api_server.py:349` | Accessed in route handler |
| `app.include_router()` | FastAPI standard | Attaches new route to server |

---

## Alternatives Considered

### Alternative 1: Core vLLM Change

Add `scale_max_loras` directly to `AsyncLLM`, `EngineCore`, and `GPUWorker` in the vLLM
codebase. This is the "clean" long-term approach but requires vLLM PR review, CI, and
release cycle alignment. The plugin approach lets us ship and validate faster.

**Rejected for now**: Deliver as plugin first, upstream as core feature once validated.

### Alternative 2: Restart with State Preservation

Serialize loaded LoRA adapter metadata to disk, restart with new `--max-loras`, reload
adapters. This avoids GPU tensor reallocation complexity but still causes downtime during
restart and weight reload.

**Rejected**: Causes the downtime we are explicitly trying to eliminate.

### Alternative 3: Over-provision at Startup

Start with `--max-loras=N_max` and use LRU eviction to simulate fewer active slots.
Simple, no new code. But wastes GPU memory proportional to `N_max - N_current` even
when most slots are unused — directly reducing KV cache capacity.

**Rejected**: Memory waste is the original problem motivation.

---

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| vLLM internal API changes break monkey-patches | Medium | Pin compatible vLLM version range; CI tests catch breakage |
| CUDA OOM on scale-up | Low-Medium | Catch exception in handler, return 500 with clear message; document memory formula |
| In-flight request loss during `abort` pause | Expected | Document that requests are aborted; clients should retry. Future: support `mode="wait"` |
| `collective_rpc` succeeds on some workers, fails on others | Low | If any worker raises, the exception propagates; state may be inconsistent — document restart as recovery |
| `register_vllm_serve_api_routers` renamed in a vLLM update | Low | Wrap with try/except; fall back to alternative attachment method |

---

## Future Work

- Upstream as a proper core vLLM feature once validated in production
- Support `mode="wait"` to drain requests gracefully before scaling (no aborts)
- Expose memory estimation endpoint: `GET /v1/lora_memory_estimate?max_loras=N`
- Integration with vLLM's autoscaling infrastructure for automatic slot adjustment based on queue depth

---

## References

- vLLM Plugin System: `docs/design/plugin_system.md`
- vLLM LoRA Resolver Plugins: `docs/design/lora_resolver_plugins.md`
- `EngineClient` protocol: `vllm/engine/protocol.py`
- `worker_extension_cls` docs: `vllm/config/parallel.py:235-239`
- `collective_rpc` implementation: `vllm/v1/executor/multiproc_executor.py:930-957`
