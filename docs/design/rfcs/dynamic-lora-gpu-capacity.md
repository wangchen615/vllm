# RFC: Dynamic LoRA GPU Capacity

- **Authors**: Chen Wang ([@wangchen615](https://github.com/wangchen615)), Yue Zhu ([@yuezhu1](https://github.com/yuezhu1))
- **Status**: Draft
- **Created**: 2026-03-19
- **Related**: [LoRA Resolver Plugins](../lora_resolver_plugins.md),
  [Plugin System](../plugin_system.md),
  [In-Place LoRA Reloading](../../features/lora.md#in-place-lora-reloading)

---

## Summary

This RFC proposes dynamically scaling the number of concurrent LoRA adapter GPU slots
(`max_loras`) in a running vLLM instance — without requiring a server restart.

The feature is delivered in two complementary layers:

1. **Core vLLM mechanism**: `resize_lora_slots(N)` on `LoRAModelManager` and workers,
   callable via `collective_rpc`. A `LoRAMemoryNotifier` interface lets external memory
   managers (e.g., [kvcached](https://github.com/ovg-project/kvcached)) signal when GPU
   memory becomes available or constrained. The `LoRAResolver` interface is extended with
   an optional `get_desired_lora_slots()` method for policy-driven scaling.

2. **Plugin package** (`vllm-lora-scaler`): A standalone vLLM general plugin that exposes
   a REST endpoint `POST /v1/scale_max_loras` for operator-triggered scaling, and implements
   the `LoRAResolver` policy interface for automatic memory-pressure-driven scaling.

Together, these enable full runtime control over both *how many* and *which* LoRA adapters
occupy GPU memory without restarting vLLM.

---

## Motivation

### Current Limitations

`max_loras` is a static configuration set at server startup. Each LoRA-enabled linear layer
pre-allocates stacked weight tensors of shape:

```text
lora_a: [max_loras, 1, max_lora_rank, hidden_dim]
lora_b: [max_loras, 1, output_dim,    max_lora_rank]
```

Once the server starts, `max_loras` is immutable. This creates a fundamental tension:

- **Set too low**: Heavy multi-LoRA workloads queue up; popular LoRAs compete for a small
  number of slots, increasing latency.
- **Set too high**: GPU memory is wasted on idle LoRA slots when only a few adapters are
  active — memory that could otherwise be used for KV cache, reducing throughput.

Operators who need to serve more concurrent LoRA tenants must stop serving traffic, restart
vLLM with a larger `--max-loras`, and wait for model weights to reload (minutes on large
models). This causes unnecessary downtime.

### The Opportunity

Several vLLM capabilities make dynamic LoRA capacity tractable:

1. **LoRA Resolver Plugins**: Plugins resolve and load LoRA adapters at request time without
   server restarts. The resolver is a natural policy point for deciding *which* and *how many*
   LoRAs should be in GPU.

2. **In-Place LoRA Reloading** (`load_inplace=True` on `LoRARequest`): A LoRA in a GPU slot
   can be hot-swapped without freeing the slot first, enabling low-overhead adapter rotation.

3. **`collective_rpc` / `pause_generation` / `resume_generation`**: Already-public
   `EngineClient` methods enable coordinated, zero-downtime reallocation across all TP workers.

4. **External memory managers** like [kvcached](https://github.com/ovg-project/kvcached):
   These dynamically shrink and grow the KV cache on the GPU based on load. When kvcached
   releases GPU memory, that memory could be used to load more LoRA adapters — and vice versa.

### Use Cases

- **Bursty multi-LoRA serving**: Scale up GPU slots when many distinct LoRAs are needed;
  scale down during quiet periods to free memory for KV cache.
- **kvcached co-deployment**: LoRA capacity and KV cache capacity share GPU memory
  cooperatively rather than competing with fixed allocations.
- **Traffic bursts**: Scale from 4 → 16 LoRA slots when demand spikes, back to 4 afterwards.
- **Hot-adapter rotation**: Evict cold LoRAs and swap in hot ones without server restart
  using `load_inplace=True`.
- **Multi-tenant SaaS**: Fine-grained resource control per deployment without downtime.
- **Cost efficiency**: Run more LoRA variants on the same GPU by dynamically time-sharing
  GPU memory between KV cache and LoRA weights.

---

## Design

### High-Level Architecture

```text
┌──────────────────────────────────────────────────────────────────┐
│                          vLLM Engine                             │
│                                                                  │
│  ┌──────────────┐  post-batch   ┌────────────────────────────┐   │
│  │  Scheduler   │ ────────────▶ │  _maybe_resize_lora_slots()│   │
│  └──────────────┘               └─────────────┬──────────────┘   │
│                                               │                  │
│  ┌─────────────────────────────┐   ┌──────────▼──────────────┐   │
│  │  POST /v1/scale_max_loras   │   │      LoRAResolver        │   │
│  │  (plugin REST endpoint)     │   │  .get_desired_slots()    │   │
│  └──────────────┬──────────────┘   └──────────┬──────────────┘   │
│                 │  pause / rpc / resume        │                  │
│                 └──────────────┬───────────────┘                  │
│                                │                                  │
│                     ┌──────────▼──────────────┐                   │
│                     │   WorkerLoRAManager      │                   │
│                     │   .resize_lora_slots()   │                   │
│                     └──────────┬──────────────┘                   │
│                                │ collective_rpc                   │
│              ┌─────────────────┼─────────────┐                    │
│              ▼                 ▼             ▼                    │
│           Worker0           Worker1        ...                    │
│        (TP rank 0)       (TP rank 1)                              │
└──────────────────────────────────────────────────────────────────┘

         ┌──────────────────────────────────────┐
         │       External Memory Manager         │
         │  (kvcached / KV Cache Manager)        │
         │                                       │
         │  notify(MEMORY_FREED, bytes) ─────────┼──▶ LoRAMemoryNotifier
         │  notify(MEMORY_CLAIMED, bytes) ───────┼──▶ (triggers resize eval)
         └──────────────────────────────────────┘
```

### Two Scaling Modes

This RFC supports two complementary modes of triggering a resize:

| Mode | Trigger | Use Case |
| --- | --- | --- |
| **Automatic** | Memory watermark crossed or `LoRAMemoryNotifier` event | kvcached co-deployment, continuous load-based scaling |
| **Operator-triggered** | `POST /v1/scale_max_loras` REST call | Manual scaling, SaaS tenant onboarding, A/B testing |

Both modes use the same underlying `resize_lora_slots(N)` mechanism in the core.

### Design Principles

1. **No static upper bound**: GPU tensors are allocated and freed dynamically; there is no
   `max_loras_limit` pre-allocation.
2. **Policy in plugin, mechanism in core**: The *when* and *how many* decision lives in the
   plugin. The *how* (reallocation, LRU eviction, TP coordination) lives in vLLM core.
3. **Between-batch only**: All resizing happens strictly between batches (or during a
   `pause_generation` window) to avoid corrupting kernel index metadata during inference.
4. **Safe bounds**: User-configurable `min_loras` and `max_loras` act as floor and ceiling,
   preventing runaway allocation.
5. **Watermark-based automatic triggering**: Automatic resizing is triggered by GPU memory
   utilization crossing configurable thresholds, not by batch count.
6. **Ship fast via plugin, upstream as core**: Operator-triggered scaling is delivered as a
   plugin first. Once validated, it will be proposed as a core vLLM feature.

---

## Part 1: Core vLLM Changes

### 1. `LoRAConfig` Changes

**File**: `vllm/config/lora.py`

```python
@dataclass
class LoRAConfig:
    # Existing fields (unchanged)
    max_loras: int = 1
    max_cpu_loras: Optional[int] = None
    max_lora_rank: int = 16
    # ...

    # New fields
    min_loras: int = 1
    """Minimum number of LoRA GPU slots. Acts as floor for dynamic resizing.
    Must be >= 1. Only meaningful when dynamic_lora_slots=True."""

    dynamic_lora_slots: bool = False
    """Enable automatic dynamic resizing of GPU LoRA slots at runtime.
    When True, max_loras becomes the initial value and upper bound for
    automatic scaling. Operator-triggered scaling via POST /v1/scale_max_loras
    is always available regardless of this flag."""

    lora_mem_high_watermark: float = 0.8
    """GPU memory utilization above which LoRA slots are proactively reduced.
    Range: 0.0 - 1.0. Only used when dynamic_lora_slots=True."""

    lora_mem_low_watermark: float = 0.5
    """GPU memory utilization below which LoRA slots may be expanded.
    Range: 0.0 - 1.0. Only used when dynamic_lora_slots=True."""

    lora_slot_resize_cooldown_s: float = 1.0
    """Minimum seconds between consecutive automatic LoRA slot resizes.
    Prevents thrashing when memory utilization oscillates near a watermark.
    Only used when dynamic_lora_slots=True."""

    def __post_init__(self):
        # NOTE: existing max_loras validation (must be >= 1) is preserved.
        # New fields are validated only when dynamic_lora_slots is enabled.
        if self.dynamic_lora_slots:
            if self.min_loras < 1:
                raise ValueError("min_loras must be >= 1")
            if self.min_loras > self.max_loras:
                raise ValueError("min_loras must be <= max_loras")
            if not (0.0 < self.lora_mem_low_watermark
                    < self.lora_mem_high_watermark < 1.0):
                raise ValueError(
                    "lora_mem_low_watermark must be less than "
                    "lora_mem_high_watermark, both in (0, 1)")
            if self.lora_slot_resize_cooldown_s < 0:
                raise ValueError(
                    "lora_slot_resize_cooldown_s must be >= 0")
```

---

### 2. `LoRAResolver` Interface Extension

**File**: `vllm/lora/resolver.py`

```python
class LoRAResolver(ABC):

    @abstractmethod
    async def resolve_lora(
        self,
        base_model_name: str,
        lora_name: str,
    ) -> Optional[LoRARequest]:
        """Resolve a LoRA adapter by name and return a LoRARequest."""
        ...

    async def get_desired_lora_slots(
        self,
        current_slots: int,
        active_loras: list[str],
        free_gpu_memory_bytes: int,
        total_gpu_memory_bytes: int,
    ) -> Optional[int]:
        """
        Optional. Return the desired number of GPU LoRA slots, or None to
        keep the current value.

        Called by the engine between batches when dynamic_lora_slots=True.
        The returned value is clamped to [min_loras, max_loras] by the engine.

        Args:
            current_slots: Currently active GPU LoRA slot count.
            active_loras: Names of LoRAs currently occupying GPU slots.
            free_gpu_memory_bytes: Available GPU memory from
                torch.cuda.mem_get_info().
            total_gpu_memory_bytes: Total GPU memory.

        Returns:
            Desired slot count, or None to leave unchanged.
        """
        return None
```

**Backward compatibility**: `get_desired_lora_slots` has a default `return None`
implementation, so all existing `LoRAResolver` implementations continue to work without
any changes.

---

### 3. `LoRAMemoryNotifier` (New File)

**File**: `vllm/lora/memory_notifier.py`

A thread-safe notification interface for external memory managers:

```python
import threading
from enum import Enum
from typing import Callable, Optional
import logging

logger = logging.getLogger(__name__)


class GPUMemoryEvent(Enum):
    MEMORY_FREED = "freed"      # External system released GPU memory
    MEMORY_CLAIMED = "claimed"  # External system is about to allocate memory


class LoRAMemoryNotifier:
    """
    Interface for external GPU memory managers (e.g., kvcached) to notify
    the vLLM LoRA subsystem of memory availability changes.

    External systems call notify() when they free or claim GPU memory.
    vLLM evaluates whether to resize LoRA slots in response.

    Usage (from kvcached or similar):
        notifier = LoRAMemoryNotifier.get_instance()
        notifier.notify(GPUMemoryEvent.MEMORY_FREED, bytes_freed)
    """

    _instance: Optional["LoRAMemoryNotifier"] = None
    _instance_lock: threading.Lock = threading.Lock()

    def __init__(self):
        self._lock = threading.Lock()
        self._callbacks: list[Callable[[GPUMemoryEvent, int], None]] = []
        self._pending_event: Optional[tuple[GPUMemoryEvent, int]] = None

    @classmethod
    def get_instance(cls) -> "LoRAMemoryNotifier":
        # Double-checked locking for thread-safe singleton creation.
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def register_resize_callback(
        self,
        cb: Callable[[GPUMemoryEvent, int], None],
    ) -> None:
        """Register a callback to be invoked on memory events."""
        with self._lock:
            self._callbacks.append(cb)

    def notify(self, event: GPUMemoryEvent, bytes_delta: int) -> None:
        """
        Called by external memory manager to signal a memory change.
        Thread-safe: may be called from kvcached's thread.
        """
        logger.debug("LoRAMemoryNotifier: event=%s bytes_delta=%d",
                     event.value, bytes_delta)
        with self._lock:
            self._pending_event = (event, bytes_delta)
            callbacks = list(self._callbacks)
        for cb in callbacks:
            try:
                cb(event, bytes_delta)
            except Exception:
                logger.exception("LoRA memory resize callback failed")

    def consume_pending_event(
        self,
    ) -> Optional[tuple[GPUMemoryEvent, int]]:
        """Consume and return any pending event (called by engine loop)."""
        with self._lock:
            event = self._pending_event
            self._pending_event = None
        return event
```

---

### 4. Per-Layer Tensor Reallocation

**File**: `vllm/lora/layers/base_linear.py`

```python
def reallocate_lora_weights(self, new_slots: int) -> None:
    """
    Reallocate stacked LoRA tensors for new_slots GPU slots.
    Copies weights for slots that survive the resize.
    Called between batches only.

    NOTE: Does NOT call torch.cuda.empty_cache() — the caller
    (resize_lora_slots) does this once after all layers are done,
    to avoid stalling the CUDA stream once per layer.
    """
    if not hasattr(self, 'lora_a_stacked') or self.lora_a_stacked is None:
        return

    surviving_slots = min(self.lora_a_stacked[0].shape[0], new_slots)
    new_lora_a, new_lora_b = [], []

    for a, b in zip(self.lora_a_stacked, self.lora_b_stacked):
        new_a = torch.zeros(new_slots, *a.shape[1:],
                            dtype=a.dtype, device=a.device)
        new_b = torch.zeros(new_slots, *b.shape[1:],
                            dtype=b.dtype, device=b.device)
        new_a[:surviving_slots].copy_(a[:surviving_slots])
        new_b[:surviving_slots].copy_(b[:surviving_slots])
        new_lora_a.append(new_a)
        new_lora_b.append(new_b)

    del self.lora_a_stacked
    del self.lora_b_stacked

    self.lora_a_stacked = tuple(new_lora_a)
    self.lora_b_stacked = tuple(new_lora_b)
```

The same pattern applies to:

- `vllm/lora/layers/column_parallel_linear.py`
- `vllm/lora/layers/row_parallel_linear.py`
- `vllm/lora/layers/logits_processor.py`
- `vllm/lora/layers/fused_moe.py`

---

### 5. `LoRAModelManager.resize_lora_slots()`

**File**: `vllm/lora/model_manager.py`

The following structures are resized on every worker:

| Structure | Location | Change |
| --- | --- | --- |
| `lora_a_stacked`, `lora_b_stacked` | Each LoRA layer module | Freed and reallocated with first dim = N |
| `lora_index_to_id` | `LoRAModelManager` | List resized to length N |
| `token_mapping_meta`, `prompt_mapping_meta` | `PunicaWrapperGPU` | `LoRAKernelMeta` rebuilt at size N |
| `_active_adapters` LRU cache | `LRUCacheLoRAModelManager` | New `LoRALRUCache(N, ...)` |
| `lora_config.max_loras` | `LoRAConfig` | Integer field updated to N |

```python
@property
def lora_slots(self) -> int:
    return self._lora_slots

def resize_lora_slots(self, new_slots: int) -> None:
    """
    Dynamically resize the number of GPU LoRA slots.

    If shrinking: LRU-evicts active LoRAs down to new_slots.
    If growing: allocates additional slots (empty, loaded on demand).
    Reallocates all per-layer stacked weight tensors.

    Must be called between batches only (or within a pause_generation window).
    """
    if new_slots == self._lora_slots:
        return
    if new_slots < 1:
        raise ValueError(f"new_slots must be >= 1, got {new_slots}")

    logger.info("Resizing LoRA GPU slots: %d -> %d",
                self._lora_slots, new_slots)

    # Evict active LoRAs if shrinking.
    # _active_adapters is a LoRALRUCache; remove_oldest() returns (key, value).
    if new_slots < self._lora_slots:
        while len(self._active_adapters) > new_slots:
            evicted_id, _ = self._active_adapters.remove_oldest()
            self._deactivate_adapter(evicted_id)
            logger.debug("Evicted LoRA %d to free GPU slot", evicted_id)

    # Reallocate per-layer tensors, then release cached GPU memory once.
    for module in self.modules.values():
        if hasattr(module, 'reallocate_lora_weights'):
            module.reallocate_lora_weights(new_slots)
    torch.cuda.empty_cache()  # Called once after all layers are done

    # Resize slot-to-id mapping
    old_mapping = self.lora_index_to_id[:]
    self.lora_index_to_id = old_mapping[:new_slots]
    while len(self.lora_index_to_id) < new_slots:
        self.lora_index_to_id.append(None)

    self._lora_slots = new_slots

    # Re-load surviving active LoRAs into their slots
    for idx, lora_id in enumerate(self.lora_index_to_id):
        if lora_id is not None and lora_id in self._registered_adapters:
            self._load_adapter_to_slot(
                self._registered_adapters[lora_id], idx)
```

---

### 6. `WorkerLoRAManager` Exposure

**File**: `vllm/lora/worker_manager.py`

```python
class WorkerLoRAManager:

    def resize_lora_slots(self, new_slots: int) -> None:
        """Resize GPU LoRA slots. Called by engine via collective RPC."""
        self._adapter_manager.resize_lora_slots(new_slots)

    @property
    def lora_slots(self) -> int:
        return self._adapter_manager.lora_slots
```

---

### 7. Worker RPC Targets

**File**: `vllm/v1/worker/gpu_worker.py`

```python
class Worker:

    def resize_lora_slots(self, new_slots: int) -> None:
        """RPC target: resize LoRA slots on this worker."""
        if self.lora_manager is not None:
            self.lora_manager.resize_lora_slots(new_slots)

    def get_lora_slots(self) -> int:
        """RPC target: return current LoRA slot count."""
        if self.lora_manager is not None:
            return self.lora_manager.lora_slots
        return 0
```

All TP workers are called via `collective_rpc`, ensuring they resize in lockstep.

---

### 8. Engine Automatic Scaling Hook

**File**: `vllm/v1/engine/llm_engine.py`

```python
class LLMEngine:

    def __init__(self, ...):
        if self.lora_config and self.lora_config.dynamic_lora_slots:
            self._setup_dynamic_lora()

    def _setup_dynamic_lora(self) -> None:
        self._lora_resize_pending = False
        self._last_lora_resize_time: float = 0.0
        LoRAMemoryNotifier.get_instance().register_resize_callback(
            self._on_memory_event)

    def _on_memory_event(self, event: GPUMemoryEvent, bytes_delta: int) -> None:
        """Thread-safe: sets flag consumed at next batch boundary."""
        self._lora_resize_pending = True

    async def _maybe_resize_lora_slots(self) -> None:
        """Called post-batch when dynamic_lora_slots=True."""
        if not (self.lora_config and self.lora_config.dynamic_lora_slots):
            return
        pending = LoRAMemoryNotifier.get_instance().consume_pending_event()
        if pending is None and not self._lora_resize_pending:
            return
        self._lora_resize_pending = False

        cfg = self.lora_config
        now = time.monotonic()
        if now - self._last_lora_resize_time < cfg.lora_slot_resize_cooldown_s:
            return

        free, total = torch.cuda.mem_get_info()
        utilization = 1.0 - free / total
        current = self.executor.collective_rpc_single("get_lora_slots")[0]

        if utilization > cfg.lora_mem_high_watermark:
            desired: Optional[int] = max(cfg.min_loras, current - 1)
        elif utilization < cfg.lora_mem_low_watermark:
            desired = min(cfg.max_loras, current + 1)
        else:
            desired = None  # In green zone — defer to resolver

        for resolver in LoRAResolverRegistry.get_all_resolvers():
            resolver_desired = await resolver.get_desired_lora_slots(
                current_slots=current,
                active_loras=self._get_active_lora_names(),
                free_gpu_memory_bytes=free,
                total_gpu_memory_bytes=total,
            )
            if resolver_desired is not None:
                desired = resolver_desired
                break

        if desired is None or desired == current:
            return
        desired = max(cfg.min_loras, min(cfg.max_loras, desired))
        if desired != current:
            self.executor.collective_rpc("resize_lora_slots", args=(desired,))
            self._last_lora_resize_time = time.monotonic()
            logger.info("LoRA GPU slots auto-resized to %d", desired)
```

---

### 9. CudaGraph Handling

**File**: `vllm/v1/cudagraph_dispatcher.py`

Dynamic slot reallocation changes tensor addresses, invalidating captured CUDA graphs.
LoRA cudagraph specialization is disabled when `dynamic_lora_slots=True`:

```python
def _get_lora_cases(self) -> list[int]:
    if self.lora_config is None:
        return [0]
    if self.lora_config.dynamic_lora_slots:
        logger.warning(
            "dynamic_lora_slots=True: disabling LoRA cudagraph "
            "specialization. This may reduce throughput slightly.")
        return [0]
    # Existing logic unchanged
    ...
```

Re-capturing cudagraphs after each resize is deferred as a future optimization.

---

## Part 2: Plugin Package (`vllm-lora-scaler`)

### Why a Plugin (for Operator-Triggered Scaling)

The `POST /v1/scale_max_loras` endpoint is delivered as a vLLM general plugin for two
reasons:

1. It ships independently of the vLLM release cycle and can be validated in production faster.
2. The required `pause_generation`, `resume_generation`, and `collective_rpc` are all part of
   the public `EngineClient` interface — no new engine API surface is needed.

Once battle-tested, the operator-triggered path will be proposed as a core vLLM feature.

### Package Layout

```text
vllm-lora-scaler/
├── setup.py                         # entry_point: vllm.general_plugins
├── vllm_lora_scaler/
│   ├── __init__.py                  # register() — runs at vLLM import time
│   ├── patches.py                   # monkey-patches for LoRA classes
│   ├── worker_extension.py          # WorkerLoRAScalerMixin
│   └── api_router.py                # FastAPI route handler
```

### Operation Sequence (Operator-Triggered)

```text
1. Operator sends POST /v1/scale_max_loras {"new_max_loras": N}
2. Plugin handler calls engine_client.pause_generation(mode="abort")
   → All in-flight requests are aborted; new requests are queued
3. Plugin handler calls engine_client.collective_rpc("resize_lora_slots", (N,))
   → Broadcast to all TP/PP workers simultaneously
   → Each worker: evict LoRAs if shrinking, reallocate tensors, update metadata
4. Plugin handler updates app.state.vllm_config.lora_config.max_loras = N
5. Plugin handler calls engine_client.resume_generation()
   → Queued requests begin processing; LoRAs reload on demand via LRU
6. 200 OK returned to client
```

### REST API

```text
POST /v1/scale_max_loras
```

**Request body:**

```json
{ "new_max_loras": 8 }
```

**Responses:**

- `200 OK` — scaling completed successfully
- `400 Bad Request` — `new_max_loras < 1` or LoRA not enabled
- `500 Internal Server Error` — worker reallocation failed (e.g., OOM)

**Prerequisites:**

```bash
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
export VLLM_PLUGINS=lora_scaler
# Server started with --enable-lora
```

**Example:**

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

### Extension Points Used

| vLLM Extension Point | Where Defined | How Plugin Uses It |
| --- | --- | --- |
| `vllm.general_plugins` entry point | `vllm/plugins/__init__.py` | Plugin's `register()` called at startup |
| `GPUWorker.__bases__` mutation | `vllm/v1/worker/worker_base.py` | Injects `resize_lora_slots` RPC method |
| `collective_rpc(method_name, args)` | `vllm/engine/protocol.py` | Broadcasts realloc to all workers |
| `pause_generation(mode)` | `vllm/engine/protocol.py` | Drains in-flight requests |
| `resume_generation()` | `vllm/engine/protocol.py` | Restarts scheduling |
| `app.state.engine_client` | `vllm/entrypoints/openai/api_server.py` | Accessed in route handler |
| `app.include_router()` | FastAPI standard | Attaches new route to server |

### Plugin Policy Example (Automatic Scaling)

```python
# vllm_lora_scaler/resolver.py

from vllm.lora.resolver import LoRAResolver, LoRAResolverRegistry
from vllm.lora.request import LoRARequest
from vllm.lora.memory_notifier import LoRAMemoryNotifier, GPUMemoryEvent


class DynamicCapacityLoRAResolver(LoRAResolver):

    async def resolve_lora(
        self, base_model_name: str, lora_name: str
    ) -> Optional[LoRARequest]:
        # Delegate to filesystem or HF resolver
        ...

    async def get_desired_lora_slots(
        self,
        current_slots: int,
        active_loras: list[str],
        free_gpu_memory_bytes: int,
        total_gpu_memory_bytes: int,
    ) -> Optional[int]:
        """Scale slots based on distinct active LoRAs and memory pressure."""
        util = 1.0 - free_gpu_memory_bytes / total_gpu_memory_bytes
        n_active = len(active_loras)

        if util > 0.85:
            return max(1, n_active - 1)
        elif util < 0.5 and n_active >= current_slots:
            return current_slots + 1
        return None  # No change needed


def register():
    resolver = DynamicCapacityLoRAResolver()
    LoRAResolverRegistry.register_resolver("DynamicCapacityResolver", resolver)

    # kvcached integration: called by kvcached when it frees/claims memory
    notifier = LoRAMemoryNotifier.get_instance()
    # kvcached calls:  notifier.notify(GPUMemoryEvent.MEMORY_FREED, bytes)
```

---

## Interaction with In-Place LoRA Reloading

In-place LoRA reloading (`load_inplace=True` on `LoRARequest`) hot-swaps a LoRA in an
existing GPU slot without freeing and re-allocating the slot. Combined with dynamic slot
sizing:

1. **Scale down**: `resize_lora_slots(n - 1)` → evicts LRU LoRA, frees a slot's GPU memory.
2. **Swap**: Resolver returns `LoRARequest(load_inplace=True)` for a new LoRA → replaces
   a slot's weights in-place (low overhead, no reallocation).
3. **Scale up**: `resize_lora_slots(n + 1)` → allocates a new slot from freed GPU memory.

---

## Alternatives Considered

### Alternative 1: Static Over-Provisioning

Start with `--max-loras=N_max` and use LRU eviction to simulate fewer active slots.
Simple, no new code, but wastes GPU memory proportional to `N_max - N_current` at all
times — directly reducing KV cache capacity. **Rejected**: memory waste is the original
problem.

### Alternative 2: Restart with State Preservation

Serialize loaded LoRA adapter metadata to disk, restart with new `--max-loras`, reload.
This avoids GPU tensor reallocation complexity but still causes downtime. **Rejected**:
causes the downtime we are explicitly trying to eliminate.

### Alternative 3: Pre-Allocate Upper Bound (`max_loras_limit`)

Pre-allocate tensors at `max_loras_limit` and expose a smaller active window. Avoids
reallocation overhead but wastes static GPU memory proportional to `max_loras_limit`.
This directly conflicts with the kvcached use case where available GPU memory is dynamic.
**Rejected**: incompatible with co-deployment with dynamic KV cache managers.

---

## Failure Modes and Safety

| Scenario | Handling |
| --- | --- |
| Resize during inference batch | Not possible — automatic resize is post-batch only; operator resize uses `pause_generation` |
| Resize to 0 | Clamped to `min_loras` (≥ 1) |
| Resize above `max_loras` | Clamped to `max_loras` |
| OOM during tensor realloc | PyTorch OOM propagated; current slots unchanged (best-effort); see Open Question 3 |
| TP worker resize failure | `collective_rpc` failure — engine logs error and retains current slots |
| Resolver raises exception | Logged and ignored; watermark decision used as fallback |
| Cooldown active | Automatic resize skipped; re-evaluated on next batch boundary |
| kvcached notifies while batch running | Event queued; consumed at next batch boundary |
| In-flight request loss (operator scale) | Requests aborted during `pause_generation`; clients should retry. Future: `mode="wait"` |
| `collective_rpc` partial failure | If some workers raise, state may be inconsistent — document restart as recovery |

---

## Configuration Reference

```bash
# Automatic scaling with kvcached co-deployment
vllm serve meta-llama/Llama-2-7b-hf \
  --enable-lora \
  --max-loras 8 \
  --lora-extra-config '{
    "min_loras": 1,
    "dynamic_lora_slots": true,
    "lora_mem_high_watermark": 0.80,
    "lora_mem_low_watermark": 0.50,
    "lora_slot_resize_cooldown_s": 1.0
  }'

# Operator-triggered scaling only (plugin, no automatic scaling)
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
export VLLM_PLUGINS=lora_scaler
vllm serve meta-llama/Llama-2-7b-hf --enable-lora --max-loras 4
```

---

## Files to Modify (Core vLLM)

| File | Change Type | Description |
| --- | --- | --- |
| `vllm/config/lora.py` | Modify | Add `min_loras`, `dynamic_lora_slots`, watermark fields |
| `vllm/lora/resolver.py` | Modify | Add `get_desired_lora_slots()` default method |
| `vllm/lora/memory_notifier.py` | **New** | `LoRAMemoryNotifier` + `GPUMemoryEvent` |
| `vllm/lora/model_manager.py` | Modify | Add `resize_lora_slots()`, `_lora_slots` property |
| `vllm/lora/layers/base_linear.py` | Modify | Add `reallocate_lora_weights()` |
| `vllm/lora/layers/column_parallel_linear.py` | Modify | Add `reallocate_lora_weights()` |
| `vllm/lora/layers/row_parallel_linear.py` | Modify | Add `reallocate_lora_weights()` |
| `vllm/lora/layers/logits_processor.py` | Modify | Add `reallocate_lora_weights()` |
| `vllm/lora/layers/fused_moe.py` | Modify | Add `reallocate_lora_weights()` |
| `vllm/lora/worker_manager.py` | Modify | Expose `resize_lora_slots()` as RPC target |
| `vllm/v1/engine/llm_engine.py` | Modify | Add `_maybe_resize_lora_slots()` post-batch hook |
| `vllm/v1/worker/gpu_worker.py` | Modify | Add `resize_lora_slots()` + `get_lora_slots()` RPC methods |
| `vllm/v1/cudagraph_dispatcher.py` | Modify | Disable LoRA cudagraph when `dynamic_lora_slots=True` |

---

## Open Questions

1. **Metrics**: Should current `lora_slots` be exposed as a Prometheus gauge so operators can
   observe dynamic behavior?
2. **PP (Pipeline Parallelism)**: This RFC is scoped to TP coordination only. PP support
   (coordinating resize across pipeline stages) is deferred as a follow-up.
3. **Rollback on OOM**: On `torch.cuda.OutOfMemoryError` during reallocation, should we
   attempt to restore the previous slot count via a try/except around the reallocation loop,
   or log and retain whatever partial state exists?
4. **kvcached API**: Should `LoRAMemoryNotifier` be the stable integration point, or should
   we coordinate with the kvcached project on a richer bidirectional interface?
5. **Operator scaling `mode="wait"`**: The current REST endpoint uses `mode="abort"` which
   drops in-flight requests. A `mode="wait"` option that drains requests gracefully before
   scaling is desirable — deferred as future work.

> **Resolved**: Resize frequency throttling is handled by the `lora_slot_resize_cooldown_s`
> config field (default 1.0s), validated in `__post_init__`.
