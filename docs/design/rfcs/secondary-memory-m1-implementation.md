# Secondary memory M1 — implementation plan

| | |
| --- | --- |
| **Status** | Draft |
| **Branch** | `dev` |
| **Owner** | @wangchen615 |
| **Created** | 2026-05-04 |
| **Last revised** | 2026-05-31 (rebased onto `OffloadingConnector`; added Host Memory Pool Manager + cross-deployment recovery) |
| **Component design** | [exclusive-tiered-caching.md](exclusive-tiered-caching.md) |
| **Parent** | [secondary-memory-system-overview.md](secondary-memory-system-overview.md) |

This is the **file-by-file edit list** for M1. Read [exclusive-tiered-caching.md](exclusive-tiered-caching.md) first for *what* M1 does; this document is *which files change*.

> **Terminology**: This document uses neutral hardware-agnostic names — **secondary fast memory** for the novel middle tier and **accelerator** for any device that owns HBM. See [secondary-memory-system-overview.md §Terminology](secondary-memory-system-overview.md#terminology).

## 1. Context (recap)

M1 ships under `OffloadingConnector` (`vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py`), not under `SimpleCPUOffloadConnector`. The earlier plan to fork `SimpleCPUOffloadConnector` is dropped. Reasons:

- `OffloadingConnector` already provides every primitive M1 needed to invent — `OffloadingManager` (medium-keyed, swappable), `CachePolicy` plug-ins (`vllm/v1/kv_offload/cpu/policies/`, registry in `cpu/manager.py:19-22`), `OffloadingSpecFactory` (`factory.py`, register a new spec via `extra_config.spec_name`), `OffloadingEvent.medium`, manager-wrapping pattern (`reuse_manager.py`).
- `OffloadingSpec.get_handlers()` yields `(src_type, dst_type, handler)` tuples; a two-medium spec yields four handlers and the existing worker dispatches by type — no scheduler-side or worker-side fork.
- The Host Memory Pool Manager (HMPM) — required for cross-deployment recovery, which is now part of M1 acceptance — replaces the per-instance `SharedOffloadRegion` (`vllm/v1/kv_offload/cpu/shared_offload_region.py`). The existing `CpuGpuOffloadingHandlers` constructor already takes an optional `mmap_region` (`gpu_worker.py:381`); the HMPM-backed region drops in via that argument.

## 2. Design committed for M1

- **New `OffloadingSpec`**: `SecondaryMemoryOffloadingSpec`, registered as `"SecondaryMemoryOffloadingSpec"` in `OffloadingSpecFactory`. Opt-in via `kv_connector_extra_config.spec_name`.
- **Two CPU media**: `FAST_CPU` and `SLOW_CPU`. Each backed by a `CPUOffloadingManager` instance. Composed by a new `MultiMediaOffloadingManager`.
- **Admission**: a new `AdmissionPolicy` Protocol + `PriorityAdmissionPolicy` default, reading `req_context.priority` against `priority_threshold` (default `1`).
- **Eviction**: each medium uses its own `CachePolicy` (default LRU). No demotion, no promotion, no inter-medium copies.
- **Load**: prefix-cache lookup checks both media (fast first); longer hit wins on prefix-range overlap.
- **HMPM**: a new external service component owning the shared mmap region. vLLM instances attach via Unix domain socket. Exposes `attach`, `detach`, `allocate`, `free`, `lookup_peer`, `claim_orphan`, plus heartbeat. M1: process-level Python service in the `vllm/v1/kv_offload/hmpm/` package.
- **Cross-deployment recovery**: a survivor instance can call `lookup_peer` + `claim_orphan` against the HMPM and prefetch a failed peer's blocks through its own existing handler stack.
- **Config**: new keys in `kv_connector_extra_config` (see [exclusive-tiered §8](exclusive-tiered-caching.md#8-configuration-surface)). Legacy `cpu_bytes_to_use` continues to work and maps to `slow_cpu_bytes` with `fast_cpu_bytes=0`.

## 3. Key constraints discovered during exploration

1. **`SharedOffloadRegion` is single-instance by construction**: mmap path keyed by `instance_id` (`shared_offload_region.py:50`), `O_EXCL` create (`:62-67`), creator unlinks on shutdown (`:184-191`). Cross-deployment requires inverting this lifecycle — that is the HMPM.
2. **`OffloadingManager` is medium-keyed via `OffloadingEvent.medium` and `LoadStoreSpec.medium()`** — no schema change needed to support multiple CPU media. We use distinct medium strings (`"FAST_CPU"`, `"SLOW_CPU"`) returned from two new `LoadStoreSpec` subclasses, OR — simpler — we keep one `CPULoadStoreSpec` class and tag medium internally on the `OffloadingManager` side. Decision in §4.4 below.
3. **`OffloadingSpec.get_handlers()` yields `(src, dst, handler)` tuples** (`base.py:386-398`); a two-medium spec yields four. The worker dispatch table is keyed on `(type(src), type(dst))`. So we DO need two distinct `LoadStoreSpec` subclasses — one per medium — or the worker cannot tell the handlers apart. Updates §4.4: distinct subclasses.
4. **`Request.priority` is on `Request` end-to-end** (`vllm/v1/request.py`). Plumbing to `ReqContext` is one-line: extend `ReqContext` with `priority: int = 0`, populate it where `RequestStatus.req_context` is built (`offloading/scheduler.py:141`).
5. **`CPUOffloadingManager.cache_policy` is a string-keyed plug-in dict** (`cpu/manager.py:19-22`) — adding our own policy later (e.g. heat-driven for the `inclusive` mode) is one entry in the dict, not a refactor.
6. **`FilterReusedOffloadingManager`** in `vllm/v1/kv_offload/reuse_manager.py` is already a wrapper-pattern manager. `MultiMediaOffloadingManager` follows the same pattern and composes cleanly with it (e.g. wrap each backing manager in a `FilterReusedOffloadingManager`).

## 4. Design

### 4.1 New package layout

```text
vllm/v1/kv_offload/
  cpu/
    multi_media_manager.py   (new)  — MultiMediaOffloadingManager
    admission.py             (new)  — AdmissionPolicy Protocol + PriorityAdmissionPolicy
    secondary_memory_spec.py (new)  — SecondaryMemoryOffloadingSpec, registered in factory
    fast_cpu_spec.py         (new)  — FastCpuLoadStoreSpec subclass
    slow_cpu_spec.py         (new)  — SlowCpuLoadStoreSpec subclass
  hmpm/                      (new package)
    __init__.py
    server.py                — HostMemoryPoolManager service entry point
    client.py                — HmpmClient used by the spec to attach
    pool_region.py           — HmpmSharedOffloadRegion (drop-in for SharedOffloadRegion)
    protocol.py              — RPC message definitions
    runner.py                — `python -m vllm.v1.kv_offload.hmpm` launcher
```

### 4.2 New scheduler-side code

Nothing forks `OffloadingConnectorScheduler`. The only edit is in `ReqContext`:

```python
# vllm/v1/kv_offload/base.py
@dataclass
class ReqContext:
    kv_transfer_params: dict[str, Any] | None = None
    priority: int = 0  # NEW
```

And in `offloading/scheduler.py:141`:

```python
self.req_context = ReqContext(
    kv_transfer_params=self.req.kv_transfer_params,
    priority=self.req.priority,   # NEW
)
```

That is the entirety of the upstream-touching change. Everything else is in our new package.

### 4.3 `MultiMediaOffloadingManager`

Implements the full `OffloadingManager` interface (`base.py:319-216`). Composition over inheritance — wraps two `CPUOffloadingManager` instances:

```python
class MultiMediaOffloadingManager(OffloadingManager):
    def __init__(self, fast: OffloadingManager, slow: OffloadingManager,
                 admission_policy: AdmissionPolicy):
        self._fast = fast
        self._slow = slow
        self._admission = admission_policy

    def lookup(self, key, req_context):
        # fast first, then slow; merge in-flight states
        r = self._fast.lookup(key, req_context)
        if r is True:  return True
        if r is None:  return None
        return self._slow.lookup(key, req_context)

    def prepare_load(self, keys, req_context):
        # split keys by which medium has them; build a composite LoadStoreSpec
        # M1: "longest contiguous from one medium" — return spec from whichever
        # medium has the longer contiguous prefix; the other medium's hits are
        # discarded for this load.
        ...

    def touch(self, keys):
        # touch on whichever medium has each key
        ...

    def complete_load(self, keys):
        # dispatch per-key by which medium owns each key
        ...

    def prepare_store(self, keys, req_context):
        target = self._admission.choose(req_context)  # FAST or SLOW manager
        return target.prepare_store(keys, req_context)

    def complete_store(self, keys, success=True):
        # dispatch per-key by which medium owns the key
        ...

    def take_events(self):
        yield from self._fast.take_events()
        yield from self._slow.take_events()

    def shutdown(self):
        self._fast.shutdown()
        self._slow.shutdown()
```

Internal bookkeeping: a small `dict[OffloadKey, Literal["fast", "slow"]]` tracks which medium owns each key, populated on `prepare_store` completion and consulted by `complete_load`, `touch`, `complete_store`. Updated by event consumption when blocks are evicted.

### 4.4 `LoadStoreSpec` subclasses per medium

```python
# fast_cpu_spec.py
class FastCpuLoadStoreSpec(BlockIDsLoadStoreSpec):
    @staticmethod
    def medium() -> str:
        return "FAST_CPU"

# slow_cpu_spec.py
class SlowCpuLoadStoreSpec(BlockIDsLoadStoreSpec):
    @staticmethod
    def medium() -> str:
        return "SLOW_CPU"
```

Both are thin wrappers around the existing `CPULoadStoreSpec` shape — they just carry distinct types so the worker's dispatch table (keyed on `(src_type, dst_type)`) routes to the right handler.

### 4.5 `SecondaryMemoryOffloadingSpec`

Mirrors `CPUOffloadingSpec` (`vllm/v1/kv_offload/cpu/spec.py`) but creates two manager+handler pairs and composes them.

```python
class SecondaryMemoryOffloadingSpec(OffloadingSpec):
    def __init__(self, vllm_config, kv_cache_config):
        super().__init__(vllm_config, kv_cache_config)

        # Parse capacity per medium
        fast_bytes = int(self.extra_config.get("fast_cpu_bytes", 0))
        slow_bytes = int(self.extra_config.get("slow_cpu_bytes",
                          self.extra_config.get("cpu_bytes_to_use", 0)))
        if fast_bytes == 0 and slow_bytes == 0:
            raise Exception("must specify fast_cpu_bytes or slow_cpu_bytes")

        # Compute per-medium block counts (same arithmetic as CPUOffloadingSpec)
        ...
        self.fast_blocks = fast_bytes // kv_bytes_per_offloaded_block
        self.slow_blocks = slow_bytes // kv_bytes_per_offloaded_block

        # Placement mode (M1: only "partitioned" accepted)
        mode = self.extra_config.get("placement_mode", "partitioned")
        if mode != "partitioned":
            raise Exception(f"placement_mode={mode!r} not in M1; "
                            f"see secondary-memory-resiliency.md")

        self._priority_threshold = int(
            self.extra_config.get("priority_threshold", 1))
        self._eviction_policy = self.extra_config.get("eviction_policy", "lru")
        self._hmpm_enabled = bool(self.extra_config.get("hmpm_enabled", False))
        self._hmpm_socket = self.extra_config.get(
            "hmpm_socket_path", "/var/run/hillock-hmpm.sock")
        self._hmpm_deployment_id = self.extra_config.get(
            "hmpm_deployment_id") or vllm_config.instance_id

        self._manager: MultiMediaOffloadingManager | None = None
        self._fast_handlers: CpuGpuOffloadingHandlers | None = None
        self._slow_handlers: CpuGpuOffloadingHandlers | None = None
        self._hmpm_client: HmpmClient | None = None
        self._fast_region: SharedOffloadRegion | HmpmSharedOffloadRegion | None = None
        self._slow_region: SharedOffloadRegion | HmpmSharedOffloadRegion | None = None

    def get_manager(self) -> OffloadingManager:
        if self._manager is None:
            fast_mgr = CPUOffloadingManager(self.fast_blocks, self._eviction_policy, ...)
            slow_mgr = CPUOffloadingManager(self.slow_blocks, self._eviction_policy, ...)
            policy = PriorityAdmissionPolicy(fast_mgr, slow_mgr, self._priority_threshold)
            self._manager = MultiMediaOffloadingManager(fast_mgr, slow_mgr, policy)
        return self._manager

    def get_handlers(self, kv_caches):
        if self._fast_handlers is None:
            self._setup_regions(kv_caches)
            self._fast_handlers = CpuGpuOffloadingHandlers(
                kv_caches=kv_caches,
                block_size_factor=self.block_size_factor,
                num_cpu_blocks=self.fast_blocks,
                mmap_region=self._fast_region,
            )
            self._slow_handlers = CpuGpuOffloadingHandlers(
                kv_caches=kv_caches,
                block_size_factor=self.block_size_factor,
                num_cpu_blocks=self.slow_blocks,
                mmap_region=self._slow_region,
            )

        # Four handlers: GPU↔FAST, GPU↔SLOW
        yield (GPULoadStoreSpec, FastCpuLoadStoreSpec,
               self._fast_handlers.gpu_to_cpu_handler)
        yield (FastCpuLoadStoreSpec, GPULoadStoreSpec,
               self._fast_handlers.cpu_to_gpu_handler)
        yield (GPULoadStoreSpec, SlowCpuLoadStoreSpec,
               self._slow_handlers.gpu_to_cpu_handler)
        yield (SlowCpuLoadStoreSpec, GPULoadStoreSpec,
               self._slow_handlers.cpu_to_gpu_handler)

    def _setup_regions(self, kv_caches):
        if self._hmpm_enabled:
            self._hmpm_client = HmpmClient.connect(
                socket_path=self._hmpm_socket,
                deployment_id=self._hmpm_deployment_id,
            )
            self._fast_region = HmpmSharedOffloadRegion(
                self._hmpm_client, medium="FAST_CPU", num_blocks=self.fast_blocks, ...)
            self._slow_region = HmpmSharedOffloadRegion(
                self._hmpm_client, medium="SLOW_CPU", num_blocks=self.slow_blocks, ...)
        else:
            # Fallback: use upstream SharedOffloadRegion (one per medium).
            # No cross-deployment recovery in this path.
            ...
```

Registered in `factory.py`:

```python
OffloadingSpecFactory.register_spec(
    "SecondaryMemoryOffloadingSpec",
    "vllm.v1.kv_offload.cpu.secondary_memory_spec",
    "SecondaryMemoryOffloadingSpec",
)
```

### 4.6 Host Memory Pool Manager — implementation outline

**Service** (`vllm/v1/kv_offload/hmpm/server.py`):

- Single Python process. Owns one `mmap.mmap` over `/dev/shm/hillock_secondary_pool.mmap`.
- Listens on a Unix domain socket. Length-prefixed msgpack messages — keep it simple, swap to gRPC if multi-language clients show up.
- In-memory state: `dict[deployment_id, DeploymentState]` where `DeploymentState` carries: attach handle, last heartbeat timestamp, per-medium `dict[OffloadKey, BlockId]`.
- A background thread sweeps for stale heartbeats every 1 s; deployments past timeout are marked **orphaned** (their blocks remain in the table; `claim_orphan` can move ownership).

**Client** (`vllm/v1/kv_offload/hmpm/client.py`):

- Long-lived UDS connection. Background thread sends heartbeats. Synchronous `attach`/`allocate`/`free`/`lookup_peer`/`claim_orphan` on the main thread. M1 is single-host so latency is fine; if it shows up in flame graphs we batch.

**Pool region** (`vllm/v1/kv_offload/hmpm/pool_region.py`):

- `HmpmSharedOffloadRegion` exposes the same interface as `SharedOffloadRegion` (`create_next_view`, `cleanup`, `is_pinned`) so it drops into `CpuGpuOffloadingHandlers(mmap_region=...)` unchanged. Internally it uses the HMPM-allocated slice of the shared mmap (the HMPM tells the client which byte range belongs to which medium for this deployment).

**Launcher** (`vllm/v1/kv_offload/hmpm/runner.py`):

```bash
python -m vllm.v1.kv_offload.hmpm \
    --pool-bytes 64GB \
    --socket-path /var/run/hillock-hmpm.sock \
    --heartbeat-timeout-ms 3000
```

The HMPM is launched separately from any vLLM instance. M1 ships systemd unit files as examples; users can run it under any process supervisor.

### 4.7 Cross-deployment recovery (M1 acceptance)

This is the only mechanism in M1 that crosses deployment boundaries. It lives entirely in the HMPM's API (`lookup_peer`, `claim_orphan`) and a small recovery driver.

**Recovery driver** (`vllm/v1/kv_offload/hmpm/recovery.py`, new):

```python
class CrossDeploymentRecoveryDriver:
    """Optional driver. If enabled, polls HMPM for orphaned peers and
    claims their blocks for in-flight requests on this deployment.

    M1: external trigger (orchestrator notifies this deployment which
    request-ids to recover). Future: the driver polls automatically.
    """
    def recover(self, peer_deployment_id: str, request_ids: list[str]):
        keys = self._compute_offload_keys(request_ids)
        peers = self._hmpm.lookup_peer(keys)
        present = [p for p in peers if p.found]
        self._hmpm.claim_orphan(peer_deployment_id,
                                [p.key for p in present])
        # The blocks are now ours in the HMPM. The next prefix-cache
        # lookup for these requests will hit, and the existing load
        # path moves them to GPU.
```

The hard part is *not* the load path — it's reusing the existing one verbatim. `claim_orphan` updates the HMPM's ownership table; on the next `manager.lookup(key, ...)` the multi-medium manager (re-)discovers the keys via the HMPM and returns `True`. The handler dispatches a CPU→GPU copy through the existing `CpuGpuOffloadingHandlers`.

The bookkeeping the multi-media manager needs: when `claim_orphan` succeeds, the local `MultiMediaOffloadingManager._owns` map is repopulated for the claimed keys. This is a small extra entry-point on the manager (`absorb_claimed_keys(keys, medium)`), not a structural change.

## 5. Critical files to modify or create

| Path | Action | Why |
| --- | --- | --- |
| `vllm/v1/kv_offload/base.py` | **Edit** — add `priority: int = 0` to `ReqContext` | Plumb `Request.priority` to admission policy |
| `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:141` | **Edit** — populate `req_context.priority = self.req.priority` | One line. Only upstream-touching change. |
| `vllm/v1/kv_offload/factory.py` | **Edit** — register `SecondaryMemoryOffloadingSpec` | New spec entry |
| `vllm/v1/kv_offload/cpu/secondary_memory_spec.py` | **New** | The new `OffloadingSpec` |
| `vllm/v1/kv_offload/cpu/multi_media_manager.py` | **New** | `MultiMediaOffloadingManager` |
| `vllm/v1/kv_offload/cpu/admission.py` | **New** | `AdmissionPolicy` Protocol + `PriorityAdmissionPolicy` |
| `vllm/v1/kv_offload/cpu/fast_cpu_spec.py` | **New** | `FastCpuLoadStoreSpec` (medium="FAST_CPU") |
| `vllm/v1/kv_offload/cpu/slow_cpu_spec.py` | **New** | `SlowCpuLoadStoreSpec` (medium="SLOW_CPU") |
| `vllm/v1/kv_offload/hmpm/__init__.py` | **New** | Package init |
| `vllm/v1/kv_offload/hmpm/server.py` | **New** | HMPM service |
| `vllm/v1/kv_offload/hmpm/client.py` | **New** | `HmpmClient` |
| `vllm/v1/kv_offload/hmpm/pool_region.py` | **New** | `HmpmSharedOffloadRegion` |
| `vllm/v1/kv_offload/hmpm/protocol.py` | **New** | RPC message definitions |
| `vllm/v1/kv_offload/hmpm/runner.py` | **New** | `python -m vllm.v1.kv_offload.hmpm` |
| `vllm/v1/kv_offload/hmpm/recovery.py` | **New** | `CrossDeploymentRecoveryDriver` |

**Not modified for M1**: `OffloadingConnector`, `OffloadingConnectorScheduler` (except the one-line `ReqContext` population), `OffloadingConnectorWorker`, `CPUOffloadingManager`, `CpuGpuOffloadingHandlers` (we use it as-is, twice), `SharedOffloadRegion` (we provide a drop-in alternative for HMPM mode), `BlockPool`, the cache-policy registry (`cpu/policies/`).

## 6. Out of scope for M1

- **Inclusive cascade between media** (heat tracking + demote/promote) — see [resiliency RFC](secondary-memory-resiliency.md). M1 deliberately has no inter-medium copies.
- **Asymmetric performance between the two media** — both are pinned host DRAM in the M1 emulator. Real secondary-memory backings are an HMPM-side change.
- **Placement modes other than `partitioned`** — `inclusive`, `hybrid`, `replicate` are described in the [resiliency RFC](secondary-memory-resiliency.md). The spec rejects non-`partitioned` modes today.
- **Hot-failure detection inside one deployment** (a medium going unresponsive while the deployment is alive) — see [resiliency RFC](secondary-memory-resiliency.md). M1 only handles peer-deployment failures via the HMPM heartbeat.
- **Model-sharing variant** (read-only / externally-managed tier) — see [resiliency RFC](secondary-memory-resiliency.md). M1's HMPM does not yet support read-only attaches.
- **Multi-host HMPM** — single host only. Cross-host pool is post-M1.
- **HMPM auth and quota enforcement** — M1 uses a UDS with filesystem permissions; no auth on the wire. Quota enforcement is best-effort, single-tenant.
- **Sophisticated admission policies** — only priority-thresholded routing in M1. The `AdmissionPolicy` Protocol is the seam for richer policies later.
- **Promotion on slow hit** — partitioned design has no need.
- HMA multi-group interaction — should Just Work because each medium uses an unmodified `CPUOffloadingManager`, but untested.
- Per-medium Prometheus metrics — add as a follow-up; the medium tag is already on `OffloadingEvent`.

## 7. Relationship to the resiliency proposal

[secondary-memory-resiliency.md](secondary-memory-resiliency.md) owns the design for `inclusive`, `hybrid`, `replicate` placement modes plus the failure-detection / failover / re-replication mechanisms inside one deployment. It is **not** scheduled for M1.

What M1 must *not* foreclose so the resiliency proposal remains cheap to land later:

- **`MultiMediaOffloadingManager` is replaceable per placement mode.** The spec instantiates one of `MultiMediaOffloadingManager` / `InclusiveCascadeManager` / `HybridReplicateManager` / `ReplicateManager` based on `placement_mode`. The wrapper-pattern (already present in `FilterReusedOffloadingManager`) keeps each variant ~200 LOC.
- **`AdmissionPolicy` is composable with placement modes.** `inclusive` can ignore admission entirely (cascade is admission-free); `replicate` writes to all media regardless. The Protocol stays.
- **HMPM API has room for read-only attaches.** `attach(deployment_id)` returns a handle; a future `attach_readonly(deployment_id, owner_managed=True)` covers the model-sharing variant without breaking the M1 client.
- **Per-medium events are already there.** `OffloadingEvent.medium` is in `base.py` today; no schema change needed when resiliency adds tier-health state and quarantine events.
- **Cross-deployment recovery (M1) and cross-medium failover (resiliency) are different mechanisms.** M1's HMPM heartbeat handles peer-deployment death; resiliency's `TierHealth` state machine handles a single deployment's medium going slow / quarantined. Both can coexist.

If M1 review uncovers anything that would constrain the resiliency design, we update both RFCs together.

## 8. Verification

### Unit tests (new file `tests/v1/kv_offload/cpu/test_multi_media_manager.py`)

1. **Construction**: build `SecondaryMemoryOffloadingSpec` with `fast_cpu_bytes=N*block_size`, `slow_cpu_bytes=M*block_size`, `priority_threshold=1`, `hmpm_enabled=False`. Assert `get_manager()` returns `MultiMediaOffloadingManager`; `get_handlers()` yields four entries with the expected `(src, dst)` types.
2. **Admission routing**: call `manager.prepare_store(keys, ReqContext(priority=0))` → blocks land in `_fast`. Same call with `priority=5` → blocks land in `_slow`. Assert no overlap of physical block ids.
3. **Independent eviction**: fill `_fast` to capacity with `priority=0` traffic. One more `priority=0` `prepare_store` triggers `_fast` LRU eviction; assert `_slow.num_allocated_blocks` is unchanged (no demotion).
4. **Cross-medium load lookup**: prime `_fast` with key A, `_slow` with key B. `manager.lookup(A, ...)` → `True` from fast. `manager.lookup(B, ...)` → `True` from slow. `prepare_load([A, B], ...)` → returns the longer of the two (M1 rule).
5. **Same key in both media (allowed)**: store key K with `priority=0` and again with `priority=5`. Each medium has its own physical block for K; `lookup(K, ...)` returns `True` (fast wins).
6. **Threshold disabled**: `fast_cpu_bytes=0` → all stores route to `_slow`. Backward-compat path (legacy `cpu_bytes_to_use` users see no behavior change).
7. **`ReqContext.priority` plumbing**: build a `RequestStatus` with a `Request` carrying `priority=2`. Assert `req_status.req_context.priority == 2`.

### HMPM tests (new file `tests/v1/kv_offload/hmpm/test_hmpm_service.py`)

1. **Attach/detach**: launch HMPM service in a subprocess; client attaches; allocate two blocks; detach; reattach with same `deployment_id`; assert blocks are still owned.
2. **Heartbeat orphan detection**: deployment A attaches; allocates one block; client process is killed (no clean detach). After heartbeat timeout, the HMPM marks A orphaned. Deployment B `lookup_peer`s A's key, finds it, calls `claim_orphan`; assert ownership transferred.
3. **Cross-deployment recovery end-to-end** (the M1 acceptance test): two `SecondaryMemoryOffloadingSpec` instances attached to one HMPM. Instance A stores keys for one request via `prepare_store`. Kill instance A. Instance B's `CrossDeploymentRecoveryDriver` runs `recover("A", [request_id])`; assert subsequent `manager.lookup(key, ...)` on B returns `True`.

### Integration test (extend `tests/v1/kv_offload/test_end_to_end.py` or create)

1. Run a workload mixing high-priority (`priority=0`) and low-priority (`priority=5`) requests through a small model (e.g. `facebook/opt-125m`). Inspect per-medium hit counters via `OffloadingEvent.medium`. Assert: high-priority hits come predominantly from `FAST_CPU`, low-priority hits from `SLOW_CPU`.
2. Output bit-exactness for fixed seed and identical prompts regardless of which medium served the prefix-cache hit.

### Manual smoke test (per AGENTS.md workflow)

```bash
.venv/bin/python -m pytest tests/v1/kv_offload/cpu/ tests/v1/kv_offload/hmpm/ -v
pre-commit run --all-files
```

Run on GPU with HMPM:

```bash
# Terminal 1: HMPM service
.venv/bin/python -m vllm.v1.kv_offload.hmpm --pool-bytes 1GB

# Terminal 2: vLLM instance
VLLM_USE_V1=1 .venv/bin/python -c "
from vllm import LLM, SamplingParams
import json
llm = LLM(
    model='facebook/opt-125m',
    enable_prefix_caching=True,
    kv_transfer_config=json.dumps({
        'kv_connector': 'OffloadingConnector',
        'kv_connector_extra_config': {
            'spec_name': 'SecondaryMemoryOffloadingSpec',
            'fast_cpu_bytes': 64 * 1024 * 1024,
            'slow_cpu_bytes': 256 * 1024 * 1024,
            'priority_threshold': 1,
            'hmpm_enabled': True,
        },
        'kv_role': 'kv_both',
    }),
)
out_hi = llm.generate(['Hello high priority ' * 100],
                     SamplingParams(max_tokens=32), priority=0)
out_lo = llm.generate(['Hello low priority ' * 100],
                     SamplingParams(max_tokens=32), priority=5)
print(out_hi[0].outputs[0].text)
print(out_lo[0].outputs[0].text)
"
```

Expected: HMPM logs `attached deployment <id>`, allocations distributed across `FAST_CPU` and `SLOW_CPU`, per-medium counters at the end.

## 9. Open questions deferred to during-implementation

- **`req_context.priority` vs an `extra` dict on `ReqContext`**: a typed `priority: int` field is clean for M1 but locks `ReqContext` to one extra signal. An alternative is a generic `extra: dict[str, Any] = {}` field that admission policies read freely. Decide during implementation; the visible API stays the same either way (the user sets `Request.priority`, not `req_context.extra`).
- **Default `priority_threshold`**: `1` is the proposal — default-priority requests land in `FAST_CPU`, explicit de-prioritization lands in `SLOW_CPU`. An alternative is `None` (operator must set explicitly when both media are configured) — safer against accidental misconfiguration but slightly less ergonomic. Decide during implementation.
- **HMPM transport**: msgpack-over-UDS in M1. If the protocol grows past trivial size, we move to gRPC; if cross-host shows up, we move to gRPC anyway. Pick the simplest thing that works for M1.
- **HMPM eviction semantics across deployments**: when the pool is full and a new `allocate` arrives, whose blocks get evicted? M1: round-robin per-deployment quota, no priority. Future: HMPM honors a deployment-level priority hint. Out of M1.
- **Multi-class admission**: `priority_threshold` admits a binary partition. If a deployment has 3+ priority classes that should route to different media, M1 collapses them to "above/below threshold." A future enhancement could make `AdmissionPolicy` consult a list of thresholds or a callable. Out of M1.

## 10. Branch & PR plan

- Work on branch `dev` (already created off synced upstream `main`).
- Three focused PRs into your fork's `dev` once M1 lands green:
  1. `MultiMediaOffloadingManager` + admission + spec + LoadStoreSpec subclasses (no HMPM dependency).
  2. HMPM service + client + pool region + protocol.
  3. Cross-deployment recovery driver + integration tests.
- Not upstream yet — upstream will want the broader design discussion first.
- Label issues/PRs with `project:secondary-memory`.

## 11. Suggested issue breakdown (for project board)

These map 1:1 to the task list tracked during planning. Each can become a GitHub issue under `project:secondary-memory`. Checkboxes indicate suggested dependency order (top-down).

- [ ] **[Prep] Verify dev environment + test paths** — confirm `.venv` setup, find existing `tests/v1/kv_offload/`. Confirm `Request.priority` is reachable from the offload-side scheduler hook.
- [ ] **[Plumbing] Add `priority` to `ReqContext`** — one new field on the dataclass; one-line edit in `offloading/scheduler.py:141`. PR-1 starter.
- [ ] **[Spec] `FastCpuLoadStoreSpec` / `SlowCpuLoadStoreSpec`** — two thin subclasses; tests cover `medium()` and `(src,dst)`-keyed worker dispatch.
- [ ] **[Manager] `AdmissionPolicy` Protocol + `PriorityAdmissionPolicy`** — minimal admission seam; unit tests.
- [ ] **[Manager] `MultiMediaOffloadingManager`** — composes two `CPUOffloadingManager`s; full `OffloadingManager` interface; unit tests.
- [ ] **[Spec] `SecondaryMemoryOffloadingSpec`** — wires manager + handlers; registered in factory; backward-compat for `cpu_bytes_to_use`.
- [ ] **[HMPM] Service + client + protocol** — one small service, one client, one message file. Unit tests under `tests/v1/kv_offload/hmpm/`.
- [ ] **[HMPM] Pool region drop-in** — `HmpmSharedOffloadRegion` exposes the same shape as `SharedOffloadRegion`; passes the same tests.
- [ ] **[HMPM] Heartbeat + orphan detection** — background sweep; unit tests with controlled clock.
- [ ] **[Recovery] `CrossDeploymentRecoveryDriver`** — `lookup_peer` + `claim_orphan` driver; integration test using two spec instances on one HMPM.
- [ ] **[Tests] Unit tests for multi-media manager** — admission routing, independent eviction, cross-medium load lookup, same-key-in-both-media, threshold-disabled back-compat.
- [ ] **[Tests] Integration + manual smoke test** — mixed-priority workload, per-medium hit counters, output bit-exactness, cross-deployment recovery end-to-end.
- [ ] **[Release] Lint, pre-commit, open three draft PRs** — per `AGENTS.md` workflow.
