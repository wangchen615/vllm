# Two-tier CPU KV offload — M1 plan

| | |
| --- | --- |
| **Status** | Draft |
| **Branch** | `dev` |
| **Owner** | @wangchen615 |
| **Created** | 2026-05-04 |
| **Component design** | [exclusive-tiered-caching.md](exclusive-tiered-caching.md) (component-level view of this M1 mode) |
| **Parent** | [secondary-memory-system-overview.md](secondary-memory-system-overview.md) |

## Context

> **Terminology**: This document uses neutral hardware-agnostic names — **secondary fast memory** for the novel middle tier and **accelerator** for any device that owns HBM. See [secondary-memory-system-overview.md §Terminology](secondary-memory-system-overview.md#terminology).

### The production target: a three-level memory hierarchy

This RFC targets a **three-level memory hierarchy** for LLM KV cache:

```text
  GPU HBM  ↔  secondary fast memory system  ↔  slow DRAM on host
  (smallest,      (the novel tier —                 (largest,
   fastest,        larger than HBM, faster           slowest,
   closest)        than host DRAM to reach           most distant)
                   from the accelerator)
```

The middle tier — the **secondary fast memory system** — is the novel piece. It's **larger than HBM, smaller than host DRAM, and faster than host DRAM to reach from the accelerator** (think CXL-attached memory, near-accelerator vmem, NVLink-reachable DDR). It gives the system a place to keep hot KV blocks that don't fit in HBM but would be expensive to reload from host DRAM.

### Hierarchy semantics: speed *and* placement policy

The hierarchy is not defined by speed alone. A second axis — the **placement policy** — decides which tier holds which blocks, and whether the same block can live in more than one tier. Speed alone gets you a fast lookup path; placement policy decides what each tier is *for*. The two axes are independent and equally first-class.

vLLM's existing offload connectors all implement one specific placement policy: **inclusive cascade** (write to fast, demote to slow on eviction, slow always backs up fast's overflow). This RFC deliberately does *not* do this for M1. The contrast is the project's main novelty:

| Mode | Semantics | Block can live in… | Inter-tier movement | In M1? |
| --- | --- | --- | --- | --- |
| **partitioned** | Each request is admitted to a tier based on a per-request signal (e.g. priority). The two tiers cache **different sets** of blocks. Each *physical* block lives in exactly one tier. | Exactly one tier | None — never copied between tiers | **Yes** (M1 mode) |
| **hybrid** | New stores go to fast; a bounded async mirror also writes to slow. | One or both | Async mirror on store | No (resiliency RFC) |
| **replicate** | Every store goes to both tiers. Loads prefer fast; slow is used on fast-miss or fast-failure. | Both | Sync on store | No (resiliency RFC) |

> **What we are *not* doing**: the inclusive-cascade model that `OffloadingConnector` and `SimpleCPUOffloadConnector` already implement (write to fast, demote LRU victims to slow on fast eviction). Cascade is well-served by the existing connectors and adds no novelty here. M1 explicitly avoids it.

**M1 implements `partitioned` only.** Admission is keyed on `Request.priority` (vLLM already plumbs this through `LLMEngine.add_request` and `AsyncLLM.add_request` — no API change needed). A configurable threshold `priority_threshold` decides which tier a request's blocks land in. Each tier evicts independently using its own `BlockPool`'s LRU; **no fast→slow demotion path exists**.

The companion [resiliency RFC](secondary-memory-resiliency.md) owns the design for `hybrid` and `replicate`.

### Use cases driving the design

These motivate why two *partitioned* tiers (and eventually `hybrid` / `replicate`) are worth the complexity. M1 only needs to enable the first; the others justify why M1's abstractions need to stay symmetric and not foreclose future modes.

1. **Class-of-service KV cache for mixed-priority serving** — *driven by `partitioned` mode (M1)*. A serving system handles both latency-sensitive interactive requests and best-effort batch / agentic-background traffic on the same engine. With one undifferentiated CPU pool, low-priority KV evicts (or prevents caching of) high-priority KV. Partitioning by priority gives high-priority requests a smaller, tightly-managed fast tier insulated from low-priority churn, while low-priority requests still benefit from a larger slow-tier cache. **The two tiers cache different sets of KV blocks on purpose** — they're not a fast→slow cascade for the same data.
2. **Resilient serving across memory-tier failures** — *driven by `hybrid` and `replicate` modes (resiliency RFC)*. A real three-level hierarchy is naturally redundant; placement policy decides whether we exploit that to survive a tier going unresponsive mid-serve.
3. **Model sharing for agentic workflows with rapid model switches** — *driven by `replicate` mode, with a read-only / externally-managed variant*. Multiple accelerators each load the same model from a shared secondary-fast-memory-backed pool. The model stays canonical in the shared pool; each executor brings it in to run, but the shared pool retains the master copy. This is `replicate` semantically, with two extensions: (a) the slow tier is **read-only** from the connector's perspective — only the platform writes, executors only read; (b) the slow tier is **externally managed** — the connector doesn't control evictions, the platform does. Heavy model overcommit and rapid model switches in agentic workflows make this a load-bearing use case rather than a curiosity. Out of M1 scope, mentioned here so M1's abstractions don't silently exclude it (e.g., the `CpuTier` abstraction must not assume the connector owns the tier's lifecycle).

### Why this RFC uses two CPU pools

We don't yet have the secondary fast memory system hardware available for vLLM testing. **M1 emulates the three-level hierarchy by using two separate CPU memory pools** as stand-ins for the secondary fast memory system and the slow DRAM on host. Both pools are just pinned host DRAM today, so there's no real latency asymmetry in M1 — the point of M1 is to prove that **vLLM's Simple KV-offload connector can be extended with small, well-scoped changes to manage two distinct address spaces** (allocation, eviction, exclusive placement, metadata, worker-side transfers, completion plumbing). Once that functional foundation is in place, swapping pool #0 to a real secondary fast memory backing (NUMA-local pinned, CXL, vmem, etc.) is a contained change in the worker — the scheduler logic doesn't have to move.

We considered vLLM's three existing offloading connectors before deciding to fork `SimpleCPUOffloadConnector`: `OffloadingConnector` has one CPU pool with pluggable LRU/ARC, `SimpleCPUOffloadConnector` has one CPU pool backed by a `BlockPool`, and `MultiConnector` only broadcasts saves to every child (no demotion, no exclusivity). `SimpleCPUOffloadConnector` (author: Yifan Qiao, `vllm/v1/simple_kv_offload/`, ~1350 LOC) has a clean dual-coordinator pattern (GPU + CPU `KVCacheCoordinator`) that extends naturally to a third coordinator, and its `DmaCopyBackend` has the pinned-memory / low-priority-stream plumbing we need.

### M1 goal

Demonstrate that `SimpleCPUOffloadConnector` can manage **two CPU pools as two distinct address spaces** under a `partitioned` placement policy keyed on `Request.priority`. Pool #0 is the emulated **secondary fast memory system** (code: "fast"), pool #1 is the emulated **slow DRAM on host** (code: "slow"). Both are pinned DRAM; asymmetric performance is deliberately out of scope — M1 measures *functional correctness* (blocks routed to the correct tier by priority, each tier evicts independently, no inter-tier copies, metadata + completion wiring works end-to-end), not throughput.

What's explicitly not being claimed for M1: capacity additivity (the tiers cache different sets of blocks, so capacity isn't simply additive across requests), prefix-miss speedup, resiliency, or model sharing. Those follow from a real secondary fast memory backing and/or from `hybrid` / `replicate` placement modes — covered in the [resiliency RFC](secondary-memory-resiliency.md).

### Design committed for M1

- **Admission (the only routing decision)**: when a block becomes a store candidate, the scheduler reads `request.priority` (already an `int` field on `Request`, plumbed from `LLMEngine.add_request` and `AsyncLLM.add_request` without any upstream API change). If `priority < priority_threshold` the block is routed to the fast tier; otherwise to the slow tier. The decision happens once per block at admission time and **never changes** for the lifetime of that block. (vLLM convention: lower-int = higher-priority. So with `priority_threshold = 1`, requests submitted with `priority=0` — i.e. the default-priority interactive traffic — land in fast, while requests submitted with `priority>=1` — explicitly de-prioritized batch / agentic traffic — land in slow.)
- **Eviction**: each tier is a vanilla `BlockPool` with its own LRU. When a tier is full and the next admission needs a slot, that tier evicts its own LRU victim — the block is *dropped*, not copied to the other tier. Both tiers operate independently.
- **No demotion, no promotion, no inter-tier copies.** A block lives in exactly the tier its request was admitted to, until it's evicted from that tier or the request finishes.
- **Load (cache-hit path)**: the prefix-cache lookup checks both tiers and serves from whichever holds the hit. If both have hits for different prefix ranges, we use the longer one. There's no implicit cross-tier copy on hit (no promotion).
- **Config**: three explicit knobs in `kv_connector_extra_config`:
    - `fast_cpu_bytes` — capacity of the fast (emulated secondary) tier. Default `0` (disabled → behave like single-pool slow-only).
    - `slow_cpu_bytes` — capacity of the slow (emulated host DRAM) tier.
    - `priority_threshold` — int. With `fast_cpu_bytes > 0`, defaults to `1` (default-priority requests → fast; explicitly de-prioritized requests → slow). Operators can raise it to admit more priority classes into fast or lower it (to `0` or below) to disable fast even when `fast_cpu_bytes > 0`.
    - Legacy `cpu_bytes_to_use` keeps working: it maps to `slow_cpu_bytes` with `fast_cpu_bytes=0`, which routes everything to slow. Existing deployments see no behavior change.

## Key constraints discovered during exploration

1. **Block-hash collisions across tiers**: `BlockPool.cached_block_hash_to_block` is instance-scoped (`vllm/v1/core/block_pool.py:171`) and keys embed group_id (`vllm/v1/core/kv_cache_utils.py:53-72`) — but **not** a tier identifier. With partitioned admission, the same prompt hash from two different requests at different priorities could in principle land in *both* tiers' maps. M1 accepts this: both maps may legitimately contain the same hash (different physical block, same logical content), and the load path resolves it by checking both and taking the longer hit. Each block is still in **exactly one tier** — the cross-tier-hash case happens because two requests both computed it, not because we duplicated.
2. **`DmaCopyBackend` builds copy params once at init** (`vllm/v1/simple_kv_offload/copy_backend.py:46-47`). It's tied to a single (gpu_caches, cpu_caches) pair. We'll run **one backend per tier** (two `DmaCopyBackend` instances). Each handles only GPU↔its-own-CPU-tier; **no host↔host backend is needed** in M1 since there are no inter-tier copies.
3. **`KVCacheCoordinator` is fully per-instance** (`vllm/v1/core/kv_cache_coordinator.py:28-78`) — safe to instantiate two.
4. **No cross-pool lookups in the existing API surface** — `find_longest_cache_hit` takes the block_pool explicitly, so iterating fast then slow is the natural extension.
5. **`Request.priority` is plumbed end-to-end already** (`vllm/v1/request.py:73`, `vllm/v1/engine/llm_engine.py:218`, `vllm/v1/engine/async_llm.py:292`). The connector receives the `Request` object in `update_state_after_alloc` and can read `request.priority` directly. No new API surface, no upstream changes.

## Design

### New abstraction: `CpuTier`

A thin container bundling everything that's currently singular in `SimpleCPUOffloadScheduler`/`Worker`:

- scheduler side: the `KVCacheCoordinator` + its `BlockPool` (vanilla; no subclass).
- worker side: the pinned CPU tensor dict + its `DmaCopyBackend` instance.

Scheduler holds `self._fast: CpuTier | None` and `self._slow: CpuTier | None`. Worker mirrors with two tier objects. When `fast_cpu_bytes == 0`, `_fast` is `None` and the connector behaves like today's single-pool slow-only configuration.

`CpuTier` is intentionally **symmetric** — nothing about "fast" vs "slow" leaks into its definition. Asymmetry lives at the scheduler's admission decision (which tier to route a given request's blocks to), not in the tier abstraction itself. This keeps the door open for future placement modes (`hybrid`, `replicate`) to reuse the same `CpuTier` without refactoring.

### Scheduler-side changes (`vllm/v1/simple_kv_offload/manager.py`)

**Initialization**: `SimpleCPUOffloadScheduler.__init__` builds 0, 1, or 2 coordinators based on capacity config. Each coordinator uses the existing `_derive_cpu_config` helper. The result is one `BlockPool` per active tier — no subclasses, no overrides.

**Admission helper** (new method): `_choose_tier(request: Request) -> CpuTier` — returns `_fast` if `request.priority < priority_threshold` and `_fast is not None`, else `_slow`. This is the only place that consults priority; everywhere else operates on the chosen tier opaquely.

**`get_num_new_matched_tokens` (`manager.py:211-231`)**: check both tiers' coordinators for prefix-cache hits and return the longer one. (If only one tier exists, behavior is unchanged from today.) Return the tier identity alongside the length so `update_state_after_alloc` knows which pool to touch.

**`update_state_after_alloc` (`manager.py:235-316`)**:

- For the **store path** (new admission): call `_choose_tier(request)` once and route all of this request's eligible-to-store blocks to that tier. The tier choice is recorded in `LoadRequestState` / `StoreRequestState`.
- For the **load path** (cache-hit serving): blocks may be hit in either tier. Build the `TransferMeta` with a per-block `cpu_tier: int` field (0 = fast, 1 = slow) so the worker copies from the right pool.
- Touch blocks in the correct tier's `BlockPool` to prevent eviction during the in-flight transfer.

**Store path (`_prepare_eager_store_specs`)**:

- For each request, the tier was chosen at admission. Iterate the request's new blocks and store them to that tier's `BlockPool` only.
- Each tier evicts its own LRU when full, using `BlockPool`'s native behavior — *no* `FastTierBlockPool` subclass, *no* override of `_maybe_evict_cached_block`. When eviction happens, the block is dropped (the existing `BlockPool` semantic). Other tiers are not consulted.
- Blocks already present in this tier's `cached_block_hash_to_block` are skipped (existing behavior). The other tier's map is **not** consulted for skip decisions — if the same hash exists in both tiers (because two requests at different priorities both computed it), each tier can keep its own copy. This does not violate "exclusive placement": each *physical block* still lives in exactly one tier; the duplication is logical (same hash) only.

**Metadata (`vllm/v1/simple_kv_offload/metadata.py`)**: extend `SimpleCPUOffloadMetadata`:

- `load_cpu_tiers: list[int]` parallel to `load_cpu_blocks`. The worker uses this to copy each load-source block from the right pool.
- `store_cpu_tier: int` — fast (0) or slow (1) destination for the store event. Stores in M1 are homogeneous per event (one event per tier per step); the scheduler emits up to 2 store events per step, one per tier.
- **No demote events.** The cascade design's `demote_event`/`demote_src_blocks`/`demote_dst_blocks` fields are dropped.

`build_connector_meta` emits up to **3 events per step**: load (single, can have blocks from either tier indexed by `load_cpu_tiers`), store-to-fast, store-to-slow.

**Completion handling**: existing single-tier completion logic applies to each tier independently. No cross-tier completion ordering, no `_process_demote_event`.

### Worker-side changes (`vllm/v1/simple_kv_offload/worker.py`)

**`register_kv_caches` (`worker.py:66-182`)**: allocate pinned CPU tensors for **both** tiers, size each from its own `num_cpu_blocks`. Two `DmaCopyBackend` instances: `_fast_backend`, `_slow_backend`. Each backend talks to its own tier's pinned-memory region; neither sees the other. Reuse the same `load_stream` / `store_stream` pair across backends — CUDA can serialize ops on one stream, and we get predictable ordering.

**No demotion backend.** No host→host copy path needed. (This was a known correctness risk in the previous cascade design; partitioned mode removes it entirely.)

**`get_finished`**: launch up to three event types per step (load + 2 stores). The load event may target either tier per-block (worker reads `load_cpu_tiers` from metadata to dispatch each block to the correct backend). Store events are homogeneous per tier, dispatched to whichever backend matches `store_cpu_tier`.

### Critical files to modify

| File | Change |
| --- | --- |
| `vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py` | Parse `fast_cpu_bytes`, `slow_cpu_bytes`, `priority_threshold` from `extra_config`. Backward-compat: legacy `cpu_bytes_to_use` → `slow_cpu_bytes` with `fast_cpu_bytes=0`. Log both capacities and the threshold. |
| `vllm/v1/simple_kv_offload/tier.py` **(new)** | `CpuTier` dataclass (symmetric — `KVCacheCoordinator`, `BlockPool`, capacity). No `BlockPool` subclass. |
| `vllm/v1/simple_kv_offload/manager.py` | Dual-coordinator construction; `_choose_tier(request)` admission helper; cross-tier load lookup; per-block tier hint in load metadata. |
| `vllm/v1/simple_kv_offload/worker.py` | Dual pinned-tensor allocation; two `DmaCopyBackend` instances; per-block tier dispatch on load. |
| `vllm/v1/simple_kv_offload/metadata.py` | Add `load_cpu_tiers: list[int]` and `store_cpu_tier: int` fields. **Drop** demote-event fields. |

**Not modified for M1**: `OffloadingConnector`, `MultiConnector`, `KVCacheCoordinator`, `BlockPool` (no subclass either — vanilla `BlockPool` per tier), `DmaCopyBackend` (no host→host extension needed).

## Out of scope for M1

- **Inclusive cascade (the existing-connector behavior)**: M1 does not implement fast→slow demotion on eviction. That's deliberate — cascade is what the existing connectors already do, and the project's novelty is precisely *not* doing that. If a deployment wants cascade, the existing `SimpleCPUOffloadConnector` already does it.
- **Asymmetric performance between the two pools** — M1 is a functional emulation; both pools are pinned DRAM. Real secondary fast memory backings (NUMA-local, CXL-attached, near-accelerator vmem, NVMe, etc.) are follow-up work.
- **Placement modes other than `partitioned`** — `hybrid` and `replicate` are described in the [resiliency RFC](secondary-memory-resiliency.md). M1's config surface is shaped so adding a `placement_mode` knob later does not break existing configs.
- **Resiliency (hot-failure detection & failover)** — see [resiliency RFC](secondary-memory-resiliency.md); M1 implements none of it.
- **Model sharing with read-only / externally-managed tiers** — the variant of `replicate` covering use case 3 above. Not in M1; M1's `CpuTier` abstraction does not foreclose it.
- **Sophisticated admission policies** — M1 only routes by `Request.priority` against a single threshold. Reuse-frequency-based, request-size-based, or learned admission policies are out of scope. The `_choose_tier` helper is a clear seam for swapping these in.
- **Promotion on slow-tier load hit** — partitioned design has no need for promotion (a request's blocks live where it was admitted).
- HMA multi-group interaction (slow pool shares groups with fast → should Just Work but untested).
- Metrics / observability — add only a log line per tier and per-tier hit counters.

## Relationship to the resiliency proposal

A companion RFC, [secondary-memory-resiliency.md](secondary-memory-resiliency.md), owns the design for `hybrid` and `replicate` modes plus the failure-detection / failover / re-replication mechanisms. It is **not** scheduled for M1.

What M1 must *not* foreclose so the resiliency proposal remains cheap to land later:

- **`CpuTier` is a symmetric abstraction.** Nothing about "fast" vs "slow" leaks into its definition. The only asymmetry — which tier a request is admitted to — lives in the `_choose_tier(request)` method on the manager, easy to replace per mode.
- **Worker metadata already carries per-block source-tier hints** (`load_cpu_tiers: list[int]`). That field generalizes to "valid tiers in preference order" without a schema change, which is what `replicate`'s failover needs.
- **`DmaCopyBackend.launch_copy` has no `timeout_ms` parameter today.** Leaving the signature unchanged in M1 is fine; the resiliency proposal adds it as an optional kwarg.
- **`CpuTier` does not assume the connector owns the tier's lifecycle.** M1 happens to allocate and free both pools internally, but the abstraction must allow a future tier where allocation, write, and eviction are owned by an external manager (the model-sharing / read-only variant of `replicate`). Concretely: a tier's "store" path must be conditional on a writability flag, and load-time lookup must tolerate a tier where blocks appear and disappear independently of connector actions.
- **The store path is a single function call (`_choose_tier` then per-tier store).** `hybrid` and `replicate` swap that function for "store to both" or "store to fast + async mirror to slow" without disturbing eviction or load.

If M1 review uncovers anything that would constrain the resiliency design, we update both RFCs together.

## Verification

**Unit tests** (new file `tests/v1/simple_kv_offload/test_two_tier_manager.py`):

1. **Construction**: build `SimpleCPUOffloadScheduler` with `fast_cpu_bytes=N*block_size`, `slow_cpu_bytes=M*block_size`, `priority_threshold=1`. Assert two coordinators created, two `BlockPool`s with expected `num_blocks`. Both pools are vanilla `BlockPool` (no subclasses).
2. **Admission routing**: submit one request with `priority=0` and one with `priority=5`. After `update_state_after_alloc`, assert the priority-0 request's blocks land in `_fast.block_pool` and the priority-5 request's blocks land in `_slow.block_pool`. Assert no blocks are duplicated across tiers.
3. **Independent eviction**: fill fast to capacity with priority-0 traffic. Submit another priority-0 request. Assert fast evicts an LRU victim **and slow is untouched** (no demotion).
4. **Cross-tier load lookup**: prime fast with prefix A's hash, slow with prefix B's hash. Issue a high-priority request matching prefix B (would normally route to fast for stores, but B's hit is in slow). Assert the load `TransferMeta` carries `cpu_tier=1` for B's blocks and the load is served from slow without a cross-tier copy.
5. **Same hash in both tiers (allowed)**: submit a priority-0 and a priority-5 request that compute the same prefix. Assert the same hash exists in both `cached_block_hash_to_block` maps (different physical blocks, same logical content). Subsequent loads pick the longer/equally-long hit; no duplication of physical blocks within a single tier.
6. **Threshold disabled**: with `fast_cpu_bytes=0`, all requests route to slow regardless of priority. Behavior matches single-pool today (back-compat for legacy `cpu_bytes_to_use`).

**Integration test** (extend `tests/v1/simple_kv_offload/test_end_to_end.py` if it exists, else create):

1. Run a workload mixing high-priority and low-priority requests through a small model (e.g. `facebook/opt-125m`). Monitor per-tier hit counters via the per-tier log lines. Assert: high-priority hits come predominantly from fast, low-priority hits from slow.
2. Output correctness: for a fixed seed and identical prompts, request outputs are bit-identical regardless of which tier served the prefix-cache hit.

**Manual smoke test** (per AGENTS.md workflow):

```bash
.venv/bin/python -m pytest tests/v1/simple_kv_offload/ -v
pre-commit run --all-files
```

Run on GPU:

```bash
VLLM_USE_V1=1 .venv/bin/python -c "
from vllm import LLM, SamplingParams
import json
llm = LLM(
    model='facebook/opt-125m',
    enable_prefix_caching=True,
    kv_transfer_config=json.dumps({
        'kv_connector': 'SimpleCPUOffloadConnector',
        'kv_connector_extra_config': {
            'fast_cpu_bytes': 64 * 1024 * 1024,
            'slow_cpu_bytes': 256 * 1024 * 1024,
            'priority_threshold': 1,
        },
        'kv_role': 'kv_both',
    }),
)
# High-priority request (priority=0) → fast tier
out_hi = llm.generate(['Hello high priority ' * 100], SamplingParams(max_tokens=32), priority=0)
# Low-priority request (priority=5) → slow tier
out_lo = llm.generate(['Hello low priority ' * 100], SamplingParams(max_tokens=32), priority=5)
print(out_hi[0].outputs[0].text)
print(out_lo[0].outputs[0].text)
"
```

Expected: logs show `SimpleCPUOffloadConnector: fast=64.00 MB slow=256.00 MB threshold=1`, and per-tier admission/hit counts confirming the routing happened as specified.

## Open questions deferred to during-implementation

- **Default `priority_threshold` semantics for two-pool deployments**: `1` is the proposal (default-priority requests → fast; explicit de-prioritization → slow). An alternative is `None` (operator must set explicitly when `fast_cpu_bytes > 0`) — safer against accidental misconfiguration but slightly less ergonomic. Decide during implementation based on how prominent we want the tier-routing knob to feel.
- **Multi-class admission**: `priority_threshold` admits a binary partition. If a deployment has 3+ priority classes that should route to different tiers, M1 collapses them to "above/below threshold." A future enhancement could make `_choose_tier` consult a list of thresholds or a callable. Out of M1 scope; mentioned to confirm the abstraction permits it.
- **Per-tier metrics and prefix-hit rate reporting**: the existing connector emits a single hit-rate counter. Two tiers want two counters. Plumbing TBD; trivial code, just needs a place in the metrics interface.

## Branch & PR plan

- Work on branch `dev` (already created off synced upstream `main`).
- Single focused PR into your fork's `dev` once M1 lands green. Not upstream yet — upstream will want the broader design discussion first.
- Label issues/PRs with `project:secondary-memory`.

## Suggested issue breakdown (for project board)

These map 1:1 to the task list tracked during planning. Each can become a GitHub issue under `project:secondary-memory`. Checkboxes indicate suggested dependency order (top-down).

- [ ] **[Prep] Verify dev environment + test paths** — confirm `.venv` setup, find existing `tests/v1/simple_kv_offload/`. Confirm `Request.priority` is reachable from the connector's scheduler-side hook.
- [ ] **[Connector] Parse `fast_cpu_bytes` / `slow_cpu_bytes` / `priority_threshold` config** — three new config fields; backward-compat for legacy `cpu_bytes_to_use` (→ `slow_cpu_bytes` with `fast_cpu_bytes=0`).
- [ ] **[Tier abstraction] Introduce `CpuTier`** — new `vllm/v1/simple_kv_offload/tier.py` with the `CpuTier` dataclass. Symmetric, no `BlockPool` subclass.
- [ ] **[Scheduler] Dual-coordinator construction** — refactor `SimpleCPUOffloadScheduler.__init__` to build 0/1/2 tiers based on capacities. Single-tier (slow only) fallback when `fast_cpu_bytes=0`.
- [ ] **[Scheduler] `_choose_tier(request)` admission helper** — read `request.priority`, return the destination tier. Single point of asymmetry.
- [ ] **[Scheduler] Per-tier store routing** — `_prepare_eager_store_specs` consults `_choose_tier` per request and stores to that tier's `BlockPool` only. Each tier evicts independently using vanilla `BlockPool` LRU.
- [ ] **[Scheduler] Cross-tier load lookup** — `get_num_new_matched_tokens` + `update_state_after_alloc` check both tiers' `cached_block_hash_to_block`; `TransferMeta` carries per-block `cpu_tier` so the worker copies from the right pool.
- [ ] **[Metadata] Add tier hints to `SimpleCPUOffloadMetadata`** — `load_cpu_tiers: list[int]`, `store_cpu_tier: int`. No demote-event fields.
- [ ] **[Worker] Dual pinned tensors + two `DmaCopyBackend` instances** — allocate per-tier CPU tensors; one backend per tier. No host↔host backend.
- [ ] **[Worker] Per-block tier dispatch** — load path reads `load_cpu_tiers` and routes each block to its tier's backend; store path uses `store_cpu_tier`.
- [ ] **[Tests] Unit tests for partitioned manager** — admission routing, independent eviction, cross-tier load lookup, same-hash-in-both-tiers, threshold-disabled back-compat.
- [ ] **[Tests] Integration + manual smoke test** — mixed-priority workload; per-tier hit counters; output bit-exactness.
- [ ] **[Release] Lint, pre-commit, open draft PR** — per `AGENTS.md` workflow.
