# Two-tier CPU KV offload (hillock-vmem) — M1 plan

| | |
|---|---|
| **Status** | Draft |
| **Project** | hillock-vmem |
| **Branch** | `dev` |
| **Owner** | @wangchen615 |
| **Created** | 2026-05-04 |

## Context

The `hillock-vmem` project wants vLLM to offload GPU KV cache to **two CPU memory pools with different speeds** — a fast pool and a slow pool — and to use them as an exclusive, tiered hierarchy (GPU → fast → slow). Today vLLM ships three offloading connectors and none of them do this: `OffloadingConnector` has one CPU pool with pluggable LRU/ARC, `SimpleCPUOffloadConnector` has one CPU pool backed by a `BlockPool`, and `MultiConnector` only broadcasts saves to every child (no demotion, no exclusivity).

We chose to fork `SimpleCPUOffloadConnector` (author: Yifan Qiao, `vllm/v1/simple_kv_offload/`, ~1350 LOC). It has a clean dual-coordinator pattern (GPU + CPU `KVCacheCoordinator`) that extends naturally to a third coordinator, and its `DmaCopyBackend` has the pinned-memory / low-priority-stream plumbing we need.

**M1 goal**: end-to-end GPU ↔ fast CPU ↔ slow CPU tiering working in **eager mode, exclusive placement only** (see placement modes below), with both pools physically backed by pinned host memory (same mechanism; latency asymmetry simulated in M2 or via NUMA placement later). Lazy mode, HMA hardening, resiliency (hybrid / replicate modes), and non-pinned/CXL backings are out of scope for M1 — a follow-up RFC will own those.

The design we're committing to:
- **Cascade**: Stores land in fast. When fast is full, the LRU victim is **demoted** to slow (CPU→CPU copy) to free a fast slot. When both are full, we drop the store silently (same as today's single-pool behavior at capacity).
- **Load**: Check fast first; fall back to slow. On slow hit, **promote** to fast (copying via GPU is unnecessary — a CPU→CPU copy during/after the main load suffices; M1 keeps it simple: serve from slow directly, no promotion).
- **Config**: Two explicit knobs in `kv_connector_extra_config`: `fast_cpu_bytes` and `slow_cpu_bytes`. Legacy `cpu_bytes_to_use` keeps working and maps to `fast_cpu_bytes` (slow defaults to 0 → single-pool behavior, fully backward compatible).

## Why two pools (the full motivation)

Two CPU memory pools — one faster, one larger — gives us three distinct wins, in roughly decreasing strength-of-evidence:

1. **Capacity**: total offloadable KV grows to `fast + slow`. For long-context serving workloads where prefix reuse dominates, more cached prefix directly cuts re-prefill cost.
2. **Prefix-miss latency**: a fast tier (pinned host RAM near the accelerator, ideally NVLink / CXL-close) serves prefix-miss reloads much faster than a slow tier (NUMA-far, CXL-far, or capacity-optimized memory), and the scheduler can route hot prefixes there. This is the primary win for TTFT on cache hits.
3. **Resiliency**: with both pools active and carrying (some) copies of the same blocks, the system can survive **one pool becoming unresponsive mid-serve** — the surviving pool still has enough to continue serving without forcing a cold re-prefill for every active request.

These three goals sit on a spectrum defined by **how blocks are placed across the two pools**. M1 picks one point on that spectrum; the RFC declares the whole space so the code lands in a shape that extends cleanly.

### Placement modes (future design space)

| Mode | Semantics | Effective capacity | Resiliency | In M1? |
|---|---|---|---|---|
| **exclusive** | Block lives in fast OR slow, never both. Fast-to-slow demotion on fast eviction. | `fast + slow` | **None** — losing either pool loses whatever was only there | **Yes** (only mode) |
| **hybrid** | New stores go to fast; a bounded async mirror also writes to slow. Under capacity pressure the mirror becomes the demotion (exclusive) path. | Between `min(fast,slow)` and `fast + slow` depending on pressure | Partial — recently-stored hot blocks are replicated, older demoted ones are not | No (follow-up RFC) |
| **replicate** | Every store goes to both pools. Loads prefer fast; slow is used on fast-miss or fast-failure. | `min(fast, slow)` | **Full** — either pool alone is sufficient to continue serving | No (follow-up RFC) |

These are **selectable at connector init** — future `kv_connector_extra_config.placement_mode`. Users pick where on the spectrum they sit based on workload (resiliency-critical serving vs. max-capacity batch). Defaulting to `exclusive` preserves M1 behavior.

### Failure model for resiliency (follow-up)

The resiliency modes target **hot failure**: one pool's backing store becomes unresponsive during live serving (e.g., a CXL link flaps, a remote NUMA node wedges, a pool's ioctl hangs). In that regime a loss is not a restart signal — the process is still up, other requests are still flowing, and we need to:

1. **Detect** the hang in-band (read timeout, error-return from the copy path).
2. **Fail over** the in-flight transfer to the surviving pool without aborting the request.
3. **Quarantine** the failed pool so we don't keep re-issuing against it.
4. **Optionally re-replicate** surviving-only blocks to a replacement pool when one comes back.

This is materially more complex than cold-restart recovery (which would only need "on startup, reload from whichever pool is alive"). The follow-up RFC will own detection thresholds, timeout budgets, and whether quarantine is manual or automatic.

## Key constraints discovered during exploration

1. **Block-hash collisions**: `BlockPool.cached_block_hash_to_block` is instance-scoped (`vllm/v1/core/block_pool.py:171`) and keys embed group_id (`vllm/v1/core/kv_cache_utils.py:53-72`) — but **not** a tier identifier. If the same hash lands in both pools' maps, the lookup semantics at the scheduler (`_prepare_eager_store_specs` at `vllm/v1/simple_kv_offload/manager.py:528-534`) become ambiguous. Our cascade enforces **exclusive placement** (a block is in fast OR slow, never both), which sidesteps this cleanly.
2. **`DmaCopyBackend` builds copy params once at init** (`vllm/v1/simple_kv_offload/copy_backend.py:46-47`). It's tied to a single (gpu_caches, cpu_caches) pair. We'll run **one backend per pool** (two `DmaCopyBackend` instances) rather than rewriting `build_params` — smaller blast radius and clearer ownership.
3. **KVCacheCoordinator is fully per-instance** (`vllm/v1/core/kv_cache_coordinator.py:28-78`) — safe to instantiate two.
4. **No cross-pool lookups anywhere in the existing API surface** — `find_longest_cache_hit` takes the block_pool explicitly.

## Design

### New abstraction: `CpuTier`

A thin container bundling everything that's currently singular in `SimpleCPUOffloadScheduler`/`Worker`:
- scheduler side: the `KVCacheCoordinator` + its `BlockPool` + an LRU ordering (we need explicit LRU for demotion — today's code relies on `BlockPool`'s free-list recency, which is not directly queryable for "pick LRU victim for demotion").
- worker side: the pinned CPU tensor dict + its `DmaCopyBackend` instance.

Scheduler holds `self._fast: CpuTier` and `self._slow: CpuTier | None`. Worker mirrors with two tier objects.

### Scheduler-side changes (`vllm/v1/simple_kv_offload/manager.py`)

**Initialization**: `SimpleCPUOffloadScheduler.__init__` builds two coordinators via the existing `_derive_cpu_config` helper, once per tier. If `slow_cpu_bytes == 0`, skip the slow tier and fall back to today's single-pool path (trivial branch — keeps backward-compat clean).

**`get_num_new_matched_tokens` (`manager.py:211-231`)**: check fast first, then slow; return the longer hit. Returning the union of matches requires care — stick to "longest contiguous from one tier" for M1. If fast hit ≥ slow hit, use fast; else use slow.

**`update_state_after_alloc` (`manager.py:235-316`)**: extend `cpu_hit_blocks` discovery across both tiers. Build one `LoadRequestState` with a `TransferMeta` that also carries the *source tier* for each block (new field: `cpu_tiers: list[int]` parallel to `cpu_block_ids`, values 0=fast, 1=slow). Touch blocks in the correct pool.

**`_prepare_eager_store_specs` (`manager.py:446-582`)**:
- Primary store target is fast. Same logic as today.
- Skip blocks already present in **either** fast or slow (check both `cached_block_hash_to_block` maps) to preserve the exclusive-placement invariant.
- **Demotion is allocator-driven, not capacity-driven.** The current code's `num_free <= 0` check is *not* the right demotion trigger: `BlockPool.get_num_free_blocks()` returns `free_block_queue.num_free_blocks`, which already includes cached blocks with `ref_cnt == 0` (they stay on the free queue until popped). When `num_free == 0`, every block in the pool is actively pinned (`ref_cnt > 0`) and nothing is demotable. Demoting a pinned block would corrupt an in-flight transfer.
- Correct trigger: **intercept the cached-block eviction inside the fast tier's allocator.** `BlockPool.get_new_blocks()` (block_pool.py:322) pops from `free_block_queue` and calls `_maybe_evict_cached_block()` (block_pool.py:354) whenever the popped block carries a `block_hash`. That call-site is exactly "I'm about to drop cached data on the floor" — we redirect it to "copy this cached data to slow, then drop from fast." The free queue is already LRU-ordered for cached blocks, so **no side LRU structure is needed**; `BlockPool` already picks the victim we want.

**Fast-tier demotion hook**: rather than modify upstream `BlockPool`, we subclass it (`FastTierBlockPool` in `vllm/v1/simple_kv_offload/tier.py`) and override `_maybe_evict_cached_block`. On eviction of a cached block we:
1. Look up the block's hash + payload,
2. Allocate a slow-tier block for it (falling back to the base eviction behavior — i.e. truly drop — when slow has no capacity either),
3. Enqueue a fast→slow CPU-to-CPU copy as a demote event in the next `build_connector_meta`,
4. Let the original eviction complete (block becomes free and returns to the caller).

The net effect: from the allocator's perspective `get_new_blocks` still returns as many blocks as requested; from the cache's perspective, hashes migrate fast→slow instead of disappearing. Cap queued demotions per step by the number of free slow blocks to keep `build_connector_meta` bounded.

**Metadata (`vllm/v1/simple_kv_offload/metadata.py`)**: extend `SimpleCPUOffloadMetadata`:
- `load_cpu_tiers: list[int]` parallel to `load_cpu_blocks`
- `store_cpu_tier: int` — fast (0) or slow (1) destination for each store event (stores are homogeneous per event for M1)
- New event lists for CPU→CPU demotions: `demote_event: int`, `demote_src_blocks: list[int]`, `demote_dst_blocks: list[int]`

`build_connector_meta` emits up to 3 events per step (load, store-to-fast, demote-fast-to-slow). Keep existing per-event counter semantics.

**Completion handling (`_process_store_event`, `_process_store_completion`)**: route completions to the correct tier. Add `_process_demote_event` that inserts the block's hash into the slow tier's `cached_block_hash_to_block` map (the fast-side entry was already removed synchronously by `FastTierBlockPool._maybe_evict_cached_block` before the demote was enqueued, preserving the exclusive-placement invariant even while the host-to-host copy is in flight — readers that hit the hash post-eviction will find it in slow).

### Worker-side changes (`vllm/v1/simple_kv_offload/worker.py`)

**`register_kv_caches` (`worker.py:66-182`)**: allocate pinned CPU tensors for **both** tiers, size each from its own `num_cpu_blocks`. Two `DmaCopyBackend` instances: `_fast_backend`, `_slow_backend`. Reuse the same `load_stream` / `store_stream` pair across backends — CUDA can serialize ops on one stream without issue, and we get predictable ordering.

**Demotion backend**: a **third** `DmaCopyBackend` or a direct CPU memcpy path. For M1 simplicity, add a tiny `_demote_backend` that builds `BatchMemcpyParams(src=fast_cpu_caches, dst=slow_cpu_caches, stream=store_stream)`. `cuMemcpyBatchAsync` works for host→host with pinned memory. No new kernel needed.

**`get_finished`**: launch all three event types (load, store, demote). Poll the same event lists, now split 3-ways. Report completed demote events via a new field on `SimpleCPUOffloadWorkerMetadata`.

### Critical files to modify

| File | Change |
|---|---|
| `vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py` | Parse `fast_cpu_bytes` / `slow_cpu_bytes` from `extra_config`; backward-compat alias `cpu_bytes_to_use` → `fast_cpu_bytes`. Log both capacities. |
| `vllm/v1/simple_kv_offload/tier.py` **(new)** | `CpuTier` dataclass; `FastTierBlockPool(BlockPool)` subclass overriding `_maybe_evict_cached_block` to redirect cached-block evictions into fast→slow demote events. |
| `vllm/v1/simple_kv_offload/manager.py` | Dual-coordinator construction (fast uses `FastTierBlockPool`); cross-tier load path; demote-event plumbing; `_process_demote_event`. |
| `vllm/v1/simple_kv_offload/worker.py` | Dual pinned-tensor allocation; two `DmaCopyBackend` instances + demote backend; launch + poll three event streams. |
| `vllm/v1/simple_kv_offload/metadata.py` | Extend metadata dataclasses with tier fields + demote event. |
| `vllm/v1/simple_kv_offload/copy_backend.py` | No change expected — verify host→host with pinned-memory works via existing `cuMemcpyBatchAsync`; if not, add a thin `MemcpyKind.HostToHost` path. |

**Not modified for M1**: `OffloadingConnector`, `MultiConnector`, `KVCacheCoordinator`, upstream `BlockPool` (we only *subclass* it — no edits to `vllm/v1/core/block_pool.py`).

## Out of scope for M1

- **Placement modes other than `exclusive`** — `hybrid` and `replicate` are deferred to the follow-up RFC below. The config surface is shaped so adding a `placement_mode` knob later does not break existing configs.
- **Resiliency (hot-failure detection & failover)** — single-pool failure means the blocks in that pool are lost; requests mid-serve against them will miss and re-prefill. Follow-up RFC.
- Lazy mode tiering (keep lazy path single-pool or disabled when `slow_cpu_bytes>0`)
- Promotion on slow-tier load hit
- HMA multi-group interaction (slow pool shares groups with fast → should Just Work but untested)
- Real slow-medium backings (NUMA-remote, unpinned, CXL, NVMe)
- Metrics / observability — add only a log line per tier

## Follow-up RFC: resiliency + placement modes

This is a sketch, not a commitment — the next RFC owns the details. Capturing it here so the M1 code lands in a shape the follow-up can build on without re-plumbing.

**Scope of the follow-up RFC:**
- Add `placement_mode: "exclusive" | "hybrid" | "replicate"` to `kv_connector_extra_config`. Default `"exclusive"` preserves M1 behavior.
- **Replicate mode**: store writes both tiers; loads read fast, fall back to slow on miss *or* failure. Effective capacity `min(fast, slow)`.
- **Hybrid mode**: store writes fast eagerly, mirror-writes slow under a bounded queue. When fast is full, the mirror becomes the demotion (i.e., hybrid degrades to exclusive under pressure). Effective capacity between `min(fast, slow)` and `fast + slow`.
- **Hot-failure detection**: read/write timeout on `DmaCopyBackend.launch_copy`; per-pool health state (`healthy` / `suspect` / `quarantined`). Configurable timeout budget; exponential back-off on suspect pools; manual-only re-enable in the first cut, auto-heal as a follow-follow-up.
- **Failover semantics**: on a read timeout in fast, retry on slow; on a write failure in slow under `replicate`, drop the mirror and degrade to exclusive placement for that block (log once).
- **Re-replication**: when a quarantined pool recovers, background re-replicate slow→fast (or vice versa) to restore redundancy. Bounded to avoid stepping on live serving.

**What M1 must *not* foreclose** (design constraints passed from this RFC to the follow-up):
- `CpuTier` must be a *symmetric* abstraction — nothing about "fast" vs "slow" should leak into the code paths that placement mode will later flip. The only asymmetry today (demotion direction) lives inside `FastTierBlockPool` and is easy to replace per mode.
- Metadata carries a **per-block source-tier hint** already (`load_cpu_tiers: list[int]` in the RFC above). That same field becomes the failover target when the hint pool is quarantined.
- `DmaCopyBackend.launch_copy` has no timeout today. Leaving the signature unchanged is fine for M1 but the follow-up will need to add one; we should mention this in the M1 worker change so reviewers aren't surprised later.

## Verification

**Unit tests** (new file `tests/v1/simple_kv_offload/test_two_tier_manager.py`):
1. Build `SimpleCPUOffloadScheduler` with `fast_cpu_bytes=N*block_size`, `slow_cpu_bytes=M*block_size`. Assert two coordinators created, two BlockPools with expected `num_blocks`.
2. Store 2N blocks → fast fills; next N blocks trigger N demotions to slow. Assert: slow `cached_block_hash_to_block` has N entries, fast has N entries, hashes are disjoint (exclusivity).
3. Load-path hit precedence: prime fast with hash A, slow with hash B, issue request whose prefix matches B. Assert `TransferMeta.cpu_tiers == [1, ...]` for the slow hits.
4. Backward compat: config with only legacy `cpu_bytes_to_use` set → `slow` tier is None, behavior identical to current.

**Integration test** (extend `tests/v1/simple_kv_offload/test_end_to_end.py` or equivalent; confirm file exists first):
1. Two-pool run with a small model (e.g. `facebook/opt-125m`) and artificially small `fast_cpu_bytes` so demotion fires. Assert request outputs are bit-identical to the single-pool baseline.

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
        },
        'kv_role': 'kv_both',
    }),
)
out = llm.generate(['Hello world ' * 100] * 4, SamplingParams(max_tokens=32))
for o in out: print(o.outputs[0].text)
"
```
Expected: logs show `SimpleCPUOffloadConnector: fast=... slow=...` and `demote event N completed` lines during generation.

## Open questions deferred to during-implementation

- Whether `cuMemcpyBatchAsync` accepts `HostToHost` with pinned memory today — if not, add a small path in `cuda_mem_ops.py`. Will verify in code, not blocking plan.
- Whether `FastTierBlockPool._maybe_evict_cached_block` can safely allocate a slow-tier block synchronously from inside the fast allocator's critical path, or whether we should record "evicted hash + payload pointer" and do the slow-pool allocation one layer up in the manager. First design is simpler; the second is safer if either pool ever holds a lock.

## Branch & PR plan

- Work on branch `dev` (already created off synced upstream `main`).
- Single focused PR into your fork's `dev` once M1 lands green. Not upstream yet — upstream will want the broader design discussion first.
- Label issues/PRs with `project:hillock-vmem`.

## Suggested issue breakdown (for project board)

These map 1:1 to the task list tracked during planning. Each can become a GitHub issue under `project:hillock-vmem`. Checkboxes indicate suggested dependency order (top-down).

- [ ] **[Prep] Verify dev environment + test paths** — confirm `.venv` setup, find existing `tests/v1/simple_kv_offload/`, verify `cuMemcpyBatchAsync` host→host in `cuda_mem_ops.py`.
- [ ] **[Metadata] Extend `SimpleCPUOffloadMetadata` with tier fields + demote event** — new dataclass fields; extend `aggregate()` on worker metadata.
- [ ] **[Scheduler] Introduce `CpuTier` + dual coordinators** — refactor `SimpleCPUOffloadScheduler` to hold `_fast` and optional `_slow`; single-tier fallback when `slow_cpu_bytes=0`.
- [ ] **[Scheduler] Cross-tier load path** — `get_num_new_matched_tokens` + `update_state_after_alloc` check both tiers; `TransferMeta.cpu_tiers` tracks source.
- [ ] **[Scheduler] Demotion via fast-tier allocator hook** — add `FastTierBlockPool` subclass overriding `_maybe_evict_cached_block`; redirect cached-block evictions into fast→slow demote events; cap queued demotions per step by free slow blocks.
- [ ] **[Scheduler] Completion handling for 3 event types** — split `_process_store_event` into load/store/demote; hash-map migration on demote completion.
- [ ] **[Worker] Dual pinned tensors + two `DmaCopyBackend` instances** — allocate per-tier CPU tensors; add `_demote_backend` for host→host.
- [ ] **[Worker] Launch + poll 3 event streams** — dispatch to correct backend; report completed demote events.
- [ ] **[Connector] Parse `fast_cpu_bytes` / `slow_cpu_bytes` config** — backward-compat alias; thread capacities to scheduler/worker.
- [ ] **[Tests] Unit tests for two-tier manager** — dual coordinators, demotion, load precedence, backward compat.
- [ ] **[Tests] Integration + manual smoke test** — two-pool end-to-end vs single-pool baseline.
- [ ] **[Release] Lint, pre-commit, open draft PR** — per `AGENTS.md` workflow.
