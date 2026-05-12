# Two-tier CPU KV offload (hillock-vmem) — M1 plan

| | |
|---|---|
| **Status** | Draft |
| **Project** | hillock-vmem |
| **Branch** | `dev` |
| **Owner** | @wangchen615 |
| **Created** | 2026-05-04 |

## Context

### The production target: a three-level memory hierarchy

The `hillock-vmem` project aims for a **three-level memory hierarchy** for LLM KV cache:

```
  GPU HBM  ↔  secondary fast memory system  ↔  slow DRAM on host
  (smallest,      (the novel tier —                 (largest,
   fastest,        larger than HBM, faster           slowest,
   closest)        than host DRAM to reach           most distant)
                   from the accelerator)
```

The middle tier — the **secondary fast memory system** — is the novel piece. It's **larger than HBM, smaller than host DRAM, and faster than host DRAM to reach from the accelerator** (think CXL-attached memory, near-accelerator vmem, NVLink-reachable DDR). When the GPU's KV cache is full, we'd rather spill hot blocks to the secondary fast memory system (cheap reload) than all the way to slow DRAM on the host (expensive reload). When the secondary fast memory system is full, we cascade to slow host DRAM.

### Why this RFC uses two CPU pools

We don't yet have the secondary fast memory system hardware available for vLLM testing. **M1 emulates the three-level hierarchy by using two separate CPU memory pools** as stand-ins for the secondary fast memory system and the slow DRAM on host. Both pools are just pinned host DRAM today, so there's no real latency asymmetry in M1 — the point of M1 is to prove that **vLLM's Simple KV-offload connector can be extended with small, well-scoped changes to manage two distinct address spaces** (allocation, eviction, exclusive placement, metadata, worker-side transfers, completion plumbing). Once that functional foundation is in place, swapping pool #0 to a real secondary fast memory backing (NUMA-local pinned, CXL, vmem, etc.) is a contained change in the worker — the scheduler logic doesn't have to move.

We considered vLLM's three existing offloading connectors before deciding to fork `SimpleCPUOffloadConnector`: `OffloadingConnector` has one CPU pool with pluggable LRU/ARC, `SimpleCPUOffloadConnector` has one CPU pool backed by a `BlockPool`, and `MultiConnector` only broadcasts saves to every child (no demotion, no exclusivity). `SimpleCPUOffloadConnector` (author: Yifan Qiao, `vllm/v1/simple_kv_offload/`, ~1350 LOC) has a clean dual-coordinator pattern (GPU + CPU `KVCacheCoordinator`) that extends naturally to a third coordinator, and its `DmaCopyBackend` has the pinned-memory / low-priority-stream plumbing we need.

### M1 goal

Demonstrate that `SimpleCPUOffloadConnector` can manage **two CPU pools as two distinct address spaces**, in **eager mode with exclusive placement only**. In the terminology above: pool #0 is the emulated **secondary fast memory system** (referred to in the code as "fast"), pool #1 is the emulated **slow DRAM on host** (referred to in the code as "slow"). Both are pinned DRAM; asymmetric performance is deliberately out of scope — M1 measures *functional correctness* (blocks placed and migrated correctly, exclusive placement preserved, metadata + completion wiring works end-to-end), not throughput.

What's explicitly not being claimed for M1: capacity additivity, prefix-miss speedup, or resiliency wins. Those follow from a real secondary fast memory backing (and, for resiliency, from a different placement mode) — see "Relationship to the resiliency proposal" below.

### Design committed for M1

- **Cascade**: Stores land in fast (emulated secondary fast memory system). When fast is full, the LRU victim is **demoted** to slow (emulated slow DRAM on host) via a CPU→CPU copy. When both are full, we drop the store silently (same as today's single-pool behavior at capacity).
- **Load**: Check fast first; fall back to slow. On slow hit, M1 serves from slow directly without promotion (promotion is a follow-up).
- **Config**: Two explicit knobs in `kv_connector_extra_config`: `fast_cpu_bytes` and `slow_cpu_bytes`. Legacy `cpu_bytes_to_use` keeps working and maps to `fast_cpu_bytes` (slow defaults to 0 → single-pool behavior, fully backward compatible).

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

- **Asymmetric performance between the two pools** — M1 is a functional emulation; both pools are pinned DRAM. Real secondary fast memory backings (NUMA-local, CXL-attached, near-accelerator vmem, NVMe, etc.) are follow-up work.
- **Placement modes other than `exclusive`** — `hybrid` and `replicate` motivate the resiliency proposal below. The config surface is shaped so adding a `placement_mode` knob later does not break existing configs.
- **Resiliency (hot-failure detection & failover)** — see proposal below; M1 does not implement any of it.
- Lazy mode tiering (keep lazy path single-pool or disabled when `slow_cpu_bytes>0`)
- Promotion on slow-tier load hit
- HMA multi-group interaction (slow pool shares groups with fast → should Just Work but untested)
- Metrics / observability — add only a log line per tier

## Relationship to the resiliency proposal

A companion RFC, [hillock-vmem-resiliency.md](hillock-vmem-resiliency.md), proposes how to exploit the hierarchy's natural redundancy to survive a memory-tier failure mid-serve. That proposal has its own motivation, placement modes (`exclusive` / `hybrid` / `replicate`), failure model (hot failure), and mechanism stack (detect / failover / quarantine / re-replicate). It is **not** scheduled for M1.

What M1 must *not* foreclose so the resiliency proposal remains cheap to land later:

- `CpuTier` is a **symmetric abstraction** — nothing about "fast" vs "slow" leaks into the code paths that `placement_mode` will later flip. The only asymmetry (demotion direction) lives inside `FastTierBlockPool` and is easy to replace per mode.
- Worker metadata already carries per-block source-tier hints (`load_cpu_tiers: list[int]` in the M1 design above). That field generalizes to "valid tiers in preference order" without a schema change.
- `DmaCopyBackend.launch_copy` has no `timeout_ms` parameter today. Leaving the signature unchanged in M1 is fine; the resiliency proposal adds it as an optional kwarg.

If M1 review uncovers anything that would constrain the resiliency design, we update both RFCs together.

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
