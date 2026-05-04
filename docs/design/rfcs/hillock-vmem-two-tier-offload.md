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

**M1 goal**: end-to-end GPU ↔ fast CPU ↔ slow CPU tiering working in **eager mode only**, with both pools physically backed by pinned host memory (same mechanism; latency asymmetry simulated in M2 or via NUMA placement later). Lazy mode, HMA hardening, and non-pinned/CXL backings are out of scope for M1.

The design we're committing to:
- **Cascade**: Stores land in fast. When fast is full, the LRU victim is **demoted** to slow (CPU→CPU copy) to free a fast slot. When both are full, we drop the store silently (same as today's single-pool behavior at capacity).
- **Load**: Check fast first; fall back to slow. On slow hit, **promote** to fast (copying via GPU is unnecessary — a CPU→CPU copy during/after the main load suffices; M1 keeps it simple: serve from slow directly, no promotion).
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
- **New**: when fast has no free blocks (`num_free <= 0` at line 538), instead of breaking out, attempt **demotion**: pick LRU victim from fast (see below), enqueue a fast→slow CPU-to-CPU copy, free the fast slot, continue. Cap demotions per step to keep scheduler pass bounded — `max_demotions_per_step` = `target_free` blocks (same watermark idea as lazy mode).
- Also skip blocks already present in **either** fast or slow (check both `cached_block_hash_to_block` maps).

**LRU tracking for demotion**: today's `BlockPool` doesn't expose an explicit LRU for *cached* (free-but-kept) blocks — the free queue intermixes them. Cheapest path: maintain a side `OrderedDict[block_hash, cpu_block_id]` in `CpuTier`, updated on every `_process_store_completion` (insert at MRU end) and on every cache hit during `update_state_after_alloc` (move to MRU end). To pick a demotion victim, pop from the LRU end and verify the block is still free (ref_cnt == 0); if not, skip and try next. Re-uses pattern from `vllm/v1/kv_offload/cpu/policies/lru.py` (Or Ozeri's code).

**Metadata (`vllm/v1/simple_kv_offload/metadata.py`)**: extend `SimpleCPUOffloadMetadata`:
- `load_cpu_tiers: list[int]` parallel to `load_cpu_blocks`
- `store_cpu_tier: int` — fast (0) or slow (1) destination for each store event (stores are homogeneous per event for M1)
- New event lists for CPU→CPU demotions: `demote_event: int`, `demote_src_blocks: list[int]`, `demote_dst_blocks: list[int]`

`build_connector_meta` emits up to 3 events per step (load, store-to-fast, demote-fast-to-slow). Keep existing per-event counter semantics.

**Completion handling (`_process_store_event`, `_process_store_completion`)**: route completions to the correct tier. Add `_process_demote_event` that inserts hash into slow tier's cache map + removes from fast + free refs.

### Worker-side changes (`vllm/v1/simple_kv_offload/worker.py`)

**`register_kv_caches` (`worker.py:66-182`)**: allocate pinned CPU tensors for **both** tiers, size each from its own `num_cpu_blocks`. Two `DmaCopyBackend` instances: `_fast_backend`, `_slow_backend`. Reuse the same `load_stream` / `store_stream` pair across backends — CUDA can serialize ops on one stream without issue, and we get predictable ordering.

**Demotion backend**: a **third** `DmaCopyBackend` or a direct CPU memcpy path. For M1 simplicity, add a tiny `_demote_backend` that builds `BatchMemcpyParams(src=fast_cpu_caches, dst=slow_cpu_caches, stream=store_stream)`. `cuMemcpyBatchAsync` works for host→host with pinned memory. No new kernel needed.

**`get_finished`**: launch all three event types (load, store, demote). Poll the same event lists, now split 3-ways. Report completed demote events via a new field on `SimpleCPUOffloadWorkerMetadata`.

### Critical files to modify

| File | Change |
|---|---|
| `vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py` | Parse `fast_cpu_bytes` / `slow_cpu_bytes` from `extra_config`; backward-compat alias `cpu_bytes_to_use` → `fast_cpu_bytes`. Log both capacities. |
| `vllm/v1/simple_kv_offload/manager.py` | Introduce `CpuTier`; dual-coordinator construction; cross-tier load + demotion store logic; LRU side-map; new event types. |
| `vllm/v1/simple_kv_offload/worker.py` | Dual pinned-tensor allocation; two `DmaCopyBackend` instances + demote backend; launch + poll three event streams. |
| `vllm/v1/simple_kv_offload/metadata.py` | Extend metadata dataclasses with tier fields + demote event. |
| `vllm/v1/simple_kv_offload/copy_backend.py` | No change expected — verify host→host with pinned-memory works via existing `cuMemcpyBatchAsync`; if not, add a thin `MemcpyKind.HostToHost` path. |

**Not modified for M1**: `OffloadingConnector`, `MultiConnector`, `KVCacheCoordinator`, `BlockPool`. Zero-change upstream code.

## Out of scope for M1

- Lazy mode tiering (keep lazy path single-pool or disabled when `slow_cpu_bytes>0`)
- Promotion on slow-tier load hit
- HMA multi-group interaction (slow pool shares groups with fast → should Just Work but untested)
- Real slow-medium backings (NUMA-remote, unpinned, CXL, NVMe)
- Metrics / observability — add only a log line per tier

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
- Whether the LRU side-map needs thread-safety — the scheduler runs single-threaded today but worth a re-check once the code is written.

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
- [ ] **[Scheduler] Demotion in eager store path** — LRU side-map per tier; demote fast→slow when fast is full; cap demotions per step.
- [ ] **[Scheduler] Completion handling for 3 event types** — split `_process_store_event` into load/store/demote; hash-map migration on demote completion.
- [ ] **[Worker] Dual pinned tensors + two `DmaCopyBackend` instances** — allocate per-tier CPU tensors; add `_demote_backend` for host→host.
- [ ] **[Worker] Launch + poll 3 event streams** — dispatch to correct backend; report completed demote events.
- [ ] **[Connector] Parse `fast_cpu_bytes` / `slow_cpu_bytes` config** — backward-compat alias; thread capacities to scheduler/worker.
- [ ] **[Tests] Unit tests for two-tier manager** — dual coordinators, demotion, load precedence, backward compat.
- [ ] **[Tests] Integration + manual smoke test** — two-pool end-to-end vs single-pool baseline.
- [ ] **[Release] Lint, pre-commit, open draft PR** — per `AGENTS.md` workflow.
