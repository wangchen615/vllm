# Exclusive tiered caching — component design

| | |
| --- | --- |
| **Status** | Draft |
| **Branch** | `dev` |
| **Owner** | @wangchen615 |
| **Created** | 2026-05-20 |
| **Implements** | `placement_mode = "partitioned"` for M1 |
| **Companion** | [secondary-memory-m1-implementation.md](secondary-memory-m1-implementation.md) (M1 implementation plan) |
| **Parent** | [secondary-memory-system-overview.md](secondary-memory-system-overview.md) |

This document is the **component-level design** for the M1 placement model. The M1 implementation plan ([secondary-memory-m1-implementation.md](secondary-memory-m1-implementation.md)) gives the file-by-file edit list; this document gives the *behavior* — how the scheduler, worker, metadata, and tier abstraction interact.

> **Naming**: "exclusive tiered caching" and the M1 config token `placement_mode = "partitioned"` refer to the same model. The user-facing prose in the RFC family uses **exclusive tiered**; the config string keeps the token `partitioned` so the M1 implementation does not need to rename a knob.

## 1. Placement model

Each KV block lives in **exactly one** of the two CPU tiers for the lifetime of that block. Admission is decided once, at store time, from a per-request signal (M1: `Request.priority`). A block is never copied between tiers; each tier evicts independently using its own `BlockPool`'s LRU.

<img alt="Exclusive tiered placement: high-priority requests' blocks live in the secondary fast memory pool, lower-priority requests' blocks live in the slow host DRAM pool, no inter-tier movement" src="imgs/svg/exclusive-tiered-placement.svg" width="720">

Source: [`imgs/mmd/exclusive-tiered-placement.mmd`](imgs/mmd/exclusive-tiered-placement.mmd).

The "warmer" colors in the diagram represent the *intent*: the fast tier is meant for blocks the workload reuses most. M1 does not measure reuse — it uses request priority as a coarse proxy. A finer reuse-tracking signal (a heat counter per block) is reserved for the [inclusive hierarchical](inclusive-hierarchical-caching.md) mode, where it actually drives runtime decisions.

### What "exclusive" means and does not mean

- **Each *physical* block is in exactly one tier.** The block ID returned by a tier's `BlockPool` is unique to that tier. There is no shared physical address space.
- **The same *content* may legitimately appear in both tiers' hash maps.** If two requests at different priorities both produce the same KV block (same prompt prefix, same model, same group), the high-priority instance lives in fast and the low-priority instance lives in slow. Each is a different physical block. The load path is responsible for resolving which to serve from.
- **No demotion path.** When the fast tier fills, the next admission causes fast's own LRU victim to be **dropped**, not copied to slow. No CPU→CPU transfer is launched.
- **No promotion path.** A slow-tier hit serves directly from slow. There is no implicit copy into fast on access.

### Why no demote / promote in this mode

The two tiers are admitting *different populations of requests* on purpose, not buffering the same requests at two speeds. Mixing in demote/promote would re-introduce the "fast is just a smaller cache of slow" semantics that M1 deliberately avoids — that model is the [inclusive hierarchical](inclusive-hierarchical-caching.md) one and is covered separately.

## 2. End-to-end flow

The figure below shows one scheduler step plus the corresponding worker activity for both the **store** path (KV blocks just produced on the GPU need to land in CPU) and the **load** path (a new request hits the prefix cache).

<img alt="Sequence: scheduler chooses tier from request priority, emits one event per tier per step, worker launches GPU↔tier copies independently" src="imgs/svg/exclusive-tiered-flow.svg" width="780">

Source: [`imgs/mmd/exclusive-tiered-flow.mmd`](imgs/mmd/exclusive-tiered-flow.mmd).

### Store path — admission and tier choice

1. The scheduler reads `request.priority` for each request that has freshly committed KV blocks.
2. `_choose_tier(request)` returns the fast tier if `priority < priority_threshold` (default `1`), else the slow tier.
3. The scheduler enqueues at most **one store event per tier per scheduler step**. The metadata carries `store_cpu_tier` (0 or 1) plus the per-block list.
4. The worker dequeues the event, calls the appropriate `DmaCopyBackend.launch_copy(GPU → tier, is_store=True)`, and keeps the `torch.Event` for completion polling.

The "one event per tier per step" cap keeps the worker's scheduling loop bounded and gives the per-tier counters a simple, race-free shape.

### Load path — prefix-cache lookup

1. The scheduler calls `find_longest_cache_hit` on the **fast tier first**, then the **slow tier**, with the same prompt prefix.
2. The longer of the two hits wins. Ties go to fast (cheaper to load, by design intent — even though M1 has no real latency difference).
3. The metadata carries `load_cpu_tiers: list[int]` parallel to `load_cpu_blocks`, so the worker knows which `DmaCopyBackend` to use for each block.
4. The worker issues per-tier `launch_copy(tier → GPU, is_store=False)` calls. There is no implicit "promote on load" copy.

### Independent eviction

Each tier's `BlockPool` runs its own LRU. When tier *T* needs a free slot:

1. `BlockPool.get_new_block()` returns the LRU victim.
2. The block is removed from `cached_block_hash_to_block` for *T* only.
3. **No copy is issued anywhere.** The block content is gone.

A request that was relying on that block will simply prefix-miss the next time it tries to use it. Whether that miss is acceptable is a workload-level question; M1 does not promise capacity additivity.

## 3. The `CpuTier` abstraction

The two tiers are managed through a single small abstraction so the scheduler and worker do not branch on "fast vs slow" everywhere. Asymmetry lives only at the admission decision (`_choose_tier`); everything below that point is symmetric.

<img alt="Class diagram: CpuTier composes a KVCacheCoordinator, BlockPool, and DmaCopyBackend; SimpleCPUOffloadScheduler and Worker each hold up to two CpuTiers" src="imgs/svg/cputier-class.svg" width="780">

Source: [`imgs/mmd/cputier-class.mmd`](imgs/mmd/cputier-class.mmd).

`CpuTier` deliberately carries the resiliency-RFC fields (`health`, `writable`, `externally_managed`) as `None` / default values in M1 — they exist in the type so future placement modes can flip them without an invasive refactor. M1 reads none of them.

## 4. Block-hash relationships

The figure below shows how a logical block hash relates to physical blocks across tiers and to requests. It is the data-model the scheduler reasons about.

<img alt="ER diagram: BLOCK_HASH ↔ TIER_RESIDENCE ↔ TIER ↔ BLOCK_POOL ↔ PHYSICAL_BLOCK; REQUEST produces/consumes BLOCK_HASH and carries priority" src="imgs/svg/block-tier-er.svg" width="720">

Source: [`imgs/mmd/block-tier-er.mmd`](imgs/mmd/block-tier-er.mmd).

In **exclusive tiered**, the cardinality `BLOCK_HASH ||--o{ TIER_RESIDENCE` is interpreted as "lives in *N* tiers, where *N = 1*." The same diagram is reused by the [inclusive hierarchical](inclusive-hierarchical-caching.md) RFC with `N ≥ 1` — a single ER source covers both modes.

## 5. Store-event lifecycle

Each store event the scheduler emits goes through the lifecycle shown below.

<img alt="State machine: a store event flows Pending → InFlight → Completed → Reported, with a Failed/Quarantined branch handled by the resiliency RFC" src="imgs/svg/store-event-state.svg" width="720">

Source: [`imgs/mmd/store-event-state.mmd`](imgs/mmd/store-event-state.mmd).

M1 implements the green path only: `Pending → InFlight → Completed → Reported`. The `Failed → Quarantined` branch is referenced from the [resiliency RFC](secondary-memory-resiliency.md); M1's worker does not detect failed copies — a `torch.Event` either completes or the process dies.

## 6. Configuration surface

This is the user-visible knobs the M1 connector exposes. See [secondary-memory-m1-implementation.md §Design committed for M1](secondary-memory-m1-implementation.md) for the precise defaults and the legacy-key alias.

| Key | Type | Effect |
| --- | --- | --- |
| `fast_cpu_bytes` | int | Capacity of the emulated secondary fast memory tier. `0` disables fast. |
| `slow_cpu_bytes` | int | Capacity of the emulated slow host DRAM tier. |
| `priority_threshold` | int | Admission boundary. `request.priority < threshold` → fast, else → slow. |
| `cpu_bytes_to_use` (legacy) | int | Maps to `slow_cpu_bytes` with `fast_cpu_bytes=0`. Backwards-compatible. |

`placement_mode` is implicit and fixed to `"partitioned"` in M1. The keyword exists in the design so future RFCs can add `"replicate"`, `"hybrid"`, and `"inclusive"` without breaking the M1 config shape.

## 7. Worker-side mechanics

- **Two `DmaCopyBackend` instances**, one per tier. Each is parameterized with that tier's pinned CPU tensor dict at init time.
- **No host↔host backend** — exclusive tiered has no inter-tier copies, so the demote backend planned in earlier drafts is **not** instantiated in M1. (It is added in [inclusive hierarchical](inclusive-hierarchical-caching.md).)
- **Streams**: M1 reuses the existing `load_stream` / `store_stream` pair across both backends. CUDA serialization on a single stream is fine; ordering is preserved.
- **Completion polling**: `get_finished()` walks both tiers' event lists and reports per-tier completed counters back to the scheduler via `SimpleCPUOffloadWorkerMetadata`.

## 8. What this RFC does not cover

- **Heat tracking and reuse-driven placement** — see [inclusive hierarchical](inclusive-hierarchical-caching.md). M1 is priority-driven, not reuse-driven.
- **Cross-tier failover** — see [resiliency RFC](secondary-memory-resiliency.md). When the fast tier fails in exclusive tiered mode, M1's behavior is "those requests re-prefill"; the resiliency RFC formalizes detection / quarantine / re-route.
- **Non-priority admission signals** (model id, request size, prompt length) — orthogonal to M1; can be added at `_choose_tier(request)` later without changing any tier-side data structure.
- **Real secondary-memory backing** — the M1 emulator uses pinned host DRAM for both pools. Switching pool #0 to a real backing is a worker-only change.
