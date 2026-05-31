# Secondary memory: resiliency & advanced placement modes — design proposal

| | |
| --- | --- |
| **Status** | Proposal (no implementation yet) |
| **Owner** | @wangchen615 |
| **Created** | 2026-05-12 |
| **Last revised** | 2026-05-31 (rebased onto `OffloadingConnector`; absorbed inclusive-hierarchical placement; cross-deployment recovery moved to M1) |
| **Companion** | [secondary-memory-m1-implementation.md](secondary-memory-m1-implementation.md) (M1 — prerequisite) |
| **Parent** | [secondary-memory-system-overview.md](secondary-memory-system-overview.md) |
| **Sibling** | [exclusive-tiered-caching.md](exclusive-tiered-caching.md) |

## Context

This RFC targets a **three-level memory hierarchy** for LLM KV cache:

```text
  Accelerator HBM  ↔  secondary fast memory system  ↔  slow DRAM on host
```

A companion RFC ([secondary-memory-m1-implementation.md](secondary-memory-m1-implementation.md)) covers **M1**: a functional emulation of that hierarchy on top of vLLM's `OffloadingConnector`, using one shared host-memory pool surfaced as two CPU media (`FAST_CPU` and `SLOW_CPU`) and arbitrated by an external **Host Memory Pool Manager** (HMPM). M1 ships `placement_mode = "partitioned"` plus cross-deployment recovery on peer-deployment failure.

> **Terminology**: This document uses neutral hardware-agnostic names — **secondary fast memory** for the novel middle tier and **accelerator** for any device that owns HBM. **Medium** is the connector-side identifier for a tier (`"FAST_CPU"`, `"SLOW_CPU"`). See [secondary-memory-system-overview.md §Terminology](secondary-memory-system-overview.md#terminology).

This RFC covers four follow-on capabilities:

1. **Three additional placement modes** — `inclusive`, `hybrid`, `replicate` — each filling a different point on the capacity-vs-resiliency-vs-survivor-bandwidth trade-off.
2. **Hot-failure detection inside one deployment** — a medium becoming unresponsive while the deployment is alive.
3. **In-deployment failover** — when a medium fails, redirect reads to a surviving medium where one exists; recompute or re-prefill where it does not.
4. **The externally-managed read-only tier** (model-sharing variant) — multiple accelerators reading one canonical model from a shared, externally-owned tier.

This is a **proposal, not a plan**. No implementation is scheduled. It exists on the M1 PR so reviewers can evaluate the long-term shape alongside the M1 code — the M1 abstractions must not foreclose this design.

## Motivation

Three production-ish scenarios drive this RFC. They are distinct from M1's cross-deployment recovery (which handles a *peer deployment dying entirely*) — these are about a single deployment surviving the trouble of a single medium misbehaving, plus the model-sharing use case that wants the same multi-medium plumbing for a different reason.

### Use case 1: hot tier failure inside one deployment

A deployment stays alive but one of its CPU media goes sick — a CXL link flaps, a NUMA node wedges, an ioctl hangs, an out-of-band firmware event takes the secondary fast memory pool unresponsive for seconds. The deployment is fine; *one tier* is not. Goal: **transparent failover to the surviving tier (where one exists), with at most a latency bump, no request failure, no re-prefill.**

The diagrams below illustrate the two within-deployment recovery paths this RFC has to cover. (Cross-deployment recovery — Scenario A in the previous draft of this doc — is now M1; see [exclusive-tiered §5](exclusive-tiered-caching.md#5-the-host-memory-pool-manager-hmpm).)

#### Scenario B — recompute on tier failure (`partitioned` mode)

Under `partitioned`, a high-priority request's KV blocks live **only** in `FAST_CPU`. If that medium fails, those requests **re-prefill** on the accelerator. Slow-medium requests are unaffected.

<img alt="FAST_CPU unresponsive; high-priority requests recompute on the accelerator; SLOW_CPU requests continue normally" src="imgs/svg/resilience-recompute-tiered.svg" width="780">

Source: [`imgs/mmd/resilience-recompute-tiered.mmd`](imgs/mmd/resilience-recompute-tiered.mmd).

#### Scenario C — prefetch from slow on tier failure (`inclusive` mode)

Under `inclusive`, the slow medium is a **superset** of the fast medium. Every block that was in fast also exists in slow. When the fast medium fails, recovery is a **prefetch from slow into HBM** — no re-prefill needed.

<img alt="FAST_CPU unresponsive; SLOW_CPU is a superset and prefetches every missing block back into HBM; requests resume" src="imgs/svg/resilience-prefetch-hierarchical.svg" width="780">

Source: [`imgs/mmd/resilience-prefetch-hierarchical.mmd`](imgs/mmd/resilience-prefetch-hierarchical.mmd).

The placement-mode menu below makes the trade-off explicit.

### Use case 2: model sharing for agentic workflows

In a heavy-overcommit agentic deployment, many small models are loaded and unloaded across many accelerators on rapid time-scales — request-driven model switches measured in seconds, not minutes. Re-fetching each model from object storage on every switch is unacceptably slow.

The secondary-memory hierarchy can address this by treating the secondary fast memory tier as a **shared, canonical home for model weights**:

- The model is loaded once into the secondary fast memory pool by an external manager (the platform, not vLLM).
- Each accelerator that wants to serve the model **reads** it in from the shared tier on demand.
- The model **stays canonical in the secondary fast memory pool** even after an accelerator finishes — the next switch back is a fast read, not a re-load from object storage.
- Multiple accelerators can read the same model concurrently.

This is `replicate` semantically (the same data lives in the shared tier *and* in each executor's working set), with two extensions that distinguish it from the resiliency case:

- **Read-only**: the connector only reads from the shared medium — only the external manager writes. Writability is per-medium and per-mode.
- **Externally managed lifecycle**: the connector does not control allocation or eviction in the shared medium; the platform does. Blocks may appear and disappear independently of connector actions.

This use case does **not** require the failure-detection / failover machinery from use case 1 — model weights either load successfully or the request fails up-front. But it shares the placement-mode plumbing, which is why it lives in this RFC.

### Why this belongs in the manager, not the scheduler or connector

For both use cases, the relevant property — tier health (use case 1) or tier writability/ownership (use case 2) — is a property of the offload substrate, not of request scheduling. Keeping mode handling, detection, and failover inside the `OffloadingManager` layer (specifically: by swapping which `OffloadingManager` subclass the spec instantiates) means the scheduler stays unaware of medium topology — it just sees "KV cache hit" or "miss" as today. That keeps the blast radius small and makes the feature opt-in via a config flag.

## Design

### Placement modes (the core design space)

Resiliency and capacity are fundamentally **placement** questions: *do blocks live in one medium, the other, or both?* M1's `partitioned` mode is the non-resilient endpoint optimized for capacity-by-class-of-service. Three more modes fill out the spectrum:

| Mode | Semantics | Block can live in… | Inter-medium movement | In M1? |
| --- | --- | --- | --- | --- |
| **partitioned** | Each request is admitted to one medium based on `Request.priority`. The two media cache **different sets** of blocks. | Exactly one | None | **Yes** (M1) |
| **inclusive** | HBM eviction cascades into the hierarchy. Slow is a **superset** of fast: every block in fast is also in slow. Reuse-driven (heat counter) decides what's in fast. Demote on fast LRU; optional promote on slow hit. | Both (slow is superset) | Demote (fast→slow) on fast LRU; optional async promote (slow→fast) | No |
| **hybrid** | New stores go to fast; a bounded async mirror also writes to slow. Under capacity pressure the mirror is dropped. | One or both | Async mirror on store | No |
| **replicate** | Every store goes to both media synchronously. Loads prefer fast; slow is used on fast-miss or fast-failure. | Both | Sync mirror on store | No |

> **Each placement mode is a separate `OffloadingManager` subclass.** The spec instantiates one based on `placement_mode`. M1 ships `MultiMediaOffloadingManager` for `partitioned`; this RFC's modes ship as `InclusiveCascadeManager`, `HybridReplicateManager`, `ReplicateManager`. The spec, the connector, and the worker do not branch on mode.

The mode is selected at connector init via `kv_connector_extra_config.placement_mode` (default `"partitioned"`, preserving M1 behavior). Users pick where on the spectrum they sit based on workload:

- Class-of-service serving (priority-routed traffic, capacity-oriented) → `partitioned` *(M1)*
- Long-context serving where slow tier should shadow fast for free recovery → `inclusive`
- Production serving with SLO guarantees on memory-tier failures → `replicate` (pay the capacity tax for survival)
- Mixed production with some SLO headroom → `hybrid`
- Agentic / multi-model workloads with rapid model switches → `replicate` with the read-only / externally-managed variant (see "Model-sharing variant" below)

### Placement mode: `inclusive`

The `inclusive` mode is what conventional tiered caches do: **fast is a cache of slow**. It absorbs the design that previously lived in the now-deleted `inclusive-hierarchical-caching.md` doc.

#### Placement model

The two CPU media form an **inclusive hierarchy**: every block resident in fast is also present in slow. Slow is a superset. Eviction from HBM cascades into the hierarchy; eviction from fast demotes (if not already in slow); eviction from slow drops the block.

#### Invariants

- `fast ⊆ slow`. Every block in the fast medium is also in the slow medium.
- A block leaves the system only when it is evicted from **slow**. Eviction from fast alone never destroys content.
- Reuse rate (a heat counter) is the placement signal — higher heat → fast, lower heat → slow only.

The superset invariant is what gives `inclusive` its resiliency story (Scenario C above): if the fast medium fails, **every block can still be served from slow**, no re-prefill.

#### Flow (HBM eviction → store)

1. The scheduler bumps the per-block heat counter on each cache hit. (Heat is a small integer, decayed periodically — concrete decay schedule deferred to implementation.)
2. When HBM evicts block *B*:
   - If `B`'s heat is high **and** the fast medium has room, store *B* into both fast and slow (paired write to maintain `fast ⊆ slow` from the start).
   - Otherwise store *B* into slow only.

#### Flow (fast-tier pressure → demote)

1. Fast picks an LRU victim *V*.
2. If *V* is hot (high heat), copy *V* into slow if not already there, then drop from fast.
3. If *V* is cold, drop from fast directly. Slow already holds the master copy.

#### Flow (load)

1. Look up in fast. If hit, copy fast → HBM. Done.
2. If fast missed, look up in slow. If hit, copy slow → HBM.
3. **Optional**: if `promote_on_load` is enabled and fast has room, kick off an async fast-medium write of the loaded block. Not on the critical path.

#### What `inclusive` adds beyond M1

- **A heat counter per cached block hash**, maintained by the manager — likely as a new `CachePolicy` plug-in in `vllm/v1/kv_offload/cpu/policies/` (the registry is already there, and `vllm/v1/kv_offload/reuse_manager.py` already maintains a frequency tracker for `FilterReusedOffloadingManager`, so the building block is in tree).
- **A host-to-host copy path** (CPU→CPU) for demote and optional promote. M1's spec yields four handlers (GPU↔fast, GPU↔slow); `inclusive` adds a fifth and sixth (FAST_CPU↔SLOW_CPU). The existing `OffloadingHandler` interface accommodates this without change.
- **A `demote` event type** in `OffloadingEvent` — an `OffloadingEvent` whose `medium` is the destination and whose source-medium hint is set on a new optional field. Or, since `OffloadingEvent.removed` already distinguishes store from removal, we can encode demote as a paired `removed=True` (from fast) + `removed=False` (to slow) event from the manager.

These are additive — none of them break M1's manager, spec, or worker.

### Placement modes: `hybrid` and `replicate`

#### `replicate`

Every `prepare_store` writes to **both** managers; loads check fast first, fall back to slow on miss or in-deployment fast-failure.

Effective capacity = `min(fast, slow)`. Resiliency = full — either medium alone is sufficient.

Implementation: `ReplicateManager(OffloadingManager)` wraps both per-medium managers. `prepare_store` calls both; `lookup` checks fast first, then slow; eviction is independent per medium (acceptable, because the cache is sized by `min`). One subtlety: a successful store must succeed on **both** before the manager returns success; a partial-success path degrades the affected key to "resident in only one medium" and emits a degradation event.

#### `hybrid`

`prepare_store` writes to fast synchronously; a **bounded async mirror queue** writes the same block to slow if and when slow has capacity. Under capacity pressure the mirror is dropped (the block is then partitioned-like, fast-only).

Effective capacity is between `min(fast, slow)` (no pressure) and roughly `fast` plus the headroom slow can shadow (under pressure). Resiliency = partial — recently-stored hot blocks are mirrored, older ones may not be.

Implementation: `HybridReplicateManager(OffloadingManager)`. Wraps `ReplicateManager`'s store path with a bounded queue feeding the slow side. Drop policy: oldest-mirror-first when queue is full. Drop event emitted so observability can track effective replication coverage.

### Model-sharing variant of `replicate`

The agentic / model-sharing use case (motivation §2) reuses `replicate` semantically — same data lives in two media — but with two extensions to the per-medium contract:

| Property | Resiliency `replicate` | Model-sharing `replicate` |
| --- | --- | --- |
| Connector writes to slow medium? | Yes (every store mirrored) | **No** — read-only from connector's perspective |
| Allocation in slow medium | Connector-driven (via HMPM) | **Externally driven** (platform manager) |
| Eviction in slow medium | Connector-driven (LRU) | **Externally driven** — blocks may disappear without connector action |
| Failure handling | Failover to other medium on hot failure | None — model load either succeeds or the request fails |

The connector exposes two new per-medium flags to encode this:

- `medium.writable: bool` — if `False`, the connector never issues store ops against this medium. Only loads.
- `medium.externally_managed: bool` — if `True`, the connector treats the medium's contents as authoritative-but-volatile; cache lookups respect what's there but don't assume a block stays around between scheduler steps. Affects the load path's "still cached?" recheck logic.

These flags are orthogonal to `placement_mode`. A `replicate` deployment for resiliency uses `writable=true, externally_managed=false` for both media. A `replicate` deployment for model sharing uses `writable=true, externally_managed=false` for the executor-local fast medium and `writable=false, externally_managed=true` for the shared slow medium. Future deployments may mix all four combinations.

The implementation impact is small once `placement_mode` is in: skip stores for `writable=false` media, and add a freshness check on load for `externally_managed=true` media. The HMPM grows a new `attach_readonly()` entry point that returns a handle which cannot allocate.

### Failure model: hot failure inside one deployment

This proposal targets **hot failure**: a medium becomes unresponsive during live serving, not at process restart. Recovery has to happen in-band, while other requests are flowing.

Cold-restart recovery (reload KV from the survivor on process boot) is a strictly easier problem and a reasonable fast-follow. It reuses the same health-state + placement infrastructure; the detection and failover paths are where the complexity lives.

Note that **peer-deployment failure** — one deployment dies entirely while another stays alive — is a different mechanism, and it is in M1, mediated by the HMPM heartbeat ([exclusive-tiered §5](exclusive-tiered-caching.md#5-the-host-memory-pool-manager-hmpm)). Hot in-deployment failure and peer-deployment failure can coexist in one deployment without fighting.

### Mechanism stack (in-deployment failure)

1. **Detect (per-medium timeout + health state)**
   - Add an optional `timeout_ms` parameter to `OffloadingHandler.start_transfer` and the per-medium copy primitives.
   - Per-medium health state machine on `MultiMediaOffloadingManager` (or its mode-specific subclass): `healthy → suspect → quarantined`.
   - `healthy → suspect`: first timeout or error. Subsequent ops still try this medium but with aggressive back-off.
   - `suspect → quarantined`: N consecutive failures within window W. No new ops issued until re-enabled.
   - `suspect → healthy`: M consecutive successes within window W.

2. **Fail over (inline on read path)**
   - On a fast-medium read timeout, retry the read on slow if the block is known to be there (i.e., `placement_mode` ∈ {`inclusive`, `hybrid`, `replicate`} *and* the per-key residence map indicates slow also holds a copy).
   - If the block is not replicated (`partitioned` mode, or `hybrid` with a non-mirrored block), the request falls back to re-prefill — same outcome as M1 today.
   - On a slow-medium write failure under `replicate`, drop the mirror for that block and degrade it to `partitioned`-like residence (log once). Store to fast still succeeds.

3. **Quarantine**
   - Once quarantined, a medium is skipped by all new ops. Existing pinned blocks in that medium stay pinned (the medium might become reachable again), but the manager no longer considers it for new admissions.
   - First cut: manual re-enable via an admin RPC or config reload. Auto-heal is a follow-up — it requires probing the medium safely without blocking live serving.

4. **Re-replication (on recovery)**
   - When a medium returns from quarantine, start a **rate-limited background re-replication** from the survivor to restore redundancy.
   - Bounded concurrency to stay off the critical path of live requests. Priority: recently-touched blocks first (most likely to be hit soon).

### Configuration surface

New fields in `kv_connector_extra_config` (in addition to the M1 fields):

| Field | Type | Default | Effect |
| --- | --- | --- | --- |
| `placement_mode` | `"partitioned" \| "inclusive" \| "hybrid" \| "replicate"` | `"partitioned"` | Picks the placement mode (and thus which manager subclass the spec instantiates). |
| `heat_promotion_threshold` | int | `2` | (`inclusive` only) Heat counter value above which a block is admitted to fast on HBM eviction. |
| `promote_on_load` | bool | `false` | (`inclusive` only) Async-write to fast on a slow load hit. |
| `heat_decay_period_steps` | int | `1024` | (`inclusive` only) Scheduler steps between decay sweeps of the heat counter. |
| `mirror_queue_size` | int | `1024` | (`hybrid` only) Bounded queue of pending fast→slow async mirrors. |
| `medium_timeout_ms` | int | `0` (disabled) | Per-op timeout. `0` = no timeout (M1 behavior). |
| `medium_failure_threshold` | int | `3` | Consecutive failures to move `suspect → quarantined`. |
| `medium_recovery_threshold` | int | `10` | Consecutive successes to move `suspect → healthy`. |
| `replication_rate_limit_mb_s` | int | `256` | (re-replication on recovery) Cap on background bandwidth. |
| `media[i].writable` | bool | `true` | Per-medium flag. `false` makes the connector skip stores against this medium (model-sharing variant). |
| `media[i].externally_managed` | bool | `false` | Per-medium flag. `true` tells the connector the medium's contents may change without its action; load path adds a freshness recheck. |

Backward compat: omitting all of these reproduces M1 behavior exactly.

### Manager subclass plan

Each placement mode is a separate `OffloadingManager` subclass that the spec picks at construction time. None of them edits `OffloadingConnectorScheduler` or `OffloadingConnectorWorker`.

```python
# Existing M1
class MultiMediaOffloadingManager(OffloadingManager): ...   # partitioned

# Future modes
class InclusiveCascadeManager(OffloadingManager):
    """fast ⊆ slow; heat-driven; demote on fast LRU; optional promote on load."""
    ...

class HybridReplicateManager(OffloadingManager):
    """Sync store to fast + bounded async mirror to slow."""
    ...

class ReplicateManager(OffloadingManager):
    """Sync store to both; loads prefer fast."""
    ...
```

`SecondaryMemoryOffloadingSpec.get_manager()` becomes:

```python
def get_manager(self):
    fast = CPUOffloadingManager(self.fast_blocks, ...)
    slow = CPUOffloadingManager(self.slow_blocks, ...)
    mode = self.extra_config.get("placement_mode", "partitioned")
    if mode == "partitioned":
        return MultiMediaOffloadingManager(fast, slow,
            PriorityAdmissionPolicy(fast, slow, self.priority_threshold))
    elif mode == "inclusive":
        return InclusiveCascadeManager(fast, slow, ...)
    elif mode == "hybrid":
        return HybridReplicateManager(fast, slow, ...)
    elif mode == "replicate":
        return ReplicateManager(fast, slow, ...)
    raise ValueError(mode)
```

### Metadata changes over M1

M1's `MultiMediaOffloadingManager` already maintains a `dict[OffloadKey, Literal["fast", "slow"]]` residence map (single-medium per key). This proposal extends it:

- For replicated modes (`inclusive`, `hybrid`, `replicate`), the value type becomes `set[Medium]` — the set of media currently holding each key.
- `replica_media: set[str]` is the new per-key field. Populated on store completion, consulted on read for failover eligibility.

No change to the OffloadKey encoding: replicated modes track residence as manager-side metadata, not as duplicate entries in the per-medium policy keymaps.

## Implementation sketch

Not scoped here, but to show the work is contained:

1. **Add `placement_mode` plumbing** through `SecondaryMemoryOffloadingSpec.get_manager()` (M1 already has the spec; this is one switch statement).
2. **Implement `InclusiveCascadeManager`** with heat-counter `CachePolicy` plug-in + host-to-host copy handlers (two new entries in `get_handlers()`).
3. **Implement `ReplicateManager`** — straightforward fan-out on `prepare_store`.
4. **Implement `HybridReplicateManager`** — `ReplicateManager` + a bounded async mirror queue.
5. **Add `TierHealth` state machine** to each manager — drained on each `take_events()` cycle.
6. **Wire `medium_timeout_ms`** through `OffloadingHandler.start_transfer` and expose a failure callback to the manager, which feeds the `TierHealth` state machine.
7. **Add the failover read path** to each manager: on timeout, check `replica_media`, resubmit on a surviving medium.
8. **Add a bounded re-replication scheduler** — a new low-priority event type, rate-limited by `replication_rate_limit_mb_s`.
9. **Add per-medium `writable` and `externally_managed` flags**: skip stores against `writable=false` media; add a freshness recheck on the load path for `externally_managed=true` media. The HMPM grows `attach_readonly()`.

## Risks and open questions

- **Timeout tuning is hard**: too tight and healthy-but-slow media get quarantined under load; too loose and failures aren't detected fast enough to matter. First cut should be conservative + configurable, with telemetry to tune in production.
- **Silent corruption is not covered**: this proposal defends against unresponsiveness, not wrong data. Checksumming per block is a materially larger project and probably a separate RFC.
- **Split-brain during re-replication**: if a medium is quarantined, serves reads anyway, then returns and has stale data — we need to version blocks or treat every recovery as "wipe + re-replicate from survivor." The simpler option is the latter; worth confirming before implementation.
- **Auto-heal vs manual re-enable**: automatic recovery detection risks flapping. Manual re-enable is safer but operationally worse. Decision can be deferred to implementation time.
- **Cold-restart recovery**: arguably should be done **first** in this RFC's order since it's simpler and provides most of the value for planned maintenance. TBD whether to split this proposal into "resiliency-cold" and "resiliency-hot" mini-RFCs.
- **Externally-managed medium freshness**: the model-sharing variant assumes the connector can detect when an `OffloadKey` referenced by the manager's keymap has been evicted by the external manager. The simplest mechanism is to recheck the medium's view on every load through the HMPM client; harder cases (the manager swaps in *different* data under the same address) require either an opaque versioning token from the platform or a content hash check on read. Pick one before implementation.
- **Coordinating evictions across executors sharing a medium**: when N accelerators read from the same shared model medium, can any of them indirectly cause an eviction of a block another is depending on? In the model-sharing variant the platform manages eviction, so the answer depends on what the platform does — but the connector's load-time freshness recheck must be cheap enough that it's safe to run on the hot path.
- **Inclusive mode + HMPM interaction**: M1's HMPM tracks ownership at the (deployment, medium, block) tuple level. `inclusive` mode within one deployment writes to both media for the same key, which is fine — two HMPM allocations under one deployment id. No new HMPM API needed, but worth verifying the heartbeat / orphan-detection logic doesn't double-count.

## Relationship to M1

M1 lands no resiliency code, no inclusive/hybrid/replicate code, and no model-sharing code. M1 *does* land cross-deployment recovery on peer-deployment failure (a different mechanism from the in-deployment failover described here).

What M1 **must not foreclose** — already honored in the M1 design:

- **Each placement mode is a separate `OffloadingManager` subclass.** The spec dispatches on `placement_mode`. M1's `MultiMediaOffloadingManager` is one of four siblings; the others slot in without touching the spec's external interface, the connector, the worker, or the scheduler.
- **The residence map generalizes from `Literal["fast","slow"]` to `set[Medium]`.** The change is contained to the manager's internal bookkeeping; no public-API break.
- **`OffloadingHandler.start_transfer` has no `timeout_ms` yet.** M1 leaves the signature alone; this proposal adds the parameter as an optional kwarg.
- **The HMPM does not assume the connector owns the medium's lifecycle** — `attach_readonly()` covers the model-sharing variant. M1's HMPM API has room for it (the `attach` API is parameterized and additive).
- **Per-medium events already carry `medium`.** `OffloadingEvent.medium` is in `base.py` today; no schema change needed when resiliency adds tier-health state and quarantine events.
- **`AdmissionPolicy` is composable with placement modes.** `inclusive` ignores admission (cascade is admission-free); `replicate` writes to all media regardless. The Protocol stays.

If M1 review uncovers anything that would constrain this proposal, we update both RFCs together.
