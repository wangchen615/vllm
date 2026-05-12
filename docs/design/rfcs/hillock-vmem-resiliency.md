# hillock-vmem: resiliency across the memory hierarchy — design proposal

| | |
|---|---|
| **Status** | Proposal (no implementation yet) |
| **Project** | hillock-vmem |
| **Owner** | @wangchen615 |
| **Created** | 2026-05-12 |
| **Companion** | [hillock-vmem-two-tier-offload.md](hillock-vmem-two-tier-offload.md) (M1 functional emulation — prerequisite) |

## Context

The hillock-vmem project targets a **three-level memory hierarchy** for LLM KV cache:

```
  GPU HBM  ↔  secondary-memory tier  ↔  CPU host memory
```

A companion RFC ([hillock-vmem-two-tier-offload.md](hillock-vmem-two-tier-offload.md)) covers **M1**: a functional emulation of that hierarchy using two CPU memory pools as stand-ins for the secondary-memory tier and the host-memory tier. M1 proves the Simple KV-offload connector can manage two address spaces with small, scoped changes.

**This RFC covers the next question**: once the hierarchy is real, how do we exploit its natural redundancy to keep serving through memory-tier failures?

This is a **proposal, not a plan**. No implementation is scheduled. It exists on the M1 PR so reviewers can evaluate the long-term shape alongside the M1 code — the M1 abstractions must not foreclose this design.

## Motivation

A real three-level hierarchy is **naturally redundant**: when the secondary tier and the host tier both participate in KV offload, some blocks end up on both. A resilient design exploits that redundancy so a memory-tier hiccup doesn't take out live requests.

Concretely, a production deployment may face:

- **Transient stalls** — a CXL link flaps, a remote NUMA node wedges, an ioctl hangs, an out-of-band firmware event takes a tier unresponsive for seconds.
- **Partial failures** — a pool is healthy but slow (thermal throttle, neighbor noise, flaky DMA path).
- **Planned degradation** — a tier is rebooted or reconfigured while the serving process stays up.

In all three cases the vLLM process is alive and other requests are still flowing. Dropping in-flight requests that happen to hit the sick tier — and forcing a cold re-prefill from scratch — is a poor outcome when another copy of the KV data is already sitting in the healthy tier. The goal of this proposal: **transparent fall-over to the surviving tier, with at most a latency bump, no request failure, no re-prefill.**

### Why this belongs in the connector, not the scheduler

Tier health is a property of the offload substrate, not of request scheduling. Keeping detection and failover inside the connector (and specifically inside the existing `CpuTier` + `DmaCopyBackend` abstractions) means the scheduler stays unaware of tier topology — it just sees "KV cache hit" or "miss" as today. That keeps the blast radius small and makes the feature opt-in via a config flag.

## Design

### Placement modes (the core design space)

Resiliency is fundamentally a **placement** question: *do blocks live in one tier, the other, or both?* M1's `exclusive` mode is the non-resilient endpoint. Two more modes fill out the spectrum:

| Mode | Semantics | Effective capacity | Resiliency | In M1? |
|---|---|---|---|---|
| **exclusive** | Block lives in fast OR slow, never both. Fast-to-slow demotion on fast eviction. | `fast + slow` | **None** — losing either tier loses whatever was only there | **Yes** (the only M1 mode) |
| **hybrid** | New stores go to fast; a bounded async mirror also writes to slow. Under capacity pressure the mirror becomes the demotion (exclusive) path. | Between `min(fast, slow)` and `fast + slow` depending on pressure | Partial — recently-stored hot blocks are replicated, older demoted ones are not | No |
| **replicate** | Every store goes to both tiers. Loads prefer fast; slow is used on fast-miss or fast-failure. | `min(fast, slow)` | **Full** — either tier alone is sufficient to continue serving | No |

The mode is selected at connector init via a new `kv_connector_extra_config.placement_mode` field (default `"exclusive"`, preserving M1 behavior). Users pick where on the spectrum they sit based on workload:

- Max-capacity batch inference → `exclusive`
- Production serving with SLO guarantees → `replicate` (pay the capacity tax for survival)
- Mixed production with some SLO headroom → `hybrid`

### Failure model: hot failure

This proposal targets **hot failure**: a tier becomes unresponsive during live serving, not at process restart. That means recovery has to happen in-band, while other requests are flowing.

Cold-restart recovery (reload KV from the survivor on process boot) is a strictly easier problem and a reasonable fast-follow. It reuses the same health-state + placement infrastructure; the detection and failover paths are where the complexity lives.

### Mechanism stack

1. **Detect (per-tier timeout + health state)**
   - Add an optional `timeout_ms` parameter to `DmaCopyBackend.launch_copy`.
   - Per-tier health state machine: `healthy → suspect → quarantined`.
   - `healthy → suspect`: first timeout or error. Subsequent ops still try this tier but with aggressive back-off.
   - `suspect → quarantined`: N consecutive failures within window W. No new ops issued until re-enabled.
   - `suspect → healthy`: M consecutive successes within window W.

2. **Fail over (inline on read path)**
   - On a fast-tier read timeout, retry the read on slow if the block is known to be there (i.e., `placement_mode` ∈ {`hybrid`, `replicate`} *and* the per-block source-tier hint indicates slow also holds a copy).
   - If the block is not replicated (`exclusive` mode, or `hybrid` with a demoted-only block), the request falls back to re-prefill — same outcome as today's single-pool failure.
   - On a slow-tier write failure under `replicate`, drop the mirror for that block and degrade it to `exclusive` placement (log once). Store to fast still succeeds.

3. **Quarantine**
   - Once quarantined, a tier is skipped by all new ops. Existing pinned blocks in that tier stay pinned (they might become reachable again), but the allocator no longer considers it.
   - First cut: manual re-enable via an admin RPC or config reload. Auto-heal is a follow-follow-up — it requires probing the tier safely without blocking live serving.

4. **Re-replication (on recovery)**
   - When a tier returns from quarantine, start a **rate-limited background re-replication** from the survivor to restore redundancy.
   - Bounded concurrency to stay off the critical path of live requests. Priority: recently-touched blocks first (most likely to be hit soon).

### Configuration surface

New fields in `kv_connector_extra_config`:

| Field | Type | Default | Effect |
|---|---|---|---|
| `placement_mode` | `"exclusive" \| "hybrid" \| "replicate"` | `"exclusive"` | Picks the replication policy. |
| `tier_timeout_ms` | int | `0` (disabled) | Per-op timeout on `DmaCopyBackend.launch_copy`. `0` = no timeout (M1 behavior). |
| `tier_failure_threshold` | int | `3` | Consecutive failures to move `suspect → quarantined`. |
| `tier_recovery_threshold` | int | `10` | Consecutive successes to move `suspect → healthy`. |
| `replication_rate_limit_mb_s` | int | `256` | Cap on background re-replication bandwidth. |

Backward compat: omitting all of these reproduces M1 behavior exactly.

### Metadata changes over M1

The M1 RFC already carries per-block source-tier hints (`load_cpu_tiers: list[int]` in the worker metadata). This proposal reuses them:

- **`load_cpu_tiers`** becomes the list of **valid source tiers** for a block (not just one primary). Worker picks in order of preference; on timeout it advances to the next.
- New per-block field: `replica_tiers: set[int]` — the tiers the block is known to exist in (for `replicate` / `hybrid`). Populated on store completion, consulted on read for failover eligibility.

No change to the block-hash encoding: exclusive placement is still the invariant within a single mode's perspective, but replicated modes track "which tiers hold this hash" as metadata sidecar, not as duplicate entries in `cached_block_hash_to_block`.

## Implementation sketch

Not scoped here, but to show the work is contained:

1. **Add `placement_mode` plumbing** through `SimpleCPUOffloadConnector.__init__` to `SimpleCPUOffloadScheduler` and `SimpleCPUOffloadWorker`.
2. **Replace `FastTierBlockPool`'s single demotion path** with a mode-dispatch: `exclusive` keeps today's demote-on-evict; `hybrid` adds an async mirror on store; `replicate` stores to both unconditionally.
3. **Add `TierHealth` state** to `CpuTier` — a small state machine with counters, drained on each `build_connector_meta`.
4. **Wire `timeout_ms` through `DmaCopyBackend.launch_copy`** and expose a failure callback to the worker, which feeds the `TierHealth` state machine.
5. **Add the failover read path** in `SimpleCPUOffloadWorker.get_finished`: on timeout, check `replica_tiers`, resubmit on a surviving tier.
6. **Add a bounded re-replication scheduler** — a new low-priority event type, rate-limited by `replication_rate_limit_mb_s`.

## Risks and open questions

- **Timeout tuning is hard**: too tight and healthy-but-slow tiers get quarantined under load; too loose and failures aren't detected fast enough to matter. First cut should be conservative + configurable, with telemetry to tune in production.
- **Silent corruption is not covered**: this proposal defends against unresponsiveness, not wrong data. Checksumming per block is a materially larger project and probably a separate RFC.
- **Split-brain during re-replication**: if a tier is quarantined, serves reads anyway, then returns and has stale data — we need to version blocks or treat every recovery as "wipe + re-replicate from survivor." The simpler option is the latter; worth confirming before implementation.
- **Auto-heal vs manual re-enable**: automatic recovery detection risks flapping. Manual re-enable is safer but operationally worse. Decision can be deferred to implementation time.
- **Cold-restart recovery**: arguably should be done **first** since it's simpler and provides most of the value for planned maintenance. TBD whether to split this proposal into "resiliency-cold" and "resiliency-hot" mini-RFCs.

## Relationship to M1

M1 lands no resiliency code. What M1 **must not foreclose** — already honored in the M1 RFC's design:

- `CpuTier` is a **symmetric abstraction** — no "fast"-vs-"slow" asymmetry leaks into the code paths that `placement_mode` will later flip. The only asymmetry (demotion direction) lives inside `FastTierBlockPool`, which is easy to replace per mode.
- Worker metadata already carries per-block source-tier hints (`load_cpu_tiers`). That field generalizes to "valid tiers in preference order" without a schema change.
- `DmaCopyBackend.launch_copy` has no `timeout_ms` yet. M1 leaves the signature alone; this proposal adds the parameter as an optional kwarg.

If M1 review uncovers anything that would constrain this proposal, we update both RFCs together.
