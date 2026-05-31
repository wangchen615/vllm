# Secondary memory system — system overview & emulator design

| | |
| --- | --- |
| **Status** | Draft |
| **Branch** | `dev` |
| **Owner** | @wangchen615 |
| **Created** | 2026-05-20 |
| **Last revised** | 2026-05-30 (rebased onto `OffloadingConnector`; cross-deployment recovery now in M1) |
| **Related** | [Exclusive tiered caching (M1)](exclusive-tiered-caching.md), [Resiliency & advanced placement modes](secondary-memory-resiliency.md) |

This document describes the **production hardware target** for a three-level KV-cache memory hierarchy and the **vLLM emulator** that lets us iterate on placement, resiliency, and observability without that hardware in hand.

It is the entry point to a small family of sibling RFCs:

- This doc — system + emulator overview.
- [Exclusive tiered caching](exclusive-tiered-caching.md) — the M1 placement model (partitioned by request priority) + the **Host Memory Pool Manager** that mediates between two vLLM deployments sharing one host pool.
- [Resiliency & advanced placement modes](secondary-memory-resiliency.md) — `inclusive`, `hybrid`, and `replicate` placement modes, hot-failure detection, and the externally-managed read-only tier (model-sharing variant).

## Terminology

The diagrams and text in this RFC family use neutral names so the design stays portable across hardware platforms and avoids any single product name:

- **Secondary fast memory system** — the novel middle tier in the production hierarchy.
- **Accelerator** — any device that runs the model and owns HBM (GPU, NPU, or other AI accelerator).
- **Medium** — the connector-side identifier for a tier (`"GPU"`, `"FAST_CPU"`, `"SLOW_CPU"`). Comes from `LoadStoreSpec.medium()` in the existing `OffloadingConnector` API and is what we extend rather than fork.
- **Host Memory Pool Manager** (HMPM) — an external component (process or shared library) that owns the shared host-memory region and arbitrates allocation, eviction, and cross-deployment access between multiple vLLM instances. New in M1; details in the [exclusive-tiered RFC](exclusive-tiered-caching.md).

## 1. The production target

The production deployment pictured below is what the emulator in this repo is meant to stand in for. The middle box — the **secondary fast memory system** — is the novel piece. It sits closer to the accelerators than host DRAM (typically across a PCIe / CXL switch on the same pod), is *larger* than the accelerators' HBM, and is *faster* to reach than host DRAM. It is the place where the system can keep the tail of hot KV blocks that no longer fit in HBM.

<img alt="Production hardware target with PCIe-switched secondary fast memory; multiple accelerator deployments share one secondary pool through the Host Memory Pool Manager" src="imgs/svg/system-overview.svg" width="780">

Source: [`imgs/mmd/system-overview.mmd`](imgs/mmd/system-overview.mmd) — edit this file and re-render to update the diagram.

### What changes vs. a single-CPU-pool offload

A conventional vLLM offload connector sees one CPU pool, owned and lifecycled by one vLLM instance. Three things change in the production target:

1. There are now **two address spaces below HBM**, with very different latency / bandwidth profiles. The secondary fast memory tier is reachable from the accelerator without crossing the host-DRAM path; the slow tier is normal pinned host DRAM.
2. The **placement policy** between those two address spaces is a first-class design choice — not just "evict from fast, demote to slow." Four placement modes are anticipated in the RFC family: `partitioned` (M1, [exclusive tiered](exclusive-tiered-caching.md)) and `inclusive` / `hybrid` / `replicate` ([resiliency RFC](secondary-memory-resiliency.md)). Each one decides differently which medium holds which block and whether a block can live in more than one medium.
3. The **host pool can be shared across vLLM deployments**. When one accelerator deployment loses its HBM (hardware failure, planned maintenance, or rolling restart), a surviving deployment can prefetch its offloaded blocks from the shared host pool and resume in-flight requests without a cold re-prefill. That requires an arbitration component above the per-instance offload connector — the **Host Memory Pool Manager**.

The combination of two physical address spaces *and* a real placement-policy choice *and* a multi-deployment lifecycle is what motivates a new design rather than a small extension to an existing connector.

### Why we build on `OffloadingConnector`, not `SimpleCPUOffloadConnector`

Earlier drafts of this RFC family proposed forking `SimpleCPUOffloadConnector`. That choice has been reversed. The upstream `OffloadingConnector` (`vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py`, with abstractions in `vllm/v1/kv_offload/`) already provides the primitives we would otherwise be inventing:

| Concept this RFC family needs | Where it already lives in `OffloadingConnector` |
| --- | --- |
| Symmetric per-tier abstraction | `OffloadingManager` (`vllm/v1/kv_offload/base.py`) — already keyed on `LoadStoreSpec.medium()` |
| Pluggable eviction policy | `vllm/v1/kv_offload/cpu/policies/` — `_CACHE_POLICIES = {"lru": ..., "arc": ...}` registry in `cpu/manager.py:19-22` |
| Manager composition / wrapping | `FilterReusedOffloadingManager` in `vllm/v1/kv_offload/reuse_manager.py` is already a wrapper-pattern manager that delegates to a backing manager |
| Per-medium events | `OffloadingEvent.medium` (`base.py:75-80`) |
| Connector registration | `OffloadingSpecFactory.register_spec(name, module_path, class_name)` in `factory.py` — register a new spec via `extra_config.spec_name`, no connector fork |
| Worker handler dispatch | `OffloadingSpec.get_handlers()` yields `(src_type, dst_type, handler)` tuples — a two-medium spec yields four handlers (GPU↔fast, GPU↔slow), and the existing worker dispatches automatically |

Concretely, **M1 ships as a new `OffloadingSpec` registered through `OffloadingSpecFactory`, not as a connector fork**. The seam is the spec name in `kv_connector_extra_config.spec_name`; existing `cpu_bytes_to_use` / `eviction_policy` knobs survive and are extended.

## 2. The emulator

vLLM does not yet have access to the production hardware in CI or in dev clusters, so M1 emulates the three-level hierarchy with **one mmap-backed shared host-memory pool, surfaced to vLLM as two media** (`FAST_CPU` and `SLOW_CPU`) and arbitrated by the Host Memory Pool Manager.

<img alt="One shared mmap-backed host memory pool, surfaced to vLLM as FAST_CPU and SLOW_CPU media; the Host Memory Pool Manager owns the pool and lets multiple vLLM instances attach to it" src="imgs/svg/emulator-overview.svg" width="780">

Source: [`imgs/mmd/emulator-overview.mmd`](imgs/mmd/emulator-overview.mmd).

### Why one shared pool and not two private pools

The earlier draft of this RFC family used two private pinned-DRAM pools, each owned by one vLLM instance, as stand-ins for the secondary fast tier and the slow tier. That framing has been replaced for one concrete reason: it cannot model cross-deployment recovery, which is now in M1's scope.

vLLM's existing `SharedOffloadRegion` (`vllm/v1/kv_offload/cpu/shared_offload_region.py`) is explicitly bound to a single vLLM instance:

- The mmap path is keyed by instance id: `/dev/shm/vllm_offload_{instance_id}.mmap` (`shared_offload_region.py:50`).
- The first worker `O_EXCL`-creates the file (`:62-67`); other instances cannot share it without colliding.
- The creator `unlink`s the file on shutdown (`:184-191`); a second instance attached to the same file would lose its memory when the first instance shuts down.

Making that file shareable across vLLM deployments requires inverting the lifecycle so the file is *not* owned by any one vLLM process. That inversion **is** the Host Memory Pool Manager. M1 introduces it as an external component (process or shared library) that owns the pool, lets vLLM instances attach and detach, arbitrates allocation and eviction across them, and gives a survivor instance a way to read blocks left behind by a failed peer.

### What the emulator proves and does not prove

The emulator deliberately uses pinned host DRAM for the whole shared pool, so there is **no real latency asymmetry** between `FAST_CPU` and `SLOW_CPU` in M1. That is by design: M1 is a functional-correctness milestone, not a performance milestone. What it has to demonstrate is:

- vLLM can manage **two medium-keyed views of one shared pool** through `OffloadingConnector` with small, scoped changes — specifically a new `OffloadingSpec` and a multi-medium `OffloadingManager`.
- A **placement-policy abstraction** sits cleanly above the medium dispatch, so future modes (`inclusive`, `hybrid`, `replicate`) can be added without re-plumbing the worker.
- The **Host Memory Pool Manager** can mediate between two vLLM instances on one host: one instance writes blocks under a request id, the other reads them via the manager and serves a request without re-prefill.
- The completion + metadata wiring (events, fence semantics, per-medium counters) survives going from one medium to two and from one instance to two.

What it explicitly does **not** prove (and is not expected to) — capacity additivity, prefix-miss speedup, real resiliency under hardware failure, or model-sharing throughput. Those depend either on a real secondary-memory backing or on placement modes outside M1's scope. They are tracked in the resiliency RFC.

## 3. How the sibling RFCs fit together

```text
   secondary-memory-system-overview.md   ← this doc
        ├── exclusive-tiered-caching.md         ← M1 component design (partitioned + HMPM)
        │     └── secondary-memory-m1-implementation.md  ← M1 implementation plan
        └── secondary-memory-resiliency.md      ← inclusive / hybrid / replicate, hot failure,
                                                   externally-managed read-only tier
```

- The **exclusive-tiered RFC** describes M1 at a component-design level: the `partitioned`-by-priority placement mode, the `OffloadingSpec` shape, and the Host Memory Pool Manager's API surface and ownership boundaries with vLLM.
- The **M1 implementation plan** is the concrete file-by-file edit list. It points at `vllm/v1/kv_offload/cpu/spec.py` and friends, plus the new HMPM component. Read the component RFC for *what* the system does; read the M1 plan for *which files change*.
- The **resiliency RFC** owns the three non-M1 placement modes and the failure-detection / failover / re-replication mechanisms. The previously-separate "inclusive hierarchical caching" doc has been folded into this RFC as one of three placement modes alongside `hybrid` and `replicate`. It is the natural follow-on after M1.

## 4. Out of scope for this overview

- Concrete capacity sizing for the secondary fast memory tier — workload-dependent, will follow once a real backing is wired up.
- Real-hardware drivers / kernel changes — outside vLLM.
- Multi-host pools — M1's HMPM scopes to one physical host. Cross-host pools are a follow-on once single-host cross-deployment recovery is proven.
- Observability / metrics — to be added per-medium after the functional milestone lands.
