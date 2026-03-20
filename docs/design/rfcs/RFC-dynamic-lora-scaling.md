# RFC: Dynamic LoRA Slot Scaling for vLLM

> **Status: SUPERSEDED**
>
> This RFC has been merged with
> [RFC: Dynamic LoRA GPU Capacity](./dynamic-lora-gpu-capacity.md),
> which is the authoritative document going forward.
>
> See the [Differences and Merge Rationale](#differences-and-merge-rationale) section
> below for a full comparison of what each RFC contributed and how conflicts were resolved.

---

## Original Summary

This RFC proposed a mechanism to dynamically scale the number of concurrent LoRA adapter slots
(`max_loras`) in a running vLLM instance without requiring a server restart, delivered as a
standalone vLLM plugin (`vllm-lora-scaler`) that exposes a REST endpoint
`POST /v1/scale_max_loras`.

**Authors:** Yue Zhu ([@yuezhu1](https://github.com/yuezhu1))
**Created:** 2026-03-19

---

## Differences and Merge Rationale

### Comparison Table

| Aspect | This RFC (yuezhu1) | Chen Wang's RFC (wangchen615) | Resolution in Merged RFC |
| --- | --- | --- | --- |
| **Scaling trigger** | Operator REST call `POST /v1/scale_max_loras` | Automatic: memory watermarks + `LoRAMemoryNotifier` from kvcached | Both modes included; share one core `resize_lora_slots()` mechanism |
| **Delivery model** | Standalone plugin (`vllm-lora-scaler`) using public vLLM extension points only | Core vLLM changes (new config fields, engine hook, worker RPC) | Core carries the mechanism; plugin delivers the REST endpoint |
| **Pause strategy** | `pause_generation(mode="abort")` + `resume_generation()` wraps the resize | Between-batch only (post-batch hook) | Both: automatic uses batch boundary; operator REST uses pause/resume |
| **Scale-down eviction** | Full eviction via `remove_all_adapters()` then reload on demand | LRU eviction of only the minimum necessary adapters | LRU eviction adopted (more efficient; avoids reloading hot adapters) |
| **TP/PP coordination** | `collective_rpc` (noted PP works without extra sync) | `collective_rpc` for TP; PP scoped out | Same; PP deferred as follow-up in merged RFC |
| **Memory pressure signal** | Not addressed | `LoRAMemoryNotifier` interface for kvcached integration | Included in merged RFC |
| **In-place reloading** | Not addressed | `load_inplace=True` on `LoRARequest` for hot-swap without slot realloc | Included in merged RFC |
| **Config bounds** | No floor/ceiling | `min_loras` / `max_loras` as floor/ceiling | Included in merged RFC |
| **Resize throttling** | Not addressed | `lora_slot_resize_cooldown_s` config field | Included in merged RFC |
| **CudaGraph handling** | Not addressed | Disable LoRA cudagraph specialization when `dynamic_lora_slots=True` | Included in merged RFC |
| **Upstream strategy** | Plugin first, upstream later | Core change proposed directly | Merged RFC adopts plugin-first for REST endpoint; core changes for mechanism |

### Key Conflicts Resolved

#### 1. Full eviction vs. LRU eviction on scale-down

This RFC used `remove_all_adapters()` (evict everything, reload on demand), matching the
approach used by vLLM's sleep/wake feature. Chen Wang's RFC used LRU eviction of only the
minimum necessary adapters to free slots.

*Resolution*: LRU eviction was adopted in the merged RFC. It preserves hot adapters in GPU
slots when scaling down, avoiding unnecessary reload latency for popular LoRAs. The safety
argument for full eviction still holds (no dangling slot indices), but LRU is strictly more
efficient.

#### 2. Plugin-only vs. core change

This RFC avoided core vLLM changes entirely, using monkey-patching and `__bases__` mutation
to inject functionality. Chen Wang's RFC required modifying `LoRAConfig`, engine internals,
and worker classes.

*Resolution*: The merged RFC adopts a hybrid: core vLLM carries the `resize_lora_slots()`
mechanism (clean, reviewable, no monkey-patching), while the `POST /v1/scale_max_loras`
REST endpoint remains a plugin using `pause_generation` / `collective_rpc`. This gives the
operator-triggered path a faster shipping path while the automatic scaling path goes through
proper core review.

#### 3. Automatic vs. operator-triggered scaling

The two RFCs addressed different triggering modes and were not in conflict — they were
complementary. The merged RFC includes both modes explicitly.

---

## See Also

- **[RFC: Dynamic LoRA GPU Capacity](./dynamic-lora-gpu-capacity.md)** — The merged,
  authoritative RFC incorporating all features from both proposals.
