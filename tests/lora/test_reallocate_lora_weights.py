# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for reallocate_lora_weights() across all LoRA layer types.

Tests run on CPU — no GPU required.
"""

import pytest
import torch

from vllm.config.lora import LoRAConfig
from vllm.lora.layers import (
    ColumnParallelLinearWithLoRA,
    LogitsProcessorWithLoRA,
    MergedColumnParallelLinearWithLoRA,
    RowParallelLinearWithLoRA,
)
from vllm.lora.layers.fused_moe import FusedMoEWithLoRA
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor

DEVICE = "cpu"
MAX_LORAS = 4
MAX_LORA_RANK = 8
HIDDEN_SIZE = 32
OUTPUT_SIZE = 16
VOCAB_SIZE = 64

# FusedMoE-specific constants
MOE_NUM_EXPERTS = 4
MOE_TOP_K = 2
MOE_HIDDEN_SIZE = 16
MOE_INTERMEDIATE_SIZE = 32


def _lora_config() -> LoRAConfig:
    return LoRAConfig(
        max_loras=MAX_LORAS,
        max_lora_rank=MAX_LORA_RANK,
        lora_dtype=torch.float32,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_column_parallel_layer() -> ColumnParallelLinearWithLoRA:
    base = ColumnParallelLinear(HIDDEN_SIZE, OUTPUT_SIZE, bias=False)
    layer = ColumnParallelLinearWithLoRA(base)
    layer.create_lora_weights(MAX_LORAS, _lora_config())
    return layer


def _make_row_parallel_layer() -> RowParallelLinearWithLoRA:
    base = RowParallelLinear(HIDDEN_SIZE, OUTPUT_SIZE, bias=False)
    layer = RowParallelLinearWithLoRA(base)
    layer.create_lora_weights(MAX_LORAS, _lora_config())
    return layer


def _make_merged_column_layer() -> MergedColumnParallelLinearWithLoRA:
    base = MergedColumnParallelLinear(
        HIDDEN_SIZE, [OUTPUT_SIZE, OUTPUT_SIZE], bias=False
    )
    layer = MergedColumnParallelLinearWithLoRA(base)
    layer.create_lora_weights(MAX_LORAS, _lora_config())
    return layer


def _make_logits_processor_layer() -> LogitsProcessorWithLoRA:
    base = LogitsProcessor(VOCAB_SIZE, VOCAB_SIZE)
    layer = LogitsProcessorWithLoRA(
        base,
        hidden_size=HIDDEN_SIZE,
        dtype=torch.float32,
        device=torch.device(DEVICE),
        sharded_to_full_mapping=None,
    )
    layer.create_lora_weights(MAX_LORAS, _lora_config())
    return layer


def _fill_slots(layer, n_slots: int):
    """Fill first n_slots of lora_a_stacked with recognizable values."""
    stacked = layer.lora_a_stacked
    if isinstance(stacked, tuple):
        for t in stacked:
            for i in range(n_slots):
                t[i].fill_(float(i + 1))
    else:
        for i in range(n_slots):
            stacked[i].fill_(float(i + 1))


def _check_weights_preserved(layer, surviving: int):
    """Assert surviving slots have the expected fill values."""
    stacked = layer.lora_a_stacked
    tensors = stacked if isinstance(stacked, tuple) else (stacked,)
    for t in tensors:
        for i in range(surviving):
            assert t[i].unique().item() == float(i + 1), (
                f"slot {i} weight not preserved"
            )


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_layer",
    [
        _make_column_parallel_layer,
        _make_row_parallel_layer,
        _make_merged_column_layer,
        _make_logits_processor_layer,
    ],
    ids=["column_parallel", "row_parallel", "merged_column", "logits_processor"],
)
def test_scale_up_shape(dist_init, make_layer):
    layer = make_layer()
    new_slots = MAX_LORAS * 2
    layer.reallocate_lora_weights(new_slots)
    stacked = layer.lora_a_stacked
    tensors = stacked if isinstance(stacked, tuple) else (stacked,)
    assert tensors[0].shape[0] == new_slots


@pytest.mark.parametrize(
    "make_layer",
    [
        _make_column_parallel_layer,
        _make_row_parallel_layer,
        _make_merged_column_layer,
        _make_logits_processor_layer,
    ],
    ids=["column_parallel", "row_parallel", "merged_column", "logits_processor"],
)
def test_scale_down_shape(dist_init, make_layer):
    layer = make_layer()
    new_slots = MAX_LORAS // 2
    layer.reallocate_lora_weights(new_slots)
    stacked = layer.lora_a_stacked
    tensors = stacked if isinstance(stacked, tuple) else (stacked,)
    assert tensors[0].shape[0] == new_slots


# ---------------------------------------------------------------------------
# Weight preservation tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_layer",
    [
        _make_column_parallel_layer,
        _make_row_parallel_layer,
        _make_merged_column_layer,
        _make_logits_processor_layer,
    ],
    ids=["column_parallel", "row_parallel", "merged_column", "logits_processor"],
)
def test_scale_up_preserves_weights(dist_init, make_layer):
    layer = make_layer()
    _fill_slots(layer, MAX_LORAS)
    layer.reallocate_lora_weights(MAX_LORAS * 2)
    _check_weights_preserved(layer, MAX_LORAS)


@pytest.mark.parametrize(
    "make_layer",
    [
        _make_column_parallel_layer,
        _make_row_parallel_layer,
        _make_merged_column_layer,
        _make_logits_processor_layer,
    ],
    ids=["column_parallel", "row_parallel", "merged_column", "logits_processor"],
)
def test_scale_down_preserves_surviving_weights(dist_init, make_layer):
    layer = make_layer()
    new_slots = MAX_LORAS // 2
    _fill_slots(layer, MAX_LORAS)
    layer.reallocate_lora_weights(new_slots)
    _check_weights_preserved(layer, new_slots)


# ---------------------------------------------------------------------------
# New slots are zeroed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_layer",
    [
        _make_column_parallel_layer,
        _make_row_parallel_layer,
        _make_merged_column_layer,
        _make_logits_processor_layer,
    ],
    ids=["column_parallel", "row_parallel", "merged_column", "logits_processor"],
)
def test_scale_up_new_slots_are_zero(dist_init, make_layer):
    layer = make_layer()
    _fill_slots(layer, MAX_LORAS)
    new_slots = MAX_LORAS * 2
    layer.reallocate_lora_weights(new_slots)
    stacked = layer.lora_a_stacked
    tensors = stacked if isinstance(stacked, tuple) else (stacked,)
    for t in tensors:
        assert t[MAX_LORAS:].eq(0).all(), "new slots should be zero-initialised"


# ---------------------------------------------------------------------------
# No-op when lora tensors not initialised
# ---------------------------------------------------------------------------


def test_noop_before_create_lora_weights(dist_init):
    base = ColumnParallelLinear(HIDDEN_SIZE, OUTPUT_SIZE, bias=False)
    layer = ColumnParallelLinearWithLoRA(base)
    # Should not raise
    layer.reallocate_lora_weights(8)


# ---------------------------------------------------------------------------
# empty_cache not called
# ---------------------------------------------------------------------------


def test_no_empty_cache_called(dist_init, monkeypatch):
    called = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: called.append(1))
    layer = _make_column_parallel_layer()
    layer.reallocate_lora_weights(MAX_LORAS * 2)
    assert not called, "empty_cache should not be called inside reallocate_lora_weights"


# ---------------------------------------------------------------------------
# FusedMoEWithLoRA helpers and tests
# ---------------------------------------------------------------------------


def _make_fused_moe_layer() -> FusedMoEWithLoRA:
    base = FusedMoE(
        num_experts=MOE_NUM_EXPERTS,
        top_k=MOE_TOP_K,
        hidden_size=MOE_HIDDEN_SIZE,
        intermediate_size=MOE_INTERMEDIATE_SIZE,
        params_dtype=torch.float32,
    )
    layer = FusedMoEWithLoRA(base)
    layer.create_lora_weights(MAX_LORAS, _lora_config())
    return layer


def _flat_list_expected_len(layer: FusedMoEWithLoRA, n_slots: int) -> int:
    """Expected length of lora_a_stacked / lora_b_stacked flat list."""
    # per slot: 2 entries (w13 + w2) per expert, plus 1 extra w13 if gated
    per_expert = 2 + (1 if layer._w13_slices == 2 else 0)
    return n_slots * layer.base_layer.local_num_experts * per_expert


def test_fused_moe_scale_up_shape(dist_init):
    layer = _make_fused_moe_layer()
    new_slots = MAX_LORAS * 2
    layer.reallocate_lora_weights(new_slots)
    assert layer.w13_lora_a_stacked[0].shape[0] == new_slots
    assert layer.w2_lora_a_stacked[0].shape[0] == new_slots
    assert layer.w13_lora_b_stacked[0].shape[0] == new_slots
    assert layer.w2_lora_b_stacked[0].shape[0] == new_slots


def test_fused_moe_scale_down_shape(dist_init):
    layer = _make_fused_moe_layer()
    new_slots = MAX_LORAS // 2
    layer.reallocate_lora_weights(new_slots)
    assert layer.w13_lora_a_stacked[0].shape[0] == new_slots
    assert layer.w2_lora_a_stacked[0].shape[0] == new_slots


def test_fused_moe_scale_up_preserves_weights(dist_init):
    layer = _make_fused_moe_layer()
    # Write recognizable values into the first MAX_LORAS slots
    for i in range(MAX_LORAS):
        layer.w13_lora_a_stacked[0][i].fill_(float(i + 1))
        layer.w2_lora_a_stacked[0][i].fill_(float(i + 1))
    layer.reallocate_lora_weights(MAX_LORAS * 2)
    for i in range(MAX_LORAS):
        assert layer.w13_lora_a_stacked[0][i].unique().item() == float(i + 1), (
            f"w13_lora_a slot {i} not preserved after scale-up"
        )
        assert layer.w2_lora_a_stacked[0][i].unique().item() == float(i + 1), (
            f"w2_lora_a slot {i} not preserved after scale-up"
        )


def test_fused_moe_scale_down_preserves_surviving_weights(dist_init):
    layer = _make_fused_moe_layer()
    new_slots = MAX_LORAS // 2
    for i in range(MAX_LORAS):
        layer.w13_lora_a_stacked[0][i].fill_(float(i + 1))
        layer.w2_lora_a_stacked[0][i].fill_(float(i + 1))
    layer.reallocate_lora_weights(new_slots)
    for i in range(new_slots):
        assert layer.w13_lora_a_stacked[0][i].unique().item() == float(i + 1), (
            f"w13_lora_a slot {i} not preserved after scale-down"
        )
        assert layer.w2_lora_a_stacked[0][i].unique().item() == float(i + 1), (
            f"w2_lora_a slot {i} not preserved after scale-down"
        )


def test_fused_moe_scale_up_new_slots_are_zero(dist_init):
    layer = _make_fused_moe_layer()
    for i in range(MAX_LORAS):
        layer.w13_lora_a_stacked[0][i].fill_(float(i + 1))
    new_slots = MAX_LORAS * 2
    layer.reallocate_lora_weights(new_slots)
    assert layer.w13_lora_a_stacked[0][MAX_LORAS:].eq(0).all(), (
        "new slots in w13_lora_a should be zero-initialised"
    )
    assert layer.w2_lora_a_stacked[0][MAX_LORAS:].eq(0).all(), (
        "new slots in w2_lora_a should be zero-initialised"
    )


def test_fused_moe_adapter_enabled_length(dist_init):
    layer = _make_fused_moe_layer()
    new_slots = MAX_LORAS * 2
    layer.reallocate_lora_weights(new_slots)
    assert layer.adapter_enabled.shape[0] == new_slots + 1, (
        "adapter_enabled should have length new_slots + 1"
    )


def test_fused_moe_max_loras_updated(dist_init):
    layer = _make_fused_moe_layer()
    new_slots = MAX_LORAS * 2
    layer.reallocate_lora_weights(new_slots)
    assert layer.max_loras == new_slots


def test_fused_moe_flat_list_length_after_scale_up(dist_init):
    layer = _make_fused_moe_layer()
    new_slots = MAX_LORAS * 2
    layer.reallocate_lora_weights(new_slots)
    expected = _flat_list_expected_len(layer, new_slots)
    assert len(layer.lora_a_stacked) == expected, (
        f"lora_a_stacked length mismatch: {len(layer.lora_a_stacked)} != {expected}"
    )
    assert len(layer.lora_b_stacked) == expected, (
        f"lora_b_stacked length mismatch: {len(layer.lora_b_stacked)} != {expected}"
    )


def test_fused_moe_flat_list_length_after_scale_down(dist_init):
    layer = _make_fused_moe_layer()
    new_slots = MAX_LORAS // 2
    layer.reallocate_lora_weights(new_slots)
    expected = _flat_list_expected_len(layer, new_slots)
    assert len(layer.lora_a_stacked) == expected
    assert len(layer.lora_b_stacked) == expected


def test_fused_moe_noop_before_create_lora_weights(dist_init):
    base = FusedMoE(
        num_experts=MOE_NUM_EXPERTS,
        top_k=MOE_TOP_K,
        hidden_size=MOE_HIDDEN_SIZE,
        intermediate_size=MOE_INTERMEDIATE_SIZE,
        params_dtype=torch.float32,
    )
    layer = FusedMoEWithLoRA(base)
    # Should not raise — no weights initialised yet
    layer.reallocate_lora_weights(8)
