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
