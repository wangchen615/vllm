# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for dynamic LoRA slot scaling fields added to LoRAConfig.

These tests cover the five new fields (min_loras, dynamic_lora_slots,
lora_mem_high_watermark, lora_mem_low_watermark, lora_slot_resize_cooldown_s)
and their validation logic. All tests run without GPU.
"""

import pytest
from pydantic import ValidationError

from vllm.config.lora import LoRAConfig

# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


def test_default_values():
    cfg = LoRAConfig(max_loras=4)
    assert cfg.dynamic_lora_slots is False
    assert cfg.min_loras == 1
    assert cfg.lora_mem_high_watermark == 0.8
    assert cfg.lora_mem_low_watermark == 0.5
    assert cfg.lora_slot_resize_cooldown_s == 1.0


def test_defaults_do_not_require_dynamic_flag():
    """New fields must have safe defaults so existing configs are unaffected."""
    cfg = LoRAConfig()
    assert cfg.dynamic_lora_slots is False


# ---------------------------------------------------------------------------
# Valid dynamic configurations
# ---------------------------------------------------------------------------


def test_valid_dynamic_config_defaults():
    cfg = LoRAConfig(max_loras=8, dynamic_lora_slots=True)
    assert cfg.min_loras == 1
    assert cfg.lora_mem_high_watermark == 0.8
    assert cfg.lora_mem_low_watermark == 0.5
    assert cfg.lora_slot_resize_cooldown_s == 1.0


def test_valid_dynamic_config_custom():
    cfg = LoRAConfig(
        max_loras=16,
        dynamic_lora_slots=True,
        min_loras=2,
        lora_mem_high_watermark=0.9,
        lora_mem_low_watermark=0.4,
        lora_slot_resize_cooldown_s=2.5,
    )
    assert cfg.min_loras == 2
    assert cfg.lora_mem_high_watermark == 0.9
    assert cfg.lora_mem_low_watermark == 0.4
    assert cfg.lora_slot_resize_cooldown_s == 2.5


def test_min_loras_equal_to_max_loras_is_valid():
    cfg = LoRAConfig(max_loras=4, dynamic_lora_slots=True, min_loras=4)
    assert cfg.min_loras == 4


def test_zero_cooldown_is_valid():
    cfg = LoRAConfig(
        max_loras=4, dynamic_lora_slots=True, lora_slot_resize_cooldown_s=0.0
    )
    assert cfg.lora_slot_resize_cooldown_s == 0.0


# ---------------------------------------------------------------------------
# Validation errors: dynamic_lora_slots=True
# ---------------------------------------------------------------------------


def test_min_loras_exceeds_max_loras_raises():
    with pytest.raises(ValueError, match="min_loras"):
        LoRAConfig(max_loras=2, dynamic_lora_slots=True, min_loras=5)


def test_inverted_watermarks_raises():
    with pytest.raises(ValueError, match="lora_mem_low_watermark"):
        LoRAConfig(
            max_loras=4,
            dynamic_lora_slots=True,
            lora_mem_low_watermark=0.9,
            lora_mem_high_watermark=0.5,
        )


def test_equal_watermarks_raises():
    with pytest.raises(ValueError, match="lora_mem_low_watermark"):
        LoRAConfig(
            max_loras=4,
            dynamic_lora_slots=True,
            lora_mem_low_watermark=0.7,
            lora_mem_high_watermark=0.7,
        )


# ---------------------------------------------------------------------------
# Validation errors: field-level (always enforced, regardless of flag)
# ---------------------------------------------------------------------------


def test_min_loras_zero_raises():
    with pytest.raises(ValidationError, match="min_loras"):
        LoRAConfig(max_loras=4, min_loras=0)


def test_negative_cooldown_raises():
    with pytest.raises(ValidationError, match="lora_slot_resize_cooldown_s"):
        LoRAConfig(max_loras=4, lora_slot_resize_cooldown_s=-1.0)


def test_high_watermark_at_boundary_raises():
    with pytest.raises(ValidationError, match="lora_mem_high_watermark"):
        LoRAConfig(max_loras=4, lora_mem_high_watermark=1.0)


def test_low_watermark_at_zero_raises():
    with pytest.raises(ValidationError, match="lora_mem_low_watermark"):
        LoRAConfig(max_loras=4, lora_mem_low_watermark=0.0)


# ---------------------------------------------------------------------------
# Validation skipped when dynamic_lora_slots=False
# ---------------------------------------------------------------------------


def test_min_loras_gt_max_loras_ok_when_flag_off():
    """Cross-field check only applies when dynamic_lora_slots=True."""
    cfg = LoRAConfig(max_loras=2, dynamic_lora_slots=False, min_loras=5)
    assert cfg.min_loras == 5


# ---------------------------------------------------------------------------
# compute_hash includes dynamic_lora_slots
# ---------------------------------------------------------------------------


def test_compute_hash_differs_with_dynamic_flag():
    cfg_static = LoRAConfig(max_loras=4, dynamic_lora_slots=False)
    cfg_dynamic = LoRAConfig(max_loras=4, dynamic_lora_slots=True)
    assert cfg_static.compute_hash() != cfg_dynamic.compute_hash()


def test_compute_hash_differs_with_specialize_active_lora():
    cfg_off = LoRAConfig(max_loras=4, specialize_active_lora=False)
    cfg_on = LoRAConfig(max_loras=4, specialize_active_lora=True)
    assert cfg_off.compute_hash() != cfg_on.compute_hash()


def test_compute_hash_stable():
    cfg = LoRAConfig(max_loras=4, dynamic_lora_slots=True)
    assert cfg.compute_hash() == cfg.compute_hash()
