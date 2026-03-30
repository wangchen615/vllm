# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for LoRAModelManager.resize_lora_slots().

Tests run without GPU — all LoRA layer operations are mocked.
"""

from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import torch

from vllm.config.lora import LoRAConfig
from vllm.lora.model_manager import (
    LoRALRUCache,
    LoRAModelManager,
    LRUCacheLoRAModelManager,
)

# Initial slot count used by all manager fixtures.
# Shrink tests target 2 (below); grow tests target 8 (above).
INITIAL_SLOTS = 4


def _lora_config(max_loras: int = INITIAL_SLOTS) -> LoRAConfig:
    return LoRAConfig(max_loras=max_loras, max_lora_rank=8)


def _make_mock_module() -> MagicMock:
    m = MagicMock()
    m.reallocate_lora_weights = MagicMock()
    return m


def _make_base_manager(n_modules: int = 2) -> LoRAModelManager:
    """LoRAModelManager with mocked model and modules."""
    manager = MagicMock(spec=LoRAModelManager)
    manager.lora_config = _lora_config()
    manager._lora_slots = INITIAL_SLOTS
    manager.lora_index_to_id = [None] * INITIAL_SLOTS
    manager._active_adapters = {}
    manager.modules = {f"mod_{i}": _make_mock_module() for i in range(n_modules)}
    # lora_slots is a property backed by _lora_slots; wire it up
    type(manager).lora_slots = PropertyMock(
        side_effect=lambda self=manager: self._lora_slots
    )
    manager._evict_adapters_to_fit = LoRAModelManager._evict_adapters_to_fit.__get__(
        manager
    )
    manager._compact_slots = LoRAModelManager._compact_slots.__get__(manager)
    manager.resize_lora_slots = LoRAModelManager.resize_lora_slots.__get__(manager)
    return manager


def _make_lru_manager(n_modules: int = 2) -> LRUCacheLoRAModelManager:
    """LRUCacheLoRAModelManager with mocked model and modules."""
    manager = MagicMock(spec=LRUCacheLoRAModelManager)
    manager.lora_config = _lora_config()
    manager._lora_slots = INITIAL_SLOTS
    manager.lora_index_to_id = [None] * INITIAL_SLOTS
    manager.modules = {f"mod_{i}": _make_mock_module() for i in range(n_modules)}
    manager._deactivate_adapter = LoRAModelManager._deactivate_adapter.__get__(manager)
    manager._active_adapters = LoRALRUCache(INITIAL_SLOTS, manager._deactivate_adapter)
    # lora_slots is a property backed by _lora_slots; wire it up
    type(manager).lora_slots = PropertyMock(
        side_effect=lambda self=manager: self._lora_slots
    )
    manager._evict_adapters_to_fit = (
        LRUCacheLoRAModelManager._evict_adapters_to_fit.__get__(manager)
    )
    manager._compact_slots = LoRAModelManager._compact_slots.__get__(manager)
    manager.resize_lora_slots = LoRAModelManager.resize_lora_slots.__get__(manager)
    return manager


def test_raises_on_zero_slots():
    manager = _make_base_manager()
    with pytest.raises(ValueError, match="must be >= 1"):
        manager.resize_lora_slots(0)


def test_noop_when_size_unchanged():
    manager = _make_base_manager()
    with patch("torch.accelerator.empty_cache") as mock_empty:
        manager.resize_lora_slots(INITIAL_SLOTS)
    mock_empty.assert_not_called()
    for mod in manager.modules.values():
        mod.reallocate_lora_weights.assert_not_called()
    assert manager._lora_slots == INITIAL_SLOTS


def test_grow_adds_empty_slots():
    manager = _make_base_manager()
    manager.lora_index_to_id = [10, None, None, None]
    with patch("torch.accelerator.empty_cache"):
        manager.resize_lora_slots(8)
    # _lora_slots updated, lora_config.max_loras unchanged
    assert manager._lora_slots == 8
    assert manager.lora_config.max_loras == INITIAL_SLOTS
    # lora_index_to_id extended with None, existing entries preserved
    assert len(manager.lora_index_to_id) == 8
    assert manager.lora_index_to_id[0] == 10
    assert all(v is None for v in manager.lora_index_to_id[1:])
    # reallocate called on every module
    for mod in manager.modules.values():
        mod.reallocate_lora_weights.assert_called_once_with(8)


def test_shrink_evicts_lru_adapters():
    manager = _make_lru_manager()
    # Add adapters in order: 1 (oldest) → 4 (newest)
    for i in range(1, 5):
        manager._active_adapters[i] = None
    # Survivors 3 and 4 are in high slots; compaction must move them to 0, 1
    manager.lora_index_to_id = [1, 2, 3, 4]
    with patch("torch.accelerator.empty_cache"):
        manager.resize_lora_slots(2)
    # Oldest two (1, 2) evicted; newest two (3, 4) survive
    assert 1 not in manager._active_adapters
    assert 2 not in manager._active_adapters
    assert 3 in manager._active_adapters
    assert 4 in manager._active_adapters
    # cache rebuilt with new capacity, _lora_slots updated
    assert manager._active_adapters.capacity == 2
    assert manager._lora_slots == 2
    assert len(manager.lora_index_to_id) == 2
    # Survivors compacted into low slots
    assert manager.lora_index_to_id == [3, 4]
    for mod in manager.modules.values():
        mod.reallocate_lora_weights.assert_called_once_with(2)


def test_base_shrink_raises_when_active_adapters_overflow():
    manager = _make_base_manager()
    manager._active_adapters = {1: None, 2: None, 3: None}
    with pytest.raises(ValueError, match="3 adapters are currently active"):
        manager.resize_lora_slots(2)


def test_empty_cache_called_once():
    manager = _make_base_manager(n_modules=4)
    with patch("torch.accelerator.empty_cache") as mock_empty:
        manager.resize_lora_slots(8)
    mock_empty.assert_called_once()
