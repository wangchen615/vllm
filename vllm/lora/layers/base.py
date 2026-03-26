# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, overload

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config.lora import LoRAConfig

if TYPE_CHECKING:
    from vllm.lora.punica_wrapper import PunicaWrapperBase


class BaseLayerWithLoRA(nn.Module):
    @overload
    def slice_lora_a(
        self, lora_a: list[torch.Tensor | None]
    ) -> list[torch.Tensor | None]: ...
    @overload
    def slice_lora_a(self, lora_a: torch.Tensor) -> torch.Tensor: ...
    def slice_lora_a(
        self, lora_a: torch.Tensor | list[torch.Tensor | None]
    ) -> torch.Tensor | list[torch.Tensor | None]:
        """Slice lora a if splitting for tensor parallelism."""
        ...

    @overload
    def slice_lora_b(
        self, lora_b: list[torch.Tensor | None]
    ) -> list[torch.Tensor | None]: ...
    @overload
    def slice_lora_b(self, lora_b: torch.Tensor) -> torch.Tensor: ...
    def slice_lora_b(
        self, lora_b: torch.Tensor | list[torch.Tensor | None]
    ) -> torch.Tensor | list[torch.Tensor | None]:
        """Slice lora b if splitting with tensor parallelism."""
        ...

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: PretrainedConfig | None = None,
    ) -> None:
        """Initializes lora matrices."""
        ...

    def reset_lora(self, index: int):
        """Resets the lora weights at index back to 0."""
        ...

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor | list[torch.Tensor],
        lora_b: torch.Tensor | list[torch.Tensor],
    ):
        """Overwrites lora tensors at index."""
        ...

    def set_mapping(
        self,
        punica_wrapper,
    ):
        self.punica_wrapper: PunicaWrapperBase = punica_wrapper

    def reallocate_lora_weights(self, new_slots: int) -> None:
        """Reallocate lora_a_stacked / lora_b_stacked for new_slots.

        - Preserves weights for surviving slots (min(old, new) slots).
        - No-op if layer has no lora tensors.
        - Does NOT call empty_cache() — caller does this once after all layers.
        """
        lora_a_stacked: torch.Tensor | tuple[torch.Tensor, ...] | None = getattr(
            self, "lora_a_stacked", None
        )
        lora_b_stacked: torch.Tensor | tuple[torch.Tensor, ...] | None = getattr(
            self, "lora_b_stacked", None
        )
        if lora_a_stacked is None or lora_b_stacked is None:
            return

        def _reallocate(
            stacked: torch.Tensor | tuple[torch.Tensor, ...],
        ) -> torch.Tensor | tuple[torch.Tensor, ...]:
            is_tuple = isinstance(stacked, tuple)
            tensors = stacked if is_tuple else (stacked,)
            surviving = min(tensors[0].shape[0], new_slots)
            new_tensors = tuple(
                torch.zeros(new_slots, *t.shape[1:], dtype=t.dtype, device=t.device)
                for t in tensors
            )
            for new_t, old_t in zip(new_tensors, tensors):
                new_t[:surviving].copy_(old_t[:surviving])
            return new_tensors if is_tuple else new_tensors[0]

        object.__setattr__(self, "lora_a_stacked", _reallocate(lora_a_stacked))
        object.__setattr__(self, "lora_b_stacked", _reallocate(lora_b_stacked))

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        """Returns True if the layer can be replaced by this LoRA layer."""
        raise NotImplementedError
