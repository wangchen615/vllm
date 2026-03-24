# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any, Literal

import torch
from pydantic import ConfigDict, Field, model_validator
from typing_extensions import Self

from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.utils.hashing import safe_hash

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.config.cache import CacheConfig
else:
    ModelConfig = Any
    CacheConfig = Any

logger = init_logger(__name__)

LoRADType = Literal["auto", "float16", "bfloat16"]
MaxLoRARanks = Literal[1, 8, 16, 32, 64, 128, 256, 320, 512]
LoRAExtraVocabSize = Literal[256, 512]


@config(config=ConfigDict(arbitrary_types_allowed=True))
class LoRAConfig:
    """Configuration for LoRA."""

    max_lora_rank: MaxLoRARanks = 16
    """Max LoRA rank."""
    max_loras: int = Field(default=1, ge=1)
    """Max number of LoRAs in a single batch."""
    fully_sharded_loras: bool = False
    """By default, only half of the LoRA computation is sharded with tensor
    parallelism. Enabling this will use the fully sharded layers. At high
    sequence length, max rank or tensor parallel size, this is likely faster.
    """
    max_cpu_loras: int | None = None
    """Maximum number of LoRAs to store in CPU memory. Must be >= than
    `max_loras`."""
    lora_dtype: torch.dtype | LoRADType = "auto"
    """Data type for LoRA. If auto, will default to base model dtype."""
    target_modules: list[str] | None = None
    """Restrict LoRA to specific module suffixes (e.g., ["o_proj", "qkv_proj"]).
    If None, all supported LoRA modules are used. This allows deployment-time
    control over which modules have LoRA applied, useful for performance tuning."""
    default_mm_loras: dict[str, str] | None = None
    """Dictionary mapping specific modalities to LoRA model paths; this field
    is only applicable to multimodal models and should be leveraged when a
    model always expects a LoRA to be active when a given modality is present.
    Note that currently, if a request provides multiple additional
    modalities, each of which have their own LoRA, we do NOT apply
    default_mm_loras because we currently only support one lora adapter
    per prompt. When run in offline mode, the lora IDs for n modalities
    will be automatically assigned to 1-n with the names of the modalities
    in alphabetic order."""
    enable_tower_connector_lora: bool = False
    """If `True`, LoRA support for the tower (vision encoder) and connector 
    of multimodal models will be enabled. This is an experimental feature and 
    currently only supports some MM models such as the Qwen VL series. The default 
    is False."""
    specialize_active_lora: bool = False
    """Whether to construct lora kernel grid by the number of active LoRA adapters.
    When set to True, separate cuda graphs will be captured for different counts
    of active LoRAs (powers of 2 up to max_loras), which can improve performance
    for variable LoRA usage patterns at the cost of increased startup time and
    memory usage. Only takes effect when cudagraph_specialize_lora is True.
    """

    # --- Dynamic LoRA slot scaling fields ---

    min_loras: int = Field(default=1, ge=1)
    """Minimum number of LoRA GPU slots. Acts as the floor when dynamic
    resizing shrinks the slot count. Must be >= 1 and <= max_loras.
    Only meaningful when dynamic_lora_slots=True."""

    dynamic_lora_slots: bool = False
    """Enable automatic dynamic resizing of GPU LoRA slots at runtime.
    When True, max_loras becomes the initial value and upper bound for
    automatic scaling. Operator-triggered scaling via
    POST /v1/scale_max_loras is always available regardless of this flag."""

    lora_mem_high_watermark: float = Field(default=0.8, gt=0.0, lt=1.0)
    """GPU memory utilization fraction above which LoRA slots are
    proactively reduced. Must be in (0, 1) and greater than
    lora_mem_low_watermark. Only used when dynamic_lora_slots=True."""

    lora_mem_low_watermark: float = Field(default=0.5, gt=0.0, lt=1.0)
    """GPU memory utilization fraction below which LoRA slots may be
    expanded. Must be in (0, 1) and less than lora_mem_high_watermark.
    Only used when dynamic_lora_slots=True."""

    lora_slot_resize_cooldown_s: float = Field(default=1.0, ge=0.0)
    """Minimum seconds between consecutive automatic LoRA slot resizes.
    Prevents thrashing when memory utilization oscillates near a watermark.
    Only used when dynamic_lora_slots=True."""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []
        factors.append(self.max_lora_rank)
        factors.append(self.max_loras)
        factors.append(self.fully_sharded_loras)
        factors.append(self.lora_dtype)
        factors.append(self.enable_tower_connector_lora)
        # target_modules affects which modules get LoRA applied
        factors.append(
            tuple(sorted(self.target_modules)) if self.target_modules else None
        )
        # dynamic_lora_slots disables LoRA cudagraph specialization
        factors.append(self.dynamic_lora_slots)

        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    @model_validator(mode="after")
    def _validate_lora_config(self) -> Self:
        if self.max_cpu_loras is None:
            self.max_cpu_loras = self.max_loras
        elif self.max_cpu_loras < self.max_loras:
            raise ValueError(
                f"max_cpu_loras ({self.max_cpu_loras}) must be >= "
                f"max_loras ({self.max_loras})."
            )

        if self.dynamic_lora_slots:
            if self.min_loras > self.max_loras:
                raise ValueError(
                    f"min_loras ({self.min_loras}) must be <= "
                    f"max_loras ({self.max_loras})."
                )
            if self.lora_mem_low_watermark >= self.lora_mem_high_watermark:
                raise ValueError(
                    "lora_mem_low_watermark must be less than "
                    "lora_mem_high_watermark, both in (0, 1). Got "
                    f"{self.lora_mem_low_watermark} >= "
                    f"{self.lora_mem_high_watermark}."
                )

        return self

    def verify_with_model_config(self, model_config: ModelConfig):
        if self.lora_dtype in (None, "auto"):
            self.lora_dtype = model_config.dtype
        elif isinstance(self.lora_dtype, str):
            self.lora_dtype = getattr(torch, self.lora_dtype)
