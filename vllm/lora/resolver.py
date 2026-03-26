# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections.abc import Set
from dataclasses import dataclass, field

from vllm.logger import init_logger
from vllm.lora.request import LoRARequest

logger = init_logger(__name__)


class LoRAResolver(ABC):
    """Base class for LoRA adapter resolvers.

    This class defines the interface for resolving and fetching LoRA adapters.
    Implementations of this class should handle the logic for locating and
    downloading LoRA adapters from various sources (e.g. S3, cloud storage,
    etc.).
    """

    @abstractmethod
    async def resolve_lora(
        self, base_model_name: str, lora_name: str
    ) -> LoRARequest | None:
        """Abstract method to resolve and fetch a LoRA model adapter.

        Implements logic to locate and download LoRA adapter based on the name.
        Implementations might fetch from a blob storage or other sources.

        Args:
            base_model_name: The name/identifier of the base model to resolve.
            lora_name: The name/identifier of the LoRA model to resolve.

        Returns:
            Optional[LoRARequest]: The resolved LoRA model information, or None
            if the LoRA model cannot be found.
        """
        pass

    async def get_desired_lora_slots(
        self,
        current_slots: int,
        active_loras: list[str],
        free_gpu_memory_bytes: int,
        total_gpu_memory_bytes: int,
    ) -> int | None:
        """Optional policy hook for dynamic LoRA slot sizing.

        Called by the engine between batches when ``dynamic_lora_slots=True``.
        Implementations may use the current number of slots, the set of active
        LoRA adapters, and available GPU memory to decide how many LoRA slots
        should be allocated. The engine will clamp the returned value to the
        configured ``[min_loras, max_loras]`` range.

        Args:
            current_slots: The current number of LoRA slots allocated by the
                engine.
            active_loras: A list of adapter names for LoRA adapters that are
                currently active or scheduled on this engine. Each entry
                corresponds to a distinct adapter; ordering is
                implementation-defined and should not be relied upon.
            free_gpu_memory_bytes: The current estimate of free GPU memory,
                in bytes, available for allocating additional LoRA adapters
                on this worker.
            total_gpu_memory_bytes: The total GPU memory capacity, in bytes,
                on this worker.

        Returns:
            Optional[int]: The desired total number of LoRA slots. If an
            integer is returned, the engine will adjust the number of slots
            (after clamping to ``[min_loras, max_loras]``). If ``None`` is
            returned, the engine will keep ``current_slots`` unchanged.
        """
        return None


@dataclass
class _LoRAResolverRegistry:
    resolvers: dict[str, LoRAResolver] = field(default_factory=dict)

    def get_supported_resolvers(self) -> Set[str]:
        """Get all registered resolver names."""
        return self.resolvers.keys()

    def register_resolver(
        self,
        resolver_name: str,
        resolver: LoRAResolver,
    ) -> None:
        """Register a LoRA resolver.
        Args:
            resolver_name: Name to register the resolver under.
            resolver: The LoRA resolver instance to register.
        """
        if resolver_name in self.resolvers:
            logger.warning(
                "LoRA resolver %s is already registered, and will be "
                "overwritten by the new resolver instance %s.",
                resolver_name,
                resolver,
            )

        self.resolvers[resolver_name] = resolver

    def get_resolver(self, resolver_name: str) -> LoRAResolver:
        """Get a registered resolver instance by name.
        Args:
            resolver_name: Name of the resolver to get.
        Returns:
            The resolver instance.
        Raises:
            KeyError: If the resolver is not found in the registry.
        """
        if resolver_name not in self.resolvers:
            raise KeyError(
                f"LoRA resolver '{resolver_name}' not found. "
                f"Available resolvers: {list(self.resolvers.keys())}"
            )
        return self.resolvers[resolver_name]


LoRAResolverRegistry = _LoRAResolverRegistry()
