# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import os
import time
from collections.abc import Sequence
from typing import Callable, Optional, TypeVar, Union

import regex as re
import safetensors.torch
import torch
from torch import nn

from vllm.config.lora import LoRAConfig
from vllm.logger import init_logger
from vllm.lora.layers import BaseLayerWithLoRA, LoRAMapping
from vllm.lora.lora_weights import LoRALayerWeights, PackedLoRALayerWeights
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.punica_wrapper import get_punica_wrapper
from vllm.lora.utils import (from_layer, from_layer_logits_processor,
                             get_supported_lora_modules,
                             is_regex_target_modules,
                             parse_fine_tuned_lora_name, replace_submodule)
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
from vllm.model_executor.models import SupportsLoRA, supports_multimodal
from vllm.model_executor.models.interfaces import is_pooling_model
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.utils import PPMissingLayer, WeightsMapper
from vllm.model_executor.utils import get_packed_modules_mapping
from vllm.utils import LRUCache, is_pin_memory_available

logger = init_logger(__name__)

T = TypeVar("T")


class AdapterLRUCache(LRUCache[int, T]):

    def __init__(self, capacity: int, deactivate_fn: Callable[[int], object]):
        super().__init__(capacity)
        self.deactivate_fn = deactivate_fn

    def _on_remove(self, key: int, value: Optional[T]):
        logger.debug("Removing adapter int id: %d", key)
        self.deactivate_fn(key)
        return super()._on_remove(key, value)


_GLOBAL_LORA_ID = 0


def get_lora_id():
    global _GLOBAL_LORA_ID
    _GLOBAL_LORA_ID += 1
    return _GLOBAL_LORA_ID


def is_moe_model(model: nn.Module) -> bool:
    """Checks if the model contains FusedMoE layers and warns the user."""
    if any(isinstance(module, FusedMoE) for module in model.modules()):
        logger.warning_once(
            "For MoE models, vLLM currently does not support fused MoE LoRA "
            "inference. Please ensure that the loaded LoRA model does not "
            "contain expert weights.")
        return True
    return False


class LoRAModel:
    """A LoRA fine-tuned model."""

    def __init__(
        self,
        lora_model_id: int,
        rank: int,
        loras: dict[str, LoRALayerWeights],
    ) -> None:
        """
        Args:
            lora_model_id: The integer id for the lora model.
            rank: lora rank.
            loras: module name -> weights for lora-replaced layers.

        """
        self.id = lora_model_id

        assert (
            lora_model_id
            > 0), f"a valid lora id should be greater than 0, got {self.id}"
        self.rank = rank
        self.loras: dict[str, LoRALayerWeights] = loras

    def clone(self, lora_model_id: int) -> "LoRAModel":
        """Return a copy of the object with different ids.

        Will share the underlying tensors."""
        return self.__class__(
            lora_model_id,
            rank=self.rank,
            loras=self.loras.copy(),
        )

    @property
    def extra_vocab_size(self) -> int:
        return max(lora.extra_vocab_size
                   for lora in self.loras.values()) if self.loras else 0

    def get_lora(self, module_name: str) -> Optional[LoRALayerWeights]:
        """Get LoRA for a given module by name"""
        return self.loras.get(module_name, None)

    def check_lora_name(self, lora_name: str) -> bool:
        return lora_name in self.loras

    # (yard1): TODO see if we can derive target_embedding_padding automatically
    @classmethod
    def from_lora_tensors(
        cls,
        lora_model_id: int,
        tensors: dict[str, torch.Tensor],
        peft_helper: PEFTHelper,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
        embeddings: Optional[dict[str, torch.Tensor]] = None,
        target_embedding_padding: Optional[int] = None,
        embedding_modules: Optional[dict[str, str]] = None,
        embedding_padding_modules: Optional[list[str]] = None,
        weights_mapper: Optional[WeightsMapper] = None,
    ) -> "LoRAModel":
        """Create a LoRAModel from a dictionary of tensors."""
        pin_memory = str(device) == "cpu" and is_pin_memory_available()
        loras: dict[str, LoRALayerWeights] = {}
        for tensor_name, tensor in tensors.items():
            module_name, is_lora_a, is_bias = parse_fine_tuned_lora_name(
                tensor_name, weights_mapper)
            if module_name not in loras:
                lora_embeddings_tensor = None
                if embeddings:
                    assert embedding_modules is not None
                    embeddings_module = next(
                        (k for k in embedding_modules if k in module_name),
                        None)
                    if embeddings_module:
                        lora_embeddings_tensor = embeddings[
                            embedding_modules[embeddings_module]].to(
                                device=device, dtype=dtype)
                        if pin_memory:
                            lora_embeddings_tensor = (
                                lora_embeddings_tensor.pin_memory())
                loras[module_name] = LoRALayerWeights.from_config(
                    module_name, peft_helper, lora_embeddings_tensor)

            if is_bias:
                loras[module_name].bias = tensor.to(device=device, dtype=dtype)
                bias = tensor.to(device=device, dtype=dtype)
                if pin_memory:
                    bias = bias.pin_memory()
                loras[module_name].bias = bias
            elif is_lora_a:
                loras[module_name].lora_a = tensor.to(device=device,
                                                      dtype=dtype)
                if pin_memory:
                    loras[module_name].lora_a = loras[
                        module_name].lora_a.pin_memory()
            else:
                loras[module_name].lora_b = tensor.to(device=device,
                                                      dtype=dtype)
                assert embedding_padding_modules is not None
                if any(name in module_name
                       for name in embedding_padding_modules
                       ) and target_embedding_padding is not None:
                    lora_b = loras[module_name].lora_b
                    assert target_embedding_padding >= lora_b.shape[0]
                    addition = target_embedding_padding - lora_b.shape[0]
                    loras[module_name].lora_b = torch.nn.functional.pad(
                        lora_b, (0, 0, 0, addition))
                if pin_memory:
                    loras[module_name].lora_b = loras[
                        module_name].lora_b.pin_memory()

        for lora in loras.values():
            lora.optimize()

        return cls(lora_model_id, peft_helper.r, loras)

    @classmethod
    def from_local_checkpoint(
            cls,
            lora_dir: str,
            expected_lora_modules: list[str],
            peft_helper: PEFTHelper,
            *,
            lora_model_id: Optional[int] = None,
            device: str = "cuda",
            dtype: Optional[torch.dtype] = None,
            target_embedding_padding: Optional[int] = None,
            embedding_modules: Optional[dict[str, str]] = None,
            embedding_padding_modules: Optional[list[str]] = None,
            weights_mapper: Optional[WeightsMapper] = None,
            tensorizer_config_dict: Optional[dict] = None) -> "LoRAModel":
        """Create a LoRAModel from a local checkpoint.

        Args:
            lora_dir: The local path that has lora data.
            expected_lora_modules: Name of modules that are expected to be
                replaced by lora.
            peft_helper: Loaded lora configuration information.
            lora_model_id: LoRA model id. If not given, automatically set by
                a global counter.
            device: Device where the lora model is loaded.
            dtype: dtype of the lora model weights.

        Returns:
            Loaded LoRA Model.
        """
        lora_tensor_path = os.path.join(lora_dir, "adapter_model.safetensors")
        lora_bin_file_path = os.path.join(lora_dir, "adapter_model.bin")
        lora_pt_file_path = os.path.join(lora_dir, "adapter_model.pt")
        new_embeddings_tensor_path = os.path.join(
            lora_dir, "new_embeddings.safetensors")
        new_embeddings_bin_file_path = os.path.join(lora_dir,
                                                    "new_embeddings.bin")
        tensors: dict[str, torch.Tensor] = {}
        unexpected_modules: list[Union[list[str], str]] = []

        def check_unexpected_modules(modules: dict):
            for lora_module in modules.keys():  # noqa
                module_name, _, _ = parse_fine_tuned_lora_name(
                    lora_module, weights_mapper)
                part_name = module_name.split(".")[-1]
                if part_name not in expected_lora_modules:
                    unexpected_modules.append(module_name)
            if unexpected_modules:
                raise ValueError(
                    f"While loading {lora_dir}, expected"
                    f" target modules in {expected_lora_modules}"
                    f" but received {unexpected_modules}."
                    f" Please verify that the loaded LoRA module is correct")

        if tensorizer_config_dict:
            from tensorizer import TensorDeserializer

            tensorizer_config = TensorizerConfig(**tensorizer_config_dict)
            lora_tensor_path = os.path.join(tensorizer_config.tensorizer_dir,
                                            "adapter_model.tensors")
            tensorizer_args = tensorizer_config._construct_tensorizer_args()
            tensors = TensorDeserializer(
                lora_tensor_path,
                dtype=tensorizer_config.dtype,
                **tensorizer_args.deserialization_kwargs)
            check_unexpected_modules(tensors)

        elif os.path.isfile(lora_tensor_path):
            # Find unexpected modules.
            # Use safetensor key as a source of truth to find expected modules.
            # in peft if you have target_modules A, B, C and C does not exist
            # in the model it won’t error and model will be trained with A, B
            # loraified. C won’t exist in the safetensor but it will exist in
            # the target_modules of the adapter_config.json.
            unexpected_modules = []
            with safetensors.safe_open(lora_tensor_path,
                                       framework="pt") as f:  # type: ignore
                # Load tensors if there are only expected modules.
                check_unexpected_modules(f)
                for module in f.keys():  # noqa
                    tensors[module] = f.get_tensor(module)
        elif os.path.isfile(lora_bin_file_path) or os.path.isfile(
                lora_pt_file_path):
            # When a bin/pt file is provided, we rely on config to find
            # unexpected modules.
            unexpected_modules = []
            target_modules = peft_helper.target_modules
            if not isinstance(target_modules, list):
                target_modules = [target_modules]
            for module in target_modules:
                # Compatible with more modules,
                # such as:layers.11.self_attn.k_proj
                part_name = module.split(".")[-1]
                if part_name not in expected_lora_modules:
                    unexpected_modules.append(module)
            # loaded lora's target modules must be a subset of
            # expected_lora_modules. It is not reliable. See
            # https://github.com/vllm-project/vllm/pull/5909. But there's no
            # other better mechanism.
            if unexpected_modules and not is_regex_target_modules(
                    peft_helper.target_modules, expected_lora_modules):
                raise ValueError(
                    f"While loading {lora_dir}, expected"
                    f" target modules in {expected_lora_modules}"
                    f" but received {unexpected_modules}."
                    f" Please verify that the loaded LoRA module is correct")
            lora_file_path = (lora_bin_file_path
                              if os.path.isfile(lora_bin_file_path) else
                              lora_pt_file_path)
            tensors = torch.load(lora_file_path,
                                 map_location=device,
                                 weights_only=True)
        else:
            raise ValueError(f"{lora_dir} doesn't contain tensors")

        embeddings = None
        if os.path.isfile(new_embeddings_tensor_path):
            embeddings = safetensors.torch.load_file(
                new_embeddings_tensor_path)
        elif os.path.isfile(new_embeddings_bin_file_path):
            embeddings = torch.load(new_embeddings_bin_file_path,
                                    map_location=device,
                                    weights_only=True)

        return cls.from_lora_tensors(
            lora_model_id=get_lora_id()
            if lora_model_id is None else lora_model_id,
            tensors=tensors,
            peft_helper=peft_helper,
            device=device,
            dtype=dtype,
            embeddings=embeddings,
            target_embedding_padding=target_embedding_padding,
            embedding_modules=embedding_modules,
            embedding_padding_modules=embedding_padding_modules,
            weights_mapper=weights_mapper)


class LoRAModelManager:
    """A manager that manages multiple LoRA-fine-tuned models."""

    def __init__(
        self,
        model: SupportsLoRA,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        vocab_size: int,
        lora_config: LoRAConfig,
        device: torch.device,
    ):
        """Create a LoRAModelManager and adapter for a given model.

        Args:
            model: the model to be adapted.
            max_num_seqs: the maximum number of sequences model can run in a
                single batch.
            max_num_batched_tokens: the maximum number of tokens model can run
                in a single batch.
            vocab_size: the vocab size of the model.
            lora_config: the LoRA configuration.
        """
        self.model: SupportsLoRA = model
        self._registered_adapters: dict[int, LoRAModel] = {}
        # Dict instead of a set for compatibility with LRUCache.
        self._active_adapters: dict[int, None] = {}
        self.adapter_type = "LoRA"
        self.lora_config = lora_config
        self.device = device
        self.max_num_seqs = max_num_seqs
        assert self.capacity >= self.lora_slots
        self.max_num_batched_tokens = math.ceil(max_num_batched_tokens / 8) * 8
        self.lora_index_to_id: list[Optional[int]] = [None] * self.lora_slots
        self.vocab_size = vocab_size
        self.punica_wrapper = get_punica_wrapper(
            max_num_batched_tokens,
            max_batches=self.max_num_seqs,
            device=self.device,
            max_loras=self.lora_config.max_loras,
        )

        self.supported_lora_modules = get_supported_lora_modules(self.model)
        assert self.supported_lora_modules, "No supported LoRA modules found in"
        f" {self.model.__class__.__name__}."

        self.packed_modules_mapping = get_packed_modules_mapping(self.model)
        # Used to indicate whether the model is a multimodal model
        self.supports_mm: bool = (
            supports_multimodal(self.model)
            # In case the model only supports LoRA for
            # text modules (e.g. ChatGLM)
            and hasattr(self.model, "get_mm_mapping"))
        self.is_pooling_model = is_pooling_model(self.model)
        self.is_moe_model = is_moe_model(self.model)
        self.packed_modules: dict[str, list[str]] = {}
        self.modules: dict[str, BaseLayerWithLoRA] = {}
        # Dict instead of a set for compatibility with LRUCache.
        self._last_mapping: Optional[LoRAMapping] = None
        self._create_lora_modules()
        self.model.lora_manager = self

    def __len__(self) -> int:
        return len(self._registered_adapters)

    @property
    def capacity(self) -> int:
        return self.lora_config.max_cpu_loras

    @property
    def lora_slots(self) -> int:
        return self.lora_config.max_loras

    @property
    def adapter_slots(self) -> int:
        return self.lora_slots

    def activate_adapter(
        self,
        lora_id: int,
    ) -> bool:
        """Move LoRA into a GPU buffer to be used in the forward pass."""
        if lora_id in self._active_adapters:
            return False
        first_free_slot = next(
            ((i, lora_id) for i, lora_id in enumerate(self.lora_index_to_id)
             if lora_id is None), None)
        if first_free_slot is None:
            raise ValueError("No free lora slots")
        index, _ = first_free_slot
        self._active_adapters[lora_id] = None
        lora_model = self._registered_adapters[lora_id]
        logger.debug("Activating LoRA. int id: %d, slot index: %d",
                     lora_model.id, index)
        self.lora_index_to_id[index] = lora_model.id
        for module_name, module in self.modules.items():
            module_lora = self._get_lora_layer_weights(lora_model, module_name)
            if module_lora:
                module_lora.optimize()
                # Bias is not explicitly enabled with the flag enable_lora_bias.
                bias = module_lora.bias
                if ((torch.is_tensor(bias) or
                     (isinstance(bias, Sequence) and any(b is not None
                                                         for b in bias)))
                        and not self.lora_config.bias_enabled):
                    module_lora.bias = None
                    raise ValueError(
                        f"Adapter bias cannot be used for {module_name}"
                        " without --enable-lora-bias.")
                module.set_lora(index, module_lora.lora_a, module_lora.lora_b,
                                module_lora.embeddings_tensor,
                                module_lora.bias)
            else:
                module.reset_lora(index)
        return True

    def _deactivate_adapter(self, lora_id: int):
        try:
            index = self.lora_index_to_id.index(lora_id)
            self.lora_index_to_id[index] = None
        except ValueError:
            pass

    def _add_adapter(self, lora: LoRAModel):
        self._create_merged_loras_inplace(lora)
        self._registered_adapters[lora.id] = lora

    def pin_adapter(self, lora_id: int) -> bool:
        """Pin a LoRAModel in the manager cache."""
        raise NotImplementedError(
            "Pinning is not supported in LoRAModelManager. "
            "Use LRUCacheLoRAModelManager for pinning")  # type: ignore

    def _set_adapter_mapping(self, mapping: LoRAMapping) -> None:
        # update lora states
        self.punica_wrapper.update_metadata(
            mapping,
            self.lora_index_to_id,
            self.lora_slots + 1,
            self.vocab_size,
            self.lora_config.lora_extra_vocab_size,
        )

    def remove_all_adapters(self):
        """Remove all LoRAModels from the manager."""
        self._registered_adapters.clear()
        self.lora_index_to_id = [None] * self.lora_slots
        self._active_adapters.clear()

    def _create_lora_modules(self):

        def _parent_module(module_name: str) -> str:
            # module name is a dot separated name.
            # for example:
            #  - given an input 'x.y.z' return 'x.y'
            #  - given an input 'x' return ''
            return module_name.rpartition('.')[0]

        for module_name, module in self.model.named_modules(
                remove_duplicate=False):
            if isinstance(module, PPMissingLayer):
                continue
            if not self._match_target_modules(module_name):
                continue
            # A temporary approach for multimodal models to support LoRA
            # TODO: Remove this restriction
            if self._filter_unsupported_mm_module(module_name):
                logger.warning(
                    "Regarding multimodal models, vLLM currently only supports "
                    "adding LoRA to language model, %s will be ignored.",
                    module_name,
                )
                continue
            parts = module_name.split(".")[-1]
            packed_moduled_lst = self.packed_modules_mapping.get(parts, [])
            new_module = replace_submodule(
                self.model, module_name,
                from_layer(module, self.lora_slots, self.lora_config,
                           packed_moduled_lst, self.model.config))

            # (yard1): TODO make this more robust
            if "lm_head" in module_name:
                logits_processor_module_name = 'logits_processor'
                parent_module = _parent_module(module_name)
                if parent_module:
                    logits_processor_module_name = (
                        f"{parent_module}.{logits_processor_module_name}")

                logits_processor_module = self.model.get_submodule(
                    logits_processor_module_name)

                new_module = replace_submodule(
                    self.model, logits_processor_module_name,
                    from_layer_logits_processor(logits_processor_module,
                                                module, self.lora_slots,
                                                self.lora_config,
                                                self.model.config))

            # In some models, especially multimodal ones, layers with the same
            # name may have different types, such as nn.Linear and
            # ReplicatedLinear. The nn.Linear layers cannot be replaced with
            # LoRA layers, leading to assertion error. The following check
            # aims to prevent this error
            if self.supports_mm and not isinstance(new_module,
                                                   BaseLayerWithLoRA):
                continue
            self.register_module(module_name, new_module)
            self._register_packed_modules(module_name)
            # All lora layers share the same punica_wrapper based on reference.
            new_module.set_mapping(self.punica_wrapper)

    def register_module(self, module_name: str, module: "BaseLayerWithLoRA"):
        assert isinstance(module, BaseLayerWithLoRA)
        self.modules[module_name] = module

    def create_dummy_lora(
            self,
            lora_id: int,
            rank: int,
            embedding_modules: Optional[dict[str, str]] = None) -> LoRAModel:
        """Create zero-initialized LoRAModel for warmup."""
        model = LoRAModel(lora_id, rank, {})
        for module_name, module in self.model.named_modules():
            bias_enabled = self.lora_config.bias_enabled
            if (not self._match_target_modules(module_name)
                    or not isinstance(module, BaseLayerWithLoRA)
                    or self._filter_unsupported_mm_module(module_name)):
                continue
            parts = module_name.split(".")
            if module_name not in self.packed_modules:
                assert embedding_modules is not None
                if parts[-1] in embedding_modules:
                    input_dim = (module.base_layer.org_vocab_size +
                                 self.lora_config.lora_extra_vocab_size if
                                 hasattr(module.base_layer, "org_vocab_size")
                                 else module.base_layer.weight.shape[1])
                    output_dim = module.base_layer.embedding_dim if hasattr(
                        module.base_layer,
                        "embedding_dim") else module.base_layer.weight.shape[0]
                    embeddings_tensor_dim = (module.base_layer.embedding_dim if
                                             hasattr(module.base_layer,
                                                     "embedding_dim") else
                                             module.base_layer.weight.shape[1])
                    lora = LoRALayerWeights.create_dummy_lora_weights(
                        module_name,
                        input_dim,
                        output_dim,
                        rank,
                        module.lora_a_stacked[0].dtype,
                        "cpu",
                        embeddings_tensor_dim=embeddings_tensor_dim,
                        bias_enabled=bias_enabled)
                else:
                    lora = LoRALayerWeights.create_dummy_lora_weights(
                        module_name,
                        module.lora_a_stacked[0].shape[-1],
                        module.lora_b_stacked[0].shape[-2],
                        rank,
                        module.lora_a_stacked[0].dtype,
                        "cpu",
                        bias_enabled=bias_enabled,
                    )
            else:
                parts = module_name.split(".")
                replacements = self.packed_modules_mapping[parts[-1]]
                subloras: list[Optional[LoRALayerWeights]] = []
                for i, r in enumerate(replacements):
                    lora = LoRALayerWeights.create_dummy_lora_weights(
                        module_name + "." + r,
                        module.lora_a_stacked[i].shape[-1],
                        module.lora_b_stacked[i].shape[-2],
                        rank,
                        module.lora_a_stacked[i].dtype,
                        "cpu",
                        bias_enabled=bias_enabled,
                    )
                    subloras.append(lora)
                lora = PackedLoRALayerWeights.pack(subloras)
            model.loras[module_name] = lora
        return model

    def _match_target_modules(self, module_name: str):
        return any(
            re.match(
                r".*\.{target_module}$".format(target_module=target_module),
                module_name) or target_module == module_name
            for target_module in self.supported_lora_modules)

    def _filter_unsupported_mm_module(self, module_name: str) -> bool:
        """
        Regarding multimodal models, vLLM currently only supports adding LoRA to
        language model. LoRA for other modules, such as the vision tower, will
        be filtered out.
        """
        if self.supports_mm:
            module_mapping: MultiModelKeys = self.model.get_mm_mapping()
            prefix_lst = module_mapping.connector + module_mapping.tower_model
            return any(
                [module_name.startswith(prefix) for prefix in prefix_lst])
        return False

    def _register_packed_modules(self, module_full_name: str) -> None:
        parts = module_full_name.split(".")
        module_name = parts[-1]
        replacements = self.packed_modules_mapping.get(module_name, [])
        # When replacements is less than or equal to 1, it indicates that this
        # module is not a packed module.
        if len(replacements) <= 1:
            return
        prefix = ".".join(parts[:-1])
        self.packed_modules[module_full_name] = [
            prefix + "." + r if prefix else r for r in replacements
        ]

    def _create_merged_loras_inplace(self, lora_model: LoRAModel) -> None:
        for module_name, new_module_names in self.packed_modules.items():
            replacement_loras: list[Optional[LoRALayerWeights]] = []
            replaced_module: set[str] = set()
            has_replacement = False
            for r in new_module_names:
                lora = self._get_lora_layer_weights(lora_model, r)
                replacement_loras.append(lora)
                if lora:
                    has_replacement = True
                    replaced_module.add(r)
            if not has_replacement:
                continue
            for i in range(len(replacement_loras)):
                if replacement_loras[i]:
                    continue
                replacement_loras[i] = None
            # HACK Temporary solution for the pool model.
            if self.is_pooling_model and not lora_model.check_lora_name(
                    module_name):
                replaced_module_name = module_name.replace("model.", "")
                if lora_model.check_lora_name(module_name):
                    module_name = replaced_module_name
            lora_model.loras[module_name] = PackedLoRALayerWeights.pack(
                replacement_loras)
            # Remove the modules that have been replaced.
            for module in replaced_module:
                lora_model.loras.pop(module, None)

    def _get_lora_layer_weights(
            self, lora_model: LoRAModel,
            module_name: str) -> Optional[LoRALayerWeights]:
        org_module_name = module_name
        if self.is_pooling_model and not lora_model.check_lora_name(
                module_name):
            # If it's a pool model, and the layer name is not found,
            # remove the prefix 'model.' and search again.
            module_name = module_name.replace("model.", "")
            if lora_model.check_lora_name(module_name):
                org_module_name = module_name
                logger.info_once(
                    "For the pool model, successfully loaded the LoRA weights "
                    "after removing the prefix 'model.'.")
        return lora_model.get_lora(org_module_name)

    def deactivate_adapter(self, adapter_id: int) -> bool:
        if adapter_id not in self._active_adapters:
            return False
        self._deactivate_adapter(adapter_id)
        self._active_adapters.pop(adapter_id, None)
        return True

    def add_adapter(self, adapter: LoRAModel) -> bool:
        logger.debug("Adding lora. Model id: %d, "
                     "int id: %d", adapter.id, adapter.id)
        if adapter.id in self._registered_adapters:
            return False
        if len(self._registered_adapters) >= self.capacity:
            raise RuntimeError("No free adapter slots.")
        self._add_adapter(adapter)
        return True

    def set_adapter_mapping(self, mapping: LoRAMapping) -> None:
        if self._last_mapping != mapping:
            self._set_adapter_mapping(mapping)
            self._last_mapping = mapping

    def remove_adapter(self, adapter_id: int) -> bool:
        self.deactivate_adapter(adapter_id)
        if adapter_id not in self._registered_adapters:
            return False
        self._registered_adapters.pop(adapter_id, None)
        return True

    def list_adapters(self) -> dict[int, LoRAModel]:
        return dict(self._registered_adapters)

    def get_adapter(self, adapter_id: int) -> Optional[LoRAModel]:
        return self._registered_adapters.get(adapter_id)


class LoRALRUCache(AdapterLRUCache[LoRAModel]):

    def __init__(self, capacity: int, deactivate_lora_fn: Callable[[int],
                                                                   bool]):
        super().__init__(capacity, deactivate_lora_fn)


class LRUCacheLoRAModelManager(LoRAModelManager):
    """A model manager that manages multiple LoRAs with LRU cache."""

    def __init__(self, model: nn.Module, max_num_seqs: int,
                 max_num_batched_tokens: int, vocab_size: int,
                 lora_config: LoRAConfig, device: torch.device):
        super().__init__(model, max_num_seqs, max_num_batched_tokens,
                         vocab_size, lora_config, device)
        self._registered_adapters: LoRALRUCache = LoRALRUCache(
            self.capacity, self.deactivate_adapter)
        self._active_adapters: LoRALRUCache = LoRALRUCache(
            self.lora_slots, self._deactivate_adapter)

    def list_adapters(self) -> dict[int, LoRAModel]:
        """List all registered LoRAModels."""
        return dict(self._registered_adapters.cache)

    def add_adapter(self, lora: LoRAModel) -> bool:
        """Add a LoRAModel to the manager."""
        logger.debug("Adding lora. Model id: %d, "
                     "int id: %d", lora.id, lora.id)
        if lora.id not in self._registered_adapters:
            self._add_adapter(lora)
            was_added = True
        else:
            # We always touch to update the LRU cache order
            self._registered_adapters.touch(lora.id)
            was_added = False
        return was_added

    def activate_adapter(
        self,
        lora_id: int,
    ) -> bool:
        if lora_id not in self._active_adapters and len(
                self._active_adapters) >= self.lora_slots:
            self._active_adapters.remove_oldest()
        result = super().activate_adapter(lora_id)
        # We always touch to update the LRU cache order
        self._active_adapters.touch(lora_id)
        return result

    def remove_oldest_adapter(self) -> bool:
        if len(self._registered_adapters) > 0:
            self._registered_adapters.remove_oldest()
            return True
        return False

    def pin_adapter(self, lora_id: int) -> bool:
        """Pin a LoRAModel in the manager cache."""
        self._pin_lora_in_cpu_cache(lora_id)
        self._pin_lora_in_gpu_cache(lora_id)
        return True

    def _pin_lora_in_cpu_cache(self, lora_id: int):
        try:
            self._registered_adapters.pin(lora_id)
        except ValueError as err:
            raise ValueError("Pinning failed. "
                             f"LoRA {lora_id} is not registered.") from err

    def _pin_lora_in_gpu_cache(self, lora_id: int):
        if lora_id not in self._active_adapters:
            # move lora to gpu if not already active
            self.activate_adapter(lora_id)

        self._active_adapters.pin(lora_id)


def create_lora_manager(
        model: nn.Module,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        vocab_size: int,
        lora_config: LoRAConfig,
        device: torch.device,
        lora_manager_cls: type[LoRAModelManager] = LoRAModelManager,
        **kwargs) -> LoRAModelManager:
    """Create a LoRA adapter for a given model."""
    if not isinstance(model, SupportsLoRA):
        raise ValueError(f"Model {type(model)} is not supported for LoRA.")
    lora_manager = lora_manager_cls(
        model=model,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        vocab_size=vocab_size,
        lora_config=lora_config,
        device=device,
        **kwargs)
    return lora_manager


class StorageCatalogLoRAModelManager(LoRAModelManager):
    """A model manager that manages LoRAs using a storage-based catalog.

    This manager scans a storage path for available LoRA adapters and loads
    them on demand. It uses LRU cache for GPU memory management and can also
    cache adapters in CPU memory for faster loading.
    """

    def __init__(self, model: nn.Module, max_num_seqs: int,
                 max_num_batched_tokens: int, vocab_size: int,
                 lora_config: LoRAConfig, device: torch.device,
                 catalog_path: str):
        super().__init__(model, max_num_seqs, max_num_batched_tokens,
                         vocab_size, lora_config, device)

        self.catalog_path = catalog_path
        self.max_cpu_adapters = self.lora_config.max_cpu_loras or self.capacity

        # LRU cache for GPU memory management (using IDs for compatibility)
        self._active_adapters: LoRALRUCache = LoRALRUCache(
            self.lora_slots, self._deactivate_adapter)

        # LRU cache for CPU memory management (using IDs for compatibility)
        self._cpu_adapters: LoRALRUCache = LoRALRUCache(
            self.max_cpu_adapters, lambda _: True)  # No-op deactivate for CPU

        # Mapping from storage names to cache IDs
        self._name_to_id: dict[str, int] = {}
        self._id_to_name: dict[int, str] = {}

        # Cache of available adapter names in storage
        self._available_adapters_cache: Optional[set[str]] = None
        self._cache_timestamp: float = 0.0
        self._cache_ttl: float = 60.0  # 60 seconds TTL for storage scan

    def _scan_storage_catalog(self) -> set[str]:
        """Scan the storage catalog for available LoRA adapters."""
        current_time = time.time()
        
        # Return cached result if still valid
        if (self._available_adapters_cache is not None and 
            current_time - self._cache_timestamp < self._cache_ttl):
            return self._available_adapters_cache

        available_adapters = set()
        try:
            if not os.path.exists(self.catalog_path):
                logger.warning(f"Storage catalog path {self.catalog_path} does not exist")
                return available_adapters

            for item in os.listdir(self.catalog_path):
                item_path = os.path.join(self.catalog_path, item)
                if os.path.isdir(item_path) and self._is_valid_lora_directory(item_path):
                    available_adapters.add(item)
        except Exception as e:
            logger.warning(f"Error scanning storage catalog {self.catalog_path}: {e}")
        
        self._available_adapters_cache = available_adapters
        self._cache_timestamp = current_time
        return available_adapters

    def _get_or_create_adapter_id(self, adapter_name: str) -> int:
        """Get or create an adapter ID for a given adapter name.
        
        Uses the global incremental ID system for cache management
        while maintaining a mapping to storage names.
        """
        if adapter_name in self._name_to_id:
            return self._name_to_id[adapter_name]
        
        # Use global incremental ID for cache management
        adapter_id = get_lora_id()
        
        # Store the mapping
        self._name_to_id[adapter_name] = adapter_id
        self._id_to_name[adapter_id] = adapter_name
        
        return adapter_id

    def _is_valid_lora_directory(self, path: str) -> bool:
        """Check if a directory contains valid LoRA files."""
        required_files = ["adapter_config.json"]
        optional_files = ["adapter_model.safetensors", "adapter_model.bin"]
        
        # Check for required files
        for file in required_files:
            if not os.path.exists(os.path.join(path, file)):
                return False
        
        # Check for at least one optional file
        has_weights = any(os.path.exists(os.path.join(path, file)) 
                         for file in optional_files)
        if not has_weights:
            return False
        
        return True

    def _load_adapter_from_storage(self, adapter_name: str) -> Optional[LoRAModel]:
        """Load a LoRA adapter from storage."""
        adapter_path = os.path.join(self.catalog_path, adapter_name)
        
        if not os.path.exists(adapter_path):
            logger.warning(f"LoRA adapter {adapter_name} not found at {adapter_path}")
            return None
        
        try:
            # Get expected LoRA modules from the model
            expected_lora_modules = get_supported_lora_modules(self.model)
            
            # Create PEFTHelper for loading
            peft_helper = PEFTHelper(adapter_path)
            
            # Generate a unique ID for this adapter based on its name
            adapter_id = self._get_adapter_id_from_name(adapter_name)
            
            # Load the LoRA model
            lora_model = LoRAModel.from_local_checkpoint(
                lora_dir=adapter_path,
                expected_lora_modules=expected_lora_modules,
                peft_helper=peft_helper,
                lora_model_id=adapter_id,
                device=str(self.device),
                dtype=self.lora_config.lora_dtype,
                target_embedding_padding=self.lora_config.lora_vocab_padding_size,
            )
            
            # Store the adapter name for future reference
            lora_model.adapter_name = adapter_name
            
            logger.info(f"Successfully loaded LoRA adapter {adapter_name} (ID: {adapter_id}) from storage")
            return lora_model
            
        except Exception as e:
            logger.warning(f"Failed to load LoRA adapter {adapter_name} from {adapter_path}: {e}")
            return None

    def list_adapters(self) -> dict[int, LoRAModel]:
        """List all available LoRA adapters from storage and cache."""
        # Get adapters from storage
        storage_adapters = self._scan_storage_catalog()
        
        # Get adapters from GPU cache
        gpu_adapters = dict(self._active_adapters.cache)
        
        # Get adapters from CPU cache
        cpu_adapters = dict(self._cpu_adapters.cache)
        
        # Combine all adapters
        all_adapters = {}
        all_adapters.update(gpu_adapters)
        all_adapters.update(cpu_adapters)
        
        # Add storage adapters that aren't cached
        for adapter_name in storage_adapters:
            # Check if we have this adapter in our caches
            adapter_id = self._name_to_id.get(adapter_name)
            if adapter_id is None or (adapter_id not in gpu_adapters and adapter_id not in cpu_adapters):
                # Create a placeholder entry for storage-only adapters
                storage_id = self._get_or_create_adapter_id(adapter_name)
                all_adapters[storage_id] = None  # Placeholder
        
        return all_adapters

    def add_adapter(self, lora: LoRAModel) -> bool:
        """Add a LoRA adapter to the CPU cache."""
        # Get the adapter name from the lora object
        adapter_name = getattr(lora, 'adapter_name', None)
        
        if lora.id not in self._cpu_adapters:
            if len(self._cpu_adapters) >= self.max_cpu_adapters:
                self._cpu_adapters.remove_oldest()
            self._cpu_adapters.put(lora.id, lora)
            return True
        else:
            # Update LRU order
            self._cpu_adapters.touch(lora.id)
            return False

    def activate_adapter(self, lora_id: int) -> bool:
        """Activate a LoRA adapter, loading it to GPU if needed."""
        # First check if it's already active in GPU
        if lora_id in self._active_adapters:
            self._active_adapters.touch(lora_id)
            return super().activate_adapter(lora_id)
        
        # Check if it's in CPU cache
        if lora_id in self._cpu_adapters:
            lora = self._cpu_adapters.get(lora_id)
            if lora is not None:
                # Move to GPU cache
                if len(self._active_adapters) >= self.lora_slots:
                    self._active_adapters.remove_oldest()
                self._active_adapters.put(lora_id, lora)
                return super().activate_adapter(lora_id)
        
        # Try to load from storage
        storage_adapters = self._scan_storage_catalog()
        for adapter_name in storage_adapters:
            # Check if this adapter_name corresponds to the requested lora_id
            expected_id = self._get_or_create_adapter_id(adapter_name)
            if expected_id == lora_id:
                lora = self._load_adapter_from_storage(adapter_name)
                if lora is not None:
                    # Add to CPU cache first
                    self.add_adapter(lora)
                    # Then activate
                    return self.activate_adapter(lora.id)
        
        logger.warning(f"LoRA adapter {lora_id} not found in storage or cache")
        return False

    def remove_oldest_adapter(self) -> bool:
        """Remove the oldest adapter from CPU cache."""
        if len(self._cpu_adapters) > 0:
            self._cpu_adapters.remove_oldest()
            return True
        return False

    def pin_adapter(self, lora_id: int) -> bool:
        """Pin a LoRA adapter in both CPU and GPU caches."""
        # Pin in CPU cache
        try:
            self._cpu_adapters.pin(lora_id)
        except ValueError:
            pass  # Not in CPU cache, that's okay
        
        # Pin in GPU cache
        if lora_id in self._active_adapters:
            self._active_adapters.pin(lora_id)
        
        return True

    def get_adapter(self, adapter_id: int) -> Optional[LoRAModel]:
        """Get a LoRA adapter from cache or storage."""
        # Check GPU cache first
        if adapter_id in self._active_adapters:
            return self._active_adapters.get(adapter_id)
        
        # Check CPU cache
        if adapter_id in self._cpu_adapters:
            return self._cpu_adapters.get(adapter_id)
        
        # Try to load from storage
        storage_adapters = self._scan_storage_catalog()
        for adapter_name in storage_adapters:
            # Check if this adapter_name corresponds to the requested adapter_id
            expected_id = self._get_or_create_adapter_id(adapter_name)
            if expected_id == adapter_id:
                return self._load_adapter_from_storage(adapter_name)
        
        return None

    def remove_adapter(self, adapter_id: int) -> bool:
        """Remove a LoRA adapter from caches."""
        removed = False
        
        # Remove from GPU cache
        if adapter_id in self._active_adapters:
            self._active_adapters.remove(adapter_id)
            removed = True
        
        # Remove from CPU cache
        if adapter_id in self._cpu_adapters:
            self._cpu_adapters.remove(adapter_id)
            removed = True
        
        # Remove from name-to-id mapping
        if adapter_id in self._id_to_name:
            adapter_name = self._id_to_name[adapter_id]
            self._name_to_id.pop(adapter_name, None)
            self._id_to_name.pop(adapter_id, None)
        
        return removed

    def refresh_storage_catalog(self):
        """Refresh the storage catalog and clean up stale mappings.
        
        This should be called when the storage catalog is changed by external processes.
        It will remove mappings for adapters that no longer exist in storage.
        """
        current_storage_adapters = self._scan_storage_catalog()
        
        # Remove mappings for adapters that no longer exist
        stale_names = []
        for adapter_name in list(self._name_to_id.keys()):
            if adapter_name not in current_storage_adapters:
                stale_names.append(adapter_name)
        
        for adapter_name in stale_names:
            adapter_id = self._name_to_id[adapter_name]
            # Remove from caches if present
            if adapter_id in self._active_adapters:
                self._active_adapters.remove(adapter_id)
            if adapter_id in self._cpu_adapters:
                self._cpu_adapters.remove(adapter_id)
            # Remove from mappings
            self._name_to_id.pop(adapter_name, None)
            self._id_to_name.pop(adapter_id, None)
        
        # Clear the storage cache to force rescan
        self._available_adapters_cache = None
        self._cache_timestamp = 0.0
        
        logger.info(f"Refreshed storage catalog. Removed {len(stale_names)} stale mappings.")

    def get_adapter_name(self, adapter_id: int) -> Optional[str]:
        """Get the adapter name for a given adapter ID."""
        return self._id_to_name.get(adapter_id)

    def remove_all_adapters(self):
        """Remove all adapters from caches."""
        self._active_adapters.clear()
        self._cpu_adapters.clear()
        self._name_to_id.clear()
        self._id_to_name.clear()

    def __len__(self) -> int:
        """Return the number of adapters in CPU cache."""
        return len(self._cpu_adapters)

    def __contains__(self, adapter_id: int) -> bool:
        """Check if an adapter is in any cache."""
        return (adapter_id in self._active_adapters or 
                adapter_id in self._cpu_adapters)
