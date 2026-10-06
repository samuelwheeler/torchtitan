# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import Any

import torch
from torch.distributed.tensor import DTensor, Replicate

from torchtitan.models.common.rope import ComplexRoPE
from torchtitan.models.utils import MoEStateDictAdapter
from torchtitan.protocols.state_dict_adapter import dtensor_safe


class AGPTMoEStateDictAdapter(MoEStateDictAdapter):
    """Convert AGPT GQA-MoE checkpoints to the custom HF model layout."""

    _TOP_MAP = {
        "tok_embeddings.weight": "model.embed_tokens.weight",
        "norm.weight": "model.norm.weight",
        "lm_head.weight": "lm_head.weight",
    }
    _LAYER_MAP = {
        "attention.qkv_linear.wq.weight": "self_attn.q_proj.weight",
        "attention.qkv_linear.wk.weight": "self_attn.k_proj.weight",
        "attention.qkv_linear.wv.weight": "self_attn.v_proj.weight",
        "attention.wo.weight": "self_attn.o_proj.weight",
        "attention_norm.weight": "input_layernorm.weight",
        "ffn_norm.weight": "post_attention_layernorm.weight",
        "moe.router.gate.weight": "mlp.gate.weight",
        "moe.shared_experts.w1.weight": "mlp.shared_experts.gate_proj.weight",
        "moe.shared_experts.w3.weight": "mlp.shared_experts.up_proj.weight",
        "moe.shared_experts.w2.weight": "mlp.shared_experts.down_proj.weight",
        "moe.expert_bias_E": "mlp.expert_bias",
    }

    def __init__(self, model_config, hf_assets_path):
        super().__init__(model_config, hf_assets_path)
        self._validate_hf_rope_config(ComplexRoPE.Config)
        attention = model_config.layers[0].attention
        self.n_heads = attention.n_heads
        self.n_kv_heads = attention.n_kv_heads or attention.n_heads
        self.head_dim = attention.head_dim or model_config.dim // self.n_heads
        self._is_cos_sin = False
        self.num_layers = len(model_config.layers)
        self.num_experts = model_config.layers[0].moe.num_experts
        from torchtitan.models.common.feed_forward import FeedForward
        from . import _LegacyInterleavedColumnParallelLinear

        self._interleaved_keys = {
            f"{fqn}.w13.{name}"
            for fqn, config, _, _ in model_config.traverse(FeedForward.Config)
            if isinstance(config.w13, _LegacyInterleavedColumnParallelLinear.Config)
            for name in ("weight", "bias")
        }

    def _split_stacked_linear(self, state_dict, *, fused_key, logical_keys, dim):
        if fused_key not in self._interleaved_keys:
            return super()._split_stacked_linear(
                state_dict, fused_key=fused_key, logical_keys=logical_keys, dim=dim
            )
        if fused_key not in state_dict:
            return
        tensor = state_dict.pop(fused_key)
        if isinstance(tensor, DTensor):
            self._stacked_linear_sharding[fused_key] = (
                tensor.device_mesh,
                tensor.placements,
            )
            tensor = tensor.redistribute(
                tensor.device_mesh, [Replicate()] * tensor.device_mesh.ndim
            )
        # Native shared FFNs alternate gate/up rows in a [2F, D] parameter.
        projections = tensor.unflatten(0, (-1, 2)).unbind(1)
        state_dict.update(zip(logical_keys, projections, strict=True))

    def _stack_logical_linears(self, state_dict, *, fused_key, logical_keys, dim):
        if fused_key not in self._interleaved_keys:
            return super()._stack_logical_linears(
                state_dict, fused_key=fused_key, logical_keys=logical_keys, dim=dim
            )
        if not all(key in state_dict for key in logical_keys):
            return
        fused = torch.stack([state_dict.pop(key) for key in logical_keys], dim=1)
        fused = fused.flatten(0, 1)
        if fused_key in self._stacked_linear_sharding:
            mesh, placements = self._stacked_linear_sharding[fused_key]
            fused = fused.redistribute(mesh, placements)
        state_dict[fused_key] = fused

    def hf_dtype_overrides(self):
        return {
            f"model.layers.{layer}.mlp.expert_bias": torch.float32
            for layer in range(self.num_layers)
        }

    @dtensor_safe
    def _permute(self, weight, num_heads, output_dim=None):
        output_dim = weight.shape[0] if output_dim is None else output_dim
        return (
            weight.view(num_heads, output_dim // num_heads // 2, 2, weight.shape[1])
            .transpose(1, 2)
            .reshape(output_dim, weight.shape[1])
            .clone()
        )

    @dtensor_safe
    def _reverse_permute(self, weight, num_heads, output_dim=None):
        output_dim = weight.shape[0] if output_dim is None else output_dim
        return (
            weight.view(num_heads, 2, output_dim // num_heads // 2, weight.shape[1])
            .transpose(1, 2)
            .reshape(output_dim, weight.shape[1])
        )

    def _legacy_keys(self):
        keys = set(self._TOP_MAP)
        suffixes = (
            "attention.qkv_linear.wq.weight",
            "attention.qkv_linear.wk.weight",
            "attention.qkv_linear.wv.weight",
            "attention.wo.weight",
            "attention_norm.weight",
            "ffn_norm.weight",
            "moe.experts.aurora_gate",
            "moe.experts.aurora_up",
            "moe.experts.aurora_down",
            "moe.router.gate.weight",
            "moe.shared_experts.w1.weight",
            "moe.shared_experts.w2.weight",
            "moe.shared_experts.w3.weight",
            "moe.expert_bias",
        )
        keys.update(
            f"layers.{layer}.{suffix}"
            for layer in range(self.num_layers)
            for suffix in suffixes
        )
        return keys

    def _hf_keys(self):
        keys = set(self._TOP_MAP.values())
        for layer in range(self.num_layers):
            keys.update(
                f"model.layers.{layer}.{suffix}" for suffix in self._LAYER_MAP.values()
            )
            keys.update(
                f"model.layers.{layer}.mlp.experts.{expert}.{projection}_proj.weight"
                for expert in range(self.num_experts)
                for projection in ("gate", "up", "down")
            )
        return keys

    def _validate_hf_keys(self, keys):
        expected = self._hf_keys()
        actual = set(keys)
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            raise RuntimeError(
                f"AGPT MoE HF schema mismatch; missing={missing[:8]}, "
                f"unexpected={unexpected[:8]}"
            )

    def checkpoint_state_dict(self, metadata):
        marker = "layers.0.moe.experts.aurora_gate"
        if marker not in metadata:
            return None
        expected = self._legacy_keys()
        training_components = {"optimizer", "lr_scheduler", "train_state", "dataloader"}
        actual = {
            key for key in metadata if key.split(".", 1)[0] not in training_components
        }
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            raise RuntimeError(
                f"Legacy Sonic schema mismatch; missing={missing[:8]}, "
                f"unexpected={unexpected[:8]}"
            )
        return {
            key: torch.empty(value.size, dtype=value.properties.dtype)
            for key, value in metadata.items()
            if key in expected
        }

    def _attention_to_hf(self, suffix, value):
        if suffix == "attention.qkv_linear.wq.weight":
            return self._permute(value, self.n_heads)
        if suffix == "attention.qkv_linear.wk.weight":
            return self._permute(
                value, self.n_kv_heads, self.head_dim * self.n_kv_heads
            )
        return value

    def _map_logical_to_hf(self, state_dict):
        result = {}
        for key, value in state_dict.items():
            if key in self._TOP_MAP:
                result[self._TOP_MAP[key]] = value
                continue
            match = re.fullmatch(r"layers\.(\d+)\.(.+)", key)
            if match is None:
                raise KeyError(f"Unsupported AGPT MoE state key: {key}")
            layer, suffix = match.groups()
            expert = re.fullmatch(
                r"moe\.routed_experts\.(w1_EFD|w3_EFD|w2\.weight)", suffix
            )
            if expert is not None:
                projection = {
                    "w1_EFD": "gate_proj",
                    "w3_EFD": "up_proj",
                    "w2.weight": "down_proj",
                }[expert.group(1)]
                for expert_id, expert_weight in enumerate(value.unbind(0)):
                    result[
                        f"model.layers.{layer}.mlp.experts.{expert_id}."
                        f"{projection}.weight"
                    ] = expert_weight
                continue
            if suffix not in self._LAYER_MAP:
                raise KeyError(f"Unsupported AGPT MoE layer key: {key}")
            value = self._attention_to_hf(suffix, value)
            result[f"model.layers.{layer}.{self._LAYER_MAP[suffix]}"] = value
        return result

    def _legacy_to_hf(self, state_dict):
        logical = {}
        expert_names = {
            "aurora_gate": "w1_EFD",
            "aurora_up": "w3_EFD",
            "aurora_down": "w2.weight",
        }
        for key, value in state_dict.items():
            expert = re.fullmatch(
                r"layers\.(\d+)\.moe\.experts\.(aurora_gate|aurora_up|aurora_down)",
                key,
            )
            if expert is not None:
                layer, name = expert.groups()
                logical[
                    f"layers.{layer}.moe.routed_experts.{expert_names[name]}"
                ] = value.transpose(-2, -1)
            elif key.endswith(".moe.expert_bias"):
                logical[key.removesuffix("expert_bias") + "expert_bias_E"] = value
            else:
                logical[key] = value
        return self._map_logical_to_hf(logical)

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        if any(key.endswith("moe.experts.aurora_gate") for key in state_dict):
            result = self._legacy_to_hf(state_dict)
        else:
            logical = self._native_fused_linears_to_hf(
                state_dict, split_routed_experts=True
            )
            result = self._map_logical_to_hf(logical)
        self._validate_hf_keys(result)
        return result

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        self._validate_hf_keys(hf_state_dict)
        reverse_top = {value: key for key, value in self._TOP_MAP.items()}
        reverse_layer = {value: key for key, value in self._LAYER_MAP.items()}
        logical = {}
        expert_weights = {}
        for key, value in hf_state_dict.items():
            if key in reverse_top:
                logical[reverse_top[key]] = value
                continue
            expert = re.fullmatch(
                r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
                r"(gate_proj|up_proj|down_proj)\.weight",
                key,
            )
            if expert is not None:
                layer, expert_id, projection = expert.groups()
                expert_weights.setdefault((layer, projection), {})[
                    int(expert_id)
                ] = value
                continue
            match = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", key)
            if match is None or match.group(2) not in reverse_layer:
                raise KeyError(f"Unsupported AGPT MoE HF key: {key}")
            layer, hf_suffix = match.groups()
            suffix = reverse_layer[hf_suffix]
            if suffix == "attention.qkv_linear.wq.weight":
                value = self._reverse_permute(value, self.n_heads)
            elif suffix == "attention.qkv_linear.wk.weight":
                value = self._reverse_permute(
                    value, self.n_kv_heads, self.head_dim * self.n_kv_heads
                )
            logical[f"layers.{layer}.{suffix}"] = value
        projection_map = {
            "gate_proj": "w1_EFD",
            "up_proj": "w3_EFD",
            "down_proj": "w2.weight",
        }
        for (layer, projection), weights in expert_weights.items():
            if set(weights) != set(range(self.num_experts)):
                raise ValueError(
                    f"Layer {layer} {projection} does not contain all experts"
                )
            logical[
                f"layers.{layer}.moe.routed_experts.{projection_map[projection]}"
            ] = torch.stack([weights[index] for index in range(self.num_experts)])
        return self._native_fused_linears_from_hf(logical, fuse_routed_experts=True)

    def write_hf_assets(self, output_dir, export_dtype):
        from torchtitan.experiments.ezpz.eval.hf_agpt_moe.assets import write_hf_assets

        write_hf_assets(
            output_dir,
            self.model_config,
            self.hf_assets_path,
            export_dtype,
        )

    def validate_hf_assets(self):
        from torchtitan.experiments.ezpz.eval.hf_agpt_moe.assets import (
            validate_tokenizer,
        )

        validate_tokenizer(self.hf_assets_path, self.model_config.vocab_size)
