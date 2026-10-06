# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch


def logical_state_dict(source, config):
    """Decode physical parameters directly, without any production adapter."""
    state = dict(source)
    for i, layer in enumerate(config.layers):
        prefix = f"layers.{i}."
        attn = layer.attention
        heads_per_kv = attn.n_heads // attn.n_kv_heads
        head_dim = attn.head_dim or config.dim // attn.n_heads
        packed = state.pop(prefix + "attention.qkv_linear.wqkv.weight").reshape(
            attn.n_kv_heads, heads_per_kv + 2, head_dim, config.dim
        )
        projections = (
            packed[:, :heads_per_kv],
            packed[:, heads_per_kv],
            packed[:, heads_per_kv + 1],
        )
        for name, value in zip(("wq", "wk", "wv"), projections, strict=True):
            state[prefix + f"attention.qkv_linear.{name}.weight"] = value.reshape(
                -1, config.dim
            )
        if layer.moe is None:
            gate, up = state.pop(prefix + "feed_forward.w13.weight").unbind(0)
            ffn = "feed_forward."
        else:
            interleaved = state.pop(prefix + "moe.shared_experts.w13.weight")
            gate, up = interleaved[::2], interleaved[1::2]
            ffn = "moe.shared_experts."
            routed = state.pop(prefix + "moe.routed_experts.w13.weight")
            state[prefix + "moe.routed_experts.w1_EFD"] = routed[:, 0]
            state[prefix + "moe.routed_experts.w3_EFD"] = routed[:, 1]
        state[prefix + ffn + "w1.weight"] = gate
        state[prefix + ffn + "w3.weight"] = up
    return state


def reference_hf_state_dict(source, config):
    state = logical_state_dict(source, config)
    result = {
        "model.embed_tokens.weight": state["tok_embeddings.weight"],
        "model.norm.weight": state["norm.weight"],
        "lm_head.weight": state["lm_head.weight"],
    }
    for i, layer in enumerate(config.layers):
        prefix, target = f"layers.{i}.", f"model.layers.{i}."
        pairs = {
            "attention.wo.weight": "self_attn.o_proj.weight",
            "attention_norm.weight": "input_layernorm.weight",
            "ffn_norm.weight": "post_attention_layernorm.weight",
        }
        for native, hf, heads in (
            ("wq", "q_proj", layer.attention.n_heads),
            ("wk", "k_proj", layer.attention.n_kv_heads),
            ("wv", "v_proj", None),
        ):
            value = state[prefix + f"attention.qkv_linear.{native}.weight"]
            if heads is not None:
                head_rows = value.reshape(heads, -1, config.dim)
                value = torch.cat(
                    (head_rows[:, ::2], head_rows[:, 1::2]), dim=1
                ).reshape(value.shape)
            result[target + f"self_attn.{hf}.weight"] = value
        native_ffn = "feed_forward" if layer.moe is None else "moe.shared_experts"
        hf_ffn = "mlp" if layer.moe is None else "mlp.shared_experts"
        for native, hf in (("w1", "gate_proj"), ("w3", "up_proj"), ("w2", "down_proj")):
            pairs[f"{native_ffn}.{native}.weight"] = f"{hf_ffn}.{hf}.weight"
        if layer.moe is not None:
            pairs["moe.router.gate.weight"] = "mlp.gate.weight"
            pairs["moe.expert_bias_E"] = "mlp.expert_bias"
            for native, hf in (
                ("w1_EFD", "gate_proj"),
                ("w3_EFD", "up_proj"),
                ("w2.weight", "down_proj"),
            ):
                for expert, value in enumerate(
                    state[prefix + f"moe.routed_experts.{native}"]
                ):
                    result[target + f"mlp.experts.{expert}.{hf}.weight"] = value
        result.update(
            {target + hf: state[prefix + native] for native, hf in pairs.items()}
        )
    return result
