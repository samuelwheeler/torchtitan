# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch

from torchtitan.experiments.ezpz.moe.agpt_hf_state_dict_adapter import (
    AGPTMoEStateDictAdapter,
)


def _legacy_state():
    prefix = "layers.0."
    state = {
        "tok_embeddings.weight": torch.randn(7, 8),
        "norm.weight": torch.randn(8),
        "lm_head.weight": torch.randn(7, 8),
        f"{prefix}attention.qkv_linear.wq.weight": torch.arange(64).view(8, 8),
        f"{prefix}attention.qkv_linear.wk.weight": torch.arange(32).view(4, 8),
        f"{prefix}attention.qkv_linear.wv.weight": torch.randn(4, 8),
        f"{prefix}attention.wo.weight": torch.randn(8, 8),
        f"{prefix}attention_norm.weight": torch.randn(8),
        f"{prefix}ffn_norm.weight": torch.randn(8),
        f"{prefix}moe.experts.aurora_gate": torch.randn(2, 8, 6),
        f"{prefix}moe.experts.aurora_up": torch.randn(2, 8, 6),
        f"{prefix}moe.experts.aurora_down": torch.randn(2, 6, 8),
        f"{prefix}moe.router.gate.weight": torch.randn(2, 8),
        f"{prefix}moe.shared_experts.w1.weight": torch.randn(6, 8),
        f"{prefix}moe.shared_experts.w2.weight": torch.randn(8, 6),
        f"{prefix}moe.shared_experts.w3.weight": torch.randn(6, 8),
        f"{prefix}moe.expert_bias": torch.randn(2),
    }
    return state


def test_legacy_sonic_to_hf_is_strict_and_transposes_experts(tiny_moe_config):
    adapter = AGPTMoEStateDictAdapter(tiny_moe_config, None)
    source = _legacy_state()
    hf = adapter.to_hf(source)

    assert set(hf) == adapter._hf_keys()
    assert torch.equal(
        hf["model.layers.0.mlp.experts.1.gate_proj.weight"],
        source["layers.0.moe.experts.aurora_gate"][1].T,
    )
    assert torch.equal(
        hf["model.layers.0.mlp.experts.0.down_proj.weight"],
        source["layers.0.moe.experts.aurora_down"][0].T,
    )
    expected_q = source["layers.0.attention.qkv_linear.wq.weight"].view(
        2, 2, 2, 8
    ).transpose(1, 2).reshape(8, 8)
    assert torch.equal(hf["model.layers.0.self_attn.q_proj.weight"], expected_q)


def test_hf_to_native_reverses_permutation_and_stacks_experts(tiny_moe_config):
    adapter = AGPTMoEStateDictAdapter(tiny_moe_config, None)
    source = _legacy_state()
    hf = adapter.to_hf(source)
    native = adapter.from_hf(hf)
    logical = adapter._native_fused_linears_to_hf(native, split_routed_experts=True)

    assert torch.equal(
        logical["layers.0.attention.qkv_linear.wq.weight"],
        source["layers.0.attention.qkv_linear.wq.weight"],
    )
    assert torch.equal(
        logical["layers.0.moe.routed_experts.w3_EFD"],
        source["layers.0.moe.experts.aurora_up"].transpose(-2, -1),
    )


def test_legacy_checkpoint_schema_rejects_extra_model_keys(tiny_moe_config):
    adapter = AGPTMoEStateDictAdapter(tiny_moe_config, None)
    metadata = {
        key: SimpleNamespace(
            size=torch.Size([1]), properties=SimpleNamespace(dtype=torch.float32)
        )
        for key in adapter._legacy_keys()
    }
    assert set(adapter.checkpoint_state_dict(metadata)) == adapter._legacy_keys()
    metadata["layers.0.unexpected.weight"] = next(iter(metadata.values()))
    with pytest.raises(RuntimeError, match="unexpected"):
        adapter.checkpoint_state_dict(metadata)


def test_current_fused_native_round_trip_and_shared_forward(tiny_moe_config):
    model = tiny_moe_config.build()
    model.init_states(buffer_device=torch.device("cpu"))
    source = model.state_dict()
    adapter = AGPTMoEStateDictAdapter(tiny_moe_config, None)
    hf = adapter.to_hf(source)
    restored = adapter.from_hf(hf)
    assert set(restored) == set(source)
    for key in source:
        torch.testing.assert_close(restored[key], source[key], rtol=0, atol=0)
    shared = model.layers["0"].moe.shared_experts
    tokens = torch.randn(5, 8)
    gate = hf["model.layers.0.mlp.shared_experts.gate_proj.weight"]
    up = hf["model.layers.0.mlp.shared_experts.up_proj.weight"]
    down = hf["model.layers.0.mlp.shared_experts.down_proj.weight"]
    expected = torch.nn.functional.linear(
        torch.nn.functional.silu(tokens @ gate.T) * (tokens @ up.T), down
    )
    torch.testing.assert_close(shared(tokens), expected)


def test_registered_full_model_round_trip_shapes():
    from torchtitan.experiments.ezpz.moe import model_registry

    config = model_registry("AGPT_2B_50K_MOE_sdpa_aurora_full_sonic")
    with torch.device("meta"):
        source = config.build().state_dict()
    adapter = AGPTMoEStateDictAdapter(config, None)
    restored = adapter.from_hf(adapter.to_hf(source))
    assert set(restored) == set(source)
    assert {key: value.shape for key, value in restored.items()} == {
        key: value.shape for key, value in source.items()
    }
