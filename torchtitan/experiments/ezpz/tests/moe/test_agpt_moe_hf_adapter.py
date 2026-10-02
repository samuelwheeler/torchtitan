from types import SimpleNamespace

import pytest
import torch

from torchtitan.experiments.ezpz.moe.agpt_hf_state_dict_adapter import (
    AGPTMoEStateDictAdapter,
)
from torchtitan.models.common.rope import ComplexRoPE


def _adapter():
    attention = SimpleNamespace(
        n_heads=2,
        n_kv_heads=1,
        head_dim=4,
        rope=ComplexRoPE.Config(dim=4, max_context_length=16),
    )
    layer = SimpleNamespace(attention=attention, moe=SimpleNamespace(num_experts=2))
    config = SimpleNamespace(layers=[layer], dim=8)
    return AGPTMoEStateDictAdapter(config, None)


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


def test_legacy_sonic_to_hf_is_strict_and_transposes_experts():
    adapter = _adapter()
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


def test_hf_to_logical_native_reverses_permutation_and_stacks_experts():
    adapter = _adapter()
    source = _legacy_state()
    hf = adapter.to_hf(source)
    adapter._native_fused_linears_from_hf = lambda state, **kwargs: state
    native = adapter.from_hf(hf)

    assert torch.equal(
        native["layers.0.attention.qkv_linear.wq.weight"],
        source["layers.0.attention.qkv_linear.wq.weight"],
    )
    assert torch.equal(
        native["layers.0.moe.routed_experts.w3_EFD"],
        source["layers.0.moe.experts.aurora_up"].transpose(-2, -1),
    )


def test_legacy_checkpoint_schema_rejects_extra_model_keys():
    adapter = _adapter()
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
