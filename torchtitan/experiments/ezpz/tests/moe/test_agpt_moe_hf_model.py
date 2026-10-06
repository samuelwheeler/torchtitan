import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from torchtitan.experiments.ezpz.eval.hf_agpt_moe.configuration_agpt_moe import (
    AGPTMoEConfig,
)
from torchtitan.experiments.ezpz.eval.hf_agpt_moe.modeling_agpt_moe import (
    AGPTMoEForCausalLM,
    AGPTSparseMoE,
)


def _config(**overrides):
    values = dict(
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=32,
        moe_intermediate_size=6,
        shared_expert_intermediate_size=8,
        num_local_experts=3,
        num_experts_per_tok=2,
        rms_norm_eps=1e-5,
    )
    values.update(overrides)
    return AGPTMoEConfig(**values)


def test_sparse_moe_uses_bias_for_selection_but_original_unnormalized_scores():
    torch.manual_seed(1)
    moe = AGPTSparseMoE(_config(hidden_size=2, moe_intermediate_size=3))
    tokens = torch.tensor([[[1.0, -0.5]]])
    with torch.no_grad():
        moe.gate.weight.copy_(torch.tensor([[2.0, 0.0], [1.0, 0.0], [0.0, 0.0]]))
        moe.expert_bias.copy_(torch.tensor([0.0, 0.0, 5.0]))

    scores = torch.softmax(F.linear(tokens.view(1, 2), moe.gate.weight).float(), -1)
    selected = torch.topk(scores + moe.expert_bias, 2, sorted=False).indices[0]
    assert set(selected.tolist()) == {0, 2}
    assert scores[0, selected].sum() < 1
    expected = moe.shared_experts(tokens.view(1, 2))
    for expert_id in selected:
        expected = expected + moe.experts[expert_id](tokens.view(1, 2)) * scores[
            0, expert_id
        ]
    torch.testing.assert_close(moe(tokens).view(1, 2), expected)


def test_hf_causal_lm_supports_forward_and_generation():
    model = AGPTMoEForCausalLM(_config()).eval()
    input_ids = torch.tensor([[1, 2, 3]])
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        output = model(input_ids, attention_mask=attention_mask)
        generated = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=2,
            do_sample=False,
        )
    assert output.logits.shape == (1, 3, 32)
    assert generated.shape == (1, 5)
    assert model.model.layers[0].mlp.expert_bias.dtype is torch.float32


def test_auto_model_remote_code_round_trip(tmp_path):
    config = _config()
    config.auto_map = {
        "AutoConfig": "configuration_agpt_moe.AGPTMoEConfig",
        "AutoModelForCausalLM": "modeling_agpt_moe.AGPTMoEForCausalLM",
    }
    model = AGPTMoEForCausalLM(config).eval()
    with torch.no_grad():
        model.model.layers[0].mlp.expert_bias.copy_(
            torch.tensor([0.001003, 0.001004, -0.002007])
        )
    model.save_pretrained(tmp_path)
    source = Path(__file__).parents[2] / "eval/hf_agpt_moe"
    for name in ("configuration_agpt_moe.py", "modeling_agpt_moe.py"):
        shutil.copy2(f"{source}/{name}", tmp_path / name)

    loaded = AutoModelForCausalLM.from_pretrained(
        tmp_path, trust_remote_code=True
    ).eval()
    input_ids = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        expected = model(input_ids).logits
        actual = loaded(input_ids).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)
    bf16_model = AutoModelForCausalLM.from_pretrained(
        tmp_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    assert bf16_model.model.layers[0].mlp.gate.weight.dtype == torch.bfloat16
    assert bf16_model.model.layers[0].mlp.expert_bias.dtype == torch.float32
    torch.testing.assert_close(
        bf16_model.model.layers[0].mlp.expert_bias,
        model.model.layers[0].mlp.expert_bias, rtol=0, atol=0,
    )


def test_native_and_hf_logits_match(tiny_moe_config, monkeypatch):
    from torchtitan.experiments.ezpz import agpt
    from torchtitan.experiments.ezpz.eval.hf_agpt_moe.assets import _config
    from torchtitan.experiments.ezpz.moe.agpt_hf_state_dict_adapter import (
        AGPTMoEStateDictAdapter,
    )

    torch.manual_seed(17)
    native = tiny_moe_config.build().eval()
    native.init_states(buffer_device=torch.device("cpu"))
    # The native for_loop expert backend computes in BF16 even on CPU.
    for parameter in native.parameters():
        parameter.data = parameter.data.bfloat16()
    adapter = AGPTMoEStateDictAdapter(tiny_moe_config, None)
    hf = AGPTMoEForCausalLM(AGPTMoEConfig(**_config(tiny_moe_config, "bfloat16"))).eval()
    for parameter in hf.parameters():
        parameter.data = parameter.data.bfloat16()
    hf.load_state_dict(adapter.to_hf(native.state_dict()), strict=True)
    tokens = torch.tensor([1, 3, 5, 7])
    monkeypatch.setattr(agpt, "_EZPZ_MAX_CONTEXT_LENGTH", tokens.numel())
    with torch.inference_mode():
        expected = native(tokens)
        actual = hf(tokens[None], use_cache=False).logits[0]
    actual, expected = actual.float(), expected.float()
    relative_rms = (actual - expected).square().mean().sqrt() / expected.square().mean().sqrt()
    assert relative_rms < 0.02
    assert F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0) > 0.999
    assert torch.equal(actual.argmax(-1), expected.argmax(-1))
