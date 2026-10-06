import json
import shutil
from pathlib import Path

import sentencepiece as spm

from torchtitan.models.common.activation import Softmax
from torchtitan.models.common.attention import GQAttention
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import GroupedLinear, RouterGateLinear
from torchtitan.models.common.rope import ComplexRoPE


def _common_layer(model_config):
    first = model_config.layers[0]
    _validate_semantics(first)
    expected = _architecture(first)
    for layer in model_config.layers:
        if not isinstance(layer.attention, GQAttention.Config) or layer.moe is None:
            raise ValueError("AGPT MoE HF export requires GQA and MoE in every layer")
        _validate_semantics(layer)
        if _architecture(layer) != expected:
            raise ValueError("AGPT MoE HF export requires uniform transformer layers")
    return first


def _architecture(layer):
    attention, moe = layer.attention, layer.moe
    rope = attention.rope
    return (
        attention.dim,
        attention.n_heads,
        attention.n_kv_heads,
        attention.head_dim,
        rope.dim,
        rope.max_context_length,
        rope.theta,
        moe.num_experts,
        moe.router.top_k,
        moe.routed_experts.w13.out_features,
        moe.shared_experts.w13.out_features,
        layer.attention_norm.eps,
        layer.ffn_norm.eps,
    )


def _validate_semantics(layer):
    attention, moe = layer.attention, layer.moe
    router = moe.router
    dispatcher = moe.routed_experts.token_dispatcher
    if not isinstance(attention.rope, ComplexRoPE.Config):
        raise ValueError("AGPT MoE HF export supports ComplexRoPE only")
    if attention.rope.scaling != "none" or attention.qk_norm is not None:
        raise ValueError("Unsupported AGPT attention configuration")
    if not isinstance(router.score_func, Softmax.Config):
        raise ValueError("AGPT MoE HF export supports softmax routing only")
    if router.route_norm or router.route_scale != 1.0:
        raise ValueError("Unsupported AGPT routing normalization or scale")
    if dispatcher.score_before_experts or moe.load_balance_coeff is None:
        raise ValueError("Unsupported AGPT routing/bias semantics")
    if moe.shared_experts is None:
        raise ValueError("AGPT MoE HF export requires shared experts")
    bias_configs = (
        attention.qkv_linear.wqkv,
        attention.wo,
        router.gate,
        moe.shared_experts.w13,
        moe.shared_experts.w2,
    )
    if any(linear.bias for linear in bias_configs):
        raise ValueError("AGPT MoE HF export does not support linear biases")
    if not isinstance(router.gate, RouterGateLinear.Config):
        raise ValueError("AGPT MoE HF export requires an FP32 router gate")
    if not isinstance(moe.routed_experts.w13, GroupedLinear.Config):
        raise ValueError("AGPT MoE HF export requires grouped routed experts")
    if not isinstance(moe.shared_experts, FeedForward.Config):
        raise ValueError("AGPT MoE HF export requires a fused shared SwiGLU")


def _config(model_config, export_dtype):
    layer = _common_layer(model_config)
    attention, moe = layer.attention, layer.moe
    rope = attention.rope
    if model_config.enable_weight_tying or model_config.lm_head.bias:
        raise ValueError("AGPT MoE HF export requires an untied, bias-free LM head")
    return {
        "architectures": ["AGPTMoEForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_agpt_moe.AGPTMoEConfig",
            "AutoModelForCausalLM": "modeling_agpt_moe.AGPTMoEForCausalLM",
        },
        "model_type": "agpt_moe",
        "vocab_size": model_config.vocab_size,
        "hidden_size": model_config.dim,
        "intermediate_size": moe.routed_experts.w13.out_features,
        "moe_intermediate_size": moe.routed_experts.w13.out_features,
        "shared_expert_intermediate_size": moe.shared_experts.w13.out_features,
        "num_hidden_layers": len(model_config.layers),
        "num_attention_heads": attention.n_heads,
        "num_key_value_heads": attention.n_kv_heads,
        "head_dim": attention.head_dim or model_config.dim // attention.n_heads,
        "num_local_experts": moe.num_experts,
        "num_experts_per_tok": moe.router.top_k,
        "router_score_function": "softmax",
        "norm_topk_prob": False,
        "router_bias_selection": True,
        "router_scale": 1.0,
        "hidden_act": "silu",
        "max_position_embeddings": rope.max_context_length,
        "rope_theta": rope.theta,
        "rms_norm_eps": layer.attention_norm.eps,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "mlp_bias": False,
        "tie_word_embeddings": False,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": None,
        "torch_dtype": export_dtype,
    }


def validate_tokenizer(hf_assets_path, vocab_size):
    assets = Path(hf_assets_path) if hf_assets_path is not None else None
    if assets is None or not (assets / "tokenizer.model").is_file():
        raise FileNotFoundError("AGPT MoE export requires hf_assets_path/tokenizer.model")
    tokenizer = spm.SentencePieceProcessor(model_file=str(assets / "tokenizer.model"))
    if not 0 < tokenizer.get_piece_size() <= vocab_size:
        raise ValueError(
            f"Tokenizer vocabulary {tokenizer.get_piece_size()} exceeds model "
            f"vocabulary {vocab_size}"
        )
    if (
        (tokenizer.unk_id(), tokenizer.bos_id(), tokenizer.eos_id()) != (0, 1, 2)
        or [tokenizer.id_to_piece(i) for i in range(3)] != ["<unk>", "<s>", "</s>"]
        or tokenizer.pad_id() != -1
    ):
        raise ValueError("AGPT MoE export requires Llama SentencePiece special tokens")


def write_hf_assets(output_dir, model_config, hf_assets_path, export_dtype):
    validate_tokenizer(hf_assets_path, model_config.vocab_size)
    output_dir = Path(output_dir)
    assets = Path(hf_assets_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(__file__).parent
    for name in ("configuration_agpt_moe.py", "modeling_agpt_moe.py"):
        shutil.copy2(source_dir / name, output_dir / name)
    shutil.copy2(assets / "tokenizer.model", output_dir / "tokenizer.model")
    (output_dir / "config.json").write_text(
        json.dumps(_config(model_config, export_dtype), indent=2) + "\n"
    )
    tokenizer_config = {
        "add_bos_token": True,
        "add_eos_token": False,
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "model_max_length": model_config.max_context_length,
        "tokenizer_class": "LlamaTokenizer",
    }
    (output_dir / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2) + "\n"
    )
    (output_dir / "special_tokens_map.json").write_text(
        json.dumps(
            {"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>"},
            indent=2,
        )
        + "\n"
    )
