import pytest
import sentencepiece as spm
import torch

from torchtitan.experiments.ezpz.moe import (
    _dtensor_safe_fused_ffn_config,
    make_ezpz_experts_config,
    make_ezpz_router_config,
    model_registry,
)
from torchtitan.models.common import RMSNorm
from torchtitan.models.common.config_utils import make_gqa_config
from torchtitan.models.common.rope import ComplexRoPE


@pytest.fixture
def tiny_moe_config():
    config = model_registry("AGPT_2B_50K_MOE_sdpa_aurora_full_sonic")
    config.dim, config.vocab_size, config.max_context_length = 8, 32, 16
    config.tok_embeddings.num_embeddings = 32
    config.tok_embeddings.embedding_dim = 8
    config.norm.normalized_shape = 8
    config.lm_head.in_features, config.lm_head.out_features = 8, 32
    config.layers = config.layers[:1]
    layer = config.layers[0]
    init = {"weight": torch.nn.init.normal_}
    layer.attention = make_gqa_config(
        dim=8, n_heads=2, n_kv_heads=1,
        wqkv_param_init=init, wo_param_init=init,
        inner_attention=layer.attention.inner_attention,
        rope=ComplexRoPE.Config(dim=4, max_context_length=16),
    )
    layer.attention_norm = RMSNorm.Config(normalized_shape=8)
    layer.ffn_norm = RMSNorm.Config(normalized_shape=8)
    layer.moe.num_experts = 2
    layer.moe.router = make_ezpz_router_config(
        dim=8, num_experts=2, gate_param_init=init, top_k=1,
        score_func="softmax", route_norm=False,
    )
    layer.moe.routed_experts = make_ezpz_experts_config(
        dim=8, hidden_dim=6, num_experts=2, top_k=1,
        score_before_experts=False, compute_backend="for_loop",
        param_init={name: torch.nn.init.normal_ for name in ("w1_EFD", "w2_EDF", "w3_EFD")},
    )
    layer.moe.shared_experts = _dtensor_safe_fused_ffn_config(
        dim=8, hidden_dim=6, w1_param_init=init, w2w3_param_init=init,
    )
    return config


@pytest.fixture
def tokenizer_dir(tmp_path):
    assets = tmp_path / "tokenizer"
    assets.mkdir()
    corpus = assets / "corpus.txt"
    corpus.write_text("Aurora trains sparse language models.\n" * 10)
    spm.SentencePieceTrainer.train(
        input=str(corpus), model_prefix=str(assets / "tokenizer"),
        vocab_size=32, hard_vocab_limit=False, minloglevel=2,
    )
    return assets
