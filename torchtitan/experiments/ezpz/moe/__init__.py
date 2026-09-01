# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from functools import partial
from typing import Literal

import torch.nn as nn

from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.models.common import (
    Embedding,
    GQAttention,
    Linear,
    RMSNorm,
    RoPE,
    TransformerBlock,
)
from torchtitan.experiments.ezpz.agpt import (
    _default_inner_attention,
    _depth_init as _agpt_depth_init,
    _ezpz_get_attention_config,
    _linear_init as _agpt_linear_init,
)
from torchtitan.models.common.config_utils import (
    make_experts_config,
    make_ffn_config,
    make_gqa_config,
    make_moe_config,
    make_router_config,
)
from torchtitan.models.common.param_init import depth_scaled_std
from torchtitan.protocols.model_spec import ModelSpec

from .model import Attention, moeModel, moeTransformerBlock

from .parallelize import parallelize_moe
from .state_dict_adapter import moeStateDictAdapter

__all__ = [
    "parallelize_moe",
    "moeModel",
    "moe_configs",
]


_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}


def _output_linear_init(dim: int) -> dict[str, Callable]:
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


def _depth_init(layer_id: int) -> dict[str, Callable]:
    return {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }


def _depth_experts_init(layer_id: int) -> dict[str, Callable]:
    return {
        "w1": partial(nn.init.trunc_normal_, std=0.02),
        "w2": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "w3": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
    }


def _make_moe_attn_config(
    *,
    layer_id: int,
    dim: int,
    n_heads: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    mscale: float = 1.0,
    attn_backend: str = "sdpa",
) -> Attention.Config:
    """Build a fully-specified MoE MLA Attention.Config.

    All Linear and RMSNorm sub-configs have their dimensional fields set.
    When q_lora_rank == 0, sets wq (not wq_a/wq_b).
    When q_lora_rank > 0, sets wq_a/wq_b (not wq).
    """
    _inner, _mask = _ezpz_get_attention_config(attn_backend)
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

    if q_lora_rank == 0:
        wq = Linear.Config(
            in_features=dim,
            out_features=n_heads * qk_head_dim,
            param_init=_LINEAR_INIT,
        )
        wq_a = None
        wq_b = None
        # q_norm is unused when q_lora_rank == 0 (never built), but the field is
        # required on Attention.Config so we supply a placeholder.
        q_norm = RMSNorm.Config(normalized_shape=1, param_init=_NORM_INIT)
    else:
        wq = None
        wq_a = Linear.Config(
            in_features=dim,
            out_features=q_lora_rank,
            param_init=_LINEAR_INIT,
        )
        wq_b = Linear.Config(
            in_features=q_lora_rank,
            out_features=n_heads * qk_head_dim,
            param_init=_LINEAR_INIT,
        )
        q_norm = RMSNorm.Config(normalized_shape=q_lora_rank, param_init=_NORM_INIT)

    return Attention.Config(
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        mscale=mscale,
        wq=wq,
        wq_a=wq_a,
        wq_b=wq_b,
        q_norm=q_norm,
        wkv_a=Linear.Config(
            in_features=dim,
            out_features=kv_lora_rank + qk_rope_head_dim,
            param_init=_LINEAR_INIT,
        ),
        kv_norm=RMSNorm.Config(normalized_shape=kv_lora_rank, param_init=_NORM_INIT),
        wkv_b=Linear.Config(
            in_features=kv_lora_rank,
            out_features=n_heads * (qk_nope_head_dim + v_head_dim),
            param_init=_LINEAR_INIT,
        ),
        wo=Linear.Config(
            in_features=n_heads * v_head_dim,
            out_features=dim,
            param_init=_depth_init(layer_id),
        ),
        inner_attention=_inner,
        mask_type=_mask,
    )


def _build_moe_layers(
    *,
    n_layers: int,
    n_dense_layers: int,
    dim: int,
    n_heads: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    mscale: float,
    dense_hidden_dim: int,
    moe_hidden_dim: int,
    num_experts: int,
    num_shared_experts: int,
    router_top_k: int,
    router_score_func: Literal["sigmoid", "softmax"],
    router_num_expert_groups: int | None = None,
    router_num_limited_groups: int | None = None,
    router_route_scale: float = 1.0,
    router_route_norm: bool = False,
    score_before_experts: bool = False,
    attn_backend: str = "sdpa",
    moe_comm_backend: str = "standard",
) -> list[TransformerBlock.Config]:
    """Build the list of per-layer TransformerBlock configs.

    Layers with layer_id < n_dense_layers get a dense FeedForward and no MoE.
    Layers with layer_id >= n_dense_layers get a MoE and no FeedForward.

    Router and expert inits are constructed per-layer so depth-scaled
    initializers are correct for each layer's position.
    """
    layers = []
    for layer_id in range(n_layers):
        attn_cfg = _make_moe_attn_config(
            layer_id=layer_id,
            dim=dim,
            n_heads=n_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            mscale=mscale,
            attn_backend=attn_backend,
        )

        if layer_id < n_dense_layers:
            ffn_cfg = make_ffn_config(
                dim=dim,
                hidden_dim=dense_hidden_dim,
                w1_param_init=_LINEAR_INIT,
                w2w3_param_init=_depth_init(layer_id),
            )
            moe_cfg = None
        else:
            ffn_cfg = None
            moe_cfg = make_moe_config(
                num_experts=num_experts,
                router=make_router_config(
                    dim=dim,
                    num_experts=num_experts,
                    gate_param_init=_depth_init(layer_id),
                    top_k=router_top_k,
                    score_func=router_score_func,
                    num_expert_groups=router_num_expert_groups,
                    num_limited_groups=router_num_limited_groups,
                    route_scale=router_route_scale,
                    route_norm=router_route_norm,
                ),
                experts=make_experts_config(
                    dim=dim,
                    hidden_dim=moe_hidden_dim,
                    num_experts=num_experts,
                    top_k=router_top_k,
                    score_before_experts=score_before_experts,
                    comm_backend=moe_comm_backend,
                    param_init=_depth_experts_init(layer_id),
                ),
                shared_experts=make_ffn_config(
                    dim=dim,
                    hidden_dim=moe_hidden_dim * num_shared_experts,
                    w1_param_init=_LINEAR_INIT,
                    w2w3_param_init=_depth_init(layer_id),
                ),
            )

        layers.append(
            moeTransformerBlock.Config(
                attention=attn_cfg,
                attention_norm=RMSNorm.Config(
                    normalized_shape=dim, param_init=_NORM_INIT
                ),
                ffn_norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
                feed_forward=ffn_cfg,
                moe=moe_cfg,
            )
        )
    return layers


def _debugmodel() -> moeModel.Config:
    dim = 256
    n_layers = 6
    vocab_size = 256128
    n_heads = 16
    moe_hidden_dim = 256
    num_shared_experts = 2
    dense_hidden_dim = 1024
    rope_dim = 64
    num_experts = 8
    n_dense_layers = 1

    layers = _build_moe_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=0.70,
        dense_hidden_dim=dense_hidden_dim,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=3,
        router_score_func="softmax",
        score_before_experts=False,
    )
    return moeModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=rope_dim,
            max_seq_len=4096 * 4,
            theta=10000.0,
            backend="complex",
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
        layers=layers,
    )


def _debugmodel_flex_attn() -> moeModel.Config:
    dim = 256
    n_layers = 6
    vocab_size = 256128
    n_heads = 16
    moe_hidden_dim = 256
    num_shared_experts = 2
    dense_hidden_dim = 1024
    rope_dim = 64
    num_experts = 8
    n_dense_layers = 1

    layers = _build_moe_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=0.70,
        dense_hidden_dim=dense_hidden_dim,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=3,
        router_score_func="softmax",
        score_before_experts=False,
        attn_backend="flex",
    )
    return moeModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=rope_dim,
            max_seq_len=4096 * 4,
            theta=10000.0,
            backend="complex",
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
        layers=layers,
    )


def _small() -> moeModel.Config:
    dim = 2048
    n_layers = 24
    vocab_size = 256128
    n_heads = 16
    moe_hidden_dim = 512
    num_shared_experts = 2
    dense_hidden_dim = 4096
    rope_dim = 64
    num_experts = 64
    n_dense_layers = 1

    layers = _build_moe_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=0.70,
        dense_hidden_dim=dense_hidden_dim,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=6,
        router_score_func="softmax",
        router_route_norm=True,
        router_route_scale=1.0,
        score_before_experts=False,
        attn_backend="flex",
    )
    return moeModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=rope_dim,
            max_seq_len=256128,
            theta=50000.0,
            backend="complex",
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
        layers=layers,
    )


def _16b() -> moeModel.Config:
    dim = 2048
    n_layers = 27
    vocab_size = 256128
    n_heads = 16
    moe_hidden_dim = 1408
    num_shared_experts = 2
    dense_hidden_dim = 10944
    rope_dim = 64
    num_experts = 64
    n_dense_layers = 1

    layers = _build_moe_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=0.70,
        dense_hidden_dim=dense_hidden_dim,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=6,
        router_score_func="softmax",
        score_before_experts=False,
        attn_backend="flex",
    )
    return moeModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=rope_dim,
            max_seq_len=4096 * 4,
            theta=10000.0,
            backend="complex",
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
        layers=layers,
    )


def _236b() -> moeModel.Config:
    dim = 5120
    n_layers = 60
    vocab_size = 256128
    n_heads = 128
    q_lora_rank = 1536
    moe_hidden_dim = 1536
    num_shared_experts = 2
    dense_hidden_dim = 12288
    rope_dim = 64
    num_experts = 160
    n_dense_layers = 1

    layers = _build_moe_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=1.0,
        dense_hidden_dim=dense_hidden_dim,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=6,
        router_score_func="softmax",
        router_num_expert_groups=8,
        router_num_limited_groups=3,
        router_route_scale=16.0,
        score_before_experts=False,
        attn_backend="flex",
    )
    return moeModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=rope_dim,
            max_seq_len=4096 * 4,
            theta=10000.0,
            backend="complex",
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
        layers=layers,
    )


def _671b() -> moeModel.Config:
    dim = 7168
    n_layers = 61
    vocab_size = 256128
    n_heads = 128
    q_lora_rank = 1536
    moe_hidden_dim = 2048
    num_shared_experts = 1
    dense_hidden_dim = 18432
    rope_dim = 64
    num_experts = 256
    n_dense_layers = 3

    layers = _build_moe_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=1.0,
        dense_hidden_dim=dense_hidden_dim,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=8,
        router_score_func="sigmoid",
        router_num_expert_groups=8,
        router_num_limited_groups=4,
        router_route_scale=2.5,
        router_route_norm=True,
        score_before_experts=False,
        attn_backend="flex",
    )
    return moeModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=rope_dim,
            max_seq_len=4096 * 4,
            theta=10000.0,
            backend="complex",
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
        layers=layers,
    )


def _500m() -> moeModel.Config:
    """~500M active params. Halfway between debugmodel (48M) and 10B_2B."""
    dim = 512
    n_layers = 12
    vocab_size = 256128
    n_heads = 16
    moe_hidden_dim = 512
    num_shared_experts = 2
    dense_hidden_dim = 2048
    rope_dim = 64
    num_experts = 16
    n_dense_layers = 1

    layers = _build_moe_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=0.70,
        dense_hidden_dim=dense_hidden_dim,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=3,
        router_score_func="softmax",
        score_before_experts=False,
    )
    return moeModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=rope_dim,
            max_seq_len=4096 * 4,
            theta=10000.0,
            backend="complex",
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
        layers=layers,
    )


def _2b() -> moeModel.Config:
    """~2B active params. Between small and 10B_2B."""
    dim = 1024
    n_layers = 18
    vocab_size = 256128
    n_heads = 16
    moe_hidden_dim = 1024
    num_shared_experts = 2
    dense_hidden_dim = 4096
    rope_dim = 64
    num_experts = 24
    n_dense_layers = 1

    layers = _build_moe_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=0.70,
        dense_hidden_dim=dense_hidden_dim,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=3,
        router_score_func="softmax",
        score_before_experts=False,
    )
    return moeModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=rope_dim,
            max_seq_len=4096 * 4,
            theta=10000.0,
            backend="complex",
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
        layers=layers,
    )


def _4b() -> moeModel.Config:
    """~4B total / ~1B active. Optimized for 12 XPU tiles per node.

    n_heads=12 divides evenly across 12 tiles for TP.
    """
    dim = 1536
    n_layers = 22
    vocab_size = 256128
    n_heads = 12
    moe_hidden_dim = 1024
    num_shared_experts = 2
    dense_hidden_dim = 6144
    rope_dim = 64
    num_experts = 24
    n_dense_layers = 1

    layers = _build_moe_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=0.70,
        dense_hidden_dim=dense_hidden_dim,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=3,
        router_score_func="softmax",
        score_before_experts=False,
    )
    return moeModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=rope_dim,
            max_seq_len=4096 * 4,
            theta=10000.0,
            backend="complex",
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
        layers=layers,
    )


def _7b() -> moeModel.Config:
    """~7B total / ~1.5B active. Optimized for 12 XPU tiles per node.

    n_heads=24 divides by 2,3,4,6,12 for flexible TP.
    num_experts=36 matches 10B_2B routing complexity.
    """
    dim = 2048
    n_layers = 24
    vocab_size = 256128
    n_heads = 24
    moe_hidden_dim = 1280
    num_shared_experts = 2
    dense_hidden_dim = 8192
    rope_dim = 64
    num_experts = 36
    n_dense_layers = 1

    layers = _build_moe_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=0.70,
        dense_hidden_dim=dense_hidden_dim,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=3,
        router_score_func="softmax",
        score_before_experts=False,
    )
    return moeModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=rope_dim,
            max_seq_len=4096 * 4,
            theta=10000.0,
            backend="complex",
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
        layers=layers,
    )


def _10b_2b() -> moeModel.Config:
    dim = 2048
    n_layers = 27
    vocab_size = 256128
    n_heads = 16
    moe_hidden_dim = 1408
    num_shared_experts = 2
    dense_hidden_dim = 10944
    rope_dim = 64
    num_experts = 36
    n_dense_layers = 1

    layers = _build_moe_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=0.70,
        dense_hidden_dim=dense_hidden_dim,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=3,
        router_score_func="softmax",
        score_before_experts=False,
        attn_backend="flex",
    )
    return moeModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=rope_dim,
            max_seq_len=4096 * 4,
            theta=10000.0,
            backend="complex",
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
        layers=layers,
    )


def _10b_2b_sdpa() -> moeModel.Config:
    """10B_2B with SDPA instead of FlexAttention.

    Avoids FlexAttention Triton compilation and the fp32 autocast
    issue on XPU (torch.autocast doesn't support fp32 on XPU).
    """
    cfg = _10b_2b()
    sdpa_cfg = _default_inner_attention()
    for layer_cfg in cfg.layers:
        layer_cfg.attention.inner_attention = sdpa_cfg
        layer_cfg.attention.mask_type = "causal"
    return cfg


def _10b_2b_50k_sdpa() -> moeModel.Config:
    """~10.56B total / ~2.00B active with the AGPT 50K vocabulary.

    This is the MoE counterpart to ``agpt_2b_50k``: the much smaller
    embedding/output tables are reinvested in a 31-layer transformer
    backbone.  It retains the existing 36-expert, top-3, two-shared-expert
    routing geometry, which divides evenly over one Aurora node's 12 tiles.
    """
    dim = 2048
    n_layers = 31
    vocab_size = 50304
    n_heads = 16
    moe_hidden_dim = 1408
    num_shared_experts = 2
    dense_hidden_dim = 10944
    rope_dim = 64
    num_experts = 36
    n_dense_layers = 1

    layers = _build_moe_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=0.70,
        dense_hidden_dim=dense_hidden_dim,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=3,
        router_score_func="softmax",
        score_before_experts=False,
        attn_backend="sdpa",
    )
    return moeModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=rope_dim,
            max_seq_len=4096 * 4,
            theta=10000.0,
            backend="complex",
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
        layers=layers,
    )


def _10b_2b_50k_sdpa_for_loop() -> moeModel.Config:
    cfg = _10b_2b_50k_sdpa()
    for layer_cfg in cfg.layers:
        if layer_cfg.moe is not None:
            layer_cfg.moe.experts.compute_backend = "for_loop"
            layer_cfg.moe.experts.use_grouped_mm = False
    return cfg


def _10b_2b_50k_sdpa_aurora_sycl() -> moeModel.Config:
    cfg = _10b_2b_50k_sdpa()
    for layer_cfg in cfg.layers:
        if layer_cfg.moe is not None:
            layer_cfg.moe.experts.compute_backend = "aurora_sycl"
            layer_cfg.moe.experts.use_grouped_mm = False
    return cfg


def _10b_2b_50k_sdpa_aurora_full(backend: str) -> moeModel.Config:
    cfg = _10b_2b_50k_sdpa()
    for layer_cfg in cfg.layers:
        if layer_cfg.moe is not None:
            layer_cfg.moe.experts.compute_backend = backend
            layer_cfg.moe.experts.use_grouped_mm = False
    return cfg


def _10b_2b_50k_sdpa_aurora_full_loop() -> moeModel.Config:
    return _10b_2b_50k_sdpa_aurora_full("aurora_full_loop")


def _10b_2b_50k_sdpa_aurora_full_sonic() -> moeModel.Config:
    return _10b_2b_50k_sdpa_aurora_full("aurora_full_sonic")


def _10b_2b_50k_sdpa_aurora_full_1layer(backend: str) -> moeModel.Config:
    cfg = _10b_2b_50k_sdpa_aurora_full(backend)
    cfg.layers = [cfg.layers[1]]
    return cfg


def _10b_2b_50k_sdpa_aurora_full_loop_1layer() -> moeModel.Config:
    return _10b_2b_50k_sdpa_aurora_full_1layer("aurora_full_loop")


def _10b_2b_50k_sdpa_aurora_full_sonic_1layer() -> moeModel.Config:
    return _10b_2b_50k_sdpa_aurora_full_1layer("aurora_full_sonic")


def _agpt_2b_50k_moe_sdpa(
    expert_backend: str = "for_loop",
) -> moeModel.Config:
    """AGPT 2B/50K backbone with compute-matched MoE FFNs.

    Attention, depth, width, RoPE, vocabulary, normalization, and parameter
    initialization match ``agpt_2b_50k``. Every 10,496-wide dense FFN is
    replaced by 36 routed experts (top-3) plus two shared-expert equivalents.
    A 2,112-wide expert gives 10,560 active hidden units per token.
    """
    dim = 2048
    n_layers = 24
    n_heads = 16
    n_kv_heads = 4
    vocab_size = 50304
    expert_hidden_dim = 2112
    num_experts = 36
    top_k = 3
    num_shared_experts = 2
    layers = []

    for layer_id in range(n_layers):
        linear_init = _agpt_linear_init(dim)
        depth_init = _agpt_depth_init(dim, layer_id)
        expert_init = {
            "w1": linear_init["weight"],
            "w2": depth_init["weight"],
            "w3": depth_init["weight"],
        }
        layers.append(
            moeTransformerBlock.Config(
                attention_norm=RMSNorm.Config(
                    normalized_shape=dim, param_init=_NORM_INIT
                ),
                ffn_norm=RMSNorm.Config(
                    normalized_shape=dim, param_init=_NORM_INIT
                ),
                attention=make_gqa_config(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    wqkv_param_init=linear_init,
                    wo_param_init=depth_init,
                    inner_attention=_default_inner_attention(),
                    mask_type="causal",
                ),
                feed_forward=None,
                moe=make_moe_config(
                    num_experts=num_experts,
                    load_balance_coeff=1e-3,
                    router=make_router_config(
                        dim=dim,
                        num_experts=num_experts,
                        gate_param_init=depth_init,
                        top_k=top_k,
                        score_func="softmax",
                        route_norm=False,
                    ),
                    experts=make_experts_config(
                        dim=dim,
                        hidden_dim=expert_hidden_dim,
                        num_experts=num_experts,
                        top_k=top_k,
                        score_before_experts=False,
                        use_grouped_mm=False,
                        compute_backend=expert_backend,
                        comm_backend="standard",
                        param_init=expert_init,
                    ),
                    shared_experts=make_ffn_config(
                        dim=dim,
                        hidden_dim=expert_hidden_dim * num_shared_experts,
                        w1_param_init=linear_init,
                        w2w3_param_init=depth_init,
                    ),
                ),
            )
        )

    return moeModel.Config(
        dim=dim,
        vocab_size=vocab_size,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=131072,
            theta=50000,
            backend="complex",
            scaling="none",
        ),
        layers=layers,
    )


def _agpt_2b_50k_moe_sdpa_aurora_full_loop() -> moeModel.Config:
    return _agpt_2b_50k_moe_sdpa("aurora_full_loop")


def _agpt_2b_50k_moe_sdpa_aurora_full_sonic() -> moeModel.Config:
    return _agpt_2b_50k_moe_sdpa("aurora_full_sonic")


def _10b_2b_sdpa_1layer() -> moeModel.Config:
    """Single-block profiling variant of 10B_2B_sdpa.

    Keeps the same hidden size, head geometry, expert count, router setup,
    and MoE dimensions as 10B_2B_sdpa, but reduces the stack to one
    transformer block containing attention + MoE.
    """
    dim = 2048
    n_layers = 1
    vocab_size = 256128
    n_heads = 16
    moe_hidden_dim = 1408
    num_shared_experts = 2
    dense_hidden_dim = 10944
    rope_dim = 64
    num_experts = 36
    n_dense_layers = 0

    layers = _build_moe_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=0.70,
        dense_hidden_dim=dense_hidden_dim,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=3,
        router_score_func="softmax",
        score_before_experts=False,
        attn_backend="sdpa",
    )
    return moeModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=rope_dim,
            max_seq_len=4096 * 4,
            theta=10000.0,
            backend="complex",
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
        layers=layers,
    )


def _10b_2b_sdpa_batched_mm_padded() -> moeModel.Config:
    cfg = _10b_2b_sdpa()
    for layer_cfg in cfg.layers:
        if layer_cfg.moe is None:
            continue
        layer_cfg.moe.experts.compute_backend = "batched_mm_padded"
        layer_cfg.moe.experts.use_grouped_mm = False
    return cfg


def _10b_2b_sdpa_scattermoe() -> moeModel.Config:
    cfg = _10b_2b_sdpa()
    for layer_cfg in cfg.layers:
        if layer_cfg.moe is None:
            continue
        layer_cfg.moe.experts.compute_backend = "scattermoe"
        layer_cfg.moe.experts.use_grouped_mm = False
    return cfg


moe_configs = {
    "debugmodel": _debugmodel,
    "debugmodel_flex_attn": _debugmodel_flex_attn,
    "500M": _500m,
    "2B": _2b,
    "4B": _4b,
    "7B": _7b,
    "small": _small,
    "16B": _16b,
    "236B": _236b,
    "671B": _671b,
    "10B_2B": _10b_2b,
    "10B_2B_sdpa": _10b_2b_sdpa,
    "10B_2B_50K_sdpa": _10b_2b_50k_sdpa,
    "10B_2B_50K_sdpa_for_loop": _10b_2b_50k_sdpa_for_loop,
    "10B_2B_50K_sdpa_aurora_sycl": _10b_2b_50k_sdpa_aurora_sycl,
    "10B_2B_50K_sdpa_aurora_full_loop": _10b_2b_50k_sdpa_aurora_full_loop,
    "10B_2B_50K_sdpa_aurora_full_sonic": _10b_2b_50k_sdpa_aurora_full_sonic,
    "10B_2B_50K_sdpa_aurora_full_loop_1layer": _10b_2b_50k_sdpa_aurora_full_loop_1layer,
    "10B_2B_50K_sdpa_aurora_full_sonic_1layer": _10b_2b_50k_sdpa_aurora_full_sonic_1layer,
    "AGPT_2B_50K_MOE_sdpa": _agpt_2b_50k_moe_sdpa,
    "AGPT_2B_50K_MOE_sdpa_aurora_full_loop": _agpt_2b_50k_moe_sdpa_aurora_full_loop,
    "AGPT_2B_50K_MOE_sdpa_aurora_full_sonic": _agpt_2b_50k_moe_sdpa_aurora_full_sonic,
    "10B_2B_sdpa_1layer": _10b_2b_sdpa_1layer,
    "10B_2B_sdpa_batched_mm_padded": _10b_2b_sdpa_batched_mm_padded,
    "10B_2B_sdpa_scattermoe": _10b_2b_sdpa_scattermoe,
}

moe_configs["debugmodel_hf"] = moe_configs["debugmodel"]
moe_configs["debugmodel_flex_attn_hf"] = moe_configs["debugmodel_flex_attn"]


def model_registry(
    flavor: str,
    moe_comm_backend: str = "standard",
    quantization: list | None = None,
) -> ModelSpec:
    from torchtitan.components.quantization import QuantizationConverter
    from torchtitan.distributed.pipeline_parallel import pipeline_llm
    from torchtitan.models.common.config_utils import make_token_dispatcher_config

    config = moe_configs[flavor]()

    # Rebuild token dispatchers per #3125 — comm_backend is now always set
    # (default "standard"), and AllToAllTokenDispatcher falls back to local
    # dispatch when EP=1.
    for layer_cfg in config.layers:
        if layer_cfg.moe is not None:
            experts_cfg = layer_cfg.moe.experts
            experts_cfg.token_dispatcher = make_token_dispatcher_config(
                num_experts=experts_cfg.num_experts,
                top_k=experts_cfg.token_dispatcher.top_k,
                score_before_experts=experts_cfg.token_dispatcher.score_before_experts,
                comm_backend=moe_comm_backend,
            )

    # Quantization is now applied to the config at model_registry time
    # rather than to the runtime model (#3127).
    if quantization is not None:
        for q in quantization:
            assert isinstance(q, QuantizationConverter.Config)
            q.build().convert(config)

    return ModelSpec(
        name="moe",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_moe,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        # The DeepSeek adapter assumes MLA projections. Native DCP checkpoints
        # do not require an adapter; HF conversion for this GQA-MoE flavor is
        # intentionally disabled until it has an explicit mapping.
        state_dict_adapter=(
            None
            if isinstance(config.layers[0].attention, GQAttention.Config)
            else moeStateDictAdapter
        ),
    )
