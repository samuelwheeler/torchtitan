# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config-based DTensor sharding for the ezpz/moe (DeepSeek-V3-derived) model.

Mirrors `torchtitan.models.deepseek_v3.sharding` but binds against
`torchtitan.experiments.ezpz.moe.model.Attention` (our MLA Attention is a
separate class from upstream's, even though the structure is identical).

Note: the MoE block itself is NOT touched here. MoE TP/EP wiring is still
done at parallelize-time by `apply_moe_ep_tp` — that mirrors upstream's
deepseek_v3 pattern, which also leaves the MoE block out of
`set_deepseek_v3_sharding_config`.
"""

from typing import TYPE_CHECKING

from torch.distributed.tensor import Placement, Replicate, Shard

from torchtitan.experiments.ezpz.moe.model import Attention
from torchtitan.models.common.attention import GQAttention
from torchtitan.models.common.decoder_sharding import (
    colwise_config,
    dense_activation_placement,
    dense_param_placement,
    norm_config,
    rowwise_config,
    set_decoder_sharding_config,
    set_dense_ffn_sharding,
    set_gqa_attention_sharding,
    set_gqa_inner_attention_local_map,
)
from torchtitan.protocols.sharding import ShardingConfig

if TYPE_CHECKING:
    from torchtitan.experiments.ezpz.moe.model import (
        moeModel,
        moeTransformerBlock,
    )


def set_moe_sharding_config(
    config: "moeModel.Config",
    *,
    loss_parallel: bool,
    enable_sp: bool,
) -> None:
    """Fill ``sharding_config`` on all moe (non-MoE-block) sub-configs.

    No-op on MoE blocks — those are handled at parallelize-time by
    ``apply_moe_ep_tp``.
    """
    set_decoder_sharding_config(
        config, loss_parallel=loss_parallel, enable_sp=enable_sp
    )
    for layer_cfg in config.layers:
        _set_moe_layer_sharding(layer_cfg, enable_sp=enable_sp)


def _set_moe_layer_sharding(
    layer_cfg: "moeTransformerBlock.Config", *, enable_sp: bool
) -> None:
    """Set sharding on one moe transformer layer.

    MLA attention: low-rank projections (wkv_a, wq_a, kv_norm, q_norm)
    stay replicated. Up-projections (wkv_b, wq_b, wq) are colwise.

    On non-MoE layers (the dense FFN at the bottom of the stack), also
    sets dense FFN sharding. MoE-layer feed_forward sub-module is None;
    its `moe` block is left for ``apply_moe_ep_tp``.
    """
    norm = norm_config(enable_sp=enable_sp)
    layer_cfg.attention_norm.sharding_config = norm
    layer_cfg.ffn_norm.sharding_config = norm
    attn_x_placement: Placement = Shard(1) if enable_sp else Replicate()

    attention = layer_cfg.attention
    if isinstance(attention, GQAttention.Config):
        set_gqa_attention_sharding(attention, enable_sp=enable_sp)
        set_gqa_inner_attention_local_map(attention.inner_attention)
        if layer_cfg.feed_forward is not None:
            set_dense_ffn_sharding(
                layer_cfg.feed_forward,
                attn_x_placement=attn_x_placement,
                enable_sp=enable_sp,
            )
        return
    if not isinstance(attention, Attention.Config):
        raise TypeError(
            "MoE sharding supports MLA or GQA attention, got "
            f"{type(attention).__name__}."
        )

    # MLA attention input: x is gathered to Replicate; freqs_cis always Replicate.
    attention.sharding_config = ShardingConfig(
        in_src_shardings={
            "x": dense_activation_placement(tp=attn_x_placement),
            "freqs_cis": dense_param_placement(tp=Replicate()),
        },
        in_dst_shardings={
            "x": dense_activation_placement(tp=Replicate()),
            "freqs_cis": dense_param_placement(tp=Replicate()),
        },
    )
    # Low-rank projections and norms keep Replicate weights on TP. We still
    # distribute them (Replicate DTensor) so DTensor activations flow through
    # without mixing plain Tensor + DTensor in the matmul.
    replicate_weight = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=Replicate())},
    )
    attention.wkv_a.sharding_config = replicate_weight
    attention.kv_norm.sharding_config = replicate_weight

    attention.wkv_b.sharding_config = colwise_config()
    attention.wo.sharding_config = rowwise_config(output_sp=enable_sp)

    # Static LocalMapConfig on the inner-attention config (upstream #2986
    # replaced runtime DTensor detection in `LocalMapInnerAttention` with
    # this config-driven approach).
    set_gqa_inner_attention_local_map(attention.inner_attention)

    # Query projection: depends on q_lora_rank
    if attention.q_lora_rank == 0:
        assert attention.wq is not None
        attention.wq.sharding_config = colwise_config()
    else:
        # Low-rank: wq_a + q_norm stay Replicate DTensors; wq_b is Colwise.
        assert attention.wq_a is not None
        assert attention.wq_b is not None
        attention.wq_a.sharding_config = replicate_weight
        attention.q_norm.sharding_config = replicate_weight
        attention.wq_b.sharding_config = colwise_config()

    # Dense FFN (non-MoE layers only). MoE blocks are handled at
    # parallelize-time by apply_moe_ep_tp.
    if layer_cfg.feed_forward is not None:
        set_dense_ffn_sharding(
            layer_cfg.feed_forward,
            attn_x_placement=attn_x_placement,
            enable_sp=enable_sp,
        )
