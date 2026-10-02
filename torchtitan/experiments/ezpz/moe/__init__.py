# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from collections.abc import Callable
from functools import partial
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from torchtitan.models.common import (
    ComplexRoPE,
    Embedding,
    Linear,
    RMSNorm,
    RoPE,
    TransformerBlock,
)

# MoE and TokenChoiceTopKRouter come straight from upstream — we have no
# ezpz-specific override for them. Earlier this re-imported from a local
# `.moe` copy that was a byte-for-byte fork of `torchtitan/models/common/moe.py`;
# that fork has been deleted to avoid silent skew on upstream MoE/router
# fixes (e.g. the CP-friendly 3-D experts output added in upstream PR #3447).
from torchtitan.models.common.activation import Sigmoid, Softmax
from torchtitan.models.common.attention import ScaledDotProductInnerAttention
from torchtitan.models.common.config_utils import (
    fused_grouped_gate_up_param_init,
    get_attention_config,
    make_ffn_config,
    make_gqa_config,
)
from torchtitan.models.common.linear import (
    ColumnParallelLinear,
    GroupedLinear,
    RouterGateLinear,
)
from torchtitan.models.common.moe import MoE, RoutedExperts, TokenChoiceTopKRouter

from torchtitan.models.common.param_init import depth_scaled_std
from torchtitan.models.deepseek_v3 import DeepSeekV3Router

from torchtitan.protocols.module import Module

from .experts import ExpertComputeBackend
from .model import Attention, moeModel, moeTransformerBlock

from .routed_experts import EzpzRoutedExperts


class _LegacyInterleavedColumnParallelLinear(ColumnParallelLinear):
    """Stacked-projection API backed by historical interleaved 2-D storage."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(ColumnParallelLinear.Config):
        pass

    def __init__(self, config: Config):
        if config.num_linears != 2:
            raise ValueError("legacy fused FFN projection requires num_linears=2")
        # Bypass Linear.__init__'s [N, F, D] reshape. Keeping [2F, D] means
        # FSDP shards the full fused output axis, as it did pre-sync.
        nn.Linear.__init__(
            self,
            config.in_features,
            config.num_linears * config.out_features,
            bias=config.bias,
        )
        self.out_features = config.out_features
        self.num_linears = config.num_linears

    def _unflatten_output(self, output):
        return output.unflatten(-1, (-1, 2)).transpose(-2, -1)


def _dtensor_safe_fused_ffn_config(**kwargs):
    """Build a fused FFN with the pre-sync 2-D storage and shard-local init."""
    cfg = make_ffn_config(**kwargs)
    stacked_cfg = cfg.w13
    cfg.w13 = _LegacyInterleavedColumnParallelLinear.Config(
        in_features=stacked_cfg.in_features,
        out_features=stacked_cfg.out_features,
        num_linears=stacked_cfg.num_linears,
        bias=stacked_cfg.bias,
        param_init=stacked_cfg.param_init,
        sharding_config=stacked_cfg.sharding_config,
    )
    core_init = cfg.w13.param_init["weight"] if cfg.w13.param_init else None
    gate_init = kwargs["w1_param_init"].get("weight")
    up_init = kwargs["w2w3_param_init"].get("weight")

    def _init_striped(t):
        if not hasattr(t, "to_local"):
            if core_init is not None:
                gate_up = t.unflatten(0, (-1, 2))
                if gate_init is not None:
                    gate_init(gate_up[:, 0])
                if up_init is not None:
                    up_init(gate_up[:, 1])
            return
        local = t.to_local()
        if local.numel() == 0:
            return
        offset = 0
        mesh = getattr(t, "device_mesh", None)
        for axis, placement in enumerate(getattr(t, "placements", ())):
            if getattr(placement, "dim", None) == 0 and mesh is not None:
                rank = mesh.get_local_rank(axis)
                chunk, remainder = divmod(t.shape[0], mesh.size(axis))
                offset += rank * chunk + min(rank, remainder)
        gate_start = 0 if offset % 2 == 0 else 1
        gate_rows = local[gate_start::2]
        up_rows = local[1 - gate_start :: 2]
        if gate_init is not None and gate_rows.numel():
            gate_init(gate_rows)
        if up_init is not None and up_rows.numel():
            up_init(up_rows)

    assert cfg.w13.param_init is not None
    cfg.w13.param_init = {**cfg.w13.param_init, "weight": _init_striped}
    return cfg


def _model_max_context_length(layers: list[TransformerBlock.Config]) -> int:
    rope = getattr(layers[0].attention, "rope", None)
    if rope is None:
        raise ValueError("MoE attention config must define RoPE")
    return rope.max_context_length


from .token_dispatcher import (
    AllToAllTokenDispatcher,
    DeepEPTokenDispatcher,
    HybridEPTokenDispatcher,
)


class EzpzScaledDotProductAttention(ScaledDotProductInnerAttention):
    """SDPA variant that avoids set_priority=True in sdpa_kernel."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(ScaledDotProductInnerAttention.Config):
        pass

    # pyrefly: ignore [bad-override]
    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        *,
        scale: float | None = None,
        enable_gqa: bool = False,
        is_causal: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        # Positional arg names MUST be the shape-suffixed q_THK/k_THK/v_THV to
        # match set_gqa_inner_attention_local_spmd's in_dst_shardings keys --
        # the local_map contract check matches by positional-arg NAME and
        # asserts under TP>1 otherwise. Renamed from _BLNH 2026-08-25: #4121
        # (73aed7f6c) renamed the upstream keys and our port left the
        # parameters behind (MEASURED on the agpt twin, smoke 8781623 arm 2).
        # Renamed AGAIN in sync 84: #4533 rekeyed in_dst_shardings a second
        # time, _TNH -> q_THK/k_THK/v_THV, and renamed the contract fn to
        # ..._local_spmd. See the agpt twin for the full history.
        #
        # This wrapper is a straight port of the agpt one -- moe reaches it
        # through the SAME BlendCorpusDataLoader (moe/config_registry.py:115),
        # and 42f4edfaa folds [B, L] -> [T] unconditionally at the yield, so
        # these tensors arrive 3D exactly as agpt's do. The 4D assert that used
        # to stand here said "if the fold-batch-dim token layout landed, this
        # wrapper needs updating"; it landed.
        #
        # Why unflatten rather than transpose in place: on 3D input a bare
        # transpose(1, 2) swaps N with H instead of L with N, and SDPA ACCEPTS
        # the result -- a quietly degraded loss curve, not a traceback.
        #
        # B is recoverable because the loader emits fixed-length rows: T is a
        # whole number of max_context_length sequences. The divisibility check
        # keeps that verified rather than assumed; a ragged batch must not
        # silently reshape into the wrong grid.
        folded = q_THK.ndim == 3
        if folded:
            # Read agpt's module global rather than defining a second one.
            # trainer.py:443 only ever calls agpt.set_ezpz_max_context_length,
            # so a moe-local copy would stay None forever and every moe step
            # would raise the "must call set_ezpz_max_context_length" branch
            # below. Imported inside the function because agpt imports heavy
            # model deps at module scope.
            from torchtitan.experiments.ezpz.agpt import _EZPZ_MAX_CONTEXT_LENGTH

            seq_len = _EZPZ_MAX_CONTEXT_LENGTH

            if seq_len is None:
                raise ValueError(
                    "3D [T, N, H] attention input but max_context_length is "
                    "unknown; the trainer must call "
                    "set_ezpz_max_context_length() before the first forward"
                )
            num_tokens, num_heads, head_dim = q_THK.shape
            if num_tokens % seq_len != 0:
                raise ValueError(
                    f"token count {num_tokens} is not a multiple of "
                    f"max_context_length {seq_len}; this wrapper assumes the "
                    "fixed-length rows the loader emits and cannot reshape a "
                    "ragged batch"
                )
            batch = num_tokens // seq_len
            q_THK = q_THK.view(batch, seq_len, num_heads, head_dim)
            k_THK = k_THK.view(batch, seq_len, -1, head_dim)
            v_THV = v_THV.view(batch, seq_len, -1, head_dim)
        assert q_THK.ndim == 4, f"expected 4D, got {tuple(q_THK.shape)}"
        q, k, v = (
            q_THK.transpose(1, 2),
            k_THK.transpose(1, 2),
            v_THV.transpose(1, 2),
        )
        with sdpa_kernel(self.sdpa_backends):
            out = F.scaled_dot_product_attention(
                q, k, v, scale=scale, is_causal=is_causal, enable_gqa=enable_gqa
            )
        out = out.transpose(1, 2)
        if folded:
            # Restore the caller's flat [T, N, H]. The input unflatten was
            # ported in e4517ae09 but this half was not, so the wrapper took
            # 3D and returned 4D -- violating its own contract. MLA then does
            # output.view(num_tokens, -1) against a 4D tensor whose leading dim
            # is BATCH, not tokens, and the result lands Shard(dim=0) where the
            # rowwise wo declares Partial(sum). agpt/__init__.py does exactly
            # this and is why agpt trains at TP=2.
            out = out.reshape(-1, out.shape[-2], out.shape[-1])
        return out


class XPUScaledDotProductAttention(EzpzScaledDotProductAttention):
    """SDPA with OVERRIDEABLE first for XPU fused attention."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(EzpzScaledDotProductAttention.Config):
        pass

    sdpa_backends = [
        SDPBackend.OVERRIDEABLE,
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.MATH,
    ]


def _default_inner_attention() -> ScaledDotProductInnerAttention.Config:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return XPUScaledDotProductAttention.Config()
    return EzpzScaledDotProductAttention.Config()


def _ezpz_get_attention_config(backend: str) -> Module.Config:
    """XPU-aware attention config selection.

    Mirrors the agpt sibling. Upstream PR #3571 (replayed at ezpz
    `db3b916a8`) dropped the `(config, mask_type)` tuple return in
    favor of returning just the config; the caller now supplies
    mask_type separately (see `_build_moe_layers` which sets
    `_mask = "causal"` next to the call site). Returning a tuple
    here would break the downstream `inner_attention.sharding_config`
    setattr in `moe/sharding.py:set_gqa_inner_attention_local_spmd`.
    """
    if backend == "sdpa":
        return _default_inner_attention()
    return get_attention_config(backend)


def make_ezpz_router_config(
    *,
    dim: int,
    num_experts: int,
    gate_param_init: dict[str, Callable],
    top_k: int = 1,
    score_func: Literal["sigmoid", "softmax"] = "sigmoid",
    route_norm: bool = False,
    route_scale: float = 1.0,
    num_expert_groups: int | None = None,
    num_limited_groups: int | None = None,
    bias: bool = False,
) -> TokenChoiceTopKRouter.Config | DeepSeekV3Router.Config:
    # Sync 84 reshaped this config three ways at once:
    #   #4631 moved num_expert_groups/num_limited_groups OFF the stock router
    #         and onto DeepSeekV3Router, which is now the only one that does
    #         node-limited routing.
    #   score_func became a UnaryActivationFn.Config instead of a string.
    #   gate became a RouterGateLinear.Config instead of a plain Linear.Config.
    # Route to whichever router actually supports what the flavor asks for,
    # rather than passing group kwargs the stock router no longer accepts.
    score_cfg = Sigmoid.Config() if score_func == "sigmoid" else Softmax.Config()
    gate_cfg = RouterGateLinear.Config(
        in_features=dim,
        out_features=num_experts,
        bias=bias,
        param_init=gate_param_init,
    )

    if num_expert_groups is not None or num_limited_groups is not None:
        return DeepSeekV3Router.Config(
            num_experts=num_experts,
            gate=gate_cfg,
            top_k=top_k,
            score_func=score_cfg,
            route_norm=route_norm,
            route_scale=route_scale,
            num_expert_groups=num_expert_groups,
            num_limited_groups=num_limited_groups,
        )

    return TokenChoiceTopKRouter.Config(
        num_experts=num_experts,
        gate=gate_cfg,
        top_k=top_k,
        score_func=score_cfg,
        route_norm=route_norm,
        route_scale=route_scale,
    )


def make_ezpz_token_dispatcher_config(
    *,
    num_experts: int,
    top_k: int,
    score_before_experts: bool = True,
    comm_backend: str,
    non_blocking_capacity_factor: float | None = None,
):
    if comm_backend == "deepep":
        return DeepEPTokenDispatcher.Config(
            num_experts=num_experts,
            top_k=top_k,
            score_before_experts=score_before_experts,
        )
    if comm_backend == "hybridep":
        return HybridEPTokenDispatcher.Config(
            num_experts=num_experts,
            top_k=top_k,
            score_before_experts=score_before_experts,
            non_blocking_capacity_factor=non_blocking_capacity_factor,
        )
    if comm_backend == "standard":
        return AllToAllTokenDispatcher.Config(
            num_experts=num_experts,
            top_k=top_k,
            score_before_experts=score_before_experts,
        )
    raise ValueError(
        f"Unknown comm_backend: {comm_backend!r}. "
        "Must be one of 'standard', 'deepep', 'hybridep'."
    )


def make_ezpz_experts_config(
    *,
    dim: int,
    hidden_dim: int,
    num_experts: int,
    top_k: int,
    param_init: dict[str, Callable],
    score_before_experts: bool = True,
    comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
    compute_backend: ExpertComputeBackend = "grouped_mm",
) -> RoutedExperts.Config:
    """Build upstream-layout routed experts plus an ezpz backend selector."""

    dispatcher = make_ezpz_token_dispatcher_config(
        num_experts=num_experts,
        top_k=top_k,
        score_before_experts=score_before_experts,
        comm_backend=comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
    )
    # EzpzRoutedExperts, not RoutedExperts: identical behaviour for the five
    # backends that take (x, num_tokens_per_expert), but it forwards the
    # router's decision to inner_experts for aurora_full_sonic, which core
    # drops at models/common/moe.py:163.
    return EzpzRoutedExperts.Config(
        w13=GroupedLinear.Config(
            group_size=num_experts,
            in_features=dim,
            out_features=hidden_dim,
            num_linears=2,
            param_init=fused_grouped_gate_up_param_init(param_init),
        ),
        w2=GroupedLinear.Config(
            group_size=num_experts,
            in_features=hidden_dim,
            out_features=dim,
            param_init={"weight": param_init["w2_EDF"]},
        ),
        token_dispatcher=dispatcher,
        compute_backend=compute_backend,
    )


def make_ezpz_moe_config(
    *,
    num_experts: int = 8,
    router: TokenChoiceTopKRouter.Config,
    routed_experts: RoutedExperts.Config,
    shared_experts=None,
    load_balance_coeff: float | None = 1e-3,
) -> MoE.Config:
    return MoE.Config(
        num_experts=num_experts,
        load_balance_coeff=load_balance_coeff,
        router=router,
        routed_experts=routed_experts,
        shared_experts=shared_experts,
    )


__all__ = [
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
    # Param names use Shazeer shape-suffix style post upstream PR #3425
    # (41st sync): w1_EFD, w2_EDF, w3_EFD.
    return {
        "w1_EFD": partial(nn.init.trunc_normal_, std=0.02),
        "w2_EDF": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "w3_EFD": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
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
    rope: RoPE.Config,
    mscale: float = 1.0,
    attn_backend: str = "sdpa",
) -> Attention.Config:
    """Build a fully-specified MoE MLA Attention.Config.

    All Linear and RMSNorm sub-configs have their dimensional fields set.
    When q_lora_rank == 0, sets wq (not wq_a/wq_b).
    When q_lora_rank > 0, sets wq_a/wq_b (not wq).
    """
    # Upstream PR #3571 dropped the (config, mask_type) tuple — see
    # _ezpz_get_attention_config docstring. ezpz/moe's Attention.Config
    # still has its own mask_type field (see moe/model.py), so we
    # preserve the previous default of "causal" here.
    _inner = _ezpz_get_attention_config(attn_backend)
    _mask = "causal"
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
        rope=dataclasses.replace(rope),
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
    compute_backend: ExpertComputeBackend = "grouped_mm",
    rope: RoPE.Config,
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
            rope=rope,
        )

        if layer_id < n_dense_layers:
            ffn_cfg = _dtensor_safe_fused_ffn_config(
                dim=dim,
                hidden_dim=dense_hidden_dim,
                w1_param_init=_LINEAR_INIT,
                w2w3_param_init=_depth_init(layer_id),
            )
            moe_cfg = None
        else:
            ffn_cfg = None
            moe_cfg = make_ezpz_moe_config(
                num_experts=num_experts,
                router=make_ezpz_router_config(
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
                routed_experts=make_ezpz_experts_config(
                    dim=dim,
                    hidden_dim=moe_hidden_dim,
                    num_experts=num_experts,
                    top_k=router_top_k,
                    score_before_experts=score_before_experts,
                    comm_backend=moe_comm_backend,
                    param_init=_depth_experts_init(layer_id),
                    compute_backend=compute_backend,
                ),
                shared_experts=_dtensor_safe_fused_ffn_config(
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
        rope=ComplexRoPE.Config(
            dim=rope_dim,
            max_context_length=4096 * 4,
            theta=10000.0,
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
    )
    return moeModel.Config(
        max_context_length=_model_max_context_length(layers),
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
        rope=ComplexRoPE.Config(
            dim=rope_dim,
            max_context_length=4096 * 4,
            theta=10000.0,
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
    )
    return moeModel.Config(
        max_context_length=_model_max_context_length(layers),
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
        rope=ComplexRoPE.Config(
            dim=rope_dim,
            max_context_length=256128,
            theta=50000.0,
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
    )
    return moeModel.Config(
        max_context_length=_model_max_context_length(layers),
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
        rope=ComplexRoPE.Config(
            dim=rope_dim,
            max_context_length=4096 * 4,
            theta=10000.0,
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
    )
    return moeModel.Config(
        max_context_length=_model_max_context_length(layers),
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
        rope=ComplexRoPE.Config(
            dim=rope_dim,
            max_context_length=4096 * 4,
            theta=10000.0,
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
    )
    return moeModel.Config(
        max_context_length=_model_max_context_length(layers),
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
        rope=ComplexRoPE.Config(
            dim=rope_dim,
            max_context_length=4096 * 4,
            theta=10000.0,
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
    )
    return moeModel.Config(
        max_context_length=_model_max_context_length(layers),
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
        rope=ComplexRoPE.Config(
            dim=rope_dim,
            max_context_length=4096 * 4,
            theta=10000.0,
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
    )
    return moeModel.Config(
        max_context_length=_model_max_context_length(layers),
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
        rope=ComplexRoPE.Config(
            dim=rope_dim,
            max_context_length=4096 * 4,
            theta=10000.0,
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
    )
    return moeModel.Config(
        max_context_length=_model_max_context_length(layers),
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
        rope=ComplexRoPE.Config(
            dim=rope_dim,
            max_context_length=4096 * 4,
            theta=10000.0,
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
    )
    return moeModel.Config(
        max_context_length=_model_max_context_length(layers),
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
        rope=ComplexRoPE.Config(
            dim=rope_dim,
            max_context_length=4096 * 4,
            theta=10000.0,
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
    )
    return moeModel.Config(
        max_context_length=_model_max_context_length(layers),
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
        rope=ComplexRoPE.Config(
            dim=rope_dim,
            max_context_length=4096 * 4,
            theta=10000.0,
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
    )
    return moeModel.Config(
        max_context_length=_model_max_context_length(layers),
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


def _agpt_2b_50k_moe_sdpa_aurora_full_sonic() -> moeModel.Config:
    """AGPT 2B/50K geometry with compute-matched full-Sonic MoE FFNs."""
    from torchtitan.experiments.ezpz.agpt import (
        _depth_init as agpt_depth_init,
        _linear_init as agpt_linear_init,
    )

    dim = 2048
    n_heads = 16
    n_kv_heads = 4
    num_experts = 36
    top_k = 3
    expert_hidden_dim = 2112
    rope = ComplexRoPE.Config(
        dim=dim // n_heads,
        max_context_length=131072,
        theta=50000,
        scaling="none",
    )
    layers = []
    for layer_id in range(24):
        linear_init = agpt_linear_init(dim)
        depth_init = agpt_depth_init(dim, layer_id)
        layers.append(
            moeTransformerBlock.Config(
                attention_norm=RMSNorm.Config(
                    normalized_shape=dim, param_init=_NORM_INIT
                ),
                ffn_norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
                attention=make_gqa_config(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    wqkv_param_init=linear_init,
                    wo_param_init=depth_init,
                    inner_attention=_default_inner_attention(),
                    rope=rope,
                ),
                feed_forward=None,
                moe=make_ezpz_moe_config(
                    num_experts=num_experts,
                    load_balance_coeff=1e-3,
                    router=make_ezpz_router_config(
                        dim=dim,
                        num_experts=num_experts,
                        gate_param_init=depth_init,
                        top_k=top_k,
                        score_func="softmax",
                        route_norm=False,
                    ),
                    routed_experts=make_ezpz_experts_config(
                        dim=dim,
                        hidden_dim=expert_hidden_dim,
                        num_experts=num_experts,
                        top_k=top_k,
                        score_before_experts=False,
                        compute_backend="aurora_full_sonic",
                        param_init={
                            "w1_EFD": linear_init["weight"],
                            "w2_EDF": depth_init["weight"],
                            "w3_EFD": depth_init["weight"],
                        },
                    ),
                    shared_experts=_dtensor_safe_fused_ffn_config(
                        dim=dim,
                        hidden_dim=expert_hidden_dim * 2,
                        w1_param_init=linear_init,
                        w2w3_param_init=depth_init,
                    ),
                ),
            )
        )
    return moeModel.Config(
        max_context_length=rope.max_context_length,
        vocab_size=50304,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=50304, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=50304,
            param_init=_output_linear_init(dim),
        ),
        layers=layers,
    )


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
    "AGPT_2B_50K_MOE_sdpa_aurora_full_sonic": _agpt_2b_50k_moe_sdpa_aurora_full_sonic,
}

moe_configs["debugmodel_hf"] = moe_configs["debugmodel"]
moe_configs["debugmodel_flex_attn_hf"] = moe_configs["debugmodel_flex_attn"]


def model_registry(
    flavor: str,
    moe_comm_backend: str = "standard",
    quantization: list | None = None,
) -> moeModel.Config:
    from torchtitan.config.transform.quantization import QuantizationConverter

    config = moe_configs[flavor]()

    # Rebuild token dispatchers per #3125 — comm_backend is now always set
    # (default "standard"), and AllToAllTokenDispatcher falls back to local
    # dispatch when EP=1.
    for layer_cfg in config.layers:
        if layer_cfg.moe is not None:
            routed_cfg = layer_cfg.moe.routed_experts
            routed_cfg.token_dispatcher = make_ezpz_token_dispatcher_config(
                num_experts=routed_cfg.w13.group_size,
                top_k=routed_cfg.token_dispatcher.top_k,
                score_before_experts=routed_cfg.token_dispatcher.score_before_experts,
                comm_backend=moe_comm_backend,
            )

    # Quantization is now applied to the config at model_registry time
    # rather than to the runtime model (#3127).
    if quantization is not None:
        for q in quantization:
            assert isinstance(q, QuantizationConverter.Config)
            q.build().convert(config)

    # Read context length from the flavor's own RoPE config (same approach as
    # the AGPT twin) so the registry cannot drift from the model it describes.
    if config.layers[0].attention.rope is None:
        raise ValueError(
            f"moe flavor {flavor!r} has no RoPE config, so max_context_length "
            "cannot be derived"
        )

    # The AGPT GQA/Sonic flavor exports through its own HF model because the
    # generic DeepSeek adapter assumes MLA and different router semantics.
    if flavor == "AGPT_2B_50K_MOE_sdpa_aurora_full_sonic":
        from dataclasses import fields

        from .agpt_hf_state_dict_adapter import AGPTMoEStateDictAdapter

        class _SonicMoeModel(moeModel):
            state_dict_adapter_cls = AGPTMoEStateDictAdapter

            @dataclasses.dataclass(kw_only=True, slots=True)
            class Config(moeModel.Config):
                pass

        return _SonicMoeModel.Config(
            **{field.name: getattr(config, field.name) for field in fields(config)}
        )
    return config
