# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass, replace as dataclass_replace
from functools import partial
from typing import Any, Literal

_EZPZ_MAX_CONTEXT_LENGTH: int | None = None


def set_ezpz_max_context_length(seq_len: int) -> None:
    """Tell the SDPA wrapper how to unflatten #4121 flat [T, N, H] batches.

    Module-level rather than a Config field: the forward's positional-arg names
    are contract-checked under TP>1 (set_gqa_inner_attention_local_spmd matches
    in_dst_shardings by name), so the signature must not change.
    """
    global _EZPZ_MAX_CONTEXT_LENGTH
    _EZPZ_MAX_CONTEXT_LENGTH = int(seq_len)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from torchtitan.experiments.ezpz.agpt.local_rmsnorm import LocalShardRMSNorm

from torchtitan.experiments.ezpz.diagnostics import attention as _attn_diag
from torchtitan.models.common import (
    ComplexRoPE,
    compute_ffn_hidden_dim,
    CosSinRoPE,
    Embedding,
    Linear,
    RMSNorm,
    RoPE,
    TransformerBlock,
)

from torchtitan.models.common.attention import ScaledDotProductInnerAttention
from torchtitan.models.common.config_utils import get_attention_config
from torchtitan.protocols.module import Module


class EzpzScaledDotProductAttention(ScaledDotProductInnerAttention):
    """SDPA that avoids ``set_priority=True`` in the ``sdpa_kernel`` context.

    Works around a torch._dynamo bug in PyTorch 2.11 where
    ``sdpa_kernel(..., set_priority=True)`` passes FX proxy nodes to
    ``int()`` during fake tensor tracing, causing
    ``RuntimeError('Invalid backend')``.
    """

    @dataclass(kw_only=True, slots=True)
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
        # match the keys in set_gqa_inner_attention_local_spmd's
        # in_dst_shardings (models/common/decoder_sharding.py:288) -- the
        # local_map contract check matches by positional-arg NAME and asserts
        # under TP>1 if a mapped input is missing.
        #
        # These were q_BLNH until 2026-08-25. #4121 (73aed7f6c) renamed the
        # upstream keys _BLNH -> _TNH along with the layout change, and our
        # port (476d16831) adapted this function's BODY without renaming its
        # PARAMETERS -- so the contract silently broke for TP>1 while TP=1,
        # which never wraps local_map, kept passing. MEASURED on smoke 8781623
        # arm 2 (2N, TP=2):
        #   AssertionError: XPUScaledDotProductAttention: local_map is set but
        #   in_dst_shardings is missing entries for: ['q_BLNH','k_BLNH','v_BLNH']
        # Exactly the failure mode the 57th sync's q_BLNH rename was written
        # to prevent, reintroduced from the other direction.
        #
        # It happened AGAIN in sync 84. #4533 renamed the upstream class
        # (ScaledDotProductAttention -> ScaledDotProductInnerAttention) and
        # the contract fn (set_gqa_inner_attention_local_map ->
        # ..._local_spmd), and rekeyed in_dst_shardings a second time:
        # q_TNH/k_TNH/v_TNH -> q_THK/k_THK/v_THV (N heads -> H, H head_dim ->
        # K, and v's last axis is V not K). These parameters were renamed to
        # match at merge time. THIRD occurrence of the same trap: whenever an
        # upstream sync touches decoder_sharding.py, diff its in_dst_shardings
        # keys against this signature BEFORE running anything at TP>1.
        # #4121 (fold-batch-dim) reshaped the LM stack to a flat [T, N, H]
        # token layout, so these arrive 3D on the current tree. SDPA needs
        # [B, N, L, H], and a bare transpose(1, 2) on a 3D tensor swaps N with
        # H -- which SDPA ACCEPTS, producing a quietly degraded loss curve
        # rather than a traceback. Unflatten first, restore the layout after.
        #
        # B is recoverable because ConcatThenSplitPacking emits "fixed-length
        # rows": T is a whole number of max_context_length sequences (measured
        # T=20480, L=4096 -> B=5, exactly the configured LBS). The divisibility
        # check keeps that a verified property rather than an assumption -- a
        # ragged batch must not silently reshape into the wrong grid.
        #
        # Upstream's own SDPA does the same bare transpose, and its flat-layout
        # path (VarlenAttention) needs CUDA flash attention, which XPU lacks --
        # so neither upstream branch covers this and the adaptation lives here.
        folded = q_THK.ndim == 3
        if folded:
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
                    "fixed-length rows ConcatThenSplitPacking emits and cannot "
                    "reshape a ragged batch"
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
        # Avoid set_priority=True — triggers a torch._dynamo bug in
        # PyTorch 2.11 where FX proxy nodes are incorrectly passed to
        # int() during fake tensor tracing.
        # QK diagnostics. Module-level gate rather than a config on self: the
        # positional-arg names of this forward are contract-checked under TP>1
        # (see the arg-name note above), so the signature must not change. No-op
        # and near-free when disabled -- see diagnostics/attention.py.
        _attn_diag.observe(q, k, scale)
        with sdpa_kernel(self.sdpa_backends):
            out = F.scaled_dot_product_attention(
                q, k, v, scale=scale, is_causal=is_causal, enable_gqa=enable_gqa
            )
        out = out.transpose(1, 2)
        if folded:
            # Restore the caller's flat [T, N, H]: GQAttention immediately does
            # out.view(out.shape[0], -1), which reads the wrong stride off a 4D
            # tensor.
            out = out.reshape(-1, out.shape[-2], out.shape[-1])
        return out


class XPUScaledDotProductAttention(EzpzScaledDotProductAttention):
    """SDPA with OVERRIDEABLE backend for XPU-optimized fused attention.

    Adds OVERRIDEABLE to the backend priority list for the XPU fused
    attention kernel. On single-device the OVERRIDEABLE backend is 23x
    faster than MATH and avoids materializing the N×N attention matrix.

    Note: On XPU, the sdpa_kernel context manager and
    torch.backends.cuda.enable_math_sdp are not respected inside
    FSDP-wrapped modules (PyTorch XPU bug). The MATH backend is always
    used inside FSDP regardless of this setting. TP is required for
    models where the MATH attention matrix exceeds device memory
    (e.g., 80B with 72 heads at seq_len=8192 = 9 GiB per tile).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(EzpzScaledDotProductAttention.Config):
        pass

    sdpa_backends = [
        SDPBackend.OVERRIDEABLE,
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.MATH,
    ]


from torchtitan.models.common.activation import BinaryActivationFn
from torchtitan.models.common.config_utils import make_ffn_config, make_gqa_config


# ---------------------------------------------------------------------------
# Architecture tweaks for competition
# ---------------------------------------------------------------------------


class SoftcappedFlexAttention(Module):
    """FlexAttention with logit softcapping (Gemma 2 style).

    Uses FlexAttention's score_mod to apply tanh softcapping inside the
    fused kernel — no O(seq_len²) materialization. Requires torch.compile.

    TP: relies on `set_gqa_inner_attention_local_spmd` setting a static
    `LocalMapConfig` on the inner-attention sharding_config (upstream
    #2986 replaced runtime DTensor detection in `LocalMapInnerAttention`
    with config-driven local_map).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        logit_cap: float = 30.0

    def __init__(self, config: Config):
        super().__init__()
        self.logit_cap = config.logit_cap
        from torch.nn.attention.flex_attention import flex_attention

        self._flex_attention = torch.compile(flex_attention)

    # pyrefly: ignore [bad-override]
    def forward(
        self,
        q_BLNH: torch.Tensor,
        k_BLNH: torch.Tensor,
        v_BLNH: torch.Tensor,
        *,
        scale: float | None = None,
        enable_gqa: bool = False,
        is_causal: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        # 57th sync: shape-suffixed positional names required to match
        # set_gqa_inner_attention_local_spmd's in_dst_shardings under TP>1.
        # The _BLNH suffixes are a contract: 4D [B, L, N, H]. Upstream's
        # fold-batch-dim (#4121) reshapes the LM stack to a flat [T] token
        # layout, and if that ever reaches here the tensors arrive 3D --
        # transpose(1, 2) then swaps N with H instead of L with N, and SDPA
        # ACCEPTS the result. The failure is a quietly degraded loss curve,
        # not a traceback. Assert the rank so it is loud instead.
        assert q_BLNH.ndim == 4, (
            f"expected 4D [B, L, N, H], got {tuple(q_BLNH.shape)} -- if the "
            "fold-batch-dim token layout landed, this wrapper needs updating"
        )
        q, k, v = (
            q_BLNH.transpose(1, 2),
            k_BLNH.transpose(1, 2),
            v_BLNH.transpose(1, 2),
        )

        cap = self.logit_cap

        def softcap_mod(score, b, h, q_idx, kv_idx):
            return cap * torch.tanh(score / cap)

        out = self._flex_attention(
            q,
            k,
            v,
            score_mod=softcap_mod,
            scale=scale,
            enable_gqa=enable_gqa,
        )
        return out.transpose(1, 2)


class ReLUSquaredGLU(BinaryActivationFn):
    """ReLU-squared gated activation: ``relu(gate)**2 * up``.

    ReLU^2(x) = max(0, x)^2. Used in NanoGPT speedrun entries for faster
    convergence; the squared activation creates sharper sparsity patterns.

    This used to be a ReLUSquaredFeedForward subclass overriding forward() as
    ``w2(relu(w1(x))**2 * w3(x))``. #4535 made the fused gate-up projection the
    default: FeedForward.Config lost its w1/w3 fields for a single w13, and the
    base forward() now unflattens the interleaved gate/up pair and delegates to
    a configurable activation_fn. Expressing ReLU^2 as that activation keeps the
    exact same math while inheriting the base's remat regions and fused layout.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BinaryActivationFn.Config):
        pass

    def __init__(self, config: Config) -> None:
        pass

    def __call__(
        self,
        gate: torch.Tensor,
        up: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        h = F.relu(gate)
        return h * h * up


from torchtitan.experiments.ezpz.agpt.model import AgptModel

from torchtitan.models.common.param_init import depth_scaled_std
from torchtitan.models.llama3.model import Llama3TransformerBlock

__all__ = [
    "EzpzScaledDotProductAttention",
    "XPUScaledDotProductAttention",
    "_default_inner_attention",
    "model_registry",
]


_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}


def _linear_init(dim: int) -> dict[str, Callable]:
    """Weight init with std = sqrt(2/(5*d)), following Megatron-DeepSpeed.

    Reference: https://arxiv.org/pdf/2312.16903
    """
    s = (2.0 / (5 * dim)) ** 0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s),
        "bias": nn.init.zeros_,
    }


def _output_linear_init(dim: int) -> dict[str, Callable]:
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


def _depth_init(dim: int, layer_id: int) -> dict[str, Callable]:
    base_std = (2.0 / (5 * dim)) ** 0.5
    return {
        "weight": partial(
            nn.init.trunc_normal_, std=depth_scaled_std(base_std, layer_id)
        ),
        "bias": nn.init.zeros_,
    }


def _default_inner_attention() -> ScaledDotProductInnerAttention.Config:
    """Return the right SDPA config for the current device."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return XPUScaledDotProductAttention.Config()
    return EzpzScaledDotProductAttention.Config()


def _ezpz_get_attention_config(
    backend: str,
) -> Module.Config:
    """XPU-aware attention config selection.

    For the "sdpa" backend, uses the XPU-optimized SDPA classes instead
    of upstream's ScaledDotProductInnerAttention. Other backends delegate
    to upstream get_attention_config().

    Upstream PR #3571 (2026-06-09) removed SDPA + mask_type from the
    language-model attention path entirely; ezpz keeps the "sdpa"
    branch alive because XPU lacks a working FlexAttention backend.
    Returns just the config now (upstream dropped the (config, mask_type)
    tuple too).
    """
    if backend == "sdpa":
        return _default_inner_attention()
    return get_attention_config(backend)


def _build_agpt_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    hidden_dim: int,
    rope: RoPE.Config,
    n_kv_heads: int | None = None,
    attn_backend: str = "sdpa",
    qk_norm: bool = False,
    logit_softcap: float | None = None,
    relu_squared: bool = False,
) -> list[TransformerBlock.Config]:
    """Build a list of per-layer TransformerBlock configs with depth-scaled inits."""
    if logit_softcap is not None:
        inner_attention = SoftcappedFlexAttention.Config(
            logit_cap=logit_softcap,
        )
    else:
        inner_attention = _ezpz_get_attention_config(attn_backend)
    linear_init = _linear_init(dim)
    head_dim = dim // n_heads
    # QK-Norm uses LocalShardRMSNorm (a drop-in RMSNorm subclass) so the
    # per-head norm runs on the local TP shard, bypassing the native DTensor
    # RMSNorm backward that crashes on a Shard(2) 4-D tensor under AC=full
    # recompute at TP=4 ("tensor does not have a device"). Bit-identical:
    # head_dim and the norm weight are both unsharded/replicated on TP.
    # See local_rmsnorm.py.
    qk_norm_config = (
        LocalShardRMSNorm.Config(normalized_shape=head_dim, param_init=_NORM_INIT)
        if qk_norm
        else None
    )
    layers = []
    for layer_id in range(n_layers):
        # Both branches go through make_ffn_config now: #4535 fused w1/w3 into
        # a single interleaved w13, so hand-building the Config would have to
        # duplicate fused_gate_up_param_init's interleaving. ReLU^2 differs from
        # the SwiGLU default only in the activation.
        ffn_config = make_ffn_config(
            dim=dim,
            hidden_dim=hidden_dim,
            w1_param_init=linear_init,
            w2w3_param_init=_depth_init(dim, layer_id),
        )
        if relu_squared:
            ffn_config = dataclass_replace(
                ffn_config, activation_fn=ReLUSquaredGLU.Config()
            )
        layers.append(
            Llama3TransformerBlock.Config(
                attention_norm=RMSNorm.Config(
                    normalized_shape=dim, param_init=_NORM_INIT
                ),
                ffn_norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
                attention=make_gqa_config(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    wqkv_param_init=linear_init,
                    wo_param_init=_depth_init(dim, layer_id),
                    inner_attention=inner_attention,
                    rope=rope,
                    qk_norm=qk_norm_config,
                ),
                feed_forward=ffn_config,
            )
        )
    return layers


def _build_agpt_config(
    *,
    dim: int,
    n_layers: int,
    n_heads: int,
    n_kv_heads: int | None,
    rope_theta: int,
    vocab_size: int,
    hidden_dim: int,
    attn_backend: str = "sdpa",
    rope_backend: Literal["complex", "cos_sin"] = "complex",
    scaling: Literal["none", "llama", "yarn"] = "none",
    max_context_length: int = 131072,
    qk_norm: bool = False,
    logit_softcap: float | None = None,
    relu_squared: bool = False,
) -> AgptModel.Config:
    # PR #3458 (RoPE refactor): RoPE.Config split into ComplexRoPE.Config /
    # CosSinRoPE.Config; the backend= field is gone (backend is encoded in
    # the type). Top-level Model.Config.rope is gone too — each layer's
    # Attention.Config owns its own rope. ``rope_backend`` keyword on this
    # builder is kept for back-compat with existing config callers, but
    # internally it now selects a concrete subclass.
    rope_cls: type[RoPE.Config] = (
        ComplexRoPE.Config if rope_backend == "complex" else CosSinRoPE.Config
    )
    rope_cfg = rope_cls(
        dim=dim // n_heads,
        max_context_length=max_context_length,
        theta=rope_theta,
        scaling=scaling,
    )
    return AgptModel.Config(
        max_context_length=max_context_length,
        dim=dim,
        vocab_size=vocab_size,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=_build_agpt_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            hidden_dim=hidden_dim,
            attn_backend=attn_backend,
            rope=rope_cfg,
            qk_norm=qk_norm,
            logit_softcap=logit_softcap,
            relu_squared=relu_squared,
        ),
    )


agpt_configs = {
    "debugmodel": _build_agpt_config(
        dim=256,
        n_layers=6,
        n_heads=16,
        n_kv_heads=None,
        rope_theta=500000,
        vocab_size=32000,
        hidden_dim=compute_ffn_hidden_dim(256, multiple_of=256),
    ),
    # QK-norm adds 2 more RMSNorms per layer on head_dim, also initialized at
    # 1.0. Used by the master-weight-dtype ablation to test whether the
    # bf16 norm-freeze recurs for QK-norm gains -- the specific recurrence
    # risk cited in docs/production/agpt/30b-exp/README.md Section 6.
    "debugmodel_qknorm": _build_agpt_config(
        dim=256,
        n_layers=6,
        n_heads=16,
        n_kv_heads=None,
        rope_theta=500000,
        vocab_size=32000,
        hidden_dim=compute_ffn_hidden_dim(256, multiple_of=256),
        qk_norm=True,
    ),
    "debugmodel_flex_attn": _build_agpt_config(
        dim=256,
        n_layers=6,
        n_heads=16,
        n_kv_heads=None,
        rope_theta=500000,
        vocab_size=32000,
        hidden_dim=compute_ffn_hidden_dim(256, multiple_of=256),
        attn_backend="flex",
    ),
    "debugmodel_varlen_attn": _build_agpt_config(
        dim=256,
        n_layers=6,
        n_heads=16,
        n_kv_heads=None,
        rope_theta=500000,
        vocab_size=32000,
        hidden_dim=compute_ffn_hidden_dim(256, multiple_of=256),
        attn_backend="varlen",
    ),
    "2B": _build_agpt_config(
        dim=2048,
        n_layers=12,
        n_heads=16,
        n_kv_heads=4,
        rope_theta=50000,
        vocab_size=256128,
        hidden_dim=11008,
    ),
    # Historical 50,304-token AGPT whose smaller embedding/head budget is
    # reinvested in depth and FFN width.
    "2B_50K": _build_agpt_config(
        dim=2048,
        n_layers=24,
        n_heads=16,
        n_kv_heads=4,
        rope_theta=50000,
        vocab_size=50304,
        hidden_dim=10496,
    ),
    # [ezpz] agpt-2b variant matching the SFT checkpoint-900 HF config exactly
    # (vocab 256000, not the 256128 padding) + fused QKV so GRPO LoRA can target
    # ["wqkv","wo"]. Used by the RL overlay (experiments/ezpz/rl/alphabet_sort_agpt).
    "2b-rl": _build_agpt_config(
        dim=2048,
        n_layers=12,
        n_heads=16,
        n_kv_heads=4,
        rope_theta=50000,
        vocab_size=256000,
        hidden_dim=11008,
        # fuse_qkv=True was here until sync 84: #4526 removed the kwarg and
        # QKVLinear is now always fused, so this flavor's behavior is the
        # default. Checkpoints are unaffected -- attention.py:742-743 hooks
        # split/merge the logical wq/wk/wv keys on save/load.
        # flex: matches the proven-working fork run (v4/v5/v6). The RL vLLM
        # generator asserts varlen|flex (generator.py:799) then REPLACES
        # inner_attention with its own VLLMAttentionWrapper, so the generator
        # uses vLLM attention regardless; the trainer uses flex. varlen fails on
        # XPU (aten::_flash_attention_forward_no_dropout_inplace not implemented).
        # The config fn disables FlexAttention max_autotune to avoid XPU
        # OUT_OF_RESOURCES on the backward.
        attn_backend="flex",
    ),
    # [ezpz] agpt-2b variant matching the Megatron-DeepSpeed AuroraGPT-2B base
    # exactly (vocab 256000, the un-padded gemma-7b vocab the MDS run trained
    # with; the plain "2B" flavor pads to 256128). Plain (non-fused) QKV and the
    # default sdpa attention -- this is a PRETRAINING/anneal fork target, not an
    # RL/LoRA one, so it does not carry the "2b-rl" fused-QKV + flex layout.
    # Used by the mid-training anneal A/B configs (agpt_2b_mds_anneal_*), which
    # fork the converted MDS gs138650 DCP. Complex RoPE matches the production
    # "2B" flavor AND the Megatron-DeepSpeed base (adjacent-pair rotation), so
    # convert to HF with --model_flavor 2b-mds (complex): the state_dict_adapter
    # then APPLIES the Q/K permute, which is correct for a complex-trained base.
    # Do NOT convert this with a cos_sin ("_real") flavor -- that skips the
    # permute and corrupts the export.
    "2b-mds": _build_agpt_config(
        dim=2048,
        n_layers=12,
        n_heads=16,
        n_kv_heads=4,
        rope_theta=50000,
        vocab_size=256000,
        hidden_dim=11008,
    ),
    "2B_qknorm": _build_agpt_config(
        dim=2048,
        n_layers=12,
        n_heads=16,
        n_kv_heads=4,
        rope_theta=50000,
        vocab_size=256128,
        hidden_dim=11008,
        qk_norm=True,
    ),
    "2B_softcap": _build_agpt_config(
        dim=2048,
        n_layers=12,
        n_heads=16,
        n_kv_heads=4,
        rope_theta=50000,
        vocab_size=256128,
        hidden_dim=11008,
        logit_softcap=30.0,
    ),
    "2B_relu2": _build_agpt_config(
        dim=2048,
        n_layers=12,
        n_heads=16,
        n_kv_heads=4,
        rope_theta=50000,
        vocab_size=256128,
        hidden_dim=11008,
        relu_squared=True,
    ),
    "2B_kitchen_sink": _build_agpt_config(
        dim=2048,
        n_layers=12,
        n_heads=16,
        n_kv_heads=4,
        rope_theta=50000,
        vocab_size=256128,
        hidden_dim=11008,
        qk_norm=True,
        logit_softcap=30.0,
        relu_squared=True,
    ),
    "2B_flex_attn": _build_agpt_config(
        dim=2048,
        n_layers=12,
        n_heads=16,
        n_kv_heads=4,
        rope_theta=50000,
        vocab_size=256128,
        hidden_dim=11008,
        attn_backend="flex",
    ),
    "7B": _build_agpt_config(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        rope_theta=10000,
        vocab_size=32000,
        hidden_dim=11008,
    ),
    "8B": _build_agpt_config(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        rope_theta=500000,
        vocab_size=128256,
        hidden_dim=compute_ffn_hidden_dim(
            4096, multiple_of=1024, ffn_dim_multiplier=1.3
        ),
    ),
    "20B": _build_agpt_config(
        dim=5120,
        n_layers=64,
        n_heads=40,
        n_kv_heads=8,
        rope_theta=500000,
        vocab_size=256128,
        hidden_dim=compute_ffn_hidden_dim(5120, multiple_of=1024),
    ),
    "20B_flex_attn": _build_agpt_config(
        dim=5120,
        n_layers=64,
        n_heads=40,
        n_kv_heads=8,
        rope_theta=500000,
        vocab_size=256128,
        hidden_dim=compute_ffn_hidden_dim(5120, multiple_of=1024),
        attn_backend="flex",
    ),
    # 30B-exp: the proposed next flagship (docs/production/agpt/30b-exp/).
    # The proposal fixes only dim=6144; the rest is sized to sit consistently
    # between 20B (dim 5120, L=64) and 80B (dim 9216, L=84):
    #   dim=6144, L=64, H=48 (head_dim 128, matching 20B/80B), kv=8 GQA,
    #   ffn via the standard 2/3*4*dim rounded to 1024 -> 16384.
    # That lands at 28.1B params with the current 256,128 gemma vocab. NOTE the
    # proposal argues for a ~64k custom vocab, which would cut ~2.5B of
    # embedding; this config keeps gemma so it is directly comparable to the
    # existing 2B/20B/80B runs. Add a separate entry when the new tokenizer
    # exists rather than changing this one.
    "30B": _build_agpt_config(
        dim=6144,
        n_layers=64,
        n_heads=48,
        n_kv_heads=8,
        rope_theta=500000,
        vocab_size=256128,
        hidden_dim=compute_ffn_hidden_dim(6144, multiple_of=1024),
    ),
    # 30B-exp with the Llama-3 128k vocab instead of gemma's 256,128.
    #
    # The proposal asks for "~64k custom BPE", but we have no 64k tokenizer and
    # training one is its own project. Llama-3.1/3.2 (128k) is already vendored
    # in assets/hf/ and satisfies BOTH surviving arguments in the proposal:
    #   - cost: halves the embedding, 3.15B -> 1.57B params (11.2% -> 5.9% of
    #     the model). A 64k vocab would save only ~0.8B beyond this.
    #   - code fertility: the proposal's own table has gemma costing +17% on
    #     starcoder and +26% on Python vs Llama-3.1, and attributes that to
    #     gemma's merges rather than to vocab size.
    # 26.5B params. Needs assets/hf/Llama-3.1-8B (or 3.2-1B) as the tokenizer.
    #
    # NOT a drop-in swap for a gemma-trained checkpoint -- different vocab means
    # retokenizing the corpus. This is for the NEXT flagship, not a continuation.
    "30B_llama3tok": _build_agpt_config(
        dim=6144,
        n_layers=64,
        n_heads=48,
        n_kv_heads=8,
        rope_theta=500000,
        vocab_size=128256,
        hidden_dim=compute_ffn_hidden_dim(6144, multiple_of=1024),
    ),
    # 30B-exp with OLMo-2's 100,352 vocab. exp07's nine-tokenizer bake-off
    # measured this as tied with Llama-3 on fertility (225,749 vs 225,539
    # tok/MB, +0.09%) with a 22% smaller vocab, so it should cost 0.34B fewer
    # embedding params at dim=6144 (1.23B vs 1.58B) for the same tokens.
    # 100,352 (not the tokenizer's 100,278) matches OLMo-2's own config.json,
    # which pads to a multiple of 128; max added-token id is 100,277 so the
    # embedding covers the tokenizer with room to spare.
    # Needs assets/hf/OLMo-2-1124-7B as the tokenizer.
    "30B_olmo2tok": _build_agpt_config(
        dim=6144,
        n_layers=64,
        n_heads=48,
        n_kv_heads=8,
        rope_theta=500000,
        vocab_size=100352,
        hidden_dim=compute_ffn_hidden_dim(6144, multiple_of=1024),
    ),
    # Experimental node-local HSDP variant for 12-XPU Aurora/Sunspot nodes.
    # 16128 = 12 * 1344 and remains 256-aligned for efficient GEMMs. Keep this
    # separate from 30B_olmo2tok: changing the canonical flavor would make its
    # existing checkpoints shape-incompatible. Compared with 16384 this removes
    # 256 FFN channels (1.56%) while allowing dp_shard=12 parameter init.
    "30B_olmo2tok_dp12": _build_agpt_config(
        dim=6144,
        n_layers=64,
        n_heads=48,
        n_kv_heads=8,
        rope_theta=500000,
        vocab_size=100352,
        hidden_dim=16128,
    ),
    # ---- Aurora-native mid-ladder, OLMo-2 vocab ----
    #
    # The size ladder jumped 2B (dim 2048) straight to 20B (dim 5120); the 7B
    # and 8B in this dict are Llama-geometry ports on foreign vocabs (32000 /
    # 128256), not members of this family. These two fill that gap with the
    # same conventions as 20B/30B: head_dim 128, GQA with 8 kv heads,
    # compute_ffn_hidden_dim at multiple_of=1024.
    #
    # Both take OLMo-2's 100,352 vocab for the reason exp07 measured at 30B --
    # it tied Llama-3.1 on fertility (225,749 vs 225,539 tok/MB) with a 22%
    # smaller vocab -- and the argument is STRONGER here, because embedding is
    # a larger share of a smaller model. At dim 3072 gemma's 256,128 vocab
    # would be ~34% of the parameters; OLMo-2 makes it ~14%.
    #
    # Caveat on the mechanism, not the choice: exp07's framing was "freed HBM
    # converts into batch size". That held because the 30B was pinned at 76.8%
    # of a 64 GiB tile. These sizes are nowhere near that ceiling, so the vocab
    # buys parameters and fertility here, not throughput. Do not cite the
    # throughput half of exp07 for these.
    "5B_olmo2tok": _build_agpt_config(
        dim=3072,
        n_layers=40,
        n_heads=24,
        n_kv_heads=8,
        rope_theta=500000,
        vocab_size=100352,
        hidden_dim=compute_ffn_hidden_dim(3072, multiple_of=1024),
    ),
    # dim 4096 puts n_heads at 32, which is NOT divisible by 12, so TP=12 is
    # unavailable -- the 80B family deliberately picked 72 heads / 12 kv to
    # keep it. Accepted here: exp05 measured TP=4 costing 55% of throughput on
    # this stack (340 -> 151 tps), so TP>2 is not a capability these sizes
    # would use. TP in {2, 4, 8} all divide cleanly.
    "10B_olmo2tok": _build_agpt_config(
        dim=4096,
        n_layers=48,
        n_heads=32,
        n_kv_heads=8,
        rope_theta=500000,
        vocab_size=100352,
        hidden_dim=compute_ffn_hidden_dim(4096, multiple_of=1024),
    ),
    "50B": _build_agpt_config(
        dim=8192,
        n_layers=56,
        n_heads=64,
        n_kv_heads=8,
        rope_theta=500000,
        vocab_size=256128,
        hidden_dim=compute_ffn_hidden_dim(
            8192, multiple_of=1024, ffn_dim_multiplier=1.3
        ),
    ),
    # Same per-layer shape as 80B (dim=9216, 72 heads, 12 kv heads,
    # hidden_dim=25600) but only 48 layers. ~50B params total.
    # Distinct from "50B" (dim=8192) — this one keeps the 80B-family
    # head pattern (n_kv_heads=12) so it shares the TP=2 sharding plan.
    # Use as a smaller compile target that still exercises the
    # compile+AC+TP=2 path that the dense 80B configs depend on.
    "50B_wide": _build_agpt_config(
        dim=9216,
        n_layers=48,
        n_heads=72,
        n_kv_heads=12,
        rope_theta=500000,
        vocab_size=256128,
        hidden_dim=25600,
    ),
    # Same per-layer shape as 80B with 72 layers (~70B params total).
    # Bisect midpoint between the working 50B_wide (48L, no crash) and
    # the broken 80B (84L, DeviceMesh-in-saved-tensors AOT autograd
    # crash) — used to pin down whether the depth-sensitivity threshold
    # is at 72L or somewhere else in [48, 84).
    "70B_wide": _build_agpt_config(
        dim=9216,
        n_layers=72,
        n_heads=72,
        n_kv_heads=12,
        rope_theta=500000,
        vocab_size=256128,
        hidden_dim=25600,
    ),
    # Aurora-native ~80B configs: n_kv_heads=12, n_heads divisible by 12
    # so TP can be any factor of 12 (2, 3, 4, 6, 12).
    #
    # ~80.8B: Balanced width/depth. PP divides 84: {1,2,3,4,6,12}.
    "80B": _build_agpt_config(
        dim=9216,
        n_layers=84,
        n_heads=72,
        n_kv_heads=12,
        rope_theta=500000,
        vocab_size=256128,
        hidden_dim=25600,
    ),
    # 80B + QK-Norm: RMSNorm on Q,K per head before attention. Directly bounds
    # the attention-score magnitude -- the prime remaining suspect for the
    # dp>186 bf16 overflow after full-depth fp32 residual proved insufficient
    # (residual stream was necessary-but-not-sufficient; scores/another
    # activation still overflow). Near-free at train time; no ckpt-compat cost
    # (no surviving 80B production checkpoint to preserve).
    "80B_qknorm": _build_agpt_config(
        dim=9216,
        n_layers=84,
        n_heads=72,
        n_kv_heads=12,
        rope_theta=500000,
        vocab_size=256128,
        hidden_dim=25600,
        qk_norm=True,
    ),
    # 80B + logit softcap (Gemma-2 style, tanh score_mod). Caps attention
    # scores at +/-30 -- an alternative score-bounding lever to QK-Norm.
    "80B_softcap": _build_agpt_config(
        dim=9216,
        n_layers=84,
        n_heads=72,
        n_kv_heads=12,
        rope_theta=500000,
        vocab_size=256128,
        hidden_dim=25600,
        logit_softcap=30.0,
    ),
    # 80B + QK-Norm AND softcap: both score-bounding levers together, for the
    # wall test if either alone is insufficient at dp=192.
    "80B_qknorm_softcap": _build_agpt_config(
        dim=9216,
        n_layers=84,
        n_heads=72,
        n_kv_heads=12,
        rope_theta=500000,
        vocab_size=256128,
        hidden_dim=25600,
        qk_norm=True,
        logit_softcap=30.0,
    ),
    # ~80.0B: Wider (dim=10752), shallower (48 layers).
    # PP divides 48: {1,2,3,4,6,8,12,16,24}.
    "80B_wide": _build_agpt_config(
        dim=10752,
        n_layers=48,
        n_heads=84,
        n_kv_heads=12,
        rope_theta=500000,
        vocab_size=256128,
        hidden_dim=39936,
    ),
    # ~80.9B: Narrower (dim=7680), deeper (96 layers).
    # PP divides 96: {1,2,3,4,6,8,12,16,24}.
    "80B_deep": _build_agpt_config(
        dim=7680,
        n_layers=96,
        n_heads=60,
        n_kv_heads=12,
        rope_theta=500000,
        vocab_size=256128,
        hidden_dim=28672,
    ),
    # Variants with hidden_dim divisible by 12, for clean TP sharding
    # across all factors of 12 (2, 3, 4, 6, 12).
    "80B_alt": _build_agpt_config(
        dim=9216,
        n_layers=84,
        n_heads=72,
        n_kv_heads=12,
        rope_theta=500000,
        vocab_size=256128,
        hidden_dim=25596,  # 25600 -> 25596 (multiple of 12)
    ),
    "80B_deep_alt": _build_agpt_config(
        dim=7680,
        n_layers=96,
        n_heads=60,
        n_kv_heads=12,
        rope_theta=500000,
        vocab_size=256128,
        hidden_dim=28668,  # 28672 -> 28668 (multiple of 12)
    ),
}


# Case-insensitive aliases
agpt_configs["2b"] = agpt_configs["2B"]
agpt_configs["2b_50k"] = agpt_configs["2B_50K"]
agpt_configs["2b_flex_attn"] = agpt_configs["2B_flex_attn"]
agpt_configs["7b"] = agpt_configs["7B"]
agpt_configs["8b"] = agpt_configs["8B"]
agpt_configs["20b"] = agpt_configs["20B"]
agpt_configs["30b"] = agpt_configs["30B"]
agpt_configs["30b_llama3tok"] = agpt_configs["30B_llama3tok"]
agpt_configs["30b_olmo2tok"] = agpt_configs["30B_olmo2tok"]
agpt_configs["30b_olmo2tok_dp12"] = agpt_configs["30B_olmo2tok_dp12"]
agpt_configs["5b_olmo2tok"] = agpt_configs["5B_olmo2tok"]
agpt_configs["10b_olmo2tok"] = agpt_configs["10B_olmo2tok"]
agpt_configs["20b_flex_attn"] = agpt_configs["20B_flex_attn"]
agpt_configs["50b"] = agpt_configs["50B"]
agpt_configs["50b_wide"] = agpt_configs["50B_wide"]
agpt_configs["70b_wide"] = agpt_configs["70B_wide"]
agpt_configs["80b"] = agpt_configs["80B"]
agpt_configs["80b_wide"] = agpt_configs["80B_wide"]
agpt_configs["80b_qknorm"] = agpt_configs["80B_qknorm"]
agpt_configs["80b_softcap"] = agpt_configs["80B_softcap"]
agpt_configs["80b_qknorm_softcap"] = agpt_configs["80B_qknorm_softcap"]


def _as_cos_sin(config: "AgptModel.Config") -> "AgptModel.Config":
    """Return a deep copy of ``config`` with every layer's RoPE rebuilt as
    ``CosSinRoPE`` (the ``_real`` / rotate-half convention).

    The ``_real`` training flavors (see config_registry._set_rope_backend) flip
    the RoPE backend to cos_sin. For DCP->HF conversion we need a matching
    model-config flavor so convert_to_hf.py builds a cos_sin model and the
    RoPE-aware AgptStateDictAdapter skips the (complex-only) Q/K permute. This
    mirrors _set_rope_backend but operates on an AgptModel.Config directly.
    """
    from copy import deepcopy
    from dataclasses import fields

    cfg = deepcopy(config)
    for layer in cfg.layers:
        old_rope = layer.attention.rope
        if old_rope is None:
            continue
        kwargs = {f.name: getattr(old_rope, f.name) for f in fields(old_rope)}
        layer.attention.rope = CosSinRoPE.Config(**kwargs)
    return cfg


# ``_real`` (cos_sin RoPE) convert-time flavors. Production 2B/20B train with
# these (submit_agpt_{2b,20b}_autoretry.sh default CONFIG_SUFFIX=_real); pass
# e.g. ``--model_flavor 20b_real`` to convert_to_hf.py so the exported weights
# match the trained RoPE convention. See agpt/state_dict_adapter.py.
agpt_configs["2b_real"] = _as_cos_sin(agpt_configs["2B"])
agpt_configs["20b_real"] = _as_cos_sin(agpt_configs["20B"])
agpt_configs["80b_deep"] = agpt_configs["80B_deep"]
agpt_configs["80b_alt"] = agpt_configs["80B_alt"]
agpt_configs["80b_deep_alt"] = agpt_configs["80B_deep_alt"]
agpt_configs["2b_qknorm"] = agpt_configs["2B_qknorm"]
agpt_configs["2b_softcap"] = agpt_configs["2B_softcap"]
agpt_configs["2b_relu2"] = agpt_configs["2B_relu2"]
agpt_configs["2b_kitchen_sink"] = agpt_configs["2B_kitchen_sink"]


# ---------------------------------------------------------------------------
# muP (Maximal Update Parametrization) -- OPT-IN, purely additive
# ---------------------------------------------------------------------------
# Registers the muP width-ladder flavors (mup_1536 / mup_3072 / mup_6144 and a
# CPU-sized mup_tiny_* ladder). Every existing flavor above is untouched: the
# muP flavors are NEW keys, built by a separate builder, and
# register_mup_flavors raises rather than overwrite a name that already exists.
#
# Imported at the BOTTOM because agpt.mup imports the SDPA wrappers from this
# module to subclass them; at this point the module is fully populated, so the
# cycle resolves. See agpt/mup.py for the design and
# docs/experiments/mup/README.md for the audit.
from torchtitan.experiments.ezpz.agpt.mup import (  # noqa: E402
    default_mup_adamw,
    MupScaledDotProductAttention,
    MupXPUScaledDotProductAttention,
    register_mup_flavors,
)

register_mup_flavors(agpt_configs)

__all__ += [
    "default_mup_adamw",
    "MupScaledDotProductAttention",
    "MupXPUScaledDotProductAttention",
]


def model_registry(
    flavor: str,
    attn_backend: str = "sdpa",
    converters: list | None = None,
) -> AgptModel.Config:
    from copy import deepcopy

    # Upstream #4684 (post-sync-84) renamed validate_converter_order ->
    # validate_converter_compatibility. Same one-arg contract: it takes the
    # converter Config list and raises on an incompatible combination.
    from torchtitan.config.transform.converter import validate_converter_compatibility

    # [ezpz] deepcopy: agpt_configs[flavor] is a shared prebuilt config object
    # (unlike qwen3/llama3 which rebuild per call); converters mutate the tree,
    # so copy first to avoid corrupting the cached registry entry.
    config = deepcopy(agpt_configs[flavor])
    if converters is not None:
        validate_converter_compatibility(converters)
        for c in converters:
            config = c.build().convert(config)

    # The direct model contract stores this value on Decoder.Config. Keep the
    # validation here because the registry caches templates and converters may
    # replace attention subtrees before the config is returned.
    rope = config.layers[0].attention.rope
    if rope is None:
        raise ValueError(
            f"agpt flavor {flavor!r} has no RoPE config, so max_context_length "
            "cannot be validated"
        )
    if config.max_context_length != rope.max_context_length:
        raise ValueError(
            f"agpt flavor {flavor!r} has inconsistent context lengths: "
            f"model={config.max_context_length}, rope={rope.max_context_length}"
        )

    if flavor in ("2B_50K", "2b_50k"):
        from dataclasses import fields

        from .hf_state_dict_adapter import AGPTDenseStateDictAdapter

        class _Dense50KModel(AgptModel):
            state_dict_adapter_cls = AGPTDenseStateDictAdapter

            @dataclass(kw_only=True, slots=True)
            class Config(AgptModel.Config):
                pass

        return _Dense50KModel.Config(
            **{field.name: getattr(config, field.name) for field in fields(config)}
        )
    return config
