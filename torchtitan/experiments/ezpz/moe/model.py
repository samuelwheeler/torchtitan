# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.models.common.attention import (
    AttentionMasksType,
    BaseAttention,
    GQAttention,
    ScaledDotProductAttention,
)
from torchtitan.protocols.module import Module
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.rmsnorm import RMSNorm
from torchtitan.models.common.rope import apply_rotary_emb_single_complex
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability


class Attention(BaseAttention):
    """
    Multi-head latent attention (MLA) module.

    This is moe-specific and NOT shared with other models.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        n_heads: int
        dim: int
        wq: Linear.Config | None = None
        wq_a: Linear.Config | None = None
        wq_b: Linear.Config | None = None
        wkv_a: Linear.Config
        wkv_b: Linear.Config
        wo: Linear.Config
        q_lora_rank: int = 0
        kv_lora_rank: int = 512
        q_norm: RMSNorm.Config
        kv_norm: RMSNorm.Config
        qk_nope_head_dim: int = 128
        qk_rope_head_dim: int = 64
        v_head_dim: int = 128
        inner_attention: Module.Config
        mask_type: str = "causal"
        mscale: float = 1.0
        rope_factor: float = 1.0
        rope_max_seq_len: int = 4096
        rope_original_seq_len: int = 4096

    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim

        if self.q_lora_rank == 0:
            assert config.wq is not None, "wq is required when q_lora_rank == 0"
            self.wq = config.wq.build()
        else:
            assert (
                config.wq_a is not None and config.wq_b is not None
            ), "wq_a and wq_b are required when q_lora_rank > 0"
            self.wq_a = config.wq_a.build()
            self.q_norm = config.q_norm.build()
            self.wq_b = config.wq_b.build()

        # TODO(fegin): revisit
        # https://github.com/pytorch/torchtitan/pull/2785#discussion_r3034078575
        self.wkv_a = config.wkv_a.build()
        self.kv_norm = config.kv_norm.build()
        self.wkv_b = config.wkv_b.build()
        self.wo = config.wo.build()
        self.softmax_scale = self.qk_head_dim**-0.5

        if config.rope_max_seq_len > config.rope_original_seq_len:
            mscale = 0.1 * config.mscale * math.log(config.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        self.inner_attention = config.inner_attention.build()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        bsz, seqlen, _ = x.size()

        # Query projection
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_a(x)
            q = self.wq_b(self.q_norm(q))
        q = q.view(bsz, seqlen, -1, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        q_pe = apply_rotary_emb_single_complex(q_pe, freqs_cis, positions)
        q = torch.cat([q_nope, q_pe], dim=-1)

        # Key-value projection
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        k_pe = apply_rotary_emb_single_complex(k_pe.unsqueeze(2), freqs_cis, positions)

        kv = self.wkv_b(self.kv_norm(kv))
        kv = kv.view(bsz, seqlen, -1, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1)

        # NOTE: The XPU SDPA backend on Aurora doesn't properly handle
        # different head dimensions for Q/K vs V.
        # On Intel XPU (Aurora), F.scaled_dot_product_attention returns output
        # with Q/K's head dimension instead of V's.
        # Standard CUDA backends correctly return (B, H, L, Ev) when E != Ev,
        # but the XPU MATH backend does not.
        pad_v = self.qk_head_dim != self.v_head_dim
        if pad_v:
            v = F.pad(v, (0, self.qk_head_dim - self.v_head_dim))

        output = self.inner_attention(
            q, k, v, attention_masks=attention_masks, scale=self.softmax_scale
        )

        if pad_v:
            output = output[..., : self.v_head_dim]

        output = output.contiguous()
        output = output.view(bsz, seqlen, -1)
        return self.wo(output)


class moeTransformerBlock(TransformerBlock):
    """
    moe Transformer block with attention and feed-forward layers.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()
        self.attention = config.attention.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()

        self.moe_enabled = config.moe is not None
        if self.moe_enabled:
            assert config.moe is not None
            self.moe = config.moe.build()
        else:
            assert config.feed_forward is not None
            self.feed_forward = config.feed_forward.build()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        x = x + self.attention(
            self.attention_norm(x), freqs_cis, attention_masks, positions
        )
        if self.moe_enabled:
            x = x + self.moe(self.ffn_norm(x))
        else:
            x = x + self.feed_forward(self.ffn_norm(x))
        return x


class moeModel(Decoder):
    """
    moe Transformer model with attention and feed-forward layers.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 2048
        vocab_size: int = 102400

        def update_from_config(
            self,
            *,
            trainer_config,
            **kwargs,
        ) -> None:
            training = trainer_config.training
            parallelism = trainer_config.parallelism
            debug = trainer_config.debug
            seq_len = training.seq_len
            if seq_len > self.rope.max_seq_len:
                logger.warning(
                    f"Sequence length {seq_len} exceeds original maximum {self.rope.max_seq_len}."
                )
            self.rope = dataclasses.replace(self.rope, max_seq_len=seq_len)

            # MLA keeps additional rope metadata on each attention module.
            # GQA consumes the shared Decoder rope cache directly.
            for layer_cfg in self.layers:
                attention = layer_cfg.attention
                if isinstance(attention, Attention.Config):
                    attention.rope_max_seq_len = seq_len
                    attention.rope_factor = self.rope.rope_factor
                    attention.rope_original_seq_len = self.rope.original_seq_len
                elif not isinstance(attention, GQAttention.Config):
                    raise TypeError(
                        "MoE model supports MLA or GQA attention, got "
                        f"{type(attention).__name__}."
                    )

            first_attention = self.layers[0].attention
            tp = parallelism.tensor_parallel_degree
            if tp > 1 and isinstance(first_attention, GQAttention.Config):
                n_heads = first_attention.n_heads
                n_kv_heads = first_attention.n_kv_heads or n_heads
                if n_heads % tp != 0:
                    raise ValueError(
                        f"tensor_parallel_degree ({tp}) must divide n_heads ({n_heads})."
                    )
                if n_kv_heads % tp != 0:
                    raise ValueError(
                        "tensor_parallel_degree "
                        f"({tp}) must divide n_kv_heads ({n_kv_heads})."
                    )

            for layer_cfg in self.layers:
                if layer_cfg.moe is not None:
                    expert_compute_backend = layer_cfg.moe.experts.compute_backend
                    if (
                        (
                            expert_compute_backend == "grouped_mm"
                            or (
                                expert_compute_backend is None
                                and layer_cfg.moe.experts.use_grouped_mm
                            )
                        )
                        and not has_cuda_capability(9, 0)
                    ):
                        logger.warning(
                            "Failed to use grouped mm, which is only supported on SM90 or later",
                        )
                        if expert_compute_backend == "grouped_mm":
                            layer_cfg.moe.experts.compute_backend = "for_loop"
                        layer_cfg.moe.experts.use_grouped_mm = False
                    layer_cfg.moe.router._debug_force_load_balance = (
                        debug.moe_force_load_balance
                    )
                    # ETP was deprecated upstream (#3167); the comm_backend now
                    # lives on the token_dispatcher, not on parallelism config.
                    comm_backend = getattr(
                        layer_cfg.moe.experts.token_dispatcher,
                        "comm_backend",
                        "standard",
                    )
                    if comm_backend in ("deepep", "hybridep"):
                        if parallelism.expert_parallel_degree == 1:
                            raise ValueError(
                                f"{comm_backend.upper()} requires expert "
                                "parallelism (expert_parallel_degree > 1)."
                            )
                        from torchtitan.models.common.moe_deepep import DeepEPMoE

                        init_kwargs = {
                            f.name: getattr(layer_cfg.moe, f.name)
                            for f in dataclasses.fields(layer_cfg.moe)
                            if f.init
                        }
                        layer_cfg.moe = DeepEPMoE.Config(**init_kwargs)

            if parallelism.context_parallel_degree > 1 and not isinstance(
                self.layers[0].attention.inner_attention,
                ScaledDotProductAttention.Config,
            ):
                raise NotImplementedError(
                    "Context Parallel for MoE only supports "
                    "ScaledDotProductAttention. Got "
                    f"{type(self.layers[0].attention.inner_attention).__name__}."
                )

            # Fill ShardingConfig on every sub-module config so
            # Module.parallelize(tp_mesh) can distribute params/activations.
            # MoE blocks are intentionally skipped — apply_moe_ep_tp handles
            # them at parallelize-time, mirroring upstream deepseek_v3.
            from torchtitan.experiments.ezpz.moe.sharding import (
                set_moe_sharding_config,
            )

            set_moe_sharding_config(
                self,
                loss_parallel=not parallelism.disable_loss_parallel,
                enable_sp=parallelism.enable_sequence_parallel,
            )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            attention = self.layers[0].attention
            if isinstance(attention, Attention.Config):
                n_heads = attention.n_heads
                head_dims = (
                    attention.qk_nope_head_dim
                    + attention.qk_rope_head_dim
                    + attention.v_head_dim
                )
            elif isinstance(attention, GQAttention.Config):
                n_heads = attention.n_heads
                head_dim = attention.head_dim or self.dim // n_heads
                head_dims = 2 * head_dim
            else:
                raise TypeError(
                    "MoE parameter accounting supports MLA or GQA attention, got "
                    f"{type(attention).__name__}."
                )
            return get_moe_model_nparams_and_flops(
                self,
                model,
                n_heads,
                head_dims,
                seq_len,
            )
