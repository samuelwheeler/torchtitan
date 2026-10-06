# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from .configuration_agpt_moe import AGPTMoEConfig


class AGPTExpert(nn.Module):
    def __init__(self, config, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        gated = self.act_fn(self.gate_proj(hidden_states))
        return self.down_proj(gated * self.up_proj(hidden_states))


class AGPTSparseMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.router_score_function != "softmax":
            raise ValueError("AGPTSparseMoE supports softmax routing only")
        if config.norm_topk_prob:
            raise ValueError("AGPTSparseMoE requires unnormalized top-k weights")
        if not config.router_bias_selection:
            raise ValueError("AGPTSparseMoE requires routing-only expert bias")
        self.top_k = config.num_experts_per_tok
        self.router_scale = config.router_scale
        self.gate = nn.Linear(
            config.hidden_size, config.num_local_experts, bias=False
        )
        self.experts = nn.ModuleList(
            AGPTExpert(config, config.moe_intermediate_size)
            for _ in range(config.num_local_experts)
        )
        self.shared_experts = AGPTExpert(
            config, config.shared_expert_intermediate_size
        )
        self.register_buffer(
            "expert_bias",
            torch.zeros(config.num_local_experts, dtype=torch.float32),
            persistent=True,
        )

    def forward(self, hidden_states):
        output_shape = hidden_states.shape
        tokens = hidden_states.reshape(-1, output_shape[-1])
        logits = F.linear(tokens.float(), self.gate.weight.float())
        scores = torch.softmax(logits, dim=-1)
        expert_ids = torch.topk(
            scores + self.expert_bias, self.top_k, dim=-1, sorted=False
        ).indices
        weights = scores.gather(-1, expert_ids) * self.router_scale
        routed = torch.zeros_like(tokens)
        for expert_id, expert in enumerate(self.experts):
            token_ids, slots = torch.where(expert_ids == expert_id)
            if token_ids.numel() == 0:
                continue
            expert_output = expert(tokens[token_ids])
            expert_output = expert_output * weights[token_ids, slots, None].to(
                expert_output.dtype
            )
            routed.index_add_(0, token_ids, expert_output)
        output = routed + self.shared_experts(tokens)
        return output.reshape(output_shape)


class AGPTMoEForCausalLM(LlamaForCausalLM):
    config_class = AGPTMoEConfig

    def __init__(self, config):
        super().__init__(config)
        for layer in self.model.layers:
            layer.mlp = AGPTSparseMoE(config)
            layer.mlp.apply(self._init_weights)
