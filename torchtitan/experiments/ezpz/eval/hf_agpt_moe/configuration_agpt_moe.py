# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from transformers.models.llama.configuration_llama import LlamaConfig


class AGPTMoEConfig(LlamaConfig):
    model_type = "agpt_moe"

    def __init__(
        self,
        num_local_experts=36,
        num_experts_per_tok=3,
        moe_intermediate_size=2112,
        shared_expert_intermediate_size=4224,
        router_score_function="softmax",
        norm_topk_prob=False,
        router_bias_selection=True,
        router_scale=1.0,
        **kwargs,
    ):
        intermediate_size = kwargs.pop("intermediate_size", moe_intermediate_size)
        if intermediate_size != moe_intermediate_size:
            raise ValueError("intermediate_size must equal moe_intermediate_size")
        super().__init__(intermediate_size=intermediate_size, **kwargs)
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.router_score_function = router_score_function
        self.norm_topk_prob = norm_topk_prob
        self.router_bias_selection = router_bias_selection
        self.router_scale = router_scale
