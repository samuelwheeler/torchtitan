# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch import nn

from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import TokenChoiceTopKRouter
from torchtitan.models.common.token_dispatcher import LocalTokenDispatcher


class TestMoERoutingCounts(unittest.TestCase):
    def test_bf16_router_ties_have_stable_expert_order(self):
        class FixedGate(nn.Module):
            def forward(self, x):
                return torch.tensor(
                    [[0.0, 0.0, 0.0, -1.0]], dtype=torch.bfloat16
                ).expand(x.shape[0], -1)

        router = TokenChoiceTopKRouter(
            TokenChoiceTopKRouter.Config(
                num_experts=4,
                gate=Linear.Config(in_features=3, out_features=4, bias=False),
                top_k=2,
                score_func="sigmoid",
            )
        )
        router.gate = FixedGate()

        # The production router passes an FP32 load-balancing bias. Cover that
        # promotion explicitly: the choice keys must still have a deterministic
        # expert-id tiebreak when the underlying gate scores are BF16.
        _, expert_ids, counts = router(
            torch.zeros(5, 3, dtype=torch.bfloat16),
            expert_bias=torch.zeros(4, dtype=torch.float32),
        )

        expected_ids = torch.tensor([[0, 1]]).expand(5, -1)
        torch.testing.assert_close(expert_ids.sort(dim=-1).values, expected_ids)
        torch.testing.assert_close(counts, torch.tensor([5, 5, 0, 0]))

    def test_dispatch_counts_are_integral_and_exact(self):
        dispatcher = LocalTokenDispatcher(
            LocalTokenDispatcher.Config(
                num_experts=5,
                top_k=2,
                score_before_experts=True,
            )
        )
        x = torch.randn(4, 3)
        expert_ids = torch.tensor([[0, 4], [1, 4], [1, 3], [4, 0]])
        scores = torch.ones(4, 2)

        _, counts, _ = dispatcher.dispatch(x, scores, expert_ids)

        self.assertEqual(counts.dtype, torch.int64)
        torch.testing.assert_close(counts, torch.tensor([2, 2, 0, 1, 3]))


if __name__ == "__main__":
    unittest.main()
