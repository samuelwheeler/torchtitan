# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn.functional as F
from torch.testing import assert_close

from torchtitan.models.common.config_utils import make_ffn_config
from torchtitan.models.common.moe import (
    GroupedExperts,
    _run_experts_batched_mm_padded,
    _run_experts_for_loop,
)
from torchtitan.models.common.token_dispatcher import LocalTokenDispatcher
from torchtitan.experiments.ezpz.moe import moe_configs
from torchtitan.experiments.ezpz.utils.count_moe_params import count_params


def _clone_for_grad(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.clone().detach().requires_grad_(True)


def _init_grouped_experts_weights(module: GroupedExperts) -> None:
    with torch.no_grad():
        module.w1.copy_(torch.randn_like(module.w1))
        module.w2.copy_(torch.randn_like(module.w2))
        module.w3.copy_(torch.randn_like(module.w3))


class TestMoEExpertBackends(unittest.TestCase):
    def test_50k_model_parameter_contract_and_backend_variants(self):
        expected = (10_564_138_496, 1_999_894_016)
        variants = {
            "10B_2B_50K_sdpa_for_loop": "for_loop",
            "10B_2B_50K_sdpa_aurora_sycl": "aurora_sycl",
            "10B_2B_50K_sdpa_aurora_full_loop": "aurora_full_loop",
            "10B_2B_50K_sdpa_aurora_full_sonic": "aurora_full_sonic",
        }
        for flavor, backend in variants.items():
            with self.subTest(flavor=flavor):
                self.assertEqual(count_params(flavor), expected)
                cfg = moe_configs[flavor]()
                self.assertEqual(cfg.vocab_size, 50_304)
                self.assertEqual(len(cfg.layers), 31)
                moe_layers = [layer.moe for layer in cfg.layers if layer.moe is not None]
                self.assertEqual(len(moe_layers), 30)
                self.assertEqual(
                    {layer.experts.compute_backend for layer in moe_layers},
                    {backend},
                )

    def test_aurora_full_layout_initializes_like_torchtitan(self):
        initial = {
            "w1": lambda tensor: torch.nn.init.constant_(tensor, 1.0),
            "w2": lambda tensor: torch.nn.init.constant_(tensor, 2.0),
            "w3": lambda tensor: torch.nn.init.constant_(tensor, 3.0),
        }
        config = GroupedExperts.Config(
            dim=7,
            hidden_dim=11,
            num_experts=5,
            use_grouped_mm=False,
            compute_backend="aurora_full_sonic",
            param_init=initial,
            token_dispatcher=LocalTokenDispatcher.Config(
                num_experts=5,
                top_k=2,
                score_before_experts=False,
            ),
        )
        experts = config.build()
        experts.init_states(buffer_device=torch.device("cpu"))

        self.assertEqual(tuple(experts.aurora_up.shape), (5, 7, 11))
        self.assertEqual(tuple(experts.aurora_gate.shape), (5, 7, 11))
        self.assertEqual(tuple(experts.aurora_down.shape), (5, 11, 7))
        self.assertTrue(experts.aurora_up.is_contiguous())
        self.assertTrue(experts.aurora_gate.is_contiguous())
        self.assertTrue(experts.aurora_down.is_contiguous())
        torch.testing.assert_close(experts.aurora_gate, torch.ones_like(experts.aurora_gate))
        torch.testing.assert_close(experts.aurora_down, torch.full_like(experts.aurora_down, 2.0))
        torch.testing.assert_close(experts.aurora_up, torch.full_like(experts.aurora_up, 3.0))

    def test_aurora_shared_views_preserve_feed_forward(self):
        torch.manual_seed(7)
        initial = {"weight": lambda tensor: torch.nn.init.normal_(tensor)}
        experts = GroupedExperts.Config(
            dim=7,
            hidden_dim=11,
            num_experts=5,
            use_grouped_mm=False,
            compute_backend="aurora_full_sonic",
            token_dispatcher=LocalTokenDispatcher.Config(
                num_experts=5,
                top_k=2,
                score_before_experts=False,
            ),
        ).build()
        shared = make_ffn_config(
            dim=7,
            hidden_dim=22,
            w1_param_init=initial,
            w2w3_param_init=initial,
        ).build()
        shared.init_states(buffer_device=torch.device("cpu"))

        up, gate, down = experts._aurora_shared_weights(shared)
        self.assertEqual(tuple(up.shape), (2, 7, 11))
        self.assertEqual(tuple(gate.shape), (2, 7, 11))
        self.assertEqual(tuple(down.shape), (2, 11, 7))

        x = torch.randn(13, 7)
        expected = shared(x)
        actual = torch.zeros_like(x)
        for index in range(2):
            actual = actual + (F.silu(x @ gate[index]) * (x @ up[index])) @ down[index]
        assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_batched_mm_padded_matches_for_loop_kernel(self):
        torch.manual_seed(0)

        num_experts = 5
        dim = 7
        hidden_dim = 11
        num_tokens_per_expert = torch.tensor([4, 0, 3, 1, 2], dtype=torch.int64)
        total_tokens = int(num_tokens_per_expert.sum().item())

        w1 = torch.randn(num_experts, hidden_dim, dim, dtype=torch.float32)
        w2 = torch.randn(num_experts, dim, hidden_dim, dtype=torch.float32)
        w3 = torch.randn(num_experts, hidden_dim, dim, dtype=torch.float32)
        x = torch.randn(total_tokens, dim, dtype=torch.float32)
        grad_out = torch.randn(total_tokens, dim, dtype=torch.float32)

        ref_w1 = _clone_for_grad(w1)
        ref_w2 = _clone_for_grad(w2)
        ref_w3 = _clone_for_grad(w3)
        ref_x = _clone_for_grad(x)
        ref_out = _run_experts_for_loop(
            ref_w1, ref_w2, ref_w3, ref_x, num_tokens_per_expert
        )
        ref_out.backward(grad_out)

        test_w1 = _clone_for_grad(w1)
        test_w2 = _clone_for_grad(w2)
        test_w3 = _clone_for_grad(w3)
        test_x = _clone_for_grad(x)
        test_out = _run_experts_batched_mm_padded(
            test_w1, test_w2, test_w3, test_x, num_tokens_per_expert
        )
        test_out.backward(grad_out)

        assert_close(test_out, ref_out, rtol=1e-5, atol=1e-5)
        assert_close(test_x.grad, ref_x.grad, rtol=1e-5, atol=1e-5)
        assert_close(test_w1.grad, ref_w1.grad, rtol=1e-5, atol=1e-5)
        assert_close(test_w2.grad, ref_w2.grad, rtol=1e-5, atol=1e-5)
        assert_close(test_w3.grad, ref_w3.grad, rtol=1e-5, atol=1e-5)

    def test_grouped_experts_backend_selector_matches_for_loop(self):
        torch.manual_seed(1)

        num_experts = 5
        dim = 9
        hidden_dim = 13
        dispatcher_config = LocalTokenDispatcher.Config(
            num_experts=num_experts,
            top_k=1,
            score_before_experts=True,
        )
        ref = GroupedExperts(
            GroupedExperts.Config(
                dim=dim,
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                use_grouped_mm=False,
                compute_backend="for_loop",
                token_dispatcher=dispatcher_config,
            )
        )
        test = GroupedExperts(
            GroupedExperts.Config(
                dim=dim,
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                use_grouped_mm=False,
                compute_backend="batched_mm_padded",
                token_dispatcher=dispatcher_config,
            )
        )

        _init_grouped_experts_weights(ref)
        with torch.no_grad():
            test.w1.copy_(ref.w1)
            test.w2.copy_(ref.w2)
            test.w3.copy_(ref.w3)

        num_tokens_per_expert = torch.tensor([3, 0, 2, 1, 4], dtype=torch.int64)
        total_tokens = int(num_tokens_per_expert.sum().item())
        x = torch.randn(total_tokens, dim, dtype=torch.float32)
        grad_out = torch.randn(total_tokens, dim, dtype=torch.float32)

        ref_x = _clone_for_grad(x)
        ref_out = ref._experts_forward(ref_x, num_tokens_per_expert)
        ref_out.backward(grad_out)

        test_x = _clone_for_grad(x)
        test_out = test._experts_forward(test_x, num_tokens_per_expert)
        test_out.backward(grad_out)

        assert_close(test_out, ref_out, rtol=1e-5, atol=1e-5)
        assert_close(test_x.grad, ref_x.grad, rtol=1e-5, atol=1e-5)
        assert_close(test.w1.grad, ref.w1.grad, rtol=1e-5, atol=1e-5)
        assert_close(test.w2.grad, ref.w2.grad, rtol=1e-5, atol=1e-5)
        assert_close(test.w3.grad, ref.w3.grad, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
