# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan.experiments.ezpz.eval.convert_to_hf import _prepare_hf_state_dict


def test_prepare_hf_state_dict_casts_and_packs_transpose_views():
    transposed = torch.arange(24, dtype=torch.float32).view(4, 6).T
    assert not transposed.is_contiguous()

    prepared = _prepare_hf_state_dict({"expert.weight": transposed}, torch.bfloat16)

    weight = prepared["expert.weight"]
    assert weight.dtype == torch.bfloat16
    assert weight.is_contiguous()
    torch.testing.assert_close(weight.float(), transposed)


def test_prepare_hf_state_dict_packs_without_dtype_change():
    transposed = torch.arange(12, dtype=torch.float32).view(3, 4).T

    weight = _prepare_hf_state_dict({"weight": transposed}, torch.float32)["weight"]

    assert weight.dtype == torch.float32
    assert weight.is_contiguous()
    torch.testing.assert_close(weight, transposed)


def test_prepare_preserves_exact_fp32_routing_bias(tiny_moe_config, tmp_path):
    from safetensors.torch import load_file, save_file

    from torchtitan.experiments.ezpz.moe.agpt_hf_state_dict_adapter import (
        AGPTMoEStateDictAdapter,
    )

    key = "model.layers.0.mlp.expert_bias"
    bias = torch.tensor([0.001003, 0.001004], dtype=torch.float32)
    adapter = AGPTMoEStateDictAdapter(tiny_moe_config, None)
    prepared = _prepare_hf_state_dict(
        {key: bias, "weight": torch.ones(2)},
        torch.bfloat16,
        adapter.hf_dtype_overrides(),
    )
    save_file(prepared, tmp_path / "model.safetensors")
    loaded = load_file(tmp_path / "model.safetensors")
    assert loaded["weight"].dtype == torch.bfloat16
    assert loaded[key].dtype == torch.float32
    torch.testing.assert_close(loaded[key], bias, rtol=0, atol=0)
