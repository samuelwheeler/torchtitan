# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file

from torchtitan.experiments.ezpz.eval.convert_to_hf import convert_to_hf
from torchtitan.experiments.ezpz.moe.agpt_hf_state_dict_adapter import (
    AGPTMoEStateDictAdapter,
)
from torchtitan.experiments.ezpz.tests.moe.hf_reference import (
    logical_state_dict,
    reference_hf_state_dict,
)


@pytest.mark.parametrize("layout", ["native", "legacy"])
def test_dcp_to_hf_preserves_mapping_and_bias(
    layout,
    tiny_moe_config,
    tokenizer_dir,
    tmp_path,
    monkeypatch,
):
    from torchtitan.experiments.ezpz import moe

    monkeypatch.setattr(moe, "model_registry", lambda flavor: tiny_moe_config)
    model = tiny_moe_config.build()
    model.init_states(buffer_device=torch.device("cpu"))
    source = model.state_dict()
    source["layers.0.moe.expert_bias_E"].copy_(torch.tensor([0.001003, 0.001004]))
    expected = reference_hf_state_dict(source, tiny_moe_config)
    if layout == "legacy":
        logical = logical_state_dict(source, tiny_moe_config)
        source = {}
        names = {
            "w1_EFD": "aurora_gate",
            "w3_EFD": "aurora_up",
            "w2.weight": "aurora_down",
        }
        for key, value in logical.items():
            if ".moe.routed_experts." in key:
                prefix, projection = key.split(".moe.routed_experts.")
                source[f"{prefix}.moe.experts.{names[projection]}"] = value.transpose(
                    -2, -1
                ).contiguous()
            else:
                source[
                    key.replace(".expert_bias_E", ".expert_bias")
                ] = value.contiguous()
    checkpoint, output = tmp_path / "dcp", tmp_path / "hf"
    dcp.save(source, checkpoint_id=checkpoint)
    convert_to_hf(
        checkpoint,
        output,
        "experiments.ezpz.moe",
        "tiny",
        tokenizer_dir,
        "bfloat16",
    )
    actual = {}
    for shard in output.glob("*.safetensors"):
        actual.update(load_file(shard))
    assert set(actual) == set(expected)
    for key, value in expected.items():
        dtype = torch.float32 if key.endswith(".expert_bias") else torch.bfloat16
        assert actual[key].dtype == dtype
        torch.testing.assert_close(actual[key], value.to(dtype), rtol=0, atol=0)


@pytest.mark.parametrize("failure", ["weights", "assets"])
def test_converter_failure_never_publishes(
    tiny_moe_config, tokenizer_dir, tmp_path, monkeypatch, failure
):
    from torchtitan.experiments.ezpz import moe

    converter = importlib.import_module(
        "torchtitan.experiments.ezpz.eval.convert_to_hf"
    )
    monkeypatch.setattr(moe, "model_registry", lambda flavor: tiny_moe_config)
    model = tiny_moe_config.build()
    source, output = tmp_path / "dcp", tmp_path / "hf"
    dcp.save(model.state_dict(), checkpoint_id=source)

    def fail(*args, **kwargs):
        raise OSError(f"{failure} write failed")

    if failure == "weights":
        monkeypatch.setattr(converter.dcp, "save", fail)
    else:
        monkeypatch.setattr(AGPTMoEStateDictAdapter, "write_hf_assets", fail)
    with pytest.raises(OSError, match="write failed"):
        converter.convert_to_hf(
            source, output, "experiments.ezpz.moe", "tiny", tokenizer_dir, "bfloat16"
        )
    assert not output.exists()
    assert list(tmp_path.glob(".hf.staging-*"))
