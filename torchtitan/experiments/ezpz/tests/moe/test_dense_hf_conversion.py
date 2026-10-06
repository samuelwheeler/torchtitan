# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import pickle
from dataclasses import fields

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
)

from torchtitan.experiments.ezpz import agpt
from torchtitan.experiments.ezpz.agpt.hf_state_dict_adapter import (
    AGPTDenseStateDictAdapter,
)
from torchtitan.experiments.ezpz.eval.convert_to_hf import convert_to_hf
from torchtitan.experiments.ezpz.tests.moe.hf_reference import (
    logical_state_dict,
    reference_hf_state_dict,
)


def test_dense_50k_hf_config_and_full_model_shapes(tmp_path, tokenizer_dir):
    config = agpt.model_registry("2b_50k")
    assert pickle.loads(pickle.dumps(type(config))) is type(config)
    adapter = AGPTDenseStateDictAdapter(config, tokenizer_dir)
    adapter.write_hf_assets(tmp_path, "bfloat16")
    hf_config = json.loads((tmp_path / "config.json").read_text())
    assert (
        hf_config["num_hidden_layers"],
        hf_config["vocab_size"],
        hf_config["intermediate_size"],
    ) == (24, 50304, 10496)
    with torch.device("meta"):
        native = config.build()
        hf = LlamaForCausalLM(LlamaConfig(**hf_config))
    exported = adapter.to_hf(native.state_dict())
    assert exported.keys() == hf.state_dict().keys()
    assert all(exported[k].shape == v.shape for k, v in hf.state_dict().items())
    tokenizer = AutoTokenizer.from_pretrained(tmp_path)
    assert tokenizer.bos_token_id == 1 and tokenizer.eos_token_id == 2
    assert tokenizer.encode("Aurora")[0] == 1


@pytest.mark.parametrize("layout", ["native", "legacy"])
@pytest.mark.parametrize("assets_mode", ["generated", "static"])
def test_dense_dcp_hf_weights_and_logits(
    layout, assets_mode, tmp_path, tokenizer_dir, monkeypatch
):
    template = agpt.model_registry("2b_50k")
    small = agpt._build_agpt_config(
        dim=16,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        rope_theta=50000,
        vocab_size=32,
        hidden_dim=32,
        max_context_length=16,
    )
    config = type(template)(**{f.name: getattr(small, f.name) for f in fields(small)})
    monkeypatch.setattr(agpt, "model_registry", lambda flavor: config)
    monkeypatch.setattr(agpt, "_EZPZ_MAX_CONTEXT_LENGTH", 4)
    torch.manual_seed(17)
    native = config.build().eval()
    native.init_states(buffer_device=torch.device("cpu"))
    source = native.state_dict()
    expected = reference_hf_state_dict(source, config)
    if layout == "legacy":
        source = logical_state_dict(source, config)
    static_config = None
    if assets_mode == "static":
        static_assets = tmp_path / "static-assets"
        static_assets.mkdir()
        AGPTDenseStateDictAdapter(config, tokenizer_dir).write_hf_assets(
            static_assets, "float32"
        )
        AutoTokenizer.from_pretrained(static_assets).save_pretrained(static_assets)
        tokenizer_dir = static_assets
        static_config = static_assets / "config.json"
        monkeypatch.setattr(AGPTDenseStateDictAdapter, "write_hf_assets", None)
    checkpoint, output = tmp_path / "dcp", tmp_path / "hf"
    dcp.save(source, checkpoint_id=checkpoint)
    convert_to_hf(
        checkpoint,
        output,
        "experiments.ezpz.agpt",
        "2b_50k",
        tokenizer_dir,
        "float32",
        static_config,
    )
    actual = {}
    for shard in output.glob("*.safetensors"):
        actual.update(load_file(shard))
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    hf = AutoModelForCausalLM.from_pretrained(output).eval()
    tokens = torch.tensor([1, 3, 5, 7])
    with torch.inference_mode():
        native_logits = native(tokens)
        hf_logits = hf(tokens[None], use_cache=False).logits[0]
    torch.testing.assert_close(hf_logits, native_logits, rtol=1e-4, atol=1e-5)
