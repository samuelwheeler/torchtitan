# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json

import pytest

from torchtitan.experiments.ezpz.eval.hf_agpt_moe.assets import (
    validate_tokenizer,
    write_hf_assets,
)
from torchtitan.experiments.ezpz.moe import model_registry


def test_hf_assets_are_derived_from_the_registered_model(tmp_path, tokenizer_dir):
    output = tmp_path / "hf"
    config = model_registry("AGPT_2B_50K_MOE_sdpa_aurora_full_sonic")

    write_hf_assets(output, config, tokenizer_dir, "bfloat16")
    hf_config = json.loads((output / "config.json").read_text())

    assert hf_config["hidden_size"] == 2048
    assert hf_config["num_hidden_layers"] == 24
    assert hf_config["num_attention_heads"] == 16
    assert hf_config["num_key_value_heads"] == 4
    assert hf_config["num_local_experts"] == 36
    assert hf_config["num_experts_per_tok"] == 3
    assert hf_config["moe_intermediate_size"] == 2112
    assert hf_config["shared_expert_intermediate_size"] == 4224
    assert hf_config["rope_theta"] == 50000
    assert hf_config["auto_map"]["AutoModelForCausalLM"].endswith("AGPTMoEForCausalLM")
    assert (output / "modeling_agpt_moe.py").is_file()
    assert (output / "configuration_agpt_moe.py").is_file()
    assert (output / "tokenizer.model").read_bytes() == (
        tokenizer_dir / "tokenizer.model"
    ).read_bytes()


def test_tokenizer_rejects_vocabulary_outside_model(tokenizer_dir):
    with pytest.raises(ValueError, match="exceeds model vocabulary"):
        validate_tokenizer(tokenizer_dir, 3)


def test_tokenizer_rejects_incompatible_special_tokens(tokenizer_dir):
    from sentencepiece import sentencepiece_model_pb2

    path = tokenizer_dir / "tokenizer.model"
    model = sentencepiece_model_pb2.ModelProto()
    model.ParseFromString(path.read_bytes())
    model.pieces[1].piece = "<wrong-bos>"
    path.write_bytes(model.SerializeToString())
    with pytest.raises(ValueError, match="special tokens"):
        validate_tokenizer(tokenizer_dir, 32)
