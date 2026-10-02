import json

from torchtitan.experiments.ezpz.eval.hf_agpt_moe.assets import write_hf_assets
from torchtitan.experiments.ezpz.moe import model_registry


def test_hf_assets_are_derived_from_the_registered_model(tmp_path):
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "tokenizer.model").write_bytes(b"test-tokenizer")
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
    assert hf_config["auto_map"]["AutoModelForCausalLM"].endswith(
        "AGPTMoEForCausalLM"
    )
    assert (output / "modeling_agpt_moe.py").is_file()
    assert (output / "configuration_agpt_moe.py").is_file()
    assert (output / "tokenizer.model").read_bytes() == b"test-tokenizer"
