import torch

from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.experiments.ezpz.moe import model_registry, moe_configs
from torchtitan.experiments.ezpz.moe.config_registry import (
    agpt_2b_50k_moe_sdpa_aurora_full_sonic,
)
from torchtitan.models.common.attention import GQAttention


FLAVOR = "AGPT_2B_50K_MOE_sdpa_aurora_full_sonic"


def test_agpt_moe_architecture_matches_dense_backbone():
    config = moe_configs[FLAVOR]()

    assert config.dim == 2048
    assert config.vocab_size == 50304
    assert len(config.layers) == 24
    assert config.rope.dim == 128
    assert config.rope.theta == 50000
    assert config.rope.scaling == "none"

    for layer in config.layers:
        assert isinstance(layer.attention, GQAttention.Config)
        assert layer.attention.n_heads == 16
        assert layer.attention.n_kv_heads == 4
        assert layer.feed_forward is None
        assert layer.moe.num_experts == 36
        assert layer.moe.router.top_k == 3
        assert layer.moe.router.score_func == "softmax"
        assert not layer.moe.router.route_norm
        assert layer.moe.load_balance_coeff == 1e-3
        assert layer.moe.experts.hidden_dim == 2112
        assert layer.moe.shared_experts.w1.out_features == 4224
        assert layer.moe.experts.compute_backend == "aurora_full_sonic"


def test_agpt_moe_active_and_total_parameter_counts():
    config = moe_configs[FLAVOR]()
    with torch.device("meta"):
        model = config.build()

    counts = {"dense": 0, "router": 0, "shared": 0, "routed": 0}
    for name, parameter in model.named_parameters():
        if ".moe.router." in name:
            counts["router"] += parameter.numel()
        elif ".moe.shared_experts." in name:
            counts["shared"] += parameter.numel()
        elif ".moe.experts." in name:
            counts["routed"] += parameter.numel()
        else:
            counts["dense"] += parameter.numel()

    total = sum(counts.values())
    active = (
        counts["dense"]
        + counts["router"]
        + counts["shared"]
        + counts["routed"] * 3 // 36
    )
    assert total == 12_293_801_984
    assert active == 2_016_708_608
    measured_total, flops_2048 = config.get_nparams_and_flops(model, 2048)
    _, flops_4096 = config.get_nparams_and_flops(model, 4096)
    assert measured_total == total
    assert flops_2048 == 12_690_075_648
    assert flops_4096 == 13_898_035_200


def test_agpt_moe_training_config_wires_balancing_and_gqa_sharding():
    trainer_config = agpt_2b_50k_moe_sdpa_aurora_full_sonic()
    spec = trainer_config.model_spec

    assert spec.post_optimizer_build_fn is register_moe_load_balancing_hook
    assert spec.state_dict_adapter is None
    assert trainer_config.activation_checkpoint.mode == "selective"
    assert trainer_config.tokenizer.backend == "hf"
    assert trainer_config.hf_assets_path.endswith("olmo-7b-0724-hf")
    spec.model.update_from_config(trainer_config=trainer_config)
    assert spec.model.layers[0].attention.sharding_config is not None


def test_agpt_moe_model_registry_keeps_native_checkpoint_support():
    spec = model_registry(FLAVOR)
    assert spec.state_dict_adapter is None
