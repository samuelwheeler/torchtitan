# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from dataclasses import is_dataclass
from typing import Any, Literal

import ezpz
import ezpz.distributed

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.quantization.float8 import (
    Float8GroupedExpertsConverter,
    Float8LinearConverter,
)
from torchtitan.config import (
    ActivationCheckpointConfig,
    CommConfig,
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.experiments.ezpz.blendcorpus.blendcorpus_builder import (
    BlendCorpusDataLoader,
)
from torchtitan.experiments.ezpz.blendcorpus.build_tokenizer import EZPZTokenizer
from torchtitan.experiments.ezpz.trainer import FaultTolerantTrainer
from torchtitan.experiments.ft.config.job_config import FaultTolerance

from . import model_registry

TT_CONFIG_JSON_ENV = "TT_CONFIG_JSON"


def _load_json_overrides() -> dict[str, Any]:
    path = os.environ.get(TT_CONFIG_JSON_ENV, "").strip()
    if not path:
        raise ValueError(
            f"{TT_CONFIG_JSON_ENV} must point to a JSON file when using *_from_json configs."
        )

    with open(path, encoding="utf-8") as f:
        overrides = json.load(f)

    if not isinstance(overrides, dict):
        raise ValueError(
            f"Expected top-level JSON object in {path!r}, got {type(overrides).__name__}."
        )

    return overrides


def _apply_config_overrides(
    target: Any,
    overrides: dict[str, Any],
    path: str = "",
) -> None:
    for key, value in overrides.items():
        if not hasattr(target, key):
            raise KeyError(f"Unknown config field {key!r} at path {path or '<root>'}.")

        current_value = getattr(target, key)
        field_path = f"{path}.{key}" if path else key

        if isinstance(value, dict):
            if not is_dataclass(current_value):
                raise TypeError(
                    f"Expected dataclass at {field_path!r} for nested override, "
                    f"got {type(current_value).__name__}."
                )
            _apply_config_overrides(current_value, value, field_path)
            continue

        setattr(target, key, value)


def _config_from_json(base_fn) -> FaultTolerantTrainer.Config:
    cfg = base_fn()
    _apply_config_overrides(cfg, _load_json_overrides())
    return cfg


def _base_config(flavor: str) -> FaultTolerantTrainer.Config:
    return FaultTolerantTrainer.Config(
        hf_assets_path="./assets/hf/gemma-7b",
        model_spec=model_registry(flavor),
        tokenizer=EZPZTokenizer.Config(backend="hf"),
        loss=CrossEntropyLoss.Config(),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=200,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=8192,
            steps=10000,
        ),
        dataloader=BlendCorpusDataLoader.Config(dataset="c4_test"),
        metrics=MetricsProcessor.Config(log_freq=10),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
        comm=CommConfig(train_timeout_seconds=100),
        fault_tolerance=FaultTolerance(enable=False),
    )


def moe(
    flavor: str,
    local_batch_size: int = 1,
    activation_checkpoint_mode: Literal["none", "full", "selective"] = "full",
    seq_len: int = 8192,
    # See agpt/config_registry.py for the full reasoning. bf16 master
    # weights silently freeze RMSNorm.weight at init=1.0 because the
    # per-step update (~lr * exp_avg / sqrt(hessian)) is sub-ulp at
    # bf16 scale 1.0.
    dtype: Literal["bfloat16", "float32"] = "float32",
    compile: bool = True,
    checkpoint_interval: int = 50,
    hf_assets_path: str = "./assets/hf/gemma-7b",
    dataset_path: str | None = None,
) -> FaultTolerantTrainer.Config:
    cfg = _base_config(flavor)
    cfg.hf_assets_path = hf_assets_path
    cfg.debug.print_config = True
    cfg.training.local_batch_size = local_batch_size
    cfg.activation_checkpoint.mode = activation_checkpoint_mode
    cfg.training.seq_len = seq_len
    cfg.training.dtype = dtype
    cfg.dataloader.dataset = "blendcorpus"
    if dataset_path is None:
        dataset_path = f"torchtitan/experiments/ezpz/data-lists/{ezpz.distributed.get_machine().lower()}/books.txt"
    cfg.dataloader.dataset_path = dataset_path
    cfg.metrics.log_freq = 1
    cfg.metrics.enable_wandb = True
    if compile:
        cfg.compile = CompileConfig(enable=True)
    cfg.checkpoint.enable = True
    cfg.checkpoint.interval = checkpoint_interval
    return cfg


def moe_500m() -> FaultTolerantTrainer.Config:
    return moe("500M", local_batch_size=4)


def moe_2b() -> FaultTolerantTrainer.Config:
    return moe("2B", local_batch_size=16)


def moe_4b() -> FaultTolerantTrainer.Config:
    return moe("4B", local_batch_size=16)


def moe_7b() -> FaultTolerantTrainer.Config:
    return moe("7B", local_batch_size=2, activation_checkpoint_mode="none")


def moe_debugmodel() -> FaultTolerantTrainer.Config:
    return moe("debugmodel", local_batch_size=8)


def moe_debugmodel_hf() -> FaultTolerantTrainer.Config:
    cfg = moe("debugmodel", local_batch_size=8)
    cfg.dataloader.dataset_path = None
    return cfg


def moe_debugmodel_ep() -> FaultTolerantTrainer.Config:
    cfg = moe_debugmodel()
    cfg.model_spec = model_registry("debugmodel", moe_comm_backend="standard")
    cfg.parallelism.expert_parallel_degree = 2
    return cfg


def moe_debugmodel_flex_attn() -> FaultTolerantTrainer.Config:
    cfg = moe_debugmodel()
    cfg.model_spec = model_registry("debugmodel_flex_attn")
    return cfg


def moe_debugmodel_flex_attn_hf() -> FaultTolerantTrainer.Config:
    cfg = moe_debugmodel_hf()
    cfg.model_spec = model_registry("debugmodel_flex_attn_hf")
    return cfg


def moe_small() -> FaultTolerantTrainer.Config:
    return moe("small", local_batch_size=8)


def moe_small_hf() -> FaultTolerantTrainer.Config:
    cfg = moe("small", local_batch_size=8)
    cfg.dataloader.dataset_path = None
    return cfg


def moe_16b() -> FaultTolerantTrainer.Config:
    cfg = moe(
        "16B",
        local_batch_size=4,
        hf_assets_path="./assets/hf/deepseek-moe-16b-base",
    )
    cfg.optimizer.lr = 2.2e-4
    cfg.lr_scheduler.decay_type = "cosine"
    cfg.lr_scheduler.min_lr_factor = 0.1
    cfg.lr_scheduler.warmup_steps = 200
    cfg.training.steps = 1000
    cfg.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    cfg.parallelism.expert_parallel_degree = 8
    cfg.compile = CompileConfig(enable=True, components=["loss"])
    return cfg


def moe_671b() -> FaultTolerantTrainer.Config:
    cfg = moe(
        "671B",
        local_batch_size=4,
        hf_assets_path="./assets/hf/DeepSeek-V3.1-Base",
    )
    cfg.optimizer.lr = 2.2e-4
    cfg.lr_scheduler.warmup_steps = 2000
    cfg.lr_scheduler.decay_type = "cosine"
    cfg.lr_scheduler.min_lr_factor = 0.1
    cfg.training.steps = 10000
    cfg.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    cfg.checkpoint.interval = 500
    cfg.compile = CompileConfig(enable=True, components=["loss"])
    # Quantization is now applied to the config at model_registry time
    # rather than to the runtime model (#3127). Re-register with Float8.
    cfg.model_spec = model_registry(
        "671B",
        quantization=[
            Float8LinearConverter.Config(filter_fqns=["output", "router.gate"]),
            Float8GroupedExpertsConverter.Config(),
        ],
    )
    return cfg


def moe_7b_ep() -> FaultTolerantTrainer.Config:
    cfg = moe_7b()
    cfg.model_spec = model_registry("7B", moe_comm_backend="standard")
    cfg.parallelism.expert_parallel_degree = 2
    return cfg


def moe_10b_2b_sdpa_ep() -> FaultTolerantTrainer.Config:
    cfg = moe_10b_2b_sdpa()
    cfg.model_spec = model_registry("10B_2B_sdpa", moe_comm_backend="standard")
    cfg.parallelism.expert_parallel_degree = 2
    return cfg


def moe_2b_ep() -> FaultTolerantTrainer.Config:
    cfg = moe("2B", local_batch_size=16)
    cfg.model_spec = model_registry("2B", moe_comm_backend="standard")
    cfg.parallelism.expert_parallel_degree = 2
    return cfg


def moe_10b_2b() -> FaultTolerantTrainer.Config:
    cfg = moe("10B_2B", local_batch_size=1)
    cfg.optimizer.lr = 2.2e-4
    cfg.lr_scheduler.decay_type = "cosine"
    cfg.lr_scheduler.min_lr_factor = 0.1
    cfg.training.steps = 1000
    cfg.checkpoint.interval = 100
    return cfg


def moe_10b_2b_sdpa() -> FaultTolerantTrainer.Config:
    cfg = moe("10B_2B_sdpa", local_batch_size=2,
              activation_checkpoint_mode="none")
    cfg.optimizer.lr = 2.2e-4
    cfg.lr_scheduler.decay_type = "cosine"
    cfg.lr_scheduler.min_lr_factor = 0.1
    cfg.training.steps = 1000
    cfg.checkpoint.interval = 100
    return cfg


def _moe_10b_2b_50k(flavor: str) -> FaultTolerantTrainer.Config:
    cfg = moe(
        flavor,
        local_batch_size=1,
        activation_checkpoint_mode="none",
        seq_len=2048,
        compile=False,
        checkpoint_interval=100,
        hf_assets_path="./assets/hf/llama-2-32k-sp",
    )
    cfg.tokenizer.backend = "sptoken"
    cfg.optimizer.lr = 2.2e-4
    cfg.lr_scheduler.decay_type = "cosine"
    cfg.lr_scheduler.min_lr_factor = 0.1
    cfg.training.steps = 1100
    return cfg


def moe_10b_2b_50k_sdpa_for_loop() -> FaultTolerantTrainer.Config:
    return _moe_10b_2b_50k("10B_2B_50K_sdpa_for_loop")


def moe_10b_2b_50k_sdpa_aurora_sycl() -> FaultTolerantTrainer.Config:
    return _moe_10b_2b_50k("10B_2B_50K_sdpa_aurora_sycl")


def moe_10b_2b_50k_sdpa_aurora_full_loop() -> FaultTolerantTrainer.Config:
    return _moe_10b_2b_50k("10B_2B_50K_sdpa_aurora_full_loop")


def moe_10b_2b_50k_sdpa_aurora_full_sonic() -> FaultTolerantTrainer.Config:
    return _moe_10b_2b_50k("10B_2B_50K_sdpa_aurora_full_sonic")


def _agpt_2b_50k_moe(flavor: str) -> FaultTolerantTrainer.Config:
    cfg = _moe_10b_2b_50k(flavor)
    cfg.hf_assets_path = "./assets/hf/olmo-7b-0724-hf"
    cfg.tokenizer.backend = "hf"
    cfg.activation_checkpoint.mode = "selective"
    return cfg


def agpt_2b_50k_moe_sdpa() -> FaultTolerantTrainer.Config:
    return _agpt_2b_50k_moe("AGPT_2B_50K_MOE_sdpa")


def agpt_2b_50k_moe_sdpa_aurora_full_loop() -> FaultTolerantTrainer.Config:
    return _agpt_2b_50k_moe("AGPT_2B_50K_MOE_sdpa_aurora_full_loop")


def agpt_2b_50k_moe_sdpa_aurora_full_sonic() -> FaultTolerantTrainer.Config:
    return _agpt_2b_50k_moe("AGPT_2B_50K_MOE_sdpa_aurora_full_sonic")


def moe_10b_2b_50k_sdpa_aurora_full_loop_1layer() -> FaultTolerantTrainer.Config:
    return _moe_10b_2b_50k("10B_2B_50K_sdpa_aurora_full_loop_1layer")


def moe_10b_2b_50k_sdpa_aurora_full_sonic_1layer() -> FaultTolerantTrainer.Config:
    return _moe_10b_2b_50k("10B_2B_50K_sdpa_aurora_full_sonic_1layer")


def moe_10b_2b_sdpa_1layer() -> FaultTolerantTrainer.Config:
    cfg = moe("10B_2B_sdpa_1layer", local_batch_size=2,
              activation_checkpoint_mode="none")
    cfg.optimizer.lr = 2.2e-4
    cfg.lr_scheduler.decay_type = "cosine"
    cfg.lr_scheduler.min_lr_factor = 0.1
    cfg.training.steps = 1000
    cfg.checkpoint.interval = 100
    return cfg


def moe_10b_2b_sdpa_batched_mm_padded() -> FaultTolerantTrainer.Config:
    cfg = moe("10B_2B_sdpa_batched_mm_padded", local_batch_size=2,
              activation_checkpoint_mode="none")
    cfg.optimizer.lr = 2.2e-4
    cfg.lr_scheduler.decay_type = "cosine"
    cfg.lr_scheduler.min_lr_factor = 0.1
    cfg.training.steps = 1000
    cfg.checkpoint.interval = 100
    return cfg


def moe_10b_2b_sdpa_scattermoe() -> FaultTolerantTrainer.Config:
    cfg = moe("10B_2B_sdpa_scattermoe", local_batch_size=2,
              activation_checkpoint_mode="none")
    cfg.optimizer.lr = 2.2e-4
    cfg.lr_scheduler.decay_type = "cosine"
    cfg.lr_scheduler.min_lr_factor = 0.1
    cfg.training.steps = 1000
    cfg.checkpoint.interval = 100
    return cfg


def smoke_moe_500m_50steps() -> FaultTolerantTrainer.Config:
    """50-step moe smoke test for the post-#2963/#2937 replay.

    Smallest non-debug moe flavor (500M) on 2 nodes, AdamW, no checkpoint.
    Verifies imports, model build, sharding-config population on MLA
    attention + dense FFN, Module.parallelize, apply_moe_ep_tp,
    per-block compile, FSDP wrap, optimizer step, loss decreasing.
    Uses fineweb-edu HF stream so no local data is required.
    """
    cfg = moe(
        "500M",
        local_batch_size=2,
        activation_checkpoint_mode="none",
        seq_len=8192,
        compile=True,
        checkpoint_interval=10_000,
    )
    cfg.dataloader.dataset = "HuggingFaceFW/fineweb-edu"
    cfg.dataloader.dataset_path = None
    cfg.training.steps = 50
    cfg.checkpoint.enable = False
    cfg.optimizer.lr = 8e-4
    cfg.lr_scheduler.warmup_steps = 5
    cfg.lr_scheduler.decay_ratio = 0.0
    cfg.metrics.log_freq = 1
    return cfg


def moe_debugmodel_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json(moe_debugmodel)


def moe_small_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json(moe_small)


def moe_16b_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json(moe_16b)


def moe_10b_2b_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json(moe_10b_2b)


def moe_10b_2b_sdpa_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json(moe_10b_2b_sdpa)


def moe_10b_2b_50k_sdpa_for_loop_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json(moe_10b_2b_50k_sdpa_for_loop)


def moe_10b_2b_50k_sdpa_aurora_sycl_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json(moe_10b_2b_50k_sdpa_aurora_sycl)


def moe_10b_2b_50k_sdpa_aurora_full_loop_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json(moe_10b_2b_50k_sdpa_aurora_full_loop)


def moe_10b_2b_50k_sdpa_aurora_full_sonic_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json(moe_10b_2b_50k_sdpa_aurora_full_sonic)


def agpt_2b_50k_moe_sdpa_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json(agpt_2b_50k_moe_sdpa)


def agpt_2b_50k_moe_sdpa_aurora_full_loop_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json(agpt_2b_50k_moe_sdpa_aurora_full_loop)


def agpt_2b_50k_moe_sdpa_aurora_full_sonic_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json(agpt_2b_50k_moe_sdpa_aurora_full_sonic)


def moe_10b_2b_50k_sdpa_aurora_full_loop_1layer_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json(moe_10b_2b_50k_sdpa_aurora_full_loop_1layer)


def moe_10b_2b_50k_sdpa_aurora_full_sonic_1layer_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json(moe_10b_2b_50k_sdpa_aurora_full_sonic_1layer)


def moe_10b_2b_sdpa_1layer_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json(moe_10b_2b_sdpa_1layer)


def moe_10b_2b_sdpa_batched_mm_padded_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json(moe_10b_2b_sdpa_batched_mm_padded)


def moe_10b_2b_sdpa_scattermoe_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json(moe_10b_2b_sdpa_scattermoe)


def moe_671b_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json(moe_671b)
