# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from copy import deepcopy
from dataclasses import is_dataclass
from typing import Any

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.quantization.float8 import (
    Float8GroupedMMConverter,
    Float8LinearConverter,
)
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.experiments.ezpz.blendcorpus.blendcorpus_builder import (
    BlendCorpusDataLoader,
)
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.trainer import Trainer

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


def _deepseek_v3_10b_2b_model_spec():
    model_spec = deepcopy(model_registry("16B"))
    model_spec.flavor = "10B_2B_EP12"
    model_cfg = model_spec.model

    model_cfg.layer.attention.qk_nope_head_dim = 64
    model_cfg.layer.attention.qk_rope_head_dim = 64

    assert model_cfg.layer.moe is not None
    model_cfg.layer.n_dense_layers = 1
    model_cfg.layer.moe.num_experts = 36
    model_cfg.layer.moe.top_k = 3
    model_cfg.layer.moe.num_shared_experts = 2
    return model_spec


def deepseek_v3_debugmodel() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("debugmodel"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
            expert_tensor_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
    )


def deepseek_v3_debugmodel_ep() -> Trainer.Config:
    config = deepseek_v3_debugmodel()
    config.model_spec = model_registry("debugmodel", moe_comm_backend="standard")
    return config


def deepseek_v3_debugmodel_flex_attn() -> Trainer.Config:
    config = deepseek_v3_debugmodel()
    config.model_spec = model_registry("debugmodel", attn_backend="flex")
    return config


def deepseek_v3_debugmodel_flex_attn_ep() -> Trainer.Config:
    config = deepseek_v3_debugmodel()
    config.model_spec = model_registry(
        "debugmodel", attn_backend="flex", moe_comm_backend="standard"
    )
    return config


def deepseek_v3_16b() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./assets/hf/deepseek-moe-16b-base",
        model_spec=model_registry(
            "16B", attn_backend="flex", moe_comm_backend="standard"
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        optimizer=OptimizersContainer.Config(lr=2.2e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=1000,
        ),
        parallelism=ParallelismConfig(
            pipeline_parallel_schedule="Interleaved1F1B",
            expert_parallel_degree=8,
            expert_tensor_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(interval=10),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
        compile=CompileConfig(enable=True, components=["loss"]),
    )


def deepseek_v3_671b() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./assets/hf/DeepSeek-V3.1-Base",
        model_spec=model_registry(
            "671B", attn_backend="flex", moe_comm_backend="torchao"
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        optimizer=OptimizersContainer.Config(lr=2.2e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2000,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=10000,
        ),
        parallelism=ParallelismConfig(
            pipeline_parallel_schedule="Interleaved1F1B",
            expert_parallel_degree=1,
            expert_tensor_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
        compile=CompileConfig(enable=True, components=["loss"]),
        model_converters=ModelConvertersContainer.Config(
            converters=[
                Float8LinearConverter.Config(filter_fqns=["output", "router.gate"]),
                Float8GroupedMMConverter.Config(fqns=["experts"]),
            ]
        ),
    )


def deepseek_v3_10b_2b_ep12() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./assets/hf/gemma-7b",
        metrics=MetricsProcessor.Config(log_freq=1, enable_wandb=True),
        model_spec=_deepseek_v3_10b_2b_model_spec(),
        dataloader=BlendCorpusDataLoader.Config(dataset="blendcorpus"),
        optimizer=OptimizersContainer.Config(lr=2.2e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=200,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            global_batch_size=48,
            seq_len=4096,
            steps=1000,
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=2,
            data_parallel_shard_degree=12,
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
            context_parallel_degree=1,
            expert_parallel_degree=12,
            expert_tensor_parallel_degree=1,
            pipeline_parallel_schedule="1F1B",
        ),
        checkpoint=CheckpointManager.Config(
            enable=True,
            folder="checkpoints",
            interval=100,
            last_save_model_only=False,
            load_step=-1,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="op",
        ),
        compile=CompileConfig(enable=False, components=["loss"]),
    )


def deepseek_v3_10b_2b_ep12_from_json() -> Trainer.Config:
    cfg = deepseek_v3_10b_2b_ep12()
    _apply_config_overrides(cfg, _load_json_overrides())
    return cfg
