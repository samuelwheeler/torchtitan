import ezpz
import json
import os
from dataclasses import is_dataclass
from typing import Any, Literal

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.validate import Validator
from torchtitan.config import ActivationCheckpointConfig, CommConfig, TrainingConfig
from torchtitan.config.configs import CompileConfig
from torchtitan.experiments.ezpz._distributed_compat import ezpz_distributed
from torchtitan.experiments.ezpz.blendcorpus.blendcorpus_builder import (
    BlendCorpusDataLoader,
)
from torchtitan.experiments.ezpz.blendcorpus.build_tokenizer import EZPZTokenizer
from torchtitan.experiments.ft.config.job_config import FaultTolerance
from torchtitan.experiments.ezpz.trainer import FaultTolerantTrainer

from . import model_registry

TT_CONFIG_JSON_ENV = "TT_CONFIG_JSON"


def agpt_debugmodel() -> FaultTolerantTrainer.Config:
    return ezpz_agpt_debugmodel()


def agpt_2b() -> FaultTolerantTrainer.Config:
    return ezpz_agpt_2b()


def agpt_2b_hf() -> FaultTolerantTrainer.Config:
    cfg = ezpz_agpt_2b()
    cfg.dataloader.dataset_path = None
    return cfg


def agpt_2b_flex_attn() -> FaultTolerantTrainer.Config:
    return ezpz_agpt_2b_flex_attn()


def agpt_7b() -> FaultTolerantTrainer.Config:
    return ezpz_agpt_7b()


def agpt_7b_hf() -> FaultTolerantTrainer.Config:
    cfg = ezpz_agpt_7b()
    cfg.dataloader.dataset_path = None
    return cfg


def ezpz_agpt_8b() -> FaultTolerantTrainer.Config:
    return _base_config("8B")


def agpt_8b() -> FaultTolerantTrainer.Config:
    return ezpz_agpt_8b()


def agpt(
    flavor: str,
    local_batch_size: int = 1,
    activation_checkpoint_mode: Literal["none", "full"] = "full",
    seq_len: int = 8192,
    dtype: Literal["bfloat16", "float32"] = "bfloat16",
    compile: bool = True,
    fsdp_reshard_after_forward: Literal["default", "always", "never"] = "default",
    tensor_parallel_degree: int = 1,
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
        dataset_path = f"torchtitan/experiments/ezpz/data-lists/{ezpz_distributed.get_machine().lower()}/books.txt"
    cfg.dataloader.dataset_path = dataset_path
    cfg.metrics.log_freq = 1
    cfg.metrics.enable_wandb = True
    if compile:
        cfg.compile = CompileConfig(enable=True)
    cfg.parallelism.fsdp_reshard_after_forward = fsdp_reshard_after_forward
    cfg.parallelism.tensor_parallel_degree = tensor_parallel_degree
    cfg.checkpoint.enable = True
    cfg.checkpoint.interval = checkpoint_interval
    return cfg


def _base_config(flavor: str) -> FaultTolerantTrainer.Config:
    return FaultTolerantTrainer.Config(
        hf_assets_path="./tests/assets/hf/gemma-7b",
        model_spec=model_registry(flavor),
        tokenizer=EZPZTokenizer.Config(backend="hf"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=200,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10000,
        ),
        dataloader=BlendCorpusDataLoader.Config(dataset="c4_test"),
        metrics=MetricsProcessor.Config(log_freq=10),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="full",
        ),
        comm=CommConfig(train_timeout_seconds=100),
        fault_tolerance=FaultTolerance(enable=False),
        validator=Validator.Config(enable=False),
    )


def ezpz_agpt_debugmodel() -> FaultTolerantTrainer.Config:
    return agpt("debugmodel", local_batch_size=2)


def ezpz_agpt_2b() -> FaultTolerantTrainer.Config:
    return agpt("2b", activation_checkpoint_mode="none")


def ezpz_agpt_2b_flex_attn() -> FaultTolerantTrainer.Config:
    return agpt("2b_flex_attn", local_batch_size=2)


def ezpz_agpt_20b_flex_attn() -> FaultTolerantTrainer.Config:
    return agpt("20b_flex_attn")


def agpt_20b_flex_attn() -> FaultTolerantTrainer.Config:
    return agpt("20b_flex_attn")


def ezpz_agpt_7b() -> FaultTolerantTrainer.Config:
    return agpt(
        "7b",
        local_batch_size=2,
        seq_len=4096,
        hf_assets_path="./assets/hf/llama-2-7b-hf",
    )


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


def _config_from_json(flavor: str) -> FaultTolerantTrainer.Config:
    cfg = _base_config(flavor)
    _apply_config_overrides(cfg, _load_json_overrides())
    return cfg


def ezpz_agpt_debugmodel_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json("debugmodel")


def ezpz_agpt_2b_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json("2b")


def ezpz_agpt_7b_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json("7b")


def ezpz_agpt_8b_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json("8B")


def ezpz_agpt_blendcorpus_debugmodel() -> FaultTolerantTrainer.Config:
    cfg = _base_config("debugmodel")
    cfg.dataloader.dataset = "blendcorpus"
    if isinstance(cfg.tokenizer, EZPZTokenizer.Config):
        cfg.tokenizer.backend = "sptoken"
    return cfg


def ezpz_agpt_20b() -> FaultTolerantTrainer.Config:
    return agpt("20b")


def agpt_20b() -> FaultTolerantTrainer.Config:
    return agpt("20b")


def ezpz_agpt_50b() -> FaultTolerantTrainer.Config:
    return agpt("50b")


def agpt_50b() -> FaultTolerantTrainer.Config:
    return agpt("50b")


def ezpz_agpt_80b() -> FaultTolerantTrainer.Config:
    return agpt("80B", tensor_parallel_degree=2)


def agpt_80b() -> FaultTolerantTrainer.Config:
    return agpt("80B", tensor_parallel_degree=2)


def ezpz_agpt_80b_alt() -> FaultTolerantTrainer.Config:
    return agpt("80B_alt", tensor_parallel_degree=2)


def agpt_80b_alt() -> FaultTolerantTrainer.Config:
    return agpt("80B_alt", tensor_parallel_degree=2)


def ezpz_agpt_80b_wide() -> FaultTolerantTrainer.Config:
    return agpt("80B_wide", tensor_parallel_degree=2)


def agpt_80b_wide() -> FaultTolerantTrainer.Config:
    return agpt("80B_wide", tensor_parallel_degree=2)


def ezpz_agpt_80b_deep() -> FaultTolerantTrainer.Config:
    return agpt("80B_deep", tensor_parallel_degree=2)


def agpt_80b_deep() -> FaultTolerantTrainer.Config:
    return agpt("80B_deep", tensor_parallel_degree=2)


def ezpz_agpt_80b_deep_alt() -> FaultTolerantTrainer.Config:
    return agpt("80B_deep_alt", tensor_parallel_degree=2)


def agpt_80b_deep_alt() -> FaultTolerantTrainer.Config:
    return agpt("80B_deep_alt", tensor_parallel_degree=2)


def ezpz_agpt_80b_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json("80B")


def ezpz_agpt_80b_wide_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json("80B_wide")


def ezpz_agpt_80b_deep_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json("80B_deep")
