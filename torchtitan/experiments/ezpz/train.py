# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import datetime
import json
import os
import sys

from pathlib import Path

from typing import Any

import ezpz
import ezpz.utils
import torch
import torch.distributed
from torch.distributed import get_rank, get_world_size, is_initialized

from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import ConfigManager
from torchtitan.experiments.ezpz._distributed_compat import ezpz_distributed
from torchtitan.experiments.ezpz.logging import init_logger
from torchtitan.experiments.ezpz.optimizer import (
    ADOPTOptimizersContainer,
    MuonClipOptimizersContainer,
    MuonOptimizersContainer,
    SophiaGOptimizersContainer,
)
from torchtitan.tools.logging import logger

DEFAULT_MODULE = "ezpz.agpt"
DEFAULT_CONFIG = "ezpz_agpt_2b"

fp = Path(__file__)
WBPROJ_NAME = f"torchtitan.{fp.parent.stem}.{fp.stem}"
os.environ.setdefault("WANDB_PROJECT", f"{WBPROJ_NAME}")

# try:
#     import intel_extension_for_pytorch as ipex
# except Exception:
#     pass


_LEGACY_KEY_REMAP = {
    "job.dump-folder": "dump-folder",
    "job.print-config": "debug.print-config",
    "job.print-args": "debug.print-config",
    "job.no-print-config": "debug.no-print-config",
    "job.save-config-file": "debug.save-config-file",
    "model.hf-assets-path": "hf-assets-path",
    "model.tokenizer-path": "hf-assets-path",
    "training.dataset": "dataloader.dataset",
    "training.dataset-path": "dataloader.dataset-path",
    "validation.enable": "validator.enable",
    "validation.no-enable": "validator.no-enable",
    "validation.freq": "validator.freq",
    "validation.steps": "validator.steps",
    "validation.dataset": "validator.dataloader.dataset",
    "validation.dataset-path": "validator.dataloader.dataset-path",
}

_FLAVOR_TO_CONFIG = {
    "debug": "ezpz_agpt_debugmodel",
    "debugmodel": "ezpz_agpt_debugmodel",
    "2b": "ezpz_agpt_2b",
    "7b": "ezpz_agpt_7b",
    "8b": "ezpz_agpt_8b",
    "auroragpt-2b": "ezpz_agpt_2b",
    "auroragpt2b": "ezpz_agpt_2b",
    "auroragpt-7b": "ezpz_agpt_7b",
    "auroragpt7b": "ezpz_agpt_7b",
    "llama3-8b": "ezpz_agpt_8b",
}

_OPTIMIZER_CONFIGS: dict[str, type[OptimizersContainer.Config]] = {
    "adamw": OptimizersContainer.Config,
    "adam": OptimizersContainer.Config,
    "adopt": ADOPTOptimizersContainer.Config,
    "sophiag": SophiaGOptimizersContainer.Config,
    "muon": MuonOptimizersContainer.Config,
    "muonclip": MuonClipOptimizersContainer.Config,
}


def _update_env() -> dict:
    now = datetime.datetime.now()
    dstr = now.strftime("%Y-%m-%d-%H%M%S")
    env_dict: dict[str, Any] = {
        f"env.{k}": v
        for k, v in dict(os.environ).items()
        if not k.startswith("_") and "API" not in k and "LS_" not in k
    }
    env_dict |= {
        "created_at": dstr,
        "day": ezpz.utils.get_timestamp("%d"),
        "DIST_INFO": ezpz_distributed.get_dist_info(),
        "ezpz_file": ezpz.__file__,
        "ezpz_version": getattr(ezpz, "__version__", "0.0"),
        "hostname": ezpz_distributed.get_hostname(),
        "month": ezpz.utils.get_timestamp("%m"),
        "machine": ezpz_distributed.get_machine(),
        "pytorch_backend": str(ezpz_distributed.get_torch_backend()).lower(),
        "project": WBPROJ_NAME,
        "torch_version": torch.__version__,
        "torch_file": torch.__file__,
        "world_size": str(ezpz_distributed.get_world_size()),
        "year": ezpz.utils.get_timestamp("%Y"),
        "working_directory": os.getcwd(),
    }
    _ = env_dict.pop("LS_COLORS", None)
    _ = env_dict.pop("PS1", None)
    logger.info(f"Running on {ezpz_distributed.get_machine()=}")
    # logger.info(f"environment={json.dumps(env_dict, indent=4, sort_keys=True)}")

    return env_dict


def _has_flag(args: list[str], name: str) -> bool:
    key = f"--{name}"
    return any(arg == key or arg.startswith(f"{key}=") for arg in args)


def _inject_default_module_and_config(args: list[str]) -> list[str]:
    merged = list(args)
    if not _has_flag(merged, "module"):
        merged = ["--module", DEFAULT_MODULE, *merged]
    if not _has_flag(merged, "config"):
        merged = ["--config", DEFAULT_CONFIG, *merged]
    return merged


def _extract_optimizer_args(
    args: list[str],
) -> tuple[str | None, dict[str, str], list[str]]:
    """Extract ``--optimizer name`` and ``--optimizer.*`` overrides from args.

    Only activates when ``--optimizer`` (bare) is present. If absent,
    all args pass through to tyro unchanged.

    Returns:
        (optimizer_name, overrides_dict, remaining_args)
    """
    # Quick check: is --optimizer present as a standalone flag?
    has_bare_optimizer = False
    for i, arg in enumerate(args):
        if arg == "--optimizer":
            has_bare_optimizer = True
            break
    if not has_bare_optimizer:
        return None, {}, list(args)

    optimizer_name: str | None = None
    overrides: dict[str, str] = {}
    remaining: list[str] = []
    i = 0
    while i < len(args):
        token = args[i]

        # --optimizer <name> (bare, no dot)
        if token == "--optimizer":
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                optimizer_name = args[i + 1].strip().lower()
                i += 2
                continue
            else:
                raise ValueError("--optimizer requires a name (e.g. --optimizer muon)")

        # --optimizer.field value  or  --optimizer.field=value
        if token.startswith("--optimizer."):
            if "=" in token:
                key_part, value = token.split("=", 1)
                field_name = key_part.removeprefix("--optimizer.")
                overrides[field_name] = value
                i += 1
            else:
                field_name = token.removeprefix("--optimizer.")
                if i + 1 < len(args) and not args[i + 1].startswith("--"):
                    overrides[field_name] = args[i + 1]
                    i += 2
                else:
                    # Boolean flag with no value — treat as "true"
                    overrides[field_name] = "true"
                    i += 1
            continue

        remaining.append(token)
        i += 1

    if optimizer_name is None:
        raise ValueError("--optimizer flag found but no name provided")

    if optimizer_name not in _OPTIMIZER_CONFIGS:
        available = ", ".join(sorted(_OPTIMIZER_CONFIGS.keys()))
        raise ValueError(
            f"Unknown optimizer '{optimizer_name}'. Available: {available}"
        )

    return optimizer_name, overrides, remaining


def _build_optimizer_config(
    name: str,
    base: OptimizersContainer.Config,
    overrides: dict[str, str],
) -> OptimizersContainer.Config:
    """Build optimizer Config from name, base config, and CLI overrides."""
    config_cls = _OPTIMIZER_CONFIGS[name]
    base_cls = OptimizersContainer.Config
    kwargs: dict[str, Any] = {}

    # Collect base class field defaults so we can detect subclass overrides
    base_field_defaults: dict[str, Any] = {
        f.name: f.default
        for f in dataclasses.fields(base_cls)
        if f.default is not dataclasses.MISSING
    }

    for field in dataclasses.fields(config_cls):
        if not hasattr(base, field.name):
            continue  # subclass-only field — let its own default apply

        base_default = base_field_defaults.get(field.name, dataclasses.MISSING)
        sub_default = (
            field.default
            if field.default is not dataclasses.MISSING
            else dataclasses.MISSING
        )

        if (
            base_default is not dataclasses.MISSING
            and sub_default is not dataclasses.MISSING
            and base_default != sub_default
        ):
            # Subclass intentionally overrode this default (e.g. name="Muon",
            # beta1=0.95) — keep the subclass value, don't clobber with base
            kwargs[field.name] = sub_default
        else:
            # Shared field with same default — copy from base so config
            # registry values (lr, weight_decay, etc.) propagate
            kwargs[field.name] = getattr(base, field.name)

    # Apply CLI overrides with type coercion
    for raw_key, raw_value in overrides.items():
        field_name = raw_key.replace("-", "_")
        # Find the matching field for type info
        matching = [f for f in dataclasses.fields(config_cls) if f.name == field_name]
        if not matching:
            available = [f.name for f in dataclasses.fields(config_cls)]
            raise ValueError(
                f"Unknown optimizer field '{field_name}' for {name}. "
                f"Available: {available}"
            )
        field = matching[0]
        # Coerce string to field type
        if field.type is bool or field.type == "bool":
            kwargs[field_name] = raw_value.lower() in ("true", "1", "yes")
        elif field.type is int or field.type == "int":
            kwargs[field_name] = int(raw_value)
        elif field.type is float or field.type == "float":
            kwargs[field_name] = float(raw_value)
        else:
            kwargs[field_name] = raw_value

    return config_cls(**kwargs)


def _canonicalize_option(option: str) -> str:
    return option.removeprefix("--").replace("_", "-")


def _config_name_from_flavor(flavor: str) -> str:
    normalized = flavor.strip().lower()
    return _FLAVOR_TO_CONFIG.get(normalized, f"ezpz_agpt_{normalized}")


def _translate_legacy_args(args: list[str]) -> list[str]:
    translated: list[str] = []
    legacy_tokenizer_backend: str | None = None
    i = 0

    while i < len(args):
        token = args[i]
        if not token.startswith("--"):
            translated.append(token)
            i += 1
            continue

        inline_value = "=" in token
        if inline_value:
            option, value = token.split("=", 1)
            consume_next = False
        else:
            option = token
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                value = args[i + 1]
                consume_next = True
            else:
                value = None
                consume_next = False

        key = _canonicalize_option(option)

        if key == "job.config-file":
            raise ValueError(
                "`--job.config-file` is no longer supported for ezpz. "
                "Use `--module ezpz.agpt --config ezpz_agpt_debugmodel` and CLI overrides instead."
            )

        if key in {"experimental.custom-args-module", "experimental.custom-import"}:
            logger.warning("Ignoring deprecated --experimental.* flag for ezpz.")
            i += 2 if consume_next else 1
            continue

        if key == "model.name":
            if value is not None:
                module_name = value
                if value.strip().lower() == "blendcorpus":
                    module_name = DEFAULT_MODULE
                translated.extend(["--module", module_name])
            i += 2 if consume_next else 1
            continue

        if key == "model.flavor":
            if value is not None:
                translated.extend(["--config", _config_name_from_flavor(value)])
            i += 2 if consume_next else 1
            continue

        if key == "model.tokenizer-backend":
            if value is not None:
                legacy_tokenizer_backend = value
            i += 2 if consume_next else 1
            continue

        if key.startswith("blendcorpus."):
            remapped = f"dataloader.{key.removeprefix('blendcorpus.')}"
        else:
            remapped = _LEGACY_KEY_REMAP.get(key, key)

        if value is None:
            translated.append(f"--{remapped}")
        else:
            translated.extend([f"--{remapped}", value])

        i += 2 if consume_next else 1

    if legacy_tokenizer_backend is not None:
        translated.extend(
            ["tokenizer:config", "--tokenizer.backend", legacy_tokenizer_backend]
        )

    return translated


def _ensure_rank_env() -> None:
    os.environ.setdefault("LOCAL_RANK", str(ezpz_distributed.get_local_rank()))
    if is_initialized():
        os.environ.setdefault("RANK", str(get_rank()))
        os.environ.setdefault("WORLD_SIZE", str(get_world_size()))


def main(args: list[str] | None = None) -> None:
    init_logger()

    import torchtitan

    logger.info(
        "torchtitan version: %s (0.0.0 means __version__ is not defined correctly).",
        torchtitan.__version__,
    )

    raw_args = sys.argv[1:] if args is None else args
    parsed_args = _inject_default_module_and_config(_translate_legacy_args(raw_args))

    # Extract --optimizer before tyro sees it (tyro only knows the base Config)
    optimizer_name, optimizer_overrides, parsed_args = _extract_optimizer_args(
        parsed_args
    )

    logger.info(f"\n{json.dumps(parsed_args, indent=4, sort_keys=True)}")
    config_manager = ConfigManager()
    config: Any = config_manager.parse_args(parsed_args)

    # Swap in the correct optimizer Config subclass if --optimizer was specified
    if optimizer_name is not None:
        config.optimizer = _build_optimizer_config(
            optimizer_name,
            config.optimizer,
            optimizer_overrides,
        )
        logger.info(
            "Using optimizer: %s (%s)", optimizer_name, type(config.optimizer).__name__
        )

    trainer = None

    try:
        if config.comm.mode == "local_tensor":
            logger.info("Local tensor mode enabled - skipping training execution")
            return

        trainer = config.build()

        # SophiaG requires a hessian EMA update each step before the param update
        if isinstance(trainer.optimizers, SophiaGOptimizersContainer):
            trainer.optimizers.register_step_pre_hook(
                lambda *_args, **_kwargs: trainer.optimizers.update_hessian()
            )

        if ezpz_distributed.get_rank() == 0 and ezpz_distributed.verify_wandb():
            try:
                run = ezpz_distributed.setup_wandb(
                    project_name=WBPROJ_NAME,
                    settings={"console": "wrap"},
                )
                wbconfig = {}
                wbconfig |= {"env": _update_env()}
                wbconfig |= config.to_dict()
                # wbconfig |= {"config": asdict(config)}
                wbconfig |= {"dist": ezpz_distributed.get_dist_info()}
                if run is not None:
                    run.config.update(wbconfig)
            except Exception as e:
                logger.warning("Unable to update `wandb.run.config`, continuing!")
                if ezpz_distributed.get_rank() == 0:
                    logger.exception(e)

        if config.checkpoint.create_seed_checkpoint:
            assert (
                int(os.environ["WORLD_SIZE"]) == 1
            ), "Must create seed checkpoint using a single device, to disable sharding."
            assert (
                config.checkpoint.enable
            ), "Must enable checkpointing when creating a seed checkpoint."
            trainer.checkpointer.save(curr_step=0, last_step=True)
            logger.info("Created seed checkpoint")
        elif config.lr_finder.enable:
            from torchtitan.experiments.ezpz.lr_finder import run_lr_finder

            run_lr_finder(trainer)
        else:
            trainer.train()
    except Exception:
        if trainer:
            trainer.close()
        raise
    else:
        trainer.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        logger.info("Process group destroyed")


if __name__ == "__main__":
    ezpz_distributed.setup_torch()
    _ensure_rank_env()
    main()
