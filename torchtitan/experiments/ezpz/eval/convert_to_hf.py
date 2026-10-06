# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from scripts/checkpoint_conversion/convert_to_hf.py
# Modified to support experiment model paths (torchtitan.experiments.*)

import argparse
import importlib
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import HuggingFaceStorageWriter

from torchtitan.components.checkpointer import ModelWrapper
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.experiments.ezpz.eval.hf_export import (
    staged_export,
    validate_export,
    write_export_manifest,
)


def _checkpoint_load_state_dict(
    state_dict: dict[str, Any],
    sd_adapter,
    checkpoint_keys: set[str],
) -> dict[str, Any]:
    """Match today's native model state to the schema stored in a DCP.

    Older TorchTitan checkpoints expose fused QKV and gate/up parameters as
    logical ``wq/wk/wv`` and ``w1/w3`` keys through state-dict hooks. Current
    model definitions retain the physical ``wqkv`` and ``w13`` parameters.
    DCP requires the destination keys to match the on-disk schema, so derive
    both representations through the authoritative model adapter and select
    the representation actually present in checkpoint metadata.
    """
    logical_state_dict = sd_adapter._native_fused_linears_to_hf(state_dict)
    load_state_dict: dict[str, Any] = {}
    missing: list[str] = []

    for key, value in state_dict.items():
        if key in checkpoint_keys:
            load_state_dict[key] = value
            continue

        logical_keys = [
            logical_key
            for logical_key in logical_state_dict
            if logical_key in checkpoint_keys
            and (
                logical_key.startswith(key.rsplit(".", 2)[0] + ".")
                if key.endswith(("wqkv.weight", "wqkv.bias", "w13.weight", "w13.bias"))
                else False
            )
        ]
        if logical_keys:
            load_state_dict.update(
                (logical_key, logical_state_dict[logical_key])
                for logical_key in logical_keys
            )
        else:
            missing.append(key)

    if missing:
        preview = ", ".join(missing[:8])
        raise RuntimeError(
            "Checkpoint schema cannot represent current model state keys: "
            f"{preview}{' ...' if len(missing) > 8 else ''}"
        )

    return load_state_dict


def _prepare_hf_state_dict(
    state_dict: dict[str, torch.Tensor],
    target_dtype: torch.dtype,
    dtype_overrides: dict[str, torch.dtype] | None = None,
) -> dict[str, torch.Tensor]:
    """Cast and pack tensors for safetensors serialization.

    State-dict adapters may return inexpensive transpose or slice views.  The
    Hugging Face storage writer ultimately serializes with safetensors, which
    requires every tensor to be contiguous.  Pack after the dtype conversion
    so a lower-precision export does not first duplicate the full checkpoint
    in FP32.
    """
    dtype_overrides = dtype_overrides or {}
    return {
        key: value.to(dtype=dtype_overrides.get(key, target_dtype)).contiguous()
        for key, value in state_dict.items()
    }


@torch.inference_mode()
def _convert_to_hf(
    input_dir,
    output_dir,
    model_name,
    model_flavor,
    hf_assets_path,
    export_dtype,
    hf_config_path=None,
):
    # load model and model args so that we can get the state dict shape
    # Support both core models (torchtitan.models.*) and experiment models
    if "." in model_name:
        model_module = importlib.import_module(f"torchtitan.{model_name}")
    else:
        model_module = importlib.import_module(f"torchtitan.models.{model_name}")
    model_config = model_module.model_registry(model_flavor)

    with torch.device("meta"):
        model_probe = model_config.build()
    adapter_cls = type(model_probe).state_dict_adapter_cls
    del model_probe

    assert adapter_cls is not None, (
        "trying to convert checkpoint from DCP to HF safetensors format, "
        "but the model has no state dict adapter."
    )
    sd_adapter = adapter_cls(model_config, hf_assets_path)
    validate_hf_assets = getattr(sd_adapter, "validate_hf_assets", None)
    if validate_hf_assets is not None:
        validate_hf_assets()

    # RoPE convention is NOT recoverable from the checkpoint: both rope caches
    # are registered with persistent=False, so nothing on disk says which one
    # the run used. The adapter reads it from the flavor passed on the command
    # line, and picking the wrong one applies (or skips) the Q/K permute --
    # which loads without error and only shows up as gibberish at generation
    # time.
    #
    # There is NO safe per-chain default: the chains changed convention
    # MID-FLIGHT when 5ffb850a1 (2026-06-25) flipped CONFIG_SUFFIX to _real and
    # the running chains picked it up on their next resume. Per W&B run
    # metadata (the authoritative record of what each run executed):
    #
    #   20b_v2_256  agpt_20b -> agpt_20b_real  2026-07-10  (loss 2.69 -> 6.14)
    #   20b_v2_512  agpt_20b -> agpt_20b_real  2026-07-05  (loss 2.57 -> 6.03)
    #   2b_v2_512   agpt_2b  -> agpt_2b_real   2026-08-05
    #   2b_v2_256   agpt_2b  throughout
    #
    # So the right flavor depends on WHICH STEP is being converted. Resolve it
    # with scripts/eval/rope_flavor_for_step.py; do not guess, and do not infer
    # from a submit script's default.
    #
    # Announce the convention loudly and write it next to the weights, so a
    # mismatch is visible in the log and auditable afterwards.
    rope_is_cos_sin = getattr(sd_adapter, "_is_cos_sin", None)
    rope_name = {True: "cos_sin", False: "complex", None: "unknown"}[rope_is_cos_sin]
    print(
        f"[convert_to_hf] flavor={model_flavor!r} -> RoPE={rope_name} "
        f"(Q/K permute {'SKIPPED' if rope_is_cos_sin else 'APPLIED'}). "
        "The checkpoint does not record its convention -- if this does not "
        "match how it was TRAINED, the export is silently corrupt. Resolve "
        "with scripts/eval/rope_flavor_for_step.py; see "
        "docs/guides/known-bugs/rope-flavor-mismatch.md"
    )

    metadata = dcp.FileSystemReader(input_dir).read_metadata().state_dict_metadata
    checkpoint_keys = set(metadata)
    checkpoint_state_dict = getattr(sd_adapter, "checkpoint_state_dict", None)
    custom_state_dict = (
        checkpoint_state_dict(metadata) if checkpoint_state_dict is not None else None
    )
    if custom_state_dict is not None:
        state_dict = custom_state_dict
    else:
        # Match the live model representation to the schema in this checkpoint.
        # Historical dense AGPT checkpoints store logical Q/K/V and gate/up
        # tensors, while current models expose physically fused parameters.
        with torch.device("cpu"):
            model = ModelWrapper(model_config.build())
        state_dict = _checkpoint_load_state_dict(
            model._get_state_dict(), sd_adapter, checkpoint_keys
        )
    dcp.load(
        state_dict,
        checkpoint_id=input_dir,
    )

    # convert state dict tt->hf
    hf_state_dict = sd_adapter.to_hf(state_dict)

    storage_writer = HuggingFaceStorageWriter(
        path=output_dir,
        save_distributed=True,
        fqn_to_index_mapping=sd_adapter.fqn_to_index_mapping,
        enable_consolidation=True,
        thread_count_consolidation=5,
    )

    # Map to the export dtype and materialize adapter-produced tensor views.
    # safetensors rejects non-contiguous views (notably transposed MoE experts).
    target_dtype = TORCH_DTYPE_MAP[export_dtype]
    dtype_overrides = getattr(sd_adapter, "hf_dtype_overrides", lambda: {})()
    hf_state_dict = _prepare_hf_state_dict(hf_state_dict, target_dtype, dtype_overrides)

    dcp.save(
        hf_state_dict,
        storage_writer=storage_writer,
    )

    write_hf_assets = getattr(sd_adapter, "write_hf_assets", None)
    if write_hf_assets is not None:
        write_hf_assets(output_dir, export_dtype)
    elif hf_config_path is not None:
        shutil.copy2(hf_config_path, Path(output_dir) / "config.json")
        for name in (
            "tokenizer.model",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ):
            shutil.copy2(Path(hf_assets_path) / name, Path(output_dir) / name)

    # Provenance for the export: which flavor produced it, and therefore which
    # RoPE convention the weights are in. Without this there is no way to audit
    # an existing HF dir after the fact -- the weights look identical either way.
    write_export_manifest(
        output_dir,
        {
            "source_dcp": str(input_dir),
            "model_name": model_name,
            "model_flavor": model_flavor,
            "rope": rope_name,
            "export_dtype": export_dtype,
        },
    )
    if write_hf_assets is not None or hf_config_path is not None:
        validate_export(output_dir)


def convert_to_hf(
    input_dir,
    output_dir,
    model_name,
    model_flavor,
    hf_assets_path,
    export_dtype,
    hf_config_path=None,
):
    with staged_export(output_dir) as staging:
        _convert_to_hf(
            input_dir,
            staging,
            model_name,
            model_flavor,
            hf_assets_path,
            export_dtype,
            hf_config_path,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DCP weights to HF format.")
    parser.add_argument("--hf_config_path", type=Path, default=None)
    parser.add_argument(
        "input_dir", type=Path, help="Input directory with DCP weights."
    )
    parser.add_argument(
        "output_dir", type=Path, help="Output directory for HF checkpoint."
    )
    parser.add_argument(
        "--hf_assets_path",
        type=Path,
        help="Path to HF assets directory.",
        default="./assets/hf/gemma-7b",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        nargs="?",
        default="experiments.ezpz.agpt",
    )
    parser.add_argument(
        # REQUIRED, deliberately no default. This selects the RoPE convention,
        # the chains switched convention mid-flight, and a wrong value produces
        # an export that loads cleanly and generates gibberish. The old default
        # of "2b" was silently wrong for every post-2026-07 20B checkpoint.
        # Resolve with scripts/eval/rope_flavor_for_step.py.
        "--model_flavor",
        type=str,
        required=True,
        help=(
            "Model flavor, e.g. 2b / 2b_real / 20b / 20b_real / 80b. REQUIRED: "
            "this picks the RoPE convention and there is no safe default -- "
            "the chains switched mid-flight. Resolve a specific step with "
            "scripts/eval/rope_flavor_for_step.py --chain <k> --step <N>."
        ),
    )
    parser.add_argument(
        "--export_dtype",
        type=str,
        nargs="?",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
        help="Export dtype for HF checkpoint (default: bfloat16)",
    )
    args = parser.parse_args()

    convert_to_hf(
        args.input_dir,
        args.output_dir,
        args.model_name,
        args.model_flavor,
        args.hf_assets_path,
        args.export_dtype,
        args.hf_config_path,
    )
