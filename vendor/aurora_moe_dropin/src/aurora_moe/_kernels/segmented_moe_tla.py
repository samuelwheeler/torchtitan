"""Specialized SYCL*TLA GEMM for compact source/expert MoE fragments."""

from __future__ import annotations

import os
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import torch

from aurora_moe._kernels.xetla_grouped_gemm import (
    _enable_pvc_link_flags,
    _load_without_sycl2020,
    _tla_root,
)


_MODULE: ModuleType | None = None


def load_segmented_moe_tla_ops(verbose: bool = False) -> ModuleType:
    """Build/load the physical-fragment PVC MoE scheduler specialization."""

    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.xpu.is_available():
        raise RuntimeError("Aurora XPU is required to use the segmented MoE TLA kernel")
    prebuilt = os.environ.get("AURORA_MOE_SEGMENTED_MOE_TLA_OPS_SO")
    if prebuilt:
        spec = spec_from_file_location("aurora_moe_segmented_moe_tla", prebuilt)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load prebuilt segmented MoE TLA kernel: {prebuilt}")
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "segmented_moe_grouped_gemm_bf16"):
            raise RuntimeError("prebuilt segmented MoE TLA kernel is missing its GEMM entry point")
        _MODULE = module
        return _MODULE

    root = _tla_root()
    example = root / "examples" / "12_xe20_moe_gemm_cute_interface"
    if not (example / "moe_tile_scheduler.hpp").is_file():
        raise RuntimeError(f"SYCL*TLA PVC MoE example is missing under {example}")
    source = Path(__file__).with_name("csrc") / "segmented_moe_tla.sycl"
    build_dir = os.environ.get("AURORA_MOE_SYCL_BUILD_DIR")
    if build_dir:
        build_dir = str(Path(build_dir) / "segmented_moe_tla")
        Path(build_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_XPU_ARCH_LIST", "pvc")
    _enable_pvc_link_flags()
    _MODULE = _load_without_sycl2020(
        name="aurora_moe_segmented_moe_tla",
        sources=[str(source)],
        extra_cflags=[
            "-O3",
            "-DNDEBUG",
            "-DCUTLASS_ENABLE_SYCL",
            "-DSYCL_INTEL_TARGET=1",
            "-DAURORA_MOE_TLA_IGC_VECTOR_ALIAS=1",
            "-Wno-pass-failed",
            "-Wno-deprecated-declarations",
        ],
        extra_sycl_cflags=["-fno-sycl-instrument-device-code", "-fsycl-targets=spir64_gen"],
        extra_include_paths=[
            str(Path(__file__).with_name("csrc")),
            str(root / "include"),
            str(root / "tools" / "util" / "include"),
            str(example),
        ],
        with_cuda=False,
        with_sycl=True,
        verbose=verbose,
        build_directory=build_dir,
    )
    return _MODULE


def segmented_moe_grouped_gemm_bf16(
    activations: torch.Tensor,
    weights: torch.Tensor,
    source_expert_counts: torch.Tensor,
) -> torch.Tensor:
    """Apply local expert weights to compact source/expert physical fragments."""

    return load_segmented_moe_tla_ops().segmented_moe_grouped_gemm_bf16(
        activations.contiguous(), weights.contiguous(), source_expert_counts.contiguous()
    )
