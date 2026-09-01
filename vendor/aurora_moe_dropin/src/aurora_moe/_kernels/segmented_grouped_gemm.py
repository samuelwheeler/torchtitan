"""SYCL*TLA grouped GEMMs over exact source/expert route segments."""

from __future__ import annotations

import os
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import torch

from aurora_moe._kernels.xetla_grouped_gemm import _enable_pvc_link_flags, _load_without_sycl2020, _tla_root


_MODULE: ModuleType | None = None


def load_segmented_grouped_ops(verbose: bool = False) -> ModuleType:
    """Build or load the exact source-segmented SYCL*TLA extension."""

    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.xpu.is_available():
        raise RuntimeError("Aurora XPU is required to use segmented grouped GEMM")
    prebuilt = os.environ.get("AURORA_MOE_SEGMENTED_GROUPED_OPS_SO")
    if prebuilt:
        spec = spec_from_file_location("aurora_moe_segmented_grouped_gemm", prebuilt)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load prebuilt segmented grouped GEMM: {prebuilt}")
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        required = ("segmented_grouped_gemm_bf16", "segmented_weight_grad_bf16")
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            raise RuntimeError(f"prebuilt segmented grouped GEMM is missing symbols: {missing}")
        _MODULE = module
        return _MODULE

    root = _tla_root()
    source = Path(__file__).with_name("csrc") / "segmented_grouped_gemm.sycl"
    build_dir = os.environ.get("AURORA_MOE_SYCL_BUILD_DIR")
    if build_dir:
        build_dir = str(Path(build_dir) / "segmented_grouped_gemm")
        Path(build_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_XPU_ARCH_LIST", "pvc")
    _enable_pvc_link_flags()
    _MODULE = _load_without_sycl2020(
        name="aurora_moe_segmented_grouped_gemm",
        sources=[str(source)],
        extra_cflags=[
            "-O3",
            "-DNDEBUG",
            "-DCUTLASS_ENABLE_SYCL",
            "-DSYCL_INTEL_TARGET=1",
            "-Wno-pass-failed",
            "-Wno-deprecated-declarations",
        ],
        extra_sycl_cflags=["-fno-sycl-instrument-device-code", "-fsycl-targets=spir64_gen"],
        extra_include_paths=[str(root / "include"), str(root / "tools" / "util" / "include")],
        with_cuda=False,
        with_sycl=True,
        verbose=verbose,
        build_directory=build_dir,
    )
    return _MODULE


def segmented_grouped_gemm_bf16(
    activations: torch.Tensor,
    weights: torch.Tensor,
    source_expert_counts: torch.Tensor,
    source_expert_offsets: torch.Tensor,
    source_offsets: torch.Tensor,
) -> torch.Tensor:
    """Apply each expert to its exact source-local contiguous route segments."""

    return load_segmented_grouped_ops().segmented_grouped_gemm_bf16(
        activations,
        weights,
        source_expert_counts,
        source_expert_offsets,
        source_offsets,
    )


def segmented_weight_grad_bf16(
    activations: torch.Tensor,
    grad_output: torch.Tensor,
    source_expert_counts: torch.Tensor,
    source_expert_offsets: torch.Tensor,
    source_offsets: torch.Tensor,
) -> torch.Tensor:
    """Return exact per-expert ``activations.T @ grad_output`` without row padding."""

    return load_segmented_grouped_ops().segmented_weight_grad_bf16(
        activations,
        grad_output,
        source_expert_counts,
        source_expert_offsets,
        source_offsets,
    )
