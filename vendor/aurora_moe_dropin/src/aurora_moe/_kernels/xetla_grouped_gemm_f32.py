"""FP32-output diagnostic for the generic SYCL*TLA grouped BF16 GEMM."""

from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType

import torch
from torch.utils import cpp_extension

from aurora_moe._kernels.xetla_grouped_gemm import (
    _enable_pvc_link_flags,
    _load_without_sycl2020,
    _tla_root,
)


_MODULE: ModuleType | None = None
_HOST_METADATA_MODULE: ModuleType | None = None


def load_xetla_grouped_f32_ops(verbose: bool = False) -> ModuleType:
    """Build the isolated FP32-output control without changing the BF16 module."""

    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.xpu.is_available():
        raise RuntimeError("Aurora XPU is required to build/use grouped GEMM")
    root = _tla_root()
    source = Path(__file__).with_name("csrc") / "xetla_grouped_gemm_f32.sycl"
    build_dir = os.environ.get("AURORA_MOE_SYCL_BUILD_DIR")
    if build_dir:
        build_dir = str(Path(build_dir) / "xetla_grouped_gemm_f32")
        Path(build_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_XPU_ARCH_LIST", "pvc")
    _enable_pvc_link_flags()
    _MODULE = _load_without_sycl2020(
        name="aurora_moe_xetla_grouped_gemm_f32",
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


def grouped_gemm_bf16_to_f32(
    activations: torch.Tensor, weights: torch.Tensor, group_offsets: torch.Tensor
) -> torch.Tensor:
    """Run the same compact grouped BF16 mainloop, retaining FP32 output."""

    return load_xetla_grouped_f32_ops().grouped_gemm_bf16_to_f32(
        activations.contiguous(), weights.contiguous(), group_offsets.contiguous()
    )


def load_xetla_grouped_f32_host_metadata_ops(verbose: bool = False) -> ModuleType:
    """Build the upstream-style host-metadata control in an isolated cache.

    This deliberately uses a distinct extension name and build directory from
    the asynchronous device-metadata FP32 control, so testing it cannot reuse
    that module's binary or alter its cache.
    """

    global _HOST_METADATA_MODULE
    if _HOST_METADATA_MODULE is not None:
        return _HOST_METADATA_MODULE
    if not torch.xpu.is_available():
        raise RuntimeError("Aurora XPU is required to build/use grouped GEMM")
    root = _tla_root()
    source = Path(__file__).with_name("csrc") / "xetla_grouped_gemm_f32.sycl"
    build_dir = os.environ.get("AURORA_MOE_SYCL_BUILD_DIR")
    if build_dir:
        build_dir = str(Path(build_dir) / "xetla_grouped_gemm_f32_hostmeta")
        Path(build_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_XPU_ARCH_LIST", "pvc")
    _enable_pvc_link_flags()
    _HOST_METADATA_MODULE = _load_without_sycl2020(
        name="aurora_moe_xetla_grouped_gemm_f32_hostmeta",
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
    return _HOST_METADATA_MODULE


def grouped_gemm_bf16_to_f32_host_metadata(
    activations: torch.Tensor,
    weights: torch.Tensor,
    host_group_offsets: torch.Tensor,
) -> torch.Tensor:
    """Run the synchronous upstream-style grouped descriptor control.

    ``host_group_offsets`` must be a contiguous CPU int32 tensor.  This API is
    diagnostic-only: it uploads host-built metadata and waits for the result.
    """

    if host_group_offsets.device.type != "cpu":
        raise ValueError("host_group_offsets must be a CPU tensor")
    return load_xetla_grouped_f32_host_metadata_ops().grouped_gemm_bf16_to_f32_host_metadata(
        activations.contiguous(), weights.contiguous(), host_group_offsets.contiguous()
    )
