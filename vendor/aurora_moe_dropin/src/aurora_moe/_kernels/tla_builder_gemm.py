"""SYCL*TLA CollectiveBuilder BF16 GEMM diagnostic."""

from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType

import torch
from torch.utils import cpp_extension

from aurora_moe._kernels.tla_ops import _enable_tla_spirv_extensions, _load_without_forced_sycl2020


_MODULE: ModuleType | None = None


def _tla_root() -> Path:
    configured = os.environ.get("AURORA_MOE_SYCL_TLA_ROOT")
    if not configured:
        raise RuntimeError("set AURORA_MOE_SYCL_TLA_ROOT to the SYCL*TLA checkout")
    root = Path(configured).expanduser().resolve()
    if not (root / "include" / "cutlass" / "cutlass.h").is_file():
        raise RuntimeError(f"SYCL*TLA headers were not found under {root}")
    return root


def load_tla_builder_gemm_ops(verbose: bool = False) -> ModuleType:
    """Build and return the direct CollectiveBuilder diagnostic extension."""

    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.xpu.is_available():
        raise RuntimeError("Aurora XPU is required to build/use this extension")
    root = _tla_root()
    source = Path(__file__).with_name("csrc") / "tla_builder_gemm.sycl"
    build_dir = os.environ.get("AURORA_MOE_SYCL_BUILD_DIR")
    if build_dir:
        build_dir = str(Path(build_dir) / "tla_builder_gemm")
        Path(build_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_XPU_ARCH_LIST", "pvc")
    _enable_tla_spirv_extensions()
    _MODULE = _load_without_forced_sycl2020(
        name="aurora_moe_tla_builder_gemm",
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


def builder_gemm_bf16(a: torch.Tensor, b: torch.Tensor, *, zero_c: bool = False) -> torch.Tensor:
    """Return ``a @ b`` through the direct SYCL*TLA CollectiveBuilder path."""

    return load_tla_builder_gemm_ops().builder_gemm_bf16(
        a.contiguous(), b.contiguous(), zero_c
    )
