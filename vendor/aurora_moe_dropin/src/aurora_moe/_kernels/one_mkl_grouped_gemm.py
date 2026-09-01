"""Raw exact ragged oneMKL grouped BF16 GEMM prototype for Aurora."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import torch
from torch.utils.cpp_extension import load


_MODULE: ModuleType | None = None


def load_one_mkl_grouped_gemm_ops(verbose: bool = False) -> ModuleType:
    """Build/load the raw oneMKL grouped-GEMM prototype."""

    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.xpu.is_available():
        raise RuntimeError("Aurora XPU is required to use oneMKL grouped GEMM")
    root = Path(os.environ.get("MKLROOT", "/opt/aurora/26.26.0/oneapi/mkl/2025.3"))
    include = root / "include"
    library = root / "lib"
    if not (include / "oneapi" / "mkl" / "blas.hpp").is_file():
        raise RuntimeError(f"oneMKL headers were not found under {root}")
    if not (library / "libmkl_sycl_blas.so.5").is_file():
        raise RuntimeError(f"oneMKL SYCL BLAS library was not found under {root}")
    build_dir = os.environ.get("AURORA_MOE_SYCL_BUILD_DIR")
    if build_dir:
        build_dir = str(Path(build_dir) / "one_mkl_grouped_gemm")
        Path(build_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_XPU_ARCH_LIST", "pvc")
    _MODULE = load(
        name="aurora_moe_one_mkl_grouped_gemm",
        sources=[str(Path(__file__).with_name("csrc") / "one_mkl_grouped_gemm.sycl")],
        extra_cflags=["-O3", "-DNDEBUG"],
        extra_sycl_cflags=["-fno-sycl-instrument-device-code"],
        extra_include_paths=[str(include)],
        extra_ldflags=[f"-L{library}", "-lmkl_sycl_blas", f"-Wl,-rpath,{library}"],
        with_cuda=False,
        with_sycl=True,
        verbose=verbose,
        build_directory=build_dir,
    )
    return _MODULE


def _host_rows(group_rows: torch.Tensor | Sequence[int]) -> list[int]:
    if isinstance(group_rows, torch.Tensor):
        if group_rows.ndim != 1 or group_rows.dtype not in (torch.int32, torch.int64):
            raise ValueError("group_rows tensor must be a rank-1 int32 or int64 tensor")
        return [int(value) for value in group_rows.cpu().tolist()]
    return [int(value) for value in group_rows]


def grouped_gemm_bf16(
    activations: torch.Tensor,
    weights: torch.Tensor,
    group_rows: torch.Tensor | Sequence[int],
) -> torch.Tensor:
    """Return exact active expert rows for `[E, max_rows, K] @ [E, K, N]`.

    ``group_rows`` supplies the dynamic valid row count of every expert.  It
    is intentionally host metadata because oneMKL's grouped API accepts host
    pointer/extent arrays.  The result is zero in inactive row tails.  This is
    a forward-only probe, not an autograd integration.
    """

    return load_one_mkl_grouped_gemm_ops().grouped_gemm_bf16(
        activations, weights, _host_rows(group_rows)
    )


def reap_completed_grouped_gemm_launches() -> int:
    """Release metadata retained for completed asynchronous grouped calls."""

    return int(load_one_mkl_grouped_gemm_ops().reap_grouped_gemm_launches())
