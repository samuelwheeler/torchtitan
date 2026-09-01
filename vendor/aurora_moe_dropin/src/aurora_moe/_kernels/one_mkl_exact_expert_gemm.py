"""Exact compact expert-major oneMKL BF16 GEMMs for Aurora."""

from __future__ import annotations

import os
from collections.abc import Sequence
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import torch
from torch.utils.cpp_extension import load


_MODULE: ModuleType | None = None


def load_one_mkl_exact_expert_ops(verbose: bool = False) -> ModuleType:
    """Load runtime-shape, no-padding oneMKL expert GEMM operations."""

    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.xpu.is_available():
        raise RuntimeError("Aurora XPU is required to use exact oneMKL expert GEMMs")
    prebuilt = os.environ.get("AURORA_MOE_ONE_MKL_EXACT_OPS_SO")
    if prebuilt:
        spec = spec_from_file_location("aurora_moe_one_mkl_exact_expert", prebuilt)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load prebuilt oneMKL exact-expert ops: {prebuilt}")
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        required = (
            "exact_expert_gemm_bf16",
            "exact_expert_gemm_transposed_weight_bf16",
            "exact_expert_sum_transposed_weight_bf16",
            "exact_expert_weight_grad_bf16",
        )
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            raise RuntimeError(f"prebuilt oneMKL exact-expert ops is missing symbols: {missing}")
        _MODULE = module
        return _MODULE

    root = Path(os.environ.get("MKLROOT", "/opt/aurora/26.26.0/oneapi/mkl/2025.3"))
    include = root / "include"
    library = root / "lib"
    if not (include / "oneapi" / "mkl" / "blas.hpp").is_file():
        raise RuntimeError(f"oneMKL headers were not found under {root}")
    if not (library / "libmkl_sycl_blas.so.5").is_file():
        raise RuntimeError(f"oneMKL SYCL BLAS library was not found under {root}")
    build_dir = os.environ.get("AURORA_MOE_SYCL_BUILD_DIR")
    if build_dir:
        build_dir = str(Path(build_dir) / "one_mkl_exact_expert")
        Path(build_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_XPU_ARCH_LIST", "pvc")
    _MODULE = load(
        name="aurora_moe_one_mkl_exact_expert",
        sources=[str(Path(__file__).with_name("csrc") / "one_mkl_exact_expert_gemm.sycl")],
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


def _rows(expert_rows: torch.Tensor | Sequence[int]) -> list[int]:
    if isinstance(expert_rows, torch.Tensor):
        if expert_rows.ndim != 1 or expert_rows.dtype not in (torch.int32, torch.int64):
            raise ValueError("expert_rows tensor must be a rank-1 int32 or int64 tensor")
        return [int(value) for value in expert_rows.cpu().tolist()]
    return [int(value) for value in expert_rows]


def exact_expert_gemm_bf16(
    activations: torch.Tensor, weights: torch.Tensor, expert_rows: torch.Tensor | Sequence[int]
) -> torch.Tensor:
    """Apply one dense, true-length GEMM to every compact expert interval."""

    return load_one_mkl_exact_expert_ops().exact_expert_gemm_bf16(
        activations, weights, _rows(expert_rows)
    )


def exact_expert_gemm_transposed_weight_bf16(
    activations: torch.Tensor, weights: torch.Tensor, expert_rows: torch.Tensor | Sequence[int]
) -> torch.Tensor:
    """Apply exact GEMMs using each forward weight through oneMKL's transpose flag.

    ``weights`` stays in normal forward layout
    ``[experts, forward_input, forward_output]``; relative to this dX GEMM it
    is ``[experts, output_dim, input_dim]``. This is the dX counterpart to
    :func:`exact_expert_gemm_bf16`; it avoids a D/H-wide transpose copy for
    every expert projection in backward.
    """

    return load_one_mkl_exact_expert_ops().exact_expert_gemm_transposed_weight_bf16(
        activations, weights, _rows(expert_rows)
    )


def exact_expert_sum_transposed_weight_bf16(
    left_activations: torch.Tensor,
    left_weights: torch.Tensor,
    right_activations: torch.Tensor,
    right_weights: torch.Tensor,
    expert_rows: torch.Tensor | Sequence[int],
) -> torch.Tensor:
    """Return two exact dX GEMMs accumulated into one compact BF16 tensor."""

    return load_one_mkl_exact_expert_ops().exact_expert_sum_transposed_weight_bf16(
        left_activations,
        left_weights,
        right_activations,
        right_weights,
        _rows(expert_rows),
    )


def exact_expert_weight_grad_bf16(
    activations: torch.Tensor, gradients: torch.Tensor, expert_rows: torch.Tensor | Sequence[int]
) -> torch.Tensor:
    """Return exact expert-wise ``activations.T @ gradients`` without padding."""

    return load_one_mkl_exact_expert_ops().exact_expert_weight_grad_bf16(
        activations, gradients, _rows(expert_rows)
    )
