"""Runtime-shape BF16 grouped GEMM through SYCL*TLA's generic Xe API."""

from __future__ import annotations

import os
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import torch
from torch.utils import cpp_extension


_MODULE: ModuleType | None = None


def _tla_root() -> Path:
    configured = os.environ.get("AURORA_MOE_SYCL_TLA_ROOT")
    if not configured:
        raise RuntimeError("set AURORA_MOE_SYCL_TLA_ROOT to the SYCL*TLA checkout")
    root = Path(configured).expanduser().resolve()
    if not (root / "include" / "cutlass" / "gemm" / "group_array_problem_shape.hpp").is_file():
        raise RuntimeError(f"SYCL*TLA grouped-GEMM headers were not found under {root}")
    return root


def _enable_pvc_link_flags() -> None:
    flags = getattr(cpp_extension, "_SYCL_DLINK_FLAGS", None)
    if flags is None:
        raise RuntimeError("this PyTorch build does not expose SYCL device linking")
    while "--offload-compress" in flags:
        flags.remove("--offload-compress")
    required = [
        "-fno-sycl-instrument-device-code",
        "-Xspirv-translator",
        "-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate",
        "-Xs",
        '"-options \\"-igc_opts \'VectorAliasBBThreshold=10000\'\\""',
    ]
    for flag in required:
        if flag not in flags:
            flags.append(flag)


def _load_without_sycl2020(*args: object, **kwargs: object) -> ModuleType:
    append_standard = getattr(cpp_extension, "_append_sycl_std_if_no_std_present", None)
    if append_standard is None:
        raise RuntimeError("this PyTorch build does not expose its SYCL standard helper")

    def leave_compiler_default(_: list[str]) -> None:
        return None

    cpp_extension._append_sycl_std_if_no_std_present = leave_compiler_default
    try:
        return cpp_extension.load(*args, **kwargs)
    finally:
        cpp_extension._append_sycl_std_if_no_std_present = append_standard


def load_xetla_grouped_ops(verbose: bool = False) -> ModuleType:
    """Build and return the generic SYCL*TLA grouped-GEMM prototype.

    A multi-rank DP×EP process must not let every rank race the first JIT
    compile.  ``AURORA_MOE_XETLA_GROUPED_OPS_SO`` therefore accepts a shared
    object built once with the canonical extension name
    ``aurora_moe_xetla_grouped_gemm``.  Source JIT remains the development
    default.
    """

    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.xpu.is_available():
        raise RuntimeError("Aurora XPU is required to build/use grouped GEMM")
    prebuilt = os.environ.get("AURORA_MOE_XETLA_GROUPED_OPS_SO")
    if prebuilt:
        spec = spec_from_file_location("aurora_moe_xetla_grouped_gemm", prebuilt)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load prebuilt XeTLA grouped GEMM: {prebuilt}")
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        required = ("grouped_gemm_bf16", "grouped_weight_grad_bf16")
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            raise RuntimeError(
                f"prebuilt XeTLA grouped GEMM is missing required symbols: {missing}"
            )
        _MODULE = module
        return _MODULE
    root = _tla_root()
    source = Path(__file__).with_name("csrc") / "xetla_grouped_gemm.sycl"
    build_dir = os.environ.get("AURORA_MOE_SYCL_BUILD_DIR")
    if build_dir:
        build_dir = str(Path(build_dir) / "xetla_grouped_gemm")
        Path(build_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_XPU_ARCH_LIST", "pvc")
    _enable_pvc_link_flags()
    _MODULE = _load_without_sycl2020(
        name="aurora_moe_xetla_grouped_gemm",
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


def grouped_gemm_bf16(
    activations: torch.Tensor, weights: torch.Tensor, group_offsets: torch.Tensor
) -> torch.Tensor:
    """Return packed expert-group ``activations @ weights`` on XPU.

    ``activations`` is ``[sum(rows), K]``, ``weights`` is ``[groups, K, N]``,
    and the device-resident int32 ``group_offsets`` has shape ``[groups + 1]``.
    This is a raw forward primitive; autograd integration intentionally remains
    separate while its forward correctness and performance are established.
    """

    return load_xetla_grouped_ops().grouped_gemm_bf16(
        activations.contiguous(), weights.contiguous(), group_offsets.contiguous()
    )


def grouped_weight_grad_bf16(
    activations: torch.Tensor, grad_output: torch.Tensor, group_offsets: torch.Tensor
) -> torch.Tensor:
    """Return exact per-expert ``activations.T @ grad_output`` on XPU.

    ``activations`` and ``grad_output`` are both contiguous expert-segmented
    route matrices with a shared leading ``sum(rows)`` dimension.  The generic
    SYCL*TLA grouped scheduler consumes each expert's true reduction extent
    from the int32 device ``group_offsets``; it does not allocate a padded
    ``[experts, max_rows, ...]`` operand.
    """

    return load_xetla_grouped_ops().grouped_weight_grad_bf16(
        activations.contiguous(), grad_output.contiguous(), group_offsets.contiguous()
    )
