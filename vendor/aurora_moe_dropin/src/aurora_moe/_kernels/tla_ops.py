"""Build and load the SYCL*TLA grouped-MoE GEMM extension for PVC."""

from __future__ import annotations

import os
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import torch
from torch.utils import cpp_extension


_MODULE: ModuleType | None = None


def _enable_tla_spirv_extensions() -> None:
    """Add the device-link extensions required by SYCL*TLA's PVC copy/MMA ops.

    PyTorch 2.10 exposes SYCL compile flags but not a public device-link flag
    argument.  Its extension helper keeps this list as a module-level value;
    extend it narrowly rather than reimplementing its XPU build machinery.
    """

    flags = getattr(cpp_extension, "_SYCL_DLINK_FLAGS", None)
    if flags is None:
        raise RuntimeError("this PyTorch build does not expose SYCL device linking")
    # SYCL*TLA's standalone CMake path emits an uncompressed PVC AOT image.
    # Keep this narrowly scoped to the TLA extension.
    while "--offload-compress" in flags:
        flags.remove("--offload-compress")
    required = [
        "-fno-sycl-instrument-device-code",
        "-Xspirv-translator",
        "-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate",
        # This is the upstream PVC MoE example's IGC workaround.  It is a
        # device-link option, not the similarly named runtime environment
        # variable, and materially affects generated block-2D code.
        "-Xs",
        '"-options \\"-igc_opts \'VectorAliasBBThreshold=10000\'\\""',
    ]
    for flag in required:
        if flag not in flags:
            flags.append(flag)


def _load_without_forced_sycl2020(*args: object, **kwargs: object) -> ModuleType:
    """Invoke PyTorch's extension builder without its forced SYCL 2020 mode.

    The current Aurora PyTorch extension helper adds ``-sycl-std=2020`` to
    every SYCL source, while SYCL*TLA's supported PVC CMake build uses the
    compiler default.  Keep this extension aligned with the upstream build
    while its exact supported shape envelope is being established.

    PyTorch exposes no public switch for this, so temporarily suppress the
    helper's private flag insertion while constructing this one extension and
    immediately restore it.  Route kernels and all subsequent extensions keep
    PyTorch's normal behavior.
    """

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


def _tla_root() -> Path:
    configured = os.environ.get("AURORA_MOE_SYCL_TLA_ROOT")
    if not configured:
        raise RuntimeError(
            "set AURORA_MOE_SYCL_TLA_ROOT to a checkout of Intel's SYCL*TLA "
            "repository before loading the grouped GEMM extension"
        )
    root = Path(configured).expanduser().resolve()
    if not (root / "include" / "cutlass" / "cutlass.h").is_file():
        raise RuntimeError(f"SYCL*TLA headers were not found under {root}")
    example = root / "examples" / "12_xe20_moe_gemm_cute_interface"
    if not (example / "moe_grouped_gemm.hpp").is_file():
        raise RuntimeError(
            "the checkout does not contain SYCL*TLA's PVC grouped-MoE example: "
            f"{example}"
        )
    return root


def load_tla_ops(verbose: bool = False) -> ModuleType:
    """Return a runtime-shape BF16 grouped GEMM running on PyTorch's XPU stream.

    The caller supplies packed rows and one row count per expert.  No expert
    count, hidden size, or token count is compiled into the public interface.
    """

    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.xpu.is_available():
        raise RuntimeError("Aurora XPU is required to build/use SYCL*TLA kernels")

    # A validated shared object is mandatory for a multi-rank performance
    # measurement: otherwise all ranks can contend on the first source JIT.
    # The filename must carry the canonical extension name below so CPython's
    # PyInit symbol matches it.
    prebuilt = os.environ.get("AURORA_MOE_TLA_OPS_SO")
    if prebuilt:
        spec = spec_from_file_location("aurora_moe_tla_ops", prebuilt)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load prebuilt TLA ops: {prebuilt}")
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "grouped_gemm_bf16"):
            raise RuntimeError("prebuilt TLA ops is missing grouped_gemm_bf16")
        _MODULE = module
        return _MODULE

    root = _tla_root()
    source = Path(__file__).with_name("csrc") / "tla_grouped_gemm.sycl"
    build_dir = os.environ.get("AURORA_MOE_SYCL_BUILD_DIR")
    if build_dir:
        build_dir = str(Path(build_dir) / "tla_ops")
        Path(build_dir).mkdir(parents=True, exist_ok=True)

    # cpp_extension's native SYCL mode generates the PVC device-link command
    # correctly when this architecture is set.  TLA's headers need these two
    # feature defines in both host and device compilation passes.
    os.environ.setdefault("TORCH_XPU_ARCH_LIST", "pvc")
    _enable_tla_spirv_extensions()
    cflags = [
        "-O3",
        "-DNDEBUG",
        "-DCUTLASS_ENABLE_SYCL",
        "-DSYCL_INTEL_TARGET=1",
        "-DAURORA_MOE_TLA_IGC_VECTOR_ALIAS=1",
        "-Wno-pass-failed",
        "-Wno-deprecated-declarations",
    ]
    _MODULE = _load_without_forced_sycl2020(
        name="aurora_moe_tla_ops",
        sources=[str(source)],
        extra_cflags=cflags,
        # TLA's PVC instructions require an AOT PVC image; unlike a generic
        # route kernel, we intentionally do not emit a fallback spir64 image.
        extra_sycl_cflags=[
            "-fno-sycl-instrument-device-code",
            "-fsycl-targets=spir64_gen",
        ],
        extra_include_paths=[
            str(root / "include"),
            str(root / "tools" / "util" / "include"),
            str(root / "examples" / "12_xe20_moe_gemm_cute_interface"),
        ],
        with_cuda=False,
        with_sycl=True,
        verbose=verbose,
        build_directory=build_dir,
    )
    return _MODULE
