"""Build and load the fused BF16 SwiGLU SYCL pointwise extension."""

from __future__ import annotations

import os
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import torch
from torch.utils.cpp_extension import load


_MODULE: ModuleType | None = None


def load_swiglu_ops(verbose: bool = False) -> ModuleType:
    """Return generic XPU BF16 SwiGLU forward/backward pointwise kernels."""

    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.xpu.is_available():
        raise RuntimeError("Aurora XPU is required to build/use the SwiGLU kernels")
    # Use a single canonical image across DP×EP ranks rather than triggering a
    # first-step source build from every process.
    prebuilt = os.environ.get("AURORA_MOE_SWIGLU_OPS_SO")
    if prebuilt:
        spec = spec_from_file_location("aurora_moe_swiglu_ops", prebuilt)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load prebuilt SwiGLU ops: {prebuilt}")
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        required = ("swiglu_forward_bf16", "swiglu_backward_bf16")
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            raise RuntimeError(f"prebuilt SwiGLU ops is missing symbols: {missing}")
        _MODULE = module
        return _MODULE
    source = Path(__file__).with_name("csrc") / "swiglu_ops.sycl"
    build_dir = os.environ.get("AURORA_MOE_SYCL_BUILD_DIR")
    if build_dir:
        build_dir = str(Path(build_dir) / "swiglu_ops")
        Path(build_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_XPU_ARCH_LIST", "pvc")
    _MODULE = load(
        name="aurora_moe_swiglu_ops",
        sources=[str(source)],
        extra_cflags=["-O3", "-DNDEBUG"],
        extra_sycl_cflags=["-fno-sycl-instrument-device-code"],
        with_cuda=False,
        with_sycl=True,
        verbose=verbose,
        build_directory=build_dir,
    )
    return _MODULE
