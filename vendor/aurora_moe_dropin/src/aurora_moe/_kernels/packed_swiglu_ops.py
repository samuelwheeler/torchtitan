"""Differentiable packed-[up|gate] BF16 SwiGLU kernels for Aurora."""

from __future__ import annotations

import os
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import torch
from torch.utils.cpp_extension import load


_MODULE: ModuleType | None = None


def load_packed_swiglu_ops(verbose: bool = False) -> ModuleType:
    """Build/load the generic packed-projection SwiGLU SYCL extension."""

    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.xpu.is_available():
        raise RuntimeError("Aurora XPU is required to build/use packed SwiGLU operations")
    # A packed up/gate projection is used by the distributed candidate, where
    # source JIT compilation from every DPxEP rank would otherwise dominate
    # the first step.  Match the other Sonic-pipeline extensions and allow a
    # canonical image built once on an allocated tile to be supplied.
    prebuilt = os.environ.get("AURORA_MOE_PACKED_SWIGLU_OPS_SO")
    if prebuilt:
        spec = spec_from_file_location("aurora_moe_packed_swiglu_ops", prebuilt)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load prebuilt packed SwiGLU ops: {prebuilt}")
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        required = ("packed_swiglu_forward_bf16", "packed_swiglu_backward_bf16")
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            raise RuntimeError(f"prebuilt packed SwiGLU ops is missing symbols: {missing}")
        _MODULE = module
        return _MODULE
    source = Path(__file__).with_name("csrc") / "packed_swiglu_ops.sycl"
    build_dir = os.environ.get("AURORA_MOE_SYCL_BUILD_DIR")
    if build_dir:
        build_dir = str(Path(build_dir) / "packed_swiglu_ops")
        Path(build_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_XPU_ARCH_LIST", "pvc")
    _MODULE = load(
        name="aurora_moe_packed_swiglu_ops",
        sources=[str(source)],
        extra_cflags=["-O3", "-DNDEBUG"],
        extra_sycl_cflags=["-fno-sycl-instrument-device-code"],
        with_cuda=False,
        with_sycl=True,
        verbose=verbose,
        build_directory=build_dir,
    )
    return _MODULE


class _PackedSwiGLU(torch.autograd.Function):
    """First-order autograd for a contiguous ``[..., 2H]`` projection."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, projection: torch.Tensor
    ) -> torch.Tensor:
        ctx.save_for_backward(projection)
        return load_packed_swiglu_ops().packed_swiglu_forward_bf16(projection)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_hidden: torch.Tensor
    ) -> tuple[torch.Tensor]:
        (projection,) = ctx.saved_tensors
        return (
            load_packed_swiglu_ops().packed_swiglu_backward_bf16(
                grad_hidden.contiguous(), projection
            ),
        )


def packed_swiglu_bf16(projection: torch.Tensor) -> torch.Tensor:
    """Apply exact BF16 SwiGLU to contiguous packed ``[..., up | gate]`` values."""

    return _PackedSwiGLU.apply(projection)
