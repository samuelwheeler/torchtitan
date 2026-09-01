"""Differentiable runtime-shape BF16 GEMM using PVC XMX joint_matrix."""

from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType

import torch
from torch.utils.cpp_extension import load


_MODULE: ModuleType | None = None


def load_xmx_gemm_ops(verbose: bool = False) -> ModuleType:
    """Build/load the PVC XMX BF16 GEMM extension."""

    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.xpu.is_available():
        raise RuntimeError("Aurora XPU is required to build/use XMX GEMM")
    source = Path(__file__).with_name("csrc") / "xmx_gemm.sycl"
    build_dir = os.environ.get("AURORA_MOE_SYCL_BUILD_DIR")
    if build_dir:
        build_dir = str(Path(build_dir) / "xmx_gemm")
        Path(build_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_XPU_ARCH_LIST", "pvc")
    _MODULE = load(
        name="aurora_moe_xmx_gemm",
        sources=[str(source)],
        extra_cflags=["-O3", "-DNDEBUG"],
        extra_sycl_cflags=[
            "-fno-sycl-instrument-device-code",
            "-fsycl-targets=spir64_gen",
        ],
        with_cuda=False,
        with_sycl=True,
        verbose=verbose,
        build_directory=build_dir,
    )
    return _MODULE


class _XmxBf16Gemm(torch.autograd.Function):
    """First-order autograd built from the same runtime-shape XMX primitive."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, a: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        a = a.contiguous()
        b = b.contiguous()
        ctx.save_for_backward(a, b)
        return load_xmx_gemm_ops().xmx_gemm_bf16(a, b)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        a, b = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        ops = load_xmx_gemm_ops()
        grad_a = grad_b = None
        if ctx.needs_input_grad[0]:
            grad_a = ops.xmx_gemm_bf16(grad_output, b.transpose(0, 1).contiguous())
        if ctx.needs_input_grad[1]:
            grad_b = ops.xmx_gemm_bf16(a.transpose(0, 1).contiguous(), grad_output)
        return grad_a, grad_b


def xmx_gemm_bf16(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return differentiable BF16 ``a @ b`` for runtime 2-D XPU shapes.

    The implementation pads only the internal DPAS launch to 8x16x16 tiles,
    then trims the result.  It does not encode model, expert, or batch sizes.
    """

    return _XmxBf16Gemm.apply(a, b)
