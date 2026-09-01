"""Direct oneMKL strided-batched BF16 GEMM on PyTorch's current XPU stream."""

from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType

import torch
from torch.utils.cpp_extension import load


_MODULE: ModuleType | None = None


def load_one_mkl_ops(verbose: bool = False) -> ModuleType:
    """Load the experimental direct-oneMKL BMM extension for Aurora."""

    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.xpu.is_available():
        raise RuntimeError("Aurora XPU is required to build/use oneMKL operations")

    root = Path(os.environ.get("MKLROOT", "/opt/aurora/26.26.0/oneapi/mkl/2025.3"))
    include = root / "include"
    library = root / "lib"
    if not (include / "oneapi" / "mkl" / "blas.hpp").is_file():
        raise RuntimeError(f"oneMKL headers were not found under {root}")
    if not (library / "libmkl_sycl_blas.so.5").is_file():
        raise RuntimeError(f"oneMKL SYCL BLAS library was not found under {root}")

    source = Path(__file__).with_name("csrc") / "one_mkl_batched_gemm.sycl"
    build_dir = os.environ.get("AURORA_MOE_SYCL_BUILD_DIR")
    if build_dir:
        build_dir = str(Path(build_dir) / "one_mkl_ops")
        Path(build_dir).mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TORCH_XPU_ARCH_LIST", "pvc")
    _MODULE = load(
        name="aurora_moe_one_mkl_ops",
        sources=[str(source)],
        extra_cflags=["-O3", "-DNDEBUG"],
        extra_sycl_cflags=["-fno-sycl-instrument-device-code"],
        extra_include_paths=[str(include)],
        extra_ldflags=[
            f"-L{library}",
            "-lmkl_sycl_blas",
            f"-Wl,-rpath,{library}",
        ],
        with_cuda=False,
        with_sycl=True,
        verbose=verbose,
        build_directory=build_dir,
    )
    return _MODULE


def _raw_strided_batched_gemm_bf16(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    trans_a: bool = False,
    trans_b: bool = False,
    batched: bool = True,
) -> torch.Tensor:
    """Return ``op(a) @ op(b)`` for contiguous BF16 XPU 3-D tensors.

    The transpose flags describe logical transpose operations over contiguous
    base tensors.  Keeping the bases contiguous avoids materializing every
    transpose in the routed-expert backward path.
    """

    ops = load_one_mkl_ops()
    if batched:
        return ops.strided_batched_gemm_bf16(a, b, trans_a, trans_b)
    return ops.gemm_loop_bf16(a, b, trans_a, trans_b)


def _raw_shared_a_gemm_bf16(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    trans_a: bool = False,
    trans_b: bool = False,
) -> torch.Tensor:
    return load_one_mkl_ops().shared_a_gemm_bf16(a, b, trans_a, trans_b)


class _OneMklBmm(torch.autograd.Function):
    """First-order autograd using direct oneMKL GEMMs in both passes."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        a: torch.Tensor,
        b: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        batched: bool,
    ) -> torch.Tensor:
        ctx.trans_a = trans_a
        ctx.trans_b = trans_b
        ctx.batched = batched
        ctx.save_for_backward(a, b)
        return _raw_strided_batched_gemm_bf16(
            a, b, trans_a=trans_a, trans_b=trans_b, batched=batched
        )

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None, None]:
        a, b = ctx.saved_tensors
        grad_a = grad_b = None
        if ctx.needs_input_grad[0]:
            grad_a_op = _raw_strided_batched_gemm_bf16(
                grad_output.contiguous(), b, trans_b=not ctx.trans_b, batched=ctx.batched
            )
            grad_a = grad_a_op.transpose(-1, -2) if ctx.trans_a else grad_a_op
        if ctx.needs_input_grad[1]:
            grad_b_op = _raw_strided_batched_gemm_bf16(
                a, grad_output.contiguous(), trans_a=not ctx.trans_a, batched=ctx.batched
            )
            grad_b = grad_b_op.transpose(-1, -2) if ctx.trans_b else grad_b_op
        return grad_a, grad_b, None, None, None


def strided_batched_gemm_bf16(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    trans_a: bool = False,
    trans_b: bool = False,
    batched: bool = True,
) -> torch.Tensor:
    """Return differentiable ``op(a) @ op(b)`` using direct oneMKL calls.

    This supports first-order gradients for contiguous BF16 base tensors.  A
    caller that already owns its backward, such as the routed-expert custom
    autograd function, can use the raw extension through ``load_one_mkl_ops``.
    """

    return _OneMklBmm.apply(a, b, trans_a, trans_b, batched)


class _OneMklSharedABmm(torch.autograd.Function):
    """First-order autograd for GEMMs that reuse one activation matrix."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        a: torch.Tensor,
        b: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        batched: bool,
    ) -> torch.Tensor:
        ctx.trans_a = trans_a
        ctx.trans_b = trans_b
        ctx.batched = batched
        ctx.save_for_backward(a, b)
        return _raw_shared_a_gemm_bf16(a, b, trans_a=trans_a, trans_b=trans_b)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None, None]:
        a, b = ctx.saved_tensors
        grad_a = grad_b = None
        if ctx.needs_input_grad[0]:
            grad_a_per_group = _raw_strided_batched_gemm_bf16(
                grad_output.contiguous(),
                b,
                trans_b=not ctx.trans_b,
                batched=ctx.batched,
            )
            grad_a_op = grad_a_per_group.sum(dim=0)
            grad_a = grad_a_op.transpose(-1, -2) if ctx.trans_a else grad_a_op
        if ctx.needs_input_grad[1]:
            grad_b_op = _raw_shared_a_gemm_bf16(
                a, grad_output.contiguous(), trans_a=not ctx.trans_a
            )
            grad_b = grad_b_op.transpose(-1, -2) if ctx.trans_b else grad_b_op
        return grad_a, grad_b, None, None, None


def shared_a_gemm_bf16(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    trans_a: bool = False,
    trans_b: bool = False,
    batched: bool = True,
) -> torch.Tensor:
    """Return one ``op(a) @ op(b[group])`` result per BF16 weight group."""

    return _OneMklSharedABmm.apply(a, b, trans_a, trans_b, batched)
