"""Persistent IPC push-kernel helpers for same-node all-to-all diagnostics."""

from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType

import torch
from torch.utils.cpp_extension import load

from ._prebuilt_extension import load_prebuilt_ipc_extension


_MODULE: ModuleType | None = None


def load_level_zero_ipc_push_ops(verbose: bool = False) -> ModuleType:
    """Build the isolated persistent Level Zero IPC peer-push kernel."""

    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.xpu.is_available():
        raise RuntimeError("Aurora XPU is required for the IPC push-kernel probe")
    _MODULE = load_prebuilt_ipc_extension(
        component="level_zero_ipc_push",
        module_name="aurora_moe_level_zero_ipc_push",
        required=(
            "push_alltoall_bf16_ccl_typed_async",
            "push_alltoall_bf16_ccl_typed_direct_async",
        ),
    )
    if _MODULE is not None:
        return _MODULE
    source = Path(__file__).with_name("csrc") / "level_zero_ipc_push.sycl"
    build_dir = os.environ.get("AURORA_MOE_SYCL_BUILD_DIR")
    if build_dir:
        build_dir = str(Path(build_dir) / "level_zero_ipc_push")
        Path(build_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_XPU_ARCH_LIST", "pvc")
    _MODULE = load(
        name="aurora_moe_level_zero_ipc_push",
        sources=[str(source)],
        extra_cflags=["-O3", "-DNDEBUG"],
        extra_sycl_cflags=["-fno-sycl-instrument-device-code"],
        with_cuda=False,
        with_sycl=True,
        verbose=verbose,
        build_directory=build_dir,
    )
    return _MODULE
