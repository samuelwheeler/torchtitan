"""Build and load the small SYCL routing-kernel extension.

The extension is compiled only when explicitly requested.  Keeping the build
lazy lets the normal PyTorch reference implementation run on systems without
an Intel GPU or an Intel SYCL compiler.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType

import torch
from torch.utils.cpp_extension import load


_MODULE: ModuleType | None = None


def load_route_ops(verbose: bool = False) -> ModuleType:
    """Return the BF16 XPU route-pack/reduce extension.

    ``AURORA_MOE_SYCL_BUILD_DIR`` may be set to put build products outside the
    repository.  Aurora's ``frameworks`` module supplies ``icpx`` and the
    matching PyTorch XPU libraries.
    """

    global _MODULE
    if _MODULE is not None:
        return _MODULE

    if not torch.xpu.is_available():
        raise RuntimeError("Aurora XPU is required to build/use the SYCL route kernels")

    source = Path(__file__).with_name("csrc") / "route_ops.sycl"
    build_dir = os.environ.get("AURORA_MOE_SYCL_BUILD_DIR")
    if build_dir:
        build_dir = str(Path(build_dir) / "route_ops")
        Path(build_dir).mkdir(parents=True, exist_ok=True)

    # Use PyTorch's native SYCL-extension path.  Setting its XPU architecture
    # causes cpp_extension to emit a PVC AOT image and to quote the backend
    # ``-device pvc`` argument correctly during SYCL device linking.
    os.environ.setdefault("TORCH_XPU_ARCH_LIST", "pvc")
    common = ["-O3", "-DNDEBUG"]
    sycl = ["-fno-sycl-instrument-device-code"]

    _MODULE = load(
        name="aurora_moe_route_ops",
        sources=[str(source)],
        extra_cflags=common,
        extra_sycl_cflags=sycl,
        with_cuda=False,
        with_sycl=True,
        verbose=verbose,
        build_directory=build_dir,
    )
    return _MODULE
