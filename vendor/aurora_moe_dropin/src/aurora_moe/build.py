"""Prebuild the native-oneCCL and SYCL extensions used by AuroraMoE."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex

import torch


def prebuild(build_dir: Path, *, verbose: bool = False) -> dict[str, Path]:
    """Build the validated runtime extensions once and return their paths."""

    if not torch.xpu.is_available():
        raise RuntimeError("prebuild must run inside an Aurora XPU allocation")
    build_dir = build_dir.expanduser().resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    os.environ["AURORA_MOE_SYCL_BUILD_DIR"] = str(build_dir)

    from ._kernels.ep_local_ops import load_ep_local_ops
    from ._kernels.ep_route_ops import load_ep_route_ops
    from ._kernels.native_ccl_a2a import load_native_ccl_a2a_ops
    from ._kernels.one_mkl_exact_expert_gemm import load_one_mkl_exact_expert_ops
    from ._kernels.segment_expert_reorder import load_segment_expert_reorder_ops
    from ._kernels.swiglu_ops import load_swiglu_ops

    modules = {
        "AURORA_MOE_NATIVE_CCL_A2A_SO": load_native_ccl_a2a_ops(verbose),
        "AURORA_MOE_EP_LOCAL_OPS_SO": load_ep_local_ops(verbose),
        "AURORA_MOE_EP_ROUTE_OPS_SO": load_ep_route_ops(verbose),
        "AURORA_MOE_SEGMENT_EXPERT_REORDER_OPS_SO": load_segment_expert_reorder_ops(verbose),
        "AURORA_MOE_ONE_MKL_EXACT_OPS_SO": load_one_mkl_exact_expert_ops(verbose),
        "AURORA_MOE_SWIGLU_OPS_SO": load_swiglu_ops(verbose),
    }
    result = {name: Path(str(module.__file__)).resolve() for name, module in modules.items()}
    if not all(path.is_file() for path in result.values()):
        raise RuntimeError("one or more extension builds returned no shared object")
    return result


def prebuild_torchtitan_experts(
    build_dir: Path, *, verbose: bool = False
) -> dict[str, Path]:
    """Build only the two local kernels used by the TorchTitan adapter."""

    if not torch.xpu.is_available():
        raise RuntimeError("prebuild must run inside an Aurora XPU allocation")
    build_dir = build_dir.expanduser().resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    os.environ["AURORA_MOE_SYCL_BUILD_DIR"] = str(build_dir)

    from ._kernels.one_mkl_exact_expert_gemm import load_one_mkl_exact_expert_ops
    from ._kernels.swiglu_ops import load_swiglu_ops

    modules = {
        "AURORA_MOE_ONE_MKL_EXACT_OPS_SO": load_one_mkl_exact_expert_ops(verbose),
        "AURORA_MOE_SWIGLU_OPS_SO": load_swiglu_ops(verbose),
    }
    result = {name: Path(str(module.__file__)).resolve() for name, module in modules.items()}
    if not all(path.is_file() for path in result.values()):
        raise RuntimeError("one or more extension builds returned no shared object")
    return result


def prebuild_torchtitan_full_sonic(
    build_dir: Path, *, verbose: bool = False
) -> dict[str, Path]:
    """Build the four kernels used by TorchTitan's full Sonic backend."""

    if not torch.xpu.is_available():
        raise RuntimeError("prebuild must run inside an Aurora XPU allocation")
    build_dir = build_dir.expanduser().resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    os.environ["AURORA_MOE_SYCL_BUILD_DIR"] = str(build_dir)

    from ._kernels.ep_route_ops import load_ep_route_ops
    from ._kernels.one_mkl_exact_expert_gemm import load_one_mkl_exact_expert_ops
    from ._kernels.segment_expert_reorder import load_segment_expert_reorder_ops
    from ._kernels.swiglu_ops import load_swiglu_ops

    modules = {
        "AURORA_MOE_EP_ROUTE_OPS_SO": load_ep_route_ops(verbose),
        "AURORA_MOE_ONE_MKL_EXACT_OPS_SO": load_one_mkl_exact_expert_ops(verbose),
        "AURORA_MOE_SEGMENT_EXPERT_REORDER_OPS_SO": load_segment_expert_reorder_ops(verbose),
        "AURORA_MOE_SWIGLU_OPS_SO": load_swiglu_ops(verbose),
    }
    result = {name: Path(str(module.__file__)).resolve() for name, module in modules.items()}
    if not all(path.is_file() for path in result.values()):
        raise RuntimeError("one or more extension builds returned no shared object")
    return result


def write_environment(build_dir: Path, modules: dict[str, Path], output: Path) -> Path:
    """Write a sourceable shell file for the extensions compiled by ``prebuild``."""

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    exports = {"AURORA_MOE_SYCL_BUILD_DIR": build_dir.expanduser().resolve(), **modules}
    output.write_text(
        "\n".join(f"export {name}={shlex.quote(str(path))}" for name, path in exports.items())
        + "\n"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--torchtitan-experts-only",
        action="store_true",
        help="build only exact expert GEMM and fused SwiGLU kernels",
    )
    parser.add_argument(
        "--torchtitan-full-sonic",
        action="store_true",
        help="build the four kernels used by TorchTitan's full Sonic backend",
    )
    args = parser.parse_args()
    if args.torchtitan_experts_only and args.torchtitan_full_sonic:
        parser.error("choose at most one TorchTitan prebuild mode")
    if args.torchtitan_experts_only:
        build_fn = prebuild_torchtitan_experts
    elif args.torchtitan_full_sonic:
        build_fn = prebuild_torchtitan_full_sonic
    else:
        build_fn = prebuild
    modules = build_fn(args.build_dir, verbose=args.verbose)
    if args.env_file is not None:
        print(f"wrote {write_environment(args.build_dir, modules, args.env_file)}")
        return
    print(f"export AURORA_MOE_SYCL_BUILD_DIR={args.build_dir.expanduser().resolve()}")
    for name, path in modules.items():
        print(f"export {name}={path}")


if __name__ == "__main__":
    main()
