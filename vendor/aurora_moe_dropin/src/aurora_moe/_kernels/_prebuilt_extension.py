"""Opt-in direct imports for prebuilt isolated IPC extensions."""

from __future__ import annotations

import ctypes
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Iterable


_PREBUILT_ENV = "AURORA_MOE_L0_IPC_PREBUILT"
_BUILD_ENV = "AURORA_MOE_SYCL_BUILD_DIR"
# IPC transport artifacts can intentionally come from a different build image
# than the GEMM/reorder artifacts.  The two-node launcher sources the worker
# environment after an experiment's environment, which otherwise overwrites
# the generic SYCL build root and makes a valid raw-IPC image undiscoverable.
_IPC_BUILD_ENV = "AURORA_MOE_L0_IPC_BUILD_DIR"


def _prebuilt_requested() -> bool:
    value = os.environ.get(_PREBUILT_ENV, "0")
    if value in {"", "0"}:
        return False
    if value != "1":
        raise ValueError(f"{_PREBUILT_ENV} must be 0 or 1")
    return True


def _validate_module(module: ModuleType, path: Path, required: Iterable[str]) -> None:
    if module.__name__ != path.stem:
        raise RuntimeError(f"prebuilt extension name mismatch: {module.__name__} != {path.stem}")
    module_path = Path(str(getattr(module, "__file__", ""))).resolve()
    if module_path != path:
        raise RuntimeError(f"prebuilt extension path mismatch: {module_path} != {path}")
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(f"prebuilt extension {path} is missing symbols: {missing}")


def load_prebuilt_ipc_extension(
    *,
    component: str,
    module_name: str,
    required: Iterable[str],
) -> ModuleType | None:
    """Directly import a validated cached IPC extension when explicitly enabled."""

    if not _prebuilt_requested():
        return None
    build_root = os.environ.get(_IPC_BUILD_ENV) or os.environ.get(_BUILD_ENV)
    if not build_root:
        raise RuntimeError(
            f"{_PREBUILT_ENV}=1 requires {_IPC_BUILD_ENV} or {_BUILD_ENV}"
        )
    component_dir = (Path(build_root) / component).resolve()
    path = (component_dir / f"{module_name}.so").resolve()
    if path.parent != component_dir or not path.is_file():
        raise RuntimeError(f"prebuilt IPC extension is unavailable: {path}")

    existing = sys.modules.get(module_name)
    if existing is not None:
        _validate_module(existing, path, required)
        return existing

    init_symbol = f"PyInit_{module_name}"
    mode = getattr(os, "RTLD_NOW", 0) | getattr(os, "RTLD_LOCAL", 0)
    try:
        library = ctypes.CDLL(str(path), mode=mode)
        getattr(library, init_symbol)
    except (OSError, AttributeError) as exc:
        raise RuntimeError(f"prebuilt IPC extension ABI validation failed for {path}: {init_symbol}") from exc

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or not isinstance(spec.loader, importlib.machinery.ExtensionFileLoader):
        raise RuntimeError(f"could not create CPython extension loader for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if sys.modules.get(module_name) is module:
            sys.modules.pop(module_name)
        raise
    _validate_module(module, path, required)
    return module
