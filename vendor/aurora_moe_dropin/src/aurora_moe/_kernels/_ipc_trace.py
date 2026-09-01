"""Opt-in host-side traces for persistent Level Zero IPC diagnostics."""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Mapping


_TRACE_ENV = "AURORA_MOE_L0_IPC_TRACE"
_TRACE_DIR_ENV = "AURORA_MOE_TRACE_DIR"


def ipc_trace_enabled() -> bool:
    """Return whether per-rank IPC control-flow tracing is enabled."""

    value = os.environ.get(_TRACE_ENV, "0")
    if value in {"", "0"}:
        return False
    if value != "1":
        raise ValueError(f"{_TRACE_ENV} must be 0 or 1")
    return True


def _global_rank() -> int | None:
    values: list[int] = []
    for name in ("RANK", "PALS_RANKID"):
        value = os.environ.get(name)
        if value is not None:
            try:
                values.append(int(value))
            except ValueError as exc:
                raise ValueError(f"{name} must be an integer") from exc
    if values and any(value != values[0] for value in values[1:]):
        raise RuntimeError("RANK and PALS_RANKID disagree")
    return values[0] if values else None


def emit_ipc_trace(
    rank: int,
    component: str,
    operation: str,
    state: str,
    fields: Mapping[str, object] | None = None,
) -> None:
    """Print one flushed, non-collective trace record for a local IPC phase."""

    parts = [
        "L0_IPC_TRACE",
        f"time_ns={time.monotonic_ns()}",
        f"pid={os.getpid()}",
        f"rank={rank}",
        f"component={component}",
        f"operation={operation}",
        f"state={state}",
    ]
    if fields:
        parts.extend(f"{name}={value}" for name, value in fields.items())
    record = " ".join(parts)
    trace_dir = os.environ.get(_TRACE_DIR_ENV)
    if trace_dir:
        global_rank = _global_rank()
        file_rank = rank if global_rank is None else global_rank
        root = Path(trace_dir)
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"rank{file_rank:02d}.log"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{time.monotonic():.6f} {record}\n")
        return
    print(record, flush=True)
