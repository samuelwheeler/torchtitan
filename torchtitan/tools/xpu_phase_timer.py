# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import hashlib
import inspect
import json
import math
import os
import statistics
import tempfile
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.distributed._composable.fsdp import FSDPModule
from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState

from torchtitan.config import Configurable


EXPECTED_TORCH_GIT = "808e7f2bb128dc3fd517bd7883cf3a66070a1607"
EXPECTED_FSDP_CALLBACK_SHA256 = (
    "400f2f1f9dc5fe4c615e983a1017a551bec921e8b2f297ed597eb4ecd449a206"
)
POST_BACKWARD_SPAN_SEMANTICS = (
    "Rank-local default-stream time from autograd graph completion, before the "
    "FSDP final callback, until the first gradient consumer is runnable. It "
    "equals exposed FSDP communication drain only for the current non-pipeline "
    "HSDP configuration."
)


@dataclass(slots=True)
class _Sample:
    step: int
    events: dict[str, Any]
    backward_callbacks: int = 0


class XPUPhaseTimer(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        enable: bool = False
        warmup_steps: int = 2
        sample_freq: int = 1
        max_records: int = 128
        save_folder: str = "phase_timing"

        def __post_init__(self) -> None:
            if self.warmup_steps < 0:
                raise ValueError("phase timer warmup_steps must be non-negative")
            if self.sample_freq < 1:
                raise ValueError("phase timer sample_freq must be positive")
            if self.max_records < 1:
                raise ValueError("phase timer max_records must be positive")
            if (
                not self.save_folder
                or self.save_folder in (".", "..")
                or Path(self.save_folder).name != self.save_folder
            ):
                raise ValueError("phase timer save_folder must be one directory name")

    _EVENT_NAMES = (
        "step_start",
        "backward_compute_done",
        "grad_ready",
        "norm_done",
        "optimizer_start",
        "optimizer_done",
        "step_done",
    )

    def __init__(
        self,
        config: Config,
        *,
        device: torch.device,
        event_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        self.device = device
        if config.enable and device.type != "xpu":
            raise ValueError(f"XPU phase timing requires an XPU device, got {device}")
        self._event_factory = event_factory or torch.xpu.Event
        self._open_step: int | None = None
        self._active: _Sample | None = None
        self._pending: deque[_Sample] = deque()
        self.records: list[dict[str, float | int]] = []
        self.samples_skipped_by_limit = 0

    @property
    def enabled(self) -> bool:
        return self.config.enable

    def begin_step(self, step: int) -> None:
        if not self.enabled:
            return
        if self._open_step is not None:
            raise RuntimeError(f"phase timer step {self._open_step} is still open")
        self._open_step = step
        if (
            step <= self.config.warmup_steps
            or (step - self.config.warmup_steps - 1) % self.config.sample_freq
        ):
            return
        if len(self.records) + len(self._pending) >= self.config.max_records:
            self.samples_skipped_by_limit += 1
            return
        events = {
            name: self._event_factory(enable_timing=True) for name in self._EVENT_NAMES
        }
        self._active = _Sample(step, events)
        events["step_start"].record()

    def record_backward_compute_done(self, stream: Any) -> None:
        if self._active is None:
            return
        self._active.events["backward_compute_done"].record(stream)
        self._active.backward_callbacks += 1

    def mark_grad_ready(self) -> None:
        if self._active is not None and self._active.backward_callbacks == 0:
            raise RuntimeError("no FSDP final callback ran before gradient consumption")
        self._record("grad_ready")

    def mark_norm_done(self) -> None:
        self._record("norm_done")

    def mark_optimizer_start(self) -> None:
        self._record("optimizer_start")

    def mark_optimizer_done(self) -> None:
        self._record("optimizer_done")

    def end_step(self) -> None:
        if not self.enabled:
            return
        if self._open_step is None:
            raise RuntimeError("phase timer has no open step")
        if self._active is not None:
            self._active.events["step_done"].record()
            self._pending.append(self._active)
        self._active = None
        self._open_step = None

    def cancel_step(self) -> None:
        self._active = None
        self._open_step = None

    def collect_ready(self) -> list[dict[str, float | int]]:
        ready = []
        while self._pending and self._pending[0].events["step_done"].query():
            sample = self._pending.popleft()
            record = self._elapsed_record(sample)
            self.records.append(record)
            ready.append(record)
        return ready

    def collect_ready_metrics(self) -> dict[str, float]:
        records = self.collect_ready()
        if not records:
            return {}
        names = (
            "step_core_ms",
            "compute_frontier_ms",
            "post_backward_to_grad_ready_ms",
            "grad_norm_ms",
            "pre_optimizer_gap_ms",
            "optimizer_ms",
        )
        return {
            f"phase_timing/rank_local/{name}": statistics.fmean(
                float(record[name]) for record in records
            )
            for name in names
        }

    def finalize(self) -> list[dict[str, float | int]]:
        if not self.enabled:
            return []
        if self._open_step is not None:
            raise RuntimeError(f"phase timer step {self._open_step} is still open")
        if self._pending:
            # Training has ended, so this synchronization is outside every timed span.
            self._pending[-1].events["step_done"].synchronize()
            self.collect_ready()
        if self._pending:
            raise RuntimeError("phase timer terminal event completed out of order")
        return self.records

    def finalize_to_artifact(self, base_folder: str, rank: int) -> Path | None:
        if not self.enabled:
            return None
        records = self.finalize()
        output_dir = Path(base_folder) / self.config.save_folder
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"rank_{rank:06d}.json"
        payload = {
            "schema": 2,
            "rank": rank,
            "semantics": {
                "post_backward_to_grad_ready_ms": POST_BACKWARD_SPAN_SEMANTICS,
            },
            "sampling": {
                "warmup_steps": self.config.warmup_steps,
                "sample_freq": self.config.sample_freq,
                "max_records": self.config.max_records,
                "samples_skipped_by_limit": self.samples_skipped_by_limit,
            },
            "samples": records,
        }
        fd, temporary_path = tempfile.mkstemp(
            prefix=f".{output_path.name}.", suffix=".tmp", dir=output_dir
        )
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(payload, output, sort_keys=True, separators=(",", ":"))
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, output_path)
        return output_path

    def _record(self, name: str) -> None:
        if self._active is not None:
            self._active.events[name].record()

    @staticmethod
    def _elapsed_record(sample: _Sample) -> dict[str, float | int]:
        events = sample.events
        spans = {
            "step_core_ms": ("step_start", "step_done"),
            "compute_frontier_ms": ("step_start", "backward_compute_done"),
            "post_backward_to_grad_ready_ms": (
                "backward_compute_done",
                "grad_ready",
            ),
            "grad_norm_ms": ("grad_ready", "norm_done"),
            "pre_optimizer_gap_ms": ("norm_done", "optimizer_start"),
            "optimizer_ms": ("optimizer_start", "optimizer_done"),
        }
        record: dict[str, float | int] = {
            "step": sample.step,
            "backward_callbacks": sample.backward_callbacks,
        }
        for name, (start, end) in spans.items():
            value = float(events[start].elapsed_time(events[end]))
            if not math.isfinite(value) or value < 0:
                raise RuntimeError(f"invalid phase timing {name}={value}")
            record[name] = value
        return record


def validate_fsdp_phase_timer_contract() -> None:
    git_version = torch.version.git_version
    if git_version != EXPECTED_TORCH_GIT:
        raise RuntimeError(
            "XPU phase timer only supports the audited Torch build: "
            f"expected git {EXPECTED_TORCH_GIT}, got {git_version}"
        )
    callback_source = inspect.getsource(FSDPState._root_post_backward_final_callback)
    callback_sha = hashlib.sha256(callback_source.encode()).hexdigest()
    if callback_sha != EXPECTED_FSDP_CALLBACK_SHA256:
        raise RuntimeError(
            "FSDP final callback changed; refusing to install phase timing hook: "
            f"expected {EXPECTED_FSDP_CALLBACK_SHA256}, got {callback_sha}"
        )


def validate_phase_timer_training_mode(
    *, enabled: bool, native_ddp: bool, lr_finder: bool, seed_checkpoint: bool
) -> None:
    if not enabled:
        return
    if native_ddp:
        raise ValueError(
            "XPU phase timing requires FSDP/HSDP and does not support native DDP"
        )
    if lr_finder:
        raise ValueError("XPU phase timing is not supported by the LR-finder path")
    if seed_checkpoint:
        raise ValueError(
            "XPU phase timing is not supported by the seed-checkpoint-only path"
        )


def _wrap_fsdp_callback(
    state: Any, timer: XPUPhaseTimer, original: Callable
) -> Callable:
    @functools.wraps(original)
    def wrapped() -> None:
        timer.record_backward_compute_done(state._device_handle.current_stream())
        original()

    return torch._dynamo.disable(wrapped)


def install_fsdp_phase_timer_hooks(
    model_parts: Iterable[nn.Module], timer: XPUPhaseTimer
) -> int:
    if not timer.enabled:
        return 0
    validate_fsdp_phase_timer_contract()
    states = {}
    for model_part in model_parts:
        for module in model_part.modules():
            if isinstance(module, FSDPModule):
                state = module._get_fsdp_state()
                states[id(state)] = state
    if not states:
        raise RuntimeError(
            "XPU phase timing requires an FSDP/HSDP model; unsharded and native "
            "DDP models are unsupported"
        )
    expected = FSDPState._root_post_backward_final_callback
    for state in states.values():
        original = state._root_post_backward_final_callback
        if getattr(original, "__func__", None) is not expected:
            raise RuntimeError("FSDP final callback was already replaced")
        state._root_post_backward_final_callback = _wrap_fsdp_callback(
            state, timer, original
        )
    return len(states)
