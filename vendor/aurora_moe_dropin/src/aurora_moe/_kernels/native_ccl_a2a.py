"""Experimental direct-oneCCL collectives for an existing XCCL job.

This deliberately does not reach into ProcessGroupXCCL's private communicator.
Instead it creates one additional oneCCL communicator for the supplied process
group, using a one-time KVS-address broadcast.  The hot path calls oneCCL's C++
``alltoall``/``alltoallv``/``allreduce`` APIs on PyTorch's *current* XPU SYCL
queue and retains the returned ``ccl::event`` in a work object.

It is an isolated transport experiment, not a replacement for
``torch.distributed`` yet.  Equal-size all-to-all accepts contiguous BF16 or
int64 tensors; all-to-all-v and SUM all-reduce accept contiguous BF16 tensors.
Every member of the supplied group must make matching calls in the same order.
Set ``AURORA_MOE_NATIVE_CCL_PRIVATE_STREAM=1`` before communicator creation to
use a persistent in-order oneCCL queue in the current PyTorch queue's context;
each collective receives an explicit device-side producer dependency and still
bridges completion back to its PyTorch consumer stream.
"""

from __future__ import annotations

import math
import os
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType
from typing import Sequence

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load


_MODULE: ModuleType | None = None


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "on", "yes"}


def _record(name: str):
    """Create a zero-overhead-when-disabled profiler range for bridge diagnosis."""

    return torch.profiler.record_function(name) if _env_enabled("AURORA_MOE_PROFILE") else nullcontext()


def load_native_ccl_a2a_ops(verbose: bool = False) -> ModuleType:
    """Build/load the direct-oneCCL extension without launching a collective."""

    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.xpu.is_available():
        raise RuntimeError("Aurora XPU is required to use the native oneCCL prototype")

    root = Path(os.environ.get("CCL_ROOT", "/opt/aurora/26.26.0/oneapi/ccl/latest"))
    include = root / "include"
    library = root / "lib"
    if not (include / "oneapi" / "ccl.hpp").is_file():
        raise RuntimeError(f"oneCCL C++ headers were not found under {root}")
    if not (library / "libccl.so").is_file():
        raise RuntimeError(f"oneCCL shared library was not found under {root}")

    source = Path(__file__).with_name("csrc") / "native_ccl_a2a.sycl"
    build_dir = os.environ.get("AURORA_MOE_SYCL_BUILD_DIR")
    if build_dir:
        build_dir = str(Path(build_dir) / "native_ccl_a2a")
        Path(build_dir).mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TORCH_XPU_ARCH_LIST", "pvc")
    _MODULE = load(
        name="aurora_moe_native_ccl_a2a",
        sources=[str(source)],
        # oneCCL 2021.17 selects its SYCL native-device adapter from this
        # preprocessor definition.  PyTorch's JIT uses a separate host phase
        # for a `.sycl` source, where SYCL_LANGUAGE_VERSION is not reliably
        # propagated on every Aurora framework build.  Make the adapter
        # choice explicit for both that host phase and the device phase.
        extra_cflags=["-O3", "-DNDEBUG", "-DCCL_ENABLE_SYCL"],
        extra_sycl_cflags=["-fno-sycl-instrument-device-code"],
        extra_include_paths=[str(include)],
        extra_ldflags=[f"-L{library}", "-lccl", f"-Wl,-rpath,{library}"],
        with_cuda=False,
        with_sycl=True,
        verbose=verbose,
        build_directory=build_dir,
    )
    return _MODULE


def _check_xpu_contiguous(tensor: torch.Tensor, name: str) -> None:
    if tensor.device.type != "xpu" or not tensor.is_contiguous():
        raise ValueError(f"{name} must be a contiguous XPU tensor")


def _check_bf16_xpu(tensor: torch.Tensor, name: str) -> None:
    _check_xpu_contiguous(tensor, name)
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be a contiguous BF16 XPU tensor")


def _check_equal_a2a_xpu(tensor: torch.Tensor, name: str) -> None:
    _check_xpu_contiguous(tensor, name)
    if tensor.dtype not in {torch.bfloat16, torch.int64}:
        raise ValueError(f"{name} must be a contiguous BF16 or int64 XPU tensor")


def _split_sizes(values: Sequence[int], name: str, world_size: int) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if len(result) != world_size or any(value < 0 for value in result):
        raise ValueError(f"{name} must contain {world_size} nonnegative split sizes")
    return result


def _row_width(tensor: torch.Tensor) -> int:
    return math.prod(int(size) for size in tensor.shape[1:])


class NativeCclWork:
    """Owns a oneCCL event and its input/output tensors.

    The C++ launch path establishes device-side producer and consumer ordering.
    In private-stream mode it records producer readiness on the PyTorch stream,
    passes that event to oneCCL, and then places reciprocal completion fences
    on PyTorch consumer streams.  ``wait_stream`` adds another consumer fence
    after a stream change.  ``wait`` is intentionally host-blocking and is
    useful only at an explicit synchronization boundary.  Keep this object
    alive until all consumer streams have consumed ``output``.
    """

    def __init__(
        self,
        implementation: object,
        output: torch.Tensor,
        owner: "NativeCclA2A",
        *,
        bridge_current_stream: bool = True,
    ):
        self._implementation = implementation
        self.output = output
        # Retain the communicator independently of a caller dropping its
        # NativeCclA2A reference while this event remains in flight.
        self._owner = owner
        # The default immediate bridge is mandatory for ordinary callers:
        # subsequent PyTorch operations on the current stream must observe the
        # oneCCL output.  A host-thread submitter may defer it only when it
        # will explicitly call wait_stream on the eventual consumer stream
        # before that stream enqueues any output use or a stream-fence reaper
        # can observe this work.
        if bridge_current_stream:
            with _record("moe.comm.native_ccl_consumer_fence"):
                self._implementation.wait_stream()

    def wait(self) -> None:
        """Block the host until the direct-oneCCL event is complete."""

        self._implementation.wait()

    def wait_stream(self) -> None:
        """Add an event dependency to the then-current PyTorch XPU stream."""

        self._implementation.wait_stream()

    def is_completed(self) -> bool:
        return bool(self._implementation.is_completed())

    def is_stream_fence_completed(self) -> bool:
        """Whether every registered consumer-stream fence has completed."""

        return bool(self._implementation.is_stream_fence_completed())


class NativeCclA2A:
    """A dynamic-size direct-oneCCL communicator for one PyTorch group.

    ``require_async=True`` is the safe default: oneCCL's global
    ``CCL_OP_SYNC=1`` workaround calls ``ccl::event.wait`` inside collective
    submission, so it defeats the purpose of this prototype and cannot be
    bypassed per communicator.  Set it to ``False`` only to perform a
    correctness/control run under that inherited safety setting.
    """

    def __init__(self, group: dist.ProcessGroup | None = None, *, require_async: bool = True):
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("initialize torch.distributed before constructing NativeCclA2A")
        self._group = group
        self._world_size = dist.get_world_size(group)
        self._rank = dist.get_rank(group)
        self._global_ranks = tuple(dist.get_process_group_ranks(group))
        if len(self._global_ranks) != self._world_size:
            raise RuntimeError("could not resolve the supplied process group's ordered ranks")
        self._require_async = require_async
        self._implementation: object | None = None
        self._device_index: int | None = None

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def rank(self) -> int:
        return self._rank

    def _check_async_environment(self) -> None:
        if self._require_async and _env_enabled("CCL_OP_SYNC"):
            raise RuntimeError(
                "CCL_OP_SYNC=1 forces oneCCL's internal ccl::event.wait() during "
                "collective submission. Start a separately validated run with "
                "CCL_OP_SYNC=0 before using this async prototype, or construct "
                "NativeCclA2A(require_async=False) for a correctness-only control."
            )

    def _ensure_communicator(self, tensor: torch.Tensor) -> object:
        _check_xpu_contiguous(tensor, "input")
        self._check_async_environment()
        if self._implementation is not None:
            if tensor.get_device() != self._device_index:
                raise ValueError("one NativeCclA2A instance cannot span XPU devices")
            return self._implementation

        ops = load_native_ccl_a2a_ops()
        implementation = ops.NativeCclCommunicator()
        # The main KVS has to stay alive on group rank zero.  The extension
        # retains it; object broadcast is only a one-time control-plane step.
        root_address = implementation.create_root_kvs_address() if self._rank == 0 else b""
        address_box = [root_address]
        dist.broadcast_object_list(
            address_box,
            src=self._global_ranks[0],
            group=self._group,
            device=tensor.device,
        )
        if not isinstance(address_box[0], bytes):
            raise RuntimeError("oneCCL KVS bootstrap returned a non-byte address")
        implementation.initialize(
            address_box[0], self._world_size, self._rank, tensor.get_device()
        )
        self._implementation = implementation
        self._device_index = tensor.get_device()
        return implementation

    def warm_up(self, tensor: torch.Tensor) -> None:
        """Create the communicator before a latency-sensitive collective."""

        _check_xpu_contiguous(tensor, "input")
        self._ensure_communicator(tensor)

    def all_to_all_single(
        self,
        input: torch.Tensor,
        *,
        output: torch.Tensor | None = None,
        async_op: bool = True,
        bridge_current_stream: bool = True,
    ) -> tuple[torch.Tensor, NativeCclWork] | torch.Tensor:
        """Run equal-size BF16 or int64 all-to-all on the current XPU stream.

        Set ``bridge_current_stream=False`` only for a host-thread handoff
        that will explicitly bridge ``NativeCclWork`` to its real consumer
        before work reaping or output use.
        """

        _check_equal_a2a_xpu(input, "input")
        if output is None:
            output = torch.empty_like(input)
        _check_equal_a2a_xpu(output, "output")
        if (
            output.device != input.device
            or output.dtype != input.dtype
            or output.numel() != input.numel()
        ):
            raise ValueError("input and output must share a device, dtype, and number of elements")
        if input.numel() % self._world_size:
            raise ValueError("equal all-to-all input elements must divide world_size")
        implementation = self._ensure_communicator(input)
        with _record("moe.comm.native_ccl_launch"):
            native_work = implementation.alltoall_equal(input, output)
        work = NativeCclWork(
            native_work, output, self, bridge_current_stream=bridge_current_stream
        )
        if async_op:
            return output, work
        work.wait()
        return output

    def all_reduce_sum(
        self,
        input: torch.Tensor,
        *,
        output: torch.Tensor | None = None,
        async_op: bool = True,
        bridge_current_stream: bool = True,
    ) -> tuple[torch.Tensor, NativeCclWork] | torch.Tensor:
        """Run a BF16 SUM all-reduce, in place when ``output`` is omitted.

        Set ``bridge_current_stream=False`` only for a host-thread handoff
        that will explicitly bridge ``NativeCclWork`` to its real consumer
        before work reaping or output use.
        """

        _check_bf16_xpu(input, "input")
        if output is None:
            output = input
        _check_bf16_xpu(output, "output")
        if output.device != input.device or output.numel() != input.numel():
            raise ValueError("input and output must share a device and number of elements")
        implementation = self._ensure_communicator(input)
        with _record("moe.comm.native_ccl_launch"):
            native_work = implementation.allreduce_sum_bf16(input, output)
        work = NativeCclWork(
            native_work, output, self, bridge_current_stream=bridge_current_stream
        )
        if async_op:
            return output, work
        work.wait()
        return output

    def all_reduce(
        self,
        input: torch.Tensor,
        *,
        output: torch.Tensor | None = None,
        async_op: bool = True,
        bridge_current_stream: bool = True,
    ) -> tuple[torch.Tensor, NativeCclWork] | torch.Tensor:
        """Alias for the bridge's only reduction: BF16 SUM all-reduce."""

        return self.all_reduce_sum(
            input,
            output=output,
            async_op=async_op,
            bridge_current_stream=bridge_current_stream,
        )

    def all_to_all_v(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        output_split_sizes: Sequence[int],
        input_split_sizes: Sequence[int],
        *,
        async_op: bool = True,
        bridge_current_stream: bool = True,
    ) -> tuple[torch.Tensor, NativeCclWork] | torch.Tensor:
        """Run BF16 all-to-all-v using PyTorch's dim-0 split-size convention.

        Set ``bridge_current_stream=False`` only for a host-thread handoff
        that will explicitly bridge ``NativeCclWork`` to its real consumer
        before work reaping or output use.
        """

        _check_bf16_xpu(input, "input")
        _check_bf16_xpu(output, "output")
        if input.device != output.device or input.ndim < 1 or output.ndim < 1:
            raise ValueError("input/output must be same-device rank-1-or-higher tensors")
        send_rows = _split_sizes(input_split_sizes, "input_split_sizes", self._world_size)
        recv_rows = _split_sizes(output_split_sizes, "output_split_sizes", self._world_size)
        if sum(send_rows) != input.size(0) or sum(recv_rows) != output.size(0):
            raise ValueError("all-to-all-v split sizes must sum to input/output dim-0 sizes")
        send_counts = tuple(rows * _row_width(input) for rows in send_rows)
        recv_counts = tuple(rows * _row_width(output) for rows in recv_rows)
        implementation = self._ensure_communicator(input)
        with _record("moe.comm.native_ccl_launch"):
            native_work = implementation.alltoallv_bf16(
                input, output, send_counts, recv_counts
            )
        work = NativeCclWork(
            native_work, output, self, bridge_current_stream=bridge_current_stream
        )
        if async_op:
            return output, work
        work.wait()
        return output
