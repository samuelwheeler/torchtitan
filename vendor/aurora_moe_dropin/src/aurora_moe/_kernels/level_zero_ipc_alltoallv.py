"""Exact compact variable-split Level Zero IPC transport for Aurora EP.

This module deliberately owns a flat exported receive staging allocation rather
than a ``[peer, capacity]`` buffer.  Each sender writes only its live compact
route block into the destination's source-major offset.  The allocation is
reusable transport storage; the tensor returned to the MoE is always sized to
the actual receive split sum.
"""

from __future__ import annotations

import os
from math import prod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load

from ._prebuilt_extension import load_prebuilt_ipc_extension
from .level_zero_ipc import load_level_zero_ipc_ops
from .level_zero_ipc_device_barrier import IpcDeviceBarrierEpoch
from .level_zero_ipc_equal_a2a import _Descriptor, _client_exchange, _root_exchange


_MODULE: ModuleType | None = None

# A compact forward dispatch result is retained by the local expert autograd
# node until backward.  The three other all-to-all-v results are consumed on
# the current stream before the next use of their staging slot.  Keep those
# lifetimes physically separate so an opt-in zero-copy transient output never
# aliases a saved forward activation.
_OUTPUT_SLOTS = 2
_RETAINED_OUTPUT_SLOT = 0
_TRANSIENT_OUTPUT_SLOT = 1
_EXPERT_MAJOR_VECTORS_PER_WORKGROUP = 8192


@dataclass(frozen=True)
class ExpertMajorIpcBlockPlan:
    """Exact runtime fragments for direct physical/expert-major IPC layouts."""

    forward_blocks: tuple[tuple[int, int, int, int], ...]
    inverse_blocks: tuple[tuple[int, int, int, int], ...]
    forward_output_rows: int
    inverse_output_rows: int
    forward_capacity_rows: int
    inverse_capacity_rows: int
    peers: int
    experts: int
    forward_input_offsets: tuple[int, ...]
    forward_remote_offsets: tuple[int, ...]
    forward_counts: tuple[int, ...]
    inverse_input_offsets: tuple[int, ...]
    inverse_remote_offsets: tuple[int, ...]
    inverse_counts: tuple[int, ...]


def make_expert_major_ipc_block_plan(
    counts: torch.Tensor, rank: int
) -> ExpertMajorIpcBlockPlan:
    """Map exact ``[source, destination, expert]`` counts into live IPC blocks.

    A forward block maps the sender's destination-major/source-expert input
    into the receiver's expert-major interval.  An inverse block maps that
    receiver-local expert-major interval back into the original sender's
    destination-major physical order.  Every offset and count is derived from
    the supplied runtime count tensor; zero-size fragments are omitted.
    """

    if (
        counts.device.type != "cpu"
        or counts.dtype != torch.int64
        or counts.ndim != 3
        or counts.size(0) != counts.size(1)
        or not counts.is_contiguous()
    ):
        raise ValueError(
            "counts must be contiguous CPU int64 [source, destination, expert]"
        )
    peers, _, experts = counts.shape
    if peers < 1 or experts < 1 or not 0 <= rank < peers:
        raise ValueError("counts must have nonempty peer/expert axes and a valid rank")
    if bool((counts < 0).any().item()):
        raise ValueError("expert fragment counts must be nonnegative")

    def _value(*index: int) -> int:
        return int(counts[index].item())

    forward_input_offsets: list[int] = []
    forward_remote_offsets: list[int] = []
    forward_counts: list[int] = []
    forward: list[tuple[int, int, int, int]] = []
    for expert in range(experts):
        for destination in range(peers):
            input_offset = int(counts[rank, :destination, :].sum().item()) + int(
                counts[rank, destination, :expert].sum().item()
            )
            output_offset = int(counts[:, destination, :expert].sum().item()) + int(
                counts[:rank, destination, expert].sum().item()
            )
            rows = _value(rank, destination, expert)
            forward_input_offsets.append(input_offset)
            forward_remote_offsets.append(output_offset)
            forward_counts.append(rows)
            if rows:
                forward.append((input_offset, output_offset, rows, destination))

    inverse_input_offsets: list[int] = []
    inverse_remote_offsets: list[int] = []
    inverse_counts: list[int] = []
    inverse: list[tuple[int, int, int, int]] = []
    for expert in range(experts):
        for source in range(peers):
            input_offset = int(counts[:, rank, :expert].sum().item()) + int(
                counts[:source, rank, expert].sum().item()
            )
            output_offset = int(counts[source, :rank, :].sum().item()) + int(
                counts[source, rank, :expert].sum().item()
            )
            rows = _value(source, rank, expert)
            inverse_input_offsets.append(input_offset)
            inverse_remote_offsets.append(output_offset)
            inverse_counts.append(rows)
            if rows:
                inverse.append((input_offset, output_offset, rows, source))

    return ExpertMajorIpcBlockPlan(
        forward_blocks=tuple(forward),
        inverse_blocks=tuple(inverse),
        forward_output_rows=int(counts[:, rank, :].sum().item()),
        inverse_output_rows=int(counts[rank, :, :].sum().item()),
        forward_capacity_rows=int(counts.sum(dim=(0, 2)).max().item()),
        inverse_capacity_rows=int(counts.sum(dim=(1, 2)).max().item()),
        peers=peers,
        experts=experts,
        forward_input_offsets=tuple(forward_input_offsets),
        forward_remote_offsets=tuple(forward_remote_offsets),
        forward_counts=tuple(forward_counts),
        inverse_input_offsets=tuple(inverse_input_offsets),
        inverse_remote_offsets=tuple(inverse_remote_offsets),
        inverse_counts=tuple(inverse_counts),
    )


def load_level_zero_ipc_alltoallv_ops(verbose: bool = False) -> ModuleType:
    """Load the isolated exact all-to-all-v peer-write extension."""

    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.xpu.is_available():
        raise RuntimeError("Aurora XPU is required for the IPC alltoallv transport")
    _MODULE = load_prebuilt_ipc_extension(
        component="level_zero_ipc_alltoallv",
        module_name="aurora_moe_level_zero_ipc_alltoallv",
        required=("push_alltoallv_bf16_typed_direct_async",),
    )
    if _MODULE is not None:
        return _MODULE
    source = Path(__file__).with_name("csrc") / "level_zero_ipc_alltoallv.sycl"
    build_dir = os.environ.get("AURORA_MOE_SYCL_BUILD_DIR")
    if build_dir:
        build_dir = str(Path(build_dir) / "level_zero_ipc_alltoallv")
        Path(build_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_XPU_ARCH_LIST", "pvc")
    _MODULE = load(
        name="aurora_moe_level_zero_ipc_alltoallv",
        sources=[str(source)],
        extra_cflags=["-O3", "-DNDEBUG"],
        extra_sycl_cflags=["-fno-sycl-instrument-device-code"],
        with_cuda=False,
        with_sycl=True,
        verbose=verbose,
        build_directory=build_dir,
    )
    return _MODULE


class LevelZeroIpcAllToAllV:
    """Asynchronous exact BF16 all-to-all-v over one node-local EP group.

    ``exchange_async`` accepts compact source-major input and ordinary exact
    all-to-all-v split descriptors.  ``remote_receive_offsets`` is indexed by
    destination rank and identifies where this rank's source block belongs in
    each remote source-major output.  It is derived from the globally shared
    route-count matrix, not inferred from a capacity or a local heuristic.
    """

    def __init__(
        self,
        group: dist.ProcessGroup | None = None,
        *,
        socket_path: str | os.PathLike[str] | None = None,
    ) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("initialize torch.distributed before IPC alltoallv")
        if not torch.xpu.is_available():
            raise RuntimeError("Aurora XPU is required for IPC alltoallv")
        self._group = group
        self._world_size = dist.get_world_size(group)
        self._rank = dist.get_rank(group)
        if self._world_size < 2 or self._world_size > 12:
            raise ValueError("IPC alltoallv supports 2--12 node-local ranks")
        self._device_index = torch.xpu.current_device()
        self._device = torch.device("xpu", self._device_index)
        base = socket_path or os.environ.get("AURORA_MOE_L0_IPC_ALLTOALLV_SOCKET")
        if not base:
            raise RuntimeError(
                "set AURORA_MOE_L0_IPC_ALLTOALLV_SOCKET for IPC descriptor exchange"
            )
        self._socket_base = Path(base)
        self._ipc_bias, self._ipc_bias_flags = self._mapping_policy()
        self._import_backend = self._import_policy()
        self._ipc_ops = load_level_zero_ipc_ops()
        self._push_ops = load_level_zero_ipc_alltoallv_ops()
        # A transient SYCL out-of-order queue can be destroyed while its
        # current-stream completion bridge is still live.  Retaining one queue
        # per submitting host thread avoids that per-exchange Level Zero
        # teardown path.  It is a transport implementation detail, not a
        # routing-shape restriction.
        persistent_ooo_queue = os.environ.get(
            "AURORA_MOE_L0_IPC_ALLTOALLV_PERSISTENT_OOO_QUEUE", "1"
        )
        if persistent_ooo_queue not in {"0", "1"}:
            raise ValueError(
                "AURORA_MOE_L0_IPC_ALLTOALLV_PERSISTENT_OOO_QUEUE must be 0 or 1"
            )
        self._persistent_ooo_queue: object | None = None
        if persistent_ooo_queue == "1":
            queue_type = getattr(
                self._push_ops, "PersistentOooAllToAllVQueue", None
            )
            if queue_type is None:
                raise RuntimeError(
                    "AURORA_MOE_L0_IPC_ALLTOALLV_PERSISTENT_OOO_QUEUE=1 requires "
                    "a rebuilt level_zero_ipc_alltoallv extension with "
                    "PersistentOooAllToAllVQueue"
                )
            self._persistent_ooo_queue = queue_type()
        self._readiness = IpcDeviceBarrierEpoch(
            group, socket_path=f"{self._socket_base}.readiness"
        )
        self._receive_staging: torch.Tensor | None = None
        self._remote_receive_addresses: tuple[int, ...] = ()
        self._remote_receive_ptrs: torch.Tensor | None = None
        self._exporter: object | None = None
        self._imports: list[object | None] = []
        self._descriptors: list[_Descriptor] = []
        # ``_capacity_elements`` is per output slot, not the total exported
        # allocation.  The exported allocation has two equal slots so a
        # transient zero-copy result can never overwrite an output retained by
        # forward autograd.
        self._capacity_elements = 0
        self._dtype: torch.dtype | None = None
        minimum_capacity = os.environ.get(
            "AURORA_MOE_L0_IPC_ALLTOALLV_MIN_CAPACITY_ELEMENTS", "0"
        )
        try:
            self._minimum_capacity_elements = int(minimum_capacity)
        except ValueError as error:
            raise ValueError(
                "AURORA_MOE_L0_IPC_ALLTOALLV_MIN_CAPACITY_ELEMENTS must be a "
                "nonnegative integer"
            ) from error
        if self._minimum_capacity_elements < 0:
            raise ValueError(
                "AURORA_MOE_L0_IPC_ALLTOALLV_MIN_CAPACITY_ELEMENTS must be a "
                "nonnegative integer"
            )
        log_capacity = os.environ.get(
            "AURORA_MOE_L0_IPC_ALLTOALLV_LOG_CAPACITY", "0"
        )
        if log_capacity not in {"0", "1"}:
            raise ValueError(
                "AURORA_MOE_L0_IPC_ALLTOALLV_LOG_CAPACITY must be 0 or 1"
            )
        self._log_capacity = log_capacity == "1"
        self._epoch = 0
        self._closed = False

    @staticmethod
    def _mapping_policy() -> tuple[str, int]:
        policy = os.environ.get("AURORA_MOE_L0_IPC_BIAS", "default")
        flags = {"default": 0, "cached": 1, "uncached": 2}
        if policy not in flags:
            raise ValueError("AURORA_MOE_L0_IPC_BIAS must be default, cached, or uncached")
        return policy, flags[policy]

    @staticmethod
    def _import_policy() -> str:
        backend = os.environ.get("AURORA_MOE_L0_IPC_IMPORT_BACKEND", "scm")
        if backend not in {"scm", "pidfd"}:
            raise ValueError("AURORA_MOE_L0_IPC_IMPORT_BACKEND must be scm or pidfd")
        return backend

    @staticmethod
    def _signed_pointer(address: int) -> int:
        return address - (1 << 64) if address >= 1 << 63 else address

    @property
    def capacity_elements(self) -> int:
        """Current reusable receive-staging capacity per output slot in BF16 elements."""

        return self._capacity_elements

    def _barrier(self) -> None:
        dist.barrier(group=self._group)

    def _epoch_path(self) -> Path:
        path = Path(f"{self._socket_base}.payload.epoch{self._epoch}")
        if len(os.fsencode(path)) >= 108:
            raise RuntimeError(f"IPC socket path is too long: {path}")
        return path

    def _close_imports(self) -> None:
        for imported in self._imports:
            if imported is not None:
                imported.close()
        self._imports = []
        for descriptor in self._descriptors:
            if descriptor.fd >= 0 and not descriptor.exporter_owned_fd:
                descriptor.close_received_fd()
        self._descriptors = []
        self._remote_receive_addresses = ()
        self._remote_receive_ptrs = None

    def _close_exporter(self) -> None:
        if self._exporter is not None:
            self._exporter.close()
        self._exporter = None
        self._receive_staging = None
        self._capacity_elements = 0

    def _retire_epoch(self) -> None:
        self._close_imports()
        self._barrier()
        self._close_exporter()
        self._barrier()

    def _create_epoch(self, capacity_elements: int, dtype: torch.dtype) -> None:
        if capacity_elements <= 0:
            raise ValueError("IPC alltoallv staging capacity must be positive")
        self._barrier()
        torch.xpu.synchronize(self._device)
        self._retire_epoch()
        self._epoch += 1
        self._receive_staging = torch.empty(
            (_OUTPUT_SLOTS * capacity_elements,), dtype=dtype, device=self._device
        )
        if self._import_backend == "pidfd":
            # Match oneCCL's parent-only ptrace grant before descriptor
            # duplication. The readiness channel performs the same setup for
            # its control allocation; the payload allocation needs it too.
            self._ipc_ops.enable_pidfd_parent_tracer()
        self._exporter = self._ipc_ops.IpcExport(self._receive_staging)
        own = _Descriptor(
            self._exporter.fd(),
            os.getpid(),
            self._exporter.tensor_offset_bytes(),
            self._exporter.handle_bytes(),
            exporter_owned_fd=True,
        )
        path = self._epoch_path()
        if self._rank == 0:
            descriptors = _root_exchange(
                path, self._world_size, own, send_fd=self._import_backend == "scm"
            )
        else:
            descriptors = _client_exchange(
                path,
                self._rank,
                self._world_size,
                own,
                send_fd=self._import_backend == "scm",
            )
        imports: list[object | None] = [None] * self._world_size
        for source, descriptor in enumerate(descriptors):
            if source != self._rank:
                imports[source] = self._ipc_ops.IpcImport(
                    descriptor.handle,
                    descriptor.take_fd() if self._import_backend == "scm" else -1,
                    self._device_index,
                    self._ipc_bias == "uncached",
                    descriptor.pid,
                    self._import_backend == "pidfd",
                    self._ipc_bias_flags,
                )
        own_descriptor = descriptors[self._rank]
        if own_descriptor.exporter_owned_fd:
            if self._rank != 0:
                raise RuntimeError("only root may retain its exporter-owned descriptor")
        elif own_descriptor.fd >= 0:
            own_descriptor.close_received_fd()

        addresses: list[int] = []
        for source, descriptor in enumerate(descriptors):
            if source == self._rank:
                address = self._receive_staging.data_ptr()
            else:
                imported = imports[source]
                if imported is None:
                    raise RuntimeError(f"missing IPC receive mapping for rank {source}")
                address = int(imported.mapping_address()) + descriptor.tensor_offset_bytes
            if address % 2:
                raise RuntimeError("IPC receive mapping is not BF16 aligned")
            addresses.append(self._signed_pointer(address))

        self._imports = imports
        self._descriptors = descriptors
        self._remote_receive_addresses = tuple(addresses)
        self._remote_receive_ptrs = torch.tensor(
            addresses, dtype=torch.int64, device=self._device
        )
        self._capacity_elements = capacity_elements
        self._dtype = dtype
        torch.xpu.synchronize(self._device)
        self._barrier()

    def _ensure_capacity(self, capacity_elements: int, dtype: torch.dtype) -> None:
        if self._closed:
            raise RuntimeError("IPC alltoallv is closed")
        if dtype != torch.bfloat16:
            raise ValueError("IPC alltoallv supports BF16 payloads only")
        if capacity_elements <= 0:
            raise ValueError("IPC alltoallv capacity must be positive")
        if self._dtype is not None and dtype != self._dtype:
            raise ValueError("IPC alltoallv payload dtype cannot change after initialization")
        requested = capacity_elements
        capacity_elements = max(capacity_elements, self._minimum_capacity_elements)
        if capacity_elements > self._capacity_elements:
            # `capacity_elements` is a maximum over the globally replicated
            # route-count matrix, so every rank takes this collective epoch
            # change together.  Internal geometric growth avoids repeated
            # descriptor exchanges as token counts vary.
            grown = max(capacity_elements, max(1, self._capacity_elements * 2))
            if self._log_capacity and self._rank == 0:
                print(
                    "AURORA_MOE_L0_IPC_ALLTOALLV_CAPACITY "
                    f"epoch={self._epoch + 1} requested_elements={requested} "
                    f"minimum_elements={self._minimum_capacity_elements} "
                    f"previous_elements={self._capacity_elements} "
                    f"allocated_elements={grown} slots={_OUTPUT_SLOTS} "
                    f"allocated_bytes={_OUTPUT_SLOTS * grown * torch.tensor([], dtype=dtype).element_size()}",
                    flush=True,
                )
            self._create_epoch(grown, dtype)

    @staticmethod
    def _zero_copy_transient_requested() -> bool:
        """Whether eligible short-lived outputs may alias their IPC slot.

        This remains opt-in because a caller must prove that it has consumed a
        transient result on the current stream before issuing another
        transient exchange.  ``exchange_async`` itself keeps saved-forward
        dispatch results on a distinct slot regardless of this setting.
        """

        value = os.environ.get("AURORA_MOE_L0_IPC_ALLTOALLV_ZERO_COPY_TRANSIENT", "0")
        if value not in {"0", "1"}:
            raise ValueError(
                "AURORA_MOE_L0_IPC_ALLTOALLV_ZERO_COPY_TRANSIENT must be 0 or 1"
            )
        return value == "1"

    @staticmethod
    def _output_slot(*, retain_output: bool) -> int:
        return _RETAINED_OUTPUT_SLOT if retain_output else _TRANSIENT_OUTPUT_SLOT

    @staticmethod
    def _prefix(values: Sequence[int]) -> tuple[int, ...]:
        total = 0
        offsets: list[int] = []
        for value in values:
            if value < 0:
                raise ValueError("alltoallv split sizes must be nonnegative")
            offsets.append(total)
            total += int(value)
        return tuple(offsets)

    def exchange_async(
        self,
        input: torch.Tensor,
        input_splits: Sequence[int],
        output_splits: Sequence[int],
        remote_receive_offsets: Sequence[int],
        *,
        global_capacity_rows: int,
        output: torch.Tensor | None = None,
        retain_output: bool = False,
    ) -> torch.Tensor:
        """Enqueue an exact compact all-to-all-v without a host completion wait.

        ``retain_output`` identifies a forward-dispatch result that is saved
        through backward.  It always uses the dedicated retained slot and is
        copied into independent storage.  Other calls may return a current-
        stream view of the transient IPC slot only under the explicit
        zero-copy opt-in.
        """

        if self._closed:
            raise RuntimeError("IPC alltoallv is closed")
        if input.device != self._device or input.dtype != torch.bfloat16:
            raise ValueError("input must be a BF16 tensor on the local XPU tile")
        if not input.is_contiguous() or input.ndim < 1:
            raise ValueError("input must be contiguous with a leading route dimension")
        input_splits = tuple(int(value) for value in input_splits)
        output_splits = tuple(int(value) for value in output_splits)
        remote_receive_offsets = tuple(int(value) for value in remote_receive_offsets)
        if not (
            len(input_splits)
            == len(output_splits)
            == len(remote_receive_offsets)
            == self._world_size
        ):
            raise ValueError("alltoallv metadata must have one value per EP peer")
        if input.size(0) != sum(input_splits):
            raise ValueError("input rows do not match alltoallv input splits")
        if any(value < 0 for value in remote_receive_offsets):
            raise ValueError("remote alltoallv receive offsets must be nonnegative")
        width = input.numel() // input.size(0) if input.size(0) else prod(input.shape[1:])
        if width <= 0:
            raise ValueError("alltoallv payload width must be positive")
        capacity_rows = int(global_capacity_rows)
        if capacity_rows < sum(output_splits):
            raise ValueError("global alltoallv capacity is smaller than local receive rows")
        if any(
            offset + count > capacity_rows
            for offset, count in zip(remote_receive_offsets, input_splits)
        ):
            raise ValueError("remote alltoallv offsets exceed global staging capacity")
        if capacity_rows == 0:
            if any(input_splits) or any(output_splits):
                raise ValueError("zero capacity is valid only for all-zero split vectors")
            return input.new_empty((0, *input.shape[1:]))

        capacity_elements = capacity_rows * width
        self._ensure_capacity(capacity_elements, input.dtype)
        if self._receive_staging is None or not self._remote_receive_addresses:
            raise RuntimeError("IPC alltoallv staging was not initialized")
        slot = self._output_slot(retain_output=retain_output)
        slot_offset = slot * self._capacity_elements
        receive_slot = self._receive_staging.narrow(
            0, slot_offset, self._capacity_elements
        )
        remote_receive_addresses = tuple(
            address + slot_offset * input.element_size()
            for address in self._remote_receive_addresses
        )
        output_shape = (sum(output_splits), *input.shape[1:])
        zero_copy_transient = (
            not retain_output
            and output is None
            and self._zero_copy_transient_requested()
        )
        if output is None and not zero_copy_transient:
            output = input.new_empty(output_shape)
        elif output is not None and (
            output.device != input.device
            or output.dtype != input.dtype
            or tuple(output.shape) != output_shape
            or not output.is_contiguous()
        ):
            raise ValueError("output must be contiguous and match exact alltoallv shape")

        # The current stream owns both route packing and the output copy.  The
        # barriers are GPU-resident; neither waits on host collective progress.
        self._readiness.enqueue()
        input_rows = input.reshape(input.size(0), width)
        if self._persistent_ooo_queue is None:
            self._push_ops.push_alltoallv_bf16_typed_direct_async(
                input_rows,
                receive_slot,
                remote_receive_addresses,
                self._prefix(input_splits),
                remote_receive_offsets,
                input_splits,
                self._rank,
            )
        else:
            self._persistent_ooo_queue.push_alltoallv_bf16_typed_direct_async(
                input_rows,
                receive_slot,
                remote_receive_addresses,
                self._prefix(input_splits),
                remote_receive_offsets,
                input_splits,
                self._rank,
            )
        self._readiness.enqueue()
        live_output = receive_slot.narrow(0, 0, int(prod(output_shape)))
        if zero_copy_transient:
            return live_output.reshape(output_shape)
        assert output is not None
        output.reshape(-1).copy_(live_output)
        return output

    def exchange_expert_major_async(
        self,
        input: torch.Tensor,
        blocks: Sequence[tuple[int, int, int, int]],
        *,
        output_rows: int,
        global_capacity_rows: int,
        output: torch.Tensor | None = None,
        retain_output: bool = False,
    ) -> torch.Tensor:
        """Exchange exact live fragments directly into expert-major layout.

        Each block is ``(input_offset_rows, remote_output_offset_rows, rows,
        destination_rank)``.  The host planner derives it from the full live
        ``[source, destination, local_expert]`` count tensor, so this method
        does not infer a capacity, pad a fragment, or impose a fixed expert
        count.
        """

        if self._closed:
            raise RuntimeError("IPC alltoallv is closed")
        if input.device != self._device or input.dtype != torch.bfloat16:
            raise ValueError("input must be a BF16 tensor on the local XPU tile")
        if not input.is_contiguous() or input.ndim < 1:
            raise ValueError("input must be contiguous with a leading route dimension")
        if output_rows < 0 or global_capacity_rows < output_rows:
            raise ValueError("expert-major output rows exceed IPC staging capacity")
        width = input.numel() // input.size(0) if input.size(0) else prod(input.shape[1:])
        if width <= 0:
            raise ValueError("expert-major IPC payload width must be positive")
        normalized = tuple(tuple(int(value) for value in block) for block in blocks)
        if any(len(block) != 4 for block in normalized):
            raise ValueError("expert-major IPC blocks must have four values")
        input_rows = input.size(0)
        covered_rows = 0
        for input_offset, output_offset, rows, destination in normalized:
            if (
                input_offset < 0
                or output_offset < 0
                or rows <= 0
                or not 0 <= destination < self._world_size
                or input_offset + rows > input_rows
                or output_offset + rows > global_capacity_rows
            ):
                raise ValueError("expert-major IPC block is outside its exact live extent")
            covered_rows += rows
        if covered_rows != input_rows:
            raise ValueError("expert-major IPC blocks must cover every input row exactly once")
        if not hasattr(self._push_ops, "push_alltoallv_bf16_expert_major_blocks_async"):
            raise RuntimeError(
                "expert-major IPC requires a rebuilt level_zero_ipc_alltoallv extension"
            )

        capacity_elements = global_capacity_rows * width
        if capacity_elements == 0:
            if input_rows:
                raise ValueError("zero expert-major capacity cannot carry input rows")
            return input.new_empty((0, *input.shape[1:]))
        self._ensure_capacity(capacity_elements, input.dtype)
        if (
            self._receive_staging is None
            or self._remote_receive_ptrs is None
            or not self._remote_receive_addresses
        ):
            raise RuntimeError("IPC alltoallv staging was not initialized")
        slot = self._output_slot(retain_output=retain_output)
        slot_offset = slot * self._capacity_elements
        receive_slot = self._receive_staging.narrow(
            0, slot_offset, self._capacity_elements
        )
        remote_receive_ptrs = self._remote_receive_ptrs + (
            slot_offset * input.element_size()
        )
        output_shape = (output_rows, *input.shape[1:])
        zero_copy_transient = (
            not retain_output
            and output is None
            and self._zero_copy_transient_requested()
        )
        if output is None and not zero_copy_transient:
            output = input.new_empty(output_shape)
        elif output is not None and (
            output.device != input.device
            or output.dtype != input.dtype
            or tuple(output.shape) != output_shape
            or not output.is_contiguous()
        ):
            raise ValueError("output must be contiguous and match expert-major output shape")
        metadata = torch.tensor(normalized, dtype=torch.int64, device=self._device)
        if metadata.numel() == 0:
            metadata = metadata.reshape(0, 4)
        max_elements = max((block[2] * width for block in normalized), default=0)
        groups_per_block = max(
            1,
            (max_elements + 4 * _EXPERT_MAJOR_VECTORS_PER_WORKGROUP - 1)
            // (4 * _EXPERT_MAJOR_VECTORS_PER_WORKGROUP),
        )

        self._readiness.enqueue()
        input_rows_view = input.reshape(input_rows, width)
        if self._persistent_ooo_queue is None:
            self._push_ops.push_alltoallv_bf16_expert_major_blocks_async(
                input_rows_view,
                receive_slot,
                remote_receive_ptrs,
                metadata,
                groups_per_block,
            )
        else:
            self._persistent_ooo_queue.push_alltoallv_bf16_expert_major_blocks_async(
                input_rows_view,
                receive_slot,
                remote_receive_ptrs,
                metadata,
                groups_per_block,
            )
        self._readiness.enqueue()
        live_output = receive_slot.narrow(0, 0, output_rows * width)
        if zero_copy_transient:
            return live_output.reshape(output_shape)
        assert output is not None
        output.reshape(-1).copy_(live_output)
        return output

    def exchange_expert_major_typed_async(
        self,
        input: torch.Tensor,
        input_offsets: Sequence[int],
        remote_output_offsets: Sequence[int],
        row_counts: Sequence[int],
        *,
        output_rows: int,
        global_capacity_rows: int,
        output: torch.Tensor | None = None,
        retain_output: bool = False,
    ) -> torch.Tensor:
        """Exchange source/expert fragments with one CCL-shaped peer group per expert."""

        if self._closed:
            raise RuntimeError("IPC alltoallv is closed")
        if input.device != self._device or input.dtype != torch.bfloat16:
            raise ValueError("input must be a BF16 tensor on the local XPU tile")
        if not input.is_contiguous() or input.ndim < 1:
            raise ValueError("input must be contiguous with a leading route dimension")
        input_offsets = tuple(int(value) for value in input_offsets)
        remote_output_offsets = tuple(int(value) for value in remote_output_offsets)
        row_counts = tuple(int(value) for value in row_counts)
        if not (
            len(input_offsets)
            == len(remote_output_offsets)
            == len(row_counts)
            and len(row_counts) % self._world_size == 0
        ):
            raise ValueError("expert-major typed metadata must contain whole peer groups")
        if output_rows < 0 or global_capacity_rows < output_rows:
            raise ValueError("expert-major output rows exceed IPC staging capacity")
        input_rows = input.size(0)
        if sum(row_counts) != input_rows:
            raise ValueError("expert-major typed row counts must cover every input row")
        if any(
            input_offset < 0
            or output_offset < 0
            or rows < 0
            or input_offset + rows > input_rows
            or output_offset + rows > global_capacity_rows
            for input_offset, output_offset, rows in zip(
                input_offsets, remote_output_offsets, row_counts
            )
        ):
            raise ValueError("expert-major typed metadata is outside the live extent")
        if not hasattr(
            self._push_ops, "push_alltoallv_bf16_expert_major_typed_async"
        ):
            raise RuntimeError(
                "expert-major typed IPC requires a rebuilt level_zero_ipc_alltoallv extension"
            )
        width = input.numel() // input_rows if input_rows else prod(input.shape[1:])
        if width <= 0:
            raise ValueError("expert-major IPC payload width must be positive")
        capacity_elements = global_capacity_rows * width
        if capacity_elements == 0:
            return input.new_empty((0, *input.shape[1:]))
        self._ensure_capacity(capacity_elements, input.dtype)
        if self._receive_staging is None or not self._remote_receive_addresses:
            raise RuntimeError("IPC alltoallv staging was not initialized")
        slot = self._output_slot(retain_output=retain_output)
        slot_offset = slot * self._capacity_elements
        receive_slot = self._receive_staging.narrow(
            0, slot_offset, self._capacity_elements
        )
        remote_receive_addresses = tuple(
            address + slot_offset * input.element_size()
            for address in self._remote_receive_addresses
        )
        output_shape = (output_rows, *input.shape[1:])
        zero_copy_transient = (
            not retain_output
            and output is None
            and self._zero_copy_transient_requested()
        )
        if output is None and not zero_copy_transient:
            output = input.new_empty(output_shape)
        elif output is not None and (
            output.device != input.device
            or output.dtype != input.dtype
            or tuple(output.shape) != output_shape
            or not output.is_contiguous()
        ):
            raise ValueError("output must be contiguous and match expert-major output shape")

        self._readiness.enqueue()
        input_rows_view = input.reshape(input_rows, width)
        if self._persistent_ooo_queue is None:
            self._push_ops.push_alltoallv_bf16_expert_major_typed_async(
                input_rows_view,
                receive_slot,
                remote_receive_addresses,
                input_offsets,
                remote_output_offsets,
                row_counts,
                self._rank,
            )
        else:
            self._persistent_ooo_queue.push_alltoallv_bf16_expert_major_typed_async(
                input_rows_view,
                receive_slot,
                remote_receive_addresses,
                input_offsets,
                remote_output_offsets,
                row_counts,
                self._rank,
            )
        self._readiness.enqueue()
        live_output = receive_slot.narrow(0, 0, output_rows * width)
        if zero_copy_transient:
            return live_output.reshape(output_shape)
        assert output is not None
        output.reshape(-1).copy_(live_output)
        return output

    def close(self) -> None:
        """Collectively retire imported mappings after queued work completes."""

        if self._closed:
            return
        torch.xpu.synchronize(self._device)
        if self._persistent_ooo_queue is not None:
            self._persistent_ooo_queue.close()
            self._persistent_ooo_queue = None
        self._retire_epoch()
        self._readiness.close()
        self._closed = True
