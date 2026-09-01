"""Persistent Level Zero IPC control counters and current-stream barriers."""

from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load

from ._ipc_trace import emit_ipc_trace, ipc_trace_enabled
from ._prebuilt_extension import load_prebuilt_ipc_extension
from .level_zero_ipc import load_level_zero_ipc_ops
from .level_zero_ipc_equal_a2a import _Descriptor, _client_exchange, _root_exchange


_MODULE: ModuleType | None = None
_SLOTS = 3
_COUNTER_STRIDE_WORDS = 8
_CACHE_LINE_BYTES = 64
_COUNTER_WORD_BYTES = 8
_DIAGNOSTIC_STAMP_WORD = 1


def load_level_zero_ipc_device_barrier_ops(verbose: bool = False) -> ModuleType:
    """Build and load the isolated IPC device-barrier extension."""

    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.xpu.is_available():
        raise RuntimeError("Aurora XPU is required for the IPC device-barrier probe")
    _MODULE = load_prebuilt_ipc_extension(
        component="level_zero_ipc_device_barrier",
        module_name="aurora_moe_level_zero_ipc_device_barrier",
        required=("enqueue_ipc_device_barrier", "slots", "counter_stride_words"),
    )
    if _MODULE is not None:
        return _MODULE
    source = Path(__file__).with_name("csrc") / "level_zero_ipc_device_barrier.sycl"
    build_dir = os.environ.get("AURORA_MOE_SYCL_BUILD_DIR")
    if build_dir:
        build_dir = str(Path(build_dir) / "level_zero_ipc_device_barrier")
        Path(build_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_XPU_ARCH_LIST", "pvc")
    _MODULE = load(
        name="aurora_moe_level_zero_ipc_device_barrier",
        sources=[str(source)],
        extra_cflags=["-O3", "-DNDEBUG"],
        extra_sycl_cflags=["-fno-sycl-instrument-device-code"],
        with_cuda=False,
        with_sycl=True,
        verbose=verbose,
        build_directory=build_dir,
    )
    return _MODULE


class IpcDeviceBarrierEpoch:
    """Own persistent control mappings for a same-node IPC barrier channel."""

    def __init__(
        self,
        group: dist.ProcessGroup | None = None,
        *,
        socket_path: str | os.PathLike[str] | None = None,
    ) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("initialize torch.distributed before constructing an IPC barrier")
        if not torch.xpu.is_available():
            raise RuntimeError("Aurora XPU is required for an IPC barrier")
        self._group = group
        self._world_size = dist.get_world_size(group)
        self._rank = dist.get_rank(group)
        self._trace_enabled = ipc_trace_enabled()
        if self._trace_enabled:
            emit_ipc_trace(
                self._rank,
                "readiness",
                "init",
                "enter",
                {"world_size": self._world_size},
            )
        if self._world_size < 2 or self._world_size > 256:
            raise ValueError("IPC device barriers support world sizes in [2, 256]")
        self._device_index = torch.xpu.current_device()
        self._device = torch.device("xpu", self._device_index)
        base = socket_path or os.environ.get("AURORA_MOE_L0_IPC_BARRIER_SOCKET")
        if not base:
            raise RuntimeError("set AURORA_MOE_L0_IPC_BARRIER_SOCKET for IPC descriptor exchange")
        self._socket_base = Path(base)
        self._ipc_bias, self._ipc_bias_flags = self._mapping_policy()
        self._import_backend = self._import_policy()
        self._pointer_mode = self._pointer_policy()
        self._ops = load_level_zero_ipc_ops()
        self._barrier_ops = load_level_zero_ipc_device_barrier_ops()
        if self._trace_enabled:
            emit_ipc_trace(self._rank, "readiness", "init", "post_ops_load")
        if int(self._barrier_ops.slots) != _SLOTS or int(
            self._barrier_ops.counter_stride_words
        ) != _COUNTER_STRIDE_WORDS:
            raise RuntimeError("IPC barrier Python and SYCL control layouts disagree")
        self._control_storage: torch.Tensor | None = None
        self._local_counters: torch.Tensor | None = None
        self._remote_counter_ptrs: torch.Tensor | None = None
        self._remote_counter_addresses: tuple[int, ...] = ()
        self._exporter: object | None = None
        self._imports: list[object | None] = []
        self._descriptors: list[_Descriptor] = []
        self._pidfd_parent_tracer_enabled: bool | None = None
        self._ordinal = 0
        self._epoch = 0
        self._closed = False
        self._create_epoch()
        if self._trace_enabled:
            emit_ipc_trace(
                self._rank,
                "readiness",
                "init",
                "exit",
                {"epoch": self._epoch, "ordinal": self._ordinal},
            )

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
    def _pointer_policy() -> str:
        """Choose the isolated barrier's peer-pointer representation.

        ``device_table`` retains the original XPU-resident int64 table for
        diagnostics.  ``host_capture`` captures a fixed-size host-side peer
        pointer array in the SYCL kernel, matching oneCCL's normal pattern on
        Aurora (2--12 same-node ranks).
        """

        policy = os.environ.get(
            "AURORA_MOE_L0_IPC_BARRIER_POINTER_MODE", "device_table"
        )
        if policy not in {"device_table", "host_capture"}:
            raise ValueError(
                "AURORA_MOE_L0_IPC_BARRIER_POINTER_MODE must be "
                "device_table or host_capture"
            )
        return policy

    @staticmethod
    def _signed_pointer(address: int) -> int:
        return address - (1 << 64) if address >= 1 << 63 else address

    def _barrier(self) -> None:
        dist.barrier(group=self._group)

    def _epoch_path(self) -> Path:
        path = Path(f"{self._socket_base}.barrier.epoch{self._epoch}")
        if len(os.fsencode(path)) >= 108:
            raise RuntimeError(f"IPC socket path is too long: {path}")
        return path

    def _allocate_counters(self) -> tuple[torch.Tensor, torch.Tensor]:
        storage = torch.zeros(
            _SLOTS * _COUNTER_STRIDE_WORDS + _COUNTER_STRIDE_WORDS,
            dtype=torch.int64,
            device=self._device,
        )
        offset_bytes = (-storage.data_ptr()) % _CACHE_LINE_BYTES
        if offset_bytes % _COUNTER_WORD_BYTES:
            raise RuntimeError("could not align IPC control storage to an int64 boundary")
        offset_words = offset_bytes // _COUNTER_WORD_BYTES
        local = storage.narrow(0, offset_words, _SLOTS * _COUNTER_STRIDE_WORDS).view(
            _SLOTS, _COUNTER_STRIDE_WORDS
        )
        if not local.is_contiguous() or local.data_ptr() % _CACHE_LINE_BYTES:
            raise RuntimeError("could not create cache-line-aligned IPC counters")
        return storage, local

    def _create_epoch(self) -> None:
        if self._trace_enabled:
            emit_ipc_trace(
                self._rank,
                "readiness",
                "create_epoch",
                "enter",
                {"epoch": self._epoch},
            )
        self._barrier()
        self._epoch += 1
        storage, local = self._allocate_counters()
        if self._import_backend == "pidfd":
            # Match oneCCL's parent-only ptrace grant before duplicating IPC
            # descriptors through pidfd_getfd.  SCM exchange does not need it.
            self._pidfd_parent_tracer_enabled = bool(
                self._ops.enable_pidfd_parent_tracer()
            )
        exporter = self._ops.IpcExport(local)
        own = _Descriptor(
            exporter.fd(),
            os.getpid(),
            exporter.tensor_offset_bytes(),
            exporter.handle_bytes(),
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
                imports[source] = self._ops.IpcImport(
                    descriptor.handle,
                    descriptor.take_fd() if self._import_backend == "scm" else -1,
                    self._device_index,
                    False,
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
                address = local.data_ptr()
            else:
                imported = imports[source]
                if imported is None:
                    raise RuntimeError(f"missing IPC control mapping for rank {source}")
                address = int(imported.mapping_address()) + descriptor.tensor_offset_bytes
            if address % _CACHE_LINE_BYTES:
                raise RuntimeError(
                    f"IPC control mapping for rank {source} is not cache-line aligned"
                )
            addresses.append(self._signed_pointer(address))

        self._control_storage = storage
        self._local_counters = local
        self._remote_counter_addresses = tuple(addresses)
        self._remote_counter_ptrs = torch.tensor(
            addresses, dtype=torch.int64, device=self._device
        )
        self._exporter = exporter
        self._imports = imports
        self._descriptors = descriptors
        torch.xpu.synchronize(self._device)
        self._barrier()
        # Exporting a freshly allocated PyTorch tensor does not by itself make
        # its earlier asynchronous zero visible through every newly opened IPC
        # alias.  Reinitialize the local controls only after all ranks have
        # opened their persistent mappings, then complete one setup-only host
        # rendezvous.  The steady-state enqueue path has no host barrier.
        self.local_counters.zero_()
        torch.xpu.synchronize(self._device)
        self._barrier()
        if self._trace_enabled:
            emit_ipc_trace(
                self._rank,
                "readiness",
                "create_epoch",
                "exit",
                {"epoch": self._epoch},
            )

    def _close_imports(self) -> None:
        for imported in self._imports:
            if imported is not None:
                imported.close()
        self._imports = []
        for descriptor in self._descriptors:
            if descriptor.fd >= 0 and not descriptor.exporter_owned_fd:
                descriptor.close_received_fd()
        self._descriptors = []
        self._remote_counter_ptrs = None
        self._remote_counter_addresses = ()

    def _close_exporter(self) -> None:
        if self._exporter is not None:
            self._exporter.close()
        self._exporter = None
        self._local_counters = None
        self._control_storage = None

    @property
    def ordinal(self) -> int:
        """Next monotonically increasing barrier ordinal."""

        return self._ordinal

    @property
    def epoch(self) -> int:
        """Persistent control-allocation generation."""

        return self._epoch

    @property
    def ipc_bias(self) -> str:
        """Level Zero mapping policy used by the imported control buffers."""

        return self._ipc_bias

    @property
    def import_backend(self) -> str:
        """Descriptor import mechanism used by the control epoch."""

        return self._import_backend

    @property
    def pointer_mode(self) -> str:
        """Peer-pointer representation used by normal barrier enqueue calls."""

        return self._pointer_mode

    @property
    def local_counters(self) -> torch.Tensor:
        """The local [3, 8] control tensor retained by this epoch."""

        if self._local_counters is None:
            raise RuntimeError("IPC device barrier is closed")
        return self._local_counters

    @property
    def remote_counter_ptrs(self) -> torch.Tensor:
        """Persistent XPU pointer table, ordered by group rank."""

        if self._remote_counter_ptrs is None:
            raise RuntimeError("IPC device barrier is closed")
        return self._remote_counter_ptrs

    @property
    def remote_counter_addresses(self) -> tuple[int, ...]:
        """Host copy of the persistent peer-pointer table, ordered by rank."""

        if not self._remote_counter_addresses:
            raise RuntimeError("IPC device barrier is closed")
        return self._remote_counter_addresses

    def enqueue(self) -> int:
        """Enqueue the next device barrier on the current XPU stream."""

        if self._closed:
            raise RuntimeError("IPC device barrier is closed")
        ordinal = self._ordinal
        if self._trace_enabled:
            emit_ipc_trace(
                self._rank,
                "readiness",
                "enqueue",
                "enter",
                {"ordinal": ordinal, "pointer_mode": self._pointer_mode},
            )
        if self._pointer_mode == "device_table":
            self._barrier_ops.enqueue_ipc_device_barrier(
                self.local_counters,
                self.remote_counter_ptrs,
                self._world_size,
                ordinal,
            )
        else:
            self._barrier_ops.enqueue_ipc_device_barrier_host_ptrs(
                self.local_counters,
                self.remote_counter_addresses,
                self._world_size,
                self._rank,
                ordinal,
            )
        self._ordinal += 1
        if self._trace_enabled:
            emit_ipc_trace(
                self._rank,
                "readiness",
                "enqueue",
                "exit",
                {"ordinal": ordinal, "next_ordinal": self._ordinal},
            )
        return ordinal

    def _all_gather_control_values(self, values: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
        local = torch.tensor(values, dtype=torch.int64, device=self._device)
        gathered = [torch.empty_like(local) for _ in range(self._world_size)]
        dist.all_gather(gathered, local, group=self._group)
        return tuple(
            tuple(int(value) for value in item.cpu().tolist()) for item in gathered
        )

    def validate_device_pointer_table(self) -> tuple[tuple[int, ...], ...]:
        """Prove the XPU pointer table reaches each rank's local control word.

        This is a setup-only diagnostic.  Each sender is activated alone and
        atomically stamps every target's unused ``[0, 1]`` word.  The gathered
        rows are indexed by sender and contain the marker observed on every
        target rank.  Normal three-slot counter words are never modified.
        """

        if self._closed:
            raise RuntimeError("IPC device barrier is closed")
        rows: list[tuple[int, ...]] = []
        for sender in range(self._world_size):
            self._barrier()
            self._barrier_ops.stamp_ipc_device_barrier_device_ptrs(
                self.local_counters,
                self.remote_counter_ptrs,
                self._world_size,
                self._rank == sender,
                sender + 1,
            )
            torch.xpu.synchronize(self._device)
            self._barrier()
            torch.xpu.synchronize(self._device)
            marker = int(self.local_counters[0, _DIAGNOSTIC_STAMP_WORD].cpu().item())
            gathered_markers = self._all_gather_control_values((marker,))
            rows.append(tuple(values[0] for values in gathered_markers))

        self._barrier()
        self.local_counters[:, _DIAGNOSTIC_STAMP_WORD].zero_()
        torch.xpu.synchronize(self._device)
        self._barrier()
        failures = [
            sender
            for sender, row in enumerate(rows)
            if row != (sender + 1,) * self._world_size
        ]
        if failures:
            raise AssertionError(
                "XPU IPC pointer-table stamp mismatch for senders "
                f"{failures}: observed={rows}"
            )
        return tuple(rows)

    def diagnose_device_table_arrivals(
        self, barriers: int
    ) -> tuple[tuple[int, int, int], ...]:
        """Count remote arrivals without waiting, then restore zeroed controls.

        It distinguishes pointer/atomic delivery faults from a device-side
        wait deadlock because no rank spins in this diagnostic path.
        """

        if self._closed:
            raise RuntimeError("IPC device barrier is closed")
        if barriers <= 0:
            raise ValueError("arrival diagnostic requires a positive barrier count")
        if self._ordinal:
            raise RuntimeError("arrival diagnostic must run before normal barriers")
        self._barrier()
        self.local_counters.zero_()
        torch.xpu.synchronize(self._device)
        self._barrier()
        for ordinal in range(barriers):
            self._barrier_ops.enqueue_ipc_device_barrier_arrive_only(
                self.local_counters,
                self.remote_counter_ptrs,
                self._world_size,
                ordinal % _SLOTS,
            )
        torch.xpu.synchronize(self._device)
        self._barrier()
        torch.xpu.synchronize(self._device)
        local = tuple(int(value) for value in self.local_counters[:, 0].cpu().tolist())
        observed = self._all_gather_control_values(local)
        expected = self._expected_counters_for_count(barriers)

        self._barrier()
        self.local_counters.zero_()
        torch.xpu.synchronize(self._device)
        self._barrier()
        failures = [rank for rank, values in enumerate(observed) if values != expected]
        if failures:
            raise AssertionError(
                "XPU IPC arrive-only counter mismatch for ranks "
                f"{failures}: observed={observed}, expected={expected}"
            )
        return observed

    def _expected_counters_for_count(self, count: int) -> tuple[int, int, int]:
        return tuple(
            self._world_size * ((count + (_SLOTS - 1 - slot)) // _SLOTS)
            for slot in range(_SLOTS)
        )

    def expected_counters(self) -> tuple[int, int, int]:
        """Return exact counter values expected after all enqueued barriers finish."""

        return self._expected_counters_for_count(self._ordinal)

    def validate_after_synchronize(
        self, *, strict: bool = True
    ) -> tuple[int, int, int]:
        """Synchronize the current stream and validate all local monotonic counters."""

        torch.xpu.synchronize(self._device)
        values = tuple(int(value) for value in self.local_counters[:, 0].cpu().tolist())
        expected = self.expected_counters()
        if strict and values != expected:
            raise AssertionError(f"IPC barrier counters are {values}, expected {expected}")
        return values

    def close(self) -> None:
        """Collectively retire the persistent mappings after stream completion."""

        if self._closed:
            return
        self._barrier()
        torch.xpu.synchronize(self._device)
        self._close_imports()
        self._barrier()
        self._close_exporter()
        self._barrier()
        self._closed = True
