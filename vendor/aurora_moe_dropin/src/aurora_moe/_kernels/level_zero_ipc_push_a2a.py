"""Persistent Level Zero IPC push all-to-all for isolated diagnostics."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import time

import torch
import torch.distributed as dist

from ._ipc_trace import emit_ipc_trace, ipc_trace_enabled
from .level_zero_ipc import load_level_zero_ipc_ops
from .level_zero_ipc_equal_a2a import _Descriptor, _client_exchange, _root_exchange
from .level_zero_ipc_push import load_level_zero_ipc_push_ops


@dataclass(frozen=True)
class PushExchangeTimes:
    """Host-measured steady-state stages of one synchronous IPC exchange."""

    pack_seconds: float
    producer_barrier_seconds: float
    push_kernel_seconds: float
    consumer_barrier_seconds: float
    unpack_seconds: float


class LevelZeroIpcPushAllToAll:
    """Exact same-node BF16 equal all-to-all using a persistent IPC push kernel.

    The class owns a local send staging tensor and an exported receive staging
    tensor shaped ``[world_size, capacity]``.  Each rank pushes one local send
    row to every rank's receive row.  The descriptor exchange happens only on
    allocation growth, never in the steady-state exchange path.
    """

    def __init__(
        self,
        group: dist.ProcessGroup | None = None,
        *,
        socket_path: str | os.PathLike[str] | None = None,
    ) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("initialize torch.distributed before IPC all-to-all")
        if not torch.xpu.is_available():
            raise RuntimeError("Aurora XPU is required for IPC all-to-all")
        self._group = group
        self._world_size = dist.get_world_size(group)
        self._rank = dist.get_rank(group)
        self._trace_enabled = ipc_trace_enabled()
        if self._world_size <= 1:
            raise ValueError("IPC all-to-all requires at least two same-node ranks")
        self._device_index = torch.xpu.current_device()
        self._device = torch.device("xpu", self._device_index)
        base = socket_path or os.environ.get("AURORA_MOE_L0_IPC_PUSH_SOCKET")
        if not base:
            raise RuntimeError("set AURORA_MOE_L0_IPC_PUSH_SOCKET for IPC descriptor exchange")
        self._socket_base = Path(base)
        cache_policy = os.environ.get("AURORA_MOE_L0_IPC_BIAS", "cached")
        if cache_policy not in {"default", "cached", "uncached"}:
            raise ValueError(
                "AURORA_MOE_L0_IPC_BIAS must be default, cached, or uncached"
            )
        # Level Zero treats zero as a distinct no-bias/default setting; it is
        # the flag value used by oneCCL's IPC mapping path.  Keep cached and
        # uncached as explicit A/B controls rather than conflating default
        # with cached.
        self._ipc_bias = cache_policy
        self._ipc_bias_flags = {"default": 0, "cached": 1, "uncached": 2}[cache_policy]
        self._ipc_bias_uncached = cache_policy == "uncached"
        copy_impl = os.environ.get("AURORA_MOE_L0_IPC_PUSH_IMPL", "fast")
        if copy_impl not in {"fast", "ccl_equiv", "ccl_typed"}:
            raise ValueError(
                "AURORA_MOE_L0_IPC_PUSH_IMPL must be fast, ccl_equiv, or ccl_typed"
            )
        self._copy_impl = copy_impl
        direct_input = os.environ.get("AURORA_MOE_L0_IPC_DIRECT_INPUT", "0")
        if direct_input not in {"0", "1"}:
            raise ValueError("AURORA_MOE_L0_IPC_DIRECT_INPUT must be 0 or 1")
        # The direct-input path is opt-in until it has been validated in the
        # full MoE dispatcher.  It removes the transient local pack copy but
        # records the source allocation on the bridged current stream so an
        # OOO peer kernel cannot outlive its storage.
        self._direct_input = direct_input == "1"
        persistent_ooo_queue = os.environ.get(
            "AURORA_MOE_L0_IPC_PERSISTENT_OOO_QUEUE", "0"
        )
        if persistent_ooo_queue not in {"0", "1"}:
            raise ValueError("AURORA_MOE_L0_IPC_PERSISTENT_OOO_QUEUE must be 0 or 1")
        self._persistent_ooo_queue = persistent_ooo_queue == "1"
        if self._persistent_ooo_queue and self._copy_impl != "ccl_typed":
            raise ValueError(
                "AURORA_MOE_L0_IPC_PERSISTENT_OOO_QUEUE=1 requires "
                "AURORA_MOE_L0_IPC_PUSH_IMPL=ccl_typed"
            )
        import_backend = os.environ.get("AURORA_MOE_L0_IPC_IMPORT_BACKEND", "scm")
        if import_backend not in {"scm", "pidfd"}:
            raise ValueError("AURORA_MOE_L0_IPC_IMPORT_BACKEND must be scm or pidfd")
        self._import_backend = import_backend
        self._pidfd_parent_tracer_enabled: bool | None = None
        self._ops = load_level_zero_ipc_ops()
        self._push_ops = load_level_zero_ipc_push_ops()
        self._persistent_push_queue: object | None = None
        if self._persistent_ooo_queue:
            queue_type = getattr(self._push_ops, "PersistentOooTypedPushQueue", None)
            if queue_type is None:
                raise RuntimeError(
                    "AURORA_MOE_L0_IPC_PERSISTENT_OOO_QUEUE=1 requires a rebuilt "
                    "level_zero_ipc_push extension with PersistentOooTypedPushQueue"
                )
            self._persistent_push_queue = queue_type()
        self._send_staging: torch.Tensor | None = None
        self._receive_staging: torch.Tensor | None = None
        self._remote_receive_ptrs: torch.Tensor | None = None
        self._remote_receive_addresses: list[int] = []
        self._exporter: object | None = None
        self._imports: list[object | None] = []
        self._descriptors: list[_Descriptor] = []
        self._capacity_elements = 0
        self._dtype: torch.dtype | None = None
        self._epoch = 0
        self._closed = False

    @property
    def capacity_elements(self) -> int:
        """Per-peer staging capacity, rounded only for internal vector alignment."""

        return self._capacity_elements

    @property
    def epoch(self) -> int:
        """Number of exported receive-staging epochs created so far."""

        return self._epoch

    @property
    def ipc_bias(self) -> str:
        """Level Zero IPC mapping cache policy for this transport instance."""

        return self._ipc_bias

    @property
    def ipc_bias_flags(self) -> int:
        """Exact ``zeMemOpenIpcHandle`` flag value used for peer mappings."""

        return self._ipc_bias_flags

    @property
    def copy_impl(self) -> str:
        """Selected isolated peer-copy implementation."""

        return self._copy_impl

    @property
    def import_backend(self) -> str:
        """Descriptor-to-mapping import mechanism used by this instance."""

        return self._import_backend

    @property
    def direct_input(self) -> bool:
        """Whether async typed pushes read the caller input without a local pack copy."""

        return self._direct_input

    @property
    def persistent_ooo_queue(self) -> bool:
        """Whether typed async pushes retain one OOO SYCL queue per host thread."""

        return self._persistent_ooo_queue

    @property
    def pidfd_parent_tracer(self) -> str:
        """Whether oneCCL-style parent tracer setup succeeded for this epoch."""

        if self._import_backend != "pidfd":
            return "na"
        return "1" if self._pidfd_parent_tracer_enabled else "0"

    def _barrier(self) -> None:
        dist.barrier(group=self._group)

    def _matching_value(self, value: int, name: str) -> None:
        control = torch.tensor([value], dtype=torch.int64, device=self._device)
        gathered = [torch.empty_like(control) for _ in range(self._world_size)]
        dist.all_gather(gathered, control, group=self._group)
        values = [int(item.item()) for item in gathered]
        if any(item != value for item in values):
            raise ValueError(f"{name} must match across ranks, got {values}")

    def _epoch_path(self) -> Path:
        path = Path(f"{self._socket_base}.push.epoch{self._epoch}")
        if len(os.fsencode(path)) >= 108:
            raise RuntimeError(f"IPC socket path is too long: {path}")
        return path

    def _close_imports(self) -> None:
        for item in self._imports:
            if item is not None:
                item.close()
        self._imports = []
        for descriptor in self._descriptors:
            if descriptor.fd >= 0 and not descriptor.exporter_owned_fd:
                descriptor.close_received_fd()
        self._descriptors = []
        self._remote_receive_ptrs = None
        self._remote_receive_addresses = []

    def _close_exporter(self) -> None:
        if self._exporter is not None:
            self._exporter.close()
        self._exporter = None
        self._send_staging = None
        self._receive_staging = None
        self._capacity_elements = 0

    def _retire_epoch(self) -> None:
        self._close_imports()
        self._barrier()
        self._close_exporter()
        self._barrier()

    @staticmethod
    def _signed_pointer(address: int) -> int:
        if address >= 1 << 63:
            return address - (1 << 64)
        return address

    def _create_epoch(self, capacity_elements: int, dtype: torch.dtype) -> None:
        if self._trace_enabled:
            emit_ipc_trace(
                self._rank,
                "payload",
                "create_epoch",
                "enter",
                {
                    "epoch": self._epoch,
                    "capacity_elements": capacity_elements,
                    "dtype": str(dtype),
                },
            )
        if capacity_elements <= 0 or capacity_elements % 2:
            raise ValueError("IPC push staging capacity must be positive and even")
        self._barrier()
        torch.xpu.synchronize(self._device)
        self._retire_epoch()
        self._epoch += 1
        self._send_staging = torch.empty(
            (self._world_size, capacity_elements), dtype=dtype, device=self._device
        )
        self._receive_staging = torch.empty_like(self._send_staging)
        if self._import_backend == "pidfd":
            # oneCCL does this during Level Zero initialization for its default
            # pidfd exchange mode.  It is intentionally the parent-only form,
            # not a broad PR_SET_PTRACER_ANY grant.
            self._pidfd_parent_tracer_enabled = bool(
                self._ops.enable_pidfd_parent_tracer()
            )
        self._exporter = self._ops.IpcExport(self._receive_staging)
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
                path,
                self._world_size,
                own,
                send_fd=self._import_backend == "scm",
            )
        else:
            descriptors = _client_exchange(
                path,
                self._rank,
                self._world_size,
                own,
                send_fd=self._import_backend == "scm",
            )
        self._descriptors = descriptors
        imports: list[object | None] = [None] * self._world_size
        for source, descriptor in enumerate(descriptors):
            if source != self._rank:
                imports[source] = self._ops.IpcImport(
                    descriptor.handle,
                    descriptor.take_fd() if self._import_backend == "scm" else -1,
                    self._device_index,
                    self._ipc_bias_uncached,
                    descriptor.pid,
                    self._import_backend == "pidfd",
                    self._ipc_bias_flags,
                )
        own_descriptor = descriptors[self._rank]
        if own_descriptor.exporter_owned_fd:
            if self._rank != 0:
                raise RuntimeError("only root may retain its exporter-owned descriptor")
        elif own_descriptor.fd >= 0:
            # The SCM transport returns a receiver-local duplicate of our own
            # descriptor through rank zero.  The pidfd transport deliberately
            # carries no SCM descriptor, so there is nothing to close here.
            own_descriptor.close_received_fd()
        addresses: list[int] = []
        for source, descriptor in enumerate(descriptors):
            if source == self._rank:
                address = self._receive_staging.data_ptr()
            else:
                item = imports[source]
                if item is None:
                    raise RuntimeError(f"missing IPC receive mapping for rank {source}")
                address = int(item.mapping_address()) + descriptor.tensor_offset_bytes
            if address % 4:
                raise RuntimeError(
                    f"IPC receive mapping for rank {source} is not 32-bit aligned"
                )
            addresses.append(self._signed_pointer(address))
        self._imports = imports
        self._remote_receive_ptrs = torch.tensor(
            addresses, dtype=torch.int64, device=self._device
        )
        self._remote_receive_addresses = addresses
        self._capacity_elements = capacity_elements
        self._dtype = dtype
        torch.xpu.synchronize(self._device)
        self._barrier()
        if self._trace_enabled:
            emit_ipc_trace(
                self._rank,
                "payload",
                "create_epoch",
                "exit",
                {
                    "epoch": self._epoch,
                    "capacity_elements": self._capacity_elements,
                },
            )

    def _ensure_capacity(self, elements_per_peer: int, dtype: torch.dtype) -> None:
        if self._trace_enabled:
            emit_ipc_trace(
                self._rank,
                "payload",
                "ensure_capacity",
                "enter",
                {
                    "elements_per_peer": elements_per_peer,
                    "capacity_elements": self._capacity_elements,
                    "epoch": self._epoch,
                },
            )
        if self._closed:
            raise RuntimeError("IPC push all-to-all is closed")
        self._matching_value(elements_per_peer, "elements per peer")
        if dtype != torch.bfloat16:
            raise ValueError("IPC push all-to-all currently supports BF16 payloads only")
        if self._dtype is not None and dtype != self._dtype:
            raise ValueError("IPC push all-to-all payload dtype cannot change after initialization")
        if elements_per_peer > self._capacity_elements:
            grown = max(elements_per_peer, max(1, self._capacity_elements * 2))
            grown += grown % 2
            self._create_epoch(grown, dtype)
        if self._trace_enabled:
            emit_ipc_trace(
                self._rank,
                "payload",
                "ensure_capacity",
                "exit",
                {
                    "elements_per_peer": elements_per_peer,
                    "capacity_elements": self._capacity_elements,
                    "epoch": self._epoch,
                },
            )

    def _output_for(self, input: torch.Tensor, output: torch.Tensor | None) -> torch.Tensor:
        if input.device != self._device or not input.is_contiguous():
            raise ValueError("input must be a contiguous tensor on the local XPU tile")
        if input.ndim < 1 or input.size(0) != self._world_size:
            raise ValueError(f"input must have leading dimension {self._world_size}")
        if input.dtype != torch.bfloat16:
            raise ValueError("input must use BF16")
        if output is None:
            return torch.empty_like(input)
        if (
            output.device != input.device
            or output.dtype != input.dtype
            or output.shape != input.shape
            or not output.is_contiguous()
        ):
            raise ValueError("output must be contiguous and match input device, dtype, and shape")
        return output

    def exchange(self, input: torch.Tensor, output: torch.Tensor | None = None) -> torch.Tensor:
        """Exchange contiguous BF16 ``[world_size, ...]`` payloads exactly."""

        received, _ = self.exchange_profiled(input, output)
        return received

    def exchange_async(
        self,
        input: torch.Tensor,
        output: torch.Tensor | None = None,
        *,
        pre_remote_write: Callable[[], None] | None = None,
        post_remote_write: Callable[[], None] | None = None,
    ) -> torch.Tensor:
        """Enqueue typed IPC push and local output copy without a host wait.

        ``pre_remote_write`` runs after the normal staging pack (or after the
        direct input's current-stream producer tail) and before the typed peer push;
        ``post_remote_write`` runs after the local completion bridge and
        before unpack.  They can enqueue device barriers without exposing
        staging buffers.  Without a post callback, callers must establish an
        incoming-write barrier before consuming remote output rows or reusing
        the staging epoch.
        Steady state deliberately performs no shape-control collective: all
        ranks must supply the same equal all-to-all shape.  Initial allocation
        and capacity growth remain collective/exact through ``_ensure_capacity``.
        """

        if self._trace_enabled:
            emit_ipc_trace(
                self._rank,
                "payload",
                "exchange_async",
                "enter",
                {
                    "input_elements": input.numel(),
                    "capacity_elements": self._capacity_elements,
                    "direct_input": int(self._direct_input),
                    "pre_callback": int(pre_remote_write is not None),
                    "post_callback": int(post_remote_write is not None),
                },
            )
        output = self._output_for(input, output)
        elements_per_peer = input.numel() // self._world_size
        if elements_per_peer <= 0:
            raise ValueError("input must contain at least one element per peer")
        if self._closed:
            raise RuntimeError("IPC push all-to-all is closed")
        if self._dtype is not None and input.dtype != self._dtype:
            raise ValueError("IPC push payload dtype cannot change after initialization")
        if elements_per_peer > self._capacity_elements:
            self._ensure_capacity(elements_per_peer, input.dtype)
        if self._trace_enabled:
            emit_ipc_trace(
                self._rank,
                "payload",
                "exchange_async",
                "post_capacity",
                {
                    "elements_per_peer": elements_per_peer,
                    "capacity_elements": self._capacity_elements,
                    "epoch": self._epoch,
                },
            )
        if (
            self._send_staging is None
            or self._receive_staging is None
            or self._capacity_elements < elements_per_peer
        ):
            raise RuntimeError("IPC push staging was not initialized")
        if self._copy_impl != "ccl_typed":
            raise RuntimeError(
                "exchange_async requires AURORA_MOE_L0_IPC_PUSH_IMPL=ccl_typed"
            )

        input_rows = input.reshape(self._world_size, elements_per_peer)
        output_rows = output.reshape(self._world_size, elements_per_peer)
        if not self._direct_input:
            self._send_staging[:, :elements_per_peer].copy_(input_rows)
        if self._trace_enabled:
            emit_ipc_trace(
                self._rank,
                "payload",
                "exchange_async",
                "post_pack",
                {"elements_per_peer": elements_per_peer},
            )
        if pre_remote_write is not None:
            pre_remote_write()
        if self._trace_enabled:
            emit_ipc_trace(
                self._rank,
                "payload",
                "exchange_async",
                "post_pre_callback",
                {"elements_per_peer": elements_per_peer},
            )
        if self._direct_input:
            if self._persistent_push_queue is None:
                self._push_ops.push_alltoall_bf16_ccl_typed_direct_async(
                    input_rows,
                    self._receive_staging,
                    self._remote_receive_addresses,
                    self._rank,
                    elements_per_peer,
                )
            else:
                self._persistent_push_queue.push_alltoall_bf16_ccl_typed_direct_async(
                    input_rows,
                    self._receive_staging,
                    self._remote_receive_addresses,
                    self._rank,
                    elements_per_peer,
                )
        else:
            if self._persistent_push_queue is None:
                self._push_ops.push_alltoall_bf16_ccl_typed_async(
                    self._send_staging,
                    self._receive_staging,
                    self._remote_receive_addresses,
                    self._rank,
                    elements_per_peer,
                )
            else:
                self._persistent_push_queue.push_alltoall_bf16_ccl_typed_async(
                    self._send_staging,
                    self._receive_staging,
                    self._remote_receive_addresses,
                    self._rank,
                    elements_per_peer,
                )
        if self._trace_enabled:
            emit_ipc_trace(
                self._rank,
                "payload",
                "exchange_async",
                "post_push",
                {"elements_per_peer": elements_per_peer},
            )
        if post_remote_write is not None:
            post_remote_write()
        if self._trace_enabled:
            emit_ipc_trace(
                self._rank,
                "payload",
                "exchange_async",
                "post_post_callback",
                {"elements_per_peer": elements_per_peer},
            )
        output_rows.copy_(self._receive_staging[:, :elements_per_peer])
        if self._trace_enabled:
            emit_ipc_trace(
                self._rank,
                "payload",
                "exchange_async",
                "exit",
                {"elements_per_peer": elements_per_peer},
            )
        return output

    def exchange_profiled(
        self, input: torch.Tensor, output: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, PushExchangeTimes]:
        """Exchange exactly and return synchronized stage timings for diagnostics."""

        output = self._output_for(input, output)
        elements_per_peer = input.numel() // self._world_size
        if elements_per_peer <= 0:
            raise ValueError("input must contain at least one element per peer")
        self._ensure_capacity(elements_per_peer, input.dtype)
        if (
            self._send_staging is None
            or self._receive_staging is None
            or self._remote_receive_ptrs is None
            or self._capacity_elements < elements_per_peer
        ):
            raise RuntimeError("IPC push staging was not initialized")
        input_rows = input.reshape(self._world_size, elements_per_peer)
        output_rows = output.reshape(self._world_size, elements_per_peer)

        start = time.perf_counter()
        self._send_staging[:, :elements_per_peer].copy_(input_rows)
        torch.xpu.synchronize(self._device)
        pack_seconds = time.perf_counter() - start

        start = time.perf_counter()
        self._barrier()
        producer_barrier_seconds = time.perf_counter() - start

        start = time.perf_counter()
        if self._copy_impl == "ccl_typed":
            self._push_ops.push_alltoall_bf16_ccl_typed(
                self._send_staging,
                self._receive_staging,
                self._remote_receive_addresses,
                self._rank,
                elements_per_peer,
            )
        elif self._copy_impl == "ccl_equiv":
            self._push_ops.push_alltoall_bf16_ccl_equiv(
                self._send_staging,
                self._receive_staging,
                self._remote_receive_addresses,
                self._rank,
                elements_per_peer,
            )
        elif self._world_size <= 12:
            self._push_ops.push_alltoall_bf16_fast(
                self._send_staging,
                self._receive_staging,
                self._remote_receive_addresses,
                self._rank,
                elements_per_peer,
            )
        else:
            self._push_ops.push_alltoall_bf16_pairs(
                self._send_staging,
                self._receive_staging,
                self._remote_receive_ptrs,
                self._rank,
                elements_per_peer,
            )
        torch.xpu.synchronize(self._device)
        push_kernel_seconds = time.perf_counter() - start

        start = time.perf_counter()
        self._barrier()
        consumer_barrier_seconds = time.perf_counter() - start

        start = time.perf_counter()
        output_rows.copy_(self._receive_staging[:, :elements_per_peer])
        torch.xpu.synchronize(self._device)
        unpack_seconds = time.perf_counter() - start
        return output, PushExchangeTimes(
            pack_seconds,
            producer_barrier_seconds,
            push_kernel_seconds,
            consumer_barrier_seconds,
            unpack_seconds,
        )

    def close(self) -> None:
        """Collectively retire mappings only after all exchanges complete."""

        if self._closed:
            return
        self._barrier()
        torch.xpu.synchronize(self._device)
        if self._persistent_push_queue is not None:
            self._persistent_push_queue.close()
            self._persistent_push_queue = None
        self._retire_epoch()
        self._closed = True
