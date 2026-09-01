"""Synchronous same-node equal all-to-all over Level Zero IPC mappings."""

from __future__ import annotations

import array
from dataclasses import dataclass
import os
from pathlib import Path
import socket
import struct
import time

import torch
import torch.distributed as dist

from .level_zero_ipc import load_level_zero_ipc_ops


_HANDLE_BYTES = 64
# Rank, exporter PID, and tensor offset.  The PID lets the opt-in pidfd
# transport duplicate the original descriptor encoded in the Level Zero IPC
# handle without passing an SCM_RIGHTS file descriptor through this socket.
_DESCRIPTOR_HEADER = struct.Struct("!IQQ")


@dataclass
class _Descriptor:
    fd: int
    pid: int
    tensor_offset_bytes: int
    handle: bytes
    exporter_owned_fd: bool = False

    def take_fd(self) -> int:
        """Transfer this received file descriptor to an IpcImport."""

        if self.fd < 0:
            raise RuntimeError("IPC descriptor file descriptor was already consumed")
        fd = self.fd
        self.fd = -1
        return fd

    def close_received_fd(self) -> None:
        """Close a descriptor that is not owned by the local exporter."""

        if self.exporter_owned_fd:
            return
        if self.fd < 0:
            raise RuntimeError("IPC descriptor file descriptor was already consumed")
        os.close(self.fd)
        self.fd = -1


def _send_descriptor(
    connection: socket.socket,
    rank: int,
    descriptor: _Descriptor,
    *,
    send_fd: bool = True,
) -> None:
    if len(descriptor.handle) != _HANDLE_BYTES:
        raise RuntimeError(f"unexpected Level Zero IPC handle length {len(descriptor.handle)}")
    if descriptor.pid <= 0:
        raise RuntimeError(f"invalid IPC descriptor exporter PID {descriptor.pid}")
    message = _DESCRIPTOR_HEADER.pack(
        rank, descriptor.pid, descriptor.tensor_offset_bytes
    ) + descriptor.handle
    if send_fd:
        if descriptor.fd < 0:
            raise RuntimeError("SCM_RIGHTS IPC descriptor is missing its file descriptor")
        rights = array.array("i", [descriptor.fd])
        sent = connection.sendmsg(
            [message], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, rights.tobytes())]
        )
    else:
        sent = connection.sendmsg([message])
    if sent != len(message):
        raise RuntimeError(f"short Unix IPC send: {sent} of {len(message)} bytes")


def _recv_descriptor(
    connection: socket.socket, *, expect_fd: bool = True
) -> tuple[int, _Descriptor]:
    expected = _DESCRIPTOR_HEADER.size + _HANDLE_BYTES
    message, ancillary, flags, _ = connection.recvmsg(
        expected, socket.CMSG_SPACE(array.array("i").itemsize)
    )
    if len(message) != expected or flags & (socket.MSG_CTRUNC | socket.MSG_TRUNC):
        raise RuntimeError("malformed Unix IPC descriptor message")
    received: list[int] = []
    for level, kind, payload in ancillary:
        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
            fds = array.array("i")
            fds.frombytes(payload[: len(payload) - len(payload) % fds.itemsize])
            received.extend(fds)
    if expect_fd:
        if len(received) != 1:
            raise RuntimeError(f"expected one Unix IPC descriptor, got {len(received)}")
        fd = received[0]
    else:
        if received:
            for fd in received:
                os.close(fd)
            raise RuntimeError("pidfd IPC descriptor unexpectedly carried SCM_RIGHTS data")
        fd = -1
    rank, pid, offset = _DESCRIPTOR_HEADER.unpack(message[: _DESCRIPTOR_HEADER.size])
    if pid <= 0:
        raise RuntimeError(f"invalid IPC descriptor exporter PID {pid}")
    return rank, _Descriptor(fd, pid, offset, message[_DESCRIPTOR_HEADER.size :])


def _root_exchange(
    path: Path, world_size: int, own: _Descriptor, *, send_fd: bool = True
) -> list[_Descriptor]:
    path.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    server.bind(str(path))
    server.listen(world_size - 1)
    server.settimeout(60.0)
    descriptors: dict[int, _Descriptor] = {0: own}
    clients: dict[int, socket.socket] = {}
    try:
        while len(clients) < world_size - 1:
            connection, _ = server.accept()
            rank, descriptor = _recv_descriptor(connection, expect_fd=send_fd)
            if rank == 0 or rank >= world_size or rank in clients:
                if descriptor.fd >= 0:
                    descriptor.close_received_fd()
                connection.close()
                raise RuntimeError(f"invalid duplicate Unix IPC rank {rank}")
            clients[rank] = connection
            descriptors[rank] = descriptor
        for connection in clients.values():
            for rank in range(world_size):
                _send_descriptor(connection, rank, descriptors[rank], send_fd=send_fd)
            connection.close()
    finally:
        server.close()
        path.unlink(missing_ok=True)
    if len(descriptors) != world_size:
        raise RuntimeError("Unix IPC descriptor exchange is incomplete")
    return [descriptors[rank] for rank in range(world_size)]


def _client_exchange(
    path: Path,
    rank: int,
    world_size: int,
    own: _Descriptor,
    *,
    send_fd: bool = True,
) -> list[_Descriptor]:
    deadline = time.monotonic() + 60.0
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    while True:
        try:
            connection.connect(str(path))
            break
        except FileNotFoundError:
            if time.monotonic() >= deadline:
                connection.close()
                raise RuntimeError("timed out waiting for the Unix IPC server")
            time.sleep(0.05)
        except ConnectionRefusedError:
            if time.monotonic() >= deadline:
                connection.close()
                raise RuntimeError("Unix IPC server refused connections for 60 seconds")
            time.sleep(0.05)
    try:
        _send_descriptor(connection, rank, own, send_fd=send_fd)
        descriptors: list[_Descriptor | None] = [None] * world_size
        for _ in range(world_size):
            source, descriptor = _recv_descriptor(connection, expect_fd=send_fd)
            if source >= world_size or descriptors[source] is not None:
                if descriptor.fd >= 0:
                    descriptor.close_received_fd()
                raise RuntimeError(f"invalid Unix IPC source rank {source}")
            descriptors[source] = descriptor
    finally:
        connection.close()
    if any(descriptor is None for descriptor in descriptors):
        raise RuntimeError("Unix IPC descriptor exchange is incomplete")
    return [descriptor for descriptor in descriptors if descriptor is not None]


class LevelZeroIpcEqualAllToAll:
    """Exact synchronous BF16 equal all-to-all with persistent IPC staging.

    All ranks in ``group`` must be on one host and call ``exchange`` in the
    same order.  An exchange copies an input shaped ``[world_size, ...]``;
    output source row ``s`` receives input row ``rank`` from source ``s``.
    Staging capacity grows geometrically when needed, re-exporting only at a
    global host-synchronized boundary.  It never drops, clips, or pads routes.

    This is deliberately a correctness/control transport: producer staging,
    remote copies, and each reuse boundary are host synchronized.  Set
    ``parallel_copy_streams=True`` only to test whether independent PyTorch
    queues improve the serial ``queue.memcpy`` control; it does not make a
    claim about a fused peer-copy kernel.  The peer mappings and their file
    descriptors stay live until ``close`` or a growth epoch, and no process
    may destroy the transport while another rank uses it.
    """

    def __init__(
        self,
        group: dist.ProcessGroup | None = None,
        *,
        socket_path: str | os.PathLike[str] | None = None,
        parallel_copy_streams: bool = False,
    ) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("initialize torch.distributed before IPC all-to-all")
        if not torch.xpu.is_available():
            raise RuntimeError("Aurora XPU is required for IPC all-to-all")
        self._group = group
        self._world_size = dist.get_world_size(group)
        self._rank = dist.get_rank(group)
        if self._world_size <= 1:
            raise ValueError("IPC all-to-all requires at least two same-node ranks")
        device_index = torch.xpu.current_device()
        self._device = torch.device("xpu", device_index)
        self._device_index = device_index
        base = socket_path or os.environ.get("AURORA_MOE_L0_IPC_SOCKET")
        if not base:
            raise RuntimeError("set AURORA_MOE_L0_IPC_SOCKET for IPC descriptor exchange")
        self._socket_base = Path(base)
        self._ops = load_level_zero_ipc_ops()
        self._staging: torch.Tensor | None = None
        self._exporter: object | None = None
        self._imports: list[object | None] = []
        self._descriptors: list[_Descriptor] = []
        self._capacity_elements = 0
        self._dtype: torch.dtype | None = None
        self._epoch = 0
        self._closed = False
        self._parallel_copy_streams = parallel_copy_streams
        self._copy_streams = (
            {
                source: torch.xpu.Stream(device=self._device)
                for source in range(self._world_size)
                if source != self._rank
            }
            if parallel_copy_streams
            else {}
        )

    @property
    def capacity_elements(self) -> int:
        """Current exact per-peer staging capacity in BF16 elements."""

        return self._capacity_elements

    @property
    def epoch(self) -> int:
        """Number of exported staging epochs created so far."""

        return self._epoch

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
        path = Path(f"{self._socket_base}.epoch{self._epoch}")
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

    def _close_exporter(self) -> None:
        if self._exporter is not None:
            self._exporter.close()
        self._exporter = None
        self._staging = None
        self._capacity_elements = 0

    def _retire_epoch(self) -> None:
        self._close_imports()
        # No exporter handle may be released until every peer has retired the
        # mapping it opened from that handle.
        self._barrier()
        self._close_exporter()
        self._barrier()

    def _create_epoch(self, capacity_elements: int, dtype: torch.dtype) -> None:
        if capacity_elements <= 0:
            raise ValueError("IPC staging capacity must be positive")
        self._barrier()
        torch.xpu.synchronize(self._device)
        self._retire_epoch()
        self._epoch += 1
        self._staging = torch.empty(
            (self._world_size, capacity_elements), dtype=dtype, device=self._device
        )
        self._exporter = self._ops.IpcExport(self._staging)
        own = _Descriptor(
            self._exporter.fd(),
            os.getpid(),
            self._exporter.tensor_offset_bytes(),
            self._exporter.handle_bytes(),
            exporter_owned_fd=True,
        )
        path = self._epoch_path()
        if self._rank == 0:
            descriptors = _root_exchange(path, self._world_size, own)
        else:
            descriptors = _client_exchange(path, self._rank, self._world_size, own)
        self._descriptors = descriptors
        imports: list[object | None] = [None] * self._world_size
        for source, descriptor in enumerate(descriptors):
            if source != self._rank:
                imports[source] = self._ops.IpcImport(
                    descriptor.handle, descriptor.take_fd(), self._device_index
                )
        own_descriptor = descriptors[self._rank]
        if own_descriptor.exporter_owned_fd:
            if self._rank != 0:
                raise RuntimeError("only root may retain its exporter-owned descriptor")
        else:
            own_descriptor.close_received_fd()
        self._imports = imports
        self._capacity_elements = capacity_elements
        self._dtype = dtype
        self._barrier()

    def _ensure_capacity(self, elements_per_peer: int, dtype: torch.dtype) -> None:
        if self._closed:
            raise RuntimeError("IPC all-to-all is closed")
        self._matching_value(elements_per_peer, "elements per peer")
        if dtype != torch.bfloat16:
            raise ValueError("IPC all-to-all currently supports BF16 payloads only")
        if self._dtype is not None and dtype != self._dtype:
            raise ValueError("IPC all-to-all payload dtype cannot change after initialization")
        if elements_per_peer > self._capacity_elements:
            grown = max(elements_per_peer, max(1, self._capacity_elements * 2))
            self._create_epoch(grown, dtype)

    def exchange(self, input: torch.Tensor, output: torch.Tensor | None = None) -> torch.Tensor:
        """Synchronously exchange contiguous BF16 ``[world_size, ...]`` payloads."""

        if input.device != self._device or not input.is_contiguous():
            raise ValueError("input must be a contiguous tensor on the local XPU tile")
        if input.ndim < 1 or input.size(0) != self._world_size:
            raise ValueError(f"input must have leading dimension {self._world_size}")
        if input.dtype != torch.bfloat16:
            raise ValueError("input must use BF16")
        elements_per_peer = input.numel() // self._world_size
        if elements_per_peer <= 0:
            raise ValueError("input must contain at least one element per peer")
        if output is None:
            output = torch.empty_like(input)
        if (
            output.device != input.device
            or output.dtype != input.dtype
            or output.shape != input.shape
            or not output.is_contiguous()
        ):
            raise ValueError("output must be contiguous and match input device, dtype, and shape")
        self._ensure_capacity(elements_per_peer, input.dtype)
        if self._staging is None or self._capacity_elements < elements_per_peer:
            raise RuntimeError("IPC staging was not initialized")
        input_rows = input.reshape(self._world_size, elements_per_peer)
        output_rows = output.reshape(self._world_size, elements_per_peer)
        self._staging[:, :elements_per_peer].copy_(input_rows)
        torch.xpu.synchronize(self._device)
        self._barrier()
        byte_count = elements_per_peer * input.element_size()
        output_rows[self._rank].copy_(self._staging[self._rank, :elements_per_peer])
        for source in range(self._world_size):
            if source == self._rank:
                continue
            item = self._imports[source]
            if item is None:
                raise RuntimeError(f"missing IPC mapping for source rank {source}")
            descriptor = self._descriptors[source]
            source_offset = (
                descriptor.tensor_offset_bytes
                + self._rank * self._capacity_elements * input.element_size()
            )
            if self._parallel_copy_streams:
                with torch.xpu.stream(self._copy_streams[source]):
                    item.copy_to(output_rows[source], source_offset, byte_count)
            else:
                item.copy_to(output_rows[source], source_offset, byte_count)
        torch.xpu.synchronize(self._device)
        for source, item in enumerate(self._imports):
            if item is not None:
                if self._parallel_copy_streams:
                    with torch.xpu.stream(self._copy_streams[source]):
                        item.synchronize()
                else:
                    item.synchronize()
        # This prevents a fast rank from overwriting its exported staging rows
        # while a peer is still reading them in the current exchange.
        self._barrier()
        return output

    def close(self) -> None:
        """Collectively retire mappings after all exchanges have completed."""

        if self._closed:
            return
        self._barrier()
        torch.xpu.synchronize(self._device)
        self._retire_epoch()
        self._closed = True
