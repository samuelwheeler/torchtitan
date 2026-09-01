"""Raw-SYCL control-allocation counterpart to the IPC device-barrier probe.

This isolated A/B path deliberately allocates the three shared counter words
with ``sycl::malloc_device`` instead of PyTorch storage.  It is useful for
comparing our control protocol with oneCCL's ``ptrs0`` setup without changing
the normal persistent barrier epoch used by the composed EP transport.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load

from .level_zero_ipc_equal_a2a import _Descriptor, _client_exchange, _root_exchange


_MODULE: ModuleType | None = None
_SLOTS = 3


def load_level_zero_ipc_raw_device_barrier_ops(verbose: bool = False) -> ModuleType:
    """Build and load the isolated raw-SYCL IPC control extension."""

    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.xpu.is_available():
        raise RuntimeError("Aurora XPU is required for a raw IPC barrier probe")
    source = Path(__file__).with_name("csrc") / "level_zero_ipc_raw_device_barrier.sycl"
    build_dir = os.environ.get("AURORA_MOE_SYCL_BUILD_DIR")
    if build_dir:
        build_dir = str(Path(build_dir) / "level_zero_ipc_raw_device_barrier")
        Path(build_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_XPU_ARCH_LIST", "pvc")
    _MODULE = load(
        name="aurora_moe_level_zero_ipc_raw_device_barrier",
        sources=[str(source)],
        extra_cflags=["-O3", "-DNDEBUG"],
        extra_sycl_cflags=["-fno-sycl-instrument-device-code"],
        extra_ldflags=["-lze_loader"],
        with_cuda=False,
        with_sycl=True,
        verbose=verbose,
        build_directory=build_dir,
    )
    return _MODULE


class RawIpcDeviceBarrierEpoch:
    """Own oneCCL-shaped raw device controls and persistent SCM mappings.

    This class is intentionally a standalone diagnostic.  Its normal public
    counterpart remains :class:`IpcDeviceBarrierEpoch`, whose tensor-backed
    controls are already used by the end-to-end EP transport gate.
    """

    def __init__(
        self,
        group: dist.ProcessGroup | None = None,
        *,
        socket_path: str | os.PathLike[str] | None = None,
    ) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("initialize torch.distributed before constructing an IPC barrier")
        if not torch.xpu.is_available():
            raise RuntimeError("Aurora XPU is required for a raw IPC barrier")
        self._group = group
        self._world_size = dist.get_world_size(group)
        self._rank = dist.get_rank(group)
        if self._world_size < 2 or self._world_size > 12:
            raise ValueError("raw IPC device barriers support same-node world sizes in [2, 12]")
        self._device_index = torch.xpu.current_device()
        self._device = torch.device("xpu", self._device_index)
        base = socket_path or os.environ.get("AURORA_MOE_L0_IPC_RAW_BARRIER_SOCKET")
        if not base:
            raise RuntimeError("set AURORA_MOE_L0_IPC_RAW_BARRIER_SOCKET for raw IPC exchange")
        self._socket_base = Path(base)
        self._ipc_bias, self._ipc_bias_flags = self._mapping_policy()
        self._import_backend = os.environ.get("AURORA_MOE_L0_IPC_IMPORT_BACKEND", "scm")
        if self._import_backend != "scm":
            raise ValueError("raw IPC device barrier comparison currently supports SCM exchange only")
        self._ops = load_level_zero_ipc_raw_device_barrier_ops()
        if int(self._ops.slots) != _SLOTS:
            raise RuntimeError("raw IPC Python and SYCL slot counts disagree")
        self._local_control: object | None = None
        self._imports: list[object | None] = []
        self._descriptors: list[_Descriptor] = []
        self._remote_addresses: tuple[int, ...] = ()
        self._ordinal = 0
        self._epoch = 0
        self._closed = False
        self._create_epoch()

    @staticmethod
    def _mapping_policy() -> tuple[str, int]:
        policy = os.environ.get("AURORA_MOE_L0_IPC_BIAS", "default")
        flags = {"default": 0, "cached": 1, "uncached": 2}
        if policy not in flags:
            raise ValueError("AURORA_MOE_L0_IPC_BIAS must be default, cached, or uncached")
        return policy, flags[policy]

    @staticmethod
    def _signed_pointer(address: int) -> int:
        return address - (1 << 64) if address >= 1 << 63 else address

    def _barrier(self) -> None:
        dist.barrier(group=self._group)

    def _epoch_path(self) -> Path:
        path = Path(f"{self._socket_base}.raw.epoch{self._epoch}")
        if len(os.fsencode(path)) >= 108:
            raise RuntimeError(f"raw IPC socket path is too long: {path}")
        return path

    def _require_local_control(self) -> object:
        if self._local_control is None:
            raise RuntimeError("raw IPC device barrier is closed")
        return self._local_control

    def _create_epoch(self) -> None:
        self._barrier()
        self._epoch += 1
        local = self._ops.RawDeviceControl(self._device_index)
        own = _Descriptor(
            local.fd(),
            os.getpid(),
            local.control_offset_bytes(),
            local.handle_bytes(),
            exporter_owned_fd=True,
        )
        path = self._epoch_path()
        if self._rank == 0:
            descriptors = _root_exchange(path, self._world_size, own, send_fd=True)
        else:
            descriptors = _client_exchange(
                path, self._rank, self._world_size, own, send_fd=True
            )
        imports: list[object | None] = [None] * self._world_size
        for source, descriptor in enumerate(descriptors):
            if source != self._rank:
                imports[source] = self._ops.RawIpcImport(
                    descriptor.handle,
                    descriptor.take_fd(),
                    self._device_index,
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
                address = int(local.control_address())
            else:
                imported = imports[source]
                if imported is None:
                    raise RuntimeError(f"missing raw IPC control mapping for rank {source}")
                address = int(imported.mapping_address()) + descriptor.tensor_offset_bytes
            addresses.append(self._signed_pointer(address))

        self._local_control = local
        self._imports = imports
        self._descriptors = descriptors
        self._remote_addresses = tuple(addresses)
        torch.xpu.synchronize(self._device)
        self._barrier()
        # Mirror oneCCL's explicit post-allocation initialization, after every
        # peer has opened the raw IPC mapping.  This is setup-only.
        local.zero()
        torch.xpu.synchronize(self._device)
        self._barrier()

    def _close_imports(self) -> None:
        for imported in self._imports:
            if imported is not None:
                imported.close()
        self._imports = []
        for descriptor in self._descriptors:
            if descriptor.fd >= 0 and not descriptor.exporter_owned_fd:
                descriptor.close_received_fd()
        self._descriptors = []
        self._remote_addresses = ()

    @property
    def ordinal(self) -> int:
        return self._ordinal

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def ipc_bias(self) -> str:
        return self._ipc_bias

    @property
    def remote_addresses(self) -> tuple[int, ...]:
        if not self._remote_addresses:
            raise RuntimeError("raw IPC device barrier is closed")
        return self._remote_addresses

    def _expected_counters_for_count(self, count: int) -> tuple[int, int, int]:
        return tuple(
            self._world_size * ((count + (_SLOTS - 1 - slot)) // _SLOTS)
            for slot in range(_SLOTS)
        )

    def expected_counters(self) -> tuple[int, int, int]:
        return self._expected_counters_for_count(self._ordinal)

    def enqueue(self) -> int:
        if self._closed:
            raise RuntimeError("raw IPC device barrier is closed")
        ordinal = self._ordinal
        local = self._require_local_control()
        self._ops.enqueue_raw_ipc_device_barrier(
            self.remote_addresses,
            self._world_size,
            self._rank,
            ordinal,
            self._device_index,
            self._signed_pointer(int(local.control_address())),
        )
        self._ordinal += 1
        return ordinal

    def validate_after_synchronize(self) -> tuple[int, int, int]:
        torch.xpu.synchronize(self._device)
        values = tuple(int(value) for value in self._require_local_control().counters())
        expected = self.expected_counters()
        if values != expected:
            raise AssertionError(f"raw IPC barrier counters are {values}, expected {expected}")
        return values

    def close(self) -> None:
        if self._closed:
            return
        self._barrier()
        torch.xpu.synchronize(self._device)
        self._close_imports()
        self._barrier()
        self._require_local_control().close()
        self._local_control = None
        self._barrier()
        self._closed = True
