"""Asynchronous same-node BF16 EP all-to-all over persistent Level Zero IPC.

This is the production-facing composition of the persistent push payload
kernel and its device-side readiness protocol.  It deliberately has no MoE
shape knowledge: the input is any contiguous equal-size ``[EP, ...]`` BF16
tensor and the backing allocation grows exactly when that runtime payload
grows.  The implementation is limited by the Aurora node-local typed peer
kernel to 2--12 ranks, not by expert, hidden, route, or token dimensions.
"""

from __future__ import annotations

import os
from os import PathLike

import torch
import torch.distributed as dist

from .level_zero_ipc_device_barrier import IpcDeviceBarrierEpoch
from .level_zero_ipc_push_a2a import LevelZeroIpcPushAllToAll


class LevelZeroIpcEpAllToAll:
    """Exact current-stream BF16 equal all-to-all for one node-local EP group.

    ``exchange_async`` does not wait on the host.  Its two device barriers
    ensure that every rank has finished packing before peer writes begin, and
    that every peer write is visible before the receive staging is copied into
    a fresh output tensor.  Consequently a later current-stream consumer can
    use that output normally, while the persistent receive staging can be
    safely reused by the next exchange.

    All ranks in ``group`` must call the method in the same order with equal
    runtime shapes.  Shape equality is established collectively only when an
    epoch is first allocated or grows; the steady path has no host collective,
    capacity factor, or route clipping.
    """

    def __init__(
        self,
        group: dist.ProcessGroup | None = None,
        *,
        socket_path: str | PathLike[str] | None = None,
    ) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("initialize torch.distributed before IPC EP all-to-all")
        base = socket_path or os.environ.get("AURORA_MOE_L0_IPC_EP_SOCKET")
        if not base:
            raise RuntimeError(
                "set AURORA_MOE_L0_IPC_EP_SOCKET for persistent EP IPC descriptor exchange"
            )
        # The two components perform independent collective descriptor
        # exchanges.  Their distinct bases prevent a control socket from ever
        # colliding with a payload epoch, including during allocation growth.
        base_text = os.fspath(base)
        self._payload = LevelZeroIpcPushAllToAll(
            group, socket_path=f"{base_text}.payload"
        )
        if self._payload.copy_impl != "ccl_typed":
            raise RuntimeError(
                "LevelZeroIpcEpAllToAll requires AURORA_MOE_L0_IPC_PUSH_IMPL=ccl_typed"
            )
        self._readiness = IpcDeviceBarrierEpoch(
            group, socket_path=f"{base_text}.readiness"
        )
        self._closed = False

    @property
    def capacity_elements(self) -> int:
        """Current exact per-peer payload staging capacity in BF16 elements."""

        return self._payload.capacity_elements

    @property
    def barrier_ordinal(self) -> int:
        """Next readiness-barrier ordinal, useful for diagnostics only."""

        return self._readiness.ordinal

    def exchange_async(
        self, input: torch.Tensor, output: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Enqueue an exact equal all-to-all without host completion waits."""

        if self._closed:
            raise RuntimeError("IPC EP all-to-all is closed")
        return self._payload.exchange_async(
            input,
            output,
            pre_remote_write=self._readiness.enqueue,
            post_remote_write=self._readiness.enqueue,
        )

    def close(self) -> None:
        """Collectively retire payload and readiness mappings after queued work."""

        if self._closed:
            return
        # Each component synchronizes before releasing its own imported
        # mappings.  Retire payload first because its post barrier may be the
        # final current-stream dependency on the control epoch.
        self._payload.close()
        self._readiness.close()
        self._closed = True
