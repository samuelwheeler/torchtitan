import math
import os
import threading
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from aurora_moe._kernels.ddp_anchor import ddp_parameter_anchor
from aurora_moe._kernels.joint_moe_autograd import joint_routed_shared_moe
from aurora_moe._kernels.phase_shared_expert import PhaseSharedExpertController


PROFILE = os.environ.get("AURORA_MOE_PROFILE") == "1"


# Opt-in direct-oneCCL state.  This stays empty unless
# AURORA_MOE_NATIVE_CCL=1 is set before launching the training process.  A
# state retains its ProcessGroup identity, so an id cannot be reused for a
# newly-created group while the old communicator is still cached.
_NATIVE_CCL_BY_GROUP = {}
_NATIVE_CCL_DEDICATED_A4_BY_GROUP = {}
# Opt-in persistent same-node IPC transport.  The group object, rather than
# an MoE shape, owns its staging/control epochs so one transport can serve all
# exact BF16 EP payloads on that group.
_L0_IPC_EP_BY_GROUP = {}
# The compact transport is deliberately separate from the equal-size path:
# its exported staging is one flat source-major receive buffer and it needs
# runtime split-offset metadata from the exact route-count matrix.
_L0_IPC_ALLTOALLV_BY_GROUP = {}
_DIRECT_LAYOUT_TWO_PHASE_TAIL_STREAM_BY_DEVICE = {}
_DIRECT_LAYOUT_TWO_PHASE_A4_STREAM_BY_DEVICE = {}


def _record(name):
    return torch.profiler.record_function(name) if PROFILE else nullcontext()


def _randn(shape, seed, scale, device):
    gen = torch.Generator()
    gen.manual_seed(seed)
    return (torch.randn(*shape, generator=gen) * scale).to(device)


def _module_dtype(config):
    if config.dtype in ("bf16", "bfloat16"):
        return torch.bfloat16
    if config.dtype in ("fp32", "float32"):
        return torch.float32
    raise ValueError("unsupported dtype %s" % config.dtype)


def _init_experts(up, gate, down, expert_ids, config, layer_id):
    with torch.no_grad():
        for i, expert_id in enumerate(expert_ids):
            base = config.seed + 10000 * (layer_id + 1) + expert_id
            up[i].copy_(_randn(up[i].shape, base, 1.0 / math.sqrt(up.shape[1]), up.device))
            gate[i].copy_(_randn(gate[i].shape, base + 5000, 1.0 / math.sqrt(gate.shape[1]), gate.device))
            down[i].copy_(_randn(down[i].shape, base + 10000, 1.0 / math.sqrt(down.shape[1]), down.device))


def _swiglu(x, up, gate, down):
    return (F.silu(x.matmul(gate)) * x.matmul(up)).matmul(down)


def _native_ccl_requested():
    return os.environ.get("AURORA_MOE_NATIVE_CCL") == "1"


def _native_ccl_ep_requested():
    """Whether EP all-to-all uses the experimental direct oneCCL bridge.

    The reducer may remain direct-oneCCL while an experiment routes EP through
    ProcessGroupXCCL.  This keeps its strict CCL_OP_SYNC=0 requirement while
    allowing an apples-to-apples transport comparison without changing expert
    compute, routing, or DDP-reduction semantics.
    """

    return _native_ccl_requested() and os.environ.get(
        "AURORA_MOE_NATIVE_CCL_EP_XCCL"
    ) != "1"


def _native_ccl_alltoallv_requested(mesh):
    """Whether exact BF16 EP all-to-all-v uses the direct oneCCL bridge."""

    requested = os.environ.get("AURORA_MOE_NATIVE_CCL_ALLTOALLV") == "1"
    if not requested or mesh.group_size["ep_dispatch"] == 1:
        return False
    if not _native_ccl_ep_requested():
        raise RuntimeError(
            "AURORA_MOE_NATIVE_CCL_ALLTOALLV=1 requires direct native EP CCL; "
            "unset AURORA_MOE_NATIVE_CCL_EP_XCCL"
        )
    _require_native_ccl_environment()
    return True


def _exact_equal_alltoall_fastpath_requested():
    """Whether an exactly uniform compact route matrix may use all-to-all.

    This is deliberately not a capacity/padding shortcut.  It is available
    only when the *entire* observed ``[source, destination]`` route-count
    matrix has one nonzero value, so every rank takes the same collective
    branch and an equal-size exchange moves precisely the rows that the
    ordinary all-to-all-v would move.  Dynamic or imbalanced routing remains
    on all-to-all-v without a behavioural change.
    """

    value = os.environ.get("AURORA_MOE_EXACT_EQUAL_A2A_FASTPATH", "0")
    if value not in {"0", "1"}:
        raise ValueError("AURORA_MOE_EXACT_EQUAL_A2A_FASTPATH must be 0 or 1")
    return value == "1"


def _l0_ipc_ep_requested(mesh):
    """Whether BF16 equal-size EP payloads use persistent node-local IPC.

    This is intentionally an explicit transport experiment.  It applies only
    to the equal-size BF16 payload calls in ``_all_to_all``; arbitrary-ID
    int64 metadata and all-to-all-v retain their established transport.  The
    typed SYCL peer kernel supports the Aurora node-local EP range 2--12,
    while all model, expert, route, and token dimensions remain runtime data.
    """

    if os.environ.get("AURORA_MOE_L0_IPC_EP") != "1":
        return False
    ep_size = mesh.group_size["ep_dispatch"]
    if ep_size <= 1:
        return False
    if ep_size > 12:
        raise RuntimeError(
            "AURORA_MOE_L0_IPC_EP supports one node-local EP group of 2--12 Aurora tiles"
        )
    if not torch.xpu.is_available():
        raise RuntimeError("AURORA_MOE_L0_IPC_EP requires Aurora XPU tensors")
    local_size = int(
        os.environ.get(
            "LOCAL_WORLD_SIZE", os.environ.get("PALS_LOCAL_SIZE", torch.xpu.device_count())
        )
    )
    if ep_size > local_size:
        raise RuntimeError(
            "AURORA_MOE_L0_IPC_EP requires every EP-dispatch rank to reside on one node"
        )
    return True


def _l0_ipc_alltoallv_requested(mesh):
    """Whether exact BF16 compact EP all-to-all-v uses Level Zero IPC.

    This is not the equal-size IPC fast path repurposed with padded route
    blocks.  The implementation receives the true per-peer split vectors and
    copies only those rows into compact source-major output segments.  It is
    node-local because Level Zero IPC handles are not inter-node transport.
    """

    if os.environ.get("AURORA_MOE_L0_IPC_ALLTOALLV") != "1":
        return False
    ep_size = mesh.group_size["ep_dispatch"]
    if ep_size <= 1:
        return False
    if ep_size > 12:
        raise RuntimeError(
            "AURORA_MOE_L0_IPC_ALLTOALLV supports one node-local EP group of 2--12 Aurora tiles"
        )
    if not torch.xpu.is_available():
        raise RuntimeError("AURORA_MOE_L0_IPC_ALLTOALLV requires Aurora XPU tensors")
    local_size = int(
        os.environ.get(
            "LOCAL_WORLD_SIZE", os.environ.get("PALS_LOCAL_SIZE", torch.xpu.device_count())
        )
    )
    if ep_size > local_size:
        raise RuntimeError(
            "AURORA_MOE_L0_IPC_ALLTOALLV requires every EP-dispatch rank to reside on one node"
        )
    return True


def _l0_ipc_expert_major_requested(mesh):
    """Whether compact IPC places live fragments directly in expert-major order."""

    value = os.environ.get("AURORA_MOE_L0_IPC_EXPERT_MAJOR", "0")
    if value not in {"0", "1"}:
        raise ValueError("AURORA_MOE_L0_IPC_EXPERT_MAJOR must be 0 or 1")
    if value == "0":
        return False
    if not _l0_ipc_alltoallv_requested(mesh):
        raise RuntimeError(
            "AURORA_MOE_L0_IPC_EXPERT_MAJOR=1 requires "
            "AURORA_MOE_L0_IPC_ALLTOALLV=1"
        )
    if os.environ.get("AURORA_MOE_L0_IPC_ALLTOALLV_CCL_TYPED_STORE") != "1":
        raise RuntimeError(
            "AURORA_MOE_L0_IPC_EXPERT_MAJOR=1 requires the CCL-128 IPC store"
        )
    return True


def _native_ccl_count_alltoall_requested(mesh):
    """Whether exact EP route counts use a tiny direct-oneCCL exchange.

    The regular XCCL all-gather moves only ``EP`` int64 values, but its
    blocking host control path is material in the profiled critical path.  An
    equal-size native all-to-all of a replicated ``[EP, EP]`` count matrix
    returns the identical source-major count matrix with a device-side event
    bridge.  This remains strictly opt-in because all ranks must enter the
    same direct-oneCCL collective ordering as their payload exchanges.
    """

    requested = os.environ.get("AURORA_MOE_NATIVE_CCL_COUNT_ALLTOALL") == "1"
    if not requested or mesh.group_size["ep_dispatch"] == 1:
        return False
    if not _native_ccl_ep_requested():
        raise RuntimeError(
            "AURORA_MOE_NATIVE_CCL_COUNT_ALLTOALL=1 requires direct native EP CCL; "
            "unset AURORA_MOE_NATIVE_CCL_EP_XCCL"
        )
    _require_native_ccl_environment()
    return True


def _native_ccl_reducer_requested():
    """Whether to replace DDP's XCCL bucket reducer with direct oneCCL SUMs."""

    requested = os.environ.get("AURORA_MOE_NATIVE_CCL_REDUCER") == "1"
    if requested and not _native_ccl_requested():
        raise RuntimeError(
            "AURORA_MOE_NATIVE_CCL_REDUCER=1 requires AURORA_MOE_NATIVE_CCL=1"
        )
    if requested:
        _require_native_ccl_environment()
    return requested


def _native_reducer_two_streams_requested():
    """Whether to launch dense and sparse native gradient SUMs concurrently."""

    if os.environ.get("AURORA_MOE_NATIVE_SPARSE_REDUCER_OVERLAP") == "1":
        raise RuntimeError(
            "AURORA_MOE_NATIVE_SPARSE_REDUCER_OVERLAP uses unsafe autograd hooks; "
            "use AURORA_MOE_NATIVE_REDUCER_TWO_STREAMS=1 instead"
        )
    requested = os.environ.get("AURORA_MOE_NATIVE_REDUCER_TWO_STREAMS") == "1"
    if requested and not _native_ccl_reducer_requested():
        raise RuntimeError(
            "AURORA_MOE_NATIVE_REDUCER_TWO_STREAMS=1 requires "
            "AURORA_MOE_NATIVE_CCL_REDUCER=1"
        )
    return requested


def _expert_major_two_phase_requested():
    """Whether compact router-free A4 may overlap its exact dW tail.

    This is deliberately narrower than the ordinary expert-major path.  The
    split state returns a zero score-gradient column, so it is valid only for
    the explicitly router-gradient-free experiment.  Its operands remain
    exact compact rows and true runtime expert lengths; no capacity tensor or
    padded BMM is involved.
    """

    value = os.environ.get("AURORA_MOE_EXPERT_MAJOR_TWO_PHASE", "0")
    if value not in {"0", "1"}:
        raise ValueError("AURORA_MOE_EXPERT_MAJOR_TWO_PHASE must be 0 or 1")
    if value != "1":
        return False
    if os.environ.get("AURORA_MOE_IGNORE_ROUTER_GRAD") != "1":
        raise RuntimeError(
            "AURORA_MOE_EXPERT_MAJOR_TWO_PHASE=1 requires "
            "AURORA_MOE_IGNORE_ROUTER_GRAD=1"
        )
    if os.environ.get("AURORA_MOE_EXPERT_MAJOR_GEMM") != "onemkl" or os.environ.get(
        "AURORA_MOE_EXPERT_MAJOR_DW"
    ) != "onemkl":
        raise RuntimeError(
            "AURORA_MOE_EXPERT_MAJOR_TWO_PHASE=1 requires exact oneMKL "
            "forward and dW backends"
        )
    if os.environ.get("AURORA_MOE_EXPERT_MAJOR_PACKED_UP_GATE", "0") != "0":
        raise RuntimeError(
            "AURORA_MOE_EXPERT_MAJOR_TWO_PHASE=1 does not yet support "
            "packed up/gate weights"
        )
    return True


def _native_reducer_pack_sync_requested():
    """Whether the two-stream reducer uses a diagnostic post-pack device sync."""

    return os.environ.get("AURORA_MOE_NATIVE_REDUCER_PACK_SYNC") == "1"


def _native_reducer_serialize_collectives_requested():
    """Whether two-stream gradient buckets serialize dense then sparse CCL work.

    This preserves the parallel producer-side gradient packs while avoiding
    concurrent direct-oneCCL launches on the overlapping dense-DP and
    sparse-DP process groups.  It is deliberately opt-in until the ordering
    is validated on the target distributed configuration.
    """

    value = os.environ.get("AURORA_MOE_NATIVE_REDUCER_SERIALIZE_COLLECTIVES", "0")
    if value not in {"0", "1"}:
        raise ValueError(
            "AURORA_MOE_NATIVE_REDUCER_SERIALIZE_COLLECTIVES must be 0 or 1"
        )
    return value == "1"


def _direct_expert_layout_requested():
    """Use the exact receive-padded-to-expert layout kernels when opted in.

    This is deliberately independent of ``AURORA_MOE_ALLTOALLV``: it consumes
    the standard padded receive buffer directly and does not impose a capacity
    factor or a fixed route count.
    """

    return os.environ.get("AURORA_MOE_DIRECT_EXPERT_LAYOUT") == "1"


def _direct_layout_precompute_expert_counts_requested():
    """Exchange exact per-expert receive counts before payload dispatch.

    This optional direct-layout fast path is mathematically identical to the
    receive-side ID histogram: it merely moves that histogram into the small
    EP control collective that already determines the exact padded transport
    width.  It is opt-in while the full distributed correctness and timing
    gate is being established.
    """

    requested = os.environ.get("AURORA_MOE_DIRECT_LAYOUT_PRECOMPUTE_EXPERT_COUNTS") == "1"
    if requested and not _direct_expert_layout_requested():
        raise RuntimeError(
            "AURORA_MOE_DIRECT_LAYOUT_PRECOMPUTE_EXPERT_COUNTS=1 requires "
            "AURORA_MOE_DIRECT_EXPERT_LAYOUT=1"
        )
    return requested


def _direct_layout_precompute_native_count_alltoall_requested():
    """Opt in to native oneCCL for the widened pre-dispatch count matrix.

    The ordinary EP count vector is only ``EP`` entries wide.  The direct
    pre-dispatch layout control is wider by the number of local experts and
    should normally use ProcessGroupXCCL's tiny all-gather instead: Aurora's
    oneCCL large-alltoall implementation performs a synchronous IPC-handle
    descriptor exchange for every invocation, which has caused an otherwise
    correct target-shape validation process to stall.  The payload collectives
    remain direct oneCCL; this only selects the negligible control plane.
    Keep native control available as an explicit diagnostic opt-in.
    """

    return os.environ.get(
        "AURORA_MOE_DIRECT_LAYOUT_PRECOMPUTE_NATIVE_COUNT_ALLTOALL"
    ) == "1"


def _direct_layout_two_phase_requested():
    """Enable the opt-in A4/dW split for the direct padded expert layout."""

    requested = os.environ.get("AURORA_MOE_DIRECT_LAYOUT_TWO_PHASE") == "1"
    if requested and not _direct_expert_layout_requested():
        raise RuntimeError(
            "AURORA_MOE_DIRECT_LAYOUT_TWO_PHASE=1 requires "
            "AURORA_MOE_DIRECT_EXPERT_LAYOUT=1"
        )
    return requested


def _phase_shared_experts_requested(mesh):
    """Whether shared XMX work is split around exact EP A1/A3/A4 windows.

    The phase controller has two exact routed-backend implementations:

    * the older receive-padded/direct-layout path, whose backward exposes a
      separate dW tail; and
    * the compact all-to-all-v path, whose local nested autograd finishes all
      of dX, dscore, and dW before reverse A4 is submitted.

    Both have the same useful communication windows.  The controller accepts
    either the validated private native-oneCCL path or the exact node-local
    compact IPC path; it still rejects ordinary ProcessGroupXCCL because that
    path has no explicit GPU-resident producer/completion protocol.
    """

    requested = os.environ.get("AURORA_MOE_PHASE_SHARED_EXPERTS") == "1"
    reordered = _phase_shared_forward_reorder_requested()
    if reordered and not requested:
        raise RuntimeError(
            "AURORA_MOE_PHASE_SHARED_FORWARD_REORDER=1 requires "
            "AURORA_MOE_PHASE_SHARED_EXPERTS=1"
        )
    if not requested:
        return False
    compact_alltoallv = os.environ.get("AURORA_MOE_ALLTOALLV") == "1"
    direct_two_phase = _direct_layout_two_phase_requested()
    if compact_alltoallv and direct_two_phase:
        raise RuntimeError(
            "AURORA_MOE_PHASE_SHARED_EXPERTS=1 cannot combine "
            "AURORA_MOE_ALLTOALLV=1 with AURORA_MOE_DIRECT_LAYOUT_TWO_PHASE=1"
        )
    if not compact_alltoallv and not direct_two_phase:
        raise RuntimeError(
            "AURORA_MOE_PHASE_SHARED_EXPERTS=1 requires either "
            "AURORA_MOE_ALLTOALLV=1 or AURORA_MOE_DIRECT_LAYOUT_TWO_PHASE=1"
        )
    if mesh.group_size["ep_dispatch"] <= 1:
        raise RuntimeError("AURORA_MOE_PHASE_SHARED_EXPERTS=1 requires EP size greater than one")
    compact_l0_ipc = compact_alltoallv and _l0_ipc_alltoallv_requested(mesh)
    if compact_l0_ipc:
        # The IPC implementation inserts its own current-stream bridges and
        # device-side incoming-write barrier, so the controller can begin a
        # shared phase immediately after each nonblocking submission.  It has
        # no oneCCL private-stream dependency.
        return True
    if not _native_ccl_ep_requested():
        raise RuntimeError(
            "AURORA_MOE_PHASE_SHARED_EXPERTS=1 requires direct native EP CCL; "
            "unset AURORA_MOE_NATIVE_CCL_EP_XCCL"
        )
    if os.environ.get("AURORA_MOE_NATIVE_CCL_PRIVATE_STREAM") != "1":
        raise RuntimeError(
            "AURORA_MOE_PHASE_SHARED_EXPERTS=1 requires "
            "AURORA_MOE_NATIVE_CCL_PRIVATE_STREAM=1"
        )
    if os.environ.get("AURORA_MOE_NATIVE_CCL_STREAM_FENCE_REAP") != "1":
        raise RuntimeError(
            "AURORA_MOE_PHASE_SHARED_EXPERTS=1 requires "
            "AURORA_MOE_NATIVE_CCL_STREAM_FENCE_REAP=1"
        )
    if compact_alltoallv and not _native_ccl_alltoallv_requested(mesh):
        raise RuntimeError(
            "AURORA_MOE_PHASE_SHARED_EXPERTS=1 with AURORA_MOE_ALLTOALLV=1 "
            "requires AURORA_MOE_NATIVE_CCL_ALLTOALLV=1"
        )
    _require_native_ccl_environment()
    return True


def _phase_shared_forward_reorder_requested():
    """Start each independent shared phase before, rather than after, A1/A3 submit."""

    value = os.environ.get("AURORA_MOE_PHASE_SHARED_FORWARD_REORDER", "0")
    if value not in ("0", "1"):
        raise ValueError("AURORA_MOE_PHASE_SHARED_FORWARD_REORDER must be '0' or '1'")
    return value == "1"


def _phase_shared_a4_submit_wait_requested():
    """Whether shared backward waits for the native A4 host call to return.

    The default preserves the conservative ordering used by the validated
    native-oneCCL phase path.  The opt-in ``0`` setting is a narrowly scoped
    diagnostic for whether the independent shared backward can be queued once
    the routed payload producer event exists, while the host thread is still
    submitting reverse A4.  It does not relax the device dependency between
    the routed payload and the shared stream; it only removes a host-side
    gate.  Use it only with the distributed correctness and multi-step stress
    gates, because earlier Aurora runs observed a nested-autograd/oneCCL
    liveness race when this ordering was changed.
    """

    value = os.environ.get("AURORA_MOE_PHASE_SHARED_A4_SUBMIT_WAIT", "1")
    if value not in ("0", "1"):
        raise ValueError(
            "AURORA_MOE_PHASE_SHARED_A4_SUBMIT_WAIT must be '0' or '1'"
        )
    return value == "1"


def _dedicated_native_a4_requested(mesh, direct_layout_two_phase):
    """Whether reverse EP A4 uses its own prewarmed native communicator."""

    requested = os.environ.get("AURORA_MOE_NATIVE_CCL_DEDICATED_A4") == "1"
    if not requested:
        return False
    if not direct_layout_two_phase:
        raise RuntimeError(
            "AURORA_MOE_NATIVE_CCL_DEDICATED_A4=1 requires "
            "AURORA_MOE_DIRECT_LAYOUT_TWO_PHASE=1"
        )
    if mesh.group_size["ep_dispatch"] <= 1:
        raise RuntimeError(
            "AURORA_MOE_NATIVE_CCL_DEDICATED_A4=1 requires EP size greater than one"
        )
    if not _native_ccl_ep_requested():
        raise RuntimeError(
            "AURORA_MOE_NATIVE_CCL_DEDICATED_A4=1 requires direct native EP CCL; "
            "unset AURORA_MOE_NATIVE_CCL_EP_XCCL"
        )
    if os.environ.get("AURORA_MOE_NATIVE_CCL_PRIVATE_STREAM") != "1":
        raise RuntimeError(
            "AURORA_MOE_NATIVE_CCL_DEDICATED_A4=1 requires "
            "AURORA_MOE_NATIVE_CCL_PRIVATE_STREAM=1"
        )
    _require_native_ccl_environment()
    return True


def _direct_layout_two_phase_tail_stream(device):
    """Return this process's persistent weight-gradient tail stream for ``device``."""

    if device.type != "xpu":
        raise ValueError("direct-layout two-phase tail requires an XPU device")
    index = torch.xpu.current_device() if device.index is None else device.index
    stream = _DIRECT_LAYOUT_TWO_PHASE_TAIL_STREAM_BY_DEVICE.get(index)
    if stream is None:
        stream = torch.xpu.Stream(device=torch.device("xpu", index))
        _DIRECT_LAYOUT_TWO_PHASE_TAIL_STREAM_BY_DEVICE[index] = stream
    return stream


def _direct_layout_two_phase_a4_stream(device):
    """Return this process's persistent native-A4 submission stream for ``device``."""

    if device.type != "xpu":
        raise ValueError("direct-layout two-phase A4 requires an XPU device")
    index = torch.xpu.current_device() if device.index is None else device.index
    stream = _DIRECT_LAYOUT_TWO_PHASE_A4_STREAM_BY_DEVICE.get(index)
    if stream is None:
        stream = torch.xpu.Stream(device=torch.device("xpu", index))
        _DIRECT_LAYOUT_TWO_PHASE_A4_STREAM_BY_DEVICE[index] = stream
    return stream


def _native_ccl_reap_mode():
    """Select the explicit lifetime policy for direct-oneCCL work objects."""

    defer = os.environ.get("AURORA_MOE_NATIVE_CCL_DEFER_REAP") == "1"
    stream_fence = os.environ.get("AURORA_MOE_NATIVE_CCL_STREAM_FENCE_REAP") == "1"
    if defer and stream_fence:
        raise RuntimeError(
            "AURORA_MOE_NATIVE_CCL_DEFER_REAP and "
            "AURORA_MOE_NATIVE_CCL_STREAM_FENCE_REAP cannot both be enabled"
        )
    if defer:
        return "defer"
    if stream_fence:
        return "stream_fence"
    return "ccl_test"


def _native_ccl_defer_reap_limit():
    value = os.environ.get("AURORA_MOE_NATIVE_CCL_DEFER_REAP_MAX_WORKS", "12")
    try:
        limit = int(value)
    except ValueError as error:
        raise ValueError("AURORA_MOE_NATIVE_CCL_DEFER_REAP_MAX_WORKS must be positive") from error
    if limit <= 0:
        raise ValueError("AURORA_MOE_NATIVE_CCL_DEFER_REAP_MAX_WORKS must be positive")
    return limit


def _require_native_ccl_environment():
    # CCL_OP_SYNC is read when oneCCL initializes; changing it after XCCL has
    # made a communicator cannot make the direct path asynchronous.  Require
    # the exact fresh-process setting used by the standalone correctness gate.
    if os.environ.get("CCL_OP_SYNC") != "0":
        raise RuntimeError(
            "AURORA_MOE_NATIVE_CCL=1 requires a fresh process started with "
            "CCL_OP_SYNC=0 before XCCL/oneCCL initialization"
        )


class _NativeCclGroupState:
    """One direct-oneCCL communicator and its in-flight collective work."""

    def __init__(self, group):
        # Keep the import lazy: default MoE runs retain their normal XCCL-only
        # dependency path and never JIT-build the experimental extension.
        from aurora_moe._kernels.native_ccl_a2a import NativeCclA2A

        self.group = group
        self.communicator = NativeCclA2A(group, require_async=True)
        self.pending = []
        self._launch_lock = threading.RLock()
        self._reap_mode = _native_ccl_reap_mode()
        self._defer_limit = (
            _native_ccl_defer_reap_limit() if self._reap_mode == "defer" else None
        )
        self._warmed_up = False

    def reap(self):
        # NativeCclWork owns tensor storage, oneCCL state, and its consumer
        # fences.  CCL completion alone can precede a queued PyTorch consumer;
        # the explicit stream-fence mode retains work through that consumer.
        with self._launch_lock:
            if self._reap_mode == "defer":
                with _record("moe.comm.native_reap_deferred"):
                    if len(self.pending) >= self._defer_limit:
                        raise RuntimeError(
                            "deferred native CCL reap reached its pending-work limit; "
                            "use only a short benchmark or disable "
                            "AURORA_MOE_NATIVE_CCL_DEFER_REAP"
                        )
                return
            if self._reap_mode == "stream_fence":
                with _record("moe.comm.native_reap_stream_fence"):
                    self.pending = [
                        work for work in self.pending if not work.is_stream_fence_completed()
                    ]
                return
            with _record("moe.comm.native_reap_ccl_test"):
                self.pending = [work for work in self.pending if not work.is_completed()]

    def all_to_all(self, x):
        with self._launch_lock:
            self.reap()
            y = torch.empty_like(x)
            with _record("moe.comm.native_alltoall_submit"):
                result = self.communicator.all_to_all_single(x, output=y, async_op=True)
            # async_op=True always returns (output, NativeCclWork).  Keep the work
            # alive until its event completes; this also retains a temporary
            # contiguous input passed by _all_to_all.
            output, work = result
            self.pending.append(work)
            return output

    def all_to_all_v(self, x, input_splits, output_splits):
        """Submit exact BF16 all-to-all-v and retain its consumer fence."""

        with self._launch_lock:
            self.reap()
            y = x.new_empty((sum(output_splits), *x.shape[1:]))
            with _record("moe.comm.native_alltoallv_submit"):
                output, work = self.communicator.all_to_all_v(
                    x,
                    y,
                    output_splits,
                    input_splits,
                    async_op=True,
                )
            self.pending.append(work)
            return output

    def prepare_threaded_all_to_all_v(self, x, output_splits):
        """Allocate exact A4v output before a host-thread launch.

        The caller must subsequently submit exactly the matching split vectors
        through :meth:`submit_threaded_all_to_all_v`.  Keeping allocation and
        reaping on the main thread preserves the group's work lifetime while
        allowing the potentially blocking oneCCL launch itself to proceed in
        parallel with independent dW GEMMs.
        """

        with self._launch_lock:
            self.reap()
            return x.new_empty((sum(output_splits), *x.shape[1:]))

    def submit_threaded_all_to_all_v(
        self, x, output, input_splits, output_splits
    ):
        """Submit A4v without bridging it to the submitter's stream yet."""

        with self._launch_lock:
            with _record("moe.comm.native_alltoallv_thread_submit"):
                # The host-thread handoff explicitly calls ``wait_stream`` on
                # the real consumer before it exposes ``output``.  Deferring
                # the default bridge here avoids making the submitter's
                # auxiliary stream a redundant consumer and permits dW tail
                # enqueueing while oneCCL waits on its producer fence.
                return self.communicator.all_to_all_v(
                    x,
                    output,
                    output_splits,
                    input_splits,
                    async_op=True,
                    bridge_current_stream=False,
                )


    def all_reduce_sum(self, x):
        """Submit an out-of-place BF16 SUM reduction without a host wait."""

        with self._launch_lock:
            self.reap()
            y = torch.empty_like(x)
            with _record("moe.comm.native_allreduce_submit"):
                output, work = self.communicator.all_reduce_sum(x, output=y, async_op=True)
            self.pending.append(work)
            return output

    def all_reduce_sum_(self, x):
        """Submit an in-place BF16 SUM while retaining its event lifetime.

        Callers which consume ``x`` on PyTorch's current stream immediately
        after this return are safe: the native bridge has queued the oneCCL
        event dependency on that stream.  In-place operation is important for
        the persistent gradient buckets below: it avoids a second large
        allocation/copy and, unlike a temporary output, its storage outlives
        an event which becomes complete before a queued stream consumer runs.
        """

        self.all_reduce_sum_work_(x)
        return x

    def all_reduce_sum_work_(self, x):
        """Submit an in-place SUM and return its live work/event handle."""

        with self._launch_lock:
            self.reap()
            with _record("moe.comm.native_allreduce_submit"):
                output, work = self.communicator.all_reduce_sum(x, output=x, async_op=True)
            if output is not x:
                raise RuntimeError("native in-place all-reduce replaced its input buffer")
            self.pending.append(work)
            return work

    def warm_up(self, x):
        """Create this group's oneCCL communicator outside backward hooks."""

        with self._launch_lock:
            self.reap()
            self.communicator.warm_up(x)

    def warm_up_once(self, x):
        """Create this communicator once before its first latency-sensitive use."""

        with self._launch_lock:
            if self._warmed_up:
                return
            self.reap()
            self.communicator.warm_up(x)
            self._warmed_up = True

    def prepare_threaded_all_to_all(self, x):
        """Allocate an A4 output after reaping before a host-thread launch."""

        with self._launch_lock:
            self.reap()
            return torch.empty_like(x)

    def submit_threaded_all_to_all(self, x, output):
        """Submit A4 under the communicator-order lock from its host thread."""

        with self._launch_lock:
            with _record("moe.comm.native_alltoall_thread_submit"):
                # Keep the conservative launch-stream bridge.  The optional
                # deferred bridge API is useful for controlled experiments,
                # but Aurora's current runtime produced a GPU context abort
                # in a full DP=2/EP=12 stress run without this fence.
                return self.communicator.all_to_all_single(x, output=output, async_op=True)

    def retain_threaded_work(self, work):
        """Keep a host-thread native work/event alive on the main thread."""

        with self._launch_lock:
            self.pending.append(work)


def _native_ccl_group_state(group):
    key = id(group)
    state = _NATIVE_CCL_BY_GROUP.get(key)
    if state is None or state.group is not group:
        state = _NativeCclGroupState(group)
        _NATIVE_CCL_BY_GROUP[key] = state
    return state


def _native_ccl_ep_state(mesh):
    group = mesh.groups["ep_dispatch"]
    return _native_ccl_group_state(group)


def _native_ccl_dedicated_a4_state(mesh):
    """Return the separate oneCCL communicator reserved for reverse EP A4."""

    group = mesh.groups["ep_dispatch"]
    key = id(group)
    state = _NATIVE_CCL_DEDICATED_A4_BY_GROUP.get(key)
    if state is None or state.group is not group:
        state = _NativeCclGroupState(group)
        _NATIVE_CCL_DEDICATED_A4_BY_GROUP[key] = state
    return state


class _LevelZeroIpcEpGroupState:
    """One persistent payload/control transport for a node-local EP group."""

    def __init__(self, group, socket_path):
        # Keep the import and its SYCL extension loads lazy so ordinary XCCL
        # and native-oneCCL runs never build or allocate this experiment.
        from aurora_moe._kernels.level_zero_ipc_ep_a2a import LevelZeroIpcEpAllToAll

        self.group = group
        self.transport = LevelZeroIpcEpAllToAll(group, socket_path=socket_path)

    def all_to_all(self, x):
        return self.transport.exchange_async(x)


class _LevelZeroIpcAllToAllVGroupState:
    """One persistent exact compact IPC transport for a node-local EP group."""

    def __init__(self, group, socket_path):
        # Keep this distinct from the equal-size transport so its flat
        # source-major staging never becomes an accidental capacity-padded
        # route buffer.
        from aurora_moe._kernels.level_zero_ipc_alltoallv import LevelZeroIpcAllToAllV

        self.group = group
        self.transport = LevelZeroIpcAllToAllV(group, socket_path=socket_path)

    def all_to_all_v(
        self,
        x,
        input_splits,
        output_splits,
        remote_receive_offsets,
        *,
        global_capacity_rows,
        retain_output=False,
    ):
        return self.transport.exchange_async(
            x,
            input_splits,
            output_splits,
            remote_receive_offsets,
            global_capacity_rows=global_capacity_rows,
            retain_output=retain_output,
        )

    def all_to_all_v_expert_major(self, x, plan, *, inverse, retain_output=False):
        """Run the exact CCL-shaped direct expert-major placement for one phase."""

        return self.transport.exchange_expert_major_typed_async(
            x,
            plan.inverse_input_offsets if inverse else plan.forward_input_offsets,
            plan.inverse_remote_offsets if inverse else plan.forward_remote_offsets,
            plan.inverse_counts if inverse else plan.forward_counts,
            output_rows=plan.inverse_output_rows if inverse else plan.forward_output_rows,
            global_capacity_rows=(
                plan.inverse_capacity_rows if inverse else plan.forward_capacity_rows
            ),
            retain_output=retain_output,
        )


def _l0_ipc_ep_group_state(mesh):
    """Return the cached persistent IPC state for this exact EP group."""

    group = mesh.groups["ep_dispatch"]
    key = id(group)
    state = _L0_IPC_EP_BY_GROUP.get(key)
    if state is not None and state.group is group:
        return state

    base = os.environ.get("AURORA_MOE_L0_IPC_EP_SOCKET")
    if not base:
        raise RuntimeError(
            "set AURORA_MOE_L0_IPC_EP_SOCKET to a node-local Unix-socket base"
        )
    ranks = tuple(mesh.ranks["ep_dispatch"])
    if not ranks:
        raise RuntimeError("EP-dispatch group has no ranks")
    # A rank-list suffix permits more than one independent EP group on a host
    # without hardcoding DP layout.  `/tmp` is node-local on Aurora, so the
    # same base may safely be supplied by all ranks on different nodes.
    socket_path = f"{base}.ep{ranks[0]}_{ranks[-1]}"
    state = _LevelZeroIpcEpGroupState(group, socket_path)
    _L0_IPC_EP_BY_GROUP[key] = state
    return state


def _l0_ipc_alltoallv_group_state(mesh):
    """Return the cached exact compact IPC state for this EP group."""

    group = mesh.groups["ep_dispatch"]
    key = id(group)
    state = _L0_IPC_ALLTOALLV_BY_GROUP.get(key)
    if state is not None and state.group is group:
        return state

    base = os.environ.get("AURORA_MOE_L0_IPC_ALLTOALLV_SOCKET")
    if not base:
        # A caller that already supplies the equal-size base may reuse it:
        # the group/ragged suffix below makes its descriptor channel distinct.
        base = os.environ.get("AURORA_MOE_L0_IPC_EP_SOCKET")
    if not base:
        raise RuntimeError(
            "set AURORA_MOE_L0_IPC_ALLTOALLV_SOCKET to a node-local Unix-socket base"
        )
    ranks = tuple(mesh.ranks["ep_dispatch"])
    if not ranks:
        raise RuntimeError("EP-dispatch group has no ranks")
    socket_path = f"{base}.a2av.ep{ranks[0]}_{ranks[-1]}"
    state = _LevelZeroIpcAllToAllVGroupState(group, socket_path)
    _L0_IPC_ALLTOALLV_BY_GROUP[key] = state
    return state


class _NativeCclThreadedA4:
    """One host-thread native A4 submission and its retained work state."""

    def __init__(self, state, source, output, stream, device_index):
        self.state = state
        self.source = source
        self.output = output
        self.stream = stream
        self.device_index = device_index
        self.started = threading.Event()
        # ``started`` preserves the asynchronous host-submit behavior used by
        # the ordinary two-phase path.  Phase-shared mode additionally waits
        # for ``submitted`` before it enqueues competing XMX work, so it never
        # races oneCCL's host-side submission/progress initialization.
        self.submitted = threading.Event()
        self.work = None
        self.error = None
        self.thread = threading.Thread(target=self._submit, name="moe-native-a4-submit")

    def _submit(self):
        try:
            # XPU current-device and stream state are thread-local.  The
            # native binding releases the GIL while oneCCL submits A4.
            torch.xpu.set_device(self.device_index)
            with torch.xpu.stream(self.stream):
                self.started.set()
                _, self.work = self.state.submit_threaded_all_to_all(
                    self.source, self.output
                )
                self.submitted.set()
        except BaseException as error:
            self.error = error
            self.started.set()
            self.submitted.set()

    def start(self):
        self.thread.start()
        self.started.wait()
        if self.error is not None:
            self.thread.join()
            raise self.error

    def wait_submitted(self):
        """Wait only for host submission, never for collective completion."""

        self.submitted.wait()
        if self.error is not None:
            self.thread.join()
            raise self.error

    def bridge(self, producer_stream):
        self.wait_submitted()
        self.thread.join()
        if self.error is not None:
            raise self.error
        if self.work is None:
            raise RuntimeError("native A4 submission returned no work object")
        self.state.retain_threaded_work(self.work)
        with torch.xpu.stream(producer_stream):
            self.work.wait_stream()
        return self.output


class _NativeCclThreadedA4V:
    """Host-thread exact all-to-all-v A4 with an explicit final bridge.

    The direct oneCCL call can spend host time waiting for the producer fence.
    For the router-free split backward, the independent dW tail may be queued
    during that wait.  This object owns the exact compact source/output and
    runtime split vectors until the real consumer stream is fenced.
    """

    def __init__(
        self,
        state,
        source,
        output,
        input_splits,
        output_splits,
        stream,
        device_index,
    ):
        self.state = state
        self.source = source
        self.output = output
        self.input_splits = tuple(int(value) for value in input_splits)
        self.output_splits = tuple(int(value) for value in output_splits)
        self.stream = stream
        self.device_index = device_index
        self.started = threading.Event()
        self.submitted = threading.Event()
        self.work = None
        self.error = None
        self.thread = threading.Thread(target=self._submit, name="moe-native-a4v-submit")

    def _submit(self):
        try:
            torch.xpu.set_device(self.device_index)
            with torch.xpu.stream(self.stream):
                self.started.set()
                _, self.work = self.state.submit_threaded_all_to_all_v(
                    self.source,
                    self.output,
                    self.input_splits,
                    self.output_splits,
                )
                self.submitted.set()
        except BaseException as error:
            self.error = error
            self.started.set()
            self.submitted.set()

    def start(self):
        self.thread.start()
        self.started.wait()
        if self.error is not None:
            self.thread.join()
            raise self.error

    def wait_submitted(self):
        self.submitted.wait()
        if self.error is not None:
            self.thread.join()
            raise self.error

    def bridge(self, consumer_stream):
        self.wait_submitted()
        self.thread.join()
        if self.error is not None:
            raise self.error
        if self.work is None:
            raise RuntimeError("native A4v submission returned no work object")
        self.state.retain_threaded_work(self.work)
        with torch.xpu.stream(consumer_stream):
            self.work.wait_stream()
        return self.output


def _threaded_native_a4_requested(mesh):
    """Use a host thread only for the validated native-private-stream path."""

    return (
        mesh.group_size["ep_dispatch"] > 1
        and not _l0_ipc_ep_requested(mesh)
        and not _l0_ipc_alltoallv_requested(mesh)
        and _native_ccl_ep_requested()
        and os.environ.get("AURORA_MOE_NATIVE_CCL_PRIVATE_STREAM") == "1"
    )


def _threaded_native_a3v_requested(mesh, phase_controller):
    """Whether forward compact A3 may submit on the validated A4v handoff.

    A3's returned rows are not consumed until route reduction, while the
    shared suffix is independent of them.  The host thread can therefore
    submit the exact native A2Av while the suffix occupies its own stream.
    This remains opt-in because the deferred consumer bridge uses the same
    oneCCL handoff contract as reverse A4.
    """

    value = os.environ.get("AURORA_MOE_THREADED_NATIVE_A3V", "0")
    if value not in ("0", "1"):
        raise ValueError("AURORA_MOE_THREADED_NATIVE_A3V must be '0' or '1'")
    return (
        value == "1"
        and phase_controller is not None
        and _threaded_native_a4_requested(mesh)
        and _native_ccl_alltoallv_requested(mesh)
    )


def _start_threaded_native_a4(mesh, source, producer_stream, state=None):
    """Start native A4 submission after making its private stream depend on phase 1."""

    if state is None:
        state = _native_ccl_ep_state(mesh)
    output = state.prepare_threaded_all_to_all(source)
    stream = _direct_layout_two_phase_a4_stream(source.device)
    with torch.xpu.stream(stream):
        stream.wait_stream(producer_stream)
    launch = _NativeCclThreadedA4(
        state, source, output, stream, source.get_device()
    )
    launch.start()
    return launch


def _start_threaded_native_a4v(
    mesh, source, input_splits, output_splits, producer_stream, state=None
):
    """Start exact compact A4v on a host thread after its producer phase."""

    if state is None:
        state = _native_ccl_ep_state(mesh)
    output = state.prepare_threaded_all_to_all_v(source, output_splits)
    stream = _direct_layout_two_phase_a4_stream(source.device)
    with torch.xpu.stream(stream):
        stream.wait_stream(producer_stream)
    launch = _NativeCclThreadedA4V(
        state,
        source,
        output,
        input_splits,
        output_splits,
        stream,
        source.get_device(),
    )
    launch.start()
    return launch


def _all_to_all(x, mesh, name):
    direct_ccl_requested = _native_ccl_ep_requested()
    if _native_ccl_requested():
        _require_native_ccl_environment()
    with _record(name):
        if mesh.group_size["ep_dispatch"] > 1:
            # The persistent Level Zero path is deliberately BF16-only: the
            # common compact-ID payload carries no separate ID collective,
            # while arbitrary local-expert IDs retain the established int64
            # fallback.  Its device barriers make this exact dynamic payload
            # transport safe without a host barrier or per-call descriptor
            # exchange.
            if (
                _l0_ipc_ep_requested(mesh)
                and x.device.type == "xpu"
                and x.dtype == torch.bfloat16
            ):
                with _record("moe.comm.l0_ipc_alltoall_submit"):
                    return _l0_ipc_ep_group_state(mesh).all_to_all(x.contiguous())
            # The direct equal-size path handles both data payloads and the
            # generic int64 expert-ID fallback.  alltoallv remains on the
            # explicit compact-transport path below.
            if direct_ccl_requested and x.device.type == "xpu" and x.dtype in (
                torch.bfloat16,
                torch.int64,
            ):
                return _native_ccl_ep_state(mesh).all_to_all(x.contiguous())
            y = torch.empty_like(x)
            if os.environ.get("AURORA_MOE_XCCL_ASYNC_EP") == "1":
                # For XCCL this mirrors CUDA ProcessGroup semantics: wait()
                # establishes the device-stream dependency for the output
                # rather than requiring Python to retain a work object until
                # an unknown downstream consumer.  It is opt-in because that
                # behavior must be measured on the installed Aurora stack.
                with _record("moe.comm.xccl_alltoall_async_submit_fence"):
                    work = dist.all_to_all_single(
                        y, x.contiguous(), group=mesh.groups["ep_dispatch"], async_op=True
                    )
                    work.wait()
            else:
                dist.all_to_all_single(y, x.contiguous(), group=mesh.groups["ep_dispatch"])
        else:
            y = torch.empty_like(x)
            y.copy_(x)
        return y


def _all_to_allv_split_sizes_from_cpu(count_matrix_cpu, mesh):
    """Derive exact split descriptors from a host-visible EP count matrix."""

    ep_size = mesh.group_size["ep_dispatch"]
    rank = mesh.group_rank["ep_dispatch"]
    if count_matrix_cpu.shape != (ep_size, ep_size):
        raise ValueError("EP count matrix shape does not match dispatch group")
    send_splits = tuple(int(value) for value in count_matrix_cpu[rank].tolist())
    recv_splits = tuple(int(value) for value in count_matrix_cpu[:, rank].tolist())
    if any(value < 0 for value in send_splits + recv_splits):
        raise ValueError("EP alltoallv split sizes must be nonnegative")
    flat_counts = count_matrix_cpu.reshape(-1)
    uniform_nonzero = bool(
        flat_counts.numel()
        and int(flat_counts[0]) > 0
        and torch.equal(flat_counts, torch.full_like(flat_counts, flat_counts[0]))
    )
    return send_splits, recv_splits, uniform_nonzero


def _all_to_allv_split_sizes(count_matrix, mesh, *, return_count_matrix_cpu=False):
    """Return exact EP splits plus a globally-agreed equal-size predicate.

    ``count_matrix[source, destination]`` is device-resident after the EP
    count all-gather.  ``all_to_all_single`` requires host-side Python split
    sizes, so copy this small control-plane matrix once, synchronously, and
    retain the resulting tuples for the matching backward collectives.  The
    same copy also determines whether every entry in the global matrix is the
    same positive count.  Because every EP rank holds this matrix, that
    predicate is identical on all of them and can safely select the exact
    equal-size transport fast path without another control collective.
    """

    with _record("moe.comm.alltoallv_count_splits_to_cpu"):
        count_matrix_cpu = count_matrix.detach().to(
            device="cpu", dtype=torch.int64, non_blocking=False
        )
    result = _all_to_allv_split_sizes_from_cpu(count_matrix_cpu, mesh)
    if return_count_matrix_cpu:
        return (*result, count_matrix_cpu)
    return result


def _segmented_all_to_allv_split_sizes(
    gathered_destination_expert_counts,
    mesh,
    *,
    return_count_matrix_cpu=False,
    return_counts_cpu=False,
):
    """Return exact EP splits and local true expert rows from one D2H copy.

    Compact segmented routing already needs host split vectors.  Copying its
    richer count tensor instead of only the destination-summed matrix also
    gives the oneMKL path its per-local-expert true row counts before A1.
    That removes its otherwise unavoidable post-A1 ``counts.cpu()`` sync.
    """

    ep_size = mesh.group_size["ep_dispatch"]
    local_count = gathered_destination_expert_counts.size(2)
    expected_shape = (ep_size, ep_size, local_count)
    if tuple(gathered_destination_expert_counts.shape) != expected_shape:
        raise ValueError("segmented EP expert count tensor shape does not match dispatch group")
    with _record("moe.comm.segmented_alltoallv_count_splits_to_cpu"):
        counts_cpu = gathered_destination_expert_counts.detach().to(
            device="cpu", dtype=torch.int64, non_blocking=False
        )
    count_matrix_cpu = counts_cpu.sum(dim=2)
    send_splits, recv_splits, uniform_nonzero = _all_to_allv_split_sizes_from_cpu(
        count_matrix_cpu, mesh
    )
    rank = mesh.group_rank["ep_dispatch"]
    local_expert_rows = tuple(
        int(value) for value in counts_cpu[:, rank, :].sum(dim=0).tolist()
    )
    if sum(local_expert_rows) != sum(recv_splits):
        raise RuntimeError("segmented local expert row counts disagree with receive splits")
    result = (send_splits, recv_splits, uniform_nonzero, local_expert_rows)
    if return_count_matrix_cpu and return_counts_cpu:
        return (*result, count_matrix_cpu, counts_cpu)
    if return_count_matrix_cpu:
        return (*result, count_matrix_cpu)
    return result


def _l0_ipc_alltoallv_metadata_from_count_matrix_cpu(count_matrix_cpu, mesh):
    """Build exact remote compact offsets from ``C[source, destination]``.

    The count matrix is already globally replicated by the route-control
    exchange.  For a source rank ``r`` and destination ``d``, its live input
    range begins at ``sum(C[r, :d])`` and the remote source-major output range
    begins at ``sum(C[:r, d])``.  The reverse exchange swaps these row/column
    roles.  Both capacities are maxima of *observed* compact output rows; they
    size reusable IPC storage only and never add rows to the returned tensor.
    """

    ep_size = mesh.group_size["ep_dispatch"]
    rank = mesh.group_rank["ep_dispatch"]
    if tuple(count_matrix_cpu.shape) != (ep_size, ep_size):
        raise ValueError("EP count matrix shape does not match dispatch group")
    if count_matrix_cpu.dtype != torch.int64:
        count_matrix_cpu = count_matrix_cpu.to(dtype=torch.int64)
    if bool((count_matrix_cpu < 0).any().item()):
        raise ValueError("EP alltoallv count matrix must be nonnegative")
    forward_remote_offsets = tuple(
        int(value)
        for value in count_matrix_cpu[:rank, :].sum(dim=0).tolist()
    )
    reverse_remote_offsets = tuple(
        int(value)
        for value in count_matrix_cpu[:, :rank].sum(dim=1).tolist()
    )
    forward_capacity_rows = int(count_matrix_cpu.sum(dim=0).max().item())
    reverse_capacity_rows = int(count_matrix_cpu.sum(dim=1).max().item())
    return (
        forward_remote_offsets,
        forward_capacity_rows,
        reverse_remote_offsets,
        reverse_capacity_rows,
    )


def _all_to_allv(
    x,
    input_splits,
    output_splits,
    mesh,
    name,
    *,
    exact_equal_fastpath=False,
    l0_remote_receive_offsets=None,
    l0_global_capacity_rows=None,
):
    """Variable-row all-to-all over the EP group with exact split metadata."""

    ep_size = mesh.group_size["ep_dispatch"]
    input_splits = tuple(int(value) for value in input_splits)
    output_splits = tuple(int(value) for value in output_splits)
    if x.ndim < 1:
        raise ValueError("all_to_allv input must have a leading route dimension")
    if len(input_splits) != ep_size or len(output_splits) != ep_size:
        raise ValueError("all_to_allv split count does not match EP group size")
    if any(value < 0 for value in input_splits + output_splits):
        raise ValueError("all_to_allv split sizes must be nonnegative")
    if x.size(0) != sum(input_splits):
        raise ValueError("all_to_allv input rows do not match input split sizes")
    if exact_equal_fastpath:
        expected = input_splits[0] if input_splits else 0
        if (
            expected <= 0
            or any(value != expected for value in input_splits)
            or any(value != expected for value in output_splits)
        ):
            raise ValueError(
                "exact equal all-to-all fast path requires identical positive "
                "input and output row splits"
            )
    with _record(name):
        if ep_size > 1:
            if exact_equal_fastpath:
                # Equal-size all-to-all has the same source/destination block
                # ordering as all-to-all-v here.  This branch never creates
                # rows; it merely avoids all-to-all-v's per-call descriptor
                # path after observing an exactly uniform live route matrix.
                return _all_to_all(x, mesh, f"{name}.exact_equal")
            if (
                _l0_ipc_alltoallv_requested(mesh)
                and x.device.type == "xpu"
                and x.dtype == torch.bfloat16
            ):
                if (
                    l0_remote_receive_offsets is None
                    or l0_global_capacity_rows is None
                ):
                    raise RuntimeError(
                        "exact IPC alltoallv requires count-matrix remote offsets and capacity"
                    )
                with _record("moe.comm.l0_ipc_alltoallv_submit"):
                    return _l0_ipc_alltoallv_group_state(mesh).all_to_all_v(
                        x.contiguous(),
                        input_splits,
                        output_splits,
                        l0_remote_receive_offsets,
                        global_capacity_rows=l0_global_capacity_rows,
                        # A1's compact payload is saved by local expert
                        # autograd until backward.  A3/A2/A4 outputs are
                        # consumed on the current stream before the next
                        # transient exchange, so only the former needs an
                        # independently allocated result under the opt-in
                        # zero-copy transport policy.
                        retain_output="fwd_dispatch" in name,
                    )
            if (
                _native_ccl_alltoallv_requested(mesh)
                and x.device.type == "xpu"
                and x.dtype == torch.bfloat16
            ):
                return _native_ccl_ep_state(mesh).all_to_all_v(
                    x.contiguous(), input_splits, output_splits
                )
            y = x.new_empty((sum(output_splits), *x.shape[1:]))
            dist.all_to_all_single(
                y,
                x.contiguous(),
                output_split_sizes=list(output_splits),
                input_split_sizes=list(input_splits),
                group=mesh.groups["ep_dispatch"],
            )
        else:
            y = x.new_empty((sum(output_splits), *x.shape[1:]))
            if input_splits != output_splits:
                raise ValueError("single-rank all_to_allv must have equal splits")
            y.copy_(x)
    return y


def _all_gather_counts(counts, mesh, *, use_native_count_alltoall=True):
    """Gather a same-width exact EP control vector from every source rank.

    ``counts`` is normally the per-destination vector of width ``EP``.  The
    direct-layout precompute path may instead use width
    ``EP * local_experts`` to make receive-side expert-row bounds available
    before the payload all-to-all.  The normal narrow vector can use the
    native equal all-to-all protocol; the wider pre-dispatch control defaults
    to XCCL unless its dedicated native diagnostic opt-in is set.  Both remain
    device-resident until the one scalar padded-transport shape is required.
    """

    if counts.ndim != 1 or not counts.is_contiguous():
        raise ValueError("EP count control must be a contiguous rank-1 tensor")
    with _record("moe.comm.fwd_count_all_gather"):
        if mesh.group_size["ep_dispatch"] == 1:
            return counts.reshape(1, -1)
        if use_native_count_alltoall and _native_ccl_count_alltoall_requested(mesh):
            ep_size = mesh.group_size["ep_dispatch"]
            if (
                counts.device.type != "xpu"
                or counts.dtype != torch.int64
                or counts.numel() == 0
            ):
                raise ValueError(
                    "native EP count exchange requires a nonempty int64 XPU vector"
                )
            # Every outgoing peer receives this rank's complete control
            # vector.  The equal all-to-all output is therefore
            # ``matrix[source, control_column]``, exactly matching
            # all_gather's source-major stack while retaining the fully
            # dynamic route set.
            with _record("moe.comm.fwd_count_native_alltoall"):
                replicated = counts.reshape(1, -1).expand(ep_size, -1).contiguous()
                return _native_ccl_ep_state(mesh).all_to_all(replicated)
        gathered = [torch.empty_like(counts) for _ in range(mesh.group_size["ep_dispatch"])]
        dist.all_gather(gathered, counts, group=mesh.groups["ep_dispatch"])
        return torch.stack(gathered)


def _wrap_ddp(module, group, group_size, find_unused):
    if group_size == 1 or not dist.is_initialized():
        return module
    # Direct oneCCL reductions are intentionally performed after backward in
    # MOELayer.scale_ddp_grads().  This avoids ProcessGroupXCCL's
    # all_reduce_coalesced path, which is not valid with CCL_OP_SYNC=0 on the
    # current Aurora framework stack.
    if _native_ccl_reducer_requested():
        return module
    kwargs = {
        "process_group": group,
        "find_unused_parameters": find_unused,
        "gradient_as_bucket_view": True,
    }
    # Keep the framework default unchanged unless a performance experiment
    # explicitly asks DDP to coalesce this module's gradients into fewer
    # collectives.  This is useful on Aurora because CCL_OP_SYNC serializes
    # host submission for every individual bucket.
    bucket_cap = os.environ.get("AURORA_MOE_DDP_BUCKET_CAP_MB")
    if bucket_cap is not None:
        try:
            bucket_cap_value = float(bucket_cap)
        except ValueError as error:
            raise ValueError("AURORA_MOE_DDP_BUCKET_CAP_MB must be positive") from error
        if bucket_cap_value <= 0:
            raise ValueError("AURORA_MOE_DDP_BUCKET_CAP_MB must be positive")
        kwargs["bucket_cap_mb"] = bucket_cap_value
    if os.environ.get("AURORA_MOE_DDP_STATIC_GRAPH") == "1":
        kwargs["static_graph"] = True
    return DDP(module, **kwargs)


def _unwrap_ddp(module):
    return module.module if isinstance(module, DDP) else module


def _scale_grad(param, scale):
    if param.grad is not None:
        param.grad.mul_(scale)


class _NativeGradBucket:
    """Persistent flat storage for one direct-oneCCL MoE gradient bucket.

    The old experimental reducer allocated a temporary ``torch.cat`` input
    and a second temporary all-reduce output each training step.  Aside from
    the allocation/copy cost, an output could be released after its oneCCL
    event completed but before the current PyTorch stream executed the queued
    unpack copy.  This bucket keeps one fixed flat buffer for the life of the
    layer and reduces it in place.  Shapes are derived from actual parameters,
    so this remains dynamic with respect to expert count and hidden sizes.
    """

    def __init__(self, name, parameters, group, group_size):
        self.name = name
        self.parameters = tuple(param for param in parameters if param.numel() != 0)
        self.group = group
        self.group_size = group_size
        self.offsets = []
        offset = 0
        for param in self.parameters:
            if param.device.type != "xpu" or param.dtype != torch.bfloat16:
                raise ValueError("native CCL reducer requires BF16 XPU parameters")
            next_offset = offset + param.numel()
            self.offsets.append((offset, next_offset))
            offset = next_offset
        self.numel = offset
        self.flat = (
            torch.empty(self.numel, device=self.parameters[0].device, dtype=torch.bfloat16)
            if self.numel
            else None
        )

    def _gradients(self):
        if self.flat is None:
            return ()
        gradients = []
        for param in self.parameters:
            grad = param.grad
            if grad is None:
                raise RuntimeError(
                    "native CCL reducer requires a gradient for every nonempty parameter"
                )
            if grad.device != self.flat.device or grad.dtype != self.flat.dtype:
                raise ValueError("native CCL reducer gradient device/dtype changed after initialization")
            gradients.append(grad)
        return tuple(gradients)

    def _pack(self):
        for grad, (start, stop) in zip(self._gradients(), self.offsets):
            self.flat[start:stop].copy_(grad.reshape(-1))

    def _unpack(self):
        for grad, (start, stop) in zip(self._gradients(), self.offsets):
            grad.copy_(self.flat[start:stop].reshape_as(grad))

    @property
    def active(self):
        return self.flat is not None and self.group_size > 1

    def warm_up(self):
        """Bootstrap this bucket's direct communicator on the caller stream."""

        if self.active:
            _native_ccl_group_state(self.group).warm_up(self.flat)

    def pack(self):
        if self.active:
            self._pack()

    def launch_sum_(self):
        if not self.active:
            return None
        return _native_ccl_group_state(self.group).all_reduce_sum_work_(self.flat)

    def unpack(self):
        if self.active:
            self._unpack()

    def reduce_sum_(self):
        """Pack parameter gradients, issue one in-place SUM, then unpack."""

        # These copies execute on the current PyTorch stream.  The direct
        # oneCCL bridge records that stream dependency before it launches the
        # collective, and records the reciprocal event dependency before the
        # unpack copies below.  No host wait is needed in the hot path.
        self.pack()
        self.launch_sum_()
        self.unpack()


class _NativeMoEGradReducer:
    """The dense-DP and sparse-DP persistent buckets of one MoE layer."""

    def __init__(self, layer):
        self.dense = _NativeGradBucket(
            "dense",
            (layer.router_weight, layer.shared_up, layer.shared_gate, layer.shared_down),
            layer.mesh.groups["dense_dp"],
            layer.mesh.group_size["dense_dp"],
        )
        self.sparse = _NativeGradBucket(
            "sparse",
            (layer.experts_up, layer.experts_gate, layer.experts_down),
            layer.mesh.groups["sparse_dp"],
            layer.mesh.group_size["sparse_dp"],
        )
        self._two_streams = False
        self._dense_stream = None
        self._sparse_stream = None
        self._synchronize_after_pack = False
        self._serialize_collectives = False

    def enable_two_streams(self):
        """Use private streams only after backward has produced all gradients."""

        if self._two_streams:
            return
        if self.dense.active:
            self._dense_stream = torch.xpu.Stream(device=self.dense.flat.device)
        if self.sparse.active:
            self._sparse_stream = torch.xpu.Stream(device=self.sparse.flat.device)
        self._two_streams = self._dense_stream is not None or self._sparse_stream is not None

    def enable_synchronized_pack(self):
        """Use a device sync instead of producer-to-reducer stream edges.

        This is a correctness diagnostic for the two-stream reducer only.  It
        deliberately destroys overlap so it must not be enabled in a
        performance configuration.
        """

        if not self._two_streams:
            raise RuntimeError(
                "AURORA_MOE_NATIVE_REDUCER_PACK_SYNC=1 requires "
                "AURORA_MOE_NATIVE_REDUCER_TWO_STREAMS=1"
            )
        self._synchronize_after_pack = True

    def enable_serialized_collectives(self):
        """Serialize dense then sparse CCL events without serializing gradient packs.

        The dense and sparse data-parallel communicators overlap in rank
        membership.  On Aurora, concurrently launching their direct-oneCCL
        work from independent reducer streams is a liveness diagnostic, not
        an assumed-safe optimization.  The sparse launch below inherits a
        device dependency on the dense event, so this method adds no host
        wait and retains both normal producer-side packs.
        """

        if not self._two_streams:
            raise RuntimeError(
                "serialized native reducer collectives require "
                "AURORA_MOE_NATIVE_REDUCER_TWO_STREAMS=1"
            )
        self._serialize_collectives = True

    def _reduce_sum_two_streams_(self):
        """Pack on the producer, reduce on private streams, then join/unpack."""

        # KVS bootstrap is host-side control plane work.  Do it in matching
        # dense/sparse order before entering the concurrent device phase.
        self.dense.warm_up()
        self.sparse.warm_up()
        device = self.dense.flat.device if self.dense.flat is not None else self.sparse.flat.device
        producer = torch.xpu.current_stream(device)
        with _record("moe.comm.native_grad_pack"):
            with torch.xpu.stream(producer):
                self.dense.pack()
                self.sparse.pack()

        if self._synchronize_after_pack:
            # Diagnostic only: make the producer writes visible before either
            # reducer stream launches, without inserting P -> D/S edges.  The
            # host wait removes all intended overlap but isolates that edge.
            with _record("moe.comm.native_grad_pack_sync"):
                torch.xpu.synchronize(device)
        else:
            if self._dense_stream is not None:
                self._dense_stream.wait_stream(producer)
            if self._sparse_stream is not None:
                self._sparse_stream.wait_stream(producer)
        dense_work = None
        sparse_work = None
        with _record("moe.comm.native_dense_grad_allreduce"):
            if self._dense_stream is not None:
                with torch.xpu.stream(self._dense_stream):
                    dense_work = self.dense.launch_sum_()
        with _record("moe.comm.native_sparse_grad_allreduce"):
            if self._sparse_stream is not None:
                with torch.xpu.stream(self._sparse_stream):
                    if self._serialize_collectives and dense_work is not None:
                        # ``wait_stream`` inserts an event dependency on the
                        # sparse producer stream.  The subsequent native CCL
                        # launch captures that stream tail as its producer
                        # fence, so sparse communication begins only once the
                        # dense CCL event has completed.  This is device-side
                        # ordering: do not replace it with a host ``wait()``.
                        dense_work.wait_stream()
                    sparse_work = self.sparse.launch_sum_()

        # Both event dependencies are inserted on the producer/default stream
        # before either flat is consumed.  This is intentionally post-backward
        # and has no autograd hooks or host-side event waits.
        with _record("moe.comm.native_grad_unpack"):
            with torch.xpu.stream(producer):
                if dense_work is not None:
                    dense_work.wait_stream()
                if sparse_work is not None:
                    sparse_work.wait_stream()
                self.dense.unpack()
                self.sparse.unpack()

    def reduce_sum_(self):
        # Every rank issues matching collective order, including the
        # zero-shared-expert case where the dense router bucket remains live.
        if self._two_streams:
            self._reduce_sum_two_streams_()
            return
        with _record("moe.comm.native_dense_grad_allreduce"):
            self.dense.reduce_sum_()
        with _record("moe.comm.native_sparse_grad_allreduce"):
            self.sparse.reduce_sum_()


class _RoutedMOE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, topk_scores, topk_indices, up, gate, down, mesh, local_count):
        with _record("moe.local.fwd_route_metadata"):
            flat = x.reshape(-1, x.shape[-1])
            ep_size = mesh.group_size["ep_dispatch"]
            dest = torch.div(topk_indices, local_count, rounding_mode="floor")
            local_ids = topk_indices - dest * local_count
            counts = torch.bincount(dest.reshape(-1), minlength=ep_size).to(torch.long)
            count_matrix = _all_gather_counts(counts, mesh)
            cap = int(count_matrix.max().item())

        with _record("moe.local.fwd_pack_by_destination"):
            send_x = flat.new_zeros((ep_size, cap, flat.shape[-1]))
            send_s = topk_scores.new_zeros((ep_size, cap))
            send_tok = torch.full((ep_size, cap), -1, dtype=torch.long, device=flat.device)
            send_choice = torch.full((ep_size, cap), -1, dtype=torch.long, device=flat.device)
            send_lid = torch.full((ep_size, cap), -1, dtype=torch.long, device=flat.device)
            for rank in range(ep_size):
                token_ids, choice_ids = (dest == rank).nonzero(as_tuple=True)
                n = token_ids.numel()
                send_x[rank, :n] = flat[token_ids]
                send_s[rank, :n] = topk_scores[token_ids, choice_ids]
                send_tok[rank, :n] = token_ids
                send_choice[rank, :n] = choice_ids
                send_lid[rank, :n] = local_ids[token_ids, choice_ids]

        recv_tokens = _all_to_all(send_x, mesh, "moe.comm.fwd_dispatch_tokens")
        recv_scores = _all_to_all(send_s, mesh, "moe.comm.fwd_dispatch_scores")
        recv_lid = _all_to_all(send_lid, mesh, "moe.comm.fwd_dispatch_expert_ids")
        recv_counts = count_matrix[:, mesh.group_rank["ep_dispatch"]]

        with _record("moe.local.fwd_bin_by_expert"):
            flat_tokens = recv_tokens.reshape(ep_size * cap, flat.shape[-1])
            flat_scores = recv_scores.reshape(ep_size * cap)
            flat_lid = recv_lid.reshape(ep_size * cap)
            slot_ids = torch.arange(cap, device=x.device).unsqueeze(0)
            valid_positions = (slot_ids < recv_counts.unsqueeze(1)).reshape(-1).nonzero(as_tuple=True)[0]
            valid_lid = flat_lid[valid_positions]
            local_counts = torch.bincount(valid_lid, minlength=local_count)
            expert_cap = int(local_counts.max().item())

            expert_tokens = flat.new_zeros((local_count, expert_cap, flat.shape[-1]))
            expert_scores = topk_scores.new_zeros((local_count, expert_cap))
            expert_positions = torch.empty((local_count, expert_cap), dtype=torch.long, device=x.device)
            for local_id in range(local_count):
                idx = valid_positions[valid_lid == local_id]
                n = idx.numel()
                if n == 0:
                    continue
                expert_tokens[local_id, :n] = flat_tokens[idx]
                expert_scores[local_id, :n] = flat_scores[idx]
                expert_positions[local_id, :n] = idx

        with _record("moe.local.fwd_expert_compute_and_scatter"):
            hidden_dim = up.shape[-1]
            up_act = expert_tokens.new_zeros((local_count, expert_cap, hidden_dim))
            gate_act = expert_tokens.new_zeros((local_count, expert_cap, hidden_dim))
            hidden = expert_tokens.new_zeros((local_count, expert_cap, hidden_dim))
            raw = flat.new_zeros((local_count, expert_cap, flat.shape[-1]))
            local_flat = flat.new_zeros((ep_size * cap, flat.shape[-1]))
            for local_id in range(local_count):
                n = int(local_counts[local_id].item())
                if n == 0:
                    continue
                tokens_i = expert_tokens[local_id, :n]
                up_i = tokens_i.matmul(up[local_id])
                gate_i = tokens_i.matmul(gate[local_id])
                hidden_i = F.silu(gate_i) * up_i
                raw_i = hidden_i.matmul(down[local_id])
                up_act[local_id, :n] = up_i
                gate_act[local_id, :n] = gate_i
                hidden[local_id, :n] = hidden_i
                raw[local_id, :n] = raw_i
                local_flat[expert_positions[local_id, :n]] = raw_i * expert_scores[local_id, :n].unsqueeze(-1)

        returned = _all_to_all(
            local_flat.reshape(ep_size, cap, flat.shape[-1]),
            mesh,
            "moe.comm.fwd_combine_outputs",
        )
        with _record("moe.local.fwd_reduce_to_tokens"):
            out = flat.new_zeros(flat.shape)
            for rank in range(ep_size):
                n = int(counts[rank].item())
                if n != 0:
                    out.index_add_(0, send_tok[rank, :n], returned[rank, :n])

        ctx.mesh = mesh
        ctx.local_count = local_count
        ctx.ep_size = ep_size
        ctx.cap = cap
        ctx.x_shape = tuple(x.shape)
        ctx.topk_shape = tuple(topk_scores.shape)
        ctx.save_for_backward(
            send_tok,
            send_choice,
            counts,
            local_counts,
            expert_positions,
            expert_tokens,
            expert_scores,
            up,
            gate,
            down,
            up_act,
            gate_act,
            hidden,
            raw,
        )
        return out.reshape_as(x)

    def backward(ctx, grad_out):
        (
            send_tok,
            send_choice,
            counts,
            local_counts,
            expert_positions,
            expert_tokens,
            expert_scores,
            up,
            gate,
            down,
            up_act,
            gate_act,
            hidden,
            raw,
        ) = ctx.saved_tensors
        mesh = ctx.mesh
        ep_size = ctx.ep_size
        cap = ctx.cap
        local_count = ctx.local_count
        model_dim = grad_out.shape[-1]

        with _record("moe.local.bwd_pack_output_grads"):
            grad_flat = grad_out.reshape(-1, model_dim)
            grad_returned = grad_flat.new_zeros((ep_size, cap, model_dim))
            for rank in range(ep_size):
                n = int(counts[rank].item())
                if n != 0:
                    grad_returned[rank, :n] = grad_flat[send_tok[rank, :n]]
        grad_local = _all_to_all(
            grad_returned,
            mesh,
            "moe.comm.bwd_dispatch_output_grads",
        )
        grad_local_flat = grad_local.reshape(ep_size * cap, model_dim)

        with _record("moe.local.bwd_expert_compute_and_scatter"):
            grad_up = torch.zeros_like(up)
            grad_gate = torch.zeros_like(gate)
            grad_down = torch.zeros_like(down)
            grad_recv_tokens_flat = grad_flat.new_zeros((ep_size * cap, model_dim))
            grad_recv_scores_flat = expert_scores.new_zeros(ep_size * cap)
            for local_id in range(local_count):
                n = int(local_counts[local_id].item())
                if n == 0:
                    continue
                pos = expert_positions[local_id, :n]
                grad_expert_out = grad_local_flat[pos]
                raw_i = raw[local_id, :n]
                score_i = expert_scores[local_id, :n]
                grad_recv_scores_flat[pos] = (grad_expert_out * raw_i).sum(-1)
                grad_raw = grad_expert_out * score_i.unsqueeze(-1)

                hidden_i = hidden[local_id, :n]
                down_i = down[local_id]
                grad_down[local_id] = hidden_i.transpose(0, 1).matmul(grad_raw)
                grad_hidden = grad_raw.matmul(down_i.transpose(0, 1))

                gate_i = gate_act[local_id, :n]
                up_i = up_act[local_id, :n]
                grad_up_act = grad_hidden * F.silu(gate_i)
                grad_silu = grad_hidden * up_i
                sigmoid_gate = torch.sigmoid(gate_i)
                grad_gate_act = grad_silu * sigmoid_gate * (1.0 + gate_i * (1.0 - sigmoid_gate))

                tokens_i = expert_tokens[local_id, :n]
                grad_up[local_id] = tokens_i.transpose(0, 1).matmul(grad_up_act)
                grad_gate[local_id] = tokens_i.transpose(0, 1).matmul(grad_gate_act)
                grad_recv_tokens_flat[pos] = (
                    grad_up_act.matmul(up[local_id].transpose(0, 1))
                    + grad_gate_act.matmul(gate[local_id].transpose(0, 1))
                )

        grad_send_x = _all_to_all(
            grad_recv_tokens_flat.reshape(ep_size, cap, model_dim),
            mesh,
            "moe.comm.bwd_combine_input_grads",
        )
        grad_send_scores = _all_to_all(
            grad_recv_scores_flat.reshape(ep_size, cap),
            mesh,
            "moe.comm.bwd_combine_score_grads",
        )
        with _record("moe.local.bwd_reduce_to_tokens"):
            grad_x = grad_flat.new_zeros((grad_flat.shape[0], model_dim))
            grad_scores = expert_scores.new_zeros(ctx.topk_shape)
            grad_scores_flat = grad_scores.reshape(-1)
            top_k = ctx.topk_shape[1]
            for rank in range(ep_size):
                n = int(counts[rank].item())
                if n == 0:
                    continue
                tok = send_tok[rank, :n]
                choice = send_choice[rank, :n]
                grad_x.index_add_(0, tok, grad_send_x[rank, :n])
                grad_scores_flat.index_add_(0, tok * top_k + choice, grad_send_scores[rank, :n])

        return grad_x.reshape(ctx.x_shape), grad_scores, None, grad_up, grad_gate, grad_down, None, None


class _RoutedMOESyclBMM(torch.autograd.Function):
    """EP transport with exact local SYCL routing and XPU BMM experts."""

    @staticmethod
    def forward(ctx, x, topk_scores, topk_indices, up, gate, down, mesh, local_count):
        if x.device.type != "xpu" or x.dtype != torch.bfloat16:
            raise ValueError("sycl_bmm expert backend requires BF16 XPU tensors")

        with _record("moe.local.fwd_route_metadata"):
            flat = x.reshape(-1, x.shape[-1])
            ep_size = mesh.group_size["ep_dispatch"]
            dest = torch.div(topk_indices, local_count, rounding_mode="floor")
            local_ids = topk_indices - dest * local_count
            counts = torch.bincount(dest.reshape(-1), minlength=ep_size).to(torch.long)
            count_matrix = _all_gather_counts(counts, mesh)
            cap = int(count_matrix.max().item())

        with _record("moe.local.fwd_pack_by_destination"):
            send_x = flat.new_zeros((ep_size, cap, flat.shape[-1]))
            send_s = topk_scores.new_zeros((ep_size, cap))
            send_tok = torch.full((ep_size, cap), -1, dtype=torch.long, device=flat.device)
            send_choice = torch.full((ep_size, cap), -1, dtype=torch.long, device=flat.device)
            send_lid = torch.full((ep_size, cap), -1, dtype=torch.long, device=flat.device)
            for rank in range(ep_size):
                token_ids, choice_ids = (dest == rank).nonzero(as_tuple=True)
                n = token_ids.numel()
                send_x[rank, :n] = flat[token_ids]
                send_s[rank, :n] = topk_scores[token_ids, choice_ids]
                send_tok[rank, :n] = token_ids
                send_choice[rank, :n] = choice_ids
                send_lid[rank, :n] = local_ids[token_ids, choice_ids]

        recv_tokens = _all_to_all(send_x, mesh, "moe.comm.fwd_dispatch_tokens")
        recv_scores = _all_to_all(send_s, mesh, "moe.comm.fwd_dispatch_scores")
        recv_lid = _all_to_all(send_lid, mesh, "moe.comm.fwd_dispatch_expert_ids")
        recv_counts = count_matrix[:, mesh.group_rank["ep_dispatch"]]

        with _record("moe.sycl_bmm.fwd_prepare_local_routes"):
            flat_tokens = recv_tokens.reshape(ep_size * cap, flat.shape[-1])
            flat_scores = recv_scores.reshape(ep_size * cap)
            flat_lid = recv_lid.reshape(ep_size * cap)
            slot_ids = torch.arange(cap, device=x.device).unsqueeze(0)
            valid_positions = (slot_ids < recv_counts.unsqueeze(1)).reshape(-1).nonzero(as_tuple=True)[0]
            valid_lid = flat_lid[valid_positions]

        has_valid_routes = valid_positions.numel() != 0
        if has_valid_routes:
            from aurora_moe._kernels.padded_bmm_moe import exact_routed_expert_bmm, make_exact_route_plan

            with _record("moe.sycl_bmm.fwd_exact_local_experts"):
                with torch.enable_grad():
                    expert_tokens = flat_tokens.index_select(0, valid_positions).detach().requires_grad_(True)
                    expert_scores = (
                        flat_scores.index_select(0, valid_positions)
                        .detach()
                        .reshape(-1, 1)
                        .requires_grad_(True)
                    )
                    # Detached leaves keep nested autograd separate from DDP.
                    expert_up = up.detach().requires_grad_(True)
                    expert_gate = gate.detach().requires_grad_(True)
                    expert_down = down.detach().requires_grad_(True)
                    local_indices = valid_lid.to(torch.long).reshape(-1, 1).contiguous()
                    plan = make_exact_route_plan(local_indices, local_count)
                    weighted_valid = exact_routed_expert_bmm(
                        expert_tokens,
                        expert_scores,
                        expert_up,
                        expert_gate,
                        expert_down,
                        plan,
                        activation="swiglu",
                    )
            local_flat = flat.new_zeros((ep_size * cap, flat.shape[-1]))
            local_flat.index_copy_(0, valid_positions, weighted_valid.detach())
        else:
            expert_tokens = flat.new_empty((0, flat.shape[-1]), requires_grad=True)
            expert_scores = topk_scores.new_empty((0, 1), requires_grad=True)
            expert_up = up.detach().requires_grad_(True)
            expert_gate = gate.detach().requires_grad_(True)
            expert_down = down.detach().requires_grad_(True)
            weighted_valid = None
            local_flat = flat.new_zeros((ep_size * cap, flat.shape[-1]))

        returned = _all_to_all(
            local_flat.reshape(ep_size, cap, flat.shape[-1]),
            mesh,
            "moe.comm.fwd_combine_outputs",
        )
        with _record("moe.local.fwd_reduce_to_tokens"):
            out = flat.new_zeros(flat.shape)
            for rank in range(ep_size):
                n = int(counts[rank].item())
                if n != 0:
                    out.index_add_(0, send_tok[rank, :n], returned[rank, :n])

        ctx.mesh = mesh
        ctx.ep_size = ep_size
        ctx.cap = cap
        ctx.x_shape = tuple(x.shape)
        ctx.topk_shape = tuple(topk_scores.shape)
        ctx.has_valid_routes = has_valid_routes
        ctx.weighted_valid = weighted_valid
        ctx.save_for_backward(
            send_tok,
            send_choice,
            counts,
            valid_positions,
            expert_tokens,
            expert_scores,
            expert_up,
            expert_gate,
            expert_down,
            up,
            gate,
            down,
        )
        return out.reshape_as(x)

    @staticmethod
    def backward(ctx, grad_out):
        (
            send_tok,
            send_choice,
            counts,
            valid_positions,
            expert_tokens,
            expert_scores,
            expert_up,
            expert_gate,
            expert_down,
            up,
            gate,
            down,
        ) = ctx.saved_tensors
        mesh = ctx.mesh
        ep_size = ctx.ep_size
        cap = ctx.cap
        model_dim = grad_out.shape[-1]

        with _record("moe.local.bwd_pack_output_grads"):
            grad_flat = grad_out.reshape(-1, model_dim)
            grad_returned = grad_flat.new_zeros((ep_size, cap, model_dim))
            for rank in range(ep_size):
                n = int(counts[rank].item())
                if n != 0:
                    grad_returned[rank, :n] = grad_flat[send_tok[rank, :n]]
        grad_local = _all_to_all(
            grad_returned,
            mesh,
            "moe.comm.bwd_dispatch_output_grads",
        )
        grad_local_flat = grad_local.reshape(ep_size * cap, model_dim)

        with _record("moe.sycl_bmm.bwd_exact_local_experts"):
            grad_recv_tokens_flat = grad_flat.new_zeros((ep_size * cap, model_dim))
            grad_recv_scores_flat = expert_scores.new_zeros(ep_size * cap)
            grad_up = torch.zeros_like(up) if ctx.needs_input_grad[3] else None
            grad_gate = torch.zeros_like(gate) if ctx.needs_input_grad[4] else None
            grad_down = torch.zeros_like(down) if ctx.needs_input_grad[5] else None

            if ctx.has_valid_routes:
                grad_weighted_valid = grad_local_flat.index_select(0, valid_positions)
                with torch.enable_grad():
                    (
                        grad_tokens,
                        grad_scores,
                        grad_expert_up,
                        grad_expert_gate,
                        grad_expert_down,
                    ) = torch.autograd.grad(
                        ctx.weighted_valid,
                        (expert_tokens, expert_scores, expert_up, expert_gate, expert_down),
                        grad_weighted_valid,
                        allow_unused=True,
                    )
                if grad_tokens is not None:
                    grad_recv_tokens_flat.index_copy_(0, valid_positions, grad_tokens)
                if grad_scores is not None:
                    grad_recv_scores_flat.index_copy_(0, valid_positions, grad_scores.reshape(-1))
                if grad_up is not None and grad_expert_up is not None:
                    grad_up = grad_expert_up
                if grad_gate is not None and grad_expert_gate is not None:
                    grad_gate = grad_expert_gate
                if grad_down is not None and grad_expert_down is not None:
                    grad_down = grad_expert_down

        grad_send_x = _all_to_all(
            grad_recv_tokens_flat.reshape(ep_size, cap, model_dim),
            mesh,
            "moe.comm.bwd_combine_input_grads",
        )
        grad_send_scores = _all_to_all(
            grad_recv_scores_flat.reshape(ep_size, cap),
            mesh,
            "moe.comm.bwd_combine_score_grads",
        )
        with _record("moe.local.bwd_reduce_to_tokens"):
            grad_x = grad_flat.new_zeros((grad_flat.shape[0], model_dim))
            grad_scores_out = expert_scores.new_zeros(ctx.topk_shape)
            grad_scores_flat = grad_scores_out.reshape(-1)
            top_k = ctx.topk_shape[1]
            for rank in range(ep_size):
                n = int(counts[rank].item())
                if n == 0:
                    continue
                tok = send_tok[rank, :n]
                choice = send_choice[rank, :n]
                grad_x.index_add_(0, tok, grad_send_x[rank, :n])
                grad_scores_flat.index_add_(0, tok * top_k + choice, grad_send_scores[rank, :n])

        return (
            grad_x.reshape(ctx.x_shape) if ctx.needs_input_grad[0] else None,
            grad_scores_out if ctx.needs_input_grad[1] else None,
            None,
            grad_up,
            grad_gate,
            grad_down,
            None,
            None,
        )


class _RoutedMOESyclEP(torch.autograd.Function):
    """Exact EP transport with device-side route movement and local XPU BMMs.

    The default transport has exact observed padding and no route clipping.
    ``AURORA_MOE_ALLTOALLV=1`` selects a compact alltoallv layout with no
    transport padding.  Both paths replace source-side Python destination
    loops and contended ``index_add_`` calls with owner-per-output SYCL
    kernels and fuse token/score payloads where possible.
    """

    @staticmethod
    def forward(
        ctx,
        x,
        topk_scores,
        topk_indices,
        up,
        gate,
        down,
        mesh,
        local_count,
        phase_controller=None,
    ):
        if x.device.type != "xpu" or x.dtype != torch.bfloat16:
            raise ValueError("sycl_ep expert backend requires BF16 XPU tensors")
        if topk_scores.dtype != torch.bfloat16:
            raise ValueError("sycl_ep expert backend requires BF16 router scores")
        direct_layout_two_phase = _direct_layout_two_phase_requested()
        direct_layout = _direct_expert_layout_requested()
        precompute_expert_counts = _direct_layout_precompute_expert_counts_requested()
        dedicated_native_a4 = _dedicated_native_a4_requested(
            mesh, direct_layout_two_phase
        )
        if os.environ.get("AURORA_MOE_ALLTOALLV") == "1":
            if precompute_expert_counts:
                raise RuntimeError(
                    "AURORA_MOE_DIRECT_LAYOUT_PRECOMPUTE_EXPERT_COUNTS is unavailable "
                    "with AURORA_MOE_ALLTOALLV=1"
                )
            if direct_layout_two_phase:
                raise RuntimeError(
                    "AURORA_MOE_DIRECT_LAYOUT_TWO_PHASE is unavailable with "
                    "AURORA_MOE_ALLTOALLV=1"
                )
            return _RoutedMOESyclEP._forward_alltoallv(
                ctx,
                x,
                topk_scores,
                topk_indices,
                up,
                gate,
                down,
                mesh,
                local_count,
                phase_controller=phase_controller,
            )

        from aurora_moe._kernels.ep_route_ops import (
            destination_counts,
            destination_expert_counts,
            make_route_slots,
            pack_routes_payload,
            pack_routes_payload_ids,
            reduce_route_rows,
        )

        known_group_rows = None
        known_max_rows = None
        known_max_tail = None
        with _record("moe.sycl_ep.fwd_route_metadata"):
            flat = x.reshape(-1, x.shape[-1]).contiguous()
            ep_size = mesh.group_size["ep_dispatch"]
            dest = torch.div(topk_indices, local_count, rounding_mode="floor").contiguous()
            local_ids = (topk_indices - dest * local_count).contiguous()
            if direct_layout and precompute_expert_counts:
                # The source owns both the destination rank and the
                # destination-local expert ID for every route.  Exchange the
                # exact 2-D histogram once, before A1, so direct layout never
                # has to count received IDs or synchronously discover its BMM
                # row shape after A1.  The payload width is still the exact
                # observed maximum destination count; this is not a capacity
                # factor and never changes the route set.
                local_destination_expert_counts = destination_expert_counts(
                    dest, local_ids, ep_size, local_count
                ).contiguous()
                gathered_destination_expert_counts = _all_gather_counts(
                    local_destination_expert_counts.reshape(-1).contiguous(),
                    mesh,
                    use_native_count_alltoall=(
                        _direct_layout_precompute_native_count_alltoall_requested()
                    ),
                ).reshape(ep_size, ep_size, local_count)
                destination_count_matrix = gathered_destination_expert_counts.sum(dim=2)
                recv_counts = destination_count_matrix[
                    :, mesh.group_rank["ep_dispatch"]
                ].contiguous()
                recv_expert_counts = gathered_destination_expert_counts[
                    :, mesh.group_rank["ep_dispatch"], :
                ].sum(dim=0).contiguous()
                # One small control readback replaces the old cap readback
                # plus the direct layout's second receive-side readback.  It
                # occurs before dispatch, leaving the A1->pack->BMM path
                # entirely device-scheduled.
                shape_control = torch.stack(
                    (
                        destination_count_matrix.max(),
                        recv_expert_counts.max(),
                        recv_expert_counts.min(),
                    )
                ).detach().to(device="cpu", dtype=torch.int64, non_blocking=False)
                cap, known_max_rows, known_min_rows = (
                    int(value) for value in shape_control.tolist()
                )
                if known_max_rows > torch.iinfo(torch.int32).max:
                    raise RuntimeError("direct expert row count exceeds int32 layout storage")
                known_max_tail = known_max_rows - known_min_rows
                known_group_rows = recv_expert_counts.to(torch.int32).contiguous()
            else:
                counts = destination_counts(dest, ep_size)
                count_matrix = _all_gather_counts(counts, mesh)
                # The shape of the exact padded transport is the only
                # required scalar control-plane synchronization.  All route
                # assignments and data movement below remain on the device.
                cap = int(count_matrix.max().item())
                recv_counts = count_matrix[:, mesh.group_rank["ep_dispatch"]].contiguous()
            route_slots = make_route_slots(dest, ep_size, cap)

        rows = ep_size * cap
        # BF16 represents integer local expert IDs through 256 exactly.  For
        # the overwhelmingly common case below that bound, carry the ID in
        # the token/score payload and remove an entire EP all-to-all.  The
        # arbitrary-local-expert-count path below deliberately retains the
        # int64 transport, so this is a layout optimization rather than a
        # model-shape restriction.
        small_id_payload = 1 <= local_count <= 256
        with _record("moe.sycl_ep.fwd_pack_exact_routes"):
            if small_id_payload:
                send_payload = pack_routes_payload_ids(
                    flat,
                    topk_scores.contiguous(),
                    local_ids,
                    route_slots,
                    rows,
                    local_count,
                )
            else:
                send = pack_routes_payload(
                    flat,
                    topk_scores.contiguous(),
                    local_ids,
                    route_slots,
                    rows,
                )

        if phase_controller is not None:
            phase_controller.mark_dispatch_ready(torch.xpu.current_stream(x.device))
            if _phase_shared_forward_reorder_requested():
                phase_controller.start_prefix_after_dispatch()

        if dedicated_native_a4:
            warmup_source = send_payload.payload if small_id_payload else send.payload
            with _record("moe.comm.native_dedicated_a4_prewarm"):
                _native_ccl_dedicated_a4_state(mesh).warm_up_once(
                    warmup_source.detach().contiguous()
                )

        if small_id_payload:
            recv_payload = _all_to_all(
                send_payload.payload.reshape(ep_size, cap, flat.shape[-1] + 2),
                mesh,
                "moe.comm.sycl_ep_fwd_dispatch_token_scores_ids",
            )
        else:
            recv_payload = _all_to_all(
                send.payload.reshape(ep_size, cap, flat.shape[-1] + 1),
                mesh,
                "moe.comm.sycl_ep_fwd_dispatch_token_scores",
            )
            recv_lid = _all_to_all(
                send.expert_ids.reshape(ep_size, cap),
                mesh,
                "moe.comm.sycl_ep_fwd_dispatch_expert_ids",
            )
        if phase_controller is not None and not _phase_shared_forward_reorder_requested():
            phase_controller.start_prefix_after_dispatch()

        if direct_layout:
            # Convert the received source-major padded rows directly to an
            # expert-major dynamic BMM layout.  This replaces receive compact,
            # sort, gather, padded scatter, and their reverse-side analogues
            # without changing the exact observed route set.
            from aurora_moe._kernels.direct_layout_bmm_moe import (
                direct_padded_expert_bmm,
                prepare_direct_padded_expert_bmm_two_phase,
            )

            with _record("moe.sycl_ep.fwd_direct_expert_layout"):
                if small_id_payload:
                    direct_payload = (
                        recv_payload.reshape(rows, flat.shape[-1] + 2)
                        .contiguous()
                        .detach()
                        .requires_grad_(True)
                    )
                    direct_local_ids = None
                else:
                    direct_payload = (
                        recv_payload.reshape(rows, flat.shape[-1] + 1)
                        .contiguous()
                        .detach()
                        .requires_grad_(True)
                    )
                    direct_local_ids = recv_lid.reshape(rows).contiguous()
                if phase_controller is not None:
                    phase_controller.wait_prefix_before_local_xmx(
                        torch.xpu.current_stream(x.device)
                    )
                if direct_layout_two_phase:
                    expert_up = up.detach().requires_grad_(True)
                    expert_gate = gate.detach().requires_grad_(True)
                    expert_down = down.detach().requires_grad_(True)
                    weighted_valid, two_phase_state = (
                        prepare_direct_padded_expert_bmm_two_phase(
                            direct_payload,
                            recv_counts,
                            cap,
                            local_count,
                            expert_up,
                            expert_gate,
                            expert_down,
                            local_ids=direct_local_ids,
                            activation="swiglu",
                            known_group_rows=known_group_rows,
                            known_max_rows=known_max_rows,
                            known_max_tail=known_max_tail,
                        )
                    )
                else:
                    with torch.enable_grad():
                        expert_up = up.detach().requires_grad_(True)
                        expert_gate = gate.detach().requires_grad_(True)
                        expert_down = down.detach().requires_grad_(True)
                        weighted_valid = direct_padded_expert_bmm(
                            direct_payload,
                            recv_counts,
                            cap,
                            local_count,
                            expert_up,
                            expert_gate,
                            expert_down,
                            local_ids=direct_local_ids,
                            activation="swiglu",
                            known_group_rows=known_group_rows,
                            known_max_rows=known_max_rows,
                            known_max_tail=known_max_tail,
                        )
                local_flat = weighted_valid.detach()
            # The direct custom autograd node handles an empty received route
            # set as a differentiable all-zero result, so no special BMM
            # branch or capacity-derived route clipping is necessary.
            has_valid_routes = True
        else:
            with _record("moe.sycl_ep.fwd_prepare_local_routes"):
                if small_id_payload:
                    from aurora_moe._kernels.ep_local_ops import compact_payload_rows_with_ids_from_counts

                    flat_payload = recv_payload.reshape(rows, flat.shape[-1] + 2).contiguous()
                    compact = compact_payload_rows_with_ids_from_counts(
                        flat_payload, recv_counts, cap
                    )
                    valid_lid = compact.local_ids
                    # This placeholder is retained only for the generic-ID
                    # fallback's saved-tensor layout; the small-ID path never
                    # materializes a position vector.
                    valid_positions = torch.empty(0, device=x.device, dtype=torch.long)
                    has_valid_routes = compact.tokens.numel() != 0
                else:
                    flat_payload = recv_payload.reshape(rows, flat.shape[-1] + 1)
                    flat_lid = recv_lid.reshape(rows)
                    slot_ids = torch.arange(cap, device=x.device).unsqueeze(0)
                    valid_positions = (slot_ids < recv_counts.unsqueeze(1)).reshape(-1).nonzero(as_tuple=True)[0]
                    valid_lid = flat_lid.index_select(0, valid_positions)
                    has_valid_routes = valid_positions.numel() != 0

            if has_valid_routes:
                if small_id_payload:
                    from aurora_moe._kernels.ep_local_ops import expand_rows_to_counts
                else:
                    from aurora_moe._kernels.ep_local_ops import gather_payload_rows, scatter_rows
                from aurora_moe._kernels.padded_bmm_moe import exact_routed_expert_bmm, make_exact_route_plan

                with _record("moe.sycl_ep.fwd_exact_local_experts"):
                    # Nested leaves make this custom autograd function explicit
                    # while keeping DDP's parameter hooks on the outer module.
                    with torch.enable_grad():
                        if not small_id_payload:
                            compact = gather_payload_rows(flat_payload, valid_positions)
                        expert_tokens = compact.tokens.detach().requires_grad_(True)
                        expert_scores = (
                            compact.scores.detach()
                            .reshape(-1, 1)
                            .requires_grad_(True)
                        )
                        expert_up = up.detach().requires_grad_(True)
                        expert_gate = gate.detach().requires_grad_(True)
                        expert_down = down.detach().requires_grad_(True)
                        local_indices = valid_lid.to(torch.long).reshape(-1, 1).contiguous()
                        plan = make_exact_route_plan(local_indices, local_count)
                        if phase_controller is not None:
                            phase_controller.wait_prefix_before_local_xmx(
                                torch.xpu.current_stream(x.device)
                            )
                        weighted_valid = exact_routed_expert_bmm(
                            expert_tokens,
                            expert_scores,
                            expert_up,
                            expert_gate,
                            expert_down,
                            plan,
                            activation="swiglu",
                        )
                if small_id_payload:
                    local_flat = expand_rows_to_counts(
                        weighted_valid.detach().contiguous(), recv_counts, cap
                    )
                else:
                    local_flat = scatter_rows(
                        weighted_valid.detach().contiguous(), valid_positions, rows
                    )
            else:
                expert_tokens = flat.new_empty((0, flat.shape[-1]), requires_grad=True)
                expert_scores = topk_scores.new_empty((0, 1), requires_grad=True)
                expert_up = up.detach().requires_grad_(True)
                expert_gate = gate.detach().requires_grad_(True)
                expert_down = down.detach().requires_grad_(True)
                weighted_valid = None
                local_flat = flat.new_zeros((rows, flat.shape[-1]))

        if phase_controller is not None:
            phase_controller.mark_return_ready(torch.xpu.current_stream(x.device))
            if _phase_shared_forward_reorder_requested():
                phase_controller.start_suffix_after_return()
        returned = _all_to_all(
            local_flat.reshape(ep_size, cap, flat.shape[-1]),
            mesh,
            "moe.comm.sycl_ep_fwd_combine_outputs",
        )
        if phase_controller is not None and not _phase_shared_forward_reorder_requested():
            phase_controller.start_suffix_after_return()
        with _record("moe.sycl_ep.fwd_reduce_exact_routes"):
            out = reduce_route_rows(returned.reshape(rows, flat.shape[-1]), route_slots)

        ctx.mesh = mesh
        ctx.ep_size = ep_size
        ctx.cap = cap
        ctx.x_shape = tuple(x.shape)
        ctx.has_valid_routes = has_valid_routes
        ctx.small_id_payload = small_id_payload
        ctx.direct_layout = direct_layout
        ctx.direct_layout_two_phase = direct_layout_two_phase
        ctx.phase_controller = phase_controller
        ctx.dedicated_native_a4 = dedicated_native_a4
        ctx.direct_layout_two_phase_state = two_phase_state if direct_layout_two_phase else None
        ctx.weighted_valid = None if direct_layout_two_phase else weighted_valid
        if direct_layout:
            ctx.save_for_backward(
                route_slots,
                recv_counts,
                direct_payload,
                expert_up,
                expert_gate,
                expert_down,
                up,
                gate,
                down,
            )
        else:
            ctx.save_for_backward(
                route_slots,
                recv_counts,
                valid_positions,
                expert_tokens,
                expert_scores,
                expert_up,
                expert_gate,
                expert_down,
                up,
                gate,
                down,
            )
        return out.reshape_as(x)

    @staticmethod
    def _forward_alltoallv(
        ctx,
        x,
        topk_scores,
        topk_indices,
        up,
        gate,
        down,
        mesh,
        local_count,
        *,
        ragged_backend=False,
        phase_controller=None,
    ):
        """Run the exact non-padded EP transport selected by AURORA_MOE_ALLTOALLV."""

        from aurora_moe._kernels.ep_route_ops import (
            destination_counts,
            destination_expert_counts,
            make_compact_destination_expert_slots,
            make_compact_route_slots,
            make_destination_expert_offsets,
            pack_compact_tokens_scores,
            pack_compact_tokens_score_payload,
            pack_compact_routes_payload,
            pack_compact_routes_payload_ids,
            reduce_route_rows,
        )

        # This path deliberately uses the Sonic-style source/expert physical
        # layout only with the ragged backend.  A sender packs every peer block
        # as contiguous local-expert segments, so the receiver can infer both
        # the expert and its exact row range from the exchanged [peer, expert]
        # count matrix.  There is no ID payload, expert-major D-wide reorder,
        # capacity factor, or maximum-row padding.
        segmented_backend = (
            ragged_backend and os.environ.get("AURORA_MOE_SEGMENTED_SONIC") == "1"
        )
        # This keeps the exact source/expert transport but turns its compact
        # physical receive buffer into one true-length expert-major segment
        # per local expert before the PVC GEMMs.  It is deliberately a second
        # opt-in: the direct physical-fragment path remains a useful A/B for
        # the reorder-versus-fragmentation tradeoff.
        expert_major_segmented_backend = (
            segmented_backend
            and os.environ.get("AURORA_MOE_SEGMENTED_EXPERT_MAJOR") == "1"
        )
        # The expert-major path can retain the score beside each token through
        # the exact all-to-all-v.  Its fused source/expert permutation restores
        # a dense D-column GEMM operand, so this removes the scalar collective
        # without feeding a strided D+1 tensor to GEMM.  Keep it independently
        # opt-in until the XPU correctness gate has exercised every backend.
        expert_major_fused_payload = (
            expert_major_segmented_backend
            and os.environ.get("AURORA_MOE_SEGMENTED_FUSED_SCORE_PAYLOAD") == "1"
        )
        direct_expert_major_ipc = (
            expert_major_fused_payload and _l0_ipc_expert_major_requested(mesh)
        )

        with _record("moe.sycl_ep_alltoallv.fwd_route_metadata"):
            flat = x.reshape(-1, x.shape[-1]).contiguous()
            ep_size = mesh.group_size["ep_dispatch"]
            destination = torch.div(
                topk_indices, local_count, rounding_mode="floor"
            ).contiguous()
            local_ids = (topk_indices - destination * local_count).contiguous()
            if segmented_backend:
                local_destination_expert_counts = destination_expert_counts(
                    destination, local_ids, ep_size, local_count
                ).contiguous()
                gathered_destination_expert_counts = _all_gather_counts(
                    local_destination_expert_counts.reshape(-1).contiguous(), mesh
                ).reshape(ep_size, ep_size, local_count)
                # Count matrix shape stays [source, destination], which is
                # exactly what all_to_all_single requires for dynamic splits.
                # Keep the richer tensor device-resident for the local grouped
                # GEMMs; only this narrow matrix takes the existing mandatory
                # control-plane CPU path for split sizes.
                send_counts = local_destination_expert_counts.sum(dim=1).contiguous()
                split_metadata = _segmented_all_to_allv_split_sizes(
                    gathered_destination_expert_counts,
                    mesh,
                    return_count_matrix_cpu=True,
                    return_counts_cpu=direct_expert_major_ipc,
                )
                if direct_expert_major_ipc:
                    (
                        send_splits,
                        recv_splits,
                        uniform_nonzero_counts,
                        local_expert_rows,
                        count_matrix_cpu,
                        expert_count_matrix_cpu,
                    ) = split_metadata
                else:
                    (
                        send_splits,
                        recv_splits,
                        uniform_nonzero_counts,
                        local_expert_rows,
                        count_matrix_cpu,
                    ) = split_metadata
            else:
                send_counts = destination_counts(destination, ep_size)
                count_matrix = _all_gather_counts(send_counts, mesh)
                (
                    send_splits,
                    recv_splits,
                    uniform_nonzero_counts,
                    count_matrix_cpu,
                ) = _all_to_allv_split_sizes(
                    count_matrix,
                    mesh,
                    return_count_matrix_cpu=True,
                )
                local_expert_rows = None
            (
                l0_forward_remote_offsets,
                l0_forward_capacity_rows,
                l0_reverse_remote_offsets,
                l0_reverse_capacity_rows,
            ) = _l0_ipc_alltoallv_metadata_from_count_matrix_cpu(
                count_matrix_cpu, mesh
            )
            # The matrix is globally replicated by the route-count exchange,
            # so this branch choice is collective-order safe.  It remains an
            # opt-in experimental transport selection, not a routing policy.
            exact_equal_fastpath = (
                _exact_equal_alltoall_fastpath_requested() and uniform_nonzero_counts
            )
            destination_offsets = torch.cat(
                (
                    send_counts.new_zeros(1),
                    torch.cumsum(send_counts, dim=0)[:-1],
                )
            ).contiguous()
            if segmented_backend:
                destination_expert_offsets = make_destination_expert_offsets(
                    local_destination_expert_counts, destination_offsets
                )
                route_slots = make_compact_destination_expert_slots(
                    destination, local_ids, destination_expert_offsets
                )
                source_expert_counts = gathered_destination_expert_counts[
                    :, mesh.group_rank["ep_dispatch"], :
                ].to(torch.int32).contiguous()
                if direct_expert_major_ipc:
                    from aurora_moe._kernels.level_zero_ipc_alltoallv import (
                        make_expert_major_ipc_block_plan,
                    )

                    l0_expert_major_plan = make_expert_major_ipc_block_plan(
                        expert_count_matrix_cpu, mesh.group_rank["ep_dispatch"]
                    )
                else:
                    l0_expert_major_plan = None
            else:
                route_slots = make_compact_route_slots(destination, destination_offsets)
                l0_expert_major_plan = None
            route_rows = route_slots.numel()
            if sum(send_splits) != route_rows:
                raise RuntimeError("compact EP send splits do not cover every route")

        # BF16 carries the local ID exactly through 255.  Larger local expert
        # sets use the generic int64 transport below without any shape limit.
        small_id_payload = not segmented_backend and 1 <= local_count <= 256
        with _record("moe.sycl_ep_alltoallv.fwd_pack_exact_routes"):
            if segmented_backend:
                if expert_major_fused_payload:
                    segmented_payload_send = pack_compact_tokens_score_payload(
                        flat, topk_scores.contiguous(), route_slots
                    )
                else:
                    segmented_send = pack_compact_tokens_scores(
                        flat, topk_scores.contiguous(), route_slots
                    )
            elif small_id_payload:
                send_payload = pack_compact_routes_payload_ids(
                    flat,
                    topk_scores.contiguous(),
                    local_ids,
                    route_slots,
                    local_count,
                )
            else:
                send = pack_compact_routes_payload(
                    flat,
                    topk_scores.contiguous(),
                    local_ids,
                    route_slots,
                )

        # The shared-prefix reads only x and shared weights, so it can run
        # while A1 is in flight.  Its producer event must include the route
        # pack because native oneCCL's private stream consumes that pack
        # immediately after this point.  The controller deliberately gates
        # the local expert GEMMs below to avoid competing for XMX execution.
        if phase_controller is not None:
            phase_controller.mark_dispatch_ready(torch.xpu.current_stream(x.device))
            if _phase_shared_forward_reorder_requested():
                phase_controller.start_prefix_after_dispatch()

        if segmented_backend:
            if expert_major_fused_payload:
                if direct_expert_major_ipc:
                    assert l0_expert_major_plan is not None
                    with _record(
                        "moe.comm.l0_ipc_expert_major_fwd_dispatch_payload"
                    ):
                        recv_payload = _l0_ipc_alltoallv_group_state(
                            mesh
                        ).all_to_all_v_expert_major(
                            segmented_payload_send,
                            l0_expert_major_plan,
                            inverse=False,
                            retain_output=True,
                        )
                else:
                    recv_payload = _all_to_allv(
                        segmented_payload_send,
                        send_splits,
                        recv_splits,
                        mesh,
                        "moe.comm.sycl_segmented_expert_major_fwd_dispatch_payload",
                        exact_equal_fastpath=exact_equal_fastpath,
                        l0_remote_receive_offsets=l0_forward_remote_offsets,
                        l0_global_capacity_rows=l0_forward_capacity_rows,
                    )
            else:
                # Tokens remain a normal [routes, D] tensor for the direct
                # physical-fragment path; its tiny score buffer uses a
                # separate collective rather than a strided GEMM input.
                recv_tokens = _all_to_allv(
                    segmented_send.tokens,
                    send_splits,
                    recv_splits,
                    mesh,
                    "moe.comm.sycl_segmented_fwd_dispatch_tokens",
                    exact_equal_fastpath=exact_equal_fastpath,
                    l0_remote_receive_offsets=l0_forward_remote_offsets,
                    l0_global_capacity_rows=l0_forward_capacity_rows,
                )
                recv_scores = _all_to_allv(
                    segmented_send.scores,
                    send_splits,
                    recv_splits,
                    mesh,
                    "moe.comm.sycl_segmented_fwd_dispatch_scores",
                    exact_equal_fastpath=exact_equal_fastpath,
                    l0_remote_receive_offsets=l0_forward_remote_offsets,
                    l0_global_capacity_rows=l0_forward_capacity_rows,
                )
        elif small_id_payload:
            recv_payload = _all_to_allv(
                send_payload.payload,
                send_splits,
                recv_splits,
                mesh,
                "moe.comm.sycl_ep_alltoallv_fwd_dispatch_token_scores_ids",
                exact_equal_fastpath=exact_equal_fastpath,
                l0_remote_receive_offsets=l0_forward_remote_offsets,
                l0_global_capacity_rows=l0_forward_capacity_rows,
            )
            from aurora_moe._kernels.ep_local_ops import split_compact_payload_rows_with_ids

            with _record("moe.sycl_ep_alltoallv.fwd_prepare_local_routes"):
                compact = split_compact_payload_rows_with_ids(recv_payload.contiguous())
                valid_lid = compact.local_ids
        else:
            recv_payload = _all_to_allv(
                send.payload,
                send_splits,
                recv_splits,
                mesh,
                "moe.comm.sycl_ep_alltoallv_fwd_dispatch_token_scores",
                exact_equal_fastpath=exact_equal_fastpath,
                l0_remote_receive_offsets=l0_forward_remote_offsets,
                l0_global_capacity_rows=l0_forward_capacity_rows,
            )
            recv_lid = _all_to_allv(
                send.expert_ids,
                send_splits,
                recv_splits,
                mesh,
                "moe.comm.sycl_ep_alltoallv_fwd_dispatch_expert_ids",
                exact_equal_fastpath=exact_equal_fastpath,
            )
            from aurora_moe._kernels.ep_local_ops import split_compact_payload_rows

            with _record("moe.sycl_ep_alltoallv.fwd_prepare_local_routes"):
                compact = split_compact_payload_rows(recv_payload.contiguous())
                valid_lid = recv_lid.contiguous()

        if phase_controller is not None and not _phase_shared_forward_reorder_requested():
            # Submission has returned, so enqueue the independent shared
            # prefix on its private PyTorch stream while oneCCL advances A1.
            phase_controller.start_prefix_after_dispatch()

        if segmented_backend:
            if expert_major_fused_payload:
                if recv_payload.size(0) != sum(recv_splits) or recv_payload.size(1) != flat.size(1) + 1:
                    raise RuntimeError("fused segmented EP receive payload does not match received routes")
                has_valid_routes = recv_payload.size(0) != 0
            else:
                if recv_tokens.size(0) != sum(recv_splits) or recv_scores.size(0) != sum(recv_splits):
                    raise RuntimeError("segmented EP receive splits do not match received routes")
                has_valid_routes = recv_tokens.size(0) != 0
        else:
            if compact.tokens.size(0) != sum(recv_splits):
                raise RuntimeError("compact EP receive splits do not match received routes")
            has_valid_routes = compact.tokens.size(0) != 0
        expert_major_two_phase_state = None
        if has_valid_routes:
            if phase_controller is not None:
                phase_controller.wait_prefix_before_local_xmx(
                    torch.xpu.current_stream(x.device)
                )
            with _record(
                "moe.sycl_expert_major.fwd_local_experts"
                if expert_major_segmented_backend
                else "moe.sycl_segmented.fwd_local_experts"
                if segmented_backend
                else "moe.sycl_sonic.fwd_ragged_local_experts"
                if ragged_backend
                else "moe.sycl_ep_alltoallv.fwd_exact_local_experts"
            ):
                # Nested leaves keep this custom autograd node explicit while
                # DDP continues to own hooks on the outer expert parameters.
                with torch.enable_grad():
                    expert_tokens = (
                        recv_payload.detach().requires_grad_(True)
                        if expert_major_fused_payload
                        else recv_tokens.detach().requires_grad_(True)
                        if segmented_backend
                        else compact.tokens.detach().requires_grad_(True)
                    )
                    expert_scores = (
                        topk_scores.new_empty((0,), requires_grad=True)
                        if expert_major_fused_payload
                        else recv_scores.detach().requires_grad_(True)
                        if segmented_backend
                        else compact.scores.detach().reshape(-1, 1).requires_grad_(True)
                    )
                    expert_up = up.detach().requires_grad_(True)
                    expert_gate = gate.detach().requires_grad_(True)
                    expert_down = down.detach().requires_grad_(True)
                    if segmented_backend:
                        from aurora_moe._kernels.segmented_sonic import make_peer_expert_segments

                        segments = make_peer_expert_segments(source_expert_counts)
                        if expert_major_segmented_backend:
                            from aurora_moe._kernels.expert_major_segmented_sonic import (
                                expert_major_segmented_local_moe,
                                expert_major_segmented_local_moe_payload,
                            )

                            if expert_major_fused_payload:
                                if _expert_major_two_phase_requested():
                                    from aurora_moe._kernels.expert_major_two_phase import (
                                        prepare_exact_expert_major_two_phase,
                                    )

                                    (
                                        weighted_valid,
                                        expert_major_two_phase_state,
                                    ) = prepare_exact_expert_major_two_phase(
                                        expert_tokens,
                                        segments,
                                        expert_up,
                                        expert_gate,
                                        expert_down,
                                        expert_rows=local_expert_rows,
                                        reorder_backend=os.environ.get(
                                            "AURORA_MOE_EXPERT_MAJOR_REORDER", "parallel"
                                        ),
                                        pointwise_backend=os.environ.get(
                                            "AURORA_MOE_SEGMENTED_POINTWISE", "torch"
                                        ),
                                        already_expert_major=direct_expert_major_ipc,
                                    )
                                else:
                                    weighted_valid = expert_major_segmented_local_moe_payload(
                                        expert_tokens,
                                        segments,
                                        expert_up,
                                        expert_gate,
                                        expert_down,
                                        activation="swiglu",
                                        expert_rows=local_expert_rows,
                                        already_expert_major=direct_expert_major_ipc,
                                    )
                            else:
                                weighted_valid = expert_major_segmented_local_moe(
                                    expert_tokens,
                                    expert_scores,
                                    segments,
                                    expert_up,
                                    expert_gate,
                                    expert_down,
                                    activation="swiglu",
                                    expert_rows=local_expert_rows,
                                )
                        else:
                            from aurora_moe._kernels.segmented_sonic import segmented_local_moe

                            weighted_valid = segmented_local_moe(
                                expert_tokens,
                                expert_scores,
                                segments,
                                expert_up,
                                expert_gate,
                                expert_down,
                                activation="swiglu",
                            )
                    elif ragged_backend:
                        from aurora_moe._kernels.sonic_ragged import ragged_local_moe

                        weighted_valid = ragged_local_moe(
                            expert_tokens,
                            expert_scores,
                            valid_lid.to(torch.long).contiguous(),
                            expert_up,
                            expert_gate,
                            expert_down,
                            activation="swiglu",
                            backend=os.environ.get("AURORA_MOE_SONIC_RAGGED_GEMM", "reference"),
                        )
                    else:
                        from aurora_moe._kernels.padded_bmm_moe import exact_routed_expert_bmm, make_exact_route_plan

                        local_indices = valid_lid.to(torch.long).reshape(-1, 1).contiguous()
                        plan = make_exact_route_plan(local_indices, local_count)
                        weighted_valid = exact_routed_expert_bmm(
                            expert_tokens,
                            expert_scores,
                            expert_up,
                            expert_gate,
                            expert_down,
                            plan,
                            activation="swiglu",
                        )
            local_return = weighted_valid.detach().contiguous()
        else:
            expert_tokens = flat.new_empty(
                (0, flat.shape[-1] + 1)
                if expert_major_fused_payload
                else (0, flat.shape[-1]),
                requires_grad=True,
            )
            expert_scores = topk_scores.new_empty(
                (0,) if segmented_backend else (0, 1), requires_grad=True
            )
            expert_up = up.detach().requires_grad_(True)
            expert_gate = gate.detach().requires_grad_(True)
            expert_down = down.detach().requires_grad_(True)
            weighted_valid = None
            local_return = flat.new_empty((0, flat.shape[-1]))

        # Received rows are source-major.  Sending recv_splits back to those
        # sources restores the destination-major compact order of route_slots.
        # The shared suffix is safe after this event and can use the A3 window
        # because it has no dependency on the local routed output.
        if phase_controller is not None:
            phase_controller.mark_return_ready(torch.xpu.current_stream(x.device))
            if _phase_shared_forward_reorder_requested():
                phase_controller.start_suffix_after_return()
        native_a3_launch = None
        if _threaded_native_a3v_requested(mesh, phase_controller):
            # Capture the A3 producer before the main stream waits for the
            # independent shared suffix.  The host thread can then submit the
            # exact ragged A2Av while that suffix runs, and bridge its event
            # only at the existing route-reduction dependency.
            producer_stream = torch.xpu.current_stream(x.device)
            with _record("moe.comm.sycl_ep_alltoallv_fwd_combine_outputs_thread_start"):
                native_a3_launch = _start_threaded_native_a4v(
                    mesh,
                    local_return,
                    recv_splits,
                    send_splits,
                    producer_stream,
                )
            if not _phase_shared_forward_reorder_requested():
                assert phase_controller is not None
                phase_controller.start_suffix_after_return()
            # Queue the already-required shared completion dependency before
            # joining the host submitter.  This is what overlaps A3 host
            # submission with suffix XMX work without exposing the return
            # buffer before its oneCCL event is bridged.
            assert phase_controller is not None
            phase_controller.finish_forward(producer_stream)
            with _record("moe.comm.sycl_ep_alltoallv_fwd_combine_outputs_thread_join_bridge"):
                returned = native_a3_launch.bridge(producer_stream)
        else:
            if direct_expert_major_ipc:
                assert l0_expert_major_plan is not None
                with _record("moe.comm.l0_ipc_expert_major_fwd_combine_outputs"):
                    returned = _l0_ipc_alltoallv_group_state(
                        mesh
                    ).all_to_all_v_expert_major(
                        local_return,
                        l0_expert_major_plan,
                        inverse=True,
                    )
            else:
                returned = _all_to_allv(
                    local_return,
                    recv_splits,
                    send_splits,
                    mesh,
                    "moe.comm.sycl_ep_alltoallv_fwd_combine_outputs",
                    exact_equal_fastpath=exact_equal_fastpath,
                    l0_remote_receive_offsets=l0_reverse_remote_offsets,
                    l0_global_capacity_rows=l0_reverse_capacity_rows,
                )
            if phase_controller is not None and not _phase_shared_forward_reorder_requested():
                phase_controller.start_suffix_after_return()
        with _record("moe.sycl_ep_alltoallv.fwd_reduce_exact_routes"):
            out = reduce_route_rows(returned.contiguous(), route_slots)

        ctx.mesh = mesh
        ctx.x_shape = tuple(x.shape)
        ctx.has_valid_routes = has_valid_routes
        ctx.small_id_payload = small_id_payload
        ctx.alltoallv = True
        ctx.sonic_ragged = ragged_backend
        ctx.segmented_sonic = segmented_backend
        ctx.segmented_fused_payload = expert_major_fused_payload
        ctx.phase_controller = phase_controller
        ctx.alltoallv_send_splits = send_splits
        ctx.alltoallv_recv_splits = recv_splits
        ctx.alltoallv_exact_equal_fastpath = exact_equal_fastpath
        ctx.alltoallv_l0_forward_remote_offsets = l0_forward_remote_offsets
        ctx.alltoallv_l0_forward_capacity_rows = l0_forward_capacity_rows
        ctx.alltoallv_l0_reverse_remote_offsets = l0_reverse_remote_offsets
        ctx.alltoallv_l0_reverse_capacity_rows = l0_reverse_capacity_rows
        ctx.l0_expert_major_plan = l0_expert_major_plan
        ctx.l0_expert_major_ipc = direct_expert_major_ipc
        ctx.weighted_valid = weighted_valid
        ctx.expert_major_two_phase_state = expert_major_two_phase_state
        ctx.save_for_backward(
            route_slots,
            expert_tokens,
            expert_scores,
            expert_up,
            expert_gate,
            expert_down,
            up,
            gate,
            down,
        )
        return out.reshape_as(x)

    @staticmethod
    def backward(ctx, grad_out):
        if getattr(ctx, "alltoallv", False):
            return _RoutedMOESyclEP._backward_alltoallv(ctx, grad_out)

        direct_layout = getattr(ctx, "direct_layout", False)
        if direct_layout:
            (
                route_slots,
                recv_counts,
                direct_payload,
                expert_up,
                expert_gate,
                expert_down,
                up,
                gate,
                down,
            ) = ctx.saved_tensors
        else:
            (
                route_slots,
                recv_counts,
                valid_positions,
                expert_tokens,
                expert_scores,
                expert_up,
                expert_gate,
                expert_down,
                up,
                gate,
                down,
            ) = ctx.saved_tensors
        mesh = ctx.mesh
        ep_size = ctx.ep_size
        cap = ctx.cap
        model_dim = grad_out.shape[-1]
        rows = ep_size * cap
        small_id_payload = ctx.small_id_payload
        phase_controller = getattr(ctx, "phase_controller", None)
        if phase_controller is not None and not getattr(ctx, "direct_layout_two_phase", False):
            raise RuntimeError(
                "phase-separated shared experts require direct two-phase routed backward"
            )

        from aurora_moe._kernels.ep_route_ops import (
            fuse_route_rows,
            pack_token_rows,
            reduce_route_payload,
        )

        with _record("moe.sycl_ep.bwd_pack_output_grads"):
            grad_flat = grad_out.reshape(-1, model_dim).contiguous()
            grad_returned = pack_token_rows(grad_flat, route_slots, rows)
        # The shared graph needs only the outer gradient.  Launch its first
        # exact backward chunk before A2, then make the local routed payload
        # phase wait for that chunk below.  This uses the A2 communication
        # window without ever overlapping the two XMX-heavy local backwards.
        if (
            phase_controller is not None
            and phase_controller.early_backward_requested()
        ):
            phase_controller.start_backward_early()
        grad_local = _all_to_all(
            grad_returned.reshape(ep_size, cap, model_dim),
            mesh,
            "moe.comm.sycl_ep_bwd_dispatch_output_grads",
        )
        grad_local_flat = grad_local.reshape(rows, model_dim)
        if (
            phase_controller is not None
            and phase_controller.backward_started_early
        ):
            phase_controller.wait_backward_before_local_xmx(
                torch.xpu.current_stream(grad_out.device)
            )
        weight_tail_stream = None
        weight_tail_state = None
        weight_tail_producer = None

        if direct_layout:
            if getattr(ctx, "direct_layout_two_phase", False):
                weight_tail_state = ctx.direct_layout_two_phase_state
                if weight_tail_state is None:
                    raise RuntimeError("direct-layout two-phase backward state is missing")
                with _record("moe.sycl_ep.bwd_direct_expert_payload_phase"):
                    grad_direct_payload = weight_tail_state.payload_gradient(
                        grad_local_flat.contiguous(), payload_columns=model_dim + 1
                    )
                # The tail waits only for payload-gradient production.  A4 is
                # submitted below on the producer stream before the dW BMMs
                # are enqueued, allowing the two independent paths to overlap.
                weight_tail_producer = torch.xpu.current_stream(grad_local_flat.device)
                weight_tail_stream = _direct_layout_two_phase_tail_stream(
                    grad_local_flat.device
                )
                weight_tail_stream.wait_stream(weight_tail_producer)
                grad_payload = grad_direct_payload
            else:
                with _record("moe.sycl_ep.bwd_direct_expert_layout"):
                    # ``weighted_valid`` is directly source-padded in this path,
                    # so the received output gradients need neither compaction
                    # nor an inverse scatter before the nested autograd call.
                    with torch.enable_grad():
                        (
                            grad_direct_payload,
                            grad_expert_up,
                            grad_expert_gate,
                            grad_expert_down,
                        ) = torch.autograd.grad(
                            ctx.weighted_valid,
                            (direct_payload, expert_up, expert_gate, expert_down),
                            grad_local_flat.contiguous(),
                            allow_unused=True,
                        )
                    grad_up = (
                        grad_expert_up
                        if ctx.needs_input_grad[3] and grad_expert_up is not None
                        else (torch.zeros_like(up) if ctx.needs_input_grad[3] else None)
                    )
                    grad_gate = (
                        grad_expert_gate
                        if ctx.needs_input_grad[4] and grad_expert_gate is not None
                        else (torch.zeros_like(gate) if ctx.needs_input_grad[4] else None)
                    )
                    grad_down = (
                        grad_expert_down
                        if ctx.needs_input_grad[5] and grad_expert_down is not None
                        else (torch.zeros_like(down) if ctx.needs_input_grad[5] else None)
                    )
                    if grad_direct_payload is None:
                        grad_direct_payload = torch.zeros_like(direct_payload)
                    # BF16-ID transport appends the exact local ID after this
                    # tensor's final score column.  Its gradient is deliberately
                    # ignored; both transport variants therefore return the same
                    # [token..., score] payload to the reverse all-to-all.
                    grad_payload = grad_direct_payload[:, : model_dim + 1].contiguous()
        else:
            with _record("moe.sycl_ep.bwd_exact_local_experts"):
                grad_tokens = None
                grad_scores = None
                if small_id_payload:
                    from aurora_moe._kernels.ep_local_ops import compact_rows_from_counts
                else:
                    from aurora_moe._kernels.ep_local_ops import gather_rows, scatter_rows
                    grad_recv_tokens_flat = grad_flat.new_zeros((rows, model_dim))
                    grad_recv_scores_flat = expert_scores.new_zeros(rows)
                grad_up = torch.zeros_like(up) if ctx.needs_input_grad[3] else None
                grad_gate = torch.zeros_like(gate) if ctx.needs_input_grad[4] else None
                grad_down = torch.zeros_like(down) if ctx.needs_input_grad[5] else None

                if ctx.has_valid_routes:
                    if small_id_payload:
                        grad_weighted_valid = compact_rows_from_counts(
                            grad_local_flat.contiguous(),
                            recv_counts,
                            cap,
                            known_rows=expert_tokens.size(0),
                        )
                    else:
                        grad_weighted_valid = gather_rows(
                            grad_local_flat.contiguous(), valid_positions
                        )
                    with torch.enable_grad():
                        (
                            grad_tokens,
                            grad_scores,
                            grad_expert_up,
                            grad_expert_gate,
                            grad_expert_down,
                        ) = torch.autograd.grad(
                            ctx.weighted_valid,
                            (expert_tokens, expert_scores, expert_up, expert_gate, expert_down),
                            grad_weighted_valid,
                            allow_unused=True,
                        )
                    if not small_id_payload and grad_tokens is not None:
                        grad_recv_tokens_flat = scatter_rows(
                            grad_tokens.contiguous(), valid_positions, rows
                        )
                    if not small_id_payload and grad_scores is not None:
                        grad_recv_scores_flat.index_copy_(0, valid_positions, grad_scores.reshape(-1))
                    if grad_up is not None and grad_expert_up is not None:
                        grad_up = grad_expert_up
                    if grad_gate is not None and grad_expert_gate is not None:
                        grad_gate = grad_expert_gate
                    if grad_down is not None and grad_expert_down is not None:
                        grad_down = grad_expert_down

            with _record("moe.sycl_ep.bwd_fuse_input_score_grads"):
                if small_id_payload:
                    from aurora_moe._kernels.ep_local_ops import expand_rows_to_counts

                    if grad_tokens is None:
                        grad_tokens = torch.zeros_like(expert_tokens)
                    if grad_scores is None:
                        grad_scores = torch.zeros_like(expert_scores)
                    grad_payload = expand_rows_to_counts(
                        fuse_route_rows(
                            grad_tokens.contiguous(), grad_scores.reshape(-1).contiguous()
                        ),
                        recv_counts,
                        cap,
                    )
                else:
                    grad_payload = fuse_route_rows(grad_recv_tokens_flat, grad_recv_scores_flat)
        native_a4_launch = None
        if phase_controller is not None:
            if weight_tail_producer is None or weight_tail_stream is None:
                raise RuntimeError("phase-separated shared experts require a routed weight tail")
            phase_controller.mark_payload_ready(weight_tail_producer)
        if weight_tail_stream is not None and _threaded_native_a4_requested(mesh):
            assert weight_tail_producer is not None
            dedicated_native_a4 = getattr(ctx, "dedicated_native_a4", False)
            native_a4_state = (
                _native_ccl_dedicated_a4_state(mesh)
                if dedicated_native_a4
                else None
            )
            thread_start_name = (
                "moe.comm.sycl_ep_bwd_combine_input_score_grads_dedicated_thread_start"
                if dedicated_native_a4
                else "moe.comm.sycl_ep_bwd_combine_input_score_grads_thread_start"
            )
            with _record(thread_start_name):
                native_a4_launch = _start_threaded_native_a4(
                    mesh,
                    grad_payload.reshape(ep_size, cap, model_dim + 1),
                    weight_tail_producer,
                    state=native_a4_state,
                )
        else:
            grad_send_payload = _all_to_all(
                grad_payload.reshape(ep_size, cap, model_dim + 1),
                mesh,
                "moe.comm.sycl_ep_bwd_combine_input_score_grads",
            )
        if phase_controller is not None:
            if native_a4_launch is None:
                raise RuntimeError("phase-separated shared experts require threaded native reverse A4")
            # Do not start competing shared XMX work until the oneCCL host
            # submission itself has returned.  This still overlaps it with
            # device-side A4 progress, but avoids the observed multi-step
            # liveness race between nested autograd enqueueing and oneCCL.
            # An explicitly gated diagnostic may remove only this host
            # dependency; the shared stream still waits for payload_ready.
            if _phase_shared_a4_submit_wait_requested():
                native_a4_launch.wait_submitted()
            if phase_controller.backward_after_a4_required:
                phase_controller.start_backward_after_a4()
        if weight_tail_stream is not None:
            assert weight_tail_state is not None
            with _record("moe.sycl_ep.bwd_direct_expert_weight_tail"):
                with torch.xpu.stream(weight_tail_stream):
                    if phase_controller is not None:
                        phase_controller.wait_backward_for_weight_tail(weight_tail_stream)
                    grad_up, grad_gate, grad_down = weight_tail_state.weight_gradients(
                        need_up=ctx.needs_input_grad[3],
                        need_gate=ctx.needs_input_grad[4],
                        need_down=ctx.needs_input_grad[5],
                    )
            weight_tail_state.record_weight_tail_stream(weight_tail_stream)
        if native_a4_launch is not None:
            assert weight_tail_producer is not None
            with _record("moe.comm.sycl_ep_bwd_combine_input_score_grads_thread_join_bridge"):
                grad_send_payload = native_a4_launch.bridge(weight_tail_producer)
        with _record("moe.sycl_ep.bwd_reduce_exact_routes"):
            grad_x, grad_scores_out = reduce_route_payload(
                grad_send_payload.reshape(rows, model_dim + 1), route_slots
            )
        if weight_tail_stream is not None:
            assert weight_tail_producer is not None
            with _record("moe.sycl_ep.bwd_direct_expert_weight_tail_join"):
                weight_tail_producer.wait_stream(weight_tail_stream)

        return (
            grad_x.reshape(ctx.x_shape) if ctx.needs_input_grad[0] else None,
            grad_scores_out if ctx.needs_input_grad[1] else None,
            None,
            grad_up,
            grad_gate,
            grad_down,
            None,
            None,
            None,
        )

    @staticmethod
    def _backward_alltoallv_two_phase(ctx, grad_out, two_phase_state):
        """Run compact A4 while exact expert dW GEMMs occupy a tail stream.

        ``two_phase_state`` has already retained the exact expert-major
        operands from forward.  Its payload phase is the only predecessor of
        reverse A4; dW has no dependency on the returned source-order payload.
        This preserves the compact variable-row algebra while moving the
        blocking direct-oneCCL A4 submission off the critical Python thread.
        """

        (
            route_slots,
            _expert_tokens,
            _expert_scores,
            _expert_up,
            _expert_gate,
            _expert_down,
            _up,
            _gate,
            _down,
        ) = ctx.saved_tensors
        mesh = ctx.mesh
        send_splits = ctx.alltoallv_send_splits
        recv_splits = ctx.alltoallv_recv_splits
        l0_forward_remote_offsets = ctx.alltoallv_l0_forward_remote_offsets
        l0_forward_capacity_rows = ctx.alltoallv_l0_forward_capacity_rows
        l0_reverse_remote_offsets = ctx.alltoallv_l0_reverse_remote_offsets
        l0_reverse_capacity_rows = ctx.alltoallv_l0_reverse_capacity_rows
        direct_expert_major_ipc = getattr(ctx, "l0_expert_major_ipc", False)
        l0_expert_major_plan = getattr(ctx, "l0_expert_major_plan", None)
        if direct_expert_major_ipc and l0_expert_major_plan is None:
            raise RuntimeError("direct expert-major IPC backward state is missing its plan")
        model_dim = grad_out.shape[-1]
        phase_controller = getattr(ctx, "phase_controller", None)

        from aurora_moe._kernels.ep_route_ops import pack_token_rows, reduce_route_payload

        # Shared backward depends only on the outer gradient and its saved
        # shared-expert graph, both of which the joint wrapper has already
        # armed.  In the opt-in early schedule it occupies the exact A2
        # communication window; the reciprocal wait below keeps the routed
        # local payload GEMMs disjoint from its XMX work.
        if (
            phase_controller is not None
            and phase_controller.early_backward_requested()
        ):
            phase_controller.start_backward_early()

        with _record("moe.sycl_ep_alltoallv.bwd_pack_output_grads"):
            grad_flat = grad_out.reshape(-1, model_dim).contiguous()
            grad_returned = pack_token_rows(
                grad_flat, route_slots, route_slots.numel()
            )
        if direct_expert_major_ipc:
            with _record("moe.comm.l0_ipc_expert_major_bwd_dispatch_output_grads"):
                grad_local = _l0_ipc_alltoallv_group_state(
                    mesh
                ).all_to_all_v_expert_major(
                    grad_returned, l0_expert_major_plan, inverse=False
                )
        else:
            grad_local = _all_to_allv(
                grad_returned,
                send_splits,
                recv_splits,
                mesh,
                "moe.comm.sycl_ep_alltoallv_bwd_dispatch_output_grads",
                exact_equal_fastpath=getattr(ctx, "alltoallv_exact_equal_fastpath", False),
                l0_remote_receive_offsets=l0_forward_remote_offsets,
                l0_global_capacity_rows=l0_forward_capacity_rows,
            )
        if (
            phase_controller is not None
            and phase_controller.backward_started_early
        ):
            phase_controller.wait_backward_before_local_xmx(
                torch.xpu.current_stream(grad_out.device)
            )
        with _record("moe.sycl_ep_alltoallv.bwd_exact_local_payload_phase"):
            grad_payload = two_phase_state.payload_gradient(grad_local.contiguous())

        producer_stream = torch.xpu.current_stream(grad_out.device)
        weight_tail_stream = _direct_layout_two_phase_tail_stream(grad_out.device)
        weight_tail_stream.wait_stream(producer_stream)
        if phase_controller is not None:
            phase_controller.mark_payload_ready(producer_stream)
        native_a4_launch = None
        if direct_expert_major_ipc:
            with _record("moe.comm.l0_ipc_expert_major_bwd_combine_input_score_grads"):
                grad_send_payload = _l0_ipc_alltoallv_group_state(
                    mesh
                ).all_to_all_v_expert_major(
                    grad_payload, l0_expert_major_plan, inverse=True
                )
        elif _threaded_native_a4_requested(mesh):
            with _record("moe.comm.sycl_ep_alltoallv_bwd_combine_input_score_grads_thread_start"):
                native_a4_launch = _start_threaded_native_a4v(
                    mesh,
                    grad_payload,
                    recv_splits,
                    send_splits,
                    producer_stream,
                )
        else:
            grad_send_payload = _all_to_allv(
                grad_payload,
                recv_splits,
                send_splits,
                mesh,
                "moe.comm.sycl_ep_alltoallv_bwd_combine_input_score_grads",
                exact_equal_fastpath=getattr(ctx, "alltoallv_exact_equal_fastpath", False),
                l0_remote_receive_offsets=l0_reverse_remote_offsets,
                l0_global_capacity_rows=l0_reverse_capacity_rows,
            )

        if phase_controller is not None:
            if native_a4_launch is not None:
                if phase_controller.backward_after_a4_required:
                    # Start shared backward only once oneCCL's host-side
                    # submission is complete. It then shares A4's device
                    # window, while the routed dW tail waits below to avoid
                    # concurrent XMX GEMMs. A chunked early schedule has
                    # already consumed its first exact shared chunk in A2;
                    # this invocation queues only the remaining live chunks.
                    if _phase_shared_a4_submit_wait_requested():
                        native_a4_launch.wait_submitted()
                    phase_controller.start_backward_after_a4()
            elif _l0_ipc_alltoallv_requested(mesh):
                if phase_controller.backward_after_a4_required:
                    # The exact IPC call has already enqueued its
                    # current-stream completion bridge and post-write device
                    # barrier. Its host return is the corresponding safe
                    # submission point for remaining independent shared
                    # backward chunks.
                    phase_controller.start_backward_after_a4()
            else:
                raise RuntimeError(
                    "phase-shared compact two-phase backward requires native A4 or exact IPC A2Av"
                )

        # Queue the independent dW tail before joining the host submitter.  A
        # oneCCL implementation that blocks while it observes the producer
        # fence can therefore make progress concurrently with XMX dW work.
        with _record("moe.sycl_ep_alltoallv.bwd_exact_expert_weight_tail"):
            with torch.xpu.stream(weight_tail_stream):
                if phase_controller is not None:
                    phase_controller.wait_backward_for_weight_tail(weight_tail_stream)
                grad_up, grad_gate, grad_down = two_phase_state.weight_gradients(
                    need_up=ctx.needs_input_grad[3],
                    need_gate=ctx.needs_input_grad[4],
                    need_down=ctx.needs_input_grad[5],
                )
        two_phase_state.record_weight_tail_stream(weight_tail_stream)

        if native_a4_launch is not None:
            with _record("moe.comm.sycl_ep_alltoallv_bwd_combine_input_score_grads_thread_join_bridge"):
                grad_send_payload = native_a4_launch.bridge(producer_stream)
        with _record("moe.sycl_ep_alltoallv.bwd_reduce_exact_routes"):
            grad_x, grad_scores_out = reduce_route_payload(
                grad_send_payload.contiguous(), route_slots
            )
        # Returned parameter gradients may be consumed by the outer autograd
        # engine on the producer stream, so join the tail only after routing
        # dX has been enqueued.  This retains A4/dW overlap without exposing a
        # cross-stream gradient race.
        with _record("moe.sycl_ep_alltoallv.bwd_exact_expert_weight_tail_join"):
            producer_stream.wait_stream(weight_tail_stream)

        return (
            grad_x.reshape(ctx.x_shape) if ctx.needs_input_grad[0] else None,
            grad_scores_out if ctx.needs_input_grad[1] else None,
            None,
            grad_up,
            grad_gate,
            grad_down,
            None,
            None,
            None,
        )

    @staticmethod
    def _backward_alltoallv(ctx, grad_out):
        """Differentiate the exact non-padded EP transport in compact row order."""

        two_phase_state = getattr(ctx, "expert_major_two_phase_state", None)
        if two_phase_state is not None:
            return _RoutedMOESyclEP._backward_alltoallv_two_phase(
                ctx, grad_out, two_phase_state
            )

        (
            route_slots,
            expert_tokens,
            expert_scores,
            expert_up,
            expert_gate,
            expert_down,
            up,
            gate,
            down,
        ) = ctx.saved_tensors
        mesh = ctx.mesh
        send_splits = ctx.alltoallv_send_splits
        recv_splits = ctx.alltoallv_recv_splits
        l0_forward_remote_offsets = ctx.alltoallv_l0_forward_remote_offsets
        l0_forward_capacity_rows = ctx.alltoallv_l0_forward_capacity_rows
        l0_reverse_remote_offsets = ctx.alltoallv_l0_reverse_remote_offsets
        l0_reverse_capacity_rows = ctx.alltoallv_l0_reverse_capacity_rows
        direct_expert_major_ipc = getattr(ctx, "l0_expert_major_ipc", False)
        l0_expert_major_plan = getattr(ctx, "l0_expert_major_plan", None)
        if direct_expert_major_ipc and l0_expert_major_plan is None:
            raise RuntimeError("direct expert-major IPC backward state is missing its plan")
        model_dim = grad_out.shape[-1]
        phase_controller = getattr(ctx, "phase_controller", None)

        from aurora_moe._kernels.ep_route_ops import (
            fuse_route_rows,
            pack_token_rows,
            reduce_route_payload,
        )

        # route_slots is an exact compact permutation, so no padding is
        # introduced even though the shared route pack primitive is reused.
        with _record("moe.sycl_ep_alltoallv.bwd_pack_output_grads"):
            grad_flat = grad_out.reshape(-1, model_dim).contiguous()
            grad_returned = pack_token_rows(
                grad_flat, route_slots, route_slots.numel()
            )
        # The exact generic compact path keeps router-score gradients, unlike
        # the expert-major two-phase experiment.  It nevertheless has the
        # same A2 communication interval: run the configured first shared
        # backward chunk there, then gate the routed nested autograd below so
        # the two local XMX computations remain disjoint.
        if (
            phase_controller is not None
            and phase_controller.early_backward_requested()
        ):
            phase_controller.start_backward_early()
        if direct_expert_major_ipc:
            with _record("moe.comm.l0_ipc_expert_major_bwd_dispatch_output_grads"):
                grad_local = _l0_ipc_alltoallv_group_state(
                    mesh
                ).all_to_all_v_expert_major(
                    grad_returned, l0_expert_major_plan, inverse=False
                )
        else:
            grad_local = _all_to_allv(
                grad_returned,
                send_splits,
                recv_splits,
                mesh,
                "moe.comm.sycl_ep_alltoallv_bwd_dispatch_output_grads",
                exact_equal_fastpath=getattr(ctx, "alltoallv_exact_equal_fastpath", False),
                l0_remote_receive_offsets=l0_forward_remote_offsets,
                l0_global_capacity_rows=l0_forward_capacity_rows,
            )
        if (
            phase_controller is not None
            and phase_controller.backward_started_early
        ):
            phase_controller.wait_backward_before_local_xmx(
                torch.xpu.current_stream(grad_out.device)
            )

        with _record("moe.sycl_ep_alltoallv.bwd_exact_local_experts"):
            grad_tokens = None
            grad_scores = None
            grad_up = torch.zeros_like(up) if ctx.needs_input_grad[3] else None
            grad_gate = torch.zeros_like(gate) if ctx.needs_input_grad[4] else None
            grad_down = torch.zeros_like(down) if ctx.needs_input_grad[5] else None

            if ctx.has_valid_routes:
                with torch.enable_grad():
                    (
                        grad_tokens,
                        grad_scores,
                        grad_expert_up,
                        grad_expert_gate,
                        grad_expert_down,
                    ) = torch.autograd.grad(
                        ctx.weighted_valid,
                        (expert_tokens, expert_scores, expert_up, expert_gate, expert_down),
                        grad_local.contiguous(),
                        allow_unused=True,
                    )
                if grad_up is not None and grad_expert_up is not None:
                    grad_up = grad_expert_up
                if grad_gate is not None and grad_expert_gate is not None:
                    grad_gate = grad_expert_gate
                if grad_down is not None and grad_expert_down is not None:
                    grad_down = grad_expert_down

        with _record("moe.sycl_ep_alltoallv.bwd_fuse_input_score_grads"):
            if getattr(ctx, "segmented_fused_payload", False):
                # The expert-major custom node has already performed the exact
                # inverse source/expert permutation and fused dX/dscore into
                # the compact [D + 1] transport payload.  Re-fusing here would
                # both be incorrect (expert_tokens is itself D+1) and add a
                # full activation-memory pass.
                grad_payload = (
                    grad_tokens.contiguous()
                    if grad_tokens is not None
                    else torch.zeros_like(expert_tokens)
                )
            else:
                if grad_tokens is None:
                    grad_tokens = torch.zeros_like(expert_tokens)
                if grad_scores is None:
                    grad_scores = torch.zeros_like(expert_scores)
                grad_payload = fuse_route_rows(
                    grad_tokens.contiguous(), grad_scores.reshape(-1).contiguous()
                )

        # The local rows above are source-major.  Reverse the forward split
        # directions to restore destination-major route_slots on this rank.
        # At this point local dX, dscore, and routed dW are all complete, so
        # shared backward can safely use A4's communication window without
        # sharing XMX with the local expert gradient calculation.
        if phase_controller is not None:
            phase_controller.mark_payload_ready(torch.xpu.current_stream(grad_out.device))
        if direct_expert_major_ipc:
            with _record("moe.comm.l0_ipc_expert_major_bwd_combine_input_score_grads"):
                grad_send_payload = _l0_ipc_alltoallv_group_state(
                    mesh
                ).all_to_all_v_expert_major(
                    grad_payload, l0_expert_major_plan, inverse=True
                )
        else:
            grad_send_payload = _all_to_allv(
                grad_payload,
                recv_splits,
                send_splits,
                mesh,
                "moe.comm.sycl_ep_alltoallv_bwd_combine_input_score_grads",
                exact_equal_fastpath=getattr(ctx, "alltoallv_exact_equal_fastpath", False),
                l0_remote_receive_offsets=l0_reverse_remote_offsets,
                l0_global_capacity_rows=l0_reverse_capacity_rows,
            )
        if (
            phase_controller is not None
            and phase_controller.backward_after_a4_required
        ):
            phase_controller.start_backward_after_a4()
        with _record("moe.sycl_ep_alltoallv.bwd_reduce_exact_routes"):
            grad_x, grad_scores_out = reduce_route_payload(
                grad_send_payload.contiguous(), route_slots
            )

        return (
            grad_x.reshape(ctx.x_shape) if ctx.needs_input_grad[0] else None,
            grad_scores_out if ctx.needs_input_grad[1] else None,
            None,
            grad_up,
            grad_gate,
            grad_down,
            None,
            None,
            None,
        )


class _RoutedMOESyclSonic(torch.autograd.Function):
    """Exact compact-EP MoE with ragged Sonic-style local experts."""

    @staticmethod
    def forward(
        ctx,
        x,
        topk_scores,
        topk_indices,
        up,
        gate,
        down,
        mesh,
        local_count,
        phase_controller=None,
    ):
        if x.device.type != "xpu" or x.dtype != torch.bfloat16:
            raise ValueError("sycl_sonic expert backend requires BF16 XPU tensors")
        if topk_scores.dtype != torch.bfloat16:
            raise ValueError("sycl_sonic expert backend requires BF16 router scores")
        if os.environ.get("AURORA_MOE_ALLTOALLV") != "1":
            raise RuntimeError("sycl_sonic requires AURORA_MOE_ALLTOALLV=1 for exact compact EP transport")
        return _RoutedMOESyclEP._forward_alltoallv(
            ctx,
            x,
            topk_scores,
            topk_indices,
            up,
            gate,
            down,
            mesh,
            local_count,
            ragged_backend=True,
            phase_controller=phase_controller,
        )

    @staticmethod
    def backward(ctx, grad_out):
        return _RoutedMOESyclEP._backward_alltoallv(ctx, grad_out)


def _routed_moe(
    x,
    topk_scores,
    topk_indices,
    local_expert_ids,
    mesh,
    up,
    gate,
    down,
    expert_backend="loop",
    phase_controller=None,
):
    local_count = len(local_expert_ids)
    if phase_controller is not None and expert_backend not in {"sycl_ep", "sycl_sonic"}:
        raise ValueError(
            "phase-separated shared experts require expert_backend=sycl_ep or sycl_sonic"
        )
    if expert_backend == "loop":
        return _RoutedMOE.apply(x, topk_scores, topk_indices, up, gate, down, mesh, local_count)
    if expert_backend == "sycl_bmm":
        return _RoutedMOESyclBMM.apply(x, topk_scores, topk_indices, up, gate, down, mesh, local_count)
    if expert_backend == "sycl_ep":
        return _RoutedMOESyclEP.apply(
            x,
            topk_scores,
            topk_indices,
            up,
            gate,
            down,
            mesh,
            local_count,
            phase_controller,
        )
    if expert_backend == "sycl_sonic":
        return _RoutedMOESyclSonic.apply(
            x,
            topk_scores,
            topk_indices,
            up,
            gate,
            down,
            mesh,
            local_count,
            phase_controller,
        )
    raise ValueError("unsupported expert backend %s" % expert_backend)


class Router(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.top_k = config.top_k
        self.router_proj = nn.Linear(config.model_dim, config.num_experts, bias=False)
        weight = _randn(
            (config.num_experts, config.model_dim),
            config.seed + 1000 * layer_id,
            1.0 / math.sqrt(config.model_dim),
            self.router_proj.weight.device,
        )
        with torch.no_grad():
            self.router_proj.weight.copy_(weight)

    def forward(self, x):
        logits = self.router_proj(x.reshape(-1, x.shape[-1]))
        topk_logits, topk_indices = torch.topk(logits, self.top_k, dim=-1)
        return torch.sigmoid(topk_logits), topk_indices


class LocalExperts(nn.Module):
    def __init__(self, config, local_expert_ids, layer_id):
        super().__init__()
        count = len(local_expert_ids)
        self.experts_up = nn.Parameter(torch.empty(count, config.model_dim, config.expert_hidden_dim))
        self.experts_gate = nn.Parameter(torch.empty(count, config.model_dim, config.expert_hidden_dim))
        self.experts_down = nn.Parameter(torch.empty(count, config.expert_hidden_dim, config.model_dim))
        self.expert_backend = getattr(config, "expert_backend", "loop")
        _init_experts(self.experts_up, self.experts_gate, self.experts_down, local_expert_ids, config, layer_id)

    def forward(
        self,
        x=None,
        scores=None,
        indices=None,
        local_expert_ids=None,
        mesh=None,
        *,
        ddp_anchor_only=False,
    ):
        if ddp_anchor_only:
            return ddp_parameter_anchor(self.experts_up, self.experts_gate, self.experts_down)
        if any(value is None for value in (x, scores, indices, local_expert_ids, mesh)):
            raise ValueError("LocalExperts forward requires x, scores, indices, local_expert_ids, and mesh")
        return _routed_moe(
            x,
            scores,
            indices,
            local_expert_ids,
            mesh,
            self.experts_up,
            self.experts_gate,
            self.experts_down,
            self.expert_backend,
        )


class SharedExperts(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.shared_count = config.shared_experts
        self.experts_up = nn.Parameter(torch.empty(self.shared_count, config.model_dim, config.expert_hidden_dim))
        self.experts_gate = nn.Parameter(torch.empty(self.shared_count, config.model_dim, config.expert_hidden_dim))
        self.experts_down = nn.Parameter(torch.empty(self.shared_count, config.expert_hidden_dim, config.model_dim))
        ids = range(config.num_experts, config.num_experts + self.shared_count)
        _init_experts(self.experts_up, self.experts_gate, self.experts_down, ids, config, layer_id)

    def forward(self, x=None, *, ddp_anchor_only=False):
        if ddp_anchor_only:
            return ddp_parameter_anchor(self.experts_up, self.experts_gate, self.experts_down)
        if x is None:
            raise ValueError("SharedExperts forward requires x")
        flat = x.reshape(-1, x.shape[-1])
        out = flat.new_zeros(flat.shape)
        for expert_id in range(self.shared_count):
            out = out + _swiglu(
                flat,
                self.experts_up[expert_id],
                self.experts_gate[expert_id],
                self.experts_down[expert_id],
            )
        return out.reshape_as(x)


class MOELayer(nn.Module):
    def __init__(self, config, mesh, layer_id=0):
        super().__init__()
        if _native_ccl_requested():
            _require_native_ccl_environment()
        self._native_ccl_reducer = _native_ccl_reducer_requested()
        self.mesh = mesh
        ep_size = mesh.group_size["ep_dispatch"]
        if config.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if config.shared_experts < 0:
            raise ValueError("shared_experts must be nonnegative")
        if config.num_experts % ep_size != 0:
            raise ValueError("num_experts=%d must be divisible by ep_size=%d" % (config.num_experts, ep_size))
        if not 1 <= config.top_k <= config.num_experts:
            raise ValueError("top_k=%d must be in [1, num_experts]" % config.top_k)

        self.local_count = config.num_experts // ep_size
        start = mesh.group_rank["ep_dispatch"] * self.local_count
        self.local_expert_ids = list(range(start, start + self.local_count))
        self.router = Router(config, layer_id)
        self.experts = LocalExperts(config, self.local_expert_ids, layer_id)
        self.shared_experts = SharedExperts(config, layer_id)
        self.to(device=mesh.device, dtype=_module_dtype(config))

        # The shared-expert branch is independent of EP dispatch and can use
        # a second XPU stream while the routed branch is in communication.
        # Keep this opt-in until it has been validated across Aurora's xCCL
        # stream semantics; it is shape-independent and does not alter the
        # layer's mathematical result.
        self._joint_routed_shared_autograd = (
            mesh.device.type == "xpu"
            and config.expert_backend in {"sycl_ep", "sycl_sonic"}
            # An empty shared branch has no graph edge to its empty parameter
            # tensors.  Keep it on the ordinary routed path so it retains the
            # loop implementation's ``grad is None`` behavior and does not
            # pay for phase-controller stream events.
            and config.shared_experts > 0
            and os.environ.get("AURORA_MOE_JOINT_ROUTED_SHARED_AUTOGRAD") == "1"
        )
        self._phase_shared_experts = (
            config.expert_backend in {"sycl_ep", "sycl_sonic"}
            and config.shared_experts > 0
            and _phase_shared_experts_requested(mesh)
        )
        if self._phase_shared_experts and not self._joint_routed_shared_autograd:
            raise RuntimeError(
                "AURORA_MOE_PHASE_SHARED_EXPERTS=1 requires "
                "AURORA_MOE_JOINT_ROUTED_SHARED_AUTOGRAD=1"
            )
        self._overlap_shared_experts = (
            mesh.device.type == "xpu"
            and os.environ.get("AURORA_MOE_OVERLAP_SHARED", "0") == "1"
            and not self._joint_routed_shared_autograd
        )
        self._shared_expert_stream = (
            torch.xpu.Stream()
            if self._overlap_shared_experts or self._joint_routed_shared_autograd
            else None
        )

        self.router = _wrap_ddp(self.router, mesh.groups["dense_dp"], mesh.group_size["dense_dp"], False)
        # The exact SYCL-EP autograd function always returns a gradient tensor
        # (zero-filled when no route reaches a local expert), so its expert
        # parameters are not dynamically unused.  Leave the conservative DDP
        # behavior as the default and expose the cheaper path only for a
        # correctness-gated Aurora experiment.
        expert_find_unused = not (
            config.expert_backend in {"sycl_ep", "sycl_sonic"}
            and os.environ.get("AURORA_MOE_SYCL_EP_NO_UNUSED") == "1"
        )
        self.experts = _wrap_ddp(
            self.experts,
            mesh.groups["sparse_dp"],
            mesh.group_size["sparse_dp"],
            expert_find_unused,
        )
        self.shared_experts = _wrap_ddp(self.shared_experts, mesh.groups["dense_dp"], mesh.group_size["dense_dp"], False)
        # Construct once after the optional DDP wrapping.  In native-reducer
        # mode wrapping is intentionally disabled, while in normal mode this
        # stays ``None`` and preserves the stock DDP behavior.
        self._native_grad_reducer = (
            _NativeMoEGradReducer(self) if self._native_ccl_reducer else None
        )
        self._native_reducer_two_streams = _native_reducer_two_streams_requested()
        self._native_reducer_pack_sync = _native_reducer_pack_sync_requested()
        self._native_reducer_serialize_collectives = (
            _native_reducer_serialize_collectives_requested()
        )
        if self._native_reducer_serialize_collectives and not self._native_ccl_reducer:
            raise RuntimeError(
                "AURORA_MOE_NATIVE_REDUCER_SERIALIZE_COLLECTIVES=1 requires "
                "AURORA_MOE_NATIVE_CCL_REDUCER=1"
            )
        if (
            self._native_reducer_serialize_collectives
            and not self._native_reducer_two_streams
        ):
            raise RuntimeError(
                "AURORA_MOE_NATIVE_REDUCER_SERIALIZE_COLLECTIVES=1 requires "
                "AURORA_MOE_NATIVE_REDUCER_TWO_STREAMS=1"
            )
        if self._native_reducer_pack_sync and not self._native_reducer_two_streams:
            raise RuntimeError(
                "AURORA_MOE_NATIVE_REDUCER_PACK_SYNC=1 requires "
                "AURORA_MOE_NATIVE_REDUCER_TWO_STREAMS=1"
            )
        if self._native_reducer_two_streams:
            self._native_grad_reducer.enable_two_streams()
            if self._native_reducer_pack_sync:
                self._native_grad_reducer.enable_synchronized_pack()
            if self._native_reducer_serialize_collectives:
                self._native_grad_reducer.enable_serialized_collectives()

    @property
    def router_module(self):
        return _unwrap_ddp(self.router)

    @property
    def experts_module(self):
        return _unwrap_ddp(self.experts)

    @property
    def shared_module(self):
        return _unwrap_ddp(self.shared_experts)

    @property
    def router_weight(self):
        return self.router_module.router_proj.weight

    @property
    def experts_up(self):
        return self.experts_module.experts_up

    @property
    def experts_gate(self):
        return self.experts_module.experts_gate

    @property
    def experts_down(self):
        return self.experts_module.experts_down

    @property
    def shared_up(self):
        return self.shared_module.experts_up

    @property
    def shared_gate(self):
        return self.shared_module.experts_gate

    @property
    def shared_down(self):
        return self.shared_module.experts_down

    def scale_ddp_grads(self):
        dense = self.mesh.group_size["dense_dp"]
        sparse = self.mesh.group_size["sparse_dp"]
        if self._native_ccl_reducer:
            # Keep the historic public method name so callers and benchmark
            # harnesses do not need a special post-backward path.  Unlike
            # DDP, the direct oneCCL operations return SUM rather than AVG,
            # which exactly matches the result after the legacy scaling.
            if self._native_grad_reducer is None:
                raise RuntimeError("native CCL reducer was not initialized")
            self._native_grad_reducer.reduce_sum_()
            return
        if dense > 1:
            _scale_grad(self.router_weight, dense)
            _scale_grad(self.shared_up, dense)
            _scale_grad(self.shared_gate, dense)
            _scale_grad(self.shared_down, dense)
        if sparse > 1:
            _scale_grad(self.experts_up, sparse)
            _scale_grad(self.experts_gate, sparse)
            _scale_grad(self.experts_down, sparse)

    def forward(self, x):
        with _record("moe.router"):
            scores, indices = self.router(x)
        if self._joint_routed_shared_autograd:
            assert self._shared_expert_stream is not None
            phase_controller = (
                PhaseSharedExpertController(
                    x,
                    self.shared_up,
                    self.shared_gate,
                    self.shared_down,
                    (
                        x.requires_grad,
                        self.shared_up.requires_grad,
                        self.shared_gate.requires_grad,
                        self.shared_down.requires_grad,
                    ),
                    self._shared_expert_stream,
                )
                if self._phase_shared_experts
                else None
            )
            sparse_anchor = (
                self.experts(ddp_anchor_only=True) if isinstance(self.experts, DDP) else None
            )
            dense_anchor = (
                self.shared_experts(ddp_anchor_only=True)
                if isinstance(self.shared_experts, DDP)
                else None
            )

            def routed_forward(inner_x, inner_scores, inner_up, inner_gate, inner_down):
                return _routed_moe(
                    inner_x,
                    inner_scores,
                    indices,
                    self.local_expert_ids,
                    self.mesh,
                    inner_up,
                    inner_gate,
                    inner_down,
                    self.experts_module.expert_backend,
                    phase_controller,
                )

            with _record("moe.joint_routed_shared"):
                output = joint_routed_shared_moe(
                    x,
                    scores,
                    indices,
                    self.experts_up,
                    self.experts_gate,
                    self.experts_down,
                    self.shared_up,
                    self.shared_gate,
                    self.shared_down,
                    routed_forward=routed_forward,
                    shared_stream=self._shared_expert_stream,
                    phase_controller=phase_controller,
                )
            if sparse_anchor is not None:
                output = output + sparse_anchor
            if dense_anchor is not None:
                output = output + dense_anchor
            return output
        if self._shared_expert_stream is not None:
            # The two branches only read ``x`` and have disjoint parameters.
            # Explicitly hand off the router-produced activation to the shared
            # stream, then hand the result back before addition.  Merely using
            # a tensor on another XPU stream does not establish either edge.
            current_stream = torch.xpu.current_stream(x.device)
            with torch.xpu.stream(self._shared_expert_stream):
                self._shared_expert_stream.wait_stream(current_stream)
                with _record("moe.shared_experts"):
                    shared = self.shared_experts(x)
            with _record("moe.routed_experts"):
                routed = self.experts(x, scores, indices, self.local_expert_ids, self.mesh)
            current_stream.wait_stream(self._shared_expert_stream)
        else:
            with _record("moe.routed_experts"):
                routed = self.experts(x, scores, indices, self.local_expert_ids, self.mesh)
            with _record("moe.shared_experts"):
                shared = self.shared_experts(x)
        with _record("moe.combine_routed_shared"):
            return routed + shared
