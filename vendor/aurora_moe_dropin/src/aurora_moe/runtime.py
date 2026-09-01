"""Explicit runtime presets for Aurora's validated native-oneCCL path."""

from __future__ import annotations

import os

import torch.distributed as dist


def configure_native_runtime(
    *,
    router_grad: bool = True,
    phase_shared: bool = False,
    router_free_two_phase: bool = False,
    zero_wait_a4_submit: bool = False,
    reorder: str = "row_parallel",
) -> None:
    """Configure the source-major, native-oneCCL expert-parallel implementation.

    Call this before ``torch.distributed.init_process_group``.  The default
    retains router-score gradients.  The faster two-phase profile is available
    only when ``router_grad=False`` because it deliberately returns zero router
    score gradients.
    """

    if dist.is_available() and dist.is_initialized():
        raise RuntimeError("configure_native_runtime must run before process-group initialization")
    if reorder not in {"parallel", "row_parallel"}:
        raise ValueError("reorder must be 'parallel' or 'row_parallel'")
    if router_free_two_phase and router_grad:
        raise ValueError("router_free_two_phase requires router_grad=False")
    if zero_wait_a4_submit and router_grad:
        raise ValueError("zero_wait_a4_submit is only validated for router_grad=False")

    os.environ.update(
        {
            "ZE_FLAT_DEVICE_HIERARCHY": "FLAT",
            "CCL_OP_SYNC": "0",
            "CCL_WORKER_COUNT": "1",
            "AURORA_MOE_NATIVE_CCL": "1",
            "AURORA_MOE_NATIVE_CCL_REDUCER": "1",
            "AURORA_MOE_NATIVE_CCL_PRIVATE_STREAM": "1",
            "AURORA_MOE_NATIVE_CCL_STREAM_FENCE_REAP": "1",
            "AURORA_MOE_NATIVE_CCL_ALLTOALLV": "1",
            "AURORA_MOE_ALLTOALLV": "1",
            "AURORA_MOE_SEGMENTED_SONIC": "1",
            "AURORA_MOE_SEGMENTED_EXPERT_MAJOR": "1",
            "AURORA_MOE_SEGMENTED_FUSED_SCORE_PAYLOAD": "1",
            "AURORA_MOE_EXPERT_MAJOR_GEMM": "onemkl",
            "AURORA_MOE_EXPERT_MAJOR_DW": "onemkl",
            "AURORA_MOE_EXPERT_MAJOR_REORDER": reorder,
            "AURORA_MOE_EXPERT_MAJOR_DOWN_BACKWARD": "reordered",
            "AURORA_MOE_SEGMENTED_POINTWISE": "sycl",
            "AURORA_MOE_ONEMKL_FUSE_UP_GATE_DX": "1",
            "AURORA_MOE_EXPERT_MAJOR_PACKED_UP_GATE": "0",
            "AURORA_MOE_NATIVE_REDUCER_TWO_STREAMS": "0",
        }
    )

    if router_grad:
        os.environ.pop("AURORA_MOE_IGNORE_ROUTER_GRAD", None)
        os.environ.pop("AURORA_MOE_EXPERT_MAJOR_TWO_PHASE", None)
    else:
        os.environ["AURORA_MOE_IGNORE_ROUTER_GRAD"] = "1"
        if router_free_two_phase:
            os.environ["AURORA_MOE_EXPERT_MAJOR_TWO_PHASE"] = "1"
        else:
            os.environ.pop("AURORA_MOE_EXPERT_MAJOR_TWO_PHASE", None)

    if phase_shared:
        os.environ.update(
            {
                "AURORA_MOE_JOINT_ROUTED_SHARED_AUTOGRAD": "1",
                "AURORA_MOE_PHASE_SHARED_EXPERTS": "1",
                "AURORA_MOE_PHASE_SHARED_FORWARD_REORDER": "1",
                "AURORA_MOE_PHASE_SHARED_A4_SUBMIT_WAIT": "0"
                if zero_wait_a4_submit
                else "1",
            }
        )
    else:
        for name in (
            "AURORA_MOE_JOINT_ROUTED_SHARED_AUTOGRAD",
            "AURORA_MOE_PHASE_SHARED_EXPERTS",
            "AURORA_MOE_PHASE_SHARED_FORWARD_REORDER",
            "AURORA_MOE_PHASE_SHARED_A4_SUBMIT_WAIT",
        ):
            os.environ.pop(name, None)
