import os

import pytest
import torch

from aurora_moe import AuroraMoE, configure_native_runtime, create_dp_ep_groups
from aurora_moe.build import write_environment
from aurora_moe.distributed import ParallelMesh
from aurora_moe.torchtitan_experts import _expert_rows_tuple


_RUNTIME_ENV = (
    "ZE_FLAT_DEVICE_HIERARCHY",
    "CCL_OP_SYNC",
    "CCL_WORKER_COUNT",
    "AURORA_MOE_NATIVE_CCL",
    "AURORA_MOE_NATIVE_CCL_REDUCER",
    "AURORA_MOE_NATIVE_CCL_PRIVATE_STREAM",
    "AURORA_MOE_NATIVE_CCL_STREAM_FENCE_REAP",
    "AURORA_MOE_NATIVE_CCL_ALLTOALLV",
    "AURORA_MOE_ALLTOALLV",
    "AURORA_MOE_SEGMENTED_SONIC",
    "AURORA_MOE_SEGMENTED_EXPERT_MAJOR",
    "AURORA_MOE_SEGMENTED_FUSED_SCORE_PAYLOAD",
    "AURORA_MOE_EXPERT_MAJOR_GEMM",
    "AURORA_MOE_EXPERT_MAJOR_DW",
    "AURORA_MOE_EXPERT_MAJOR_REORDER",
    "AURORA_MOE_EXPERT_MAJOR_DOWN_BACKWARD",
    "AURORA_MOE_SEGMENTED_POINTWISE",
    "AURORA_MOE_ONEMKL_FUSE_UP_GATE_DX",
    "AURORA_MOE_EXPERT_MAJOR_PACKED_UP_GATE",
    "AURORA_MOE_NATIVE_REDUCER_TWO_STREAMS",
    "AURORA_MOE_IGNORE_ROUTER_GRAD",
    "AURORA_MOE_EXPERT_MAJOR_TWO_PHASE",
    "AURORA_MOE_JOINT_ROUTED_SHARED_AUTOGRAD",
    "AURORA_MOE_PHASE_SHARED_EXPERTS",
    "AURORA_MOE_PHASE_SHARED_FORWARD_REORDER",
    "AURORA_MOE_PHASE_SHARED_A4_SUBMIT_WAIT",
)


@pytest.fixture(autouse=True)
def clean_runtime_environment(monkeypatch):
    for name in _RUNTIME_ENV:
        monkeypatch.delenv(name, raising=False)


def test_single_rank_reference_layer_has_all_gradients():
    torch.manual_seed(7)
    layer = AuroraMoE(
        model_dim=8,
        expert_hidden_dim=12,
        num_experts=4,
        top_k=2,
        shared_experts=1,
        device="cpu",
        dtype=torch.float32,
        backend="loop",
    )
    x = torch.randn(3, 5, 8, requires_grad=True)
    y = layer(x)
    assert y.shape == x.shape
    y.square().mean().backward()
    assert x.grad is not None
    assert layer.router_weight.grad is not None
    assert layer.experts_up.grad is not None
    assert layer.experts_gate.grad is not None
    assert layer.experts_down.grad is not None


def test_training_runtime_never_enables_router_free_mode(monkeypatch):
    monkeypatch.setenv("AURORA_MOE_IGNORE_ROUTER_GRAD", "1")
    configure_native_runtime(router_grad=True)
    assert "AURORA_MOE_IGNORE_ROUTER_GRAD" not in os.environ
    assert os.environ["AURORA_MOE_EXPERT_MAJOR_GEMM"] == "onemkl"
    with pytest.raises(ValueError, match="router_grad=False"):
        configure_native_runtime(router_grad=True, router_free_two_phase=True)


def test_single_rank_groups_and_prebuild_environment_file(tmp_path):
    groups = create_dp_ep_groups(dp_size=1, ep_size=1)
    assert groups.ep_dispatch is None
    mesh = ParallelMesh(groups, torch.device("cpu"))
    assert mesh.ranks == {
        "ep_dispatch": [0],
        "dense_dp": [0],
        "sparse_dp": [0],
    }
    output = write_environment(
        tmp_path / "build",
        {"AURORA_MOE_TEST_OPS_SO": tmp_path / "build" / "test.so"},
        tmp_path / "aurora_moe.env",
    )
    assert output.read_text().splitlines() == [
        f"export AURORA_MOE_SYCL_BUILD_DIR={tmp_path / 'build'}",
        f"export AURORA_MOE_TEST_OPS_SO={tmp_path / 'build' / 'test.so'}",
    ]


def test_torchtitan_expert_rows_accept_integral_histogram_counts():
    counts = torch.tensor([4.0, 0.0, 3.0, 1.0])
    assert _expert_rows_tuple(counts, num_experts=4, total_rows=8) == (4, 0, 3, 1)


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        ([1, -1], "non-negative integers"),
        ([1.5, 2], "non-negative integers"),
        ([1, 2], "sums to 3"),
        ([1, 2, 3], "entries, expected 2"),
    ],
)
def test_torchtitan_expert_rows_reject_invalid_counts(counts, message):
    with pytest.raises(ValueError, match=message):
        _expert_rows_tuple(counts, num_experts=2, total_rows=4)
