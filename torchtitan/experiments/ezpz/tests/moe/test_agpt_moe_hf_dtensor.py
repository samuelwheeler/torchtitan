# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard

from torchtitan.experiments.ezpz.moe.agpt_hf_state_dict_adapter import (
    AGPTMoEStateDictAdapter,
)


def _round_trip(rank, init_file, shard_dim):
    from torchtitan.experiments.ezpz.tests.moe.conftest import build_tiny_moe_config

    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2,
    )
    try:
        torch.manual_seed(23)
        config = build_tiny_moe_config()
        # Five rows per rank cut through gate/up pairs in the physical layout.
        config.layers[0].moe.shared_experts.w13.out_features = 5
        config.layers[0].moe.shared_experts.w2.in_features = 5
        source = config.build().state_dict()
        key = "layers.0.moe.shared_experts.w13.weight"
        mesh = init_device_mesh("cpu", (2,))
        original = source[key]
        source[key] = distribute_tensor(original, mesh, [Shard(shard_dim)])
        adapter = AGPTMoEStateDictAdapter(config, None)
        restored = adapter.from_hf(adapter.to_hf(source))[key]
        assert restored.placements == source[key].placements
        assert restored.shape == original.shape
        torch.testing.assert_close(restored.to_local(), source[key].to_local(), rtol=0, atol=0)
        torch.testing.assert_close(restored.full_tensor(), original, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("shard_dim", [0, 1])
def test_shared_expert_dtensor_round_trip(tmp_path, shard_dim):
    torch.multiprocessing.spawn(
        _round_trip, args=(str(tmp_path / "gloo-init"), shard_dim), nprocs=2,
    )
