# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

SCRIPT = (
    Path(__file__).parents[3]
    / "torchtitan/experiments/ezpz/rl/scripts/grpo/agpt2b_multihost_torchstore_validate.pbs"
)
ENTRYPOINT = SCRIPT.with_name("multihost_train_upstream.py")


def test_multihost_launcher_allowlists_network_transport_controls():
    text = SCRIPT.read_text()

    assert 'TORCHSTORE_TRANSPORT="${TORCHSTORE_TRANSPORT:-auto}"' in text
    assert "auto_no_shm)" in text
    assert "TORCHSTORE_SHARED_MEMORY_ENABLED=0" in text
    assert "gloo|xccl|monarch_rdma)" in text
    assert "unsupported TORCHSTORE_TRANSPORT" in text


def test_multihost_entrypoint_allows_slow_actor_imports():
    text = ENTRYPOINT.read_text()

    assert 'mesh_attach_config_timeout="300s"' in text
