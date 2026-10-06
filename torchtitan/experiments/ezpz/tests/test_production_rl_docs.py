# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Static guards for the current production RL documentation hierarchy."""

from pathlib import Path


DOCS = Path(__file__).parents[1] / "docs" / "production" / "rl"
CURRENT = "monarch.md"


def test_rl_index_points_to_current_production_runbook():
    text = (DOCS / "README.md").read_text()
    current = (DOCS / CURRENT).read_text()

    assert "**Current production instructions:**" in text
    assert CURRENT in text
    assert "Production / multi-node GRPO today: TRL" not in text
    assert "## Inspectable rollout examples" in current
    assert "Artifact row 16, policy version 1" in current
    assert "intended automatic (`TransportType.Unset`) preference order" in current
    assert "same host  → SharedMemory" in current
    assert "cross host → Gloo" in current
    assert "RL_MULTIHOST_VERDICT: ok rows=40 versions=[0, 1, 2, 3]" in current
    assert "cross-host runs force Gloo" not in current


def test_legacy_operator_pages_are_marked_deprecated():
    for relative in (
        "trl.md",
        "2026-07-06_multinode-grpo-root-cause.md",
    ):
        text = (DOCS / relative).read_text()
        assert "[!WARNING]" in text
        assert CURRENT in text


def test_history_index_forwards_to_current_runbook():
    text = (DOCS / "history" / "README.md").read_text()

    assert "Every page in this directory is historical" in text
    assert CURRENT in text


def test_runtime_history_pages_have_direct_deprecation_banners():
    for relative in (
        "2026-06-13-bringup-and-2026-07-01-desync.md",
        "2026-06-14_monarch-torch213-deep-dive.md",
        "2026-07-19_monarch-single-host-grpo.md",
        "grpo-lora-agpt2b-repro.md",
        "grpo-lora-xpu-repro.md",
        "upstream-rl-port-status.md",
        "vllm-xpu-current-status.md",
        "vllm-xpu-investigation.md",
        "vllm-xpu-wiring-plan.md",
    ):
        text = (DOCS / "history" / relative).read_text()
        assert "[!WARNING]" in text
        assert CURRENT in text
