# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Contracts for the JSON-backed AGPT launch configurations."""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path

import pytest

from torchtitan.distributed.activation_checkpoint import SelectiveAC


EZPZ_ROOT = Path(__file__).parents[1]
MOE_RUNS = EZPZ_ROOT / "moe_runs"
LAUNCHERS = (
    EZPZ_ROOT / "submit/aurora/submit_agpt_dense_moe_256n_50k.pbs",
    EZPZ_ROOT / "submit/aurora/submit_agpt_moe_full_sonic_2n_1100.pbs",
)
AGPT_JSONS = (
    "agpt_dense_2b_50k_hsdp256_50k.json",
    "agpt_moe_2b_50k_ep12_dp256_50k.json",
    "agpt_2b_50k_moe_ep12_dp2_full_sonic_1100.json",
    "agpt_2b_50k_moe_ep12_1node_smoke.json",
)


def _factory(module_name: str, factory_name: str, json_path: Path, monkeypatch):
    monkeypatch.setenv("TT_CONFIG_JSON", str(json_path))
    module = importlib.import_module(module_name)
    return getattr(module, factory_name)()


def _expert_backends(cfg) -> set[str]:
    return {
        layer.moe.routed_experts.compute_backend
        for layer in cfg.model.layers
        if layer.moe is not None
    }


def test_every_retained_moe_run_json_parses():
    paths = sorted(MOE_RUNS.rglob("*.json"))
    assert paths
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            assert isinstance(json.load(stream), dict), path


def test_agpt_launch_jsons_use_current_schema():
    legacy_training = {"local_batch_size", "seq_len"}
    legacy_optimizer = {"name", "lr", "beta1", "beta2", "eps", "weight_decay"}
    for json_name in AGPT_JSONS:
        path = MOE_RUNS / json_name
        data = json.loads(path.read_text())
        assert not legacy_training.intersection(data.get("training", {})), path
        assert "lr_scheduler" not in data, path
        assert "optimizer" not in data, path
        assert "lr_scheduler" in data.get("optim", {}), path
        assert not legacy_optimizer.intersection(
            data.get("optim", {}).get("optimizer", {})
        ), path
        activation_checkpoint = data.get("activation_checkpoint")
        assert not (
            isinstance(activation_checkpoint, dict) and "mode" in activation_checkpoint
        ), path
        assert "checkpoint" not in data, path
        assert "enable" not in data.get("compile", {}), path
        checkpointer = data.get("checkpointer", {})
        assert checkpointer.get("keep_latest_k", 0) == 0, path


@pytest.mark.parametrize(
    "module_name,factory_name,json_name,ac_type,backend",
    [
        (
            "torchtitan.experiments.ezpz.agpt.config_registry",
            "agpt_2b_50k_from_json",
            "agpt_dense_2b_50k_hsdp256_50k.json",
            SelectiveAC.Config,
            None,
        ),
        (
            "torchtitan.experiments.ezpz.moe.config_registry",
            "agpt_2b_50k_moe_sdpa_aurora_full_sonic_from_json",
            "agpt_moe_2b_50k_ep12_dp256_50k.json",
            None,
            "aurora_full_sonic",
        ),
        (
            "torchtitan.experiments.ezpz.moe.config_registry",
            "agpt_2b_50k_moe_sdpa_aurora_full_sonic_from_json",
            "agpt_2b_50k_moe_ep12_dp2_full_sonic_1100.json",
            None,
            "aurora_full_sonic",
        ),
        (
            "torchtitan.experiments.ezpz.moe.config_registry",
            "agpt_2b_50k_moe_sdpa_aurora_full_sonic_from_json",
            "agpt_2b_50k_moe_ep12_1node_smoke.json",
            None,
            "aurora_full_sonic",
        ),
    ],
)
def test_retained_agpt_json_factories_load(
    module_name, factory_name, json_name, ac_type, backend, monkeypatch
):
    cfg = _factory(module_name, factory_name, MOE_RUNS / json_name, monkeypatch)

    assert cfg.model.vocab_size == 50_304
    if ac_type is None:
        assert cfg.activation_checkpoint is None
    else:
        assert isinstance(cfg.activation_checkpoint, ac_type)
    if cfg.checkpointer is not None:
        assert cfg.checkpointer.keep_latest_k == 0
    assert type(cfg.optim.optimizer.optimizers[0]).__qualname__ == "AdamW.Config"
    assert cfg.optim.optimizer.optimizers[0].lr == pytest.approx(2.2e-4)
    if backend is not None:
        from torchtitan.experiments.ezpz.moe.agpt_hf_state_dict_adapter import (
            AGPTMoEStateDictAdapter,
        )

        assert cfg.parallelism.expert_parallel_degree == 12
        assert _expert_backends(cfg) == {backend}
        assert cfg.model.build().state_dict_adapter_cls is AGPTMoEStateDictAdapter


def test_agpt_json_rejects_checkpoint_rotation(monkeypatch, tmp_path):
    path = tmp_path / "unsafe.json"
    path.write_text(json.dumps({"checkpoint": {"keep_latest_k": 2}}))

    with pytest.raises(ValueError, match="keep_latest_k=2"):
        _factory(
            "torchtitan.experiments.ezpz.agpt.config_registry",
            "agpt_2b_50k_from_json",
            path,
            monkeypatch,
        )


def test_retained_launchers_reference_importable_factories():
    assignment = re.compile(
        r"^\s*(MODULE|CONFIG)=(?:\"([^\"]+)\"|([^\s;]+))", re.MULTILINE
    )
    explicit_import = re.compile(
        r"from torchtitan\.experiments\.ezpz\.(\w+)\.config_registry import (\w+)"
    )

    references: set[tuple[str, str]] = set()
    for launcher in LAUNCHERS:
        text = launcher.read_text()
        values = {key: quoted or bare for key, quoted, bare in assignment.findall(text)}
        if "MODULE" in values and "CONFIG" in values:
            references.add((values["MODULE"].removeprefix("ezpz."), values["CONFIG"]))
        references.update(explicit_import.findall(text))

    assert references
    for module_name, factory_name in references:
        module = importlib.import_module(
            f"torchtitan.experiments.ezpz.{module_name}.config_registry"
        )
        assert callable(getattr(module, factory_name)), (module_name, factory_name)


def test_full_sonic_launcher_exports_required_runtime_contract():
    launcher = EZPZ_ROOT / "submit/aurora/submit_agpt_moe_full_sonic_2n_1100.pbs"
    text = launcher.read_text()

    assert "vendor/aurora_moe_dropin/src" in text
    assert "export AURORA_MOE_ALLTOALLV=1" in text
    assert "export AURORA_MOE_SEGMENTED_SONIC=1" in text
