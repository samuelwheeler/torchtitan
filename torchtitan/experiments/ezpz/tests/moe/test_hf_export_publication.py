# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
import sys
from pathlib import Path

import pytest

from torchtitan.experiments.ezpz.eval.hf_export import (
    staged_export,
    validate_export,
    write_export_manifest,
)


def _assets(directory):
    for name in (
        "config.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "model.safetensors",
    ):
        (directory / name).write_text("{}")
    write_export_manifest(directory, {})


def test_export_is_hidden_until_complete_and_rejects_concurrent_writer(tmp_path):
    output = tmp_path / "hf"
    with staged_export(output) as staging:
        _assets(staging)
        assert not output.exists()
        code = (
            "import sys\n"
            "from torchtitan.experiments.ezpz.eval.hf_export import staged_export\n"
            "with staged_export(sys.argv[1]):\n"
            "    pass\n"
        )
        contender = subprocess.run(
            [sys.executable, "-c", code, str(output)], capture_output=True, text=True
        )
        assert contender.returncode != 0
        assert "BlockingIOError" in contender.stderr
    validate_export(output)
    with pytest.raises(FileExistsError):
        with staged_export(output):
            pytest.fail("Existing exports must not be overwritten")
    validate_export(output)


@pytest.mark.parametrize(
    "corruption", ["manifest", "asset", "shard", "truncated", "extra"]
)
def test_incomplete_exports_are_rejected(tmp_path, corruption):
    _assets(tmp_path)
    if corruption == "manifest":
        (tmp_path / "ezpz_export.json").write_text("{}")
    elif corruption == "truncated":
        (tmp_path / "model.safetensors").write_text("")
    elif corruption == "extra":
        (tmp_path / "stale.safetensors").write_text("old export")
    else:
        name = "config.json" if corruption == "asset" else "model.safetensors"
        (tmp_path / name).rename(tmp_path / f"{name}.backup")
    with pytest.raises(ValueError):
        validate_export(tmp_path)


def test_eval_only_rejects_interrupted_export(tmp_path):
    (tmp_path / "hf").mkdir()
    script = Path(__file__).parents[2] / "scripts/eval/convert_and_eval.sh"
    env = {**os.environ, "LM_EVAL_PYTHON": sys.executable}
    result = subprocess.run(
        [
            "bash",
            str(script),
            "--model",
            "12b2a",
            "--step",
            "1",
            "--eval-only",
            "--output-root",
            str(tmp_path),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "ezpz_export.json" in result.stderr
