# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import shlex
import subprocess
import sys
from pathlib import Path

from torchtitan.experiments.ezpz.eval.hf_export import write_export_manifest


def test_export_validation_needs_no_training_dependencies(tmp_path):
    output = tmp_path / "hf"
    output.mkdir()
    for name in (
        "config.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "model.safetensors",
    ):
        (output / name).write_text("{}")
    write_export_manifest(output, {})
    interpreter = tmp_path / "python-without-site-packages"
    interpreter.write_text(f'#!/bin/bash\nexec {shlex.quote(sys.executable)} -S "$@"\n')
    interpreter.chmod(0o755)
    script = Path(__file__).parents[2] / "scripts/eval/convert_and_eval.sh"
    result = subprocess.run(
        [
            "bash",
            str(script),
            "--model",
            "12b2a",
            "--step",
            "1",
            "--eval-only",
            "--convert-only",
            "--output-root",
            str(tmp_path),
        ],
        env={**os.environ, "LM_EVAL_PYTHON": str(interpreter), "PYTHONPATH": ""},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
