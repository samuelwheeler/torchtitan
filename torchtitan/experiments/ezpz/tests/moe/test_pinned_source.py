# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "change", ["clean", "ignored_output", "tracked", "staged", "untracked", "wrong_sha"]
)
def test_pinned_source_guard(tmp_path, change):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "-q")
    (tmp_path / ".gitignore").write_text("outputs/\n")
    (tmp_path / "model.py").write_text("value = 1\n")
    git("add", ".")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "initial",
    )
    sha = git("rev-parse", "HEAD")
    if change in ("tracked", "staged"):
        (tmp_path / "model.py").write_text("value = 2\n")
        if change == "staged":
            git("add", "model.py")
    elif change == "untracked":
        (tmp_path / "extra_model.py").write_text("value = 3\n")
    elif change == "ignored_output":
        (tmp_path / "outputs").mkdir()
        (tmp_path / "outputs/result.json").write_text("{}")
    elif change == "wrong_sha":
        sha = "0" * 40
    guard = (
        Path(__file__).parents[2] / ".agents/skills/alcf-job-preflight/check_source.sh"
    )
    result = subprocess.run(
        ["bash", str(guard), str(tmp_path), sha], capture_output=True, text=True
    )
    if change in ("clean", "ignored_output"):
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode == 2, result.stderr
