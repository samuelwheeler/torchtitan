# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import fcntl
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def staged_export(output_dir):
    """Lock one destination and atomically publish a successful export."""
    output_dir = Path(output_dir).absolute()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with (output_dir.parent / f".{output_dir.name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output_dir.is_symlink() or (
            output_dir.exists()
            and (not output_dir.is_dir() or any(output_dir.iterdir()))
        ):
            raise FileExistsError(f"Refusing to overwrite export: {output_dir}")
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{output_dir.name}.staging-", dir=output_dir.parent
            )
        )
        try:
            yield staging
            _validated_files(staging)
            os.replace(staging, output_dir)
        finally:
            if staging.exists():
                print(f"Unpublished export retained at {staging}", file=sys.stderr)


def write_export_manifest(output_dir, provenance):
    output_dir = Path(output_dir)
    files = {
        path.name: path.stat().st_size
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "ezpz_export.json"
    }
    (output_dir / "ezpz_export.json").write_text(
        json.dumps({**provenance, "complete": True, "files": files}, indent=2) + "\n"
    )


def _validated_files(output_dir):
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "ezpz_export.json").read_text())
    if manifest.get("complete") is not True or not manifest.get("files"):
        raise ValueError(
            "HF export lacks a completion manifest; reconvert to a fresh output root"
        )
    files = manifest["files"]
    actual = {
        path.name
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "ezpz_export.json"
    }
    if actual != set(files):
        raise ValueError("HF export file set differs from its completion manifest")
    for name, size in files.items():
        if Path(name).name != name or not (output_dir / name).is_file():
            raise ValueError(f"Missing HF export file: {name}")
        if not size or (output_dir / name).stat().st_size != size:
            raise ValueError(f"Incomplete HF export file: {name}")
    return files


def validate_export(output_dir):
    """Reject exports without a complete manifest, assets, or all weight shards."""
    output_dir = Path(output_dir)
    files = _validated_files(output_dir)
    required = {"config.json", "tokenizer_config.json"}
    if not required.issubset(files) or not {
        "tokenizer.json",
        "tokenizer.model",
    }.intersection(files):
        raise ValueError("HF export is missing model/tokenizer assets")
    index_path = output_dir / "model.safetensors.index.json"
    if index_path.is_file():
        shards = set(json.loads(index_path.read_text())["weight_map"].values())
    else:
        shards = {"model.safetensors"}
    if not shards or not shards.issubset(files):
        raise ValueError("HF export is missing weight shards")
