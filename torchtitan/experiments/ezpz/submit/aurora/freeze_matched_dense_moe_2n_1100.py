#!/usr/bin/env python3
"""Freeze one self-contained source image for Aurora MoE training."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import tarfile


HERE = Path(__file__).resolve()
REPO = HERE.parents[5]
PACKAGE = REPO / "vendor" / "aurora_moe_dropin"
NON_RUNTIME_PREFIXES = (
    ".ci/",
    ".claude/",
    ".github/",
    "assets/",
    "benchmarks/",
    "docs/",
    "tests/",
    "vendor/",
    "torchtitan/experiments/ezpz/docs/",
)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tracked_files():
    raw = subprocess.check_output(["git", "ls-files", "-z"], cwd=str(REPO))
    paths = [
        value.decode()
        for value in raw.split(b"\0")
        if value and not value.decode().startswith(NON_RUNTIME_PREFIXES)
    ]
    return sorted(set(paths))


def package_files():
    root = PACKAGE / "src" / "aurora_moe"
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO / "outputs" / "_job_inputs" / "aurora_moe_12b2a",
    )
    args = parser.parse_args()

    pair_id = "CODEX_AURORA_MOE_12B2A_{}_{}".format(
        datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"), secrets.token_hex(4)
    )
    output = args.output_root.resolve() / pair_id
    output.mkdir(parents=True)

    records = []
    repo_entries = []
    for relative in tracked_files():
        path = REPO / relative
        if not path.exists() and not path.is_symlink():
            raise FileNotFoundError(path)
        if path.is_dir():
            continue
        data = path.read_bytes() if not path.is_symlink() else os.readlink(str(path)).encode()
        records.append(
            {"path": relative, "sha256": sha256_bytes(data), "size": len(data)}
        )
        repo_entries.append((path, relative))

    package_entries = []
    for path in package_files():
        relative = path.relative_to(PACKAGE).as_posix()
        archive_name = "vendor/aurora_moe_dropin/{}".format(relative)
        data = path.read_bytes()
        records.append(
            {"path": archive_name, "sha256": sha256_bytes(data), "size": len(data)}
        )
        package_entries.append((path, archive_name))

    records.sort(key=lambda value: value["path"])
    manifest_data = b"".join(
        (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for record in records
    )
    manifest = output / "source.manifest.jsonl"
    manifest.write_bytes(manifest_data)
    manifest_sha = sha256_bytes(manifest_data)
    (output / "source.manifest.sha256").write_text(
        "{}  source.manifest.jsonl\n".format(manifest_sha)
    )

    repo_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO)
    ).decode().strip()
    status = subprocess.check_output(
        ["git", "status", "--short"], cwd=str(REPO)
    ).decode()
    provenance = {
        "schema": 1,
        "pair_id": pair_id,
        "created_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repo": str(REPO),
        "repo_head": repo_head,
        "repo_status_sha256": sha256_bytes(status.encode()),
        "source_manifest_sha256": manifest_sha,
        "source_file_count": len(records),
    }
    provenance_data = (
        json.dumps(provenance, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    provenance_path = output / "snapshot_provenance.json"
    provenance_path.write_bytes(provenance_data)

    archive = output / "source.tar.gz"
    with tarfile.open(str(archive), "w:gz", compresslevel=6) as tar:
        for path, archive_name in repo_entries + package_entries:
            tar.add(str(path), arcname=archive_name, recursive=False)
        tar.add(str(manifest), arcname=".codex_snapshot/source.manifest.jsonl", recursive=False)
        tar.add(str(provenance_path), arcname=".codex_snapshot/snapshot_provenance.json", recursive=False)

    archive_sha = sha256_file(archive)
    provenance["source_archive_sha256"] = archive_sha
    provenance_path.write_bytes(
        (json.dumps(provenance, sort_keys=True, separators=(",", ":")) + "\n").encode()
    )

    print("PAIR_ID={}".format(pair_id))
    print("TT_INPUT_DIR={}".format(output))
    print("TT_SOURCE_ARCHIVE_SHA256={}".format(archive_sha))
    print("SOURCE_MANIFEST_SHA256={}".format(manifest_sha))
    print("SOURCE_FILE_COUNT={}".format(len(records)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
