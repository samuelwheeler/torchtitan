#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Diagnose TorchStore's cross-host locality classification.

TorchStore decides whether a client and a storage volume are co-located by
comparing two strings: the volume reports ``get_local_hostname()`` from its own
process, and the client compares that against ``get_local_hostname()`` in its
process. Both resolve as::

    os.environ.get("HOSTNAME", socket.gethostname())

``HOSTNAME`` is an ordinary (often exported) shell variable, so a scheduler or
launcher that propagates the submitting shell's environment gives every remote
actor the SAME ``HOSTNAME`` while ``socket.gethostname()`` differs. Automatic
transport selection then believes a remote generator is local and picks the
SharedMemory path, which fails on the first pull with::

    Shared memory storage not found. This may indicate the storage volume is on
    a different host.

This probe spawns one actor per scheduler host, captures the inherited
``HOSTNAME``, applies ``repair_hostname_env()``, and compares the result with
the real ``socket.gethostname()``. It exits non-zero if the post-repair pair
still disagrees, so the fix is testable without allocating a model, a vLLM
engine, or a weight transfer.

Markers:
    ``TORCHSTORE_LOCALITY_OK``     resolution is consistent and distinct per host
    ``TORCHSTORE_LOCALITY_BUG``    ``HOSTNAME`` misreports actor locality
"""

from __future__ import annotations

import argparse
import json
import os
import socket
from typing import Any

import torch.distributed as dist
from monarch._src.spmd.host_mesh import host_mesh_from_store
from monarch.actor import Actor, endpoint


class LocalityProbe(Actor):
    @endpoint
    async def identity(self) -> dict[str, Any]:
        """Report pre-fix state, apply the repair, and report resolved state."""
        inherited_hostname = os.environ.get("HOSTNAME")
        real_hostname = socket.gethostname()
        from torchtitan.torchstore_compat import repair_hostname_env

        repaired = repair_hostname_env()
        env_hostname = os.environ.get("HOSTNAME")
        return {
            "inherited_hostname": inherited_hostname,
            "env_hostname": env_hostname,
            "real_hostname": real_hostname,
            # This mirrors torchstore.utils.get_local_hostname() byte for byte.
            "torchstore_resolved": os.environ.get("HOSTNAME", real_hostname),
            "repair_result": repaired,
            "pid": os.getpid(),
        }


def _rank() -> int:
    for name in ("RANK", "PMI_RANK", "PALS_RANKID"):
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    raise RuntimeError("scheduler rank environment is missing")


def classify(rows: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    """Return ``(ok, problems)`` for the collected per-actor identities."""
    problems: list[str] = []

    real = [row["real_hostname"] for row in rows]
    resolved = [row["torchstore_resolved"] for row in rows]

    if len(set(real)) != len(real):
        problems.append(
            f"actors are not on distinct physical hosts: real_hostnames={real}"
        )

    # The actual defect: TorchStore's resolution collapses distinct hosts.
    if len(set(real)) > 1 and len(set(resolved)) == 1:
        problems.append(
            "TorchStore resolves every actor to one hostname "
            f"({resolved[0]!r}) while the real hosts are {sorted(set(real))}; "
            "is_local_to_volume() will report co-location for a remote volume "
            "and automatic selection will wrongly choose SharedMemory"
        )

    for row in rows:
        inherited_hostname = row.get("inherited_hostname")
        env_hostname = row["env_hostname"]
        if (
            inherited_hostname is not None
            and inherited_hostname != row["real_hostname"]
        ):
            print(
                "LOCALITY_REPAIRED "
                f"inherited={inherited_hostname!r} "
                f"real={row['real_hostname']!r} resolved={env_hostname!r}",
                flush=True,
            )
        if env_hostname is not None and env_hostname != row["real_hostname"]:
            problems.append(
                f"HOSTNAME={env_hostname!r} does not match "
                f"socket.gethostname()={row['real_hostname']!r} "
                f"(pid={row['pid']})"
            )

    return (not problems), problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--world-size", default=2, type=int)
    args = parser.parse_args()

    rank = _rank()
    store = dist.FileStore(args.store, args.world_size)
    hosts = host_mesh_from_store(
        store,
        monarch_port=args.port,
        name="torchstore_locality_probe",
        transport="tcp",
        rank=rank,
        local_rank=0,
        world_size=args.world_size,
        local_world_size=1,
    )

    if rank != 0:
        store.get("torchstore_locality_done")
        return

    assert hosts is not None
    try:
        hosts.initialized.get()
        actors = hosts.spawn_procs(per_host={"procs": 1}).spawn(
            "locality", LocalityProbe
        )
        identities: Any = actors.identity.call().get()
        rows = [dict(value) for value in identities.values()]
        for row in rows:
            print(f"LOCALITY_ROW {json.dumps(row, sort_keys=True)}", flush=True)

        ok, problems = classify(rows)
        for problem in problems:
            print(f"LOCALITY_PROBLEM {problem}", flush=True)
        if ok:
            print("TORCHSTORE_LOCALITY_OK", flush=True)
        else:
            print("TORCHSTORE_LOCALITY_BUG", flush=True)
            raise SystemExit(1)
    finally:
        try:
            hosts.shutdown().get()
        finally:
            store.set("torchstore_locality_done", b"1")


if __name__ == "__main__":
    main()
