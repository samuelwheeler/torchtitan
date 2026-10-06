# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Contracts for TorchStore cross-host locality classification.

TorchStore decides whether a client and a storage volume share a host by
string-comparing ``os.environ.get("HOSTNAME", socket.gethostname())`` on both
sides. ``HOSTNAME`` is an ordinary exported shell variable, so MPI/scheduler
launchers propagate the submitting shell's value to every remote rank.

Measured on a two-node Sunspot allocation (job ``12479166``)::

    {"env_hostname": "x1922c6s3b0n0", "real_hostname": "x1922c6s3b0n0", ...}
    {"env_hostname": "x1922c6s3b0n0", "real_hostname": "x1922c6s5b0n0", ...}

Both ranks resolve to ``x1922c6s3b0n0``, so ``is_local_to_volume()`` reports a
remote volume as local and automatic selection picks SharedMemory.
"""

import socket

from torchtitan.torchstore_compat import hostname_env_is_unreliable, repair_hostname_env

# The exact values observed on Sunspot job 12479166.
LAUNCHER_HOST = "x1922c6s3b0n0"
REMOTE_HOST = "x1922c6s5b0n0"


def _torchstore_resolved(environ, actual_hostname):
    """Mirror torchstore.utils.get_local_hostname() exactly."""
    return environ.get("HOSTNAME", actual_hostname)


def _is_local_to_volume(client_env, client_host, volume_env, volume_host):
    """Mirror torchstore.transport.shared_memory.is_local_to_volume().

    The volume reports its hostname via ``StorageVolume.get_id()``; the client
    compares it against its own resolution.
    """
    volume_hostname = _torchstore_resolved(volume_env, volume_host)
    return volume_hostname == _torchstore_resolved(client_env, client_host)


def test_inherited_hostname_makes_a_remote_volume_look_local():
    """The reported defect, reproduced from measured values."""
    client_env = {"HOSTNAME": LAUNCHER_HOST}
    volume_env = {"HOSTNAME": LAUNCHER_HOST}

    assert _is_local_to_volume(
        client_env, LAUNCHER_HOST, volume_env, REMOTE_HOST
    ), "expected the unrepaired environment to misreport co-location"


def test_repair_restores_correct_cross_host_classification():
    client_env = {"HOSTNAME": LAUNCHER_HOST}
    volume_env = {"HOSTNAME": LAUNCHER_HOST}

    repair_hostname_env(client_env, LAUNCHER_HOST)
    repair_hostname_env(volume_env, REMOTE_HOST)

    assert not _is_local_to_volume(client_env, LAUNCHER_HOST, volume_env, REMOTE_HOST)


def test_repair_preserves_genuine_same_host_locality():
    """A real same-host pair must still select the shared-memory path."""
    client_env = {"HOSTNAME": LAUNCHER_HOST}
    volume_env = {"HOSTNAME": LAUNCHER_HOST}

    repair_hostname_env(client_env, LAUNCHER_HOST)
    repair_hostname_env(volume_env, LAUNCHER_HOST)

    assert _is_local_to_volume(client_env, LAUNCHER_HOST, volume_env, LAUNCHER_HOST)


def test_fqdn_hostname_is_treated_as_unreliable():
    """Observed on Sunspot: HOSTNAME is the FQDN, gethostname() is short.

    Left unrepaired this is the inverse failure -- a local volume looks remote,
    silently selecting a slower network transport.
    """
    fqdn = f"{LAUNCHER_HOST}.hostmgmt2001.cm.sunspot.alcf.anl.gov"
    environ = {"HOSTNAME": fqdn}

    assert hostname_env_is_unreliable(environ, LAUNCHER_HOST)
    assert repair_hostname_env(environ, LAUNCHER_HOST) == LAUNCHER_HOST
    assert environ["HOSTNAME"] == LAUNCHER_HOST


def test_repair_is_a_noop_when_hostname_is_absent_or_correct():
    absent: dict[str, str] = {}
    assert not hostname_env_is_unreliable(absent, LAUNCHER_HOST)
    assert repair_hostname_env(absent, LAUNCHER_HOST) is None
    assert "HOSTNAME" not in absent, "must not invent HOSTNAME where none was set"

    correct = {"HOSTNAME": LAUNCHER_HOST}
    assert not hostname_env_is_unreliable(correct, LAUNCHER_HOST)
    assert repair_hostname_env(correct, LAUNCHER_HOST) is None
    assert correct["HOSTNAME"] == LAUNCHER_HOST


def test_explicitly_empty_hostname_is_repaired():
    """An empty value is not the same as an absent value to TorchStore."""
    environ = {"HOSTNAME": ""}

    assert hostname_env_is_unreliable(environ, LAUNCHER_HOST)
    assert repair_hostname_env(environ, LAUNCHER_HOST) == LAUNCHER_HOST
    assert environ["HOSTNAME"] == LAUNCHER_HOST


def test_repair_defaults_to_the_real_process_hostname():
    environ = {"HOSTNAME": "definitely-not-this-host"}
    assert repair_hostname_env(environ) == socket.gethostname()
    assert environ["HOSTNAME"] == socket.gethostname()
