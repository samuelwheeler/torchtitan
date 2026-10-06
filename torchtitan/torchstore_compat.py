# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Small, dependency-light helpers for configuring TorchStore."""

import logging
import os
import socket
from importlib import import_module

logger = logging.getLogger(__name__)


def torchstore_transport_from_env():
    """Return a forced TorchStore transport, or ``None`` for auto selection."""
    requested = os.environ.get("TORCHTITAN_TORCHSTORE_TRANSPORT", "auto").lower()
    if requested in ("", "auto", "unset"):
        return None

    names = {
        "gloo": "Gloo",
        "xccl": "XCCL",
        "shared_memory": "SharedMemory",
        "monarch_rpc": "MonarchRPC",
        "monarch_rdma": "MonarchRDMA",
    }
    if requested not in names:
        raise ValueError(
            "TORCHTITAN_TORCHSTORE_TRANSPORT must be one of "
            "auto, gloo, xccl, shared_memory, monarch_rpc, monarch_rdma; "
            f"got {requested!r}"
        )

    # Validate before importing so bad configuration is diagnosed even in a
    # dependency-light environment. TorchStore does not re-export this enum.
    transport_type = import_module("torchstore.transport").TransportType
    return getattr(transport_type, names[requested])


def hostname_env_is_unreliable(environ=None, actual_hostname=None) -> bool:
    """Return True when ``HOSTNAME`` does not describe the running host.

    TorchStore decides client/volume co-location by string-comparing
    ``os.environ.get("HOSTNAME", socket.gethostname())`` on both sides
    (``torchstore.utils.get_local_hostname``). ``HOSTNAME`` is an ordinary
    exported shell variable, so an MPI/scheduler launcher that propagates the
    submitting shell's environment gives every remote rank the *launcher's*
    hostname. Measured on a two-node Sunspot allocation, a rank really running
    on ``x1922c6s5b0n0`` reported ``HOSTNAME=x1922c6s3b0n0``.

    Both failure modes matter:

    * identical value on different hosts -> a remote volume looks local, so
      automatic selection picks SharedMemory and the first pull fails with
      "Shared memory storage not found";
    * FQDN vs short name on the same host -> a local volume looks remote, so
      a slower network transport is chosen silently.
    """
    environ = os.environ if environ is None else environ
    env_hostname = environ.get("HOSTNAME")
    if env_hostname is None:
        return False
    if actual_hostname is None:
        actual_hostname = socket.gethostname()
    return env_hostname != actual_hostname


def repair_hostname_env(environ=None, actual_hostname=None) -> "str | None":
    """Align ``HOSTNAME`` with the real hostname; return the corrected value.

    Returns ``None`` when no change was needed. Call this in each process
    *before* TorchStore resolves a transport, so locality classification is
    based on the host the process actually runs on.

    This repairs the process's own view only. It is deliberately narrow: it
    does not patch TorchStore internals, and it leaves ``HOSTNAME`` unset when
    it was never set (``socket.gethostname()`` is then already used).
    """
    environ = os.environ if environ is None else environ
    if actual_hostname is None:
        actual_hostname = socket.gethostname()
    if not hostname_env_is_unreliable(environ, actual_hostname):
        return None

    stale = environ.get("HOSTNAME")
    environ["HOSTNAME"] = actual_hostname
    logger.warning(
        "Corrected inherited HOSTNAME=%r to the real hostname %r. TorchStore "
        "resolves client/volume co-location from this value, so a propagated "
        "launcher hostname makes a remote storage volume look local and "
        "automatic transport selection wrongly chooses SharedMemory.",
        stale,
        actual_hostname,
    )
    return actual_hostname
