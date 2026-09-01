"""Process-group helpers for a DP x EP MoE layout."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class MoEProcessGroups:
    """EP dispatch and the DP groups for replicated and sharded parameters."""

    ep_dispatch: object | None = None
    dense_dp: object | None = None
    sparse_dp: object | None = None


class ParallelMesh:
    """Small mesh contract consumed by the packaged exact MoE runtime."""

    def __init__(self, groups: MoEProcessGroups, device: torch.device) -> None:
        self.device = device
        self.groups = {
            "ep_dispatch": groups.ep_dispatch,
            "dense_dp": groups.dense_dp,
            "sparse_dp": groups.sparse_dp,
        }
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.group_size = {
                name: dist.get_world_size(group) for name, group in self.groups.items()
            }
            self.group_rank = {
                name: dist.get_rank(group) for name, group in self.groups.items()
            }
            # The Level Zero transports use the global rank list to derive a
            # unique node-local descriptor socket for each EP group.  Keep it
            # in the small mesh contract just as the original benchmark mesh
            # does.  ``None`` is PyTorch's world process group.
            self.ranks = {
                name: (
                    list(range(dist.get_world_size()))
                    if group is None
                    else list(dist.get_process_group_ranks(group))
                )
                for name, group in self.groups.items()
            }
        else:
            self.rank = 0
            self.group_size = {name: 1 for name in self.groups}
            self.group_rank = {name: 0 for name in self.groups}
            self.ranks = {name: [0] for name in self.groups}


def create_dp_ep_groups(dp_size: int, ep_size: int) -> MoEProcessGroups:
    """Create fixed-order process groups for rank = dp_rank * EP + ep_rank."""

    if dp_size <= 0 or ep_size <= 0:
        raise ValueError("dp_size and ep_size must be positive")
    if not dist.is_available() or not dist.is_initialized():
        if dp_size != 1 or ep_size != 1:
            raise RuntimeError("initialize torch.distributed before creating DP x EP groups")
        return MoEProcessGroups()

    world_size = dist.get_world_size()
    if world_size != dp_size * ep_size:
        raise ValueError(
            f"world_size={world_size} does not equal dp_size * ep_size={dp_size * ep_size}"
        )

    rank = dist.get_rank()
    ep_group = None
    for dp_rank in range(dp_size):
        ranks = list(range(dp_rank * ep_size, (dp_rank + 1) * ep_size))
        group = dist.new_group(ranks=ranks)
        if rank in ranks:
            ep_group = group

    sparse_dp_group = None
    for ep_rank in range(ep_size):
        ranks = [dp_rank * ep_size + ep_rank for dp_rank in range(dp_size)]
        group = dist.new_group(ranks=ranks)
        if rank in ranks:
            sparse_dp_group = group

    if ep_group is None or sparse_dp_group is None:
        raise RuntimeError("could not resolve this rank's DP x EP process groups")
    # In the original DP x EP mesh, routing and shared experts are replicated
    # over both dimensions.  ``None`` denotes PyTorch's world process group.
    return MoEProcessGroups(
        ep_dispatch=ep_group,
        dense_dp=None,
        sparse_dp=sparse_dp_group,
    )


def default_device() -> torch.device:
    """Return the active Aurora tile when available, otherwise CPU."""

    if torch.xpu.is_available():
        return torch.device("xpu", torch.xpu.current_device())
    return torch.device("cpu")
