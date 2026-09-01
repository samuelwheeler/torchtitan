"""Run one exact AuroraMoE distributed training step."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from aurora_moe import AuroraMoE, configure_native_runtime, create_dp_ep_groups


def run_train_step() -> None:
    configure_native_runtime(router_grad=True)
    dist.init_process_group("xccl")
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("PALS_LOCAL_RANKID", "0")))
    torch.xpu.set_device(local_rank)
    groups = create_dp_ep_groups(
        dp_size=int(os.environ["AURORA_MOE_DP_SIZE"]),
        ep_size=int(os.environ["AURORA_MOE_EP_SIZE"]),
    )
    moe = AuroraMoE(
        model_dim=2048,
        expert_hidden_dim=1408,
        num_experts=36,
        top_k=3,
        shared_experts=2,
        groups=groups,
        device="xpu",
        dtype=torch.bfloat16,
    )
    optimizer = torch.optim.AdamW(moe.parameters(), lr=1e-4)
    x = torch.randn(16_384, 2048, device="xpu", dtype=torch.bfloat16)
    loss = moe(x).float().square().mean()
    loss.backward()
    moe.finalize_gradients()
    optimizer.step()
    if dist.get_rank() == 0:
        print(f"loss={loss.item():.6f}")


if __name__ == "__main__":
    run_train_step()
