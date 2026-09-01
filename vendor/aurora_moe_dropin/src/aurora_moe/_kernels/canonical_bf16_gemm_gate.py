"""Correctness and throughput gate for a contiguous Aurora BF16 GEMM backend."""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable

import torch


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("torch", "tla-builder-probe"), default="tla-builder-probe")
    parser.add_argument("--m", type=int, default=16384)
    parser.add_argument("--k", type=int, default=1408)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--minimum-tflops", type=float, default=170.0)
    parser.add_argument("--allow-below-threshold", action="store_true")
    return parser.parse_args()


def _assert_close(got: torch.Tensor, expected: torch.Tensor, backend: str) -> None:
    torch.xpu.synchronize()
    maximum = (got.float() - expected.float()).abs().max().item()
    if not torch.allclose(got, expected, atol=0.25, rtol=0.05):
        raise AssertionError(f"canonical BF16 GEMM {backend} mismatch: max_abs_diff={maximum:g}")
    print(f"canonical_bf16_gemm_correctness backend={backend} max_abs_diff={maximum:g}", flush=True)


def _candidate(
    backend: str, a: torch.Tensor, b: torch.Tensor
) -> Callable[[], torch.Tensor]:
    if backend == "torch":
        return lambda: a @ b
    from aurora_moe._kernels.tla_builder_gemm_probe import builder_gemm_probe_bf16

    c = torch.zeros(a.size(0), b.size(1), device=a.device, dtype=torch.bfloat16)
    return lambda: builder_gemm_probe_bf16(a, b, c, alpha=1.0, beta=0.0)


def _time(step: Callable[[], torch.Tensor], warmup: int, steps: int) -> tuple[float, float]:
    for _ in range(warmup):
        step()
    torch.xpu.synchronize()
    samples: list[float] = []
    for _ in range(steps):
        start = time.perf_counter()
        step()
        torch.xpu.synchronize()
        samples.append(time.perf_counter() - start)
    return sum(samples) / len(samples), min(samples)


def main() -> None:
    args = _args()
    if not torch.xpu.is_available():
        raise RuntimeError("this gate requires an Aurora XPU")
    if min(args.m, args.k, args.n, args.warmup, args.steps) <= 0:
        raise ValueError("M, K, N, warmup, and steps must be positive")
    if args.minimum_tflops <= 0:
        raise ValueError("minimum-tflops must be positive")
    torch.xpu.set_device(int(os.environ.get("PALS_LOCAL_RANKID", os.environ.get("LOCAL_RANK", "0"))))
    torch.manual_seed(20260721)
    a = torch.randn(args.m, args.k, device="xpu", dtype=torch.bfloat16)
    b = torch.randn(args.k, args.n, device="xpu", dtype=torch.bfloat16)
    expected = a @ b
    step = _candidate(args.backend, a, b)
    _assert_close(step(), expected, args.backend)
    average, best = _time(step, args.warmup, args.steps)
    flops = 2.0 * args.m * args.k * args.n
    average_tflops = flops / average / 1e12
    best_tflops = flops / best / 1e12
    print(
        "implementation=canonical_bf16_gemm_gate "
        f"backend={args.backend} M={args.m} K={args.k} N={args.n} "
        f"avg_ms={average * 1e3:.3f} best_ms={best * 1e3:.3f} "
        f"avg_tflops={average_tflops:.2f} best_tflops={best_tflops:.2f} "
        f"minimum_tflops={args.minimum_tflops:.2f}",
        flush=True,
    )
    if average_tflops < args.minimum_tflops and not args.allow_below_threshold:
        raise AssertionError(
            f"canonical BF16 GEMM gate failed: {average_tflops:.2f} < {args.minimum_tflops:.2f} TF/s"
        )
    print("canonical_bf16_gemm_gate=passed", flush=True)


if __name__ == "__main__":
    main()
