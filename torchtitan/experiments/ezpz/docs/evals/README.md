# Evaluation Results

Benchmark evaluations of AuroraGPT production checkpoints using
[lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness).

> **[Eval-suite strategy review (2026-07)](eval-landscape-2026-07.md)** -- why
> the older commonsense suite is saturated, which modern benchmarks we added
> (MMLU/GSM8K/ARC-Challenge), and how we compare to SmolLM3-3B / Llama-3.2 /
> OLMo-2.

> **Modern-task coverage (audited 2026-09-30):** MMLU and GSM8K results exist,
> but the overview charts intentionally remain the seven-task commonsense
> training thermometer. The corpus has explicit MMLU-5 ladders for both 20B
> lineages and several MDS endpoints, while older 2B-v2 MMLU artifacts omit
> shot metadata. GSM8K coverage is endpoint-sparse. See the strategy review's
> coverage audit for exact counts and the policy for separate modern-task
> reporting; these artifacts must not be merged into shared curves by task
> name alone.

## All-production overlay (vs tokens)

One chart, 4 panels (HellaSwag acc_norm, ARC-Easy acc, ARC-C acc_norm,
Winogrande acc), 4 trajectories overlaid: 2B-MDS reference, 2B 256N async,
2B 512N sync, 20B 512N sync. X-axis = tokens consumed (linear) so
GBS-different trajectories are directly comparable.

![All-production eval overlay](figures/all_production_evals.svg)

**Headline (2026-07-24):** the 20B 512N sync chain (green) at
~609B tokens (step 6,050, loss ~2.44, 13.0% of the 4.67T target) is **already at the level the 2B chains reach around
~2T tokens** on HellaSwag norm and ARC-Easy. The 2B-MDS reference
(blue, ~7.77T tokens, SophiaG continuation) sets the upper-bound
ceiling for the 2B size class — the v2 256N 2B chain (salmon-red)
has since reached its full 4.67T budget (step-92,859, loss ~2.65)
and the 512N chain (dark red) is at ~3.99T (step-39,600, 85.3%).

Regenerate with:

```bash
.venv/bin/python -m torchtitan.experiments.ezpz.eval.plot_evals_combined
```

## Per-trajectory eval pages

| Model | Source | Steps Evaluated | Status |
|-------|--------|-----------------|--------|
| [agpt 2B](agpt/2b/) | torchtitan DCP (v1 + v2 256N async + v2 512N sync) | v2 256N through step-92,859 + v2 512N through step-39,600 | **🏁 Sync-mode workaround validated 2026-05-24** |
| [agpt 20B](agpt/20b/) | torchtitan DCP (v1 + v2 512N sync + v2 256N) | v2 512N through step-6,050 + v2 256N through step-5,900 | **🏁 20B 512N sync beats 2B 256N async per token on every benchmark (2026-05-27)** -- see the RoPE note below |
| [agpt 2B (MDS)](agpt/2b-mds/) | Megatron-DeepSpeed SophiaG | steps 5K–140K (28 unique × 3 replicates) | Done — clean reference baseline |

> **Chain-state note (added 2026-08-31).** The "Steps Evaluated" column and the
> 2026-07-24 headline above are a July snapshot; the chains have since advanced
> and then STOPPED. Current disk-audited heads: **20B-256 step-12,000**,
> **20B-512 step-10,600**, both idle since 2026-08-26; **2B-256 COMPLETE at
> step-92,859** and **2B-512 COMPLETE at step-46,429** (2.68687, 4.674T, 100% of
> budget). Production last trained 2026-08-26. Nothing here is live -- see
> [`production/README.md`](../production/README.md) for the authoritative state.
>
> **RoPE note on the 20B row (added 2026-08-17).** Both step figures above
> (512N step-6,050, 256N step-5,900) sit past those chains' RoPE-convention
> switches (4,401 and 3,101), so the numbers behind that headline came from
> wrongly-permuted exports. The direction is favourable -- corrupted exports
> UNDERSTATE the model, measured at up to -0.089 on ARC-C and growing with
> training -- so the "20B beats 2B per token" claim is if anything
> conservative. But it was made on data now known to be unreliable, and
> `rope-flavor-mismatch.md` names this specific comparison as confounded, so
> it should be re-derived from the corrected exports rather than assumed to
> survive. The re-eval sweep is producing them now; the 2B-256 side needs no
> redo (that chain never switched).
> *(2026-08-31: the sweep completed -- 72/72 with 32 series, per the
> 2026-08-21 session record -- and the corrected data is what
> `all_production_evals.svg` now plots. The comparison itself has still not
> been re-derived from the corrected exports.)*

## Pipelines

```
# torchtitan DCP runs
DCP checkpoint → eval/convert_to_hf.py → HF safetensors → lm-eval (HF backend, XPU)

# Megatron-DeepSpeed runs
mp_rank_00_model_states.pt → eval/mds_to_hf.py → HF safetensors → lm-eval (HF backend, XPU)
```

See `scripts/eval/convert_and_eval.sh` (DCP) and `scripts/eval/eval_mds_sweep.sh` (MDS)
for the end-to-end scripts. Aggregate with
`eval/aggregate_evals.py --model {2b,20b,2b-mds}`.

The same DCP pipeline supports the custom 12B-total/2B-active AGPT MoE model;
see the [AGPT MoE HF export guide](../guides/agpt-moe-hf-eval.md).

> Note: the canonical chart/table path is now `scripts/update_all_charts.sh`
> (via the per-model `docs/evals/agpt/{2b,20b}/plot_eval_overview.py`), which
> `scripts/refresh_all.sh` runs automatically. `aggregate_evals.py` remains a
> manual fallback aggregator.

## Environment

- **Module:** `frameworks/2025.3.1` (bare, no user venv)
- **Must set:** `HF_HUB_ENABLE_HF_TRANSFER=0`
- **Device:** `--device xpu`
- **Do NOT** use `ezpz_setup_env` — the user venv has transformers 5.6.2
  which breaks lm-eval's HF backend
