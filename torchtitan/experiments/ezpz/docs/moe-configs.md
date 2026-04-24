# MoE Training Configs

## Model Configs

All configs use `vocab_size=256128` (Gemma tokenizer) and `seq_len=8192`.

| Config | Total Params | Active Params | Layers | Experts | Top-K | Dim |
|--------|-------------|--------------|--------|---------|-------|-----|
| `moe_debugmodel` | 0.05B | 0.04B | 6 | 8 | 3 | 256 |
| `moe_500m` | 0.25B | 0.14B | 12 | 16 | 3 | 512 |
| `moe_2b` | 1.61B | 0.49B | 18 | 24 | 3 | 1024 |
| `moe_4b` | 2.89B | 0.81B | 22 | 24 | 3 | 1536 |
| `moe_7b` | 7.54B | 1.57B | 24 | 36 | 3 | 2048 |
| `moe_10b_2b` | 9.41B | 1.98B | 27 | 36 | 3 | 2048 |

### EP-enabled variants

Use `AllToAllTokenDispatcher` for expert parallelism (EP > 1):

| Config | Base | Default EP |
|--------|------|-----------|
| `moe_debugmodel_ep` | debugmodel | 2 |
| `moe_2b_ep` | 2B | 2 |
| `moe_7b_ep` | 7B | 2 |
| `moe_10b_2b_sdpa_ep` | 10B_2B_sdpa | 2 |

Override EP with `--parallelism.expert_parallel_degree=N`.

Valid EP values must divide `num_experts`. At 24 XPU tiles:
- 8 experts: EP = 2, 4, 8
- 24 experts: EP = 2, 3, 4, 6, 8, 12, 24
- 36 experts: EP = 2, 3, 4, 6, 12

### Special variants

- `moe_10b_2b_sdpa` uses SDPA instead of FlexAttention.
- `moe_*_from_json` applies JSON overrides from the `TT_CONFIG_JSON` environment variable.

## Quick Start

```bash
# Basic training
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
  --module ezpz.moe --config moe_7b --checkpoint.no-enable --training.steps 100

# With EP
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
  --module ezpz.moe --config moe_7b_ep --checkpoint.no-enable --training.steps 100

# LR finder
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
  --module ezpz.moe --config moe_7b --checkpoint.no-enable \
  --lr_finder.enable --lr_finder.init_lr 1e-6 --lr_finder.max_lr 1.0
```

## JSON Config Overrides

Set `TT_CONFIG_JSON` to a JSON file for environment-driven overrides.

### Aurora (2 nodes, 24 XPUs, EP=12)

Aurora 2-node JSON overrides live in `moe_runs/aurora_2nodes_ep12/`.
The 128-node JSON lives in `moe_runs/aurora_128nodes_ep12/`.

```bash
# 1. Smoke test (40 steps, seq_len=1024)
TT_CONFIG_JSON=torchtitan/experiments/ezpz/moe_runs/aurora_2nodes_ep12/deepseek_v3_10b2b_ep12_2nodes_smoke.json \
  ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.moe --config moe_10b_2b_from_json --checkpoint.no_enable

# 2. Baseline (1000 steps, seq_len=4096)
TT_CONFIG_JSON=torchtitan/experiments/ezpz/moe_runs/aurora_2nodes_ep12/deepseek_v3_10b2b_ep12_2nodes.json \
  ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.moe --config moe_10b_2b_from_json --checkpoint.no_enable

# 3. Throughput test (seq_len=4096, gc_freq=200)
TT_CONFIG_JSON=torchtitan/experiments/ezpz/moe_runs/aurora_2nodes_ep12/deepseek_v3_10b2b_ep12_2nodes_4096_perf.json \
  ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.moe --config moe_10b_2b_from_json --checkpoint.no_enable

# 4. Prod sim - no AC (local_batch=2)
TT_CONFIG_JSON=torchtitan/experiments/ezpz/moe_runs/aurora_2nodes_ep12/deepseek_v3_10b2b_ep12_2nodes_4096_prod_sim.json \
  ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.moe --config moe_10b_2b_from_json --checkpoint.no_enable

# 5. Prod sim + selective AC op-level (local_batch=2)
TT_CONFIG_JSON=torchtitan/experiments/ezpz/moe_runs/aurora_2nodes_ep12/deepseek_v3_10b2b_ep12_2nodes_4096_prod_sim_ac.json \
  ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.moe --config moe_10b_2b_from_json --checkpoint.no_enable

# 6. Selective AC op-level, higher batch (local_batch=3)
TT_CONFIG_JSON=torchtitan/experiments/ezpz/moe_runs/aurora_2nodes_ep12/deepseek_v3_10b2b_ep12_2nodes_4096_prod_sim_ac_lb3.json \
  ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.moe --config moe_10b_2b_from_json --checkpoint.no_enable

# 7. Full AC, higher batch (local_batch=3)
TT_CONFIG_JSON=torchtitan/experiments/ezpz/moe_runs/aurora_2nodes_ep12/deepseek_v3_10b2b_ep12_2nodes_4096_prod_sim_ac_full_lb3.json \
  ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.moe --config moe_10b_2b_from_json --checkpoint.no_enable

# 8. Layer-1 only AC, higher batch (local_batch=3)
TT_CONFIG_JSON=torchtitan/experiments/ezpz/moe_runs/aurora_2nodes_ep12/deepseek_v3_10b2b_ep12_2nodes_4096_prod_sim_ac_layer1_lb3.json \
  ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.moe --config moe_10b_2b_from_json --checkpoint.no_enable

# 9. Selective AC + flex attention + compile model+loss (local_batch=2)
TT_CONFIG_JSON=torchtitan/experiments/ezpz/moe_runs/aurora_2nodes_ep12/deepseek_v3_10b2b_ep12_2nodes_4096_prod_sim_ac_lb2_flex_compile_model.json \
  ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.moe --config moe_10b_2b_from_json --checkpoint.no_enable

# 10. 128-node scale run (1536 XPUs)
TT_CONFIG_JSON=torchtitan/experiments/ezpz/moe_runs/aurora_128nodes_ep12/deepseek_v3_10b2b_ep12_128nodes_4096_prod_sim_ac_lb2_compile_ffn.json \
  ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.moe --config moe_10b_2b_from_json --checkpoint.no_enable
```

### Polaris (2 nodes, 8 GPUs)

Polaris JSON overrides live in `moe_runs/polaris_2nodes/`.

```bash
# 1. Smoke test (40 steps, seq_len=1024)
TT_CONFIG_JSON=torchtitan/experiments/ezpz/moe_runs/polaris_2nodes/deepseek_v3_10b2b_polaris_2nodes_smoke.json \
  ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.moe --config moe_10b_2b_from_json --checkpoint.no_enable

# 2. Baseline (1000 steps, seq_len=4096)
TT_CONFIG_JSON=torchtitan/experiments/ezpz/moe_runs/polaris_2nodes/deepseek_v3_10b2b_polaris_2nodes.json \
  ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.moe --config moe_10b_2b_from_json --checkpoint.no_enable

# 3. Throughput test (seq_len=4096, gc_freq=200)
TT_CONFIG_JSON=torchtitan/experiments/ezpz/moe_runs/polaris_2nodes/deepseek_v3_10b2b_polaris_2nodes_4096_perf.json \
  ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.moe --config moe_10b_2b_from_json --checkpoint.no_enable

# 4. Prod sim - no AC (local_batch=2)
TT_CONFIG_JSON=torchtitan/experiments/ezpz/moe_runs/polaris_2nodes/deepseek_v3_10b2b_polaris_2nodes_4096_prod_sim.json \
  ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.moe --config moe_10b_2b_from_json --checkpoint.no_enable

# 5. Prod sim + selective AC op-level (local_batch=2)
TT_CONFIG_JSON=torchtitan/experiments/ezpz/moe_runs/polaris_2nodes/deepseek_v3_10b2b_polaris_2nodes_4096_prod_sim_ac.json \
  ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.moe --config moe_10b_2b_from_json --checkpoint.no_enable

# 6. Selective AC op-level, higher batch (local_batch=3)
TT_CONFIG_JSON=torchtitan/experiments/ezpz/moe_runs/polaris_2nodes/deepseek_v3_10b2b_polaris_2nodes_4096_prod_sim_ac_lb3.json \
  ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.moe --config moe_10b_2b_from_json --checkpoint.no_enable

# 7. Full AC, higher batch (local_batch=3)
TT_CONFIG_JSON=torchtitan/experiments/ezpz/moe_runs/polaris_2nodes/deepseek_v3_10b2b_polaris_2nodes_4096_prod_sim_ac_full_lb3.json \
  ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.moe --config moe_10b_2b_from_json --checkpoint.no_enable

# 8. Layer-1 only AC, higher batch (local_batch=3)
TT_CONFIG_JSON=torchtitan/experiments/ezpz/moe_runs/polaris_2nodes/deepseek_v3_10b2b_polaris_2nodes_4096_prod_sim_ac_layer1_lb3.json \
  ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.moe --config moe_10b_2b_from_json --checkpoint.no_enable

# 9. Selective AC + compile preset (local_batch=2)
TT_CONFIG_JSON=torchtitan/experiments/ezpz/moe_runs/polaris_2nodes/deepseek_v3_10b2b_polaris_2nodes_4096_prod_sim_ac_lb2_compile_ffn.json \
  ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.moe --config moe_10b_2b_from_json --checkpoint.no_enable
```

Pre-configured JSON overrides now live in cluster- and run-specific subdirectories under `moe_runs/`.

- `*_smoke.json` for quick smoke tests
- `*_perf.json` for throughput measurement
- `*_prod_sim*.json` for production-simulation variants
- `*_128nodes*.json` for larger Aurora runs

See `moe_runs/README.md` for the full list.

### Writing Custom JSON Overrides

Any trainer config field can be overridden. Nest by config section:

```json
{
  "training": {
    "local_batch_size": 4,
    "seq_len": 2048,
    "steps": 500
  },
  "parallelism": {
    "data_parallel_replicate_degree": 4,
    "data_parallel_shard_degree": -1
  },
  "activation_checkpoint": {
    "mode": "selective",
    "selective_ac_option": "op"
  }
}
```

Unknown fields raise `KeyError`, and type mismatches raise `TypeError`.

## Using CLI Flags Directly (No JSON)

Use `--config moe_10b_2b` and pass overrides as CLI flags.
`$(ezpz_get_machine_name)` auto-selects the local machine data list.

For Aurora EP=12 topology, add:
`--parallelism.expert_parallel_degree 12 --parallelism.data_parallel_shard_degree 12`

```bash
# 1. Smoke test (40 steps, seq_len=1024)
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
  --module ezpz.moe --config moe_10b_2b --checkpoint.no_enable \
  --training.local_batch_size 1 --training.seq_len 1024 --training.steps 40 \
  --metrics.log_freq 1 \
  --dataloader.dataset blendcorpus \
  --dataloader.dataset_path torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)/books.txt

# 2. Baseline (1000 steps, seq_len=4096)
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
  --module ezpz.moe --config moe_10b_2b --checkpoint.no_enable \
  --training.local_batch_size 1 --training.seq_len 4096 --training.steps 1000 \
  --metrics.log_freq 1 \
  --dataloader.dataset blendcorpus \
  --dataloader.dataset_path torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)/books.txt

# 3. Throughput test (seq_len=4096, gc_freq=200)
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
  --module ezpz.moe --config moe_10b_2b --checkpoint.no_enable \
  --training.local_batch_size 1 --training.seq_len 4096 --training.steps 1000 \
  --training.gc_freq 200 --metrics.log_freq 10 \
  --dataloader.dataset blendcorpus \
  --dataloader.dataset_path torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)/books.txt

# 4. Prod sim - no AC (local_batch=2)
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
  --module ezpz.moe --config moe_10b_2b --checkpoint.no_enable \
  --training.local_batch_size 2 --training.seq_len 4096 --training.steps 1000 \
  --training.gc_freq 1000 --metrics.log_freq 1 \
  --activation_checkpoint.mode none \
  --dataloader.dataset blendcorpus --dataloader.num_workers 2 \
  --dataloader.persistent_workers --dataloader.prefetch_factor 2 \
  --dataloader.dataset_path torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)/books.txt

# 5. Prod sim + selective AC op-level (local_batch=2)
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
  --module ezpz.moe --config moe_10b_2b --checkpoint.no_enable \
  --training.local_batch_size 2 --training.seq_len 4096 --training.steps 1000 \
  --training.gc_freq 1000 --metrics.log_freq 1 \
  --activation_checkpoint.mode selective --activation_checkpoint.selective_ac_option op \
  --dataloader.dataset blendcorpus --dataloader.num_workers 2 \
  --dataloader.persistent_workers --dataloader.prefetch_factor 2 \
  --dataloader.dataset_path torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)/books.txt

# 6. Selective AC op-level, higher batch (local_batch=3)
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
  --module ezpz.moe --config moe_10b_2b --checkpoint.no_enable \
  --training.local_batch_size 3 --training.seq_len 4096 --training.steps 1000 \
  --training.gc_freq 1000 --metrics.log_freq 1 \
  --activation_checkpoint.mode selective --activation_checkpoint.selective_ac_option op \
  --dataloader.dataset blendcorpus --dataloader.num_workers 2 \
  --dataloader.persistent_workers --dataloader.prefetch_factor 2 \
  --dataloader.dataset_path torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)/books.txt

# 7. Full AC, higher batch (local_batch=3)
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
  --module ezpz.moe --config moe_10b_2b --checkpoint.no_enable \
  --training.local_batch_size 3 --training.seq_len 4096 --training.steps 1000 \
  --training.gc_freq 1000 --metrics.log_freq 1 \
  --activation_checkpoint.mode full \
  --activation_checkpoint.no_preserve_rng_state \
  --activation_checkpoint.determinism_check none \
  --dataloader.dataset blendcorpus --dataloader.num_workers 2 \
  --dataloader.persistent_workers --dataloader.prefetch_factor 2 \
  --dataloader.dataset_path torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)/books.txt

# 8. Layer-1 only AC, higher batch (local_batch=3)
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
  --module ezpz.moe --config moe_10b_2b --checkpoint.no_enable \
  --training.local_batch_size 3 --training.seq_len 4096 --training.steps 1000 \
  --training.gc_freq 1000 --metrics.log_freq 1 \
  --activation_checkpoint.mode selective --activation_checkpoint.selective_ac_option 1 \
  --activation_checkpoint.no_preserve_rng_state \
  --activation_checkpoint.determinism_check none \
  --dataloader.dataset blendcorpus --dataloader.num_workers 2 \
  --dataloader.persistent_workers --dataloader.prefetch_factor 2 \
  --dataloader.dataset_path torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)/books.txt

# 9. Selective AC + compile model+loss (local_batch=2)
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
  --module ezpz.moe --config moe_10b_2b --checkpoint.no_enable \
  --training.local_batch_size 2 --training.seq_len 4096 --training.steps 1000 \
  --training.gc_freq 1000 --metrics.log_freq 1 \
  --activation_checkpoint.mode selective --activation_checkpoint.selective_ac_option op \
  --compile.enable --compile.components model loss --compile.backend inductor \
  --dataloader.dataset blendcorpus --dataloader.num_workers 2 \
  --dataloader.persistent_workers --dataloader.prefetch_factor 2 \
  --dataloader.dataset_path torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)/books.txt
```

## Known Limitations

- **AC incompatible with MoE 7B+**: routing is non-deterministic, causing shape mismatches on AC recomputation. Use `--activation_checkpoint.mode none`.
- **FlexAttention on XPU**: `torch.autocast(dtype=fp32)` in the MoE router can crash on XPU. Use `_sdpa` variants if needed.
- **10b_2b_sdpa OOMs at seq_len=8192** on 2 nodes with EP=1. Use EP=2+ or reduce sequence length.
- **Muon is very slow on small MoE**: Newton-Schulz overhead dominates for debugmodel and 500M.

## Experiment Reports

See [experiments/moe/README.md](experiments/moe/README.md) for benchmark reports,
and [experiments/lr-finder/moe/](experiments/lr-finder/moe/) for LR finder results.

## LR Finder Summary (Sunspot, torch 2.13)

| Config | AdamW | Muon | SophiaG |
|--------|-------|------|---------|
| debugmodel | stable | slow | stable |
| 500M | stable | slow | stable |
| 2B | stable | stable | stable |
| 4B | stable | stable | 1 NaN (high LR) |
| 7B | stable | stable | 3-5 NaN (high LR) |
| 10b_2b_sdpa (EP=2) | stable | stable | - |

All optimizers were numerically stable on MoE models with dim <= 2048, below the
bf16 overflow threshold. SophiaG showed mild instability only at high learning rates.
