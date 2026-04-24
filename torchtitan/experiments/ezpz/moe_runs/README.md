# DeepSeek V3 MoE Runs (Aurora / ezpz)

This folder contains a starter setup for launching DeepSeek V3-style MoE runs with:

- `expert_parallel_degree=12`
- 2-node Aurora runs (`24` tiles total)
- no TP/PP/CP sharding (`tp=pp=cp=1`)
- one dense layer at the beginning (`n_dense_layers=1`)

The config used by the launcher is loaded from `TT_CONFIG_JSON` and overlaid onto
the `deepseek_v3_10b_2b_ep12` base config.

By default, the launcher uses `assets/hf/gemma-7b` when available, and
`deepseek_v3` model vocab size is automatically synced from the tokenizer files.

## Layout

- `launch_deepseek_v3_moe_ep12.sh` and `run_torchtitan_ezpz_train.sh`: shared runtime launchers
- `aurora_2nodes_ep12/`: 2-node Aurora configs plus plain-training and profiling PBS submitters
- `aurora_2nodes_ep12_throughput_2h/`: dedicated 2-node Aurora throughput run with its own config and PBS submitter
- `aurora_128nodes_ep12/`: 128-node Aurora config plus its PBS submitter
- `aurora_256nodes_ep12_muon_200k/`: 256-node Aurora config plus its PBS submitter
- `polaris_2nodes/`: Polaris configs

Each launched run now gets its own directory under `outputs/moe_runs/<run_slug>/`
with:

- `configs/`: copied input JSON plus the resolved effective config
- `metadata/`: copied launcher and submitter scripts
- `logs/`: batch log redirection
- `checkpoints/`: training checkpoints

The launcher now defaults to the smoke JSON, which uses:

- `seq_len=1024`
- `global_batch_size=24`
- `steps=40`
- `attn_backend=sdpa`
- `compile.enable=false`
- checkpoint save every 5 steps (`enable_first_step_checkpoint=true`)

## Run

From repository root:

```bash
torchtitan/experiments/ezpz/moe_runs/launch_deepseek_v3_moe_ep12.sh my_wandb_run_name
```

If tokenizer assets are not present yet:

```bash
python3 scripts/download_hf_assets.py --repo_id google/gemma-7b --assets tokenizer
```

You can also force a specific tokenizer/assets path:

```bash
HF_ASSETS_PATH=/abs/path/to/assets/hf/gemma-7b \
torchtitan/experiments/ezpz/moe_runs/launch_deepseek_v3_moe_ep12.sh my_wandb_run_name
```

Optional additional TorchTitan args can be appended:

```bash
torchtitan/experiments/ezpz/moe_runs/launch_deepseek_v3_moe_ep12.sh my_wandb_run_name \
  --training.steps 200 \
  --checkpoint.interval 20
```

To use a different JSON file:

```bash
TT_CONFIG_JSON=/abs/path/to/overrides.json \
torchtitan/experiments/ezpz/moe_runs/launch_deepseek_v3_moe_ep12.sh my_wandb_run_name
```

To use the non-smoke 4096-seq config:

```bash
TT_CONFIG_JSON=torchtitan/experiments/ezpz/moe_runs/aurora_2nodes_ep12/deepseek_v3_10b2b_ep12_2nodes.json \
torchtitan/experiments/ezpz/moe_runs/launch_deepseek_v3_moe_ep12.sh my_wandb_run_name
```

To use the 4096-seq throughput-oriented config (no torch.compile, async checkpoint):

```bash
TT_CONFIG_JSON=torchtitan/experiments/ezpz/moe_runs/aurora_2nodes_ep12/deepseek_v3_10b2b_ep12_2nodes_4096_perf.json \
bash torchtitan/experiments/ezpz/moe_runs/launch_deepseek_v3_moe_ep12.sh my_wandb_run_name
```
