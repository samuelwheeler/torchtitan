# TorchTitan + 🍋 `ezpz`

> [!NOTE]
> These instructions assume we are using the fork `saforem2/torchtitan`, on
> the branch `ezpz`, i.e.:  
> [saforem2/torchtitan@ezpz](https://github.com/saforem2/torchtitan/tree/ezpz)

1. Submit job:

   - Aurora:

     ```bash
     qsub -q prod -A <project> -l walltime=06:00:00,filesystems=flare:home -l select=2 -I
     ```

   - Polaris:

     ```bash
     qsub -q preemptable -A <project> -l walltime=06:00:00,filesystems=eagle:home -l select=2 -I
     ```

1. Clone TorchTitan from [saforem2/torchtitan@ezpz](https://github.com/saforem2/torchtitan/blob/ezpz):

   ```bash
   git clone https://github.com/saforem2/torchtitan --branch ezpz
   cd torchtitan
   ```

1. Setup environment:

   ```bash
   source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_env
   ```

1. Install Dependencies

   ```bash
   uv pip install "git+https://github.com/saforem2/ezpz"
   uv pip install "git+https://github.com/zhenghh04/blendcorpus"
   uv pip install tensorboard tyro
   ```

   - Polaris:

     ```bash
     uv pip install tf-keras
     # flash-attn
     CC=$(which gcc) CXX=$(which g++) uv pip install --upgrade flash-attn --no-build-isolation --no-cache --link-mode=copy
     ```

1. Download tokenizers:

   ```bash
   # 2B model
   python3 scripts/download_hf_assets.py --repo_id google/gemma-7b --assets tokenizer
   # 7B model
   python3 scripts/download_hf_assets.py --repo_id meta-llama/llama-2-7b-hf --assets tokenizer
   ```

1. Launch Training

   - AuroraGPT-2B:

     ```bash
     MODEL=2b
     DFL=torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)/books.txt
     ezpz launch python3 -m torchtitan.experiments.ezpz.train \
         --module ezpz.agpt \
         --config "ezpz_agpt_${MODEL}" \
         --training.dataset_path "${DFL}" \
         --debug.print_config
     ```

   - AuroraGPT-7B:

     ```bash
     MODEL=7b
     DFL=torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)/books.txt
     ezpz launch python3 -m torchtitan.experiments.ezpz.train \
         --module ezpz.agpt \
         --config "ezpz_agpt_${MODEL}" \
         --training.dataset_path "${DFL}" \
         --debug.print_config
     ```

> [!TIP]
>   - To suppress the `UserWarning: Torchinductor` error seen when using
>     `--compile.enable` on Aurora, you can export:
>
>     ```bash
>     export SYCL_DISABLE_FSYCL_SYCLHPP_WARNING=1
>     ```


## Launching with `run_train.sh`

- [run_train.sh](torchtitan/experiments/ezpz/run_train.sh)

    ```bash
    # AuroraGPT-2B model:
    MODEL=2b bash torchtitan/experiments/ezpz/run_train.sh
    # or, AuroraGPT-7B model:
    MODEL=7b bash torchtitan/experiments/ezpz/run_train.sh
    # or, to specify the data-file-list:
    MODEL=7b \
        DFL=torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)/books.txt \
        bash torchtitan/experiments/ezpz/run_train.sh
    ```

## MoE Training with JSON Override Configs

The JSON override config system lets you separate model architecture from training
hyperparameters.
Set `TT_CONFIG_JSON` to a JSON file, then use a `_from_json` config:

```bash
TT_CONFIG_JSON=<path/to/config.json> \
  ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.moe --config moe_10b_2b_from_json
```

Available `--config` values for MoE:

| Config           | Model                                    | Description                   |
| ---------------- | ---------------------------------------- | ----------------------------- |
| `moe_debugmodel` | debugmodel (256d, 6L, 8 experts)         | Tiny model for fast iteration |
| `moe_small`      | small (2048d, 24L, 64 experts)           | Small model                   |
| `moe_10b_2b`     | 10B/2B (2048d, 27L, 36 experts, top_k=3) | 10B total / 2B active         |
| `moe_16b`        | 16B (2048d, 27L, 64 experts)             | Full 16B                      |
| `moe_671b`       | 671B (7168d, 61L, 256 experts)           | Full 671B                     |

Each has a `_from_json` variant (e.g. `moe_10b_2b_from_json`) that applies
overrides from `TT_CONFIG_JSON`.

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

# 4. Prod sim — no AC (local_batch=2)
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

# 4. Prod sim — no AC (local_batch=2)
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

Unknown fields raise `KeyError`, type mismatches raise `TypeError`.

## References

- 🍋 `ezpz`:
  - Documentation: [ezpz.cool](https://ezpz.cool)
  - GitHub: [saforem2/ezpz](https://github.com/saforem2/ezpz)
