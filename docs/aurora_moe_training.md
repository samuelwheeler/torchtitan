# MoE training on Aurora

This tree contains the source used for the 256-node Aurora run
`CODEX_AGPT_MOE256_SAFE_20260810T164655Z_7dee5484`. Jobs `8747196`,
`8747197`, and `8760880` used the same source archive and advanced the same
training run to checkpoints 9,000, 18,000, and 27,000. The final segment ran
at about 530 tokens/s/tile with finite loss and gradients.

## Model and run

The `AGPT_2B_50K_MOE_sdpa_aurora_full_sonic` flavor has:

- 12,293,801,984 total parameters and 2,016,708,608 active parameters;
- 24 layers, model dimension 2,048, 16 query heads, and 4 KV heads;
- 36 routed experts, top-3 routing, and two shared-expert equivalents;
- expert hidden dimension 2,112 and vocabulary size 50,304.

The production configuration uses 256 nodes and all 3,072 Aurora tiles. It
uses EP=12, a 256-way replicated DP dimension, a 12-way sharded DP dimension,
sequence length 2,048, and global batch size 3,072 sequences. This is
6,291,456 tokens per step. Parameters are BF16 with FP32 reduction and master
weights. The optimizer is fused AdamW with a peak learning rate of 2.2e-4.

The exact training configuration is
`torchtitan/experiments/ezpz/moe_runs/agpt_moe_2b_50k_ep12_dp256_50k.json`.
The PBS launcher is
`torchtitan/experiments/ezpz/submit/aurora/submit_agpt_dense_moe_256n_50k.pbs`.

## Prepare the source and kernels

Use an Aurora PyTorch environment with XPU and SYCL extension support. The
validated run loaded oneAPI 2025.3.1 and used `CCL_OP_SYNC=1` with XCCL expert
transport.

```bash
module load oneapi/release/2025.3.1 hdf5 pti-gpu mpifileutils
export PYTHONPATH="$PWD:$PWD/vendor/aurora_moe_dropin/src"

python3 -m aurora_moe.build \
  --torchtitan-full-sonic \
  --build-dir "$PWD/outputs/aurora_moe_kernels"
```

Build the extensions once on one allocated XPU. Do not let thousands of
training ranks race through JIT compilation. The launcher accepts the build
directory through `TT_MOE_KERNEL_CACHE` and a relocatable packed environment
through `TT_VENV_TAR`.

Create an immutable source archive after committing the tree:

```bash
eval "$(python3 \
  torchtitan/experiments/ezpz/submit/aurora/freeze_matched_dense_moe_2n_1100.py)"
```

The command sets `PAIR_ID`, `TT_INPUT_DIR`, and
`TT_SOURCE_ARCHIVE_SHA256`. The archive includes the vendored Aurora MoE
Python and SYCL sources and records a SHA-256 manifest.

## Submit the 256-node run

The dataset list, tokenizer, validated BlendCorpus cache, and packed Python
environment are external artifacts and are not committed. The launcher keeps
the paths from the recorded run as defaults. Override them when using another
checkout, and provide hashes for the tokenizer and dataset list.

```bash
qsub -v \
TT_ROOT="$PWD",\
TT_INPUT_DIR="$TT_INPUT_DIR",\
TT_SOURCE_ARCHIVE_SHA256="$TT_SOURCE_ARCHIVE_SHA256",\
TT_SNAPSHOT_ID="$PAIR_ID",\
TT_TRAIN_CASE=moe,\
TT_VENV_TAR=/absolute/path/to/venv.tar.gz,\
TT_MOE_KERNEL_CACHE="$PWD/outputs/aurora_moe_kernels",\
TT_DATA_CACHE_PATH=/absolute/path/to/blendcorpus-cache,\
TT_DATA_LIST=/absolute/path/to/dolma_v1_7_llama2_live.txt,\
TT_DATA_LIST_SHA256=sha256-of-data-list,\
TT_TOKENIZER=/absolute/path/to/tokenizer.model,\
TT_TOKENIZER_SHA256=sha256-of-tokenizer \
torchtitan/experiments/ezpz/submit/aurora/submit_agpt_dense_moe_256n_50k.pbs
```

The job logs to W&B, checkpoints every 500 steps, retains the newest two
checkpoints, and is resumable by submitting the same snapshot and run ID
again. W&B credentials must be available to the batch job.

## Verification

The architecture and parameter-count contract is covered by
`tests/unit_tests/test_agpt_moe_config.py`. Backend equivalence and routing
count behavior are covered by `test_moe_expert_backends.py` and
`test_moe_routing_counts.py`. The production launcher validates source,
tokenizer, and dataset-list hashes before starting distributed training.
