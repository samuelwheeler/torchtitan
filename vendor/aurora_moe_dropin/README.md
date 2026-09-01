# Aurora MoE drop-in

This directory is a standalone PyTorch package for the validated source-major,
expert-parallel MoE implementation developed for Aurora.  It includes the
Python runtime, all SYCL sources, the native-oneCCL bridge, a prebuild helper,
and a PyTorch `nn.Module` facade.

The layer uses exact observed routing counts: it does not use capacity factors,
route clipping, or padded expert activations.  Expert count, top-k, hidden
dimension, expert hidden dimension, token count, and DP/EP sizes are runtime
parameters.

## Important status

The best sustained measurement in this project was 63.790 ms / 66.66 logical
TF/s/tile on DP=2, EP=12.  That particular profile intentionally omitted
router-score gradients, so it is supplied only as an explicit benchmark mode.
It is not the default and is not appropriate for ordinary router training.

`configure_native_runtime()` defaults to the exact router-gradient path.  Use
the production default first, then validate numerical equivalence with the
target model before enabling optional scheduling settings.

## Install and prebuild

Run these commands on Aurora after loading its PyTorch framework module:

```bash
module load frameworks
cd aurora_moe_dropin
python3 -m pip install -e .
```

Compile the extensions once inside an XPU allocation, not concurrently from
all ranks in a training job.  The generated environment file prevents each
rank from attempting a JIT build:

```bash
python3 -m aurora_moe.build \
  --build-dir /flare/AuroraGPT/$USER/aurora_moe_build \
  --env-file /flare/AuroraGPT/$USER/aurora_moe_build/aurora_moe.env
source /flare/AuroraGPT/$USER/aurora_moe_build/aurora_moe.env
```

Without `--env-file`, the command prints equivalent exports.  The helper's
sources are in `src/aurora_moe/_kernels/csrc/`; no binary is shipped in this
folder.

## Use as a PyTorch layer

Configure the runtime before initializing `torch.distributed`.  The native
oneCCL path requires `CCL_OP_SYNC=0` at process-group creation time.

```python
import torch
import torch.distributed as dist
from aurora_moe import AuroraMoE, configure_native_runtime, create_dp_ep_groups

configure_native_runtime(router_grad=True, phase_shared=False)
dist.init_process_group("xccl")
groups = create_dp_ep_groups(dp_size=2, ep_size=12)

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

x = torch.randn(16_384, 2048, device="xpu", dtype=torch.bfloat16, requires_grad=True)
y = moe(x)
loss = y.float().square().mean()
loss.backward()
moe.finalize_gradients()
```

`AuroraMoE` owns the correct DP wrappers: router/shared weights reduce across
the full DP×EP group and sharded expert weights reduce across the same-EP
DP group.  Do not wrap the whole module in world-size DDP.  For the simple DP
x EP mapping used here, rank numbering is `rank = dp_rank * ep_size + ep_rank`.
`create_dp_ep_groups` constructs the required groups in the same fixed order
on every rank.

The built-in router matches the implementation benchmarked here: a bias-free
linear projection, top-k selection, and sigmoid of selected logits.  Its
weight is exposed as `moe.router_weight`; local expert weights are exposed as
`experts_up`, `experts_gate`, and `experts_down`.

`examples/train_step.py` is the same setup as a runnable one-step distributed
training example.  Set `AURORA_MOE_DP_SIZE` and `AURORA_MOE_EP_SIZE` to match
the `mpiexec` placement before launching it.

## Profiles

The exact training profile is the default:

```python
configure_native_runtime(router_grad=True, phase_shared=False)
```

The historical throughput profile is explicitly router-gradient-free:

```python
configure_native_runtime(
    router_grad=False,
    router_free_two_phase=True,
    phase_shared=True,
    zero_wait_a4_submit=True,
)
```

That profile returns a zero router-score gradient by design.  It is useful for
kernel studies where routing is outside the differentiated graph, but it must
not be used as a normal training replacement.

## Included components

- `src/aurora_moe/layer.py`: PyTorch `AuroraMoE` facade.
- `src/aurora_moe/distributed.py`: DP x EP group construction.
- `src/aurora_moe/runtime.py`: explicit runtime profiles.
- `src/aurora_moe/build.py`: one-tile extension prebuild command.
- `src/aurora_moe/_core.py`: the exact routed MoE autograd runtime.
- `src/aurora_moe/_kernels`: vendorized SYCL/oneMKL/oneCCL implementation and
  its complete C++/SYCL source set.

## Test

The included CPU test validates the public layer contract using the reference
expert loop:

```bash
PYTHONPATH=src pytest -q tests
```

Run an XPU correctness comparison against the target model before using a new
shape, routing function, or distributed process layout in production.
