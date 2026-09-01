# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Apply PT-D parallelisms + AC + compile + FSDP to the ezpz/moe model.

This is the moe mirror of `torchtitan.models.deepseek_v3.parallelize`. It
uses the new config-based DTensor sharding API for the non-MoE path: TP
on attention/norms/dense-FFN is applied via `model.parallelize(tp_mesh)`,
which reads `sharding_config` declarations filled in by
`moeModel.Config.update_from_config`.

MoE expert/router TP and EP are still applied at parallelize-time by
`apply_moe_ep_tp` — that mirrors upstream deepseek_v3, where
`set_deepseek_v3_sharding_config` also leaves the MoE block alone.

Differences vs upstream `parallelize_deepseekv3`:

- `disable_fsdp_gradient_division` enables
  `set_force_sum_reduction_for_comms(True)` for non-NCCL backends
  (CCL on XPU). Upstream's version only sets the divide factor.
- `apply_compile`: upstream uses fullgraph=True via `apply_compile_sparse`,
  which fails on XPU (MoE routing's dynamic shapes). We compile each
  block with `block.compile(backend=...)` (no fullgraph).
- `apply_fsdp` is inlined locally to avoid importing
  `ShardPlacementResult`, which doesn't exist in Aurora's PyTorch. Also
  adds a Shard(0) fallback when expert hidden dim isn't divisible by the
  FSDP world size.
"""

import os
from typing import Any

import ezpz
import ezpz.distributed
import torch
import torch.distributed
import torch.nn as nn
from ezpz.models import summarize_model
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import Partial, Replicate, Shard
from torch.distributed.tensor.parallel import (
    parallelize_module,
    PrepareModuleInputOutput,
    RowwiseParallel,
)

from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.context_parallel import apply_cp_to_forward
from torchtitan.distributed.expert_parallel import (
    ExpertParallel,
    TensorParallel,
)
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.distributed.tensor_parallel import (
    ColwiseParallelWithGradPlacement,
    maybe_enable_async_tp,
    NoParallel,
)
from torchtitan.experiments.ezpz.moe import moeModel
from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher
from torchtitan.tools.logging import logger


def _use_ep_replicate_module(
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
) -> bool:
    """Select node-local EP plus parameter-specific replicated DP.

    ``dp_shard`` supplies the 12 ranks that are reshaped into the EP mesh; it
    does not imply parameter sharding in this mode.  Shared parameters reduce
    over the full batch mesh, while each EP expert shard reduces only over the
    same local-tile coordinate on other nodes.
    """

    if not parallelism.enable_data_parallel_replicate_module:
        return False
    invalid = {
        "tensor_parallel": parallel_dims.tp,
        "pipeline_parallel": parallel_dims.pp,
        "context_parallel": parallel_dims.cp,
    }
    invalid = {name: degree for name, degree in invalid.items() if degree != 1}
    if invalid:
        raise ValueError(
            "MoE EP ReplicateModule initially requires TP=PP=CP=1, got "
            f"{invalid}"
        )
    if parallel_dims.ep <= 1 or parallel_dims.dp_shard != parallel_dims.ep:
        raise ValueError(
            "MoE EP ReplicateModule requires expert_parallel_degree > 1 and "
            "data_parallel_shard_degree == expert_parallel_degree"
        )
    if parallel_dims.dp_shard * parallel_dims.tp // parallel_dims.ep != 1:
        raise ValueError("MoE EP ReplicateModule requires efsdp degree one")
    if training.enable_cpu_offload:
        raise ValueError("MoE EP ReplicateModule does not yet support CPU offload")
    if parallelism.enable_fsdp_async_all_reduce:
        raise ValueError(
            "MoE EP ReplicateModule does not use the FSDP async-all-reduce path"
        )
    if (
        parallelism.pipeline_parallel_fsdp_overlap
        or parallelism.pipeline_parallel_fsdp_overlap_policy != "bulk"
    ):
        raise ValueError(
            "MoE EP ReplicateModule requires the bulk pipeline overlap policy"
        )
    return True


def disable_fsdp_gradient_division(model: nn.Module) -> None:
    """Disable FSDP's automatic gradient division and (on XPU/CCL) force
    sum reduction for cross-rank gradient comms.

    On NCCL the default reduce-mean works correctly. On CCL (XPU) we need
    SUM and divide ourselves to avoid losing precision.
    """
    force_sum_reduction = False
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        backend = ezpz.distributed.get_torch_backend() or str(
            torch.distributed.get_backend()
        )
        if backend and "nccl" not in str(backend).lower():
            force_sum_reduction = True

    fsdp_modules_updated = 0
    for module in model.modules():
        # Be resilient to FSDPModule class location changes across PyTorch
        # releases by going through the public method.
        set_divide_factor = getattr(module, "set_gradient_divide_factor", None)
        if callable(set_divide_factor):
            set_divide_factor(1.0)
            fsdp_modules_updated += 1
            if force_sum_reduction:
                set_force_sum = getattr(
                    module, "set_force_sum_reduction_for_comms", None
                )
                if callable(set_force_sum):
                    set_force_sum(True)

    logger.info(
        "Configured FSDP gradient division for %d modules (force_sum_reduction=%s)",
        fsdp_modules_updated,
        force_sum_reduction,
    )


def parallelize_moe(
    model: moeModel,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    """Apply CP + TP + EP + AC + compile + FSDP to the moe model.

    The passed-in model preferably should be on meta device. Otherwise
    the model must fit on GPU or CPU memory.
    """
    assert (
        training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    use_ep_replicate_module = _use_ep_replicate_module(
        parallel_dims, training, parallelism
    )
    aurora_full_backends = {
        getattr(
            getattr(getattr(block, "moe", None), "experts", None),
            "compute_backend",
            None,
        )
        for block in model.layers.values()
    } & {"aurora_full_loop", "aurora_full_sonic"}
    uses_aurora_full_runtime = bool(aurora_full_backends)
    if uses_aurora_full_runtime:
        if parallel_dims.tp_enabled:
            raise NotImplementedError(
                "Aurora full MoE runtime does not yet support tensor parallelism"
            )
        if not parallel_dims.ep_enabled:
            raise ValueError("Aurora full MoE runtime requires expert parallelism")
        if training.mixed_precision_param != "bfloat16":
            raise ValueError("Aurora full MoE runtime requires BF16 parameters")
        for block in model.layers.values():
            if not block.moe_enabled:
                continue
            if block.moe.experts.num_experts % parallel_dims.ep:
                raise ValueError(
                    f"num_experts ({block.moe.experts.num_experts}) must be "
                    f"divisible by expert_parallel_degree ({parallel_dims.ep})"
                )
        if (
            "aurora_full_sonic" in aurora_full_backends
            and os.environ.get("AURORA_MOE_ALLTOALLV") != "1"
        ):
            raise RuntimeError(
                "Aurora full Sonic requires AURORA_MOE_ALLTOALLV=1"
            )

    # CP: wrap inner attention forward BEFORE parallelize() so CP logic
    # runs inside the local_map boundary on local tensors.
    if parallel_dims.cp_enabled:
        if parallel_dims.tp_enabled:
            raise NotImplementedError(
                "Context Parallel with Tensor Parallel is not yet supported "
                "for DeepSeek-V3. "
                "See https://github.com/pytorch/torchtitan/issues/2446"
            )
        apply_cp_to_forward(
            [block.attention.inner_attention for block in model.layers.values()],
            parallel_dims.get_mesh("cp"),
            parallelism.context_parallel_rotate_method,
        )

    # TP via the config-based sharding API. The model's sharding_config
    # declarations were filled in by update_from_config (see model.py).
    # MoE blocks are intentionally not handled here — apply_moe_ep_tp
    # below does that (mirrors upstream deepseek_v3).
    if parallel_dims.tp_enabled:
        tp_mesh = parallel_dims.get_mesh("tp")
        model.parallelize(tp_mesh)
        maybe_enable_async_tp(parallelism, compile_config, tp_mesh)

    # EP/TP for MoE blocks.
    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        apply_moe_ep_tp(
            model,
            tp_mesh=parallel_dims.get_optional_mesh("tp"),
            ep_mesh=parallel_dims.get_optional_mesh("ep"),
            enable_sp=parallelism.enable_sequence_parallel,
        )

    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )

    if ac_config.mode != "none":
        if uses_aurora_full_runtime and ac_config.mode != "selective":
            raise ValueError(
                "Aurora full MoE runtime requires selective attention-only "
                "activation checkpointing"
            )
        apply_ac(
            model,
            ac_config,
            model_compile_enabled=model_compile_enabled,
            base_folder=dump_folder,
            # The optimized routed+shared runtime uses nested autograd to
            # overlap its two backward branches. Wrapping the complete block
            # makes selective AC attempt a second traversal of one cache.
            # Checkpoint attention instead; keep dynamic MoE routing and its
            # nested backward outside the checkpoint region.
            checkpoint_submodule="attention" if uses_aurora_full_runtime else None,
        )

    if model_compile_enabled:
        # Upstream apply_compile_sparse uses fullgraph=True which fails on
        # XPU after 00b7f569 removed maybe_enable_amp — MoE routing's
        # dynamic shapes cause recompilation that fullgraph=True forbids.
        # Apply compile per-block without fullgraph instead.
        torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
        for layer_id, block in model.layers.named_children():
            block.compile(backend=compile_config.backend)
            model.layers.register_module(layer_id, block)

    if use_ep_replicate_module:
        apply_ep_replicate(
            model,
            batch_mesh=parallel_dims.get_mesh("batch"),
            expert_dp_mesh=parallel_dims.get_optional_mesh("dp_replicate"),
            expert_mp_mesh=parallel_dims.get_mesh("efsdp"),
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        )
        logger.info(
            "Applied node-local EP%d plus inter-node ReplicateModule DP%d",
            parallel_dims.ep,
            parallel_dims.dp_replicate,
        )
    else:
        dp_mesh_names = (
            ["dp_replicate", "fsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["fsdp"]
        )
        dp_mesh = parallel_dims.get_mesh(dp_mesh_names)

        edp_mesh = None
        if parallel_dims.ep_enabled:
            edp_mesh_names = (
                ["dp_replicate", "efsdp"]
                if parallel_dims.dp_replicate_enabled
                else ["efsdp"]
            )
            edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)

        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=training.enable_cpu_offload,
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            ep_degree=parallel_dims.ep,
            edp_mesh=edp_mesh,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")

    if training.enable_cpu_offload:
        logger.info("Applied CPU Offloading to the model")

    logger.info(f"\n+{summarize_model(model)}")

    return model


def apply_ep_replicate(
    model: nn.Module,
    *,
    batch_mesh: DeviceMesh,
    expert_dp_mesh: DeviceMesh | None,
    expert_mp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
) -> None:
    """Apply heterogeneous replicated DP to a node-local-EP MoE model.

    Expert weights have already been sharded over EP by ``apply_moe_ep_tp``.
    They reduce only over corresponding EP ranks on other nodes.  All other
    parameters reduce over every independent input-data rank in ``batch``.

    A one-node validation has no expert-DP collective.  A degree-one FSDP2
    wrapper on the real ``efsdp`` mesh still materializes its local expert
    shard in the requested mixed precision.
    """

    from torch.distributed._composable.replicate_with_fsdp import replicate

    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    config = {"mp_policy": mp_policy}

    local_expert_counts = []
    for transformer_block in model.layers.values():
        if not transformer_block.moe_enabled:
            continue
        experts = transformer_block.moe.experts
        expert_param = next(experts.parameters())
        local_expert_counts.append(expert_param.to_local().shape[0])
        if expert_dp_mesh is None:
            fully_shard(experts, mesh=expert_mp_mesh, **config)
        else:
            replicate(experts, mesh=expert_dp_mesh, **config)

    if not local_expert_counts or any(count <= 0 for count in local_expert_counts):
        raise ValueError("EP ReplicateModule found no valid local routed experts")
    if len(set(local_expert_counts)) != 1:
        raise ValueError(
            f"inconsistent local expert counts across layers: {local_expert_counts}"
        )

    shared_config = {"mesh": batch_mesh, **config}
    if model.tok_embeddings is not None:
        replicate(model.tok_embeddings, **shared_config)
    for transformer_block in model.layers.values():
        replicate(transformer_block, **shared_config)
    if model.norm is not None and model.lm_head is not None:
        replicate([model.norm, model.lm_head], **shared_config)
    replicate(model, **shared_config)

    disable_fsdp_gradient_division(model)
    logger.info(
        "EP ReplicateModule topology: batch_size=%d expert_dp_size=%d "
        "expert_mp_size=%d local_experts=%d batch_ranks=%s expert_dp_ranks=%s",
        batch_mesh.size(),
        1 if expert_dp_mesh is None else expert_dp_mesh.size(),
        expert_mp_mesh.size(),
        local_expert_counts[0],
        batch_mesh.mesh.tolist(),
        None if expert_dp_mesh is None else expert_dp_mesh.mesh.tolist(),
    )


# ---------------------------------------------------------------------------
# Inlined from torchtitan.models.llama4.parallelize to avoid importing
# ShardPlacementResult, which doesn't exist in Aurora's PyTorch framework
# release. Also adds a Shard(0) fallback when the expert hidden dim isn't
# divisible by the FSDP world size — upstream's version assumes divisibility
# and crashes otherwise.
# ---------------------------------------------------------------------------


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
    ep_degree: int = 1,
    edp_mesh: DeviceMesh | None = None,
):
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config: dict[str, Any] = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy, pp_enabled
    )

    if model.tok_embeddings is not None:
        fully_shard(
            model.tok_embeddings,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    if model.norm is not None and model.lm_head is not None:
        fully_shard(
            [model.norm, model.lm_head],
            **fsdp_config,
            reshard_after_forward=reshard_after_forward_policy == "always",
        )

    for layer_id, transformer_block in model.layers.items():
        if transformer_block.moe_enabled:
            assert hasattr(transformer_block, "moe")
            expert_params = set(transformer_block.moe.experts.parameters())
            num_experts = transformer_block.moe.experts.num_experts

            if ep_degree > 1:
                assert edp_mesh is not None
                efsdp_ep_size = edp_mesh["efsdp"].size() * ep_degree
            else:
                efsdp_ep_size = fsdp_config["mesh"].size()

            # Shard(1) shards the hidden dim instead of expert dim when
            # there are more FSDP ranks than experts. But this requires
            # the hidden dim to be evenly divisible by the world size.
            # Fall back to Shard(0) if not (avoids uneven sharding error).
            if efsdp_ep_size > num_experts:
                expert_w = next(iter(transformer_block.moe.experts.parameters()))
                if expert_w.shape[1] % efsdp_ep_size == 0:
                    expert_shard_placement = Shard(1)
                else:
                    expert_shard_placement = Shard(0)
            else:
                expert_shard_placement = Shard(0)

            if ep_degree == 1 and expert_shard_placement == Shard(0):
                fully_shard(
                    transformer_block,
                    **fsdp_config,
                    reshard_after_forward=reshard_after_forward,
                )
            elif ep_degree == 1:
                def _experts_shard_placement_fn(
                    param: nn.Parameter,
                    _expert_params: set = expert_params,
                ) -> Shard | None:
                    if param in _expert_params:
                        return Shard(1)
                    return None

                fully_shard(
                    transformer_block,
                    **fsdp_config,
                    reshard_after_forward=reshard_after_forward,
                    shard_placement_fn=_experts_shard_placement_fn,
                )
            else:
                # ep_degree > 1: per-param mesh with ShardPlacementResult.
                # Imported lazily to avoid hard dependency on a private
                # PyTorch API path that may not exist in older releases.
                from torch.distributed.fsdp._fully_shard._fsdp_common import (
                    FSDPMeshInfo,
                    HSDPMeshInfo,
                    ShardPlacementResult,
                )

                def _mesh_dim(mesh: DeviceMesh, dim_name: str) -> int:
                    mesh_dim_names = mesh.mesh_dim_names
                    if mesh_dim_names is None:
                        if mesh.ndim == 1:
                            return 0
                        raise ValueError(
                            f"Mesh {mesh} must have dim names for 2D HSDP."
                        )
                    try:
                        return tuple(mesh_dim_names).index(dim_name)
                    except ValueError:
                        raise ValueError(
                            f"Mesh {mesh} does not contain dim {dim_name!r}; "
                            f"names={mesh_dim_names!r}"
                        ) from None

                def _fsdp_mesh_info(
                    mesh: DeviceMesh,
                    *,
                    shard_dim_name: str,
                ) -> FSDPMeshInfo | HSDPMeshInfo:
                    """Build the correct FSDP mesh metadata for 1D or 2D meshes."""
                    if mesh.ndim == 1:
                        return FSDPMeshInfo(
                            mesh=mesh,
                            shard_mesh_dim=_mesh_dim(mesh, shard_dim_name),
                        )
                    if mesh.ndim == 2:
                        return HSDPMeshInfo(
                            mesh=mesh,
                            shard_mesh_dim=_mesh_dim(mesh, shard_dim_name),
                            replicate_mesh_dim=_mesh_dim(mesh, "dp_replicate"),
                        )
                    raise ValueError(f"Expected 1D/2D FSDP mesh, got {mesh}")

                assert edp_mesh is not None
                # Match fully_shard()'s normal 2D behavior: 2D meshes must use
                # HSDPMeshInfo so DTensor placements include both replicate and
                # shard axes. Returning FSDPMeshInfo for a 2D mesh creates specs
                # like a 2D DeviceMesh with only one placement.
                edp_mesh_info = _fsdp_mesh_info(edp_mesh, shard_dim_name="efsdp")
                dp_mesh_info = _fsdp_mesh_info(dp_mesh, shard_dim_name="fsdp")

                def _shard_placement_fn(
                    param: nn.Parameter,
                    _expert_params: set = expert_params,
                    _expert_placement: Shard = expert_shard_placement,
                    _edp_mesh_info: FSDPMeshInfo | HSDPMeshInfo = edp_mesh_info,
                    _dp_mesh_info: FSDPMeshInfo | HSDPMeshInfo = dp_mesh_info,
                ) -> ShardPlacementResult:
                    if param in _expert_params:
                        return ShardPlacementResult(
                            placement=_expert_placement, mesh_info=_edp_mesh_info
                        )
                    return ShardPlacementResult(
                        placement=Shard(0), mesh_info=_dp_mesh_info
                    )

                fully_shard(
                    transformer_block,
                    **fsdp_config,
                    reshard_after_forward=reshard_after_forward,
                    shard_placement_fn=_shard_placement_fn,
                )
        else:
            fully_shard(
                transformer_block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )

    fully_shard(model, **fsdp_config)
    disable_fsdp_gradient_division(model)


def apply_moe_ep_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh | None,
    ep_mesh: DeviceMesh | None,
    enable_sp: bool = True,
):
    """Apply MoE expert/tensor parallelism plans to MoE-enabled blocks.

    Same plan structure as upstream `llama4.parallelize.apply_moe_ep_tp`,
    minus the DeepEP/HybridEP token-dispatcher plumbing (we don't use those
    backends on Aurora). Token dispatching for the standard backend is
    handled internally by the LocalTokenDispatcher class at model build.
    """
    assert ep_mesh is not None or tp_mesh is not None
    sp_layout = Shard(1) if enable_sp else Replicate()

    for transformer_block in model.layers.values():
        if not transformer_block.moe_enabled:
            continue

        if tp_mesh is not None:
            moe_layer_plan = {
                "moe": PrepareModuleInputOutput(
                    input_layouts=(sp_layout,),
                    desired_input_layouts=(Replicate(),),
                    use_local_input=False,
                    output_layouts=(Partial(),),
                    desired_output_layouts=(sp_layout,),
                    use_local_output=False,
                ),
                "moe.router.gate": NoParallel(
                    local_output_grad_placements=(Partial(),),
                ),
            }
            if transformer_block.moe.shared_experts is not None:
                moe_layer_plan.update(
                    {
                        "moe.shared_experts.w1": ColwiseParallelWithGradPlacement(
                            local_input_grad_placements=(Partial(),)
                        ),
                        "moe.shared_experts.w2": RowwiseParallel(
                            output_layouts=Partial(),
                        ),
                        "moe.shared_experts.w3": ColwiseParallelWithGradPlacement(
                            local_input_grad_placements=(Partial(),)
                        ),
                    }
                )
            parallelize_module(
                module=transformer_block,
                device_mesh=tp_mesh,
                parallelize_plan=moe_layer_plan,
            )

        # EP disabled: shard routed expert weights across TP mesh.
        # EP enabled: shard across EP mesh (ETP deprecated upstream — see #3167).
        if ep_mesh is None:
            experts_mesh = tp_mesh
            experts_plan = TensorParallel()
        else:
            experts_mesh = ep_mesh
            experts_plan = ExpertParallel()
            dispatcher = transformer_block.moe.experts.token_dispatcher
            if tp_mesh is not None and isinstance(
                dispatcher, AllToAllTokenDispatcher
            ):
                dispatcher.sp_size = tp_mesh.size()
                dispatcher.sp_rank = tp_mesh._sym_get_coordinate(0)

        parallelize_module(
            module=transformer_block.moe.experts,
            device_mesh=experts_mesh,
            parallelize_plan=experts_plan,
        )
