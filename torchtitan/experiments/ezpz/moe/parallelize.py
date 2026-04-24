# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ezpz

import torch
import torch.distributed
import torch.nn as nn

from ezpz.models import summarize_model

from typing import Any

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import Partial, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    PrepareModuleInputOutput,
    RowwiseParallel,
    SequenceParallel,
)

from torchtitan.components.quantization.float8 import find_float8_linear_config
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac

from torchtitan.distributed.context_parallel import apply_cp_to_attention_module
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp, NoParallel
from torchtitan.experiments.ezpz._distributed_compat import ezpz_distributed

# from torchtitan.models.moe import DeepSeekV3Model
from torchtitan.experiments.ezpz.moe import moeModel
from torchtitan.distributed.expert_parallel import (
    ExpertParallel,
    ExpertTensorParallel,
    TensorParallel,
)
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.distributed.tensor_parallel import (
    ColwiseParallelWithGradPlacement,
)
from torchtitan.protocols import ModelConvertersContainer
from torchtitan.tools.logging import logger


def disable_fsdp_gradient_division(model: nn.Module) -> None:
    force_sum_reduction = False
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        backend = ezpz_distributed.get_torch_backend() or str(torch.distributed.get_backend())
        if backend and "nccl" not in str(backend).lower():
            force_sum_reduction = True

    fsdp_modules_updated = 0
    for module in model.modules():
        # Be resilient to FSDPModule class location changes across PyTorch releases.
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


# Adapted from llama4/infra/parallelize.py
def parallelize_moe(
    model: moeModel,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters: ModelConvertersContainer.Config,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    # TODO: TP currently cannot handle uneven seq_len because we set
    #       `use_local_output=True` to use plain Tensors for legacy reasons.
    #       Need to revisit this.
    assert (
        training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    if parallel_dims.tp_enabled:
        float8_config = find_float8_linear_config(model_converters.converters)
        enable_float8_linear = float8_config is not None
        float8_is_rowwise = float8_config is not None and float8_config.recipe_name in (
            "rowwise",
            "rowwise_with_gw_hp",
        )

        enable_float8_tensorwise_tp = enable_float8_linear and not float8_is_rowwise
        if enable_float8_tensorwise_tp:
            # TODO(jianiw): This branch needs to be tested and enabled
            raise NotImplementedError(
                "Currently, float8 tensorwise TP is not tested for moe"
            )

        enable_sp = parallelism.enable_sequence_parallel

        tp_mesh = parallel_dims.get_mesh("tp")
        apply_non_moe_tp(
            model,
            tp_mesh,
            enable_loss_parallel=not parallelism.disable_loss_parallel,
            enable_float8_tensorwise_tp=False,
            enable_cp=parallel_dims.cp_enabled,
            enable_sp=enable_sp,
        )
        maybe_enable_async_tp(parallelism, compile_config, tp_mesh)

    # EP/TP parallelization for MoE layers. The comm_backend (standard,
    # deepep, etc.) is now configured at model creation time via the
    # token dispatcher, not at parallelization time.
    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        apply_moe_ep_tp(
            model,
            tp_mesh=parallel_dims.get_optional_mesh("tp"),
            ep_mesh=parallel_dims.get_optional_mesh("ep"),
            etp_mesh=parallel_dims.get_optional_mesh("etp"),
            ep_etp_mesh=parallel_dims.get_optional_mesh(["ep", "etp"]),
        )

    if parallel_dims.cp_enabled:
        if parallel_dims.tp_enabled:
            raise NotImplementedError(
                "Context Parallel with Tensor Parallel is not yet supported for DeepSeek-V3. "
                "See https://github.com/pytorch/torchtitan/issues/2446"
            )
        apply_cp_to_attention_module(
            # pyrefly: ignore [missing-attribute, not-callable]
            [block.attention.inner_attention for block in model.layers.values()],
            parallel_dims.get_mesh("cp"),
        )

    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )

    if ac_config.mode != "none":
        apply_ac(
            model,
            ac_config,
            model_compile_enabled=model_compile_enabled,
            base_folder=dump_folder,
        )

    if model_compile_enabled:
        # Upstream apply_compile_sparse uses fullgraph=True which fails on
        # XPU after 00b7f569 removed maybe_enable_amp — MoE routing's
        # dynamic shapes cause recompilation that fullgraph=True forbids.
        # Apply compile per-block without fullgraph instead.
        import torch

        torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
        for layer_id, block in model.layers.named_children():
            block.compile(backend=compile_config.backend)
            model.layers.register_module(layer_id, block)

    dp_mesh_names = (
        ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
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
    disable_fsdp_gradient_division(model)

    logger.info("Applied fully_shard to the model")

    if training.enable_cpu_offload:
        logger.info("Applied CPU Offloading to the model")

    logger.info(f"\n+{ezpz.models.summarize_model(model)}")

    return model


def apply_non_moe_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    enable_loss_parallel: bool,
    enable_float8_tensorwise_tp: bool,
    enable_cp: bool,
    enable_sp: bool = True,
):
    """Apply tensor parallelism."""
    sp_layout = Shard(1) if enable_sp else Replicate()
    embed_plan = RowwiseParallel(
        input_layouts=Replicate(),
        output_layouts=sp_layout,
        use_local_output=enable_sp,
    )

    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": embed_plan,
            "norm": SequenceParallel() if enable_sp else NoParallel(),
            "output": ColwiseParallel(
                input_layouts=sp_layout,
                output_layouts=Shard(-1) if enable_loss_parallel else Replicate(),
                use_local_output=not enable_loss_parallel,
            ),
        },
    )

    rowwise_parallel, colwise_parallel, prepare_module_input = (
        RowwiseParallel,
        ColwiseParallel,
        PrepareModuleInput,
    )

    attention_kernel_plan = prepare_module_input(
        input_layouts=(Shard(1), Shard(1), Shard(1)),
        desired_input_layouts=(Shard(1), Shard(1), Shard(1)),
        use_local_output=True,
    )
    positions_sharding = Replicate() if enable_cp else None
    norm_plan = SequenceParallel() if enable_sp else NoParallel()
    rowwise_output_plan = rowwise_parallel(
        output_layouts=sp_layout, use_local_output=enable_sp
    )

    # pyrefly: ignore [not-callable]
    for transformer_block in model.layers.values():
        # pyrefly: ignore [no-matching-overload]
        layer_plan = {
            "attention_norm": norm_plan,
            "attention": prepare_module_input(
                input_layouts=(sp_layout, Replicate(), None, positions_sharding),
                desired_input_layouts=(
                    Replicate(),
                    Replicate(),
                    None,
                    positions_sharding,
                ),
            ),
            "attention.wkv_a": NoParallel(),
            "attention.wkv_b": colwise_parallel(use_local_output=False),
            "attention.kv_norm": NoParallel(),
            "attention.inner_attention": attention_kernel_plan,
            "attention.wo": rowwise_output_plan,
            "ffn_norm": norm_plan,
        }

        # pyrefly: ignore [missing-attribute]
        if transformer_block.attention.q_lora_rank == 0:
            layer_plan["attention.wq"] = colwise_parallel(
                use_local_output=False
            )  # This is only used when q_lora_rank==0
        else:
            layer_plan.update(
                {
                    "attention.wq_a": NoParallel(),
                    "attention.wq_b": colwise_parallel(use_local_output=False),
                    "attention.q_norm": NoParallel(),
                }
            )

        # pyrefly: ignore [missing-attribute]
        if not transformer_block.moe_enabled:
            layer_plan.update(
                {
                    "feed_forward": prepare_module_input(
                        input_layouts=(sp_layout,),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "feed_forward.w1": colwise_parallel(),
                    "feed_forward.w2": rowwise_output_plan,
                    "feed_forward.w3": colwise_parallel(),
                }
            )

        parallelize_module(
            # pyrefly: ignore [bad-argument-type]
            module=transformer_block,
            device_mesh=tp_mesh,
            # pyrefly: ignore [bad-argument-type]
            parallelize_plan=layer_plan,
        )

    logger.info(
        f"Applied {'Float8 tensorwise ' if enable_float8_tensorwise_tp else ''}"
        "Tensor Parallelism to the model"
    )


# ---------------------------------------------------------------------------
# Inlined from torchtitan.models.llama4.parallelize to avoid importing
# ShardPlacementResult which doesn't exist in the Aurora PyTorch framework.
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
    gradient_divide_factor: int | None = None,
):
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
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
    if model.norm is not None and model.output is not None:
        fully_shard(
            [model.norm, model.output],
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
                # Check if hidden dim (dim 1 of expert weights) is divisible
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
                # ep_degree > 1: per-param mesh with ShardPlacementResult
                from torch.distributed.fsdp._fully_shard._fsdp_common import (
                    FSDPMeshInfo,
                    ShardPlacementResult,
                )

                assert edp_mesh is not None
                edp_mesh_info = FSDPMeshInfo(mesh=edp_mesh, shard_mesh_dim=0)
                dp_mesh_info = FSDPMeshInfo(mesh=dp_mesh, shard_mesh_dim=0)

                def _shard_placement_fn(
                    param: nn.Parameter,
                    _expert_params: set = expert_params,
                    _expert_placement: Shard = expert_shard_placement,
                    _edp_mesh_info: FSDPMeshInfo = edp_mesh_info,
                    _dp_mesh_info: FSDPMeshInfo = dp_mesh_info,
                ) -> ShardPlacementResult:
                    if param in _expert_params:
                        return ShardPlacementResult(
                            placement=_expert_placement, mesh_info=_edp_mesh_info
                        )
                    else:
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
    etp_mesh: DeviceMesh | None,
    ep_etp_mesh: DeviceMesh | None,
    comm_backend: str = "standard",
    hybridep_non_blocking_expert_capacity_factor: float | None = None,
):
    assert ep_mesh is not None or tp_mesh is not None

    for transformer_block in model.layers.values():
        if not transformer_block.moe_enabled:
            continue

        if tp_mesh is not None:
            moe_layer_plan = {
                "moe": PrepareModuleInputOutput(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                    use_local_input=False,
                    output_layouts=(Partial(),),
                    desired_output_layouts=(Shard(1),),
                ),
                "moe.router.gate": NoParallel(
                    local_output_grad_placements=(Partial(),),
                ),
            }
            # Token dispatching is now handled internally by the
            # TokenDispatcher classes (LocalTokenDispatcher, etc.).
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

        experts_mesh, experts_plan = None, None
        if ep_mesh is None:
            assert ep_etp_mesh is None
            experts_mesh = tp_mesh
            experts_plan = TensorParallel()
        elif tp_mesh is None or etp_mesh is None:
            assert ep_etp_mesh is None
            experts_mesh = ep_mesh
            # DeepEP/HybridEP communication is now handled by the
            # TokenDispatcher (DeepEPTokenDispatcher) configured at model
            # creation time, not at parallelization time.
            experts_plan = ExpertParallel()
        else:
            experts_mesh = ep_etp_mesh
            experts_plan = ExpertTensorParallel()

        parallelize_module(
            module=transformer_block.moe.experts,
            device_mesh=experts_mesh,
            parallelize_plan=experts_plan,
        )
