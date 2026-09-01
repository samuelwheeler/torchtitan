# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import copy
import contextlib
import hashlib
import inspect
import math
import os
from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._mesh_layout import _MeshLayout
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    _PipelineSchedule,
    _PipelineScheduleRuntime,
    _batch_p2p,
    _wait_batch_p2p,
    get_schedule_class,
    PipelineScheduleMulti,
    PipelineScheduleSingle,
    Schedule1F1B,
    ScheduleDualPipeV,
    ScheduleZBVZeroBubble,
)

from torchtitan.components.loss import ChunkedCELoss, LossFunction
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.model_spec import ParallelizeFunction
from torchtitan.protocols.module import ModuleDict, ModuleList
from torchtitan.tools.logging import logger

__all__ = [
    "pipeline_llm",
    "build_pipeline_schedule",
    "generate_llm_fqn_per_model_part",
    "pipeline_module_split",
]


_FSDP_OVERLAP_POLICIES = ("bulk", "deferred", "pp_first")
_PP_FIRST_TORCH_GIT_VERSION = "808e7f2bb128dc3fd517bd7883cf3a66070a1607"
_PP_FIRST_SCHEDULES_SHA256 = (
    "b9f2c8e349652524460bd836af354a88615ed347b81c1b52db5a3b996f73163b"
)


def _get_pipeline_fsdp_overlap_policy(parallelism: ParallelismConfig) -> str:
    policy = parallelism.pipeline_parallel_fsdp_overlap_policy
    if policy not in _FSDP_OVERLAP_POLICIES:
        raise ValueError(
            f"unsupported pipeline FSDP overlap policy {policy!r}; "
            f"expected one of {_FSDP_OVERLAP_POLICIES}"
        )
    if parallelism.pipeline_parallel_fsdp_overlap:
        if policy != "bulk":
            raise ValueError(
                "pipeline_parallel_fsdp_overlap is the deprecated alias for "
                'policy="deferred" and cannot be combined with an explicit '
                f"{policy!r} policy"
            )
        return "deferred"
    return policy


def _validate_pp_first_torch_source() -> None:
    if torch.version.git_version != _PP_FIRST_TORCH_GIT_VERSION:
        raise RuntimeError(
            "pipeline FSDP pp_first is guarded for PyTorch git "
            f"{_PP_FIRST_TORCH_GIT_VERSION}, but found "
            f"{torch.version.git_version!r}. Re-audit Schedule1F1B before "
            "enabling this experimental path."
        )
    source_path = inspect.getsourcefile(Schedule1F1B)
    if source_path is None:
        raise RuntimeError("could not locate the installed Schedule1F1B source")
    with open(source_path, "rb") as source_file:
        source_sha256 = hashlib.sha256(source_file.read()).hexdigest()
    if source_sha256 != _PP_FIRST_SCHEDULES_SHA256:
        raise RuntimeError(
            "pipeline FSDP pp_first expected schedules.py SHA256 "
            f"{_PP_FIRST_SCHEDULES_SHA256}, got {source_sha256}. Re-audit the "
            "copied Schedule1F1B control flow before enabling this experiment."
        )


def _patch_fork_rng_device_type_for_pipeline() -> None:
    """Make PyTorch pipeline metadata inference fork the accelerator RNG.

    ``torch.distributed.pipelining`` passes explicit ``torch.device`` objects
    to ``torch.random.fork_rng`` but omits its ``device_type`` argument.  The
    latter defaults to CUDA, so an XPU-only PyTorch build fails before the
    first pipeline step with ``Torch not compiled with CUDA enabled``.  Keep
    the workaround narrowly scoped to calls with a homogeneous, explicit set
    of non-CUDA devices; all other callers retain PyTorch's default behavior.
    """
    fork_rng = torch.random.fork_rng
    if getattr(fork_rng, "_torchtitan_device_type_patch", False):
        return

    @contextlib.contextmanager
    def fork_rng_for_pipeline(
        devices=None,
        enabled=True,
        _caller="fork_rng",
        _devices_kw="devices",
        device_type="cuda",
    ):
        explicit_devices = None if devices is None else list(devices)
        if explicit_devices and device_type == "cuda":
            device_types = {torch.device(device).type for device in explicit_devices}
            if len(device_types) == 1:
                inferred_device_type = next(iter(device_types))
                if inferred_device_type != "cpu":
                    device_type = inferred_device_type

        with fork_rng(
            devices=explicit_devices,
            enabled=enabled,
            _caller=_caller,
            _devices_kw=_devices_kw,
            device_type=device_type,
        ):
            yield

    fork_rng_for_pipeline._torchtitan_device_type_patch = True
    torch.random.fork_rng = fork_rng_for_pipeline


_patch_fork_rng_device_type_for_pipeline()


class _OverlappedFSDPPipelineStage(PipelineStage):
    """Launch FSDP reductions from the final 1F1B microbatch backward.

    PyTorch 2.13's pipeline FSDP path deliberately disables gradient sync for
    every microbatch and manually invokes all FSDP ``post_backward`` handlers
    after pipeline backward and P2P communication have finished. For a full
    1F1B backward, FSDP's normal microbatch controls can instead accumulate
    early-microbatch gradients and launch each parameter group's reduction as
    it becomes ready during the final microbatch backward. Finalization is
    deferred until the schedule's ``REDUCE_GRAD`` action so that waiting for
    DP communication cannot delay the activation-gradient send to the preceding
    pipeline stage.

    ``backward_maybe_with_nosync`` is not a public extension API, so keep this
    override exact-version guarded and opt-in. The non-final path delegates to
    the installed implementation unchanged.
    """

    _SUPPORTED_TORCH_GIT_VERSION = "808e7f2bb128dc3fd517bd7883cf3a66070a1607"

    def __init__(self, *args, **kwargs):
        installed_git_version = getattr(torch.version, "git_version", None)
        if installed_git_version != self._SUPPORTED_TORCH_GIT_VERSION:
            raise RuntimeError(
                "pipeline_parallel_fsdp_overlap is guarded for PyTorch git "
                f"{self._SUPPORTED_TORCH_GIT_VERSION}, but found "
                f"{installed_git_version!r}. Re-audit PipelineStage's FSDP "
                "backward behavior before enabling this experimental path."
            )
        super().__init__(*args, **kwargs)
        self._fsdp_overlap_reductions_launched = False
        self._fsdp_overlap_event_hook: Callable[[str], None] | None = None

    def _record_fsdp_overlap_event(self, name: str) -> None:
        event_hook = getattr(self, "_fsdp_overlap_event_hook", None)
        if event_hook is not None:
            event_hook(name)

    def backward_maybe_with_nosync(
        self,
        backward_type,
        bwd_kwargs: dict,
        last_backward: bool = False,
    ):
        from torch.distributed._composable.fsdp import FSDPModule

        if not (isinstance(self.submod, FSDPModule) and last_backward):
            return super().backward_maybe_with_nosync(
                backward_type,
                bwd_kwargs,
                last_backward=last_backward,
            )

        if backward_type != "full":
            raise RuntimeError(
                "pipeline_parallel_fsdp_overlap currently supports only full "
                "1F1B backward, not split input/weight backward"
            )

        # These are FSDP's public controls for microbatch accumulation. Enabling
        # gradient synchronization before autograd starts lets per-parameter-
        # group post-backward hooks reduce the accumulated gradient while earlier
        # layers compute. Keep ``is_last_backward`` false so FSDP's automatic
        # root callback does not wait for those reductions before PipelineStage
        # can send the activation gradient to the preceding stage.
        self.submod.set_is_last_backward(False)
        self.submod.set_reshard_after_backward(True)
        self.submod.set_requires_gradient_sync(True)

        # PipelineStage uses this private helper internally. It calls
        # torch.autograd.backward(), which fires the ordinary FSDP hooks. The
        # root callback still resets per-backward state, but with
        # ``is_last_backward=False`` it deliberately does not finalize/wait.
        from torch.distributed.pipelining._backward import stage_backward

        self._record_fsdp_overlap_event("FINAL_BACKWARD_BEGIN")
        grads = stage_backward(
            bwd_kwargs["stage_output"],
            bwd_kwargs["output_grads"],
            bwd_kwargs["input_values"],
        )
        self._record_fsdp_overlap_event("FINAL_BACKWARD_RETURN")
        self._fsdp_overlap_reductions_launched = True
        return grads, None

    def perform_reduce_grad(self, grad_scale_factor: int):
        from torch.distributed._composable.fsdp import FSDPModule

        if not isinstance(self.submod, FSDPModule):
            return super().perform_reduce_grad(grad_scale_factor)

        # The final microbatch backward already ran FSDP post-backward hooks.
        # Replaying PipelineStage's bulk reduction here would reduce twice.
        if not self._fsdp_overlap_reductions_launched:
            raise RuntimeError(
                "pipeline_parallel_fsdp_overlap reached REDUCE_GRAD without "
                "completing its final full backward; refusing to skip gradient "
                "synchronization"
            )

        # This is the finalization-only portion of the installed FSDP2 root
        # callback. Calling the full callback would see IDLE parameter groups
        # left by autograd's earlier non-final callback and invoke
        # ``post_backward`` a second time, duplicating reductions. Keep this
        # exact-version guarded with the override above.
        from torch.distributed._composable.replicate_with_fsdp import (
            replicate,
            ReplicateModule,
        )
        from torch.distributed.fsdp import fully_shard

        self.submod.set_is_last_backward(True)
        distributed_state = (
            replicate.state(self.submod)
            if isinstance(self.submod, ReplicateModule)
            else fully_shard.state(self.submod)
        )
        if distributed_state._state_ctx.post_backward_final_callback_queued:
            raise RuntimeError(
                "pipeline_parallel_fsdp_overlap reached REDUCE_GRAD before "
                "autograd's FSDP root callback completed"
            )
        self._record_fsdp_overlap_event("DP_FINALIZE")
        with torch.profiler.record_function(
            "FSDP::deferred_root_post_backward_finalize"
        ):
            for state in distributed_state._state_ctx.all_states:
                for fsdp_param_group in state._fsdp_param_groups:
                    fsdp_param_group.finalize_backward()
            distributed_state._comm_ctx.post_forward_order.clear()
            current_stream = distributed_state._device_handle.current_stream()
            for rs_state in distributed_state._comm_ctx.reduce_scatter_states:
                if rs_state.event is not None:
                    current_stream.wait_event(rs_state.event)
            distributed_state._comm_ctx.reduce_scatter_states.clear()

        if grad_scale_factor != 1:
            self.scale_grads(grad_scale_factor)
        self._fsdp_overlap_reductions_launched = False


class _NativeDDPPipelineStage(PipelineStage):
    """Keep dynamic pipeline metadata inference outside the DDP reducer."""

    def _forward_metadata_inference(self, *args, **kwargs):
        from torchtitan.experiments.ezpz.native_ddp import NativeDDP

        ddp = self.submod
        if not isinstance(ddp, NativeDDP):
            raise TypeError(
                "native-DDP pipeline stage metadata inference requires NativeDDP"
            )
        self.submod = ddp.module
        try:
            return super()._forward_metadata_inference(*args, **kwargs)
        finally:
            self.submod = ddp


class _PPFirstFSDPPipelineStage(_OverlappedFSDPPipelineStage):
    """Keep the final upstream PP send ahead of FSDP2 reduction launch."""

    def __init__(self, *args, **kwargs):
        _validate_pp_first_torch_source()
        super().__init__(*args, **kwargs)
        self._pp_first_final_backward_complete = False
        self._pp_first_send_submitted = False

    def backward_maybe_with_nosync(
        self,
        backward_type,
        bwd_kwargs: dict,
        last_backward: bool = False,
    ):
        from torch.distributed._composable.fsdp import FSDPModule

        if not (
            isinstance(self.submod, FSDPModule)
            and last_backward
            and not self.is_first
        ):
            return super().backward_maybe_with_nosync(
                backward_type,
                bwd_kwargs,
                last_backward=last_backward,
            )
        if backward_type != "full":
            raise RuntimeError(
                "pipeline FSDP pp_first supports only full 1F1B backward"
            )
        if self._fsdp_overlap_reductions_launched:
            raise RuntimeError("pp_first final backward began with stale DP work")

        self._record_fsdp_overlap_event("FINAL_BACKWARD_BEGIN")
        result = PipelineStage.backward_maybe_with_nosync(
            self,
            backward_type,
            bwd_kwargs,
            last_backward=last_backward,
        )
        self._record_fsdp_overlap_event("FINAL_BACKWARD_RETURN")
        self._pp_first_final_backward_complete = True
        return result

    def mark_final_backward_send_submitted(self, send_count: int) -> None:
        if self.is_first or send_count < 1:
            raise RuntimeError(
                "pp_first sending stage did not submit a final backward send"
            )
        if not self._pp_first_final_backward_complete:
            raise RuntimeError("pp_first submitted PP send before final backward")
        if self._pp_first_send_submitted:
            raise RuntimeError("pp_first submitted the final PP send twice")
        self._pp_first_send_submitted = True
        self._record_fsdp_overlap_event("PP_SEND_SUBMIT")

    def launch_fsdp_reductions_after_pp_send(self) -> None:
        from torch.distributed._composable.fsdp import FSDPModule
        from torch.distributed._composable.replicate_with_fsdp import (
            replicate,
            ReplicateModule,
        )
        from torch.distributed.fsdp import fully_shard

        if not isinstance(self.submod, FSDPModule):
            raise RuntimeError("pp_first DP launch requires an FSDP2 module")
        if self.is_first or not self._pp_first_send_submitted:
            raise RuntimeError("pp_first DP launch requires a submitted PP send")
        if self._fsdp_overlap_reductions_launched:
            raise RuntimeError("pp_first attempted to launch DP reductions twice")

        self.submod.set_is_last_backward(True)
        self.submod.set_reshard_after_backward(True)
        self.submod.set_requires_gradient_sync(True)
        distributed_state = (
            replicate.state(self.submod)
            if isinstance(self.submod, ReplicateModule)
            else fully_shard.state(self.submod)
        )
        self._record_fsdp_overlap_event("DP_LAUNCH")
        for state in distributed_state._state_ctx.all_states:
            for fsdp_param_group in state._fsdp_param_groups:
                fsdp_param_group.post_backward()
        self._fsdp_overlap_reductions_launched = True

    def perform_reduce_grad(self, grad_scale_factor: int):
        if not self.is_first and not self._pp_first_send_submitted:
            raise RuntimeError("pp_first finalization requires a submitted PP send")
        super().perform_reduce_grad(grad_scale_factor)
        self._pp_first_final_backward_complete = False
        self._pp_first_send_submitted = False


class _PPFirstSchedule1F1B(Schedule1F1B):
    """Exact-version Schedule1F1B with a post-final-send DP launch point."""

    def __init__(self, *args, **kwargs):
        _validate_pp_first_torch_source()
        super().__init__(*args, **kwargs)
        if not isinstance(self._stage, _PPFirstFSDPPipelineStage):
            raise TypeError("pp_first schedule requires a pp_first pipeline stage")

    def _launch_dp_if_final_backward(
        self,
        bwd_mb_index: int,
        bwd_sends: list[dist.P2POp],
    ) -> None:
        if bwd_mb_index != self._n_microbatches:
            return
        stage = self._stage
        if stage.is_first:
            if bwd_sends:
                raise RuntimeError("first pipeline stage produced a backward send")
            return
        stage.mark_final_backward_send_submitted(len(bwd_sends))
        stage.launch_fsdp_reductions_after_pp_send()

    def _step_microbatches(
        self,
        arg_mbs: list | None = None,
        kwarg_mbs: list | None = None,
        target_mbs: list | None = None,
        losses: list | None = None,
        return_outputs: bool = True,
        loss_kwargs: dict | None = None,
    ):
        arg_mbs, kwarg_mbs = self._check_inputs(
            arg_mbs, kwarg_mbs, target_mbs, losses
        )
        maybe_first_target = target_mbs[0] if target_mbs is not None else None
        self._initialize_stage(
            arg_mbs[0], kwarg_mbs[0], maybe_first_target, loss_kwargs
        )

        warmup_chunks = min(
            self._n_microbatches,
            self._num_stages - self._stage.stage_index,
        )
        fwd_mb_index = 0
        bwd_mb_index = 0

        send_work: list[dist.Work] = []
        fwd_sends = []
        for _ in range(warmup_chunks):
            fwd_recvs = self._stage.get_fwd_recv_ops(fwd_mb_index)
            _wait_batch_p2p(_batch_p2p(fwd_recvs, desc="fwd_recv"))
            output = self._stage.forward_one_chunk(
                fwd_mb_index,
                arg_mbs[fwd_mb_index],
                kwarg_mbs[fwd_mb_index],
                save_forward_output=return_outputs,
            )
            _wait_batch_p2p(send_work)
            fwd_sends = self._stage.get_fwd_send_ops(fwd_mb_index)
            if fwd_mb_index != warmup_chunks - 1:
                send_work = _batch_p2p(fwd_sends, desc="fwd_send")
            self._maybe_compute_loss(
                self._stage, output, target_mbs, fwd_mb_index, loss_kwargs
            )
            fwd_mb_index += 1

        while True:
            bwd_recvs = self._stage.get_bwd_recv_ops(bwd_mb_index)
            _wait_batch_p2p(
                _batch_p2p(fwd_sends + bwd_recvs, desc="fwd_send_bwd_recv")
            )
            loss = self._maybe_get_loss(self._stage, bwd_mb_index)
            self._stage.backward_one_chunk(
                bwd_mb_index,
                loss=loss,
                last_backward=bwd_mb_index == self._n_microbatches - 1,
            )
            bwd_sends = self._stage.get_bwd_send_ops(bwd_mb_index)
            bwd_mb_index += 1
            if fwd_mb_index == self._n_microbatches:
                break

            fwd_recvs = self._stage.get_fwd_recv_ops(fwd_mb_index)
            _wait_batch_p2p(
                _batch_p2p(bwd_sends + fwd_recvs, desc="bwd_send_fwd_recv")
            )
            output = self._stage.forward_one_chunk(
                fwd_mb_index,
                arg_mbs[fwd_mb_index],
                kwarg_mbs[fwd_mb_index],
                save_forward_output=return_outputs,
            )
            self._maybe_compute_loss(
                self._stage, output, target_mbs, fwd_mb_index, loss_kwargs
            )
            fwd_sends = self._stage.get_fwd_send_ops(fwd_mb_index)
            fwd_mb_index += 1

        send_work = _batch_p2p(bwd_sends, desc="bwd_send")
        self._launch_dp_if_final_backward(bwd_mb_index, bwd_sends)

        while bwd_mb_index < self._n_microbatches:
            bwd_recvs = self._stage.get_bwd_recv_ops(bwd_mb_index)
            _wait_batch_p2p(_batch_p2p(bwd_recvs, desc="bwd_recv"))
            loss = self._maybe_get_loss(self._stage, bwd_mb_index)
            self._stage.backward_one_chunk(
                bwd_mb_index,
                loss=loss,
                last_backward=bwd_mb_index == self._n_microbatches - 1,
            )
            _wait_batch_p2p(send_work)
            bwd_sends = self._stage.get_bwd_send_ops(bwd_mb_index)
            send_work = _batch_p2p(bwd_sends, desc="bwd_send")
            bwd_mb_index += 1
            self._launch_dp_if_final_backward(bwd_mb_index, bwd_sends)

        _wait_batch_p2p(send_work)
        self._update_losses(self._stage, losses)
        self._stage.perform_reduce_grad(
            self._n_microbatches if self.scale_grads else 1
        )


def _build_get_mesh_callback(
    parallel_dims: ParallelDims,
) -> Callable[[tuple[str, ...], _MeshLayout | None], DeviceMesh | None]:
    """Build a callback that resolves a DeviceMesh from dimension names.

    Pipeline parallelism requires an SPMD mesh during module split so that
    at runtime the current PP rank can reconstruct a DTensor after receiving
    a plain tensor from the previous PP rank. DTensors are not directly
    serializable across PP stages (because ProcessGroup is not serializable),
    so each stage uses this callback to obtain its local DeviceMesh and
    re-wrap incoming tensors as DTensors with the correct placements.
    """

    def _get_mesh(
        mesh_dim_names: tuple[str, ...], mesh_layout: _MeshLayout | None
    ) -> DeviceMesh | None:
        mesh = parallel_dims.get_mesh(list(mesh_dim_names))
        if mesh_layout is not None and mesh._layout != mesh_layout:
            return None
        return mesh

    return _get_mesh


def pipeline_llm(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
    device: torch.device,
    model_config: BaseModel.Config,
    parallelize_fn: ParallelizeFunction,
    loss_fn: LossFunction,
) -> tuple[_PipelineSchedule, list[nn.Module], bool, bool]:
    pp_mesh = parallel_dims.get_mesh("pp")
    fsdp_overlap_policy = _get_pipeline_fsdp_overlap_policy(parallelism)

    if fsdp_overlap_policy != "bulk":
        if parallelism.pipeline_parallel_schedule != "1F1B":
            raise ValueError(
                "pipeline FSDP overlap supports only the full-backward 1F1B "
                "pipeline schedule"
            )
        if parallelism.pipeline_parallel_schedule_csv:
            raise ValueError(
                "pipeline FSDP overlap does not support a custom schedule CSV"
            )

    # Determine the number of virtual stages based on schedule type
    schedule_class = get_schedule_class(parallelism.pipeline_parallel_schedule)
    is_single_stage_schedule = issubclass(schedule_class, PipelineScheduleSingle)
    layers_per_stage = parallelism.pipeline_parallel_layers_per_stage
    if hasattr(model_config, "layers"):
        num_layers = len(model_config.layers)
    else:
        raise ValueError("Model does not have n_layers attribute.")

    # You can adjust these weights based on the computational cost of embeddings and output layers
    # Higher weights mean these modules are treated as "heavier" in the distribution
    input_weight = parallelism.pipeline_parallel_first_stage_less_layers
    output_weight = parallelism.pipeline_parallel_last_stage_less_layers

    # Calculate number of virtual stages
    if layers_per_stage is not None:

        # Calculate number of virtual stages needed (using ceiling division)
        # This allows for unequal distribution where stages can differ by at most 1 layer
        num_virtual_stages = math.ceil(
            (num_layers + input_weight + output_weight) / layers_per_stage
        )

        # Validation: check stages per rank based on schedule type
        model_config_info = f"Model has {num_layers} layers with pipeline_parallel_layers_per_stage={layers_per_stage}"
        stage_distribution_info = (
            f"resulting in {num_virtual_stages=} across {parallel_dims.pp} PP ranks"
        )

        if num_virtual_stages % parallel_dims.pp != 0:
            raise ValueError(
                f"Number of virtual stages ({num_virtual_stages}) must be divisible by "
                f"pipeline parallel size ({parallel_dims.pp}). "
                f"{model_config_info}. "
                f"Please adjust pipeline_parallel_layers_per_stage to a value that results in a number of stages "
                f"divisible by {parallel_dims.pp}."
            )

        stages_per_rank = num_virtual_stages // parallel_dims.pp

        if is_single_stage_schedule and stages_per_rank != 1:
            raise ValueError(
                f"Single stage schedule requires exactly 1 stage per rank, but got {stages_per_rank} stages per rank. "
                f"{model_config_info}, {stage_distribution_info}. "
                f"Please increase pipeline_parallel_layers_per_stage to {num_layers // parallel_dims.pp} or higher "
                f"to achieve 1 stage per rank."
            )

        if not is_single_stage_schedule and stages_per_rank < 2:
            raise ValueError(
                f"Multi-stage schedule requires at least 2 stages per rank, but got {stages_per_rank} stages per rank. "
                f"{model_config_info}, {stage_distribution_info}. "
                f"Please decrease pipeline_parallel_layers_per_stage to achieve at least 2 stages per rank."
            )
    else:
        # Fallback to default behavior when layers_per_stage is not provided
        # For multi-stage schedules, default is 2 virtual stages per rank
        # For single-stage schedules, default is 1 virtual stage per rank
        stages_per_rank = 1 if is_single_stage_schedule else 2
        num_virtual_stages = parallel_dims.pp * stages_per_rank

    module_names_per_stage = parallelism.module_fqns_per_model_part
    if module_names_per_stage is None:
        module_names_per_stage = generate_llm_fqn_per_model_part(
            num_virtual_stages, num_layers, input_weight, output_weight
        )
    for i, stage_ms in enumerate(module_names_per_stage):
        logger.debug(f"Stage {i}: {stage_ms}")

    get_mesh_cb = _build_get_mesh_callback(parallel_dims)
    stages, model_parts = pipeline_module_split(
        model,
        pp_mesh,
        parallelism.pipeline_parallel_schedule,
        device,
        module_names_per_stage,
        get_mesh=get_mesh_cb,
        fsdp_overlap_policy=fsdp_overlap_policy,
        native_ddp=parallelism.enable_data_parallel_native_ddp,
    )

    # For PP with looped schedules, each item in model_parts is one stage-model-chunk.
    # We need to iterate through model_parts to apply SPMD parallelisms, compilation,
    # optimizer, and checkpointing
    for i, m in enumerate(model_parts):
        # apply SPMD-style PT-D techniques
        m = parallelize_fn(
            m,
            parallel_dims=parallel_dims,
            training=training,
            parallelism=parallelism,
            compile_config=compile_config,
            ac_config=ac_config,
            dump_folder=dump_folder,
        )
        model_parts[i] = m
        # NOTE: this is to update the model in the stage
        #       in case the model is modified e.g. by torch.compile
        stages[i].submod = m

    pp_schedule = build_pipeline_schedule(
        parallelism=parallelism,
        local_batch_size=training.local_batch_size,
        stages=stages,
        loss_fn=loss_fn,
    )

    # This is used in the train loop to determine whether to pass in the input_ids and labels
    has_first_stage = False
    has_last_stage = False
    for stage in stages:
        if stage.is_first:
            has_first_stage = True
        if stage.is_last:
            has_last_stage = True

    return pp_schedule, model_parts, has_first_stage, has_last_stage


def build_pipeline_schedule(
    *,
    parallelism: ParallelismConfig,
    local_batch_size: int,
    stages: list[PipelineStage],
    loss_fn: Callable,
) -> _PipelineSchedule:
    """Builds a pipeline schedule for the given job configuration and stages.

    Args:
        parallelism (ParallelismConfig): The parallelism configuration.
        local_batch_size (int): The local batch size for computing microbatches.
        stages (list[PipelineStage]): The stages to be scheduled.
        loss_fn (Callable): The loss function.

    Returns:
        _PipelineSchedule: The pipeline schedule for the given stages.
    """
    pp_schedule_csv = parallelism.pipeline_parallel_schedule_csv

    # Validate that pp_schedule_csv is a valid path
    fsdp_overlap_policy = _get_pipeline_fsdp_overlap_policy(parallelism)
    if fsdp_overlap_policy != "bulk" and isinstance(loss_fn, ChunkedCELoss):
        raise ValueError(
            "pipeline FSDP overlap requires standard schedule-owned full "
            "backward; ChunkedCELoss performs lm_head backward inside loss "
            "forward and is not supported"
        )
    if pp_schedule_csv:
        if not os.path.isfile(pp_schedule_csv):
            raise FileNotFoundError(
                f"The specified path {pp_schedule_csv} does not exist or is not a file."
            )
        schedule_class = _PipelineScheduleRuntime
    elif fsdp_overlap_policy == "pp_first":
        schedule_class = _PPFirstSchedule1F1B
    else:
        schedule_class = get_schedule_class(parallelism.pipeline_parallel_schedule)

    looped_schedule = issubclass(schedule_class, PipelineScheduleMulti)
    microbatch_size = parallelism.pipeline_parallel_microbatch_size
    batch_size = local_batch_size
    # validate that the batch size is divisible by the microbatch_size otherwise we'll hang or error during training
    if batch_size % microbatch_size != 0:
        raise ValueError(
            f"Batch size {local_batch_size} must be divisible by microbatch_size {microbatch_size}. "
            "Update the config arguments for either batch_size or pipeline_parallel_microbatch_size."
        )
    n_microbatches = batch_size // microbatch_size
    # We expect that the number of local stages (`len(stages)`) is the same across all ranks
    num_total_stages = parallelism.pipeline_parallel_degree * len(stages)
    if n_microbatches < num_total_stages:
        logger.warning(
            f"Number of microbatches ({n_microbatches}) is less than the total number "
            f"of stages ({num_total_stages}) which may result in a bubble in the pipeline."
        )

    # pyrefly: ignore [bad-instantiation]
    schedule = schedule_class(
        # pyrefly: ignore [bad-argument-type]
        stages if looped_schedule else stages[0],
        n_microbatches=n_microbatches,
        loss_fn=loss_fn,
        scale_grads=False,
    )
    logger.info(
        f"Using pipeline schedule {parallelism.pipeline_parallel_schedule} "
        f"with {n_microbatches} microbatches and {num_total_stages} stages."
    )

    if pp_schedule_csv:
        assert schedule_class in [
            PipelineScheduleSingle,
            PipelineScheduleMulti,
            _PipelineScheduleRuntime,
        ], (
            "Only PipelineScheduleSingle (single stage), PipelineScheduleMulti (multistage), "
            "and _PipelineScheduleRuntime support csv schedules"
        )
        # pyrefly: ignore [missing-attribute]
        schedule._load_csv(pp_schedule_csv)

    return schedule


def generate_llm_fqn_per_model_part(
    num_stages: int,
    num_layers: int,
    input_weight: int = 1,
    output_weight: int = 1,
) -> list[list[str]]:
    """
    Programmatically generates module names model part, focused on LLMs models.

    Args:
        num_stages: Number of pipeline stages
        num_layers: Total number of transformer layers in the model
        input_weight: Weight for input modules (tok_embeddings) in layer calculation
        output_weight: Weight for output modules (norm + output) in layer calculation

    Returns:
        List of lists containing module names for each model part

    Example:
        generate_llm_fqn_per_model_part(2, 3, input_weight=2, output_weight=2)
        treats embeddings as 2 layers and norm+output as 2 layers for distribution
    """
    if num_stages < 1:
        raise ValueError("Number of stages must be at least 1")

    if num_stages == 1:
        # Single stage gets everything
        layer_names = [f"layers.{i}" for i in range(num_layers)]
        return [["tok_embeddings"] + layer_names + ["norm", "lm_head"]]

    # Calculate effective layers including weights
    num_effective_layers = num_layers + input_weight + output_weight

    if num_stages > num_effective_layers:
        raise ValueError(
            f"Number of stages ({num_stages}) cannot be greater than effective layers ({num_effective_layers})"
        )

    # Calculate layers per stage (distribute evenly)
    layers_per_stage = num_effective_layers // num_stages
    extra_layers = num_effective_layers % num_stages

    # Feasibility check: Ensure at least 1 layer in each PP stage
    if layers_per_stage == 0:
        raise ValueError(
            f"Configuration would result in empty stages. "
            f"With {num_stages} stages and {num_effective_layers} effective layers "
            f"(num_layers={num_layers} + input_weight={input_weight} + output_weight={output_weight}), "
            f"each stage would get {layers_per_stage} layers on average. "
            f"Reduce num_stages or increase num_layers/weights."
        )

    # Balance check: Ensure weights don't exceed minimum layers per stage
    if input_weight > layers_per_stage:
        raise ValueError(
            f"input_weight ({input_weight}) exceeds minimum layers per stage ({layers_per_stage})."
        )
    if output_weight > layers_per_stage:
        raise ValueError(
            f"output_weight ({output_weight}) exceeds minimum layers per stage ({layers_per_stage})."
        )

    module_names_per_stage = []
    current_layer = 0

    for stage_idx in range(num_stages):
        stage_modules = []

        # Calculate effective layers for this stage
        effective_layers_for_stage = layers_per_stage
        if stage_idx < extra_layers:
            effective_layers_for_stage += 1

        # First stage: handle input modules with weighting
        if stage_idx == 0:
            stage_modules.append("tok_embeddings")
            # Account for input weight in layer distribution
            remaining_layers_for_stage = effective_layers_for_stage - input_weight

            # Add transformer layers
            for _ in range(remaining_layers_for_stage):
                if current_layer < num_layers:
                    stage_modules.append(f"layers.{current_layer}")
                    current_layer += 1

        # Last stage: handle output modules with weighting
        elif stage_idx == num_stages - 1:
            # Account for output weight in layer distribution
            remaining_layers_for_stage = effective_layers_for_stage - output_weight

            # Add transformer layers
            for _ in range(remaining_layers_for_stage):
                if current_layer < num_layers:
                    stage_modules.append(f"layers.{current_layer}")
                    current_layer += 1

            # Add output modules
            stage_modules.extend(["norm", "lm_head"])

        # Middle stages: only transformer layers
        else:
            for _ in range(effective_layers_for_stage):
                if current_layer < num_layers:
                    stage_modules.append(f"layers.{current_layer}")
                    current_layer += 1

        module_names_per_stage.append(stage_modules)

    return module_names_per_stage


def pipeline_module_split(
    whole_model: nn.Module,
    pp_mesh: DeviceMesh,
    pp_schedule: str,
    device: torch.device,
    module_names_per_stage: list[list[str]],
    get_mesh: Callable | None = None,
    fsdp_overlap_policy: str = "bulk",
    native_ddp: bool = False,
) -> tuple[list[PipelineStage], list[nn.Module]]:
    """
    This API creates pipeline stages based on specified module names for each stage.

    Some model restrictions include:
    - forward() method should tolerate deleted layers
    - weight initialization methods should tolerate deleted layers
    - Does not support nested moduledict and modulelist structures

    Args:
        whole_model: The complete model to be split
        pp_mesh: Pipeline parallel device mesh
        pp_schedule: Name of pipeline parallelism schedule
        device: Device
        module_names_per_stage: List of lists, where each inner list contains the module names
                               that should be included in that stage. Module names should be
                               dot-separated paths. Examples:
                               - "tok_embeddings" for token embeddings
                               - "layers.0", "layers.1" for specific transformer layers
                               - "norm" for the final normalization layer
                               - "lm_head" for the output projection layer

    Returns:
        Tuple of (stages, models) where stages are PipelineStage objects and models are the
        corresponding model chunks

    Example usage:
        module_names_per_stage = [
            ["tok_embeddings", "layers.0"],     # Stage 0: embeddings + first layer
            ["layers.1", "layers.2"],           # Stage 1: middle layers
            ["norm", "lm_head"]                  # Stage 2: final norm + output
        ]
    """
    pp_rank = pp_mesh.get_local_rank()
    pp_degree = pp_mesh.size()

    def _build_stage_from_modules(
        stage_idx: int, module_names: list[str], num_stages: int
    ) -> tuple[PipelineStage, nn.Module]:
        model = copy.deepcopy(whole_model)

        # Create a set of modules to keep for faster lookup
        modules_to_keep = set(module_names)
        for module_name, module_value in model.named_children():
            # Handle layer-like structures (e.g., "layers.0", "layers.1")
            if isinstance(module_value, (nn.ModuleDict, nn.ModuleList)):
                layers_to_keep = {
                    name.split(".", 1)[1]
                    for name in modules_to_keep
                    if name.startswith(f"{module_name}.")
                }
                if layers_to_keep:
                    # Keep only specified layers
                    if isinstance(module_value, nn.ModuleDict):
                        for layer_name in list(module_value.keys()):
                            if layer_name not in layers_to_keep:
                                del module_value[layer_name]
                    elif isinstance(module_value, nn.ModuleList):
                        indices_to_keep = {
                            int(idx) for idx in layers_to_keep if idx.isdigit()
                        }
                        new_layers = ModuleList(
                            [
                                layer
                                for i, layer in enumerate(module_value)
                                if i in indices_to_keep
                            ]
                        )
                        setattr(model, module_name, new_layers)
                else:
                    # No layers from this structure needed, set to empty structure
                    if isinstance(module_value, nn.ModuleDict):
                        setattr(model, module_name, ModuleDict())
                    elif isinstance(module_value, nn.ModuleList):
                        setattr(model, module_name, ModuleList())
            # Handle simple module attributes (e.g., "linear", "norm")
            elif module_name not in modules_to_keep:
                # Replace with None
                setattr(model, module_name, None)

        stage_classes = {
            "bulk": PipelineStage,
            "deferred": _OverlappedFSDPPipelineStage,
            "pp_first": _PPFirstFSDPPipelineStage,
        }
        if fsdp_overlap_policy not in stage_classes:
            raise ValueError(
                f"unsupported pipeline FSDP overlap policy {fsdp_overlap_policy!r}"
            )
        stage_class = stage_classes[fsdp_overlap_policy]
        if native_ddp:
            if fsdp_overlap_policy != "bulk":
                raise ValueError("native DDP does not use pipeline FSDP overlap")
            stage_class = _NativeDDPPipelineStage
        stage = stage_class(
            model,
            stage_idx,
            num_stages,
            device,
            group=pp_mesh.get_group("pp"),
            get_mesh=get_mesh,
        )
        return stage, model

    num_stages = len(module_names_per_stage)
    stages = []
    models = []

    schedule_class = get_schedule_class(pp_schedule)
    style = (
        "v" if schedule_class in (ScheduleZBVZeroBubble, ScheduleDualPipeV) else "loop"
    )

    def _get_stage_indices() -> tuple[int, ...]:
        """
        Compute the stage ids for the stages that will run on this pp rank
        for either a looped or V style schedule
        """
        assert (
            num_stages % pp_degree == 0
        ), f"num_stages {num_stages} must be evenly divisible by pp_degree {pp_degree}"
        stages_per_rank = num_stages // pp_degree
        if style == "loop":
            return tuple(pp_rank + s * pp_degree for s in range(stages_per_rank))
        elif style == "v":
            assert (
                stages_per_rank == 2
            ), f"v schedules assume 2 stages per rank, got {stages_per_rank}"
            stage_v_pairs = list(
                zip(range(pp_degree), range(num_stages - 1, pp_degree - 1, -1))
            )
            return stage_v_pairs[pp_rank]
        else:
            raise ValueError(f"Unknown style {style}")

    for stage_idx in _get_stage_indices():
        module_names = module_names_per_stage[stage_idx]
        stage, model_chunk = _build_stage_from_modules(
            stage_idx,
            module_names,
            num_stages,
        )
        logger.info(
            f"PP rank {pp_rank} is building stage_idx {stage_idx} "
            f"with modules {module_names}"
        )
        stages.append(stage)
        models.append(model_chunk)

    if fsdp_overlap_policy != "bulk":
        logger.info(
            "Enabled %s FSDP gradient overlap for full-backward 1F1B",
            fsdp_overlap_policy,
        )

    return stages, models
