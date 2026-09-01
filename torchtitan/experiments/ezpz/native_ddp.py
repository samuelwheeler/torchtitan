# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import inspect
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.autograd import Variable
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining.schedules import Schedule1F1B
from torch.distributed.utils import _free_storage
from torch.nn.parallel import DistributedDataParallel
from torch.nn.parallel.distributed import _MixedPrecision

from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims


class NativeDDP(DistributedDataParallel):
    """DDP wrapper that preserves access to the underlying model interface."""

    def _root_copy_hook(self, *args, **kwargs) -> None:
        """Order each low-precision parameter copy after its FP32 producer.

        Upstream DDP launches the FP32-master to low-precision copies on
        ``_mp_stream`` but does not make that stream wait for the current
        stream.  In particular, the next forward can otherwise read a stale
        master while the preceding optimizer update is still executing.
        Module pre-forward events only establish the opposite dependency
        (compute waits for the copy), so the producer dependency belongs here.
        """
        if self.mixed_precision is None:
            return super()._root_copy_hook(*args, **kwargs)
        self._mp_stream.wait_stream(torch.accelerator.current_stream())
        return super()._root_copy_hook(*args, **kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def record_native_ddp_grad_streams(model: nn.Module) -> None:
    """Order and keep mixed-precision DDP gradients on their consumer stream.

    DDP creates the restored FP32 gradients on its upcast stream. The pinned
    asynchronous XPU hook explicitly orders the collective and restoration on
    that stream, but the Python clip/optimizer stream must still wait for it.

    Make the caller's clip/optimizer stream wait for the hook's upcast stream,
    then record that consumer stream on every restored gradient.  The lifetime
    annotation prevents a following ``zero_grad(set_to_none=True)`` from
    releasing and reusing storage while those consumers are still executing.
    This helper must run immediately after backward and before any gradient
    read or optimizer operation.
    """
    if not isinstance(model, NativeDDP):
        raise TypeError(f"expected NativeDDP, got {type(model).__name__}")
    if model.mixed_precision is None:
        return
    if len(model._comm_hooks) != 1:
        raise RuntimeError(
            "mixed-precision NativeDDP requires exactly one communication hook"
        )
    hook_state = model._comm_hooks[0][1]
    upcast_stream = getattr(hook_state, "upcast_stream", None)
    if upcast_stream is None:
        raise RuntimeError("mixed-precision NativeDDP hook has no upcast stream")
    stream = torch.accelerator.current_stream()
    stream.wait_stream(upcast_stream)
    for parameter in model.module.parameters():
        if parameter.grad is not None:
            parameter.grad.record_stream(stream)


_XPU_OVERLAP_TORCH_GIT = "808e7f2bb128dc3fd517bd7883cf3a66070a1607"
_PIPELINE_DDP_TORCH_GIT = _XPU_OVERLAP_TORCH_GIT
_PIPELINE_DDP_SOURCE_SHA256 = {
    "PipelineStage._forward_metadata_inference": (
        "de6da701b5258550ac6a15d80c632a383ed9aceaedfee2d9600acbba0cda2a1e"
    ),
    "PipelineStage.forward_maybe_with_nosync": (
        "16321a4a4ca2f5bab4544efb1258b5f046bd8a651751a9f0539ff631c2180d36"
    ),
    "PipelineStage.backward_maybe_with_nosync": (
        "dfc180daea2ecf5a269e57368ed2060790682985bec4a549ccee5132cb53f258"
    ),
    "DistributedDataParallel._pre_forward": (
        "2350d156509e9df5a2dfa086af59bfee00a9565560c0433f24f392d4d6cddf79"
    ),
}
_PIPELINE_DDP_FILE_SHA256 = {
    "stage.py": "28b2713c0e0c351bc6b24956ee5861bf7ed61ab230acf30a6d23224d7f95bfe8",
    "schedules.py": "b9f2c8e349652524460bd836af354a88615ed347b81c1b52db5a3b996f73163b",
    "distributed.py": (
        "268b4342445697f908c45dcee51d990a86ae224e11e57adbc352427c54047a53"
    ),
}
_XPU_OVERLAP_ENV = {
    "CCL_OP_SYNC": "0",
    "CCL_ZE_DEPS_SYNC": "0",
    "CCL_SYCL_OUTPUT_EVENT": "1",
}
_UNSAFE_XPU_OVERLAP_ENV = "TORCHTITAN_UNSAFE_ENABLE_BROKEN_XPU_DDP_MP"

_AGPT_DTYPE_POLICIES = {
    "uniform_bfloat16": {
        "tok_embedding_weight": "bfloat16",
        "tok_embedding_output": "bfloat16",
        "block0_input": "bfloat16",
        "attention_norm_weight": "bfloat16",
        "attention_norm_output": "bfloat16",
        "final_norm_output": "bfloat16",
        "lm_head_weight": "bfloat16",
        "lm_head_input": "bfloat16",
        "lm_head_output": "bfloat16",
    },
    # XPU autocast leaves embedding/RMSNorm/residual tensors and stored weights
    # in FP32 while autocasting eligible linear algebra, including the LM head,
    # to BF16. This is deliberately not described as BF16 parameter parity with
    # FSDP: it is the correctness contract for ordinary DDP's safe fallback.
    "autocast": {
        "tok_embedding_weight": "float32",
        "tok_embedding_output": "float32",
        "block0_input": "float32",
        "attention_norm_weight": "float32",
        "attention_norm_output": "float32",
        "final_norm_output": "float32",
        "lm_head_weight": "float32",
        "lm_head_input": "float32",
        "lm_head_output": "bfloat16",
    },
}


def _validate_builtin_mp_runtime(device: torch.device) -> None:
    """Reject the built-in XPU MP hook when XCCL is asynchronous.

    The pinned built-in hook launches its collective before entering the FP32
    upcast stream and relies on ``Future.wait()`` for ordering. With
    ``CCL_OP_SYNC=0``, a real 8.02 GB first-iteration bucket demonstrated that
    host-future readiness can precede physical collective completion: every
    rank copied a zero gradient on step one and the preceding gradient later.
    The exact-version custom hook keeps the entire sequence on one stream.
    """
    if device.type == "xpu" and os.getenv("CCL_OP_SYNC") != "1":
        raise RuntimeError(
            "built-in DDP mixed precision on XPU requires CCL_OP_SYNC=1; "
            "use ordinary DDP with the autocast policy for asynchronous XCCL"
        )


def _validate_xpu_overlap_runtime(device: torch.device, process_group) -> None:
    """Reject the private-hook experiment outside its proven Aurora envelope."""
    if device.type != "xpu":
        raise ValueError("DDP XPU overlap policy requires an XPU model")
    if os.getenv(_UNSAFE_XPU_OVERLAP_ENV) != "1":
        raise RuntimeError(
            "ddp_mixed_precision_xpu_overlap is disabled: full-model Aurora "
            "runs consumed zero/stale first-step gradients and the device-aware "
            "Future variant also serialized execution; use autocast, or set "
            f"{_UNSAFE_XPU_OVERLAP_ENV}=1 only for an isolated diagnostic"
        )
    if torch.version.git_version != _XPU_OVERLAP_TORCH_GIT:
        raise RuntimeError(
            "DDP XPU overlap policy is pinned to torch git "
            f"{_XPU_OVERLAP_TORCH_GIT}, got {torch.version.git_version}"
        )
    if dist.get_backend(process_group) != "xccl":
        raise RuntimeError("DDP XPU overlap policy requires the XCCL backend")
    invalid = {
        key: os.environ.get(key)
        for key, expected in _XPU_OVERLAP_ENV.items()
        if os.environ.get(key) != expected
    }
    if invalid:
        raise RuntimeError(
            "DDP XPU overlap policy communication environment mismatch: "
            f"{invalid}; required={_XPU_OVERLAP_ENV}"
        )
    forbidden = {
        key: os.environ[key]
        for key in ("CCL_ATL_SYNC_COLL", "TORCH_XCCL_BLOCKING_WAIT")
        if key in os.environ
    }
    if forbidden:
        raise RuntimeError(
            "DDP XPU overlap policy requires blocking controls to be unset: "
            f"{forbidden}"
        )


def _xpu_overlap_allreduce_and_upcast_hook(
    hook_state, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """Launch FP32 XCCL reduction on the upcast stream without blocking autograd.

    The installed PyTorch hook calls ``fut.wait()`` before returning.  On the
    pinned XPU/XCCL stack that is a host-visible collective wait.  Conversely,
    an ``async_op=False`` all-reduce issued on a nondefault XPU stream with
    ``CCL_OP_SYNC=0`` returns after enqueueing work.  The existing once-per-
    backward engine callback supplies the final compute-stream dependency.
    """
    ddp = hook_state.ddp_weakref()
    if ddp is None:
        raise RuntimeError("native DDP was destroyed during its communication hook")
    diagnostic_records = getattr(ddp, "_xpu_overlap_hook_stats", None)
    started_ns = time.perf_counter_ns() if diagnostic_records is not None else None
    process_group = ddp.process_group
    compute_stream = torch.accelerator.current_stream()
    stream = hook_state.upcast_stream
    input_buffer = bucket.buffer()
    input_bytes = input_buffer.numel() * input_buffer.element_size()
    buffer = input_buffer
    with stream:
        stream.wait_stream(compute_stream)
        if ddp.mixed_precision.param_dtype != ddp.mixed_precision.reduce_dtype:
            # ``set_buffer`` may release the reducer's last reference while
            # this stream is still reading the old BF16 bucket.
            input_buffer.record_stream(stream)
            buffer = input_buffer.to(ddp.mixed_precision.reduce_dtype)
            bucket.set_buffer(buffer)
        dist.all_reduce(buffer, group=process_group, async_op=False)
        buffer.div_(process_group.size())
        # A device-aware Future records an event on the current upcast stream
        # when it becomes ready. Reducer finalization waits on that event before
        # reusing/copying bucket views; a devices=[] host Future would permit a
        # second writer to race the outstanding FP32 restoration copies.
        result = torch.futures.Future(devices=[buffer.device])

        # Match the pinned upstream hook: restore FP32 masters and enqueue FP32
        # gradient copies after the collective on the same stream.
        for parameter in bucket.parameters():
            parameter.data = parameter._fp_param
            # The low-precision storage was allocated/copied on DDP's MP
            # stream and consumed by backward on the compute stream. Unlike
            # upstream's host-blocking Future wait, this hook has not waited
            # for that compute physically to finish, so make allocator reuse
            # respect the outstanding consumer before releasing the storage.
            parameter._mp_param.record_stream(compute_stream)
            _free_storage(parameter._mp_param)
            old_grad = parameter.grad
            old_grad.record_stream(stream)
            parameter.grad.data = old_grad.to(parameter.data.dtype)

        # This must be the final device-visible operation in the stream context:
        # Future readiness records the tail event consumed by the reducer.
        result.set_result(buffer)

    def wait_for_stream_cb() -> None:
        torch.accelerator.current_stream().wait_stream(stream)
        for _, parameter in ddp.module.named_parameters():
            if hasattr(parameter, "_ddp_mp_hook_state"):
                parameter._ddp_mp_hook_state[1].remove()
                delattr(parameter, "_ddp_mp_hook_state")
            if not parameter.requires_grad and not hasattr(parameter, "_ddp_ignored"):
                parameter.data = parameter._fp_param
        hook_state.wait_for_stream_enqueued = False

    if not hook_state.wait_for_stream_enqueued:
        # This callback is queued before the reducer receives the completed
        # Future. Engine callback FIFO ordering therefore establishes stream
        # completion before reducer finalization consumes the bucket results.
        Variable._execution_engine.queue_callback(wait_for_stream_cb)
        hook_state.wait_for_stream_enqueued = True

    if diagnostic_records is not None:
        diagnostic_records.append(
            {
                "bucket_index": bucket.index(),
                "input_bytes": input_bytes,
                "reduce_bytes": buffer.numel() * buffer.element_size(),
                "host_ns": time.perf_counter_ns() - started_ns,
            }
        )
    return result


class _ScaleGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return value

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output * ctx.scale, None


def scale_native_ddp_loss(loss: torch.Tensor, dp_degree: int) -> torch.Tensor:
    """Preserve the local loss value while compensating DDP gradient averaging."""
    if dp_degree <= 1:
        raise ValueError(f"native DDP requires dp_degree > 1, got {dp_degree}")
    return _ScaleGradient.apply(loss, float(dp_degree))


class _NativeDDPPipelineLoss:
    """Scale only the gradient produced by a pipeline-owned loss."""

    def __init__(self, loss_fn: CrossEntropyLoss, dp_degree: int):
        self.loss_fn = loss_fn
        self.dp_degree = dp_degree

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        loss = self.loss_fn(*args, **kwargs)
        return scale_native_ddp_loss(loss, self.dp_degree)


def validate_native_ddp(
    *,
    model_name: str,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    loss_fn: object,
    gradient_accumulation_steps: int,
    fault_tolerance_enabled: bool,
    create_seed_checkpoint: bool,
    optimizer_has_param_groups: bool,
) -> None:
    if not parallelism.enable_data_parallel_native_ddp:
        return
    if model_name != "ezpz.agpt":
        raise ValueError("native DDP is currently supported only for ezpz.agpt")
    if not parallel_dims.dp_replicate_enabled or parallel_dims.dp_shard != 1:
        raise ValueError(
            "native DDP requires data_parallel_replicate_degree > 1 and "
            "data_parallel_shard_degree = 1"
        )
    if parallel_dims.tp != 1 or parallel_dims.cp != 1:
        raise ValueError("native DDP currently requires TP=CP=1")
    if parallel_dims.ep != 1:
        raise ValueError("native DDP does not support expert parallelism")
    if parallel_dims.pp > 1:
        if parallelism.native_ddp_bucketize_first_iteration:
            raise ValueError(
                "native DDP first-iteration bucketization is not yet supported with PP"
            )
        if parallelism.pipeline_parallel_schedule != "1F1B":
            raise ValueError("native DDP with PP requires Schedule1F1B")
        if parallelism.pipeline_parallel_schedule_csv:
            raise ValueError(
                "native DDP with PP does not support a custom schedule CSV"
            )
        if parallelism.native_ddp_compute_policy != "autocast":
            raise ValueError(
                "native DDP with PP requires BF16 autocast with FP32 parameters, "
                "gradients, and reductions"
            )
    if training.enable_cpu_offload:
        raise ValueError("native DDP does not support CPU offload")
    if training.dtype != "float32" or training.mixed_precision_reduce != "float32":
        raise ValueError("native DDP requires FP32 master parameters and reductions")
    if training.mixed_precision_param != "bfloat16":
        raise ValueError("native DDP currently requires BF16 autocast")
    if not isinstance(loss_fn, CrossEntropyLoss):
        raise ValueError("native DDP currently requires standard CrossEntropyLoss")
    if gradient_accumulation_steps != 1:
        raise ValueError("native DDP currently requires one gradient accumulation step")
    if fault_tolerance_enabled:
        raise ValueError("native DDP does not support fault tolerance")
    if create_seed_checkpoint:
        raise ValueError("native DDP cannot create a seed checkpoint")
    if optimizer_has_param_groups:
        raise ValueError("native DDP does not yet support optimizer param_groups")
    if parallelism.enable_data_parallel_replicate_module:
        raise ValueError("native DDP and ReplicateModule are mutually exclusive")
    if parallelism.enable_fsdp_async_all_reduce:
        raise ValueError("native DDP does not use the FSDP async all-reduce path")
    if (
        parallelism.pipeline_parallel_fsdp_overlap
        or parallelism.pipeline_parallel_fsdp_overlap_policy != "bulk"
    ):
        raise ValueError("native DDP does not use pipeline FSDP overlap")
    if parallelism.native_ddp_bucket_cap_mb <= 0:
        raise ValueError("native_ddp_bucket_cap_mb must be positive")


def wrap_native_ddp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    bucket_cap_mb: float,
    compute_policy: str = "autocast",
    bucketize_first_iteration: bool = False,
) -> NativeDDP:
    policies = (
        "autocast",
        "ddp_mixed_precision",
        "ddp_mixed_precision_xpu_overlap",
    )
    if compute_policy not in policies:
        raise ValueError(f"unsupported native DDP compute policy: {compute_policy}")
    device = next(model.parameters()).device
    device_ids = None if device.type == "cpu" else [device.index]
    mixed_precision = (
        _MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        if compute_policy != "autocast"
        else None
    )
    process_group = dp_mesh.get_group()
    initial_bucket_kwargs = {}
    if bucketize_first_iteration:
        signature = inspect.signature(DistributedDataParallel)
        if "bucket_cap_mb_list" not in signature.parameters:
            raise RuntimeError(
                "native DDP first-iteration bucketization requires "
                "DistributedDataParallel.bucket_cap_mb_list"
            )
        if torch._dynamo.utils.get_optimize_ddp_mode() == "python_reducer":
            raise RuntimeError(
                "native DDP first-iteration bucketization is incompatible "
                "with the Python reducer"
            )
        initial_bucket_kwargs["bucket_cap_mb_list"] = [bucket_cap_mb]

    def validate_initial_buckets(wrapped: NativeDDP) -> None:
        if not bucketize_first_iteration:
            return
        expected = int(bucket_cap_mb * 1024 * 1024)
        if wrapped.bucket_bytes_cap_list != [expected]:
            raise RuntimeError(
                "native DDP initial bucket limit mismatch: "
                f"{wrapped.bucket_bytes_cap_list} != {[expected]}"
            )
        limits = wrapped._bucket_config.compute_bucket_size_limits(
            static_graph=False,
            find_unused_parameters=False,
        )
        if limits != ([expected], [expected]):
            raise RuntimeError(
                "native DDP initial/rebuild bucket semantics mismatch: "
                f"{limits} != {([expected], [expected])}"
            )
    if compute_policy == "ddp_mixed_precision":
        _validate_builtin_mp_runtime(device)
    if compute_policy == "ddp_mixed_precision_xpu_overlap":
        _validate_xpu_overlap_runtime(device, process_group)
        # DDP imports this private hook inside its constructor. Patch only for
        # that synchronous import/registration window, then restore upstream.
        from torch.distributed.algorithms.ddp_comm_hooks import mixed_precision_hooks

        upstream_hook = mixed_precision_hooks._reducer_allreduce_and_upcast_hook
        mixed_precision_hooks._reducer_allreduce_and_upcast_hook = (
            _xpu_overlap_allreduce_and_upcast_hook
        )
        try:
            wrapped = NativeDDP(
                model,
                device_ids=device_ids,
                process_group=process_group,
                broadcast_buffers=False,
                bucket_cap_mb=bucket_cap_mb,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
                mixed_precision=mixed_precision,
                **initial_bucket_kwargs,
            )
        finally:
            mixed_precision_hooks._reducer_allreduce_and_upcast_hook = upstream_hook
        validate_initial_buckets(wrapped)
        if (
            len(wrapped._comm_hooks) != 1
            or wrapped._comm_hooks[0][0]
            is not _xpu_overlap_allreduce_and_upcast_hook
        ):
            raise RuntimeError("DDP did not register the pinned XPU overlap hook")
        wrapped._xpu_overlap_hook_stats = (
            []
            if os.getenv("TORCHTITAN_DDP_XPU_OVERLAP_DIAGNOSTICS") == "1"
            else None
        )
        return wrapped
    wrapped = NativeDDP(
        model,
        device_ids=device_ids,
        process_group=process_group,
        broadcast_buffers=False,
        bucket_cap_mb=bucket_cap_mb,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        mixed_precision=mixed_precision,
        **initial_bucket_kwargs,
    )
    validate_initial_buckets(wrapped)
    return wrapped


def _validate_native_ddp_pipeline_runtime() -> None:
    from torch.distributed.pipelining import PipelineStage
    from torch.distributed.pipelining.schedules import Schedule1F1B

    if torch.version.git_version != _PIPELINE_DDP_TORCH_GIT:
        raise RuntimeError(
            "native DDP with PP is pinned to torch git "
            f"{_PIPELINE_DDP_TORCH_GIT}, got {torch.version.git_version}"
        )
    functions = {
        "PipelineStage._forward_metadata_inference": (
            PipelineStage._forward_metadata_inference
        ),
        "PipelineStage.forward_maybe_with_nosync": (
            PipelineStage.forward_maybe_with_nosync
        ),
        "PipelineStage.backward_maybe_with_nosync": (
            PipelineStage.backward_maybe_with_nosync
        ),
        "DistributedDataParallel._pre_forward": DistributedDataParallel._pre_forward,
    }
    actual_sources = {
        name: hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
        for name, function in functions.items()
    }
    invalid = {
        name: digest
        for name, digest in actual_sources.items()
        if digest != _PIPELINE_DDP_SOURCE_SHA256[name]
    }
    if invalid:
        raise RuntimeError(
            "native DDP with PP installed-source mismatch; re-audit reducer and "
            f"pipeline lifecycle before enabling it: {invalid}"
        )
    source_files = {
        "stage.py": Path(inspect.getsourcefile(PipelineStage)),
        "schedules.py": Path(inspect.getsourcefile(Schedule1F1B)),
        "distributed.py": Path(inspect.getsourcefile(DistributedDataParallel)),
    }
    actual_files = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in source_files.items()
    }
    invalid_files = {
        name: digest
        for name, digest in actual_files.items()
        if digest != _PIPELINE_DDP_FILE_SHA256[name]
    }
    if invalid_files:
        raise RuntimeError(
            "native DDP with PP installed-file mismatch; re-audit private "
            f"pipeline/reducer integration before enabling it: {invalid_files}"
        )


def arm_native_ddp_pipeline_bucket_rebuild(model: nn.Module) -> None:
    """Arm the one-time bucket rebuild after a complete 1F1B schedule."""
    if not isinstance(model, NativeDDP):
        raise TypeError(f"expected NativeDDP, got {type(model).__name__}")
    if model._native_ddp_pipeline_buckets_rebuilt:
        return
    if model._native_ddp_pipeline_rebuild_armed:
        raise RuntimeError(
            "native-DDP pipeline bucket rebuild was armed by more than one "
            "schedule before the next optimizer step"
        )
    model._native_ddp_pipeline_rebuild_armed = True


def maybe_rebuild_native_ddp_pipeline_buckets(model: nn.Module) -> bool:
    """Rebuild once after zero_grad and before the next 1F1B schedule."""
    if not isinstance(model, NativeDDP):
        raise TypeError(f"expected NativeDDP, got {type(model).__name__}")
    if model._native_ddp_pipeline_buckets_rebuilt:
        return False
    if not model._native_ddp_pipeline_rebuild_armed:
        return False
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError(
            "native-DDP pipeline buckets may be rebuilt only after gradients "
            "have been cleared"
        )
    rebuilt = model.reducer._rebuild_buckets()
    if rebuilt is not True or model._has_rebuilt_buckets:
        raise RuntimeError(
            "native-DDP pipeline reducer did not perform the expected one-time "
            "bucket-layout transition: "
            f"return={rebuilt!r}, has_rebuilt_buckets={model._has_rebuilt_buckets!r}"
        )
    # DDP._pre_forward normally mirrors a True reducer transition into this
    # Python flag. PipelineStage keeps every DDP forward under no_sync(), so
    # this guarded out-of-band transition must perform that assignment itself.
    model._has_rebuilt_buckets = True
    rebuilt_buckets = model.reducer._get_zeros_like_grad_buckets()
    if not rebuilt_buckets:
        raise RuntimeError("native-DDP pipeline reducer rebuilt an empty bucket layout")
    model._native_ddp_pipeline_rebuilt_bucket_sizes = tuple(
        bucket.buffer().numel() * bucket.buffer().element_size()
        for bucket in rebuilt_buckets
    )
    model._native_ddp_pipeline_rebuild_armed = False
    model._native_ddp_pipeline_buckets_rebuilt = True
    model._native_ddp_pipeline_rebuild_calls += 1
    if model._native_ddp_pipeline_rebuild_calls != 1:
        raise RuntimeError("native-DDP pipeline buckets were rebuilt more than once")
    return True


def validate_native_ddp_pipeline_checkpoint_state(model: nn.Module) -> None:
    """Require a pristine reducer lifecycle after fresh construction or DCP load."""
    if not isinstance(model, NativeDDP):
        raise TypeError(f"expected NativeDDP, got {type(model).__name__}")
    state = (
        model._native_ddp_pipeline_rebuild_armed,
        model._native_ddp_pipeline_buckets_rebuilt,
        model._native_ddp_pipeline_rebuild_calls,
    )
    if state != (False, False, 0):
        raise RuntimeError(
            "native-DDP pipeline checkpoint load found ambiguous reducer bucket "
            f"lifecycle state: {state}"
        )


def wrap_native_ddp_pipeline_stage(
    model_parts: list[nn.Module],
    pp_schedule: object,
    loss_fn: CrossEntropyLoss,
    dp_mesh: DeviceMesh,
    bucket_cap_mb: float,
    dp_degree: int,
    compute_policy: str = "autocast",
) -> NativeDDP:
    """Wrap one plain 1F1B stage and its schedule-owned loss with native DDP."""
    _validate_native_ddp_pipeline_runtime()
    if type(pp_schedule) is not Schedule1F1B:
        raise TypeError(
            "native DDP with PP requires an exact Schedule1F1B instance, got "
            f"{type(pp_schedule).__name__}"
        )
    if len(model_parts) != 1:
        raise ValueError(
            "native DDP with PP requires exactly one local pipeline stage, got "
            f"{len(model_parts)}"
        )
    stage = pp_schedule._stage
    from torchtitan.distributed.pipeline_parallel import _NativeDDPPipelineStage

    if type(stage) is not _NativeDDPPipelineStage:
        raise TypeError(
            "native DDP with PP requires the metadata-safe pipeline stage, got "
            f"{type(stage).__name__}"
        )
    model = model_parts[0]
    if stage.submod is not model:
        raise RuntimeError(
            "pipeline schedule stage and model_parts do not reference the same module"
        )
    if isinstance(model, DistributedDataParallel):
        raise TypeError("pipeline stage is already wrapped by DDP")
    if pp_schedule._loss_fn is not loss_fn:
        raise RuntimeError(
            "pipeline schedule and trainer do not reference the same loss function"
        )
    if compute_policy != "autocast":
        raise ValueError("native DDP with PP requires the autocast compute policy")

    wrapped = wrap_native_ddp(
        model,
        dp_mesh,
        bucket_cap_mb,
        compute_policy,
    )
    required_reducer_attributes = (
        "_get_zeros_like_grad_buckets",
        "_rebuild_buckets",
    )
    if any(
        not callable(getattr(wrapped.reducer, name, None))
        for name in required_reducer_attributes
    ) or not hasattr(wrapped, "_has_rebuilt_buckets"):
        raise RuntimeError(
            "native-DDP pipeline reducer does not expose the pinned bucket "
            "rebuild contract"
        )
    wrapped._native_ddp_pipeline_rebuild_armed = False
    wrapped._native_ddp_pipeline_buckets_rebuilt = False
    wrapped._native_ddp_pipeline_rebuild_calls = 0
    wrapped._native_ddp_pipeline_rebuilt_bucket_sizes = ()
    model_parts[0] = wrapped
    stage.submod = wrapped
    pp_schedule._loss_fn = _NativeDDPPipelineLoss(loss_fn, dp_degree)
    return wrapped


def _first_tensor(value: object) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            if (tensor := _first_tensor(item)) is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            if (tensor := _first_tensor(item)) is not None:
                return tensor
    return None


def install_agpt_dtype_probe(model: nn.Module) -> None:
    """Record first-forward parameter and activation dtypes for an AGPT model."""
    root = model.module if isinstance(model, NativeDDP) else model
    try:
        first_block = next(iter(root.layers.values()))
        modules = {
            "tok_embedding": root.tok_embeddings,
            "attention_norm": first_block.attention_norm,
            "final_norm": root.norm,
            "lm_head": root.lm_head,
        }
    except (AttributeError, StopIteration) as exc:
        raise TypeError("AGPT dtype probe requires a non-pipeline decoder") from exc
    if any(module is None for module in modules.values()):
        raise TypeError("AGPT dtype probe requires embedding, norm, and LM head modules")

    state: dict[str, str] = {}

    def record(key: str, value: object) -> None:
        if key not in state:
            tensor = _first_tensor(value)
            if tensor is None:
                raise RuntimeError(f"AGPT dtype probe found no tensor for {key}")
            state[key] = str(tensor.dtype).removeprefix("torch.")

    def pre_hook(key: str | None = None, parameter_key: str | None = None):
        def hook(module, args):
            # FSDP installs its pre-forward unshard/cast hook before this
            # diagnostic is registered.  Therefore a parameter read here is
            # the dtype consumed by compute.  Reading it from a post-forward
            # hook is too late: FSDP's earlier post hook may already have
            # resharded/restored the FP32 master.  Ordinary DDP has no such
            # cast hook, so this continues to observe its FP32 storage.
            if parameter_key is not None:
                parameter = next(module.parameters(recurse=False), None)
                if parameter is None:
                    raise RuntimeError(
                        f"AGPT dtype probe found no parameter for {parameter_key}"
                    )
                record(parameter_key, parameter)
            if key is not None:
                record(key, args)

        return hook

    def post_hook(key: str):
        def hook(_module, _args, output):
            record(key, output)

        return hook

    modules["tok_embedding"].register_forward_pre_hook(
        pre_hook(parameter_key="tok_embedding_weight")
    )
    modules["tok_embedding"].register_forward_hook(post_hook("tok_embedding_output"))
    first_block.register_forward_pre_hook(pre_hook("block0_input"))
    modules["attention_norm"].register_forward_pre_hook(
        pre_hook(parameter_key="attention_norm_weight")
    )
    modules["attention_norm"].register_forward_hook(post_hook("attention_norm_output"))
    modules["final_norm"].register_forward_hook(post_hook("final_norm_output"))
    modules["lm_head"].register_forward_pre_hook(
        pre_hook("lm_head_input", "lm_head_weight")
    )
    modules["lm_head"].register_forward_hook(post_hook("lm_head_output"))
    model._agpt_dtype_probe = state


def get_agpt_dtype_probe_data(
    model: nn.Module, expected_policy: str = "uniform_bfloat16"
) -> dict[str, str]:
    try:
        expected = _AGPT_DTYPE_POLICIES[expected_policy]
    except KeyError as exc:
        raise ValueError(
            f"unsupported AGPT dtype probe policy: {expected_policy}"
        ) from exc
    required = expected.keys()
    state = getattr(model, "_agpt_dtype_probe", None)
    if not isinstance(state, dict) or state.keys() != required:
        missing = set(required) - (state.keys() if isinstance(state, dict) else set())
        raise RuntimeError(f"AGPT dtype probe incomplete; missing={sorted(missing)}")
    if invalid := {
        key: (value, expected[key])
        for key, value in state.items()
        if value != expected[key]
    }:
        raise RuntimeError(f"AGPT dtype parity failure: {invalid}")
    return dict(sorted(state.items()))


def get_native_ddp_logging_data(model: nn.Module) -> dict[str, object]:
    """Return reducer timing and rebuilt-bucket evidence for an active DDP model."""
    if not isinstance(model, NativeDDP):
        raise TypeError(f"expected NativeDDP, got {type(model).__name__}")
    data = model._get_ddp_logging_data()
    keys = (
        "rank",
        "world_size",
        "backend_name",
        "bucket_cap_bytes",
        "bucket_sizes",
        "has_rebuilt_buckets",
        "rebuilt_bucket_sizes",
        "rebuilt_per_bucket_param_indices",
        "prev_iteration_grad_ready_order_indices",
        "num_buckets_reduced",
        "total_parameter_size_bytes",
        "iteration",
        "avg_forward_compute_time",
        "avg_backward_compute_time",
        "avg_backward_comm_time",
        "avg_backward_compute_comm_overlap_time",
    )
    result = {key: data[key] for key in keys if key in data}
    if hasattr(model, "_native_ddp_pipeline_rebuild_calls"):
        result["has_rebuilt_buckets"] = int(model._has_rebuilt_buckets)
        result["pipeline_bucket_rebuild_calls"] = (
            model._native_ddp_pipeline_rebuild_calls
        )
        result["pipeline_rebuilt_bucket_sizes"] = list(
            model._native_ddp_pipeline_rebuilt_bucket_sizes
        )
    result["rank"] = dist.get_rank()
    result["world_size"] = dist.get_world_size()
    return result
