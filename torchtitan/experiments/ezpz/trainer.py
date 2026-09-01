# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import timedelta
from typing import cast

import ezpz

import torch
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.tensor import DTensor

from torchtitan.components.dataloader import DataloaderExhaustedError
from torchtitan.components.loss import ChunkedCELoss, IGNORE_INDEX
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.distributed.fsdp import set_fsdp_gradient_sync
from torchtitan.experiments.ezpz.lr_finder import LRFinderConfig
from torchtitan.experiments.ezpz.native_ddp import (
    get_agpt_dtype_probe_data,
    get_native_ddp_logging_data,
    install_agpt_dtype_probe,
    maybe_rebuild_native_ddp_pipeline_buckets,
    record_native_ddp_grad_streams,
    validate_native_ddp,
    validate_native_ddp_pipeline_checkpoint_state,
    wrap_native_ddp,
    wrap_native_ddp_pipeline_stage,
)
from torchtitan.experiments.ft.config.job_config import FaultTolerance
from torchtitan.experiments.ft.manager import FTManager, maybe_semi_sync_training
from torchtitan.experiments.ft.optimizer import FTOptimizersContainer
from torchtitan.protocols import BaseModel
from torchtitan.tools import utils
from torchtitan.tools.logging import logger
from torchtitan.tools.profiler import Profiler
from torchtitan.tools.xpu_phase_timer import (
    install_fsdp_phase_timer_hooks,
    validate_phase_timer_training_mode,
    XPUPhaseTimer,
)
from torchtitan.trainer import Trainer


class FaultTolerantTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        fault_tolerance: FaultTolerance = field(default_factory=FaultTolerance)
        lr_finder: LRFinderConfig = field(default_factory=LRFinderConfig)
        phase_timer: XPUPhaseTimer.Config = field(default_factory=XPUPhaseTimer.Config)

    ft_manager: FTManager

    @record
    def __init__(self, config: Config):
        torch._C._log_api_usage_once("torchtitan.train")

        self.config = config
        assert config.model_spec is not None, (
            "model_spec must be set before creating Trainer"
        )
        model_spec = config.model_spec
        dist_utils.validate_pure_model_parallel_model(
            model_spec.name, config.parallelism
        )
        validate_phase_timer_training_mode(
            enabled=config.phase_timer.enable,
            native_ddp=config.parallelism.enable_data_parallel_native_ddp,
            lr_finder=config.lr_finder.enable,
            seed_checkpoint=config.checkpoint.create_seed_checkpoint,
        )

        device_module, device_type = utils.device_module, utils.device_type
        # pyrefly: ignore [read-only]
        self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        # Device has to be set before creating TorchFT manager.
        device_module.set_device(self.device)

        # init distributed and build meshes (FT override handles ft_manager creation)
        self.parallel_dims = parallel_dims = self.init_distributed()

        # Logging needs to happen after distributed initialized
        config.maybe_log()

        if parallel_dims.dp_enabled:
            batch_mesh = parallel_dims.get_mesh("batch")
            batch_degree, batch_rank = batch_mesh.size(), batch_mesh.get_local_rank()
        else:
            batch_degree, batch_rank = 1, 0

        # FT addition: adjust dp info via ft_manager
        batch_degree, batch_rank = self.ft_manager.get_dp_info(batch_degree, batch_rank)

        # take control of garbage collection to avoid stragglers
        self.gc_handler = utils.GarbageCollection(
            gc_freq=config.training.gc_freq, debug=config.training.gc_debug
        )

        # Set random seed, and maybe enable deterministic mode
        # (mainly for debugging, expect perf loss).
        dist_utils.set_determinism(
            parallel_dims,
            self.device,
            config.debug,
            distinct_seed_mesh_dims=["pp"],
        )

        # build tokenizer
        self.tokenizer = (
            config.tokenizer.build(tokenizer_path=config.hf_assets_path)
            if config.tokenizer is not None
            else None
        )

        # build dataloader
        self.dataloader = config.dataloader.build(
            dp_world_size=batch_degree,
            dp_rank=batch_rank,
            tokenizer=self.tokenizer,
            seq_len=config.training.seq_len,
            local_batch_size=config.training.local_batch_size,
            training_steps=config.training.steps,
            global_batch_size=config.training.global_batch_size,
            parallel_dims=parallel_dims,
        )

        # build model (using meta init)
        model_config = model_spec.model
        # set the model args from training job configs
        model_config.update_from_config(
            trainer_config=config,
        )
        self.model_config = model_config

        # logger.info(
        #     f"Building {model_spec.name} {model_spec.flavor} "
        #     f"with {json.dumps(model_config.to_dict(), indent=2, ensure_ascii=False)}"
        # )
        with (
            torch.device("meta"),
            utils.set_default_dtype(TORCH_DTYPE_MAP[config.training.dtype]),
        ):
            model = model_config.build()

        # Quantization is now applied to the config at model_registry time
        # (#3127). The runtime model_converters layer is gone.

        # Verify all submodules satisfy the Module protocol
        model.verify_module_protocol()

        # Check if any quantization converter is on the model_config
        from torchtitan.components.quantization.utils import has_quantization as _has_quantization
        has_quantization = _has_quantization(model_config)

        # metrics logging (FT addition: ft_enable, ft_replica_id)
        self.metrics_processor = config.metrics.build(
            parallel_dims=parallel_dims,
            dump_folder=config.dump_folder,
            pp_schedule=config.parallelism.pipeline_parallel_schedule,
            ft_enable=config.fault_tolerance.enable,
            ft_replica_id=config.fault_tolerance.replica_id,
            config_dict=config.to_dict(),
            has_quantization=has_quantization,
        )
        color = self.metrics_processor.color

        # calculate model size and flops per token
        (
            model_param_count,
            self.metrics_processor.num_flops_per_token,
        ) = model_config.get_nparams_and_flops(model, config.training.seq_len)

        heading = 80 * "="
        logger.info(
            "\n".join([
                "\n",
                f"{heading}",
                f"{color.blue}Model: {model_spec.name} {model_spec.flavor} ",
                f"{color.red}config: {model_param_count:,} total parameters{color.reset}",
                f"{heading}",
                "\n",
            ])
        )
        # logger.info(
        #     "\n" + 80 * "="
        #     f"{color.blue}Model {model_spec.name} {model_spec.flavor} "
        #     f"{color.red}size: {model_param_count:,} total parameters{color.reset}"
        #
        # )

        # move sharded model to CPU/GPU and initialize weights via DTensor
        buffer_device: torch.device | None
        if config.checkpoint.create_seed_checkpoint:
            init_device = "cpu"
            buffer_device = None
        elif config.training.enable_cpu_offload:
            init_device = "cpu"
            buffer_device = torch.device(device_type)
        else:
            init_device = device_type
            buffer_device = None

        # Loss is now built from the JobConfig.loss field (upstream #2937 /
        # ChunkedCELoss). The FT integration no longer wraps the loss
        # function — FTOptimizersContainer below still passes ft_manager
        # for gradient sync.
        self.loss_fn = config.loss.build(compile_config=config.compile)

        # verify batch sizes
        global_batch_size = config.training.global_batch_size
        if global_batch_size < 0:
            # This global batch size results in 1 gradient accumulation
            # step.
            global_batch_size = config.training.local_batch_size * batch_degree
        assert global_batch_size > 0
        assert (
            global_batch_size % (config.training.local_batch_size * batch_degree) == 0
        ), (
            f"global batch size must be multiple of local batch size times "
            f"data-parallel degree ({global_batch_size} "
            f"% ({config.training.local_batch_size} * {batch_degree}) != 0)"
        )

        # calculate gradient accumulation steps
        self.gradient_accumulation_steps = global_batch_size // (
            config.training.local_batch_size * batch_degree
        )
        assert self.gradient_accumulation_steps > 0
        validate_native_ddp(
            model_name=model_spec.name,
            parallel_dims=parallel_dims,
            training=config.training,
            parallelism=config.parallelism,
            loss_fn=self.loss_fn,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            fault_tolerance_enabled=self.ft_manager.enabled,
            create_seed_checkpoint=config.checkpoint.create_seed_checkpoint,
            optimizer_has_param_groups=bool(
                getattr(config.optimizer, "param_groups", [])
            ),
        )
        if (
            (
                config.parallelism.pipeline_parallel_fsdp_overlap
                or config.parallelism.pipeline_parallel_fsdp_overlap_policy
                != "bulk"
            )
            and self.gradient_accumulation_steps != 1
        ):
            raise ValueError(
                "pipeline FSDP overlap currently requires exactly one "
                "pipeline schedule invocation per optimizer step; gradient "
                f"accumulation steps is {self.gradient_accumulation_steps}"
            )

        # apply parallelisms and initialization
        if parallel_dims.pp_enabled:
            from torchtitan.components.metrics import ensure_pp_loss_visible

            if not model_spec.pipelining_fn:
                raise RuntimeError(
                    f"Pipeline Parallel is enabled but {model_spec.name} "
                    f"does not support pipelining"
                )

            # apply both PT-D Pipeline Parallel and SPMD-style PT-D techniques
            (
                self.pp_schedule,
                self.model_parts,
                self.pp_has_first_stage,
                self.pp_has_last_stage,
            ) = model_spec.pipelining_fn(
                model,
                parallel_dims=parallel_dims,
                training=config.training,
                parallelism=config.parallelism,
                compile_config=config.compile,
                ac_config=config.activation_checkpoint,
                dump_folder=config.dump_folder,
                device=self.device,
                model_config=model_config,
                parallelize_fn=model_spec.parallelize_fn,
                loss_fn=self.loss_fn,
            )
            # when PP is enabled, `model` obj is no longer used after this point,
            # model_parts is used instead
            del model

            for m in self.model_parts:
                m.to_empty(device=init_device)
                with torch.no_grad():
                    cast(BaseModel, m).init_states(buffer_device=buffer_device)
                m.train()

            if config.parallelism.enable_data_parallel_native_ddp:
                wrap_native_ddp_pipeline_stage(
                    self.model_parts,
                    self.pp_schedule,
                    self.loss_fn,
                    parallel_dims.get_mesh("dp_replicate"),
                    config.parallelism.native_ddp_bucket_cap_mb,
                    parallel_dims.dp_replicate,
                    config.parallelism.native_ddp_compute_policy,
                )
                logger.info(
                    "Applied native DDP to the sole local 1F1B stage with %.1f "
                    "MiB gradient buckets and %s compute",
                    config.parallelism.native_ddp_bucket_cap_mb,
                    config.parallelism.native_ddp_compute_policy,
                )

            # confirm that user will be able to view loss metrics on the console
            ensure_pp_loss_visible(
                parallel_dims=parallel_dims,
                pp_schedule=config.parallelism.pipeline_parallel_schedule,
                color=color,
            )
        else:
            # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
            model = model_spec.parallelize_fn(
                model,
                parallel_dims=parallel_dims,
                training=config.training,
                parallelism=config.parallelism,
                compile_config=config.compile,
                ac_config=config.activation_checkpoint,
                dump_folder=config.dump_folder,
            )

            model.to_empty(device=init_device)
            with torch.no_grad():
                cast(BaseModel, model).init_states(buffer_device=buffer_device)
            model.train()

            if config.parallelism.enable_data_parallel_native_ddp:
                model = wrap_native_ddp(
                    model,
                    parallel_dims.get_mesh("dp_replicate"),
                    config.parallelism.native_ddp_bucket_cap_mb,
                    config.parallelism.native_ddp_compute_policy,
                    config.parallelism.native_ddp_bucketize_first_iteration,
                )
                logger.info(
                    "Applied native DDP with %.1f MiB gradient buckets, %s "
                    "compute, and first-iteration bucketization=%s",
                    config.parallelism.native_ddp_bucket_cap_mb,
                    config.parallelism.native_ddp_compute_policy,
                    config.parallelism.native_ddp_bucketize_first_iteration,
                )

            self.model_parts = [model]

        self.phase_timer = config.phase_timer.build(device=self.device)
        install_fsdp_phase_timer_hooks(self.model_parts, self.phase_timer)

        if os.getenv("TORCHTITAN_AGPT_DTYPE_PROBE") == "1":
            if parallel_dims.pp_enabled or len(self.model_parts) != 1:
                raise ValueError("AGPT dtype probe requires a non-pipeline model")
            install_agpt_dtype_probe(self.model_parts[0])

        # Set lm_head reference for ChunkedCELoss after model construction.
        # Replayed from upstream torchtitan/trainer.py (lines 391-411). Required
        # whenever loss=ChunkedCELoss.Config(...) — the loss object computes
        # logits from hidden states in chunks, so it needs a handle to lm_head
        # and signals the model to skip its own lm_head pass via _skip_lm_head.
        # Non-PP: single model part always has lm_head.
        # PP: only the last stage has lm_head; non-last stages skip this.
        if isinstance(self.loss_fn, ChunkedCELoss):
            if parallel_dims.pp_enabled:
                if self.pp_has_last_stage:
                    lm_head = self.model_parts[-1].lm_head
                    assert (
                        lm_head is not None
                    ), "Last PP stage must have lm_head for ChunkedCELoss"
                    self.loss_fn.set_lm_head(lm_head)
                    self.model_parts[-1]._skip_lm_head = True
            else:
                assert len(self.model_parts) == 1
                lm_head = self.model_parts[0].lm_head
                assert lm_head is not None, "Model must have lm_head for ChunkedCELoss"
                self.loss_fn.set_lm_head(lm_head)
                self.model_parts[0]._skip_lm_head = True

        # FT addition: set all reduce hook
        self.ft_manager.maybe_set_all_reduce_hook(self.model_parts)

        # initialize device memory monitor and get peak flops for MFU calculation
        device_memory_monitor = self.metrics_processor.device_memory_monitor
        gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
        logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")
        device_mem_stats = device_memory_monitor.get_peak_stats()
        logger.info(
            f"{device_type.upper()} memory usage for model: "
            f"{device_mem_stats.max_reserved_gib:.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)"
        )

        # build optimizer after applying parallelisms to the model
        # FT addition: pass ft_manager for FTOptimizersContainer
        if isinstance(config.optimizer, FTOptimizersContainer.Config):
            self.optimizers = config.optimizer.build(
                model_parts=self.model_parts, ft_manager=self.ft_manager
            )
        else:
            self.optimizers = config.optimizer.build(model_parts=self.model_parts)
        if model_spec.post_optimizer_build_fn is not None:
            model_spec.post_optimizer_build_fn(
                self.optimizers, self.model_parts, parallel_dims
            )
        self.lr_schedulers = config.lr_scheduler.build(
            optimizers=self.optimizers,
            training_steps=config.training.steps,
        )
        # The post-optimizer model_converters hook is gone in #3127 —
        # quantization is applied to the config and runs as part of
        # forward, not via a runtime post-step hook.
        self.metrics_processor.optimizers = self.optimizers
        self.metrics_processor.model_parts = self.model_parts

        # Initialize trainer states that will be saved in checkpoint.
        # These attributes must be initialized before checkpoint loading.
        self.step = 0
        self.ntokens_seen = 0

        # Build checkpoint manager.
        # When fault tolerance is enabled and config.checkpoint uses
        # FTCheckpointManager.Config, ft_manager is passed through.
        # Otherwise the base CheckpointManager is used without it.
        ckpt_kwargs: dict = dict(
            dataloader=self.dataloader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states={"train_state": self},
            sd_adapter=(
                model_spec.state_dict_adapter(model_config, config.hf_assets_path)
                if model_spec.state_dict_adapter
                else None
            ),
            base_folder=config.dump_folder,
        )
        # FTCheckpointManager accepts ft_manager; base CheckpointManager does not
        from torchtitan.experiments.ft.checkpoint import FTCheckpointManager

        if isinstance(config.checkpoint, FTCheckpointManager.Config):
            ckpt_kwargs["ft_manager"] = self.ft_manager
        self.checkpointer = config.checkpoint.build(**ckpt_kwargs)

        loss_parallel_enabled = (
            parallel_dims.tp_enabled and not config.parallelism.disable_loss_parallel
        )
        self.train_context = dist_utils.get_train_context(
            loss_parallel_enabled,
            enable_bf16_autocast=dist_utils.should_enable_bf16_autocast(
                config.parallelism
            ),
        )

        # Build validator if validation is configured
        if config.validator.enable:
            pp_schedule, pp_has_first_stage, pp_has_last_stage = (
                (
                    self.pp_schedule,
                    self.pp_has_first_stage,
                    self.pp_has_last_stage,
                )
                if parallel_dims.pp_enabled
                else (None, None, None)
            )

            self.validator = config.validator.build(
                parallelism=config.parallelism,
                job_config=config,
                dp_world_size=batch_degree,
                dp_rank=batch_rank,
                tokenizer=self.tokenizer,
                parallel_dims=parallel_dims,
                loss_fn=self.loss_fn,
                validation_context=self.train_context,
                metrics_processor=self.metrics_processor,
                seq_len=config.training.seq_len,
                local_batch_size=config.training.local_batch_size,
                pp_schedule=pp_schedule,
                pp_has_first_stage=pp_has_first_stage,
                pp_has_last_stage=pp_has_last_stage,
            )

        logger.info(
            "Trainer is initialized with "
            f"local batch size {config.training.local_batch_size}, "
            f"global batch size {global_batch_size}, "
            f"gradient accumulation steps {self.gradient_accumulation_steps}, "
            f"sequence length {config.training.seq_len}, "
            f"total steps {config.training.steps} "
            f"(warmup {config.lr_scheduler.warmup_steps})"
        )

    def init_distributed(self) -> ParallelDims:
        config = self.config

        # determine the global ranks when fault tolerance is enabled
        global_ranks = []
        ft_config = config.fault_tolerance
        if ft_config.enable:
            group_size = ft_config.group_size
            replica_id = ft_config.replica_id
            first_rank = replica_id * group_size
            last_rank = first_rank + group_size - 1
            global_ranks = list(range(first_rank, last_rank + 1))

        # init distributed and build meshes
        dist_utils.init_distributed(
            config.comm,
            enable_cpu_backend=config.training.enable_cpu_offload,
            base_folder=config.dump_folder,
            ranks=global_ranks,
        )

        # FT addition: build FTManager
        self.ft_manager = config.fault_tolerance.build()

        world_size = int(os.environ["WORLD_SIZE"])

        return ParallelDims.from_config(config.parallelism, world_size)

    def train_step(
        self,
        data_iterator: Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]],
        *,
        return_global_loss: bool = False,
    ):
        self.phase_timer.begin_step(self.step)
        self._debug_step_phase("step_start")
        self.optimizers.zero_grad()
        self._debug_step_phase("zero_grad_returned")
        if (
            self.config.parallelism.enable_data_parallel_native_ddp
            and self.parallel_dims.pp_enabled
        ):
            maybe_rebuild_native_ddp_pipeline_buckets(self.model_parts[0])
        # Save the current step learning rate for logging
        lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]

        # Keep these variables local to shorten the code as these are
        # the major variables that are used in the training loop.
        parallel_dims = self.parallel_dims

        # Collect all microbatches on CPU and count total valid tokens
        microbatches = []
        local_valid_tokens = torch.tensor(0, dtype=torch.int64)
        for _microbatch in range(self.gradient_accumulation_steps):
            input_dict, labels = next(data_iterator)
            local_valid_tokens += (labels != IGNORE_INDEX).sum()
            microbatches.append((input_dict, labels))
        self._debug_step_phase("data_ready")

        # All-reduce to get global token count across DP ranks
        # Move to GPU for distributed communication
        local_valid_tokens = local_valid_tokens.to(self.device)
        if parallel_dims.dp_enabled:
            batch_mesh = parallel_dims.get_mesh("batch")
            global_valid_tokens = dist_utils.dist_sum_tensor(
                local_valid_tokens, batch_mesh
            )
        else:
            global_valid_tokens = local_valid_tokens.float()
        self._debug_step_phase("token_count_returned")

        # Process each microbatch: move to GPU, forward/backward, then free
        accumulated_losses = []
        for microbatch_idx, (input_dict, labels) in enumerate(microbatches):
            if (
                self.gradient_accumulation_steps > 1
                and parallel_dims.dp_cp_enabled
                and not parallel_dims.pp_enabled
            ):
                set_fsdp_gradient_sync(
                    self.model_parts,
                    microbatch_idx == self.gradient_accumulation_steps - 1,
                )
            # Move tensors to GPU
            for k, v in input_dict.items():
                if isinstance(v, torch.Tensor):
                    input_dict[k] = v.to(self.device)
            labels = labels.to(self.device)

            self._debug_step_phase("forward_backward_start")
            loss = self.forward_backward_step(
                input_dict=input_dict,
                labels=labels,
                # pyrefly: ignore [bad-argument-type]
                global_valid_tokens=global_valid_tokens,
            )
            self._debug_step_phase("forward_backward_returned")
            accumulated_losses.append(loss.detach())

        self.phase_timer.mark_grad_ready()
        if self.config.parallelism.enable_data_parallel_native_ddp:
            # Experimental mixed-precision DDP restores FP32 gradients on an
            # upcast stream; ordinary DDP/autocast makes this helper a no-op.
            record_native_ddp_grad_streams(self.model_parts[0])
        self._debug_step_phase("grad_norm_start")
        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.config.training.max_norm,
            foreach=True,
            pp_mesh=parallel_dims.get_optional_mesh("pp"),
            ep_enabled=parallel_dims.ep_enabled,
        )
        self._debug_step_phase("grad_norm_returned")
        self.phase_timer.mark_norm_done()
        should_log = self.metrics_processor.should_log(self.step)
        logged_grad_norm = grad_norm
        if self.config.parallelism.enable_data_parallel_native_ddp:
            if should_log:
                logged_grad_norm = grad_norm.detach().clone()
            if (
                should_log
                and os.getenv("TORCHTITAN_NATIVE_DDP_PRE_OPT_NORM") == "1"
            ):
                print(
                    "NATIVE_DDP_PRE_OPT_NORM "
                    f"rank={torch.distributed.get_rank()} step={self.step} "
                    f"value={float(logged_grad_norm.item()):.9g}",
                    flush=True,
                )
        self.checkpointer.maybe_wait_for_staging()
        self.phase_timer.mark_optimizer_start()
        self._debug_step_phase("optimizer_start")
        self.optimizers.step()
        self._debug_step_phase("optimizer_returned")
        self.phase_timer.mark_optimizer_done()
        self.lr_schedulers.step()
        self.phase_timer.end_step()
        self._debug_step_phase("step_core_returned")
        if (
            self.config.parallelism.enable_data_parallel_native_ddp
            and should_log
            and os.getenv("TORCHTITAN_NATIVE_DDP_PRE_OPT_NORM") == "1"
        ):
            print(
                "NATIVE_DDP_POST_OPT_NORM "
                f"rank={torch.distributed.get_rank()} step={self.step} "
                f"original={float(grad_norm.item()):.9g} "
                f"snapshot={float(logged_grad_norm.item()):.9g}",
                flush=True,
            )

        # Reduce the data collected over gradient accumulation steps.
        loss = torch.sum(torch.stack(accumulated_losses))

        # Normal training does not consume a returned loss. Avoid materializing
        # an XPU scalar on non-logging steps unless the LR finder requests it.
        if not should_log and not return_global_loss:
            self._debug_step_phase("step_returned")
            return None

        self._debug_step_phase("metrics_start")
        if parallel_dims.dp_cp_enabled:
            metrics_loss = loss.detach()
            # The loss can remain a DTensor after TP loss-parallel reduction.
            # Materialize model-parallel placements before reducing the plain
            # scalar over the independent DP/CP loss mesh.  Passing the
            # DTensor directly to dist_sum() intentionally skips its mesh
            # argument to avoid double-reducing Partial placements, which
            # would omit this orthogonal DP reduction.
            if isinstance(metrics_loss, DTensor):
                metrics_loss = metrics_loss.full_tensor()
            # FT addition: use ft_manager.loss_sync_pg for extra process group
            ft_pg = self.ft_manager.loss_sync_pg
            loss_mesh = parallel_dims.get_optional_mesh("loss")

            # For global_avg_loss, we want the average loss across all ranks:
            # loss = local_loss_sum / global_valid_tokens
            # global_avg_loss = sum(local_loss_sum) / global_valid_tokens
            #                 = sum(loss)
            #
            # For global_max_loss, we want the max of local average losses across ranks:
            # local_avg_loss = local_loss_sum / local_valid_tokens
            #                = (loss * global_valid_tokens) / local_valid_tokens
            # global_max_loss = max(local_avg_loss)
            global_avg_loss_tensor = dist_utils.dist_sum_tensor(
                metrics_loss, loss_mesh, ft_pg
            )
            if should_log:
                local_avg_loss = (
                    metrics_loss * global_valid_tokens / local_valid_tokens
                )
                global_max_loss_tensor = dist_utils.dist_max_tensor(
                    local_avg_loss, loss_mesh, ft_pg
                )
                global_ntokens_seen_tensor = dist_utils.dist_sum_tensor(
                    torch.tensor(
                        self.ntokens_seen, dtype=torch.int64, device=self.device
                    ),
                    loss_mesh,
                    ft_pg,
                )
                # Enqueue all independent metric collectives before the first
                # host scalar read. This avoids serial launch/wait/launch
                # behavior on every logging step.
                global_avg_loss = float(global_avg_loss_tensor.item())
                global_max_loss = float(global_max_loss_tensor.item())
                global_ntokens_seen = int(global_ntokens_seen_tensor.item())
            else:
                # Only the LR finder reaches this branch. It explicitly needs a
                # globally reduced Python loss on every step.
                global_avg_loss = float(global_avg_loss_tensor.item())
        else:
            global_avg_loss = float(loss.detach().item())
            if should_log:
                global_max_loss = global_avg_loss
                global_ntokens_seen = self.ntokens_seen

        if should_log:
            extra_metrics = {
                "n_tokens_seen": global_ntokens_seen,
                "lr": lr,
            }
            extra_metrics.update(self.phase_timer.collect_ready_metrics())
            self.metrics_processor.log(
                self.step,
                global_avg_loss,
                global_max_loss,
                float(logged_grad_norm.item()),
                extra_metrics=extra_metrics,
            )
        self._debug_step_phase("metrics_returned")

        if not return_global_loss:
            self._debug_step_phase("step_returned")
            return None
        if parallel_dims.pp_enabled and not self.pp_has_last_stage:
            return None
        return float(global_avg_loss)

    @record
    def train(self):
        config = self.config

        self.checkpointer.load(step=config.checkpoint.load_step)
        if (
            config.parallelism.enable_data_parallel_native_ddp
            and self.parallel_dims.pp_enabled
        ):
            validate_native_ddp_pipeline_checkpoint_state(self.model_parts[0])
        logger.info(f"Training starts at step {self.step + 1}")

        # FT addition: per-replica profiling leaf folder
        leaf_folder = (
            ""
            if not self.ft_manager.enabled
            else f"replica_{self.ft_manager.replica_id}"
        )
        with (
            config.profiler.build(
                global_step=self.step,
                base_folder=config.dump_folder,
                leaf_folder=leaf_folder,
            ) as profiler,
            # FT addition: maybe_semi_sync_training context manager
            maybe_semi_sync_training(
                config.fault_tolerance,
                ft_manager=self.ft_manager,
                model=self.model_parts[0],
                n_layers=(
                    len(self.model_config.layers)
                    if hasattr(self.model_config, "layers")
                    else 0
                ),
                optimizer=self.optimizers,
                fragment_fn=(
                    config.model_spec.fragment_fn
                    if hasattr(config.model_spec, "fragment_fn")
                    else None
                ),
            ),
        ):
            data_iterator = self.batch_generator(self.dataloader)
            while self.should_continue_training():
                self.step += 1
                self.gc_handler.run(self.step)
                try:
                    self.train_step(data_iterator)
                except DataloaderExhaustedError:
                    self.phase_timer.cancel_step()
                    logger.warning("Ran out of data; last step was canceled.")
                    break

                self.checkpointer.save(
                    self.step, last_step=(self.step == config.training.steps)
                )

                # Run validation if validator is available
                if self.config.validator.enable and self.validator.should_validate(
                    self.step
                ):
                    self.validator.validate(self.model_parts, self.step)

                # signal the profiler that the next profiling step has started
                profiler.step()

                # reduce timeout after first train step for faster signal
                # (assuming lazy init and compilation are finished)
                if self.step == 1:
                    dist_utils.set_pg_timeouts(
                        timeout=timedelta(seconds=config.comm.train_timeout_seconds),
                        parallel_dims=self.parallel_dims,
                    )

        rank = torch.distributed.get_rank()
        phase_artifact = self.phase_timer.finalize_to_artifact(
            config.dump_folder, rank
        )
        if phase_artifact is not None and rank == 0:
            logger.info(
                "XPU phase timing wrote bounded rank-local artifacts under %s",
                phase_artifact.parent,
            )

        if os.getenv("TORCHTITAN_AGPT_DTYPE_PROBE") == "1":
            dtype_data = get_agpt_dtype_probe_data(
                self.model_parts[0],
                expected_policy=(
                    "autocast"
                    if dist_utils.should_enable_bf16_autocast(config.parallelism)
                    else "uniform_bfloat16"
                ),
            )
            if torch.distributed.get_rank() == 0:
                logger.info(
                    "DP_DTYPE_PROBE %s", json.dumps(dtype_data, sort_keys=True)
                )

        if config.parallelism.enable_data_parallel_native_ddp:
            print(
                "NATIVE_DDP_STATS "
                + json.dumps(
                    get_native_ddp_logging_data(self.model_parts[0]), sort_keys=True
                ),
                flush=True,
            )

        logger.info("Synchronizing ranks before training shutdown")
        torch.distributed.barrier(device_ids=[self.device.index])

        logger.info("Training completed")
