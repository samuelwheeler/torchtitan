# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import json
import os
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from typing import Annotated, Any, cast

import torch
import torch.distributed.checkpoint.stateful
import tyro
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.tensor import DTensor

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.dataloader import BaseDataLoader, DataloaderExhaustedError
from torchtitan.components.loss import BaseLoss, ChunkedCELoss, IGNORE_INDEX
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import ensure_pp_loss_visible, MetricsProcessor
from torchtitan.components.optimizer import (
    OptimizersContainer,
    OptimizersInBackwardContainer,
)
from torchtitan.components.quantization.utils import has_quantization
from torchtitan.components.tokenizer import BaseTokenizer, HuggingFaceTokenizer
from torchtitan.components.validate import BaseValidator, Validator
from torchtitan.config import Configurable, TORCH_DTYPE_MAP
from torchtitan.config.configs import (
    ActivationCheckpointConfig,
    CommConfig,
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.distributed.context_parallel import prepare_context_parallel_input
from torchtitan.distributed.fsdp import set_fsdp_gradient_sync
from torchtitan.experiments.ezpz.native_ddp import (
    arm_native_ddp_pipeline_bucket_rebuild,
    get_agpt_dtype_probe_data,
    get_native_ddp_logging_data,
    install_agpt_dtype_probe,
    maybe_rebuild_native_ddp_pipeline_buckets,
    record_native_ddp_grad_streams,
    scale_native_ddp_loss,
    validate_native_ddp,
    validate_native_ddp_pipeline_checkpoint_state,
    wrap_native_ddp,
    wrap_native_ddp_pipeline_stage,
)
from torchtitan.models.common.decoder import Decoder
from torchtitan.protocols import BaseModel
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools import utils
from torchtitan.tools.logging import logger
from torchtitan.tools.profiler import Profiler


class Trainer(torch.distributed.checkpoint.stateful.Stateful, Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """
        Default container for training configuration.
        """

        # NOTE: model_spec is suppressed from tyro CLI parsing and is always
        # set programmatically by the model registry before Trainer construction.
        model_spec: Annotated[ModelSpec | None, tyro.conf.Suppress] = None

        hf_assets_path: str = "./tests/assets/tokenizer"
        """
        Path to HF assets folder. This folder contains local copies of Hugging Face assets,
        including model weights in .safetensors format, the model.safetensor.index.json file
        (fqn to file mapping), the config.json file, generation_config.json, and tokenizer files.
        """

        dump_folder: str = "./outputs"
        """Folder to dump job outputs"""

        profiler: Profiler.Config = field(default_factory=Profiler.Config)
        metrics: MetricsProcessor.Config = field(
            default_factory=MetricsProcessor.Config
        )
        tokenizer: BaseTokenizer.Config = field(
            default_factory=HuggingFaceTokenizer.Config
        )
        dataloader: BaseDataLoader.Config = field(default_factory=BaseDataLoader.Config)
        optimizer: OptimizersContainer.Config = field(
            default_factory=OptimizersContainer.Config
        )
        lr_scheduler: LRSchedulersContainer.Config = field(
            default_factory=LRSchedulersContainer.Config
        )
        training: TrainingConfig = field(default_factory=TrainingConfig)
        parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
        checkpoint: CheckpointManager.Config = field(
            default_factory=CheckpointManager.Config
        )
        activation_checkpoint: ActivationCheckpointConfig = field(
            default_factory=ActivationCheckpointConfig
        )
        compile: CompileConfig = field(default_factory=CompileConfig)
        comm: CommConfig = field(default_factory=CommConfig)
        validator: Validator.Config = field(default_factory=Validator.Config)
        debug: DebugConfig = field(default_factory=DebugConfig)
        loss: BaseLoss.Config = field(default_factory=BaseLoss.Config)

        def __post_init__(self):
            if self.debug.batch_invariant:
                raise ValueError(
                    "Batch-invariant mode is not supported in pre-training."
                )
            if isinstance(self.optimizer, OptimizersInBackwardContainer.Config):
                if self.parallelism.expert_parallel_degree > 1:
                    raise NotImplementedError(
                        "Optimizers in backward is not supported with Expert Parallel."
                    )
                if self.parallelism.pipeline_parallel_degree > 1:
                    raise NotImplementedError(
                        "Optimizers in backward is not supported with Pipeline Parallel."
                    )

        def to_dict(self) -> dict[str, Any]:
            d = {}
            for f in dataclasses.fields(self):
                if f.name == "model_spec":
                    assert self.model_spec is not None
                    # ModelSpec contains callables that can't be serialized
                    d["model_spec"] = {
                        "name": self.model_spec.name,
                        "flavor": self.model_spec.flavor,
                    }
                else:
                    val = getattr(self, f.name)
                    if hasattr(val, "to_dict"):
                        d[f.name] = val.to_dict()
                    elif dataclasses.is_dataclass(val):
                        d[f.name] = asdict(val)
                    else:
                        d[f.name] = val
            return d

        def maybe_log(self) -> None:
            if self.debug.print_config:
                logger.info(
                    f"Running with configs: {json.dumps(self.to_dict(), indent=2, ensure_ascii=False)}"
                )

            if self.debug.save_config_file is not None:
                config_file = os.path.join(
                    self.dump_folder, self.debug.save_config_file
                )
                if torch.distributed.is_initialized():
                    if torch.distributed.get_rank() == 0:
                        os.makedirs(os.path.dirname(config_file), exist_ok=True)
                        with open(config_file, "w") as f:
                            json.dump(self.to_dict(), f, indent=2)
                    logger.info(f"Saved job configs to {config_file}")
                else:
                    logger.warning(
                        "Job configs logging is disabled due to torch.distributed not initialized."
                    )

    def _debug_step_phase(self, phase: str) -> None:
        enabled = os.getenv("TORCHTITAN_STEP_PHASE_LOG", "0")
        if enabled not in ("0", "1"):
            raise ValueError("TORCHTITAN_STEP_PHASE_LOG must be 0 or 1")
        if enabled == "1" and torch.distributed.get_rank() == 0:
            print(
                f"TORCHTITAN_STEP_PHASE step={self.step} phase={phase} "
                f"host_ns={time.monotonic_ns()}",
                flush=True,
            )

    # core configs
    config: Config
    parallel_dims: ParallelDims

    # swappable training components
    tokenizer: BaseTokenizer
    dataloader: BaseDataLoader
    model_config: BaseModel.Config
    # TODO: we should make this list[BaseModel / Decoder] but this will affect many components.
    # will do this in a separate PR
    model_parts: list[torch.nn.Module]
    loss_fn: BaseLoss
    optimizers: OptimizersContainer
    lr_schedulers: LRSchedulersContainer
    validator: BaseValidator
    metrics_processor: MetricsProcessor
    checkpointer: CheckpointManager

    # runtime utilities
    device: torch.device
    gc_handler: utils.GarbageCollection
    train_context: dist_utils.TrainContext
    gradient_accumulation_steps: int
    pp_has_first_stage: bool
    pp_has_last_stage: bool

    # additional training states
    step: int
    ntokens_seen: int

    # Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
    @record
    def __init__(self, config: Config):
        torch._C._log_api_usage_once("torchtitan.train")

        self.config = config
        assert (
            config.model_spec is not None
        ), "model_spec must be set before creating Trainer"
        model_spec = config.model_spec
        dist_utils.validate_pure_model_parallel_model(
            model_spec.name, config.parallelism
        )

        device_module, device_type = utils.device_module, utils.device_type
        # pyrefly: ignore [read-only]
        self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        # Device has to be set before creating TorchFT manager.
        device_module.set_device(self.device)

        # init distributed and build meshes
        self.parallel_dims = parallel_dims = self.init_distributed()

        # Logging needs to happen after distributed initialized
        config.maybe_log()

        if parallel_dims.dp_enabled:
            batch_mesh = parallel_dims.get_mesh("batch")
            batch_degree, batch_rank = batch_mesh.size(), batch_mesh.get_local_rank()
        else:
            batch_degree, batch_rank = 1, 0

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
        self.tokenizer = config.tokenizer.build(tokenizer_path=config.hf_assets_path)

        # build dataloader
        self.dataloader = config.dataloader.build(
            dp_world_size=batch_degree,
            dp_rank=batch_rank,
            tokenizer=self.tokenizer,
            seq_len=config.training.seq_len,
            local_batch_size=config.training.local_batch_size,
        )

        # build model (using meta init)
        model_config = model_spec.model
        # set the model args from training job configs
        model_config.update_from_config(
            trainer_config=config,
        )
        self.model_config = model_config

        logger.info(f"Building {model_spec.name} {model_spec.flavor}")

        with (
            torch.device("meta"),
            utils.set_default_dtype(TORCH_DTYPE_MAP[config.training.dtype]),
        ):
            model = model_config.build()

        # Verify all submodules satisfy the Module protocol
        # TODO: move this to module validate().
        # This is current put here to verify module build and
        # converter, which should guanrantee Module protocol.
        # On the other hand, some parallelism wrappers don't
        # have this guanrantee, e.g., fully_shard.
        model.verify_module_protocol()

        # metrics logging
        self.metrics_processor = config.metrics.build(
            parallel_dims=parallel_dims,
            dump_folder=config.dump_folder,
            pp_schedule=config.parallelism.pipeline_parallel_schedule,
            config_dict=config.to_dict(),
            has_quantization=has_quantization(model_config),
        )
        color = self.metrics_processor.color

        # calculate model size and flops per token
        (
            model_param_count,
            self.metrics_processor.num_flops_per_token,
        ) = model_config.get_nparams_and_flops(model, config.training.seq_len)

        logger.info(
            f"{color.blue}Model {model_spec.name} {model_spec.flavor} "
            f"{color.red}size: {model_param_count:,} total parameters{color.reset}"
        )

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

        self.loss_fn = config.loss.build(
            compile_config=config.compile,
        )

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
            fault_tolerance_enabled=False,
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
            if not model_spec.pipelining_fn:
                raise RuntimeError(
                    f"Pipeline Parallel is enabled but {model_spec.name} "
                    f"does not support pipelining"
                )

            # apply both Pipeline Parallel and SPMD-style scaling techniques
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
                    # TODO: Change this back to init_weights once
                    # autoparallel contains the wrap_init_states
                    cast(BaseModel, m).init_weights(buffer_device=buffer_device)
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
            if not config.checkpoint.create_seed_checkpoint:
                # Skip parallelize_fn for seed checkpoints — nothing from
                # it is needed (AC, compile, nD parallelism, mixed precision, etc.).
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
                # TODO: Change this back to init_weights once
                # autoparallel contains the wrap_init_states
                cast(BaseModel, model).init_weights(buffer_device=buffer_device)
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

        if os.getenv("TORCHTITAN_AGPT_DTYPE_PROBE") == "1":
            if parallel_dims.pp_enabled or len(self.model_parts) != 1:
                raise ValueError("AGPT dtype probe requires a non-pipeline model")
            install_agpt_dtype_probe(self.model_parts[0])

        # Set lm_head reference for ChunkedCELoss after model construction.
        # Non-PP: single model part always has lm_head.
        # PP: only the last stage has lm_head; non-last stages skip this.
        if isinstance(self.loss_fn, ChunkedCELoss):
            if parallel_dims.pp_enabled:
                if self.pp_has_last_stage:
                    lm_head = self.model_parts[-1].lm_head
                    assert (
                        lm_head is not None
                    ), "Last PP stage must have lm_head for ChunkedCELoss"
                    self.loss_fn.set_lm_head(
                        lm_head  # pyrefly: ignore[bad-argument-type]
                    )
                    self.model_parts[
                        -1
                    ]._skip_lm_head = True  # pyrefly: ignore[bad-argument-type]
            else:
                assert len(self.model_parts) == 1
                lm_head = self.model_parts[0].lm_head
                assert lm_head is not None, "Model must have lm_head for ChunkedCELoss"
                self.loss_fn.set_lm_head(lm_head)  # pyrefly: ignore[bad-argument-type]
                self.model_parts[
                    0
                ]._skip_lm_head = True  # pyrefly: ignore[bad-argument-type]

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
        self.optimizers = config.optimizer.build(model_parts=self.model_parts)
        if model_spec.post_optimizer_build_fn is not None:
            model_spec.post_optimizer_build_fn(
                self.optimizers, self.model_parts, parallel_dims
            )
        self.lr_schedulers = config.lr_scheduler.build(
            optimizers=self.optimizers,
            training_steps=config.training.steps,
        )
        self.metrics_processor.optimizers = self.optimizers
        self.metrics_processor.model_parts = self.model_parts

        # Initialize trainer states that will be saved in checkpoint.
        # These attributes must be initialized before checkpoint loading.
        self.step = 0
        self.ntokens_seen = 0

        self.checkpointer = config.checkpoint.build(
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
        world_size = dist_utils.init_distributed(
            config.comm,
            enable_cpu_backend=config.training.enable_cpu_offload,
            base_folder=config.dump_folder,
        )

        return ParallelDims.from_config(config.parallelism, world_size)

    def batch_generator(
        self, data_iterable: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        """Returns an iterator that processes batches from the data iterator.

        Note: Tensors are yielded on CPU. The caller is responsible for moving
        them to GPU when needed. This allows for more efficient memory usage
        when doing gradient accumulation.
        """
        data_iterator = iter(data_iterable)

        while True:
            data_load_start = time.perf_counter()
            try:
                batch = next(data_iterator)
            except StopIteration as ex:
                # If data runs out during gradient accumulation, that
                # entire step will not be executed.
                raise DataloaderExhaustedError() from ex
            input_dict, labels = batch
            ntokens_batch = labels.numel()
            self.metrics_processor.ntokens_since_last_log += ntokens_batch
            self.metrics_processor.data_loading_times.append(
                time.perf_counter() - data_load_start
            )

            # Tensors stay on CPU; moved to GPU per-microbatch during training
            yield input_dict, labels

    def post_dataloading_process(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        """
        Post-processing hook after data loading and before model forward pass.

        This method processes the raw data from the dataloader and prepares it for
        the model's forward pass. It separates the main input tensor from auxiliary
        inputs and constructs additional keyword arguments (e.g., attention masks).

        This method can be overridden in subclasses to customize data processing
        for different training strategies (e.g., converting tensors to DTensors,
        applying custom transformations, etc.).

        Args:
            input_dict: Dictionary containing tensors from the dataloader. Must
                contain an "input" key with the main input tensor. May contain
                additional keys for auxiliary inputs (e.g., position ids).
            labels: Target labels for the batch.

        Returns:
            A tuple of (inputs, labels, extra_inputs, extra_kwargs) where:
                - inputs: Main input tensor extracted from input_dict["input"].
                - labels: Target labels (unchanged from input parameter).
                - extra_inputs: Dict of auxiliary input tensors from input_dict
                    (excluding "input" and "positions"). These are passed to the
                    model forward but are NOT forwarded across pipeline parallel
                    stages.
                - extra_kwargs: Dict of additional keyword arguments for model
                    forward (positions, attention_masks). These ARE forwarded
                    across all pipeline parallel stages.

        Note:
            The distinction between extra_inputs and extra_kwargs is important for
            pipeline parallelism: extra_kwargs are forwarded to all pipeline stages,
            while extra_inputs are only available to the first stage. Positions
            always go into extra_kwargs so every stage can apply RoPE correctly.
        """
        inputs = input_dict["input"]
        extra_inputs = {k: v for k, v in input_dict.items() if k != "input"}
        # extra_kwargs are forwarded to all PP stages; extra_inputs are only
        # available to the first stage.  Positions go into extra_kwargs so
        # every stage can apply RoPE correctly.
        extra_kwargs: dict[str, Any] = {}

        # Resolve positions once: per-document positions for block_causal,
        # sequential positions when CP needs them for shard indexing,
        # or None (model uses sequential RoPE slice by default).
        if isinstance(self.model_config, Decoder.Config):
            layer = self.model_config.layers[0]
            attn_config = layer.attention
        else:
            attn_config = None
        mask_type = getattr(attn_config, "mask_type", "causal")

        positions = extra_inputs.pop("positions", None)
        if mask_type == "block_causal":
            # Per-document positions from the dataloader
            extra_kwargs["positions"] = positions
        elif self.parallel_dims.cp_enabled:
            # Sequential positions needed for correct RoPE after CP sharding
            extra_kwargs["positions"] = torch.arange(
                0, inputs.shape[1], dtype=torch.int32, device=self.device
            ).expand(inputs.shape)

        inner_attention = getattr(attn_config, "inner_attention", None)
        if inner_attention is not None:
            from torchtitan.models.common.attention import (
                FlexAttention,
                VarlenAttention,
            )

            if isinstance(
                inner_attention, (FlexAttention.Config, VarlenAttention.Config)
            ):
                assert (
                    self.tokenizer is not None
                ), "tokenizer is required for flex/varlen attention"
                model = cast(Decoder, self.model_parts[0])
                extra_kwargs["attention_masks"] = model.get_attention_masks(
                    input_batch=inputs,
                    tokenizer=self.tokenizer,
                    extra_inputs=extra_inputs,
                )

        if self.parallel_dims.cp_enabled:
            inputs, labels, extra_kwargs = prepare_context_parallel_input(
                inputs,
                labels,
                extra_kwargs,
                self.parallel_dims.get_mesh("cp"),
                self.device,
                self.config.parallelism.context_parallel_load_balancer,
            )

        # Accumulate after CP sharding so labels.numel() reflects the actual
        # unique tokens this rank processes (not the full pre-split sequence).
        self.ntokens_seen += labels.numel()

        return inputs, labels, extra_inputs, extra_kwargs

    def forward_backward_step(
        self,
        *,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        model_parts = self.model_parts
        parallel_dims = self.parallel_dims

        inputs, labels, extra_inputs, extra_kwargs = self.post_dataloading_process(
            input_dict, labels
        )
        self._debug_step_phase("post_dataloading_ready")

        if parallel_dims.pp_enabled:
            # Pipeline Parallel forward / backward inside step() call
            loss_kwargs = {"global_valid_tokens": global_valid_tokens}
            with self.train_context():
                targets, losses = (
                    (labels, []) if self.pp_has_last_stage else (None, None)
                )
                if self.pp_has_first_stage:
                    self.pp_schedule.step(
                        inputs,
                        **extra_inputs,
                        **extra_kwargs,
                        target=targets,
                        losses=losses,
                        loss_kwargs=loss_kwargs,
                        return_outputs=False,
                    )
                else:
                    self.pp_schedule.step(
                        **extra_kwargs,
                        target=targets,
                        losses=losses,
                        loss_kwargs=loss_kwargs,
                        return_outputs=False,
                    )

            if self.config.parallelism.enable_data_parallel_native_ddp:
                arm_native_ddp_pipeline_bucket_rebuild(model_parts[0])

            # accumulate losses across pipeline microbatches
            # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
            if self.pp_has_last_stage:
                assert losses is not None
                # All loss classes scale by global_valid_tokens internally
                loss = torch.sum(torch.stack(losses)).to(self.device)
            else:
                loss = torch.tensor([-1.0], device=self.device)
        else:
            # Non-PP forward / backward
            assert len(model_parts) == 1
            with self.train_context():
                self._debug_step_phase("forward_start")
                pred = model_parts[0](inputs, **extra_inputs, **extra_kwargs)
                self._debug_step_phase("forward_returned")
                loss = self.loss_fn(pred, labels, global_valid_tokens)
                self._debug_step_phase("loss_returned")
                del pred
                backward_loss = (
                    scale_native_ddp_loss(loss, parallel_dims.dp_replicate)
                    if self.config.parallelism.enable_data_parallel_native_ddp
                    else loss
                )
                self._debug_step_phase("backward_start")
                backward_loss.backward()
                self._debug_step_phase("backward_returned")

        # The returned loss here is local SUM loss / global_valid_tokens
        return loss

    def train_step(
        self, data_iterator: Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ):
        self.optimizers.zero_grad()
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

            loss = self.forward_backward_step(
                input_dict=input_dict,
                labels=labels,
                # pyrefly: ignore [bad-argument-type]
                global_valid_tokens=global_valid_tokens,
            )
            accumulated_losses.append(loss.detach())

        if self.config.parallelism.enable_data_parallel_native_ddp:
            # Experimental mixed-precision DDP restores FP32 gradients on an
            # upcast stream; ordinary DDP/autocast makes this helper a no-op.
            record_native_ddp_grad_streams(self.model_parts[0])
        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.config.training.max_norm,
            foreach=True,
            pp_mesh=parallel_dims.get_optional_mesh("pp"),
            ep_enabled=parallel_dims.ep_enabled,
        )
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
        self.optimizers.step()
        self.lr_schedulers.step()
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

        # log metrics
        if not should_log:
            return

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
            local_avg_loss = (
                metrics_loss * global_valid_tokens / local_valid_tokens
            )
            (
                global_avg_loss_tensor,
                global_max_loss_tensor,
                global_ntokens_seen_tensor,
            ) = (
                dist_utils.dist_sum_tensor(metrics_loss, loss_mesh),
                dist_utils.dist_max_tensor(local_avg_loss, loss_mesh),
                dist_utils.dist_sum_tensor(
                    torch.tensor(
                        self.ntokens_seen, dtype=torch.int64, device=self.device
                    ),
                    loss_mesh,
                ),
            )
            # Launch all independent metric collectives before the first host
            # scalar read so they do not serialize submission and waiting.
            global_avg_loss = float(global_avg_loss_tensor.item())
            global_max_loss = float(global_max_loss_tensor.item())
            global_ntokens_seen = int(global_ntokens_seen_tensor.item())
        else:
            global_avg_loss = global_max_loss = float(loss.detach().item())
            global_ntokens_seen = self.ntokens_seen

        extra_metrics = {
            "n_tokens_seen": global_ntokens_seen,
            "lr": lr,
        }
        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            float(logged_grad_norm.item()),
            extra_metrics=extra_metrics,
        )

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

        with config.profiler.build(
            global_step=self.step,
            base_folder=config.dump_folder,
        ) as profiler:
            data_iterator = self.batch_generator(self.dataloader)
            while self.should_continue_training():
                self.step += 1
                self.gc_handler.run(self.step)
                try:
                    self.train_step(data_iterator)
                except DataloaderExhaustedError:
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

        if torch.distributed.get_rank() == 0:
            logger.info("Sleeping 2 seconds for other ranks to complete")
            time.sleep(2)

        logger.info("Training completed")

    def should_continue_training(self) -> bool:
        return self.step < self.config.training.steps

    def state_dict(self) -> dict[str, Any]:
        return {"step": self.step, "ntokens_seen": self.ntokens_seen}

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.step = state_dict["step"]
        self.ntokens_seen = state_dict["ntokens_seen"]

    def close(self) -> None:
        if hasattr(self, "checkpointer") and self.checkpointer:
            self.checkpointer.close()
        if hasattr(self, "metrics_processor") and self.metrics_processor:
            self.metrics_processor.close()
