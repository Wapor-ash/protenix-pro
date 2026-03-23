# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import hashlib
import logging
import os
import time
from argparse import Namespace
from contextlib import nullcontext
from typing import Any, Dict, Mapping, Tuple

import torch
import torch.distributed as dist
import wandb

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_model_type import model_configs
from protenix.config.config import parse_configs, parse_sys_args, save_config
from protenix.data.pipeline.dataloader import get_dataloaders
from protenix.metrics.lddt_metrics import LDDTMetrics
from protenix.model.loss import ProtenixLoss
from protenix.model.protenix import Protenix
from protenix.utils.distributed import DIST_WRAPPER
from protenix.utils.lr_scheduler import FinetuneLRScheduler, get_lr_scheduler
from protenix.utils.metrics import SimpleMetricAggregator
from protenix.utils.permutation.permutation import SymmetricPermutation
from protenix.utils.seed import seed_everything
from protenix.utils.torch_utils import autocasting_disable_decorator, to_device
from protenix.utils.two_stage_adapter import (
    collect_required_adapter_param_substrings,
    parse_adapter_keywords,
    validate_required_adapter_matches,
)
from protenix.utils.training import get_optimizer, is_loss_nan_check
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from runner.ema import EMAWrapper

# Disable WANDB's console output capture to reduce unnecessary logging
os.environ["WANDB_CONSOLE"] = "off"

torch.serialization.add_safe_globals([Namespace])


class AF3Trainer(object):
    """
    Trainer class for the Alphafold3 model.

    Args:
        configs (Any): Configuration object for training.
    """

    def __init__(self, configs: Any) -> None:
        self.configs = configs
        self.init_env()
        self.init_basics()
        self.init_log()
        self.init_model()
        self.init_loss()
        self.init_data()
        self.try_load_checkpoint()

    def init_basics(self) -> None:
        """
        Initialize basic training parameters and directory structure.
        """
        # Step means effective step considering accumulation
        self.step = 0
        # Global_step equals to self.step * self.iters_to_accumulate
        self.global_step = 0
        self.start_step = 0
        # Add for grad accumulation, it can increase real batch size
        self.iters_to_accumulate = self.configs.iters_to_accumulate

        self.run_name = self.configs.run_name + "_" + time.strftime("%Y%m%d_%H%M%S")
        run_names = DIST_WRAPPER.all_gather_object(
            self.run_name if DIST_WRAPPER.rank == 0 else None
        )
        self.run_name = [name for name in run_names if name is not None][0]
        self.run_dir = f"{self.configs.base_dir}/{self.run_name}"
        self.checkpoint_dir = f"{self.run_dir}/checkpoints"
        self.prediction_dir = f"{self.run_dir}/predictions"
        self.structure_dir = f"{self.run_dir}/structures"
        self.dump_dir = f"{self.run_dir}/dumps"
        self.error_dir = f"{self.run_dir}/errors"

        if DIST_WRAPPER.rank == 0:
            os.makedirs(self.run_dir)
            os.makedirs(self.checkpoint_dir)
            os.makedirs(self.prediction_dir)
            os.makedirs(self.structure_dir)
            os.makedirs(self.dump_dir)
            os.makedirs(self.error_dir)
            save_config(
                self.configs,
                os.path.join(self.configs.base_dir, self.run_name, "config.yaml"),
            )

        self.print(
            f"Using run name: {self.run_name}, run dir: {self.run_dir}, "
            f"checkpoint_dir: {self.checkpoint_dir}, "
            f"prediction_dir: {self.prediction_dir}, "
            f"structure_dir: {self.structure_dir}, "
            f"error_dir: {self.error_dir}"
        )

    def init_log(self) -> None:
        """
        Initialize logging and metrics aggregation.
        """
        if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
            wandb.init(
                project=self.configs.project,
                name=self.run_name,
                config=vars(self.configs),
                id=self.configs.wandb_id or None,
            )
        self.train_metric_wrapper = SimpleMetricAggregator(["avg"])

    def init_env(self) -> None:
        """
        Initialize the PyTorch and CUDA environment for training.
        Sets up distributed training if world_size > 1 and sets random seeds.
        """
        logging.info(
            f"Distributed environment: world size: {DIST_WRAPPER.world_size}, "
            f"global rank: {DIST_WRAPPER.rank}, local rank: {DIST_WRAPPER.local_rank}"
        )
        self.use_cuda = torch.cuda.device_count() > 0
        if self.use_cuda:
            self.device = torch.device(f"cuda:{DIST_WRAPPER.local_rank}")
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            all_gpu_ids = ",".join(str(x) for x in range(torch.cuda.device_count()))
            devices = os.getenv("CUDA_VISIBLE_DEVICES", all_gpu_ids)
            logging.info(
                f"LOCAL_RANK: {DIST_WRAPPER.local_rank} - CUDA_VISIBLE_DEVICES: [{devices}]"
            )
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        if DIST_WRAPPER.world_size > 1:
            timeout_seconds = int(os.environ.get("NCCL_TIMEOUT_SECOND", 600))
            dist.init_process_group(
                backend="nccl", timeout=datetime.timedelta(seconds=timeout_seconds)
            )

        if not self.configs.deterministic_seed:
            # use rank-specific seed
            hash_string = f"({self.configs.seed},{DIST_WRAPPER.rank},init_seed)"
            rank_seed = int(hashlib.sha256(hash_string.encode("utf8")).hexdigest(), 16)
            rank_seed = rank_seed % (2**32)
        else:
            rank_seed = self.configs.seed

        seed_everything(
            seed=rank_seed,
            deterministic=self.configs.deterministic,
        )  # Different DDP processes get different seeds

        if self.configs.triangle_attention == "deepspeed":
            env = os.getenv("CUTLASS_PATH", None)
            print(f"env: {env}")
            assert env is not None, (
                "If use deepspeed (ds4sci), set CUTLASS_PATH env as per "
                "https://www.deepspeed.ai/tutorials/ds4sci_evoformerattention/"
            )
        logging.info("Finished environment initialization.")

    def init_loss(self) -> None:
        """
        Initialize loss functions, metrics, and permutation utilities.
        """
        self.loss = ProtenixLoss(self.configs)
        self.symmetric_permutation = SymmetricPermutation(
            self.configs, error_dir=self.error_dir
        )
        self.lddt_metrics = LDDTMetrics(self.configs)

    def init_model(self) -> None:
        """
        Initialize the Protenix model, optimizer, and exponential moving average (EMA).
        Sets up DistributedDataParallel (DDP) if multiple GPUs are used.
        """
        self.raw_model = Protenix(self.configs).to(self.device)
        self.use_ddp = False
        if DIST_WRAPPER.world_size > 1:
            self.print("Using DistributedDataParallel (DDP)")
            self.use_ddp = True
            # Fix DDP/checkpoint compatibility:
            # https://discuss.pytorch.org/t/ddp-and-gradient-checkpointing/132244
            self.model = DDP(
                self.raw_model,
                find_unused_parameters=self.configs.find_unused_parameters,
                device_ids=[DIST_WRAPPER.local_rank],
                output_device=DIST_WRAPPER.local_rank,
                static_graph=True,
            )
        else:
            self.model = self.raw_model

        def count_parameters(model: torch.nn.Module) -> float:
            """Count total parameters in millions."""
            total_params = sum(p.numel() for p in model.parameters())
            return total_params / 1e6

        self.print(f"Model Parameters: {count_parameters(self.model):.2f}M")

        # === Learning rate group setup ===
        two_stage_cfg = self.configs.get("two_stage", {})
        self.two_stage_enable = two_stage_cfg.get("enable", False)
        self.current_stage = 1 if self.two_stage_enable else 0  # 0 = normal, 1 = stage1, 2 = stage2
        self._lr_group_config = None  # Set by per-group setup methods

        if self.two_stage_enable:
            self._setup_stage1()
        else:
            # Normal training: setup EMA
            if self.configs.get("ema_decay", -1) > 0:
                assert self.configs.ema_decay < 1
                self.ema_wrapper = EMAWrapper(
                    self.model,
                    self.configs.ema_decay,
                    self.configs.ema_mutable_param_keywords,
                )

            torch.cuda.empty_cache()

            # Check for per-group LR config
            adapter_lr = two_stage_cfg.get("adapter_lr", -1.0)
            backbone_lr = two_stage_cfg.get("backbone_lr", -1.0)
            use_per_group = (adapter_lr != -1.0 or backbone_lr != -1.0)

            if use_per_group:
                self._setup_per_group_training(adapter_lr, backbone_lr)
            else:
                self.optimizer = get_optimizer(
                    self.configs,
                    self.model,
                    param_names=self.configs.get("finetune_params_with_substring", [""]),
                )
                self.init_scheduler()

    def _get_adapter_keywords(self) -> list:
        """Get the adapter keywords for per-group LR (supports comma-separated)."""
        raw = self.configs.two_stage.get(
            "adapter_keywords",
            "rnalm_projection,rna_projection,dna_projection,linear_rnalm,linear_rna_llm,linear_dna_llm,"
            "rnalm_alpha_logit,rnalm_gate_mlp,linear_no_bias_a_rna,rna_template_alpha,rna_template_gate,"
            "layer_weights,projection_sequence_features,projection_pairwise_features,"
            "gated_sequence_feature_injector,gated_pairwise_feature_injector,ribonanza_pairformer_stack,"
            "constraint_embedder.substructure_z_embedder,constraint_embedder.substructure_log_alpha",
        )
        return parse_adapter_keywords(raw)

    def _is_adapter_param(self, name: str, adapter_keywords: list | None = None) -> bool:
        """Check if a parameter name matches any adapter keyword."""
        if adapter_keywords is None:
            adapter_keywords = self._get_adapter_keywords()
        return any(kw in name for kw in adapter_keywords)

    def _validate_required_adapter_routing(
        self,
        param_names: list[str],
        adapter_keywords: list[str],
    ) -> None:
        required_substrings = collect_required_adapter_param_substrings(self.configs)
        validate_required_adapter_matches(
            param_names=param_names,
            adapter_keywords=adapter_keywords,
            required_substrings=required_substrings,
        )

    def _split_params(self):
        """Split model params into adapter and backbone groups.

        Skips frozen parameters (requires_grad=False), e.g. RibonanzaNet2 backbone.
        """
        adapter_keywords = self._get_adapter_keywords()
        named_params = [
            (name, param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        ]
        self._validate_required_adapter_routing(
            [name for name, _ in named_params],
            adapter_keywords,
        )

        adapter_params = []
        backbone_params = []
        n_adapter = 0
        n_backbone = 0
        for name, param in named_params:
            if self._is_adapter_param(name, adapter_keywords):
                adapter_params.append(param)
                n_adapter += param.numel()
            else:
                backbone_params.append(param)
                n_backbone += param.numel()
        return adapter_params, backbone_params, n_adapter, n_backbone

    def _build_optimizer(self, backbone_params, adapter_params, backbone_lr, adapter_lr):
        """Create optimizer with separate backbone/adapter param groups.

        Group 0 = backbone, Group 1 = adapter.
        Uses AdamW or Adam based on configs.adam.use_adamw.
        """
        param_groups = [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": adapter_params, "lr": adapter_lr},
        ]
        opt_cls = torch.optim.AdamW if self.configs.adam.get("use_adamw", False) else torch.optim.Adam
        return opt_cls(
            param_groups,
            lr=adapter_lr,
            weight_decay=self.configs.adam.weight_decay,
            betas=(self.configs.adam.beta1, self.configs.adam.beta2),
        )

    def _apply_per_group_lr(self):
        """Apply per-group LR scaling after scheduler step.

        The scheduler sets all param groups to the same LR based on scheduler_base_lr.
        This method rescales each group to its actual target LR while preserving
        the scheduler's warmup/decay shape.

        When backbone_base_lr=0, backbone LR stays 0 throughout training (frozen).
        """
        if self._lr_group_config is None:
            return
        cfg = self._lr_group_config
        # Get the lr that scheduler just set (same for all groups)
        sched_lr = self.optimizer.param_groups[cfg["adapter_idx"]]["lr"]
        base = cfg["scheduler_base_lr"]
        if base > 1e-12:
            scale = sched_lr / base
        else:
            scale = 0.0 if sched_lr == 0 else 1.0
        self.optimizer.param_groups[cfg["adapter_idx"]]["lr"] = cfg["adapter_base_lr"] * scale
        self.optimizer.param_groups[cfg["backbone_idx"]]["lr"] = cfg["backbone_base_lr"] * scale

    def _setup_per_group_training(self, adapter_lr: float, backbone_lr: float) -> None:
        """1-stage training with separate LRs for adapter (new modules) and backbone.

        When backbone_lr=0, backbone params are effectively frozen for the entire training.
        The scheduler's warmup/decay shape is applied proportionally to both groups.
        """
        global_lr = self.configs.lr
        if adapter_lr < 0:
            adapter_lr = global_lr
        if backbone_lr < 0:
            backbone_lr = global_lr

        adapter_keywords = self._get_adapter_keywords()
        adapter_params, backbone_params, n_adapter, n_backbone = self._split_params()

        self.print(
            f"[Per-group LR] adapter={n_adapter/1e6:.2f}M (lr={adapter_lr}), "
            f"backbone={n_backbone/1e6:.2f}M (lr={backbone_lr}), "
            f"keywords={adapter_keywords}"
        )
        if adapter_lr > 0 and n_adapter == 0:
            raise RuntimeError(
                "Per-group LR requested adapter training, but no parameters matched "
                f"two_stage.adapter_keywords={adapter_keywords}"
            )

        self.optimizer = self._build_optimizer(
            backbone_params, adapter_params, backbone_lr, adapter_lr
        )
        opt_name = "AdamW" if self.configs.adam.get("use_adamw", False) else "Adam"
        self.print(
            f"[Per-group LR] Optimizer: {opt_name}, backbone_lr={backbone_lr}, adapter_lr={adapter_lr}"
        )

        # Store per-group config for LR scaling
        self._lr_group_config = {
            "backbone_idx": 0,
            "adapter_idx": 1,
            "adapter_base_lr": adapter_lr,
            "backbone_base_lr": backbone_lr,
            "scheduler_base_lr": global_lr,  # init_scheduler uses configs.lr
        }

        self.init_scheduler()
        # Restore per-group LRs after scheduler init (scheduler sets uniform LR)
        self._apply_per_group_lr()
        torch.cuda.empty_cache()

    def _setup_stage1(self) -> None:
        """Stage 1: Adapter warmup with configurable backbone/adapter LRs.

        By default backbone is frozen (stage1_backbone_lr=0).
        No EMA in Stage 1.
        """
        self.current_stage = 1
        stage_cfg = self.configs.two_stage
        adapter_keywords = self._get_adapter_keywords()

        stage1_adapter_lr = stage_cfg.get("stage1_adapter_lr", stage_cfg.get("stage1_lr", 5e-3))
        stage1_backbone_lr = stage_cfg.get("stage1_backbone_lr", 0.0)

        adapter_params, backbone_params, n_adapter, n_backbone = self._split_params()

        self.print(
            f"[Stage 1] Adapter warmup: backbone={n_backbone/1e6:.2f}M (lr={stage1_backbone_lr}), "
            f"adapter={n_adapter/1e6:.2f}M (lr={stage1_adapter_lr}), keywords={adapter_keywords}"
        )
        if stage1_adapter_lr > 0 and n_adapter == 0:
            raise RuntimeError(
                "Stage 1 adapter warmup requested adapter training, but no parameters matched "
                f"two_stage.adapter_keywords={adapter_keywords}"
            )

        # No EMA in Stage 1
        if hasattr(self, "ema_wrapper"):
            delattr(self, "ema_wrapper")

        self.optimizer = self._build_optimizer(
            backbone_params, adapter_params, stage1_backbone_lr, stage1_adapter_lr
        )
        opt_name = "AdamW" if self.configs.adam.get("use_adamw", False) else "Adam"
        self.print(
            f"[Stage 1] Optimizer: {opt_name}, backbone_lr={stage1_backbone_lr}, adapter_lr={stage1_adapter_lr}"
        )

        stage1_warmup = stage_cfg.get("stage1_warmup_steps", 1)
        stage1_max = stage_cfg.get("stage1_max_steps", 400)
        self.lr_scheduler = get_lr_scheduler(
            Namespace(
                lr_scheduler="cosine_annealing",
                warmup_steps=stage1_warmup,
                max_steps=stage1_max,
                lr=stage1_adapter_lr,
                min_lr_ratio=0.01,
            ),
            self.optimizer,
        )

        self._lr_group_config = {
            "backbone_idx": 0,
            "adapter_idx": 1,
            "adapter_base_lr": stage1_adapter_lr,
            "backbone_base_lr": stage1_backbone_lr,
            "scheduler_base_lr": stage1_adapter_lr,
        }
        self._apply_per_group_lr()

        self.print(
            f"[Stage 1] Scheduler: cosine, warmup={stage1_warmup}, "
            f"max_steps={stage1_max}, adapter_lr={stage1_adapter_lr}, backbone_lr={stage1_backbone_lr}"
        )

        torch.cuda.empty_cache()

    def _transition_to_stage2(self) -> None:
        """Stage 2: Joint training with configurable backbone/adapter LRs.

        Defaults: stage2_adapter_lr = stage1_adapter_lr, stage2_backbone_lr = stage2_adapter_lr.
        Enables EMA.
        """
        self.current_stage = 2
        stage_cfg = self.configs.two_stage

        # Unfreeze all parameters except permanently frozen ones (e.g. RibonanzaNet2)
        for name, param in self.model.named_parameters():
            if "ribonanza_net." not in name:
                param.requires_grad = True

        n_total = sum(p.numel() for p in self.model.parameters()) / 1e6
        self.print(f"[Stage 2] Joint training: all {n_total:.2f}M params trainable")

        # Resolve LRs with defaults
        stage1_adapter_lr = stage_cfg.get("stage1_adapter_lr", stage_cfg.get("stage1_lr", 5e-3))
        stage2_adapter_lr = stage_cfg.get("stage2_adapter_lr", -1.0)
        if stage2_adapter_lr <= 0:
            stage2_adapter_lr = stage1_adapter_lr
        stage2_backbone_lr = stage_cfg.get("stage2_backbone_lr", -1.0)
        if stage2_backbone_lr <= 0:
            stage2_backbone_lr = stage2_adapter_lr

        adapter_params, backbone_params, _, _ = self._split_params()

        self.optimizer = self._build_optimizer(
            backbone_params, adapter_params, stage2_backbone_lr, stage2_adapter_lr
        )
        opt_name = "AdamW" if self.configs.adam.get("use_adamw", False) else "Adam"
        self.print(
            f"[Stage 2] Optimizer: {opt_name}, backbone_lr={stage2_backbone_lr}, adapter_lr={stage2_adapter_lr}"
        )

        # Create scheduler for Stage 2
        stage2_warmup = stage_cfg.get("stage2_warmup_steps", 100)
        stage2_max = self.configs.max_steps
        stage1_max = stage_cfg.get("stage1_max_steps", 400)
        stage2_total = stage2_max - stage1_max
        self.lr_scheduler = get_lr_scheduler(
            Namespace(
                lr_scheduler="cosine_annealing",
                warmup_steps=stage2_warmup,
                max_steps=stage2_total,
                lr=stage2_adapter_lr,
                min_lr_ratio=0.01,
            ),
            self.optimizer,
        )

        self._lr_group_config = {
            "backbone_idx": 0,
            "adapter_idx": 1,
            "adapter_base_lr": stage2_adapter_lr,
            "backbone_base_lr": stage2_backbone_lr,
            "scheduler_base_lr": stage2_adapter_lr,
        }
        self._apply_per_group_lr()

        self.print(
            f"[Stage 2] Scheduler: cosine, warmup={stage2_warmup}, "
            f"max_steps={stage2_total}, adapter_lr={stage2_adapter_lr}, backbone_lr={stage2_backbone_lr}"
        )

        # Enable EMA in Stage 2
        ema_decay = stage_cfg.get("stage2_ema_decay", 0.999)
        if ema_decay > 0:
            self.ema_wrapper = EMAWrapper(
                self.model,
                ema_decay,
                self.configs.ema_mutable_param_keywords,
            )
            self.ema_wrapper.register()
            self.print(f"[Stage 2] EMA enabled with decay={ema_decay}")

        torch.cuda.empty_cache()

    def init_scheduler(self, **kwargs: Any) -> None:
        """
        Initialize the learning rate scheduler.
        Supports both standard and fine-tuning schedulers.

        Args:
            **kwargs: Additional arguments passed to the scheduler.
        """
        # init finetune lr scheduler if available
        finetune_params = self.configs.get("finetune_params_with_substring", [""])
        is_finetune = len(finetune_params[0]) > 0

        if is_finetune:
            self.lr_scheduler = FinetuneLRScheduler(
                self.optimizer,
                self.configs,
                self.configs.finetune,
                **kwargs,
            )
        else:
            self.lr_scheduler = get_lr_scheduler(self.configs, self.optimizer, **kwargs)

    def init_data(self) -> None:
        """
        Initialize training and test dataloaders.
        """
        self.train_dl, self.test_dls = get_dataloaders(
            self.configs,
            DIST_WRAPPER.world_size,
            seed=self.configs.seed,
            error_dir=self.error_dir,
        )

    def save_checkpoint(self, ema_suffix: str = "") -> None:
        """
        Save the current model state, optimizer, and scheduler to a checkpoint file.

        Args:
            ema_suffix (str): Optional suffix for EMA checkpoints.
        """
        if DIST_WRAPPER.rank == 0:
            path = f"{self.checkpoint_dir}/{self.step}{ema_suffix}.pt"
            checkpoint = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": (
                    self.lr_scheduler.state_dict()
                    if self.lr_scheduler is not None
                    else None
                ),
                "step": self.step,
            }
            torch.save(checkpoint, path)
            self.print(f"Saved checkpoint to {path}")

    def try_load_checkpoint(self) -> None:
        """
        Attempt to load model and training state from specified checkpoint paths.
        Loads both standard and EMA checkpoints if configured.
        """

        def _repair_rna_template_projector_after_load(
            checkpoint_keys: set,
            checkpoint_label: str,
        ) -> None:
            """Run post-load RNA template projector repair before EMA registration."""
            rna_cfg = self.configs.get("rna_template", {})
            if rna_cfg.get("enable", False):
                result = self.raw_model.reinit_rna_projector_from_protein(
                    checkpoint_keys=checkpoint_keys
                )
                self.print(
                    f"RNA projector init after {checkpoint_label} load: {result}"
                )

        def _register_ema_shadow_if_needed() -> None:
            """Register EMA shadow once the model weights are fully repaired."""
            if hasattr(self, "ema_wrapper") and not self.ema_wrapper.shadow:
                self.ema_wrapper.register()

        def _load_checkpoint(
            checkpoint_path: str,
            load_params_only: bool,
            skip_load_optimizer: bool = False,
            skip_load_step: bool = False,
            skip_load_scheduler: bool = False,
            load_step_for_scheduler: bool = True,
        ) -> set:
            """
            Internal helper to load a single checkpoint.

            Args:
                checkpoint_path (str): Path to the checkpoint file.
                load_params_only (bool): If True, only load model parameters.
                skip_load_optimizer (bool): If True, do not load optimizer state.
                skip_load_step (bool): If True, do not load training step.
                skip_load_scheduler (bool): If True, do not load scheduler state.
                load_step_for_scheduler (bool): If True, re-initialize scheduler with loaded step.

            Returns:
                set: The set of keys present in the checkpoint model state dict.
            """
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"Given checkpoint path does not exist: [{checkpoint_path}]"
                )
            self.print(
                f"Loading from {checkpoint_path}, strict: {self.configs.load_strict}"
            )
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
            sample_key = list(checkpoint["model"].keys())[0]
            self.print(f"Sampled key: {sample_key}")
            if sample_key.startswith("module.") and not self.use_ddp:
                # DDP checkpoint has module. prefix, remove it if not using DDP
                checkpoint["model"] = {
                    k[len("module.") :]: v for k, v in checkpoint["model"].items()
                }

            checkpoint_keys = set(checkpoint["model"].keys())

            self.model.load_state_dict(
                state_dict=checkpoint["model"],
                strict=self.configs.load_strict,
            )
            if not load_params_only:
                if not skip_load_optimizer:
                    self.print("Loading optimizer state")
                    self.optimizer.load_state_dict(checkpoint["optimizer"])
                if not skip_load_step:
                    self.print("Loading checkpoint step")
                    self.step = checkpoint["step"] + 1
                    self.start_step = self.step
                    self.global_step = self.step * self.iters_to_accumulate
                if not skip_load_scheduler:
                    self.print("Loading scheduler state")
                    self.lr_scheduler.load_state_dict(checkpoint["scheduler"])
                elif load_step_for_scheduler:
                    assert (
                        not skip_load_step
                    ), "If load_step_for_scheduler is True, you must load step first"
                    # Reinitialize LR scheduler using the updated optimizer and step
                    self.init_scheduler(last_epoch=self.step - 1)

            self.print(f"Finish loading checkpoint, current step: {self.step}")
            return checkpoint_keys

        # Load EMA model parameters if configured
        if self.configs.load_ema_checkpoint_path:
            ema_ckpt_keys = _load_checkpoint(
                self.configs.load_ema_checkpoint_path,
                load_params_only=True,
            )
            _repair_rna_template_projector_after_load(
                ema_ckpt_keys, "EMA checkpoint"
            )
            _register_ema_shadow_if_needed()

        # Load standard model checkpoint if configured
        if self.configs.load_checkpoint_path:
            ckpt_keys = _load_checkpoint(
                self.configs.load_checkpoint_path,
                self.configs.load_params_only,
                skip_load_optimizer=self.configs.skip_load_optimizer,
                skip_load_scheduler=self.configs.skip_load_scheduler,
                skip_load_step=self.configs.skip_load_step,
                load_step_for_scheduler=self.configs.load_step_for_scheduler,
            )
            _repair_rna_template_projector_after_load(ckpt_keys, "checkpoint")

        _register_ema_shadow_if_needed()

    def print(self, msg: str) -> None:
        """
        Print message to log only on the master rank (rank 0).

        Args:
            msg (str): The message to log.
        """
        if DIST_WRAPPER.rank == 0:
            logging.info(msg)

    def model_forward(
        self, batch: Dict[str, Any], mode: str = "train"
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Perform a forward pass of the model.

        Args:
            batch (Dict[str, Any]): Input batch containing features and labels.
            mode (str): Execution mode, either 'train' or 'eval'.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: Updated batch with
                predictions and logging dictionary.
        """
        assert mode in ["train", "eval"]
        batch["pred_dict"], batch["label_dict"], log_dict = self.model(
            input_feature_dict=batch["input_feature_dict"],
            label_dict=batch["label_dict"],
            label_full_dict=batch["label_full_dict"],
            mode=mode,
            current_step=self.step if mode == "train" else None,
            symmetric_permutation=self.symmetric_permutation,
            mc_dropout_apply_rate=(
                0 if mode == "train" else self.configs.mc_dropout_apply_rate
            ),
        )
        return batch, log_dict

    def get_loss(
        self, batch: Dict[str, Any], mode: str = "train"
    ) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]:
        """
        Compute the loss for a given batch.

        Args:
            batch (Dict[str, Any]): Batch containing features, labels, and predictions.
            mode (str): Execution mode, either 'train' or 'eval'.

        Returns:
            Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]:
                - Total loss tensor.
                - Dictionary containing individual loss components.
                - Updated batch dictionary.
        """
        assert mode in ["train", "eval"]

        loss, loss_dict = autocasting_disable_decorator(self.configs.skip_amp.loss)(
            self.loss
        )(
            feat_dict=batch["input_feature_dict"],
            pred_dict=batch["pred_dict"],
            label_dict=batch["label_dict"],
            mode=mode,
        )
        return loss, loss_dict, batch

    @torch.no_grad()
    def get_metrics(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute evaluation metrics (e.g., lDDT) for a batch.

        Args:
            batch (Dict[str, Any]): Batch containing predictions and labels.

        Returns:
            Dict[str, Any]: Dictionary containing computed metrics.
        """
        lddt_dict = self.lddt_metrics.compute_lddt(
            batch["pred_dict"], batch["label_dict"]
        )
        return lddt_dict

    @torch.no_grad()
    def aggregate_metrics(
        self, lddt_dict: Dict[str, Any], batch: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Aggregate metrics across the batch.

        Args:
            lddt_dict (Dict[str, Any]): Dictionary of per-structure metrics.
            batch (Dict[str, Any]): Batch containing confidence summaries.

        Returns:
            Dict[str, Any]: Aggregated metrics.
        """
        simple_metrics, _ = self.lddt_metrics.aggregate_lddt(
            lddt_dict, batch["pred_dict"]["summary_confidence"]
        )
        return simple_metrics

    @torch.no_grad()
    def evaluate(self, mode: str = "eval") -> None:
        """
        Evaluate the model on all test datasets.
        Handles both standard and EMA model evaluation if applicable.

        Args:
            mode (str): Execution mode, typically 'eval'.
        """
        if not self.configs.eval_ema_only:
            self._evaluate(mode=mode)
        if hasattr(self, "ema_wrapper"):
            self.ema_wrapper.apply_shadow()
            self._evaluate(ema_suffix=f"ema{self.ema_wrapper.decay}_", mode=mode)
            self.ema_wrapper.restore()

    @torch.no_grad()
    def _evaluate(self, ema_suffix: str = "", mode: str = "eval") -> None:
        """
        Internal evaluation loop for a specific model state.

        Args:
            ema_suffix (str): Suffix for metric names if using EMA.
            mode (str): Execution mode.
        """
        # Init Metric Aggregator
        simple_metric_wrapper = SimpleMetricAggregator(["avg"])
        eval_precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.configs.dtype]
        enable_amp = (
            torch.autocast(device_type="cuda", dtype=eval_precision)
            if torch.cuda.is_available()
            else nullcontext()
        )
        self.model.eval()

        for test_name, test_dl in self.test_dls.items():
            self.print(f"Testing on {test_name}")
            evaluated_pids = []
            total_batch_num = len(test_dl)
            for index, batch in enumerate(tqdm(test_dl)):
                batch = to_device(batch, self.device)
                pid = batch["basic"]["pdb_id"]

                if index + 1 == total_batch_num and DIST_WRAPPER.world_size > 1:
                    # Gather all pids across ranks to avoid duplicate evaluation
                    # when drop_last=False
                    all_data_ids = DIST_WRAPPER.all_gather_object(evaluated_pids)
                    dedup_ids = set(sum(all_data_ids, []))
                    if pid in dedup_ids:
                        print(
                            f"Rank {DIST_WRAPPER.rank}: Dropping data_id {pid} "
                            "as it is already evaluated."
                        )
                        break
                evaluated_pids.append(pid)

                simple_metrics = {}
                with enable_amp:
                    # Model forward
                    batch, _ = self.model_forward(batch, mode=mode)
                    # Loss forward
                    _, loss_dict, batch = self.get_loss(batch, mode="eval")
                    # lDDT metrics
                    lddt_dict = self.get_metrics(batch)
                    lddt_metrics = self.aggregate_metrics(lddt_dict, batch)
                    simple_metrics.update(
                        {k: v for k, v in lddt_metrics.items() if "diff" not in k}
                    )
                    simple_metrics.update(loss_dict)

                    # RNA-only internal LDDT for monitoring
                    is_rna = batch["input_feature_dict"].get("is_rna", None)
                    if is_rna is not None and is_rna.sum() > 0:
                        rna_lddt_mask = (
                            batch["label_dict"]["lddt_mask"]
                            * is_rna.float().unsqueeze(-1)
                            * is_rna.float().unsqueeze(-2)
                        )
                        if rna_lddt_mask.sum() > 0:
                            rna_lddt = self.lddt_metrics.lddt_base.forward(
                                pred_coordinate=batch["pred_dict"]["coordinate"],
                                true_coordinate=batch["label_dict"]["coordinate"],
                                lddt_mask=rna_lddt_mask,
                                chunk_size=self.lddt_metrics.chunk_size,
                            )  # [N_sample]
                            simple_metrics["rna_lddt/mean"] = rna_lddt.mean()
                            simple_metrics["rna_lddt/best"] = rna_lddt.max()

                # Update metric aggregator
                for key, value in simple_metrics.items():
                    simple_metric_wrapper.add(
                        f"{ema_suffix}{key}", value, namespace=test_name
                    )

                del batch, simple_metrics
                if index % 5 == 0:
                    # Release memory periodically
                    torch.cuda.empty_cache()

            metrics = simple_metric_wrapper.calc()
            self.print(f"Step {self.step}, eval {test_name}: {metrics}")
            if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
                wandb.log(metrics, step=self.step)

    def update(self) -> None:
        """
        Apply gradient clipping to model parameters.
        """
        if self.configs.grad_clip_norm != 0.0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.configs.grad_clip_norm
            )

    def train_step(self, batch: Dict[str, Any]) -> None:
        """
        Perform a single training step including forward, backward, and optimization.

        Args:
            batch (Dict[str, Any]): Input batch for training.
        """
        self.model.train()
        train_precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.configs.dtype]
        enable_amp = (
            torch.autocast(
                device_type="cuda", dtype=train_precision, cache_enabled=False
            )
            if torch.cuda.is_available()
            else nullcontext()
        )

        scaler = torch.GradScaler(
            device="cuda" if torch.cuda.is_available() else "cpu",
            enabled=(self.configs.dtype == "float16"),
        )

        with enable_amp:
            batch, _ = self.model_forward(batch, mode="train")
            loss, loss_dict, _ = self.get_loss(batch, mode="train")

        if self.configs.dtype in ["bf16", "fp32"]:
            if is_loss_nan_check(loss):
                self.print(f"Skip iteration with NaN loss at step {self.step}")
                loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
        scaler.scale(loss / self.iters_to_accumulate).backward()

        # Global training step used for accumulation logic
        if (self.global_step + 1) % self.iters_to_accumulate == 0:
            # Unscale gradients before clipping
            scaler.unscale_(self.optimizer)
            # Apply gradient clipping
            self.update()
            # Optimizer and scaler step
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.lr_scheduler.step()
            self._apply_per_group_lr()

        for key, value in loss_dict.items():
            if "loss" not in key:
                continue
            self.train_metric_wrapper.add(key, value, namespace="train")
        torch.cuda.empty_cache()

    def progress_bar(self, desc: str = "") -> None:
        """
        Update the training progress bar on the master rank.

        Args:
            desc (str): Description to display on the progress bar.
        """
        if DIST_WRAPPER.rank != 0:
            return
        eval_steps = self.configs.eval_interval * self.iters_to_accumulate
        if self.global_step % eval_steps == 0 or (not hasattr(self, "_ipbar")):
            # Start a new progress bar
            self._pbar = tqdm(
                range(
                    self.global_step % eval_steps,
                    eval_steps,
                )
            )
            self._ipbar = iter(self._pbar)

        try:
            step = next(self._ipbar)
            self._pbar.set_description(
                f"[step {self.step}: {step}/{eval_steps}] {desc}"
            )
        except StopIteration:
            # If the iterator is exhausted, just return
            pass

    def run(self) -> None:
        """
        Main entry point for the AF3Trainer.
        Handles the complete training cycle including evaluation, logging, and checkpointing.
        Supports two-stage training: Stage 1 (adapter warmup) -> Stage 2 (joint training).
        """
        if self.configs.eval_only or self.configs.eval_first:
            self.evaluate()
            if self.configs.eval_only:
                return
        use_ema = hasattr(self, "ema_wrapper")
        self.print(f"Using EMA: {use_ema}")

        # Two-stage training: determine stage1 boundary
        stage1_max_steps = 0
        if self.two_stage_enable:
            stage1_max_steps = self.configs.two_stage.get("stage1_max_steps", 400)
            self.print(
                f"[Two-Stage] Stage 1: steps 0-{stage1_max_steps}, "
                f"Stage 2: steps {stage1_max_steps}-{self.configs.max_steps}"
            )

        while True:
            for batch in self.train_dl:
                is_update_step = (self.global_step + 1) % self.iters_to_accumulate == 0
                is_last_step = (self.step + 1) == self.configs.max_steps
                step_need_log = (self.step + 1) % self.configs.log_interval == 0

                step_need_eval = (
                    self.configs.eval_interval > 0
                    and (self.step + 1) % self.configs.eval_interval == 0
                )
                step_need_save = (
                    self.configs.checkpoint_interval > 0
                    and (self.step + 1) % self.configs.checkpoint_interval == 0
                )

                is_last_step &= is_update_step
                step_need_log &= is_update_step
                step_need_eval &= is_update_step
                step_need_save &= is_update_step

                batch = to_device(batch, self.device)
                self.progress_bar()
                self.train_step(batch)

                use_ema = hasattr(self, "ema_wrapper")
                if use_ema and is_update_step:
                    self.ema_wrapper.update()

                if step_need_log or is_last_step:
                    metrics = self.train_metric_wrapper.calc()
                    stage_tag = f"[Stage {self.current_stage}] " if self.two_stage_enable else ""
                    self.print(f"{stage_tag}Step {self.step} train metrics: {metrics}")
                    # Show actual param group LRs (may differ from scheduler with per-group LR)
                    if self._lr_group_config is not None:
                        last_lr = [g["lr"] for g in self.optimizer.param_groups]
                    else:
                        last_lr = self.lr_scheduler.get_last_lr()
                    if DIST_WRAPPER.rank == 0:
                        if self.configs.use_wandb:
                            lr_dict = {"train/lr": last_lr[0]}
                            for group_i, group_lr in enumerate(last_lr):
                                lr_dict[f"train/group{group_i}_lr"] = group_lr
                            wandb.log(lr_dict, step=self.step)
                        self.print(f"Step {self.step}, learning rate: {last_lr}")
                        if self.configs.use_wandb:
                            wandb.log(metrics, step=self.step)

                if step_need_save or is_last_step:
                    self.save_checkpoint()
                    if use_ema:
                        self.ema_wrapper.apply_shadow()
                        self.save_checkpoint(
                            ema_suffix=f"_ema_{self.ema_wrapper.decay}"
                        )
                        self.ema_wrapper.restore()

                if step_need_eval or is_last_step:
                    self.evaluate()

                self.global_step += 1
                if self.global_step % self.iters_to_accumulate == 0:
                    self.step += 1

                # === Two-stage: transition from Stage 1 to Stage 2 ===
                if (
                    self.two_stage_enable
                    and self.current_stage == 1
                    and self.step >= stage1_max_steps
                ):
                    self.print(
                        f"[Two-Stage] Stage 1 complete at step {self.step}. "
                        f"Transitioning to Stage 2..."
                    )
                    # Save Stage 1 checkpoint
                    self.save_checkpoint()
                    self._transition_to_stage2()
                    self.print(f"[Two-Stage] Stage 2 started at step {self.step}")

                if self.step >= self.configs.max_steps:
                    self.print(f"Finished training after {self.step} steps")
                    break
            if self.step >= self.configs.max_steps:
                break


def main() -> None:
    """
    Parse configurations and start the training process.
    """
    log_format = (
        "%(asctime)s,%(msecs)-3d %(levelname)-8s "
        "[%(filename)s:%(lineno)s %(funcName)s] %(message)s"
    )
    logging.basicConfig(
        format=log_format,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
    )
    configs_base["triangle_attention"] = os.environ.get(
        "TRIANGLE_ATTENTION", "cuequivariance"
    )
    configs_base["triangle_multiplicative"] = os.environ.get(
        "TRIANGLE_MULTIPLICATIVE", "cuequivariance"
    )
    arg_str = parse_sys_args()
    configs = {**configs_base, **{"data": data_configs}}
    # 1. First pass to get model_name
    configs = parse_configs(
        configs,
        arg_str=arg_str,
        fill_required_with_null=True,
    )
    model_name = configs.model_name

    # 2. Get model specifics and merge into base defaults
    base_configs = {**configs_base, **{"data": data_configs}}
    model_specfics_configs = model_configs[model_name]

    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, Mapping) and k in d and isinstance(d[k], Mapping):
                deep_update(d[k], v)
            else:
                d[k] = v
        return d

    deep_update(base_configs, model_specfics_configs)

    # 3. Second pass to apply sys_args with higher priority
    configs = parse_configs(
        configs=base_configs,
        arg_str=arg_str,
        fill_required_with_null=True,
    )
    logging.info(
        f"Using params for model {model_name}: "
        f"cycle={configs.model.N_cycle}, step={configs.sample_diffusion.N_step}"
    )
    print(f"Run Name: {configs.run_name}")
    trainer = AF3Trainer(configs)
    trainer.run()


if __name__ == "__main__":
    main()
