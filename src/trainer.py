"""
Core Trainer for DDP / FSDP distributed training.
"""

import contextlib
import gc
import logging
import math
import os
import time
from datetime import timedelta
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Mapping, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from hydra.utils import instantiate
from torch.utils.data import DataLoader, DistributedSampler

from src.utils.dist import get_machine_local_and_dist_rank
from src.utils.env import apply_pytorch_env_defaults, setup_env_variables
from src.utils.freeze import freeze_modules
from src.utils.fsdp import fsdp_full_state_dict, wrap_fsdp
from src.utils.general import (
    AverageMeter,
    DurationMeter,
    ProgressMeter,
    chunk_batch,
    copy_data_to_device,
    find_latest_checkpoint,
    is_dist_avail_and_initialized,
    model_summary,
    safe_makedirs,
    set_seeds,
)
from src.utils.logging import setup_logging
from src.utils.checkpoint import robust_torch_save
from src.utils.optimizer import construct_optimizers


@dataclass(eq=False)
class Trainer:
    """Orchestrates the full DDP/FSDP training lifecycle.

    The constructor receives **flat keyword args** — each top-level Hydra config
    key (``data``, ``model``, ``optim``, ``loss``, …) arrives as a separate dict.
    """

    EPSILON: ClassVar[float] = 1e-8

    # Required fields
    data: Dict[str, Any]
    model: Dict[str, Any]
    logging: Dict[str, Any]
    checkpoint: Dict[str, Any]
    max_epochs: int
    # Optional fields with defaults
    exp_name: str = ""
    mode: str = "train"
    device: str = "cuda"
    seed_value: int = 42
    val_epoch_freq: int = 1
    distributed: Optional[Dict[str, Any]] = None
    cuda: Optional[Dict[str, Any]] = None
    limit_train_batches: Optional[int] = None
    limit_val_batches: Optional[int] = None
    num_workers: int = 4
    batch_size: int = 32
    optim: Optional[Dict[str, Any]] = None
    loss: Optional[Dict[str, Any]] = None
    env_variables: Optional[Dict[str, Any]] = None
    accum_steps: int = 1

    def __post_init__(self):
        # ---- 1. Env vars (before anything reads them) ----
        apply_pytorch_env_defaults()
        setup_env_variables(self.env_variables)
        self._start_time = time.time()
        self._ckpt_time_elapsed = 0

        # Config aliases (for internal use throughout the class)
        self.data_conf = self.data
        self.model_conf = self.model
        self.loss_conf = self.loss
        self.logging_conf = self.logging
        self.checkpoint_conf = self.checkpoint
        self.optim_conf = self.optim
        self.distributed_conf = self.distributed or {}

        self.where = 0.0  # training progress ∈ [0, 1]

        # ---- 2. Device setup ----
        self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()
        device_str = self.device
        if device_str == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        else:
            self.device = torch.device("cpu")

        # ---- 3. Distributed init ----
        self._setup_backend(self.cuda)

        # ---- 4. Logging ----
        self.rank = dist.get_rank() if is_dist_avail_and_initialized() else 0
        safe_makedirs(self.logging_conf["log_dir"])
        setup_logging(
            __name__,
            output_dir=self.logging_conf["log_dir"],
            rank=self.rank,
            log_level_primary=self.logging_conf.get("log_level_primary", "INFO"),
            log_level_secondary=self.logging_conf.get("log_level_secondary", "WARNING"),
            all_ranks=self.logging_conf.get("all_ranks", False),
        )
        set_seeds(self.seed_value, self.max_epochs, self.distributed_rank)

        # ---- 5. Components ----
        self._setup_components()

        # ---- 6. Dataloaders ----
        self._setup_dataloaders()

        # ---- 7. Move model to device ----
        self.model.to(self.device)
        self.time_elapsed_meter = DurationMeter("Time Elapsed")

        # ---- 8. Optimizers ----
        if self.mode != "val":
            self.optims = construct_optimizers(self.model, self.optim_conf)

        # ---- 9. Checkpoint resume (before DDP/FSDP wrap) ----
        self.epoch = 0
        self.steps = {"train": 0, "val": 0}
        resume_path = self.checkpoint_conf.get("resume_checkpoint_path")
        if resume_path is not None:
            self._load_checkpoint(resume_path)
        else:
            auto = find_latest_checkpoint(self.checkpoint_conf["save_dir"])
            if auto is not None:
                self._load_checkpoint(auto)

        # ---- 10. Wrap model (DDP or FSDP) ----
        self._wrap_model()

        # ---- 11. Barrier ----
        if is_dist_avail_and_initialized():
            dist.barrier()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _setup_backend(self, cuda_conf):
        if cuda_conf and torch.cuda.is_available():
            torch.backends.cudnn.deterministic = cuda_conf.get("cudnn_deterministic", False)
            torch.backends.cudnn.benchmark = cuda_conf.get("cudnn_benchmark", False)
            torch.backends.cuda.matmul.allow_tf32 = cuda_conf.get("allow_tf32", True)
            torch.backends.cudnn.allow_tf32 = cuda_conf.get("allow_tf32", True)

        backend = self.distributed_conf.get("backend", "nccl")
        timeout = self.distributed_conf.get("timeout_mins", 30)
        if (
            torch.cuda.is_available()
            and not is_dist_avail_and_initialized()
            and "RANK" in os.environ
        ):
            dist.init_process_group(backend=backend, timeout=timedelta(minutes=timeout))

    def _setup_components(self):
        logging.info("Instantiating components …")
        self.tb_writer = instantiate(self.logging_conf.get("tensorboard_writer"), _recursive_=False)
        self.model = instantiate(self.model_conf, _recursive_=False)
        self.loss_fn = instantiate(self.loss_conf, _recursive_=False) if self.loss_conf else None

        # Gradient clipper (may be None if configs list is empty)
        gc_conf = self.optim_conf.get("gradient_clip") if self.optim_conf else None
        if gc_conf and gc_conf.get("configs"):
            self.gradient_clipper = instantiate(gc_conf)
        else:
            self.gradient_clipper = None

        # AMP scaler
        amp_enabled = self.optim_conf.get("amp", {}).get("enabled", False) if self.optim_conf else False
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        # Freeze modules
        frozen = self.optim_conf.get("frozen_module_names") if self.optim_conf else None
        if frozen:
            logging.info(f"Freezing modules matching: {frozen}")
            self.model = freeze_modules(self.model, patterns=frozen)

        if self.rank == 0:
            summary_path = os.path.join(self.logging_conf["log_dir"], "model.txt")
            model_summary(self.model, log_file=summary_path)

    def _setup_dataloaders(self):
        self.train_dataset = None
        self.val_dataset = None

        num_workers = self.num_workers
        batch_size = self.batch_size

        if self.mode in ("train", "val") and "val" in self.data_conf:
            val_ds = instantiate(self.data_conf["val"], _recursive_=False)
            sampler = DistributedSampler(val_ds, shuffle=False) if is_dist_avail_and_initialized() else None
            self.val_loader = DataLoader(
                val_ds, batch_size=batch_size, sampler=sampler,
                num_workers=num_workers, pin_memory=True, shuffle=False,
            )
        else:
            self.val_loader = None

        if self.mode == "train" and "train" in self.data_conf:
            train_ds = instantiate(self.data_conf["train"], _recursive_=False)
            sampler = DistributedSampler(train_ds, shuffle=True) if is_dist_avail_and_initialized() else None
            self.train_loader = DataLoader(
                train_ds, batch_size=batch_size, sampler=sampler,
                num_workers=num_workers, pin_memory=True,
                shuffle=(sampler is None),
            )
        else:
            self.train_loader = None

    def _wrap_model(self):
        strategy = self.distributed_conf.get("strategy", "ddp")

        if strategy == "fsdp":
            fsdp_conf = dict(self.distributed_conf.get("fsdp", {}))
            # Pass amp_dtype into FSDP mixed precision
            amp_dtype = self.optim_conf.get("amp", {}).get("amp_dtype") if self.optim_conf else None
            fsdp_conf["amp_dtype"] = amp_dtype
            self.model = wrap_fsdp(self.model, fsdp_conf, device_id=self.local_rank)
            logging.info("Model wrapped with FSDP")
        elif is_dist_avail_and_initialized():
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank] if self.device.type == "cuda" else [],
                find_unused_parameters=self.distributed_conf.get("find_unused_parameters", False),
                gradient_as_bucket_view=self.distributed_conf.get("gradient_as_bucket_view", True),
                bucket_cap_mb=self.distributed_conf.get("bucket_cap_mb", 25),
                broadcast_buffers=self.distributed_conf.get("broadcast_buffers", True),
            )
            logging.info("Model wrapped with DDP")

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def _load_checkpoint(self, path: str):
        logging.info(f"Resuming from {path} (rank {self.rank})")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        state = ckpt.get("model", ckpt)
        missing, unexpected = self.model.load_state_dict(state, strict=self.checkpoint_conf.get("strict", False))
        if self.rank == 0:
            logging.info(f"Loaded model — missing: {missing or 'none'}, unexpected: {unexpected or 'none'}")

        if "optimizer" in ckpt and self.mode != "val":
            for optim in self.optims:
                try:
                    optim.optimizer.load_state_dict(ckpt["optimizer"])
                except ValueError:
                    logging.warning("Optimizer state dict mismatch (param count changed) — starting optimizer from scratch")

        self.epoch = ckpt.get("prev_epoch", ckpt.get("epoch", 0))
        self.steps = ckpt.get("steps", {"train": 0, "val": 0})
        self._ckpt_time_elapsed = ckpt.get("time_elapsed", 0)

        if self.optim_conf and self.optim_conf.get("amp", {}).get("enabled") and "scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler"])

    def save_checkpoint(self, epoch: int, names: Optional[List[str]] = None):
        folder = self.checkpoint_conf["save_dir"]
        safe_makedirs(folder)

        if names is None:
            names = ["checkpoint"]
            freq = self.checkpoint_conf.get("save_freq", 0)
            if freq > 0 and epoch % freq == 0 and epoch > 0:
                names.append(f"checkpoint_{epoch}")

        content = {
            "prev_epoch": epoch,
            "steps": self.steps,
            "time_elapsed": self.time_elapsed_meter.val,
        }

        # Optimizer state
        if hasattr(self, "optims") and self.optims:
            opt_states = [o.optimizer.state_dict() for o in self.optims]
            content["optimizer"] = opt_states[0] if len(opt_states) == 1 else opt_states

        if self.scaler.is_enabled():
            content["scaler"] = self.scaler.state_dict()

        # Extract unwrapped model
        strategy = self.distributed_conf.get("strategy", "ddp")
        if strategy == "fsdp":
            model_state = fsdp_full_state_dict(self.model)
        elif isinstance(self.model, nn.parallel.DistributedDataParallel):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()

        if self.distributed_rank == 0:
            content["model"] = model_state
            for name in names:
                path = os.path.join(folder, f"{name}.pt")
                logging.info(f"Saving checkpoint epoch={epoch} → {path}")
                robust_torch_save(content, path)

    # ------------------------------------------------------------------
    # AMP dtype helper
    # ------------------------------------------------------------------

    def _get_amp_dtype(self):
        amp_str = self.optim_conf.get("amp", {}).get("amp_dtype", "bfloat16") if self.optim_conf else "bfloat16"
        return torch.bfloat16 if amp_str == "bfloat16" else torch.float16

    def _amp_enabled(self):
        return self.optim_conf.get("amp", {}).get("enabled", False) if self.optim_conf else False

    # ------------------------------------------------------------------
    # Scalar logging helpers
    # ------------------------------------------------------------------

    def _get_scalar_keys(self, phase: str) -> List[str]:
        keys_conf = self.logging_conf.get("scalar_keys_to_log")
        if keys_conf and phase in keys_conf:
            return keys_conf[phase].get("keys_to_log", [])
        return []

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self):
        assert self.mode in ("train", "val")
        try:
            if self.mode == "train":
                self.run_train()
                self.run_val()
            else:
                self.run_val()
        finally:
            if is_dist_avail_and_initialized():
                dist.destroy_process_group()

    def run_train(self):
        while self.epoch < self.max_epochs:
            set_seeds(self.seed_value + self.epoch * 100, self.max_epochs, self.distributed_rank)

            if self.train_loader is None:
                logging.warning("No training dataloader. Skipping.")
                return

            # Set epoch on distributed sampler
            if hasattr(self.train_loader, "sampler") and hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(self.epoch)

            self.train_epoch(self.train_loader)
            self.save_checkpoint(self.epoch)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            if self.epoch % self.val_epoch_freq == 0 and self.epoch < self.max_epochs - 1:
                self.run_val()

            self.epoch += 1

    def run_val(self):
        if self.val_loader is None:
            logging.info("No validation loader. Skipping.")
            return
        self.val_epoch(self.val_loader)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def train_epoch(self, loader):
        batch_time = AverageMeter("Batch", self.device, ":.4f")
        data_time = AverageMeter("Data", self.device, ":.4f")
        mem = AverageMeter("Mem(GB)", self.device, ":.1f")
        phase = "train"

        loss_keys = self._get_scalar_keys(phase)
        loss_meters = {f"Loss/{phase}_{k}": AverageMeter(f"Loss/{phase}_{k}", self.device, ":.4f") for k in loss_keys}

        # Gradient norm meters
        if self.gradient_clipper:
            for cfg in self.gradient_clipper.configs:
                key = ",".join(cfg["module_names"])
                loss_meters[f"Grad/{key}"] = AverageMeter(f"Grad/{key}", self.device, ":.4f")

        progress = ProgressMeter(
            len(loader),
            meters=[batch_time, data_time, mem, self.time_elapsed_meter, *loss_meters.values()],
            prefix=f"Train Epoch: [{self.epoch}]",
        )

        self.model.train()
        end = time.time()

        iters = len(loader)
        limit = iters if self.limit_train_batches is None else self.limit_train_batches

        if self.gradient_clipper is not None:
            self.gradient_clipper.setup_clipping(
                self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
            )

        for i, batch in enumerate(loader):
            if i > limit:
                break

            data_time.update(time.time() - end)
            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            # Split for gradient accumulation
            chunks = chunk_batch(batch, self.accum_steps)
            self._train_chunks(chunks, phase, loss_meters)

            # Scheduler step
            exact_epoch = self.epoch + float(i) / limit
            self.where = float(exact_epoch) / self.max_epochs
            if self.where < 1.0:
                for optim in self.optims:
                    optim.step_schedulers(self.where)

            # Log scheduler values
            if self.steps[phase] % self.logging_conf.get("log_freq", 10) == 0:
                self._log_optim_state(phase)

            # Clip gradients
            if self.gradient_clipper is not None:
                for optim in self.optims:
                    self.scaler.unscale_(optim.optimizer)
                norms = self.gradient_clipper(model=self.model)
                for key, val in norms.items():
                    loss_meters[f"Grad/{key}"].update(val)

            # Optimizer step
            for optim in self.optims:
                self.scaler.step(optim.optimizer)
            self.scaler.update()

            batch_time.update(time.time() - end)
            end = time.time()
            self.time_elapsed_meter.update(time.time() - self._start_time + self._ckpt_time_elapsed)

            if torch.cuda.is_available():
                mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if i % self.logging_conf.get("log_freq", 10) == 0:
                progress.display(i)

    def _train_chunks(self, chunks: List[Any], phase: str, loss_meters: dict):
        for optim in self.optims:
            optim.zero_grad(set_to_none=True)

        n = len(chunks)
        amp_dtype = self._get_amp_dtype()
        amp_on = self._amp_enabled()

        for idx, chunk in enumerate(chunks):
            # Suppress DDP sync on all but the last chunk
            sync_ctx = self.model.no_sync() if (idx < n - 1 and hasattr(self.model, "no_sync")) else contextlib.nullcontext()

            with sync_ctx:
                with torch.amp.autocast("cuda", enabled=amp_on, dtype=amp_dtype):
                    loss_dict = self._step(chunk, self.model, phase, loss_meters)

                loss = loss_dict["loss"]
                if not math.isfinite(loss.item()):
                    logging.error(f"Loss is {loss.item()}, skipping batch")
                    return

                loss = loss / n
                self.scaler.scale(loss).backward()

    # ------------------------------------------------------------------
    # Validation epoch
    # ------------------------------------------------------------------

    @torch.no_grad()
    def val_epoch(self, loader):
        batch_time = AverageMeter("Batch", self.device, ":.4f")
        data_time = AverageMeter("Data", self.device, ":.4f")
        mem = AverageMeter("Mem(GB)", self.device, ":.1f")
        phase = "val"

        loss_keys = self._get_scalar_keys(phase)
        loss_meters = {f"Loss/{phase}_{k}": AverageMeter(f"Loss/{phase}_{k}", self.device, ":.4f") for k in loss_keys}

        progress = ProgressMeter(
            len(loader),
            meters=[batch_time, data_time, mem, self.time_elapsed_meter, *loss_meters.values()],
            prefix=f"Val Epoch: [{self.epoch}]",
        )

        self.model.eval()
        end = time.time()

        iters = len(loader)
        limit = iters if self.limit_val_batches is None else self.limit_val_batches

        amp_dtype = self._get_amp_dtype()
        amp_on = self._amp_enabled()

        for i, batch in enumerate(loader):
            if i > limit:
                break

            data_time.update(time.time() - end)
            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=amp_on, dtype=amp_dtype):
                self._step(batch, self.model, phase, loss_meters)

            batch_time.update(time.time() - end)
            end = time.time()
            self.time_elapsed_meter.update(time.time() - self._start_time + self._ckpt_time_elapsed)

            if torch.cuda.is_available():
                mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if i % self.logging_conf.get("log_freq", 10) == 0:
                progress.display(i)

    # ------------------------------------------------------------------
    # Single forward step
    # ------------------------------------------------------------------

    def _step(self, batch: Mapping, model: nn.Module, phase: str, loss_meters: dict) -> dict:
        """Forward pass → loss → log. Returns loss dict."""
        outputs = model(**self._model_inputs(batch))
        loss_dict = self._compute_loss(outputs, batch)

        self._update_scalar_logs(loss_dict, batch, phase, self.steps[phase], loss_meters)
        self.steps[phase] += 1
        return loss_dict

    def _model_inputs(self, batch: Mapping) -> dict:
        """Extract model forward kwargs from a batch dict.

        Override this in subclasses for custom batch layouts.
        """
        if "video" in batch:
            out = {"x": batch["video"]}
            if "lengths" in batch:
                out["lengths"] = batch["lengths"]
            return out
        if "image" in batch:
            return {"x": batch["image"]}
        if "images" in batch:
            return {"images": batch["images"]}
        return dict(batch)

    def _compute_loss(self, outputs, batch) -> dict:
        """Compute loss from model outputs and batch targets.

        Override for custom loss logic. Returns dict with at least ``"loss"`` key.
        """
        if self.loss_fn is None:
            raise ValueError("No loss function configured")

        if getattr(self.loss_fn, "expects_batch", False) or hasattr(self.loss_fn, "dice_weight") or hasattr(self.loss_fn, "lam_curv"):
            loss = self.loss_fn(outputs, batch)
        else:
            targets = batch.get("label", batch.get("labels", batch.get("target")))
            loss = self.loss_fn(outputs, targets)
            
        if isinstance(loss, dict):
            if "loss" not in loss:
                raise ValueError("Loss dict must contain a 'loss' key")
            return loss
            
        return {"loss": loss}

    def _update_scalar_logs(self, loss_dict, batch, phase, step, loss_meters):
        keys = self._get_scalar_keys(phase)
        bs = len(next(iter(batch.values()))) if isinstance(batch, Mapping) else 1
        for key in keys:
            if key in loss_dict:
                val = loss_dict[key].item() if torch.is_tensor(loss_dict[key]) else loss_dict[key]
                meter_key = f"Loss/{phase}_{key}"
                if meter_key in loss_meters:
                    loss_meters[meter_key].update(val, bs)
                if step % self.logging_conf.get("log_freq", 10) == 0 and self.rank == 0:
                    self.tb_writer.log(f"Values/{phase}/{key}", val, step)

    def _log_optim_state(self, phase):
        if self.rank != 0 or not hasattr(self, "optims"):
            return
        for i, optim in enumerate(self.optims):
            for j, pg in enumerate(optim.optimizer.param_groups):
                if optim.schedulers and j < len(optim.schedulers):
                    for option in optim.schedulers[j]:
                        prefix = f"{i}_" if len(self.optims) > 1 else ""
                        prefix += f"{j}_" if len(optim.optimizer.param_groups) > 1 else ""
                        self.tb_writer.log(f"Optim/{prefix}{option}", pg[option], self.steps[phase])
        self.tb_writer.log("Optim/where", self.where, self.steps[phase])
