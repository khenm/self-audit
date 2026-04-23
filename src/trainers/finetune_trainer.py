"""
FinetuneTrainer — extends Trainer with cardiac evaluation metrics.

Overrides val_epoch to accumulate:
  EDV MAE / RMSE
  ESV MAE / RMSE
  EF  MAE / RMSE / R²
  Dice (mean binary Dice at ED/ES frames)

All metrics are logged to TensorBoard under Metrics/val/<name>.
"""

import logging
import time

import torch

from src.trainer import Trainer
from src.metrics.cardiac_metrics import CardiacMetrics
from src.utils.general import (
    AverageMeter,
    DurationMeter,
    ProgressMeter,
    copy_data_to_device,
    is_dist_avail_and_initialized,
)


class FinetuneTrainer(Trainer):
    """Drop-in replacement for Trainer that adds cardiac metrics at validation."""

    def _load_checkpoint(self, path: str):
        super()._load_checkpoint(path)
        reinit_names = (self.optim_conf or {}).get("reinit_module_names", [])
        if not reinit_names:
            return
        for name, mod in self.model.named_modules():
            if name in reinit_names:
                if hasattr(mod, "reset_parameters"):
                    mod.reset_parameters()
                    logging.info(f"Reinitialized module after checkpoint load: {name}")
                else:
                    logging.warning(f"Module '{name}' has no reset_parameters() — skipping reinit")

    @torch.no_grad()
    def val_epoch(self, loader):
        batch_time = AverageMeter("Batch", self.device, ":.4f")
        data_time = AverageMeter("Data", self.device, ":.4f")
        mem = AverageMeter("Mem(GB)", self.device, ":.1f")
        phase = "val"

        loss_keys = self._get_scalar_keys(phase)
        loss_meters = {
            f"Loss/{phase}_{k}": AverageMeter(f"Loss/{phase}_{k}", self.device, ":.4f")
            for k in loss_keys
        }

        progress = ProgressMeter(
            len(loader),
            meters=[batch_time, data_time, mem, self.time_elapsed_meter, *loss_meters.values()],
            prefix=f"Val Epoch: [{self.epoch}]",
        )

        self.model.eval()
        metrics = CardiacMetrics()
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
                # Forward
                outputs = self.model(**self._model_inputs(batch))
                loss_dict = self._compute_loss(outputs, batch)

            # Loss logging
            self._update_scalar_logs(loss_dict, batch, phase, self.steps[phase], loss_meters)
            self.steps[phase] += 1

            # Metrics accumulation
            metrics.update(outputs, batch)

            batch_time.update(time.time() - end)
            end = time.time()
            self.time_elapsed_meter.update(time.time() - self._start_time + self._ckpt_time_elapsed)

            if torch.cuda.is_available():
                mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if i % self.logging_conf.get("log_freq", 10) == 0:
                progress.display(i)

        # Compute and log metrics after full pass
        self._log_metrics(metrics)

    def _log_metrics(self, metrics: CardiacMetrics) -> None:
        results = metrics.compute()
        if not results:
            return

        if self.rank == 0:
            lines = ["=" * 50, f"Val Epoch {self.epoch} — Cardiac Metrics"]
            for k, v in results.items():
                lines.append(f"  {k:20s}: {v:.4f}")
            lines.append("=" * 50)
            logging.info("\n".join(lines))

            for k, v in results.items():
                self.tb_writer.log(f"Metrics/val/{k}", v, self.epoch)
