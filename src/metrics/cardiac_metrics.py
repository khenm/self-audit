"""
Cardiac evaluation metrics for EchoNet finetuning.

Accumulated per-batch and computed at the end of a validation epoch.

Metrics:
  EDV: MAE, RMSE (mL)
  ESV: MAE, RMSE (mL)
  EF:  MAE, RMSE, R²  (fraction, 0–1)
  Dice: mean binary Dice at ED/ES frames (0–1)
"""

import torch


class CardiacMetrics:
    """
    Accumulates per-sample predictions and ground-truth values across an epoch,
    then computes summary statistics on `.compute()`.

    Usage::

        metrics = CardiacMetrics()
        for batch in loader:
            outputs = model(...)
            metrics.update(outputs, batch)
        results = metrics.compute()
        metrics.reset()
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._pred_edv: list[float] = []
        self._gt_edv: list[float] = []
        self._pred_esv: list[float] = []
        self._gt_esv: list[float] = []
        self._pred_ef: list[float] = []
        self._gt_ef: list[float] = []
        self._dice: list[float] = []

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(self, outputs: dict, batch: dict) -> None:
        """
        Accumulate one batch.

        outputs keys used:
          pred_edv  (B,)  — soft phase-weighted EDV prediction
          pred_esv  (B,)  — soft phase-weighted ESV prediction
          pred_ef   (B,)  — derived EF = (EDV - ESV) / EDV
          mask_logits (B, 1, T, H, W)

        batch keys used:
          target_edv  (B,)  — GT EDV in mL; <0 means invalid
          target_esv  (B,)  — GT ESV in mL; <0 means invalid
          target_ef   (B,)  — GT EF as fraction; <0 means invalid
          label       (B, 1, T, H, W)  — binary GT masks
          frame_mask  (B, T)  — 2=ED, 1=ES, 0=none
        """
        mask_logits = outputs.get("mask_logits")  # (B, 1, T, H, W)
        label = batch.get("label")                # (B, 1, T, H, W)
        frame_mask = batch.get("frame_mask")      # (B, T)

        # --- volume / EF ---
        for key, pred_key, gt_key in [
            ("edv", "pred_edv", "target_edv"),
            ("esv", "pred_esv", "target_esv"),
            ("ef",  "pred_ef",  "target_ef"),
        ]:
            pred_t = outputs.get(pred_key)
            gt_t = batch.get(gt_key)
            if pred_t is None or gt_t is None:
                continue
            valid = gt_t >= 0
            if not valid.any():
                continue
            p = pred_t[valid].detach().cpu().float()
            g = gt_t[valid].detach().cpu().float()
            getattr(self, f"_pred_{key}").extend(p.tolist())
            getattr(self, f"_gt_{key}").extend(g.tolist())

        # --- Dice at ED/ES frames ---
        if mask_logits is None or label is None or frame_mask is None:
            return

        mask_pred = (torch.sigmoid(mask_logits) > 0.5).float()  # (B, 1, T, H, W)
        label_f = label.float()
        B, _, T, H, W = mask_logits.shape
        valid_frames = frame_mask > 0.5  # (B, T) — ED or ES

        for b in range(B):
            for t in range(T):
                if not valid_frames[b, t]:
                    continue
                pred_hw = mask_pred[b, 0, t]   # (H, W)
                gt_hw = label_f[b, 0, t]       # (H, W)
                intersection = (pred_hw * gt_hw).sum()
                denom = pred_hw.sum() + gt_hw.sum()
                if denom > 0:
                    self._dice.append((2.0 * intersection / denom).item())

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def compute(self) -> dict[str, float]:
        """Return dict of metric_name → value. Missing metrics are omitted."""
        results: dict[str, float] = {}

        def _mae(pred, gt):
            p = torch.tensor(pred)
            g = torch.tensor(gt)
            return (p - g).abs().mean().item()

        def _rmse(pred, gt):
            p = torch.tensor(pred)
            g = torch.tensor(gt)
            return ((p - g).pow(2).mean()).sqrt().item()

        def _r2(pred, gt):
            p = torch.tensor(pred)
            g = torch.tensor(gt)
            ss_res = (p - g).pow(2).sum()
            ss_tot = (g - g.mean()).pow(2).sum().clamp(min=1e-8)
            return (1.0 - ss_res / ss_tot).item()

        if self._gt_edv:
            results["edv_mae"] = _mae(self._pred_edv, self._gt_edv)
            results["edv_rmse"] = _rmse(self._pred_edv, self._gt_edv)

        if self._gt_esv:
            results["esv_mae"] = _mae(self._pred_esv, self._gt_esv)
            results["esv_rmse"] = _rmse(self._pred_esv, self._gt_esv)

        if self._gt_ef:
            results["ef_mae"] = _mae(self._pred_ef, self._gt_ef)
            results["ef_rmse"] = _rmse(self._pred_ef, self._gt_ef)
            results["ef_r2"] = _r2(self._pred_ef, self._gt_ef)

        if self._dice:
            results["dice"] = sum(self._dice) / len(self._dice)

        return results

    def __repr__(self) -> str:
        n = len(self._gt_ef)
        nd = len(self._dice)
        return f"CardiacMetrics(n_ef={n}, n_dice_frames={nd})"
