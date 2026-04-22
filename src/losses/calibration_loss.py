import torch
import torch.nn as nn
import torch.nn.functional as F


class CalibrationLoss(nn.Module):
    """
    Phase 1b — calibrates c and γ in V = c·A^γ while the rest of the model is frozen.

    Gradients flow only to volume_derivation.log_c, volume_derivation.gamma_raw,
    and volume_derivation.bias.
    The .detach() on mask_logits is the critical invariant: it guarantees zero gradient
    reaches the segmentation head regardless of whether its parameters are frozen.

    Supervision:
    - L1 on EDV at GT ED frames and ESV at GT ES frames
    - L1 on EF derived from those same GT-indexed volumes

    target_edv / target_esv are expected to be normalised by the dataset
    (divided by 300 mL); target_ef is normalised by 100. CalibrationLoss works
    in the same normalised spaces.
    """

    expects_batch = True

    def __init__(self, lam_ef: float = 1.0):
        super().__init__()
        self.lam_ef = lam_ef

    def forward(self, outputs: dict, batch: dict) -> dict:
        # (B, 1, T, H, W) — detach: no gradient to encoder / decoder
        mask_logits = outputs["mask_logits"].detach()
        log_c = outputs["log_c"]        # nn.Parameter — has grad
        gamma_raw = outputs["gamma_raw"]  # nn.Parameter — has grad
        bias = outputs["bias"]            # nn.Parameter — has grad

        frame_mask = batch["frame_mask"]           # (B, T)
        target_edv = batch.get("target_edv")       # (B,) normalised by 300
        target_esv = batch.get("target_esv")       # (B,) normalised by 300
        target_ef  = batch.get("target_ef")        # (B,) normalised by 100

        B, _, T, H, W = mask_logits.shape

        # Area per frame — purely from the frozen seg head, no grad pathway
        mask_probs = torch.sigmoid(mask_logits).squeeze(1)  # (B, T, H, W)
        areas = mask_probs.reshape(B, T, -1).sum(dim=-1) / (H * W)  # (B, T)

        # V = c · A^γ + bias — the only trainable leaf nodes are log_c, gamma_raw, and bias
        c = torch.exp(log_c)
        gamma = F.softplus(gamma_raw)
        # Clamp areas away from 0 before raising to gamma to avoid 0^gamma NaN gradients
        vol_curve = c * (areas.clamp(min=1e-6) ** gamma) + bias  # (B, T)

        # Anchor losses to the computation graph so backward() always works in DDP
        # even when a rank receives a batch with no labeled ED/ES frames.
        loss_vol = vol_curve.sum() * 0.0
        loss_ef  = vol_curve.sum() * 0.0
        n_vol = 0
        n_ef  = 0

        for b in range(B):
            b_fm = frame_mask[b]
            ed_idx = torch.where(b_fm == 2.0)[0]
            es_idx = torch.where(b_fm == 1.0)[0]

            has_ed = len(ed_idx) > 0 and target_edv is not None and target_edv[b] >= 0
            has_es = len(es_idx) > 0 and target_esv is not None and target_esv[b] >= 0

            if has_ed:
                loss_vol = loss_vol + torch.abs(vol_curve[b, ed_idx[0]] - target_edv[b])
                n_vol += 1
            if has_es:
                loss_vol = loss_vol + torch.abs(vol_curve[b, es_idx[0]] - target_esv[b])
                n_vol += 1

            if has_ed and has_es and target_ef is not None and target_ef[b] >= 0:
                pred_edv_b = vol_curve[b, ed_idx[0]]
                pred_esv_b = vol_curve[b, es_idx[0]]
                pred_ef_b  = (pred_edv_b - pred_esv_b) / pred_edv_b.clamp(min=1e-3)
                loss_ef = loss_ef + torch.abs(pred_ef_b - target_ef[b])
                n_ef += 1

        if n_vol > 0:
            loss_vol = loss_vol / n_vol
        if n_ef > 0:
            loss_ef = loss_ef / n_ef

        loss = loss_vol + self.lam_ef * loss_ef

        return {
            "loss": loss,
            "loss_vol": loss_vol.detach(),
            "loss_ef": loss_ef.detach(),
        }
