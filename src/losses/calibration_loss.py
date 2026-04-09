import torch
import torch.nn as nn
import torch.nn.functional as F


class CalibrationLoss(nn.Module):
    """
    Phase 1b — calibrates c and γ in V = c·A^γ while the rest of the model is frozen.

    Gradients flow only to volume_derivation.log_c and volume_derivation.gamma_raw.
    The .detach() on mask_logits is the critical invariant: it guarantees zero gradient
    reaches the segmentation head regardless of whether its parameters are frozen.

    Supervision: L1 against ground-truth EDV at ED frames and ESV at ES frames
    (identified by frame_mask == 2.0 and frame_mask == 1.0 respectively).

    target_edv / target_esv are expected to be normalised by the dataset
    (divided by 300 mL); CalibrationLoss works in the same normalised space.
    """

    def forward(self, outputs: dict, batch: dict) -> dict:
        device = outputs["mask_logits"].device

        # (B, 1, T, H, W) — detach: no gradient to encoder / decoder
        mask_logits = outputs["mask_logits"].detach()
        log_c = outputs["log_c"]        # nn.Parameter — has grad
        gamma_raw = outputs["gamma_raw"]  # nn.Parameter — has grad

        frame_mask = batch["frame_mask"]           # (B, T)
        target_edv = batch.get("target_edv")       # (B,) normalised
        target_esv = batch.get("target_esv")       # (B,) normalised

        B, _, T, H, W = mask_logits.shape

        # Area per frame — purely from the frozen seg head, no grad pathway
        mask_probs = torch.sigmoid(mask_logits).squeeze(1)  # (B, T, H, W)
        areas = mask_probs.reshape(B, T, -1).sum(dim=-1)    # (B, T)

        # V = c · A^γ — the only trainable leaf nodes are log_c and gamma_raw
        c = torch.exp(log_c)
        gamma = F.softplus(gamma_raw)
        vol_curve = c * (areas ** gamma)  # (B, T)

        loss_vol = torch.tensor(0.0, device=device)
        n = 0

        for b in range(B):
            b_fm = frame_mask[b]
            ed_idx = torch.where(b_fm == 2.0)[0]
            es_idx = torch.where(b_fm == 1.0)[0]

            if len(ed_idx) > 0 and target_edv is not None and target_edv[b] >= 0:
                loss_vol = loss_vol + torch.abs(vol_curve[b, ed_idx[0]] - target_edv[b])
                n += 1
            if len(es_idx) > 0 and target_esv is not None and target_esv[b] >= 0:
                loss_vol = loss_vol + torch.abs(vol_curve[b, es_idx[0]] - target_esv[b])
                n += 1

        if n > 0:
            loss_vol = loss_vol / n

        return {"loss": loss_vol, "loss_vol": loss_vol.detach()}
