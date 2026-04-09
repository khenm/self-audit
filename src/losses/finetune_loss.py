import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss

class FinetuneLoss(nn.Module):
    """
    Final Finetuning Loss:
    L = L_mask + 0.1·L_vol + 0.75·L_ef + 0.1·L_phase + 0.005·L_curv + 0.01·L_phase_smooth + 1e-5·L_L1 + 0.0005·L_prior
    """
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        lam_vol: float = 0.1,
        lam_ef: float = 0.75,
        lam_phase: float = 0.1,
        lam_curv: float = 0.005,
        lam_phase_smooth: float = 0.01,
        lam_l1: float = 1e-5,
        lam_prior: float = 0.0005,
        log_c_init: float = -6.907755,
        **kwargs
    ):
        super().__init__()
        self.expects_batch = True
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        self.lam_vol = lam_vol
        self.lam_ef = lam_ef
        self.lam_phase = lam_phase
        self.lam_curv = lam_curv
        self.lam_phase_smooth = lam_phase_smooth
        self.lam_l1 = lam_l1
        self.lam_prior = lam_prior
        self.log_c_init = log_c_init
        
        self.dice_ce = DiceCELoss(
            sigmoid=True, 
            lambda_dice=self.dice_weight, 
            lambda_ce=self.bce_weight,
            reduction='mean'
        )

    def forward(self, outputs, batch):
        device = outputs['mask_logits'].device
        
        mask_logits = outputs['mask_logits'] # (B, 1, T, H, W)
        vol_curve = outputs['pred_vol_curve'].squeeze(-1) # (B, T)
        # model returns (B, T, 2); transpose to (B, 2, T) for consistent indexing below
        _pl = outputs.get('pred_phase_logits')
        phase_logits = _pl.transpose(1, 2) if _pl is not None else None  # (B, 2, T)

        batch_masks = batch['label'] # (B, 1, T, H, W)
        frame_mask = batch['frame_mask'] # (B, T) where 2=ED, 1=ES
        lengths = batch['lengths'] # (B,)

        B, _, T, H, W = mask_logits.shape
        t_indices = torch.arange(T, device=device)[None, :]
        valid_length_mask = t_indices < lengths[:, None] # (B, T)

        # 1. L_mask: Dice + BCE at ED/ES frames
        loss_mask = torch.tensor(0.0, device=device)
        valid_frames = frame_mask > 0.5 # 1.0 or 2.0
        if valid_frames.any():
            ml_flat = mask_logits.transpose(1, 2) # (B, T, 1, H, W)
            tg_flat = batch_masks.float().transpose(1, 2)
            ml_valid = ml_flat[valid_frames]
            tg_valid = tg_flat[valid_frames]
            if ml_valid.numel() > 0:
                loss_mask = self.dice_ce(ml_valid, tg_valid)

        # GT values
        target_edv = batch.get('target_edv')
        target_esv = batch.get('target_esv')
        target_ef = batch.get('target_ef')

        loss_vol = torch.tensor(0.0, device=device)
        loss_ef = torch.tensor(0.0, device=device)
        loss_phase = torch.tensor(0.0, device=device)

        valid_vol_samples = 0
        valid_ef_samples = 0
        valid_phase_samples = 0

        # 2. L_vol and L_phase
        for b in range(B):
            b_frame_mask = frame_mask[b]
            ed_idx = torch.where(b_frame_mask == 2.0)[0]
            es_idx = torch.where(b_frame_mask == 1.0)[0]

            # Instantaneous predicted volume
            pred_edv_inst = vol_curve[b, ed_idx[0]] if len(ed_idx) > 0 else None
            pred_esv_inst = vol_curve[b, es_idx[0]] if len(es_idx) > 0 else None

            # L_vol (L1 vs GT)
            if target_edv is not None and pred_edv_inst is not None and target_edv[b] >= 0:
                loss_vol = loss_vol + torch.abs(pred_edv_inst - target_edv[b])
                valid_vol_samples += 1
            if target_esv is not None and pred_esv_inst is not None and target_esv[b] >= 0:
                loss_vol = loss_vol + torch.abs(pred_esv_inst - target_esv[b])
                valid_vol_samples += 1

            # L_phase: phase_logits is (B, 2, T); index as [b:b+1, :, t] → (1, 2)
            if phase_logits is not None:
                if len(ed_idx) > 0:
                    loss_phase += F.cross_entropy(phase_logits[b:b+1, :, ed_idx[0]], torch.tensor([1], device=device))
                    valid_phase_samples += 1
                if len(es_idx) > 0:
                    loss_phase += F.cross_entropy(phase_logits[b:b+1, :, es_idx[0]], torch.tensor([0], device=device))
                    valid_phase_samples += 1
                     
        if valid_vol_samples > 0: loss_vol = loss_vol / valid_vol_samples
        if valid_phase_samples > 0: loss_phase = loss_phase / valid_phase_samples
        
        # L_ef: Soft-weighted aggregation
        if target_ef is not None and phase_logits is not None:
            # phase_logits: (B, 2, T)
            probs = F.softmax(phase_logits, dim=1) # (B, 2, T)
            p_es = probs[:, 0, :] # (B, T)
            p_ed = probs[:, 1, :] # (B, T)

            # mask by length
            p_es = p_es * valid_length_mask.float()
            p_ed = p_ed * valid_length_mask.float()

            sum_p_es = p_es.sum(dim=1).clamp(min=1e-5)
            sum_p_ed = p_ed.sum(dim=1).clamp(min=1e-5)

            pred_esv_soft = (p_es * vol_curve).sum(dim=1) / sum_p_es
            pred_edv_soft = (p_ed * vol_curve).sum(dim=1) / sum_p_ed

            # clamp EDV to avoid div by zero
            pred_edv_soft_clamped = pred_edv_soft.clamp(min=1e-3)
            pred_ef_soft = (pred_edv_soft_clamped - pred_esv_soft) / pred_edv_soft_clamped

            valid_ef_mask = target_ef >= 0
            if valid_ef_mask.any():
                loss_ef = F.l1_loss(pred_ef_soft[valid_ef_mask], target_ef[valid_ef_mask])

        # L_curv
        loss_curv = torch.tensor(0.0, device=device)
        if T >= 3:
            d2_vol = vol_curve[:, :-2] - 2 * vol_curve[:, 1:-1] + vol_curve[:, 2:]
            valid_curv_mask = valid_length_mask[:, 2:]
            if valid_curv_mask.any():
                loss_curv = (d2_vol[valid_curv_mask] ** 2).mean()

        # L_phase_smooth
        loss_phase_smooth = torch.tensor(0.0, device=device)
        if phase_logits is not None and T >= 2:
            # phase_logits: (B, 2, T)
            diff = phase_logits[:, :, 1:] - phase_logits[:, :, :-1]
            valid_diff_mask = valid_length_mask[:, 1:]
            diff_sq = (diff ** 2).sum(dim=1) # sum over classes, shape (B, T-1)
            if valid_diff_mask.any():
                loss_phase_smooth = diff_sq[valid_diff_mask].mean()

        # L_prior
        loss_prior = torch.tensor(0.0, device=device)
        gamma_raw = outputs.get('gamma_raw')
        log_c = outputs.get('log_c')
        if gamma_raw is not None and log_c is not None:
            gamma = F.softplus(gamma_raw)
            loss_prior = (gamma - 1.5)**2 + (log_c - self.log_c_init)**2

        # L_L1
        loss_l1 = torch.tensor(0.0, device=device)
        l1_reg = outputs.get('l1_reg_loss')
        if l1_reg is not None:
            loss_l1 = l1_reg
             
        loss = (
            loss_mask + 
            self.lam_vol * loss_vol + 
            self.lam_ef * loss_ef + 
            self.lam_phase * loss_phase + 
            self.lam_curv * loss_curv + 
            self.lam_phase_smooth * loss_phase_smooth + 
            self.lam_l1 * loss_l1 + 
            self.lam_prior * loss_prior
        )
        
        return {
            "loss": loss,
            "loss_mask": loss_mask.detach(),
            "loss_vol": loss_vol.detach(),
            "loss_ef": loss_ef.detach(),
            "loss_phase": loss_phase.detach(),
            "loss_curv": loss_curv.detach(),
            "loss_phase_smooth": loss_phase_smooth.detach(),
            "loss_prior": loss_prior.detach(),
            "loss_l1": loss_l1.detach()
        }

    @classmethod
    def from_config(cls, cfg):
        loss_cfg = cfg.get("loss", {})
        return cls(
            bce_weight=loss_cfg.get("bce_weight", 0.5),
            dice_weight=loss_cfg.get("dice_weight", 0.5),
            lam_vol=loss_cfg.get("lam_vol", 0.1),
            lam_ef=loss_cfg.get("lam_ef", 0.75),
            lam_phase=loss_cfg.get("lam_phase", 0.1),
            lam_curv=loss_cfg.get("lam_curv", 0.005),
            lam_phase_smooth=loss_cfg.get("lam_phase_smooth", 0.01),
            lam_l1=loss_cfg.get("lam_l1", 1e-5),
            lam_prior=loss_cfg.get("lam_prior", 0.0005),
            log_c_init=loss_cfg.get("log_c_init", -6.907755),
        )
