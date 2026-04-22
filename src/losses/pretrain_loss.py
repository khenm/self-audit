import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss

class PretrainLoss(nn.Module):
    """
    Final Pretraining Loss:
    L = L_mask + λ_curv · L_curv + λ_prior · L_prior
    """
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        lam_curv: float = 0.01,
        lam_prior: float = 0.001,
        log_c_init: float = 1.0202,
        **kwargs
    ):
        super().__init__()
        self.expects_batch = True
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.lam_curv = lam_curv
        self.lam_prior = lam_prior
        self.log_c_init = log_c_init

        self.dice_ce = DiceCELoss(
            sigmoid=True, 
            lambda_dice=self.dice_weight, 
            lambda_ce=self.bce_weight,
            reduction='mean'
        )

    def forward(self, outputs, batch):
        mask_logits = outputs['mask_logits'] # assumed (B, 1, T, H, W)
        batch_masks = batch['label']         # assumed (B, 1, T, H, W)
        lengths = batch['lengths']           # (B,)
        
        device = mask_logits.device
        
        # 1. L_mask: Dice + BCE
        _, _, T, _, _ = mask_logits.shape
        t_indices = torch.arange(T, device=device)[None, :]
        valid_mask = t_indices < lengths[:, None] # (B, T)
        
        # Flatten sequences onto batch dimension
        ml_flat = mask_logits.transpose(1, 2) # (B, T, 1, H, W)
        tg_flat = batch_masks.float().transpose(1, 2) # (B, T, 1, H, W)
        
        ml_valid = ml_flat[valid_mask] # (N, 1, H, W)
        tg_valid = tg_flat[valid_mask] # (N, 1, H, W)
        
        if ml_valid.numel() > 0:
            loss_mask = self.dice_ce(ml_valid, tg_valid)
        else:
            loss_mask = torch.tensor(0.0, device=device)
            
        # 2. L_curv: Mean curvature penalty (finite differences, 2nd derivative)
        vol_curve = outputs['pred_vol_curve'].squeeze(-1) # (B, T)
        loss_curv = torch.tensor(0.0, device=device)
        
        if T >= 3:
            d2_vol = vol_curve[:, :-2] - 2 * vol_curve[:, 1:-1] + vol_curve[:, 2:]
            valid_curv_mask = valid_mask[:, 2:]
            
            if valid_curv_mask.any():
                loss_curv = (d2_vol[valid_curv_mask] ** 2).mean()

        # 3. L_prior: L2 on (γ - 1.5)² + (log_c - log_c_init)² + bias²
        gamma_raw = outputs['gamma_raw']
        gamma = F.softplus(gamma_raw)
        log_c = outputs['log_c']
        bias = outputs['bias']

        loss_prior = (gamma - 1.5)**2 + (log_c - self.log_c_init)**2 + bias**2

        loss = loss_mask + self.lam_curv * loss_curv + self.lam_prior * loss_prior
        
        return {
            "loss": loss,
            "loss_mask": loss_mask.detach(),
            "loss_curv": loss_curv.detach(),
            "loss_prior": loss_prior.detach()
        }

    @classmethod
    def from_config(cls, cfg):
        loss_cfg = cfg.get("loss", {})
        return cls(
            bce_weight=loss_cfg.get("bce_weight", 0.5),
            dice_weight=loss_cfg.get("dice_weight", 0.5),
            lam_curv=loss_cfg.get("lam_curv", 0.01),
            lam_prior=loss_cfg.get("lam_prior", 0.001),
        )
