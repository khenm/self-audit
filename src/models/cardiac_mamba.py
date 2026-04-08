import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.models.mamba2.block import Mamba2Block
from src.models.mamba2.vss import PureMambaEncoder, PureMambaDecoder

logger = logging.getLogger(__name__)

class MambaBlock(nn.Module):
    """
    Wrapper for Mamba-2 Block.
    """
    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.inner = Mamba2Block(dim=d_model, **kwargs)

    def forward(self, x):
        return self.inner(x)

    def step(self, x, state=None):
        return self.inner.step(x, state)

class PhaseHead(nn.Module):
    """
    Predicts the cardiac phase (0: Systole, 1: Diastole) from the temporal hidden state.
    """
    def __init__(self, hidden_dim: int, num_phases: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, num_phases)
        )

    def forward(self, h_t):
        return self.net(h_t)

class VolumeDerivation(nn.Module):
    """
    Volume Derivation Module.
    Predicts volume from soft area using a pointwise parameterized function.
    V = c * A^gamma
    """
    def __init__(self, init_log_c: float = -6.907755, init_gamma_raw: float = 1.5):
        super().__init__()
        # math.log(0.001) ~ -6.907755
        self.log_c = nn.Parameter(torch.tensor(float(init_log_c)))
        # math.log(math.exp(1.5) - 1.0) ~ 1.2475
        gamma_raw_init = math.log(math.exp(init_gamma_raw) - 1.0) if init_gamma_raw > 0 else 1.5
        self.gamma_raw = nn.Parameter(torch.tensor(float(gamma_raw_init)))

    def forward(self, mask_logits):
        """
        mask_logits: (B, T, 1, H, W) or (B, 1, H, W)
        """
        mask_probs = torch.sigmoid(mask_logits)
        if mask_probs.ndim == 5:
            a_t = mask_probs.view(mask_probs.shape[0], mask_probs.shape[1], -1).sum(dim=-1, keepdim=True)
        else:
            a_t = mask_probs.view(mask_probs.shape[0], -1).sum(dim=-1, keepdim=True)
        
        c = torch.exp(self.log_c)
        gamma = F.softplus(self.gamma_raw)
        
        v_t = c * (a_t ** gamma)
        return v_t


class CardiacMamba(nn.Module):
    """
    Streaming Cardiac Model using a Unified Pure Mamba Architecture.

    Instantiated via Hydra ``_target_: src.models.cardiac_mamba.CardiacMamba``.
    """
    def __init__(
        self,
        hidden_dim: int = 256,
        output_size: int = 112,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        chunk_size: int = 32,
    ):
        super().__init__()

        self.output_size = output_size

        # 1. Spatial Encoder (Pure Mamba)
        self.encoder = PureMambaEncoder(
            in_channels=3,
            embed_dim=96,
            depths=[2, 2, 4, 2],
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            chunk_size=chunk_size
        )
        self.feature_dim = self.encoder.feature_dims[-1]

        # Clean Linear Adapter mapping spatial bottleneck -> temporal token
        self.temporal_adapter = nn.Linear(self.feature_dim + 1, hidden_dim)

        # 2. Segmentation Decoder (Frame-wise Native Projection)
        self.decoder = PureMambaDecoder(
            encoder_channels=self.encoder.feature_dims,
            output_channels=1,
            depths=[1, 1, 1, 1],
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            chunk_size=chunk_size
        )

        # 3. Temporal Module (Mamba)
        self.temporal_mamba = MambaBlock(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            chunk_size=chunk_size,
        )

        # 4. Multi-Task Heads
        self.phase_head = PhaseHead(hidden_dim, num_phases=2)
        self.volume_derivation = VolumeDerivation()

        logger.info("CardiacMamba initialized with PureMamba backbone, PhaseHead, and VolumeDerivation module")

    def forward_spatial(self, x):
        enc_feats = self.encoder(x)
        bottleneck = enc_feats[-1]

        mask_logits = self.decoder(enc_feats)

        mask_probs = torch.sigmoid(mask_logits)

        B_T = mask_probs.shape[0]
        z_t = mask_probs.view(B_T, -1).sum(dim=1, keepdim=True) / 1000.0

        mask_downsampled = F.adaptive_avg_pool2d(mask_probs, bottleneck.shape[2:])

        attended_bottleneck = bottleneck * mask_downsampled
        pooled = F.adaptive_avg_pool2d(attended_bottleneck, 1).flatten(1)

        # Append sensor reading z_t to pooled features
        pooled_with_z = torch.cat([pooled, z_t], dim=1)
        temporal_tokens = self.temporal_adapter(pooled_with_z)

        if self.output_size is not None and mask_logits.shape[-1] != self.output_size:
            mask_logits = F.interpolate(
                mask_logits,
                size=(self.output_size, self.output_size),
                mode='bilinear',
                align_corners=False
            )

        return mask_logits, temporal_tokens, z_t

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None, **kwargs) -> dict:
        B, C, T, H, W = x.shape
        x_flat = x.transpose(1, 2).reshape(B * T, C, H, W)

        mask_logits_flat, tokens_flat, _ = self.forward_spatial(x_flat)

        mask_logits = mask_logits_flat.view(B, T, 1, *mask_logits_flat.shape[2:])
        tokens = tokens_flat.view(B, T, -1)

        temporal_out = self.temporal_mamba(tokens)

        phase_logits = self.phase_head(temporal_out)

        vol_curve = self.volume_derivation(mask_logits)

        phase_probs = torch.softmax(phase_logits, dim=-1)
        p_es = phase_probs[..., 0]
        p_ed = phase_probs[..., 1]

        if lengths is not None:
            mask_t = (torch.arange(T, device=x.device)[None, :] < lengths[:, None])
            p_es = p_es * mask_t
            p_ed = p_ed * mask_t

        vols = vol_curve.squeeze(-1)
        pred_edv = torch.sum(p_ed * vols, dim=1) / torch.sum(p_ed, dim=1).clamp(min=1e-3)
        pred_esv = torch.sum(p_es * vols, dim=1) / torch.sum(p_es, dim=1).clamp(min=1e-3)

        return {
            "mask_logits": mask_logits.transpose(1, 2),
            "pred_vol_curve": vol_curve,
            "pred_phase_logits": phase_logits,
            "pred_edv": pred_edv,
            "pred_esv": pred_esv,
            "pred_ef": (pred_edv - pred_esv) / torch.clamp(pred_edv, min=1e-3),
            "hidden_features": temporal_out,
            "log_c": self.volume_derivation.log_c,
            "gamma_raw": self.volume_derivation.gamma_raw
        }

    def step(self, x: torch.Tensor, state=None):
        if state is None:
            state = {'mamba_state': None}

        mamba_state = state.get('mamba_state')

        # 1. Spatial
        mask_logits, token, _ = self.forward_spatial(x)

        # 2. Temporal Step
        token_seq = token.unsqueeze(1)
        temporal_out_seq, next_mamba_state = self.temporal_mamba.step(token_seq, mamba_state)
        temporal_out = temporal_out_seq

        # 3. Heads
        temporal_out_flat = temporal_out.squeeze(1)
        phase_logits = self.phase_head(temporal_out_flat)
        vol_out = self.volume_derivation(mask_logits)

        next_state = {
            'mamba_state': next_mamba_state,
        }

        return {
            "mask_logits": mask_logits,
            "pred_vol": vol_out.squeeze(-1),
            "pred_phase_logits": phase_logits,
            "hidden_features": temporal_out_flat,
            "log_c": self.volume_derivation.log_c,
            "gamma_raw": self.volume_derivation.gamma_raw
        }, next_state
