import argparse
import logging
import random
import os
from typing import List, Tuple

import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.utils.general import copy_data_to_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Clean matplotlib styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Roboto", "Helvetica Neue", "Arial"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

def overlay_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[float, float, float] = (0, 1, 0), alpha: float = 0.4) -> np.ndarray:
    """Overlays a binary mask on an RGB image."""
    result = image.copy()
    for c in range(3):
        result[..., c] = np.where(mask, result[..., c] * (1 - alpha) + color[c] * alpha, result[..., c])
    return result

def plot_videos(
    videos_data: List[dict], 
    save_path: str = "chunked_streaming_plot.png",
    samples: int = 16
):
    """
    Plots stacked video predictions.
    videos_data contains: video, mask_pred, vol_curve, ed_frame, es_frame, target_edv, target_esv
    """
    n = len(videos_data)
    
    # Each video takes 2 rows (images, plot).
    # We want the plot to be "shorter in vertical". Images row: height ratio 1.5, Plot row: height ratio 1
    # Adjust total height based on number of videos.
    fig_width = 16
    fig_height_per_video = 4.0
    
    fig = plt.figure(figsize=(fig_width, fig_height_per_video * n), constrained_layout=True)
    
    # Create gridspec with 2*n rows
    height_ratios = []
    for _ in range(n):
        height_ratios.extend([1.5, 0.8])  # Plot is shorter than images
    
    gs_main = gridspec.GridSpec(2 * n, 1, figure=fig, height_ratios=height_ratios, hspace=0.1)
    
    for i, data in enumerate(videos_data):
        video = data["video"]  # (C, T, H, W)
        mask_pred = data["mask_pred"] # (T, H, W)
        vol_curve = data["vol_curve"] # (T,)
        ed_frame = data["ed_frame"]
        es_frame = data["es_frame"]
        target_edv = data["target_edv"]
        target_esv = data["target_esv"]
        case = data.get("case", f"Video {i+1}")
        
        T = video.shape[1]
        
        # Sample frames to display
        # We ensure ED and ES are included if they exist, then fill the rest.
        sampled_indices = []
        if ed_frame != -1 and ed_frame < T: sampled_indices.append(ed_frame)
        if es_frame != -1 and es_frame < T: sampled_indices.append(es_frame)
        
        remaining = samples - len(sampled_indices)
        if remaining > 0:
            # linear spacing
            lins = np.linspace(0, T - 1, remaining, dtype=int)
            sampled_indices.extend(lins)
            
        sampled_indices = sorted(list(set(sampled_indices)))
        
        # If we still have too few or too many due to set, just linspace strictly
        if len(sampled_indices) > samples:
            sampled_indices = sampled_indices[:samples] # Fallback
        elif len(sampled_indices) < samples:
            missing = samples - len(sampled_indices)
            pool = list(set(range(T)) - set(sampled_indices))
            np.random.shuffle(pool)
            sampled_indices.extend(pool[:missing])
            sampled_indices = sorted(sampled_indices)
        
        # ---- Row 1: Images Grid ----
        gs_images = gridspec.GridSpecFromSubplotSpec(1, samples, subplot_spec=gs_main[2 * i], wspace=0.02)
        
        for j, t in enumerate(sampled_indices):
            ax_img = fig.add_subplot(gs_images[0, j])
            
            img = video[:, t, :, :].cpu().numpy().transpose(1, 2, 0)
            img = np.clip(img, 0, 1) # Ensure [0, 1] range just in case
            
            mask = mask_pred[t].cpu().numpy()
            
            # Highlight border if it's an ED or ES frame
            border_color = None
            border_lw = 0
            if t == ed_frame:
                border_color = 'limegreen'
                border_lw = 4
            elif t == es_frame:
                border_color = 'mediumorchid'
                border_lw = 4
                
            img_disp = overlay_mask(img, mask, color=(0, 1, 0), alpha=0.5)
            ax_img.imshow(img_disp)
            ax_img.axis('off')
            
            # Simple top-left label
            ax_img.text(0.05, 0.95, f"F:{t}", transform=ax_img.transAxes, 
                        color="white", fontsize=8, fontweight='bold',
                        va='top', ha='left',
                        bbox=dict(facecolor='black', alpha=0.5, pad=1, edgecolor='none'))
            
            if border_color:
                rect = plt.Rectangle((0, 0), img.shape[1], img.shape[0], 
                                     linewidth=border_lw, edgecolor=border_color, facecolor='none')
                ax_img.add_patch(rect)
                
        # ---- Row 2: Volume Plot ----
        ax_plot = fig.add_subplot(gs_main[2 * i + 1])
        ax_plot.set_title(f"Inferred Volume Curve - Case: {case}", fontsize=10, pad=4)
        
        x_axis = np.arange(T)
        vol_curve_np = vol_curve.cpu().numpy()
        
        # Plot continuous predicted curve
        ax_plot.plot(x_axis, vol_curve_np, label="Predicted Volume (mL)", color="royalblue", linewidth=2)
        
        # Plot GT markers
        if ed_frame != -1 and ed_frame < T:
            ax_plot.axvline(x=ed_frame, color='limegreen', linestyle='--', linewidth=1.5, label=f"ED (Frame {ed_frame})")
            if target_edv > 0:
                ax_plot.scatter([ed_frame], [target_edv], color='darkgreen', marker='X', s=80, zorder=5, label=f"GT EDV")
                ax_plot.text(ed_frame, target_edv + 3, f"GT: {target_edv:.1f}", color='darkgreen', fontsize=8, ha='center', va='bottom', fontweight='bold')
                
        if es_frame != -1 and es_frame < T:
            ax_plot.axvline(x=es_frame, color='mediumorchid', linestyle='--', linewidth=1.5, label=f"ES (Frame {es_frame})")
            if target_esv > 0:
                ax_plot.scatter([es_frame], [target_esv], color='purple', marker='X', s=80, zorder=5, label=f"GT ESV")
                ax_plot.text(es_frame, target_esv - 3, f"GT: {target_esv:.1f}", color='purple', fontsize=8, ha='center', va='top', fontweight='bold')
                
        ax_plot.set_xlim(0, T - 1)
        # Margin for Y
        y_min, y_max = vol_curve_np.min(), vol_curve_np.max()
        ax_plot.set_ylim(max(0, y_min - 20), y_max + 20)
        
        ax_plot.set_xlabel("Frame Number", fontsize=9, labelpad=2)
        ax_plot.set_ylabel("Volume (mL)", fontsize=9, labelpad=2)
        
        ax_plot.legend(loc='upper right', fontsize=8, frameon=True, framealpha=0.9, edgecolor='none')
        
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    logging.info(f"Plot saved to {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="finetune")
    parser.add_argument("--mode", type=str, choices=["chunked", "streaming"], default="chunked")
    parser.add_argument("--n", type=int, default=3, help="Number of random videos to plot")
    parser.add_argument("--samples", type=int, default=16, help="Number of frames to sample and display per video")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="volume_plot.png")
    parser.add_argument("--inspect", action="store_true", help="Print per-video areas, c, and gamma instead of plotting")
    args, overrides = parser.parse_known_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    overrides.append("+data.test.video_level=true")
    overrides.append("data.batch_size=1") 

    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name=args.config, overrides=overrides)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info(f"Instantiating test dataset...")
    test_ds = instantiate(cfg.data.test, _recursive_=False)
    
    # Pick n random indices
    if len(test_ds) < args.n:
        args.n = len(test_ds)
    rand_indices = random.sample(range(len(test_ds)), args.n)
    subset = Subset(test_ds, rand_indices)
    
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=1)

    logging.info("Instantiating model...")
    model = instantiate(cfg.model, _recursive_=False)
    model.to(device)

    resume_path = cfg.checkpoint.get("resume_checkpoint_path")
    if resume_path and os.path.exists(resume_path):
        logging.info(f"Loading checkpoint from {resume_path}...")
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=cfg.checkpoint.get("strict", False))
    else:
        logging.warning("No checkpoint found. Plotting with random weights!")

    model.eval()
    
    amp_enabled = cfg.optim.get("amp", {}).get("enabled", False) if cfg.get("optim") else False
    if cfg.get("optim") and cfg.optim.get("amp", {}).get("amp_dtype") == "bfloat16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16
        
    chunk_size = cfg.model.get("chunk_size", cfg.model.get("max_clip_len", 32))
    
    results = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc=f"Processing {args.n} videos ({args.mode} mode)")):
            batch = copy_data_to_device(batch, device, non_blocking=True)
            video = batch.get("video")
            if video is None:
                continue
                
            B, C, T_total, H, W = video.shape
            
            with torch.amp.autocast("cuda", enabled=amp_enabled and torch.cuda.is_available(), dtype=amp_dtype):
                if args.mode == "streaming":
                    inputs = {"x": video, "lengths": batch.get("lengths")}
                    outputs = model(**inputs)
                elif args.mode == "chunked":
                    all_mask_logits = []
                    all_vol_curves = []
                    for t_start in range(0, T_total, chunk_size):
                        t_end = min(T_total, t_start + chunk_size)
                        video_chunk = video[:, :, t_start:t_end]
                        chunk_lengths = torch.tensor([video_chunk.shape[2]], device=device)
                        inputs_chunk = {"x": video_chunk, "lengths": chunk_lengths}
                        chunk_out = model(**inputs_chunk)
                        all_mask_logits.append(chunk_out["mask_logits"])
                        all_vol_curves.append(chunk_out["pred_vol_curve"])
                    mask_logits_cat = torch.cat(all_mask_logits, dim=2)
                    vol_curve_cat = torch.cat(all_vol_curves, dim=1)
                    outputs = {
                        "mask_logits": mask_logits_cat,
                        "pred_vol_curve": vol_curve_cat,
                    }
                    
            # Parse outputs
            mask_logits = outputs["mask_logits"][0, 0] # (T, H, W)
            vol_curve = outputs["pred_vol_curve"][0] # (T, 1) or (T,)
            if vol_curve.dim() > 1:
                vol_curve = vol_curve.squeeze(-1)
                
            mask_pred = (torch.sigmoid(mask_logits) > 0.5)
            
            # Predict Volume curve scaling (EchoNet targets are divided by 300.0)
            vol_curve = vol_curve * 300.0
            
            # Frame marks
            frame_mask = batch.get("frame_mask")[0] # (T,) => 2.0 is ED, 1.0 is ES
            ed_frame = (frame_mask == 2.0).nonzero(as_tuple=True)[0]
            es_frame = (frame_mask == 1.0).nonzero(as_tuple=True)[0]
            
            ed_frame_idx = ed_frame[0].item() if len(ed_frame) > 0 else -1
            es_frame_idx = es_frame[0].item() if len(es_frame) > 0 else -1
            
            target_edv = batch.get("target_edv")[0].item() * 300.0 if "target_edv" in batch else -1.0
            target_esv = batch.get("target_esv")[0].item() * 300.0 if "target_esv" in batch else -1.0
            
            case_name = batch.get("case", [f"Unknown"])[0]

            if args.inspect:
                log_c = outputs.get("log_c")
                gamma_raw = outputs.get("gamma_raw")
                bias_param = outputs.get("bias")
                c_val = torch.exp(log_c).item() if log_c is not None else float("nan")
                gamma_val = F.softplus(gamma_raw).item() if gamma_raw is not None else float("nan")
                bias_val = bias_param.item() if bias_param is not None else float("nan")

                mask_logits_full = outputs["mask_logits"][0, 0]  # (T, H, W)
                mask_probs_full = torch.sigmoid(mask_logits_full)
                H_m, W_m = mask_probs_full.shape[-2], mask_probs_full.shape[-1]
                areas_per_frame = mask_probs_full.reshape(T_total, -1).sum(dim=-1) / (H_m * W_m)  # (T,)

                print(f"\n=== {case_name} ===")
                print(f"  c      = {c_val:.6f}")
                print(f"  gamma  = {gamma_val:.6f}")
                print(f"  bias   = {bias_val:.6f}")
                print(f"  Areas (normalized, per frame):")
                for t_idx, a in enumerate(areas_per_frame.cpu().tolist()):
                    print(f"    frame {t_idx:3d}: {a:.6f}")

            results.append({
                "video": video[0], 
                "mask_pred": mask_pred,
                "vol_curve": vol_curve,
                "ed_frame": ed_frame_idx,
                "es_frame": es_frame_idx,
                "target_edv": target_edv,
                "target_esv": target_esv,
                "case": case_name
            })
            
    plot_videos(results, save_path=args.save_path, samples=args.samples)

if __name__ == "__main__":
    main()
