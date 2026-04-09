"""
Test entrypoint — Evaluates the trained model on the test set, computing video-level cardiac metrics.
Implements Chunked Inference vs Streaming Inference modes for long sequences.

Usage:
    uv run python scripts/test.py --config finetune checkpoint.resume_checkpoint_path=/path/to/checkpoint.pt --mode streaming
"""

import argparse
import logging
from tqdm import tqdm

import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from torch.utils.data import DataLoader

from src.metrics.cardiac_metrics import CardiacMetrics
from src.utils.general import copy_data_to_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    parser = argparse.ArgumentParser(description="Evaluate model on test set using Video-Level Metrics")
    parser.add_argument("--config", type=str, default="finetune")
    parser.add_argument("--mode", type=str, choices=["chunked", "streaming"], default="chunked",
                        help="Inference mode: chunked (Mode A) or streaming (Mode B).")
    args, overrides = parser.parse_known_args()

    # Override the dataset configuration to ensure video_level=True
    overrides.append("+data.test.video_level=true")
    # For dataloaders we need bs=1 since videos have variable lengths
    overrides.append("data.batch_size=1") 

    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name=args.config, overrides=overrides)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "test" not in cfg.data:
        logging.error("No 'test' split found in dataset configuration! Please define 'test' in configs/dataset/*.yaml.")
        return

    # 1. Instantiate dataset and dataloader
    logging.info(f"Instantiating test dataset (Mode: {args.mode})...")
    test_ds = instantiate(cfg.data.test, _recursive_=False)
    
    num_workers = cfg.data.get("num_workers", 4)
    # Batch size must be 1 for variable-length full videos
    test_loader = DataLoader(
        test_ds, 
        batch_size=1, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )

    # 2. Instantiate model
    logging.info("Instantiating model...")
    model = instantiate(cfg.model, _recursive_=False)
    model.to(device)

    # 3. Load checkpoint
    resume_path = cfg.checkpoint.get("resume_checkpoint_path")
    if resume_path:
        logging.info(f"Loading checkpoint from {resume_path}...")
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=cfg.checkpoint.get("strict", False))
        logging.info(f"Loaded weights. Missing: {missing}, Unexpected: {unexpected}")
    else:
        logging.warning("No checkpoint provided. Evaluating with random weights!")

    model.eval()
    
    # 4. Evaluation Loop
    logging.info("Starting Video-Level evaluation...")
    metrics = CardiacMetrics()
    
    amp_enabled = cfg.optim.get("amp", {}).get("enabled", False) if cfg.get("optim") else False
    if cfg.get("optim") and cfg.optim.get("amp", {}).get("amp_dtype") == "bfloat16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16
        
    chunk_size = cfg.model.get("chunk_size", cfg.model.get("max_clip_len", 32))
    logging.info(f"Using Mode: {args.mode} (Chunk Size: {chunk_size})")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = copy_data_to_device(batch, device, non_blocking=True)
            
            video = batch.get("video")
            if video is None:
                continue
                
            # Full video shape: (1, C, T, H, W)
            B, C, T_total, H, W = video.shape
            
            with torch.amp.autocast("cuda", enabled=amp_enabled and torch.cuda.is_available(), dtype=amp_dtype):
                if args.mode == "streaming":
                    # Mode B: Causal processing over full sequence in one pass
                    inputs = {"x": video, "lengths": batch.get("lengths")}
                    outputs = model(**inputs)
                
                elif args.mode == "chunked":
                    # Mode A: Chunked independent processing
                    all_mask_logits = []
                    all_phase_logits = []
                    all_vol_curves = []
                    
                    for t_start in range(0, T_total, chunk_size):
                        t_end = min(T_total, t_start + chunk_size)
                        video_chunk = video[:, :, t_start:t_end]
                        chunk_lengths = torch.tensor([video_chunk.shape[2]], device=device)
                        
                        inputs_chunk = {"x": video_chunk, "lengths": chunk_lengths}
                        chunk_out = model(**inputs_chunk)
                        
                        all_mask_logits.append(chunk_out["mask_logits"])
                        all_phase_logits.append(chunk_out["pred_phase_logits"])
                        all_vol_curves.append(chunk_out["pred_vol_curve"])
                        
                    # Concatenate outputs along the temporal dimension
                    mask_logits_cat = torch.cat(all_mask_logits, dim=2) # (B, 1, T, H, W)
                    phase_logits_cat = torch.cat(all_phase_logits, dim=1) # (B, T, num_phases)
                    vol_curve_cat = torch.cat(all_vol_curves, dim=1) # (B, T, 1)
                    
                    # Compute global volume predictions
                    phase_probs = torch.softmax(phase_logits_cat, dim=-1)
                    p_es = phase_probs[..., 0]
                    p_ed = phase_probs[..., 1]
                    
                    if "lengths" in batch:
                        lengths = batch["lengths"]
                        mask_t = (torch.arange(T_total, device=device)[None, :] < lengths[:, None])
                        p_es = p_es * mask_t
                        p_ed = p_ed * mask_t
                        
                    vols = vol_curve_cat.squeeze(-1)
                    pred_edv = torch.sum(p_ed * vols, dim=1) / torch.sum(p_ed, dim=1).clamp(min=1e-3)
                    pred_esv = torch.sum(p_es * vols, dim=1) / torch.sum(p_es, dim=1).clamp(min=1e-3)
                    pred_ef = (pred_edv - pred_esv) / torch.clamp(pred_edv, min=1e-3)
                    
                    outputs = {
                        "mask_logits": mask_logits_cat,
                        "pred_phase_logits": phase_logits_cat,
                        "pred_vol_curve": vol_curve_cat,
                        "pred_edv": pred_edv,
                        "pred_esv": pred_esv,
                        "pred_ef": pred_ef,
                    }
                
            metrics.update(outputs, batch)
            
    # 5. Compute and print metrics
    results = metrics.compute()
    
    logging.info("\n" + "="*50)
    logging.info(f"Video-Level Test Set Metrics ({args.mode}):")
    for k, v in results.items():
        logging.info(f"  {k:20s}: {v:.4f}")
    logging.info("="*50 + "\n")

if __name__ == "__main__":
    main()
