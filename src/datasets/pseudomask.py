import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class PseudoMaskDataset(Dataset):
    """
    Dataset for pretraining with dense pseudomask supervision.

    Each video has a corresponding `{filename}.npy` file containing per-frame
    binary masks with shape ``(T, H, W)``. Unlike EchoNetDataset which only has
    GT masks at ED/ES frames, this provides supervision at every frame.

    Instantiated via Hydra ``_target_: src.datasets.pseudomask.PseudoMaskDataset``.
    """
    root_dir: str
    mask_dir: str
    split: str = "TRAIN"
    max_clip_len: int = 32
    img_size: tuple = (112, 112)
    flow_dir: Optional[str] = None
    transform: Any = None

    def __post_init__(self):
        super().__init__()
        self.split = self.split.upper()
        self.img_size = tuple(self.img_size)
        self.overlap = 0
        self.max_retries = 5
        self.videos_dir = os.path.join(self.root_dir, "Videos")

        self.file_list = self._load_file_list()
        self.meta_lookup = self._create_meta_lookup()
        self.clips = self._generate_clips()

        self.ef_targets = self._get_normalized_column("EF", 100.0)
        self.edv_targets = self._get_normalized_column("EDV", 300.0)
        self.esv_targets = self._get_normalized_column("ESV", 300.0)

        self.fname_to_idx = {fn: i for i, fn in enumerate(self.file_list["FileName"].values)}

        logger.info(
            f"PseudoMaskDataset initialized: Split={self.split}, Clips={len(self.clips)}, "
            f"Videos={len(self.file_list)}, MaskDir={self.mask_dir}"
        )

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _load_file_list(self):
        fname_w_frames = "FileListwFrames112.csv"
        path = os.path.join(self.root_dir, fname_w_frames)
        if not os.path.exists(path):
            path = os.path.join(self.root_dir, "FileList.csv")
            logger.warning(f"{fname_w_frames} not found, falling back to FileList.csv.")

        if not os.path.exists(path):
            raise FileNotFoundError(f"FileList.csv not found at {path}")

        df = pd.read_csv(path)
        if "Split" in df.columns:
            df = df[df["Split"].str.upper() == self.split]

        df["FileName"] = df["FileName"].astype(str).apply(
            lambda x: x[:-4] if x.lower().endswith('.avi') else x
        )

        # Only keep videos that have both a .avi and a pseudomask .npy
        available_videos = set()
        if os.path.exists(self.videos_dir):
            available_videos = {
                f[:-4] if f.lower().endswith('.avi') else f
                for f in os.listdir(self.videos_dir)
            }

        available_masks = set()
        if os.path.exists(self.mask_dir):
            available_masks = {
                f[:-4] if f.lower().endswith('.npy') else f
                for f in os.listdir(self.mask_dir) if f.endswith('.npy')
            }

        available = available_videos & available_masks
        n_before = len(df)
        df = df[df["FileName"].isin(available)]
        if len(df) < n_before:
            logger.warning(
                f"Filtered {n_before - len(df)} videos (missing video or mask). "
                f"Remaining: {len(df)}"
            )

        return df

    def _get_normalized_column(self, col_name, scale):
        if col_name in self.file_list.columns:
            return self.file_list[col_name].values / scale
        return np.full(len(self.file_list), -1.0 if "V" in col_name else 0.0)

    def _create_meta_lookup(self):
        if "EDFrame" in self.file_list.columns and "ESFrame" in self.file_list.columns:
            return self.file_list.set_index("FileName")[["EDFrame", "ESFrame"]].to_dict('index')
        logger.warning("EDFrame/ESFrame columns missing.")
        return {}

    # ------------------------------------------------------------------
    # Clip generation
    # ------------------------------------------------------------------

    def _generate_clips(self):
        clips = []
        stride = max(1, self.max_clip_len - self.overlap)

        for _, row in self.file_list.iterrows():
            fname = row["FileName"]
            total_frames = int(row["NumberOfFrames"])

            for start in range(0, total_frames, stride):
                if start >= total_frames:
                    break

                end = start + self.max_clip_len

                if end > total_frames:
                    pad_len = end - total_frames
                    if pad_len > 0.2 * self.max_clip_len:
                        continue

                clips.append((fname, start, end, total_frames))

        return clips

    # ------------------------------------------------------------------
    # Video / mask loading
    # ------------------------------------------------------------------

    def _pad_tensor(self, arr, start_idx, end_idx, total_frames):
        """Pad a (T', ...) array to max_clip_len via edge replication."""
        valid_start = max(0, start_idx)
        valid_end = min(total_frames, end_idx)
        pad_left = max(0, valid_start - start_idx)
        pad_right = max(0, end_idx - valid_end)

        ndim = arr.ndim
        pad_widths = [(pad_left, pad_right)] + [(0, 0)] * (ndim - 1)

        if pad_left > 0 or pad_right > 0:
            arr = np.pad(arr, pad_widths, mode='edge')

        current_len = arr.shape[0]
        if current_len < self.max_clip_len:
            extra = [(0, self.max_clip_len - current_len)] + [(0, 0)] * (ndim - 1)
            arr = np.pad(arr, extra, mode='edge')
        elif current_len > self.max_clip_len:
            arr = arr[:self.max_clip_len]

        return arr

    def _load_video_clip(self, video_path, start_idx, end_idx, total_frames):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        valid_start = max(0, start_idx)
        valid_end = min(total_frames, end_idx)
        cap.set(cv2.CAP_PROP_POS_FRAMES, valid_start)

        frames = []
        for _ in range(valid_end - valid_start):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if (frame.shape[0], frame.shape[1]) != self.img_size:
                frame = cv2.resize(frame, (self.img_size[1], self.img_size[0]),
                                   interpolation=cv2.INTER_LINEAR)
            frames.append(frame)
        cap.release()

        if not frames:
            return np.zeros((self.max_clip_len, *self.img_size, 3), dtype=np.uint8)

        video_chunk = np.stack(frames, axis=0)
        return self._pad_tensor(video_chunk, start_idx, end_idx, total_frames)

    def _load_pseudomask_clip(self, fname, start_idx, end_idx, total_frames):
        """Load pseudomask .npy and slice the clip window."""
        mask_path = os.path.join(self.mask_dir, fname + ".npy")
        masks = np.load(mask_path)  # (T_total, H_orig, W_orig)

        valid_start = max(0, start_idx)
        valid_end = min(total_frames, end_idx)
        clip = masks[valid_start:valid_end]  # (T', H, W)

        # Resize each frame if needed
        H, W = self.img_size
        if clip.shape[1] != H or clip.shape[2] != W:
            resized = np.stack([
                cv2.resize(f.astype(np.float32), (W, H),
                           interpolation=cv2.INTER_NEAREST)
                for f in clip
            ], axis=0)
            clip = (resized > 0.5).astype(np.uint8)

        return self._pad_tensor(clip, start_idx, end_idx, total_frames)

    # ------------------------------------------------------------------
    # Flow loading
    # ------------------------------------------------------------------

    def _load_flow(self, fname, start_idx, end_idx, total_frames):
        if self.flow_dir and os.path.exists(self.flow_dir):
            flow_path = os.path.join(self.flow_dir, fname + ".pt")
            if os.path.exists(flow_path):
                full_flow = torch.load(flow_path, weights_only=True).float()
                f_start = max(0, start_idx)
                f_end = min(total_frames - 1, end_idx)
                valid_flow = full_flow[f_start:f_end]

                target_len = self.max_clip_len - 1
                curr_len = valid_flow.shape[0]
                if curr_len < target_len:
                    pad_left = max(0, f_start - start_idx)
                    pad_right = target_len - curr_len - pad_left
                    valid_flow = F.pad(valid_flow,
                                       (0, 0, 0, 0, 0, 0, pad_left, pad_right),
                                       mode='constant', value=0.0)
                else:
                    valid_flow = valid_flow[:target_len]
                return valid_flow

        return torch.zeros((self.max_clip_len - 1, 2, *self.img_size), dtype=torch.float32)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx, _retry_count=0):
        fname, start_idx, end_idx, total_frames = self.clips[idx]
        file_idx = self.fname_to_idx[fname]

        video_path = os.path.join(
            self.videos_dir,
            fname + (".avi" if not fname.lower().endswith(".avi") else "")
        )

        video = self._load_video_clip(video_path, start_idx, end_idx, total_frames)
        if video is None:
            logger.warning(f"Failed to load video clip: {video_path}")
            if _retry_count >= self.max_retries:
                raise RuntimeError(f"Failed to load {self.max_retries} consecutive samples from {idx}")
            return self.__getitem__((idx + 1) % len(self), _retry_count + 1)

        T_clip = video.shape[0]

        # Dense pseudomask for every frame
        mask_clip = self._load_pseudomask_clip(fname, start_idx, end_idx, total_frames)

        # ED/ES frame annotations (for volume targets)
        frame_mask = np.zeros((T_clip,), dtype=np.float32)
        if fname in self.meta_lookup:
            meta = self.meta_lookup[fname]
            ed_frame = int(meta.get("EDFrame", -1))
            es_frame = int(meta.get("ESFrame", -1))
            for t in range(T_clip):
                orig_idx = start_idx + t
                if orig_idx == ed_frame:
                    frame_mask[t] = 2.0
                elif orig_idx == es_frame:
                    frame_mask[t] = 1.0

        # Convert to tensors
        video = video.transpose(3, 0, 1, 2).astype(np.float32) / 255.0  # (C, T, H, W)
        mask_clip = mask_clip[None, ...].astype(np.float32)              # (1, T, H, W)

        video = torch.from_numpy(video)
        mask_clip = torch.from_numpy(mask_clip)

        if self.transform:
            data = self.transform({"video": video, "label": mask_clip})
            video = data["video"]
            mask_clip = data.get("label", mask_clip)

        flow_chunk = self._load_flow(fname, start_idx, end_idx, total_frames)

        output = {
            "video": video,
            "label": mask_clip,
            "case": fname,
            "frame_mask": torch.tensor(frame_mask, dtype=torch.float32),
            "lengths": torch.tensor(T_clip, dtype=torch.long),
            "target_ef": torch.tensor(self.ef_targets[file_idx], dtype=torch.float32),
            "target_edv": torch.tensor(self.edv_targets[file_idx], dtype=torch.float32),
            "target_esv": torch.tensor(self.esv_targets[file_idx], dtype=torch.float32),
        }

        if flow_chunk is not None:
            output["flow"] = flow_chunk

        return output
