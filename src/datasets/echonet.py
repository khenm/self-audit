import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from src.utils.dist import get_world_size

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class EchoNetDataset(Dataset):
    """
    Dataset class for EchoNet-Dynamic video segmentation and EF regression.

    Merged from reliable-echo-segment and pretrain-fm-echo.

    Supports:
    - Sliding-window clip generation with optional ED/ES enforcement (pretrain mode)
    - Coordinate-scaled mask generation for arbitrary img_size
    - Optional keypoint annotations (return_keypoints)
    - Optional pre-computed optical flow (flow_dir)

    Instantiated via Hydra ``_target_: src.datasets.echonet.EchoNetDataset``.
    """
    root_dir: str
    split: str = "TRAIN"
    max_clip_len: int = 32
    img_size: tuple = (112, 112)
    sampling_rate: int = 1
    transform: Any = None
    return_keypoints: bool = False
    pretrain: bool = False
    flow_dir: Optional[str] = None

    def __post_init__(self):
        super().__init__()
        self.split = self.split.upper()
        self.img_size = tuple(self.img_size)
        self.overlap = 0
        self.max_retries = 5
        self.videos_dir = os.path.join(self.root_dir, "Videos")

        self.file_list = self._load_file_list()
        self.meta_lookup = self._create_meta_lookup()
        self.tracings = self._load_tracings()
        self.clips = self._generate_clips()

        self.ef_targets = self._get_normalized_column("EF", 100.0)
        self.edv_targets = self._get_normalized_column("EDV", 300.0)
        self.esv_targets = self._get_normalized_column("ESV", 300.0)

        self.fname_to_idx = {fn: i for i, fn in enumerate(self.file_list["FileName"].values)}

        logger.info(
            f"EchoNetDataset initialized: Split={self.split}, Clips={len(self.clips)}, "
            f"Videos={len(self.file_list)}, FlowDir={self.flow_dir}, "
            f"ReturnKeypoints={self.return_keypoints}"
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

        if os.path.exists(self.videos_dir):
            available = {
                f[:-4] if f.lower().endswith('.avi') else f
                for f in os.listdir(self.videos_dir)
            }
            n_before = len(df)
            df = df[df["FileName"].isin(available)]
            if len(df) < n_before:
                logger.warning(f"Filtered {n_before - len(df)} missing videos. Remaining: {len(df)}")

        return df

    def _get_normalized_column(self, col_name, scale):
        if col_name in self.file_list.columns:
            return self.file_list[col_name].values / scale
        return np.full(len(self.file_list), -1.0 if "V" in col_name else 0.0)

    def _create_meta_lookup(self):
        if "EDFrame" in self.file_list.columns and "ESFrame" in self.file_list.columns:
            return self.file_list.set_index("FileName")[["EDFrame", "ESFrame"]].to_dict('index')
        logger.warning("EDFrame/ESFrame columns missing. Beat-centric sampling disabled.")
        return {}

    def _load_tracings(self):
        path = os.path.join(self.root_dir, "VolumeTracings.csv")
        if not os.path.exists(path):
            logger.warning(f"VolumeTracings.csv not found at {path}. No masks will be generated.")
            return None

        df = pd.read_csv(path)
        df["FileName"] = df["FileName"].astype(str).apply(
            lambda x: x[:-4] if x.lower().endswith('.avi') else x
        )
        available = set(self.file_list["FileName"].values)
        return df[df["FileName"].isin(available)]

    # ------------------------------------------------------------------
    # Clip generation
    # ------------------------------------------------------------------

    def _generate_clips(self):
        """
        Sliding-window clip generation.

        When ``pretrain=True``, only clips that strictly contain both the ED
        and ES frames are kept (reliable-echo-segment behaviour).
        When ``pretrain=False``, all clips are included regardless of
        landmark coverage (pretrain-fm-echo behaviour).
        """
        clips = []
        stride = max(1, self.max_clip_len - self.overlap)

        for _, row in self.file_list.iterrows():
            fname = row["FileName"]
            total_frames = int(row["NumberOfFrames"])

            for start in range(0, total_frames, stride):
                if start >= total_frames:
                    break

                end = start + self.max_clip_len

                # Skip clips that would require >20% padding
                if end > total_frames:
                    pad_len = end - total_frames
                    if pad_len > 0.2 * self.max_clip_len:
                        continue

                if fname not in self.meta_lookup:
                    continue

                meta = self.meta_lookup[fname]
                ed_frame = int(meta.get("EDFrame", -1))
                es_frame = int(meta.get("ESFrame", -1))

                if ed_frame == -1 or es_frame == -1:
                    continue

                # In pretrain mode, enforce that both landmarks lie within the clip
                if self.pretrain:
                    has_ed = start < ed_frame < end - 1
                    has_es = start < es_frame < end - 1
                    if not (has_ed and has_es):
                        continue

                clips.append((fname, start, end, total_frames))

        return clips

    # ------------------------------------------------------------------
    # Video / mask loading
    # ------------------------------------------------------------------

    def _pad_video_tensor(self, video_chunk, start_idx, end_idx, valid_start, valid_end):
        pad_left = max(0, valid_start - start_idx)
        pad_right = max(0, end_idx - valid_end)

        if pad_left > 0 or pad_right > 0:
            video_chunk = np.pad(
                video_chunk,
                ((pad_left, pad_right), (0, 0), (0, 0), (0, 0)),
                mode='edge'
            )

        if video_chunk.shape[0] != self.max_clip_len:
            current_len = video_chunk.shape[0]
            if current_len < self.max_clip_len:
                video_chunk = np.pad(
                    video_chunk,
                    ((0, self.max_clip_len - current_len), (0, 0), (0, 0), (0, 0)),
                    mode='edge'
                )
            else:
                video_chunk = video_chunk[:self.max_clip_len]
        return video_chunk

    def _load_video_clip(self, video_path, fname, start_idx, end_idx, total_frames):
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
                frame = cv2.resize(frame, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR)
            frames.append(frame)
        cap.release()

        if not frames:
            return np.zeros((self.max_clip_len, *self.img_size, 3), dtype=np.uint8)

        video_chunk = np.stack(frames, axis=0)
        return self._pad_video_tensor(video_chunk, start_idx, end_idx, valid_start, valid_end)

    @staticmethod
    def _generate_mask(points, height, width):
        """Generate a binary mask from tracing points, scaled to arbitrary (H, W)."""
        mask = np.zeros((height, width), dtype=np.uint8)
        points = points.iloc[1:]
        scale_x = width / 112.0
        scale_y = height / 112.0
        pts = np.stack([
            np.concatenate([points["X1"].values * scale_x, points["X2"].values[::-1] * scale_x]),
            np.concatenate([points["Y1"].values * scale_y, points["Y2"].values[::-1] * scale_y])
        ], axis=1).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
        return mask

    def _get_keypoints(self, t_subset, H, W):
        """Extract and normalise 42 contour keypoints to [0, 1]."""
        pts_df = t_subset.iloc[1:]
        kps = np.stack([
            np.concatenate([pts_df["X1"].values, pts_df["X2"].values[::-1]]),
            np.concatenate([pts_df["Y1"].values, pts_df["Y2"].values[::-1]])
        ], axis=1).astype(np.float32)

        if len(kps) != 42:
            if len(kps) > 42:
                kps = kps[:42]
            else:
                pad = np.tile(kps[-1:], (42 - len(kps), 1))
                kps = np.concatenate([kps, pad], axis=0)

        kps[:, 0] /= W
        kps[:, 1] /= H
        return kps

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

        video = self._load_video_clip(video_path, fname, start_idx, end_idx, total_frames)

        if video is None:
            logger.warning(f"Failed to load video clip: {video_path}")
            if _retry_count >= self.max_retries:
                raise RuntimeError(f"Failed to load {self.max_retries} consecutive samples from {idx}")
            return self.__getitem__((idx + 1) % len(self), _retry_count + 1)

        T_clip, H, W, _ = video.shape

        mask_clip = np.zeros((T_clip, H, W), dtype=np.uint8)
        frame_mask = np.zeros((T_clip,), dtype=np.float32)
        keypoints_clip = np.zeros((T_clip, 42, 2), dtype=np.float32) if self.return_keypoints else None

        # ED / ES frame annotations
        if fname in self.meta_lookup:
            meta = self.meta_lookup[fname]
            ed_frame = int(meta.get("EDFrame", -1))
            es_frame = int(meta.get("ESFrame", -1))
        else:
            ed_frame, es_frame = -1, -1

        for t in range(T_clip):
            orig_idx = start_idx + t
            if orig_idx == ed_frame:
                frame_mask[t] = 2.0
            elif orig_idx == es_frame:
                frame_mask[t] = 1.0

        # Tracings → binary masks (and optionally keypoints)
        if self.tracings is not None:
            file_tracings = self.tracings[self.tracings["FileName"] == fname]
            if not file_tracings.empty:
                for t in range(T_clip):
                    orig_idx = start_idx + t
                    if 0 <= orig_idx < total_frames:
                        t_subset = file_tracings[file_tracings["Frame"] == orig_idx]
                        if not t_subset.empty:
                            mask_clip[t] = self._generate_mask(t_subset, H, W)
                            if self.return_keypoints:
                                keypoints_clip[t] = self._get_keypoints(t_subset, H, W)

        # Convert to tensors
        video = video.transpose(3, 0, 1, 2).astype(np.float32) / 255.0  # (C, T, H, W)
        mask_clip = mask_clip[None, ...].astype(np.float32)              # (1, T, H, W)

        video = torch.from_numpy(video)
        mask_clip = torch.from_numpy(mask_clip)

        if self.transform:
            data = self.transform({"video": video, "label": mask_clip})
            video = data["video"]
            mask_clip = data.get("label", mask_clip)

        # Optional optical flow
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

        if self.return_keypoints:
            output["keypoints"] = torch.tensor(keypoints_clip, dtype=torch.float32)

        return output

    # ------------------------------------------------------------------
    # Flow loading
    # ------------------------------------------------------------------

    def _load_flow(self, fname, start_idx, end_idx, total_frames):
        """Load pre-computed optical flow for a clip, zero-padding if absent."""
        if self.flow_dir and os.path.exists(self.flow_dir):
            flow_path = os.path.join(self.flow_dir, fname + ".pt")
            if os.path.exists(flow_path):
                full_flow = torch.load(flow_path, weights_only=True).float()  # (TotalFrames-1, 2, H, W)

                f_start = max(0, start_idx)
                f_end = min(total_frames - 1, end_idx)
                valid_flow = full_flow[f_start:f_end]

                target_len = self.max_clip_len - 1
                curr_len = valid_flow.shape[0]

                if curr_len < target_len:
                    pad_left = max(0, f_start - start_idx)
                    pad_right = target_len - curr_len - pad_left
                    valid_flow = F.pad(
                        valid_flow,
                        (0, 0, 0, 0, 0, 0, pad_left, pad_right),
                        mode='constant',
                        value=0.0
                    )
                else:
                    valid_flow = valid_flow[:target_len]

                return valid_flow

        return torch.zeros((self.max_clip_len - 1, 2, *self.img_size), dtype=torch.float32)


def build_dataloaders(cfg):
    """
    Convenience factory: builds train / val / test DataLoaders from a config dict.

    Expected config keys mirror the legacy ``EchoNet.get_dataloaders`` interface::

        cfg['data']['root_dir']
        cfg['data'].get('img_size', [112, 112])
        cfg['data'].get('flow_dir', None)
        cfg['data'].get('subset_size', None)
        cfg['model'].get('max_clip_len', 32)
        cfg['model'].get('return_keypoints', False)
        cfg['training'].get('batch_size', 8)
        cfg['training'].get('num_workers', 4)
        cfg['training'].get('pretrain', False)
    """
    root_dir = cfg['data']['root_dir']
    batch_size = cfg['training'].get('batch_size', 8)
    num_workers = cfg['training'].get('num_workers', 4)
    max_clip_len = cfg['model'].get('max_clip_len', 32)
    img_size = tuple(cfg['data'].get('img_size', [112, 112]))
    flow_dir = cfg['data'].get('flow_dir', None)
    return_kps = cfg['model'].get('return_keypoints', False)
    pretrain = cfg['training'].get('pretrain', False)

    if pretrain:
        logger.info("Pretraining mode: only clips containing both ED and ES frames are used.")

    ds_tr = EchoNetDataset(root_dir, "TRAIN", max_clip_len=max_clip_len, img_size=img_size,
                           return_keypoints=return_kps, pretrain=pretrain, flow_dir=flow_dir)
    ds_va = EchoNetDataset(root_dir, "VAL",   max_clip_len=max_clip_len, img_size=img_size,
                           return_keypoints=return_kps, pretrain=pretrain, flow_dir=flow_dir)
    ds_ts = EchoNetDataset(root_dir, "TEST",  max_clip_len=max_clip_len, img_size=img_size,
                           return_keypoints=return_kps, pretrain=pretrain, flow_dir=flow_dir)

    subset_size = cfg['data'].get('subset_size')
    if subset_size:
        subset_size = int(subset_size)
        logger.info(f"Using subset of size {subset_size} for all splits")
        ds_tr = Subset(ds_tr, range(min(len(ds_tr), subset_size)))
        ds_va = Subset(ds_va, range(min(len(ds_va), subset_size)))
        ds_ts = Subset(ds_ts, range(min(len(ds_ts), subset_size)))

    sampler_tr = sampler_va = sampler_ts = None
    shuffle_tr = True

    if get_world_size() > 1:
        sampler_tr = DistributedSampler(ds_tr, shuffle=True)
        sampler_va = DistributedSampler(ds_va, shuffle=False)
        sampler_ts = DistributedSampler(ds_ts, shuffle=False)
        shuffle_tr = False

    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=shuffle_tr,
                           num_workers=num_workers, sampler=sampler_tr, pin_memory=True)
    loader_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, sampler=sampler_va, pin_memory=True)
    loader_ts = DataLoader(ds_ts, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, sampler=sampler_ts, pin_memory=True)

    return loader_tr, loader_va, loader_ts
