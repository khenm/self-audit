import logging
import math
import os
import random
from collections import defaultdict
from dataclasses import fields, is_dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, runtime_checkable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn


# ---------------------------------------------------------------------------
# Numeric safety
# ---------------------------------------------------------------------------

def sanitize_tensor(tensor, name="tensor", clamp_max=None):
    """Replace inf/nan values with zeros in a tensor, optionally clamp to a range."""
    if tensor is None:
        return tensor

    bad = torch.isnan(tensor) | torch.isinf(tensor)
    if bad.any():
        logging.warning(f"[{name}] contains inf/nan — replacing with zeros")
        tensor = torch.where(bad, torch.zeros_like(tensor), tensor)

    if clamp_max is not None:
        tensor = tensor.clamp(min=-clamp_max, max=clamp_max)

    return tensor


# ---------------------------------------------------------------------------
# Checkpoint auto-resume
# ---------------------------------------------------------------------------

def find_latest_checkpoint(save_dir):
    """Return path to `checkpoint.pt` inside *save_dir*, or None."""
    if not os.path.isdir(save_dir):
        return None
    path = os.path.join(save_dir, "checkpoint.pt")
    return path if os.path.isfile(path) else None


# ---------------------------------------------------------------------------
# Meters
# ---------------------------------------------------------------------------

class AverageMeter:
    """Track running mean of a scalar value."""

    def __init__(self, name: str, device: Optional[torch.device] = None, fmt: str = ":f"):
        self.name = name
        self.device = device
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.total = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.total += val * n
        self.count += n
        self.avg = self.total / self.count if self.count else 0.0

    def __str__(self):
        fmtstr = "{name}: {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class DurationMeter:
    """Tracks elapsed time and formats as human-readable string."""

    def __init__(self, name: str, device=None, fmt=":f"):
        self.name = name
        self.val = 0.0

    def reset(self):
        self.val = 0.0

    def update(self, val):
        self.val = val

    def __str__(self):
        secs = int(self.val)
        m, s = divmod(secs, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        return f"{self.name}: {d:02d}d {h:02d}h {m:02d}m"


class ProgressMeter:
    """Displays a formatted progress line combining multiple meters."""

    def __init__(self, num_batches, meters, real_meters=None, prefix=""):
        self.batch_fmt = self._make_batch_fmt(num_batches)
        self.meters = meters
        self.real_meters = real_meters or {}
        self.prefix = prefix

    def display(self, batch):
        parts = [self.prefix + self.batch_fmt.format(batch)]
        parts += [str(m) for m in self.meters]
        for name, meter in self.real_meters.items():
            computed = meter.compute()
            parts.append(" | ".join(f"{os.path.join(name, k)}: {v:.4f}" for k, v in computed.items()))
        logging.info(" | ".join(parts))

    @staticmethod
    def _make_batch_fmt(total):
        width = len(str(total))
        return "[{:" + str(width) + "d}/" + f"{total}]"


# ---------------------------------------------------------------------------
# Device transfer
# ---------------------------------------------------------------------------

@runtime_checkable
class _Transferable(Protocol):
    def to(self, device: torch.device, *args: Any, **kwargs: Any): ...


def _is_namedtuple(x) -> bool:
    return isinstance(x, tuple) and hasattr(x, "_asdict") and hasattr(x, "_fields")


def copy_data_to_device(data, device: torch.device, **kwargs):
    """Recursively move tensors / dataclasses / mappings / sequences to *device*."""
    if _is_namedtuple(data):
        return type(data)(**copy_data_to_device(data._asdict(), device, **kwargs))
    if isinstance(data, (list, tuple)):
        return type(data)(copy_data_to_device(v, device, **kwargs) for v in data)
    if isinstance(data, defaultdict):
        return type(data)(data.default_factory, {k: copy_data_to_device(v, device, **kwargs) for k, v in data.items()})
    if isinstance(data, Mapping) and not is_dataclass(data):
        return type(data)({k: copy_data_to_device(v, device, **kwargs) for k, v in data.items()})
    if is_dataclass(data) and not isinstance(data, type):
        init_fields = {f.name: copy_data_to_device(getattr(data, f.name), device, **kwargs) for f in fields(data) if f.init}
        obj = type(data)(**init_fields)
        for f in fields(data):
            if not f.init:
                setattr(obj, f.name, copy_data_to_device(getattr(data, f.name), device, **kwargs))
        return obj
    if isinstance(data, _Transferable):
        return data.to(device, **kwargs)
    return data


# ---------------------------------------------------------------------------
# Filesystem
# ---------------------------------------------------------------------------

def safe_makedirs(path: str) -> bool:
    """Create directory tree; returns True on success."""
    if not path:
        logging.warning("safe_makedirs called with empty path")
        return False
    os.makedirs(path, exist_ok=True)
    return True


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seeds(seed: int, max_epochs: int, rank: int = 0):
    """Set deterministic seeds across Python, NumPy, and PyTorch (rank-aware)."""
    effective = (seed + rank) * max_epochs
    logging.info(f"Setting seed: {effective} (base={seed}, rank={rank})")
    random.seed(effective)
    np.random.seed(effective)
    torch.manual_seed(effective)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(effective)
        torch.cuda.manual_seed_all(effective)


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


# ---------------------------------------------------------------------------
# Model summary
# ---------------------------------------------------------------------------

_UNITS = ("", " K", " M", " B", " T")


def _pretty_int(n: int) -> str:
    if n < 1_000:
        return f"{n:,}"
    exp = min(int(math.log10(n) // 3), len(_UNITS) - 1)
    value = n / 10 ** (3 * exp)
    return f"{value:.1f}".rstrip("0").rstrip(".") + _UNITS[exp]


def model_summary(model: nn.Module, log_file=None):
    """Print param counts; optionally dump architecture + param lists to files."""
    if get_rank() != 0:
        return

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    frozen = total - trainable

    header = (
        f"{'=' * 60}\n"
        f"Model      : {model.__class__.__name__}\n"
        f"Total      : {_pretty_int(total)} parameters\n"
        f"  trainable: {_pretty_int(trainable)}\n"
        f"  frozen   : {_pretty_int(frozen)}\n"
        f"{'=' * 60}"
    )
    print(header)

    if log_file is None:
        return

    from pathlib import Path
    log_file = Path(log_file)
    log_file.write_text(str(model))

    named = dict(model.named_parameters())

    def _dump(names: Iterable[str], fname: str):
        with open(log_file.with_name(fname), "w") as f:
            for n in names:
                p = named[n]
                f.write(f"{n:<60s} {str(tuple(p.shape)):<20} {p.numel()}\n")

    _dump([n for n, p in named.items() if p.requires_grad], "trainable.txt")
    _dump([n for n, p in named.items() if not p.requires_grad], "frozen.txt")


# ---------------------------------------------------------------------------
# Batch chunking (gradient accumulation)
# ---------------------------------------------------------------------------

def _is_primitive_sequence(data) -> bool:
    return isinstance(data, Sequence) and not isinstance(data, str) and len(data) > 0 and isinstance(data[0], (str, int, float, bool))


def chunk_batch(batch, num_chunks: int):
    """Split a batch into *num_chunks* pieces for gradient accumulation."""
    if num_chunks <= 1:
        return [batch]
    return [_get_chunk(batch, i, num_chunks) for i in range(num_chunks)]


def _get_chunk(data, idx: int, total: int):
    if isinstance(data, torch.Tensor) or _is_primitive_sequence(data):
        start = (len(data) // total) * idx
        end = (len(data) // total) * (idx + 1)
        return data[start:end]
    if isinstance(data, Mapping):
        return {k: _get_chunk(v, idx, total) for k, v in data.items()}
    if isinstance(data, str):
        return data
    if isinstance(data, Sequence):
        return [_get_chunk(v, idx, total) for v in data]
    return data
