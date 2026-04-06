import atexit
import logging
import uuid
from typing import Any, Optional, Union

import torch
from torch.utils.tensorboard import SummaryWriter

from .dist import get_machine_local_and_dist_rank


class TensorBoardLogger:
    """Rank-0 only TensorBoard writer with auto-cleanup."""

    def __init__(
        self,
        path: str,
        *args: Any,
        filename_suffix: Optional[str] = None,
        **kwargs: Any,
    ):
        self._writer: Optional[SummaryWriter] = None
        _, self._rank = get_machine_local_and_dist_rank()
        self._path = path

        if self._rank == 0:
            logging.info(f"TensorBoard logs → {path}")
            self._writer = SummaryWriter(
                log_dir=path,
                *args,
                filename_suffix=filename_suffix or str(uuid.uuid4()),
                **kwargs,
            )
        atexit.register(self.close)

    @property
    def writer(self) -> Optional[SummaryWriter]:
        return self._writer

    def flush(self):
        if self._writer:
            self._writer.flush()

    def close(self):
        if self._writer:
            self._writer.close()
            self._writer = None

    def log(self, name: str, data: Any, step: int):
        """Log a scalar value."""
        if self._writer:
            self._writer.add_scalar(name, data, global_step=step, new_style=True)

    def log_dict(self, payload: dict, step: int):
        """Log multiple scalars at once."""
        if not self._writer:
            return
        for key, value in payload.items():
            self.log(key, value, step)

    def log_visuals(self, name: str, data: Union[torch.Tensor, Any], step: int, fps: int = 4):
        """Log image (3D tensor) or video (5D tensor) data."""
        if not self._writer:
            return
        if data.ndim == 3:
            self._writer.add_image(name, data, global_step=step)
        elif data.ndim == 5:
            self._writer.add_video(name, data, global_step=step, fps=fps)
        else:
            raise ValueError(f"Expected 3D (image) or 5D (video) tensor, got {data.ndim}D")
