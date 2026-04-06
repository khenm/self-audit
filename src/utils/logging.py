import atexit
import copy
import functools
import logging
import os
import sys

from .general import safe_makedirs


@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    """Keep a single file handle per path; close on interpreter exit."""
    buf_size = 1024  # 1 KB
    fh = open(filename, "a", buffering=buf_size)
    atexit.register(fh.close)
    return fh


def setup_logging(
    name: str,
    output_dir: str = None,
    rank: int = 0,
    log_level_primary: str = "INFO",
    log_level_secondary: str = "WARNING",
    all_ranks: bool = False,
):
    """Configure Python logging for distributed training.

    Rank 0 logs to stdout at *log_level_primary* and optionally to a file.
    Other ranks log to stdout at *log_level_secondary* only.
    """
    log_filename = None
    if output_dir:
        safe_makedirs(output_dir)
        if rank == 0:
            log_filename = os.path.join(output_dir, "log.txt")
        elif all_ranks:
            log_filename = os.path.join(output_dir, f"log_{rank}.txt")

    logger = logging.getLogger(name)
    logger.setLevel(log_level_primary)

    fmt = "%(levelname)s %(asctime)s %(filename)s:%(lineno)4d: %(message)s"
    formatter = logging.Formatter(fmt)

    # Clear existing handlers
    for h in list(logger.handlers):
        logger.removeHandler(h)
    logger.root.handlers.clear()
    logging.root.handlers.clear()

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    console.setLevel(log_level_primary if rank == 0 else log_level_secondary)
    logger.addHandler(console)

    # File handler
    if log_filename is not None:
        file_handler = logging.StreamHandler(_cached_log_stream(log_filename))
        file_handler.setLevel(log_level_primary)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logging.root = logger
