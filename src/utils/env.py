import json
import logging
import os
from typing import Any, Dict, Optional


def setup_env_variables(env_conf: Optional[Dict[str, Any]] = None):
    """Apply environment variable overrides from config, then log them."""
    if env_conf:
        for key, value in env_conf.items():
            os.environ[key] = str(value)
    logging.info(f"Environment:\n{json.dumps(dict(os.environ), sort_keys=True, indent=2)}")


# Sensible defaults for PyTorch performance and debugging
PYTORCH_ENV_DEFAULTS = {
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "MKL_THREADING_LAYER": "GNU",
    "HYDRA_FULL_ERROR": "1",
    "NCCL_ASYNC_ERROR_HANDLING": "1",
}


def apply_pytorch_env_defaults():
    """Set environment variables that improve PyTorch stability and debugging."""
    for key, value in PYTORCH_ENV_DEFAULTS.items():
        if key not in os.environ:
            os.environ[key] = value
