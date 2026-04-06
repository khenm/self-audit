import functools
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    CPUOffload,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy


# ---------------------------------------------------------------------------
# dtype mapping
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    None: None,
}


def get_fsdp_mixed_precision(amp_dtype: Optional[str] = None) -> Optional[MixedPrecision]:
    """Convert a string dtype name to an FSDP ``MixedPrecision`` policy.

    Returns ``None`` when *amp_dtype* is ``None`` (full fp32 training).
    """
    dt = _DTYPE_MAP.get(amp_dtype)
    if dt is None:
        return None
    return MixedPrecision(param_dtype=dt, reduce_dtype=dt, buffer_dtype=dt)


# ---------------------------------------------------------------------------
# FSDP wrapping
# ---------------------------------------------------------------------------

def wrap_fsdp(model: nn.Module, fsdp_conf: Dict[str, Any], device_id: int = 0) -> FSDP:
    """Wrap *model* in FSDP according to *fsdp_conf*.

    Expected keys in *fsdp_conf*:
        sharding_strategy : str   — "FULL_SHARD" | "SHARD_GRAD_OP" | "NO_SHARD"
        cpu_offload       : bool
        min_num_params    : int   — threshold for size-based auto-wrap
        (amp_dtype is read from the parent optim config, passed separately)
    """
    sharding = getattr(
        ShardingStrategy,
        fsdp_conf.get("sharding_strategy", "FULL_SHARD"),
        ShardingStrategy.FULL_SHARD,
    )

    cpu_off = CPUOffload(offload_params=fsdp_conf.get("cpu_offload", False))

    min_params = int(fsdp_conf.get("min_num_params", 1e5))
    auto_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=min_params)

    # Mixed precision is applied from the trainer side via amp_dtype
    mp = get_fsdp_mixed_precision(fsdp_conf.get("amp_dtype"))

    return FSDP(
        model,
        auto_wrap_policy=auto_policy,
        mixed_precision=mp,
        sharding_strategy=sharding,
        cpu_offload=cpu_off,
        device_id=device_id,
        use_orig_params=True,
    )


# ---------------------------------------------------------------------------
# State-dict helpers for checkpoint saving / loading
# ---------------------------------------------------------------------------

def fsdp_full_state_dict(model: FSDP) -> dict:
    """Extract a full (non-sharded) state dict from an FSDP model.

    Uses ``FULL_STATE_DICT`` with ``rank0_only=True`` and CPU offload so that
    only rank 0 materialises the full model in RAM — matching the DDP convention.
    """
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        return model.state_dict()
