import os
import torch
import torch.distributed as dist


def get_machine_local_and_dist_rank():
    """Read LOCAL_RANK and RANK from the environment (set by torchrun)."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    distributed_rank = int(os.environ.get("RANK", 0))
    return local_rank, distributed_rank


def get_rank():
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_main_process():
    return get_rank() == 0


def reduce_tensor(tensor, op="mean"):
    """All-reduce a tensor across all workers."""
    if not dist.is_available() or not dist.is_initialized() or get_world_size() == 1:
        return tensor

    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if op == "mean":
        rt /= get_world_size()
    return rt
