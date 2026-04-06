import itertools
import logging
from typing import Any, Dict, List, Mapping, Optional, Set, Union

import hydra
import torch
import torch.nn as nn
from torch import Tensor
from wcmatch import fnmatch

_GLOB = fnmatch.CASE | fnmatch.DOTMATCH | fnmatch.EXTMATCH | fnmatch.SPLIT


# ---------------------------------------------------------------------------
# Optimizer wrapper
# ---------------------------------------------------------------------------

class OptimizerWrapper:
    """Wraps a torch optimizer with per-param-group schedulers keyed on ``where ∈ [0,1]``."""

    def __init__(self, optimizer: torch.optim.Optimizer, schedulers=None):
        self.optimizer = optimizer
        self.schedulers = schedulers
        self._check_scheduler_options()
        self.step_schedulers(0.0)

    def step(self, where: float = 1.0, closure=None):
        self.step_schedulers(where)
        return self.optimizer.step(closure)

    def zero_grad(self, *args, **kwargs):
        return self.optimizer.zero_grad(*args, **kwargs)

    def step_schedulers(self, where: float):
        if self.schedulers is None:
            return
        for i, pg in enumerate(self.optimizer.param_groups):
            for option, sched_fn in self.schedulers[i].items():
                pg[option] = sched_fn(where)

    def _check_scheduler_options(self):
        if self.schedulers is None:
            return
        for sched_map in self.schedulers:
            for option in sched_map:
                assert option in self.optimizer.defaults, (
                    f"Scheduler option '{option}' not in optimizer defaults "
                    f"({list(self.optimizer.defaults.keys())})"
                )


# ---------------------------------------------------------------------------
# Param-group construction helpers
# ---------------------------------------------------------------------------

def _full_param_name(module_name: str, param_name: str) -> str:
    return param_name if module_name == "" else f"{module_name}.{param_name}"


def _module_cls_to_param_names(model: nn.Module) -> Dict[type, Set[str]]:
    mapping: Dict[type, Set[str]] = {}
    for mod_name, mod in model.named_modules():
        mapping.setdefault(type(mod), set())
        for pname, _ in mod.named_parameters(recurse=False):
            mapping[type(mod)].add(_full_param_name(mod_name, pname))
    return mapping


def _match_param_patterns(patterns: Optional[List[str]], all_names: Set[str]) -> Set[str]:
    if not patterns:
        return set()
    matched = set()
    for pat in patterns:
        hits = set(fnmatch.filter(all_names, pat, flags=_GLOB))
        if not hits:
            raise ValueError(f"Pattern '{pat}' matched no parameters")
        logging.info(f"Param pattern [{pat}]: {len(hits)} params")
        matched |= hits
    return matched


def _match_cls_patterns(cls_names: Optional[List[str]], cls_map: Dict[type, Set[str]]) -> Set[str]:
    if not cls_names:
        return set()
    matched = set()
    for name in cls_names:
        cls = hydra.utils.get_class(name)
        if cls not in cls_map or not cls_map[cls]:
            raise ValueError(f"Module class '{name}' not found or has no parameters")
        matched |= cls_map[cls]
    return matched


def _resolve_names(cfg, all_names: Set[str], cls_map: Dict[type, Set[str]]):
    if "param_names" not in cfg and "module_cls_names" not in cfg:
        return None
    return _match_param_patterns(cfg.get("param_names"), all_names) | _match_cls_patterns(cfg.get("module_cls_names"), cls_map)


def _assign_defaults(cfgs: List[dict], all_names: Set[str]):
    """Ensure exactly one scheduler per option acts as the default (covers remaining params)."""
    specified = [c["parameter_names"] for c in cfgs if c["parameter_names"]]
    defaults = all_names - set.union(*specified) if specified else all_names

    n_defaults = 0
    for c in cfgs:
        if c["parameter_names"] is None:
            c["parameter_names"] = defaults
            n_defaults += 1
    assert n_defaults <= 1, "At most one default scheduler per option"
    if n_defaults == 0:
        cfgs.append({"parameter_names": defaults})


def _build_param_groups(all_cfgs, named_params: Dict[str, Tensor]):
    schedulers: List[Dict[str, Any]] = []
    param_groups: List[Dict[str, list]] = []

    for combo in itertools.product(*all_cfgs):
        constraints = [c["parameter_names"] for c in combo]
        names = set.intersection(*constraints)
        params = [v for k, v in named_params.items() if k in names]
        if not params:
            continue
        schedulers.append({c["option"]: c["scheduler"] for c in combo if "option" in c})
        param_groups.append({"params": params})

    return schedulers, param_groups


def _validate_groups(groups: List[Dict], model: nn.Module):
    for pg in groups:
        assert len(pg["params"]) == len(set(id(p) for p in pg["params"]))

    all_in_groups = set()
    for pg in groups:
        all_in_groups |= {id(p) for p in pg["params"]}

    model_params = {id(p) for p in model.parameters()}
    assert all_in_groups == model_params, (
        f"Param groups cover {len(all_in_groups)}/{len(model_params)} parameters"
    )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def construct_optimizer(
    model: nn.Module,
    optimizer_conf,
    options_conf: Optional[Mapping] = None,
) -> OptimizerWrapper:
    """Build an :class:`OptimizerWrapper` from Hydra configs."""
    named_params = dict(model.named_parameters())
    all_names = set(named_params.keys())
    cls_map = _module_cls_to_param_names(model)

    # Simple case: no schedulers
    if not options_conf:
        opt = hydra.utils.instantiate(optimizer_conf, list(named_params.values()))
        return OptimizerWrapper(opt)

    # Build per-option scheduler configs
    instantiated = hydra.utils.instantiate(options_conf)
    all_sched_cfgs: List[List[dict]] = []

    for option, cfg_list in instantiated.items():
        for cfg in cfg_list:
            cfg.option = option
            cfg.parameter_names = _resolve_names(cfg, all_names, cls_map)
        _assign_defaults(cfg_list, all_names)
        all_sched_cfgs.append(cfg_list)

    schedulers, param_groups = _build_param_groups(all_sched_cfgs, named_params)
    _validate_groups(param_groups, model)

    opt = hydra.utils.instantiate(optimizer_conf, param_groups)
    return OptimizerWrapper(opt, schedulers)


def construct_optimizers(model: nn.Module, optim_conf) -> Optional[List[OptimizerWrapper]]:
    """Convenience: build a single-optimizer list from config."""
    if optim_conf is None:
        return None
    wrapper = construct_optimizer(model, optim_conf.optimizer, optim_conf.get("options"))
    return [wrapper]
