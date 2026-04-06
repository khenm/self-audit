import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class GradientClipper:
    """Per-module-group gradient clipping.

    Each config entry specifies which parameter groups to clip and to what norm.
    All trainable parameters must be covered by exactly one config entry —
    the clipper raises if any parameter is left unassigned.
    """

    def __init__(self, configs: List[Dict[str, Any]]):
        self.configs: List[Dict[str, Any]] = []
        self._param_groups: Optional[List] = None

        for cfg in configs:
            names = cfg["module_name"]
            if isinstance(names, str):
                names = [names]
            self.configs.append({
                "module_names": names,
                "max_norm": float(cfg["max_norm"]) if cfg.get("max_norm") is not None else None,
                "norm_type": cfg.get("norm_type", 2),
            })

    def setup_clipping(self, model: nn.Module):
        """Pre-compute parameter groups. Call once before training starts."""
        groups = []
        covered = set()

        for cfg in self.configs:
            params = []
            for pname, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if any(mn in pname for mn in cfg["module_names"]):
                    params.append(param)
                    covered.add(pname)
            groups.append((cfg, params))

        # Verify full coverage
        uncovered = [n for n, p in model.named_parameters() if p.requires_grad and n not in covered]
        if uncovered:
            logging.error(f"Uncovered parameters: {uncovered}")
            raise ValueError(
                f"{len(uncovered)} trainable parameter(s) have no gradient clip config. "
                "Add a catch-all entry or explicitly list them."
            )

        self._param_groups = groups

    def __call__(self, model: nn.Module = None) -> Dict[str, float]:
        """Clip gradients and return per-group norm values for logging."""
        if self._param_groups is None:
            raise RuntimeError("Call setup_clipping(model) before clipping gradients.")

        norms: Dict[str, float] = {}
        for cfg, params in self._param_groups:
            if not params or cfg["max_norm"] is None:
                continue

            norm = nn.utils.clip_grad_norm_(
                params,
                max_norm=cfg["max_norm"],
                norm_type=cfg["norm_type"],
            )
            if norm is not None:
                key = ",".join(cfg["module_names"])
                norms[key] = norm.item()

        return norms
