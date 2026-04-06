import logging
from functools import wraps
from typing import List, Set

import torch.nn as nn
from wcmatch import fnmatch

# Glob flags: case-sensitive, dot-match, extended patterns, pipe-split
_GLOB = fnmatch.CASE | fnmatch.DOTMATCH | fnmatch.EXTMATCH | fnmatch.SPLIT


def freeze_modules(model: nn.Module, patterns: List[str], recursive: bool = True) -> nn.Module:
    """Freeze submodules whose names match any of *patterns* (glob syntax).

    Parameters
    ----------
    model : nn.Module
    patterns : list[str]
        Glob patterns (e.g. ``["encoder.*", "cls_head"]``).
    recursive : bool
        If True, freeze the entire subtree of matched modules.

    Returns
    -------
    nn.Module  — the same model, with matched parts frozen.
    """
    matched: Set[str] = set()

    for name, mod in model.named_modules():
        if any(fnmatch.fnmatch(name, p, flags=_GLOB) for p in patterns):
            matched.add(name)
            _freeze_module(mod, recursive)

    # Ensure every user pattern actually matched something
    _validate_patterns(matched, patterns)
    return model


def _freeze_module(mod: nn.Module, recursive: bool):
    """Set *mod* to eval, disable requires_grad, and lock .train()."""
    if recursive:
        mod.eval()
    else:
        mod.training = False

    original_train = mod.train

    @wraps(original_train)
    def _locked_train(mode: bool = True):
        if recursive:
            return original_train(False)
        out = original_train(mode)
        out.training = False
        return out

    mod.train = _locked_train  # type: ignore[attr-defined]

    params = mod.parameters() if recursive else mod.parameters(recurse=False)
    for p in params:
        p.requires_grad = False


def _validate_patterns(matched_names: Set[str], patterns: List[str]):
    unused = [p for p in patterns if not any(fnmatch.fnmatch(n, p, flags=_GLOB) for n in matched_names)]
    if unused:
        raise ValueError(f"Freeze patterns matched nothing: {unused}")
