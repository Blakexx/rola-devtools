"""THE DIFF EXECUTORS: one side's tensors, and the comparison of two sides.

`produce` runs in the side's own environment -- the checkout's python, its torch, its built extension -- and calls the
declaration's `module:function` once per cell. That function is given the cell's record and returns `{name: tensor}`:
the quantities this side has to show. They are written into the node's workspace and nothing else leaves it.

`compare` runs in one side's environment too (it takes the same torch to read the tensors back), loads each cell's
quantities from both workspaces, and applies the strategy to each. Its output is the DIFF and never the tensors.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

from .strategies import compare as apply_strategy


def _resolve(spec: str):
    module, _, function = spec.partition(":")
    if not module or not function:
        raise ValueError(f"an executor is module:function, got {spec!r}")
    return getattr(importlib.import_module(module), function)


def produce(ctx) -> dict:
    """One side, over every cell the node takes as a data input. A cell whose call RAISES is that cell's domain
    failure: it is recorded and the other cells still run, because one unsupported configuration is not a reason to
    lose the comparison on the rest."""
    import torch

    subject = _resolve(ctx.params["executor"])
    params = ctx.params["params"]
    out = {}
    for record in ctx.inputs:
        name = record["name"]
        try:
            quantities = subject(record, **params)
        except Exception as ex:  # noqa: BLE001 -- the cell's domain failure, recorded and carried
            out[name] = {"status": "failed", "detail": f"{type(ex).__name__}: {ex}"}
            continue
        if not isinstance(quantities, dict) or not quantities:
            raise TypeError(f"{ctx.params['executor']} returned {type(quantities).__name__} for {name}; "
                            "a side returns a non-empty {name: tensor}")
        file = f"{name}.pt"
        torch.save({k: v.detach().cpu() for k, v in quantities.items()}, ctx.workspace / file)
        out[name] = {"status": "ok", "file": file, "draw": record.get("draw"),
                     "quantities": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in quantities.items()}}
    return {"cells": out}


def _load(side, name: str) -> dict:
    import torch

    return torch.load(Path(side.dir) / side.output["cells"][name]["file"], map_location="cpu", weights_only=True)


def compare(ctx) -> dict:
    """The two sides, cell by cell and quantity by quantity. A cell either side failed to produce is NOT a comparison
    that passed: it is carried as that cell's own status, and it counts as a difference the claim has to answer for --
    a gate that went green because one side produced nothing is the failure this rule exists to prevent."""
    left, right = ctx.deps["left"], ctx.deps["right"]
    strategy, params = ctx.params["strategy"], ctx.params["params"]
    expect, on_difference = ctx.params["expect"], ctx.params["on_difference"]

    cells, differing, unusable = {}, [], []
    for name in sorted(set(left.output["cells"]) | set(right.output["cells"])):
        sides = {"left": left.output["cells"].get(name), "right": right.output["cells"].get(name)}
        missing = [s for s, state in sides.items() if state is None or state["status"] != "ok"]
        if missing:
            detail = {s: (sides[s] or {}).get("detail", "no such cell on this side") for s in missing}
            cells[name] = {"status": "unusable", "produced_by": [s for s in sides if s not in missing], **detail}
            unusable.append(name)
            continue
        if sides["left"]["draw"] != sides["right"]["draw"]:
            cells[name] = {"status": "unusable", "why": "the two sides drew this cell with different code",
                           "left_draw": sides["left"]["draw"], "right_draw": sides["right"]["draw"]}
            unusable.append(name)
            continue
        a, b = _load(left, name), _load(right, name)
        names = sorted(set(a) | set(b))
        verdicts = {}
        for quantity in names:
            if quantity not in a or quantity not in b:
                verdicts[quantity] = {"same": False, "why": f"only {'left' if quantity in a else 'right'} produced it"}
                continue
            rule = {**params, **params.get("per_quantity", {}).get(quantity, {})}
            rule.pop("per_quantity", None)
            envelope = rule.pop("envelope_from", None)
            if envelope is not None:
                rule["envelope"] = b[envelope]
            verdicts[quantity] = apply_strategy(strategy, a[quantity], b[quantity], rule)
        same = all(v.get("same") for v in verdicts.values())
        cells[name] = {"status": "ok", "same": same, "quantities": verdicts}
        if not same:
            differing.append(name)

    #: THE CLAIM, and whether this run kept it
    held = (not differing and not unusable) if expect == "same" else (bool(differing) and not unusable)
    out = {"strategy": strategy, "expect": expect, "cells": cells, "differing": differing, "unusable": unusable,
           "compared": len(cells) - len(unusable), "held": held}
    if not held and on_difference == "fail":
        raise AssertionError(_message(out))
    if not held:
        out["status"] = "failed"
        out["detail"] = _message(out)
    return out


def _message(out: dict) -> str:
    if out["unusable"]:
        return (f"{len(out['unusable'])} of {len(out['cells'])} cells produced no comparison "
                f"({', '.join(out['unusable'][:4])}): a gate that goes green on a cell neither side produced is a gate "
                f"that measured nothing")
    if out["expect"] == "same":
        worst = max((q for name in out["differing"] for q in out["cells"][name]["quantities"].values()
                     if not q.get("same")), key=lambda q: q.get("ratio", 0.0), default={})
        return (f"{len(out['differing'])} of {out['compared']} cells differ under {out['strategy']} "
                f"({', '.join(out['differing'][:4])}); the worst slot is {json.dumps(worst, sort_keys=True)}")
    return (f"every one of {out['compared']} cells is the same under {out['strategy']}, and a difference was "
            f"EXPECTED: the comparison cannot see what it was built to see")


__all__ = ["compare", "produce"]
