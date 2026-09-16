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


def _binding(module: str, cwd: str) -> dict:
    """WHERE THIS SIDE'S LIBRARY ACTUALLY CAME FROM, asserted rather than arranged for. An editable install resolves
    its package through the install's path entry -- the canonical checkout -- so a side running from another worktree
    can import the tree it is being compared against and agree with it perfectly, having compared nothing. A path
    insertion that did not take effect is exactly as silent as no insertion at all; this is what makes it loud."""
    where = Path(importlib.import_module(module).__file__).resolve()
    if not where.is_relative_to(Path(cwd).resolve()):
        raise AssertionError(f"this side runs in {cwd} and imported {module} from {where}, which is outside it: the "
                             f"comparison would have run one tree against itself")
    return {"module": module, "file": str(where)}


def produce(ctx) -> dict:
    """One side, over every cell the node takes as a data input. A cell whose call RAISES is that cell's domain
    failure: it is recorded and the other cells still run, because one unsupported configuration is not a reason to
    lose the comparison on the rest."""
    import os

    import torch

    #: THE THREADS ARE THE SLOTS HELD: torch's intra-op pool otherwise takes every core on the host, headroom included
    slots = os.environ.get("ROLA_HOST_BUDGET_SLOTS")
    if slots:
        torch.set_num_threads(max(1, int(slots)))
    binds = ctx.params.get("binds")
    binding = {**_binding(binds, str(Path.cwd())), "executor": ctx.params["executor"]} if binds else None
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
    return {"cells": out, "binding": binding}


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

    #: THE SAME FUNCTION FROM THE SAME LIBRARY COMPARED NOTHING, whatever it agreed with itself about. Two DIFFERENT
    #: functions in one checkout -- a kernel against its fp64 reference -- share a library by design.
    bindings = (left.output.get("binding"), right.output.get("binding"))
    same_file = all(bindings) and bindings[0]["file"] == bindings[1]["file"]
    if same_file and bindings[0].get("executor") == bindings[1].get("executor"):
        raise AssertionError(f"both sides ran {bindings[0].get('executor')} with {bindings[0]['module']} from "
                             f"{bindings[0]['file']}: this comparison ran one tree against itself and its verdict "
                             "means nothing")

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
        #: a quantity named as another's ENVELOPE is a bound the reference side supplies, never a thing compared
        envelopes = {params.get("envelope_from")} | {q.get("envelope_from") for q in params.get("per_quantity", {}).values()}
        names = sorted((set(a) | set(b)) - envelopes)
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
    quantities = sum(len(c.get("quantities", {})) for c in cells.values())
    held = (not differing and not unusable) if expect == "same" else (bool(differing) and not unusable)
    if quantities < ctx.params.get("minimum", 1):
        held = False
    out = {"strategy": strategy, "expect": expect, "cells": cells, "differing": differing, "unusable": unusable,
           "compared": len(cells) - len(unusable), "quantities": quantities,
           "minimum": ctx.params.get("minimum", 1), "held": held,
           "bindings": {"left": bindings[0], "right": bindings[1]}}
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
    if out["quantities"] < out["minimum"]:
        return (f"this comparison compared {out['quantities']} quantities and was declared to need "
                f"{out['minimum']}: a gate that cannot run says so, it does not report a verdict over whatever "
                f"survived")
    if out["expect"] == "same":
        worst = max((q for name in out["differing"] for q in out["cells"][name]["quantities"].values()
                     if not q.get("same")), key=lambda q: q.get("ratio", 0.0), default={})
        return (f"{len(out['differing'])} of {out['compared']} cells differ under {out['strategy']} "
                f"({', '.join(out['differing'][:4])}); the worst slot is {json.dumps(worst, sort_keys=True)}")
    return (f"every one of {out['compared']} cells is the same under {out['strategy']}, and a difference was "
            f"EXPECTED: the comparison cannot see what it was built to see")


__all__ = ["compare", "produce"]
