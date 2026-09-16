"""SIDES a diff test can run: each takes a cell record and returns named tensors, which is the whole contract a
declaration's `module:function` has to meet."""
from __future__ import annotations

import torch


def flat(record, *, value=1.0, bump=0.0, where=None, dtype="float32") -> dict:
    """A cell-shaped block of `value`, with one slot moved by `bump` -- the planted difference a diff must see."""
    n = int(record["params"]["tokens"]) // 64
    out = torch.full((n,), float(value), dtype=getattr(torch, dtype))
    if bump:
        out[0 if where is None else int(where)] += float(bump)
    return {"y": out, "state": torch.arange(n, dtype=torch.float64)}


def sparse(record, *, drop=None, **_kw) -> dict:
    """Every other slot nonzero, so a side that `drop`s one moves the SUPPORT rather than a value."""
    n = int(record["params"]["tokens"]) // 64
    out = torch.zeros(n, dtype=torch.float64)
    out[::2] = torch.arange(1, n // 2 + 1, dtype=torch.float64)
    if drop is not None:
        out[int(drop)] = 0.0
    return {"y": out}


def refuses(record, **_kw) -> dict:
    raise RuntimeError(f"this side does not support {record['name']}")


def refuses_one(record, **kw) -> dict:
    """Refuses `corner-anti` the way a binary refuses an arm it does not carry, and produces every other cell."""
    if record["name"] == "corner-anti":
        raise RuntimeError("this build carries no arm for corner-anti")
    return flat(record, **kw)
