"""QKV CELLS: attention's input as data -- queries, keys and values after their projections (moved from rola-bench's
rola_bench/measure/cells.py).

A QKV cell is ``batch`` sequences of ``tokens`` tokens, ``heads`` heads of width ``dv``, in ``dtype``, whether the
problem is causal, and the seed of the device generator its queries, keys and values are drawn from, in that order.
"""
from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True, slots=True)
class QKVCell:
    name: str
    seed: int
    tokens: int
    dv: int
    heads: int = 1
    batch: int = 1
    dtype: str = "bfloat16"
    causal: bool = True


def qkv_cell(name: str, **params) -> QKVCell:
    """A QKV cell's data provider (`qkv.json` names it)."""
    unknown = sorted(set(params) - {f.name for f in fields(QKVCell)})
    if unknown:
        raise ValueError(f"{name}: undeclared field(s) {unknown}")
    return QKVCell(name, **params)


def realize(cell: QKVCell, device: str = "cuda"):
    """``(q, k, v)``, each ``[batch, heads, tokens, dv]``."""
    import torch

    generator = torch.Generator(device=device).manual_seed(cell.seed)
    dtype = getattr(torch, cell.dtype)
    return tuple(torch.randn(cell.batch, cell.heads, cell.tokens, cell.dv, device=device, dtype=dtype, generator=generator)
                 for _ in range(3))


__all__ = ["QKVCell", "qkv_cell", "realize"]
