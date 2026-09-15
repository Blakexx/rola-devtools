"""LAYER CELLS: a sequence layer's input as data -- the hidden states a layer's projections read and the values it mixes
-- drawn from a seed (moved from rola's benchmarks/cells/layer.py, its constructor half left to the arms).

A layer cell is ONLY the input: a batch of ``B`` sequences of ``tokens`` prefill tokens (and ``decode_steps`` more, taken
one at a time after the prefill), hidden states of width ``hidden`` and values of ``H`` heads of width ``dv``, in one
dtype. How a layer is built from it -- RoLA's routing widths, its entmax alpha, its logit gain; an attention layer's
head count -- is the arm's parameters, so two arms on one layer cell read the same tensors. ``seed`` is the cell's: an
arm that initializes parameters seeds them from it (`torch.manual_seed(seed)` immediately before construction), and
the inputs come from a device generator at ``seed + 500``, so an arm's construction never moves them.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

DTYPES = {"bf16": "bfloat16", "fp32": "float32"}
#: the offset of the input generator's seed from the parameter seed
INPUT_SEED_OFFSET = 500


@dataclass(frozen=True, slots=True)
class LayerCell:
    """One layer input. Nothing derived is stored."""

    name: str
    B: int
    tokens: int
    H: int
    #: the hidden width; 0 means ``H * dv``
    hidden: int
    dv: int
    dtype: str
    seed: int
    decode_steps: int
    note: str

    @property
    def hidden_size(self) -> int:
        return self.hidden or self.H * self.dv

    @property
    def torch_dtype(self):
        import torch

        return getattr(torch, DTYPES[self.dtype])


def layer_cell(name: str, **params) -> LayerCell:
    """A layer cell's data provider (`layer.json` names it): the record's parameters, checked."""
    unknown = sorted(set(params) - {f.name for f in fields(LayerCell)})
    if unknown:
        raise ValueError(f"{name}: undeclared field(s) {unknown}")
    cell = LayerCell(name=name, **params)
    if cell.dtype not in DTYPES:
        raise ValueError(f"{name}: dtype {cell.dtype!r} is not one of {sorted(DTYPES)}")
    return cell


@dataclass(frozen=True, slots=True)
class RealizedLayer:
    """A drawn layer cell: ``x`` ``[B, tokens + decode_steps, hidden]`` and ``v`` ``[B, tokens + decode_steps, H, dv]``."""

    cell: LayerCell
    x: object
    v: object

    @property
    def prefill(self):
        """``(x, v)`` over the prefill tokens alone, contiguous."""
        T = self.cell.tokens
        return self.x[:, :T].contiguous(), self.v[:, :T].contiguous()


def realize(cell: LayerCell, device: str = "cuda") -> RealizedLayer:
    import torch

    gen = torch.Generator(device=device).manual_seed(cell.seed + INPUT_SEED_OFFSET)
    total = cell.tokens + cell.decode_steps
    x = torch.randn((cell.B, total, cell.hidden_size), device=device, dtype=cell.torch_dtype, generator=gen)
    v = torch.randn((cell.B, total, cell.H, cell.dv), device=device, dtype=cell.torch_dtype, generator=gen)
    return RealizedLayer(cell, x, v)


__all__ = ["DTYPES", "LayerCell", "RealizedLayer", "layer_cell", "realize"]
