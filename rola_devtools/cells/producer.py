"""PRODUCER CELLS: the routing producer's input as data -- logits at a stated SCALE, a mask, and the alpha they are
solved at.

The producer is the step that turns logits into routing amplitudes on the simplex (entmax at `alpha`), and the thing
that distinguishes two implementations of it is not the amplitudes in the middle of the distribution but WHERE THE
SUPPORT ENDS. A cell therefore declares its `scale`, because the scale is the axis that moves the boundary:

    0.5   a nearly dense simplex -- almost every digit survives, and a support disagreement is nearly impossible
    3     the working regime
    30    where a bisection's finite iteration count shows, and where two implementations are likeliest to disagree
          about which digits survive at all

`mask` is a RAGGED one when a cell declares it, never a uniform one: padding past a level's logical width is what a
kernel's mask does, and a uniform mask exercises none of the boundary the mask exists for. `pairs` makes the cell a
COMPOSED one: two independent logit tensors, a read side and a write side, for the union split that follows the solve.

Symbols, each at first use: `width` = the level's digit count, `rows` = independent logit rows drawn, `alpha` = the
entmax parameter (1.5 is the sparse producer, 2 is sparsemax).

Standard library until a draw is realized: `realize` imports torch where it is called.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

#: the alphas a shipped producer solves at
ALPHAS = (1.5, 2.0)


@dataclass(frozen=True, slots=True)
class ProducerCell:
    name: str
    seed: int
    width: int
    scale: float
    alpha: float = 1.5
    rows: int = 8
    masked: bool = False
    #: a composed cell: a read side and a write side, drawn independently, for the union split
    pairs: bool = False
    #: the leading dimensions a paired cell's logits carry (`[B, L, width]`), so the split is exercised over a batch
    lead: tuple = ()

    @property
    def shape(self) -> tuple:
        return (*self.lead, self.rows, self.width) if self.lead else (self.rows, self.width)


def producer_cell(name: str, **params) -> ProducerCell:
    """A producer cell's data provider (`producer.json` names it)."""
    unknown = sorted(set(params) - {f.name for f in fields(ProducerCell)})
    if unknown:
        raise ValueError(f"{name}: undeclared field(s) {unknown}")
    if "lead" in params:
        params["lead"] = tuple(params["lead"])
    cell = ProducerCell(name, **params)
    if cell.alpha not in ALPHAS:
        raise ValueError(f"{name}: alpha {cell.alpha} is not one a shipped producer solves at {ALPHAS}")
    if cell.width < 1 or cell.rows < 1:
        raise ValueError(f"{name}: a cell draws at least one row of at least one digit")
    return cell


def realize(cell: ProducerCell, device: str = "cpu"):
    """`(logits, mask)` for a plain cell, `(read_logits, write_logits, mask)` for a paired one.

    fp64 and on the CPU by default: the producer's reference is a pure-torch fp64 object, and what is being compared
    is arithmetic rather than a launch.
    """
    import torch

    gen = torch.Generator(device=device).manual_seed(cell.seed)
    draw = lambda: torch.randn(cell.shape, generator=gen, dtype=torch.float64, device=device) * cell.scale  # noqa: E731
    mask = None
    if cell.masked:
        mask = torch.ones(cell.shape, dtype=torch.bool, device=device)
        #: RAGGED, a different live width per row: a uniform mask is a narrower cell, not a masked one
        flat = mask.reshape(-1, cell.width)
        for row in range(flat.shape[0]):
            flat[row, max(1, cell.width - row % cell.width):] = False
    if cell.pairs:
        return draw(), draw(), mask
    return draw(), mask


__all__ = ["ALPHAS", "ProducerCell", "producer_cell", "realize"]
