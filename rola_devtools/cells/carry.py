"""CARRY CELLS: a routed recurrence's input as data -- per-level routing amplitudes on the simplex for both sides, a gain,
the values, and the state the sequence enters with -- with one draw behind each record (moved from rola's
benchmarks/cells).

`carry.json` holds the records; `carry_cell` validates one, and `realize` draws it: ``[B, L, H, B_l]`` read and write
amplitudes per level, a ``[B, L, H]`` gain and ``[B, L, H, DV]`` values, in bf16, plus the entry state of a carried
cell. A record declares a DRAW, a SHAPE and the STATE it binds -- which backing (`dense` or `paged`) and whether it
enters empty (`fresh`), carries a drawn state (`carried`) or binds none -- and never how a kernel runs it: a launch
shape, a window grid or a stream count is the arm's, so two kernels timed on one cell read the same tensors.

Every cell but the two degenerate ones states where in the routing distribution it sits (`regime`, on every axis of
`rola_devtools.cells.regimes.REGIME_AXES`), and `realize` proves it before any reader sees the draw.

Symbols, each at first use: ``D`` = routing depth, ``B_l`` = level ``l``'s digit count, ``N = prod_l B_l`` = leaf
capacity, ``DV`` = the value width, ``L`` = tokens, ``BH`` = batch times heads, ``k_tok`` = a token's nonzero digits
per sparse level (a property of the draw, never told to a kernel).

Standard library until a draw is realized: `realize` imports torch where it is called.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

DRAWS = ("dense", "alt", "both", "cohort", "dead", "deposit",
         "tied", "anti", "cold", "readonly", "concentrated", "onehot", "flip")
#: the draws with no support to sample, which state it themselves and declare no regime
DEGENERATE = ("dead", "deposit")
#: the draws that carry a `k_tok`
COUNTED = ("alt", "both", "cohort", "tied")
BACKINGS = ("dense", "paged")
STATES = ("none", "fresh", "carried")
#: who a cell is sized for: `oracle`, small enough for an fp64 reference; `probe`, a measurement's size; `both`
TIERS = ("oracle", "probe", "both")
#: THE TOKEN WINDOW the `tail` regime and the `flip` draw are stated against: RoLA's carry kernel window
#: (`rola.ops.carry.WINDOW`; rola's tests hold the two equal). A cell's input does not change with it, only what the
#: two claims mean.
WINDOW = 512
#: THE ONE TOKEN a ``deposit`` draw makes live. Not the window's first: a token index a kernel's segment layout has to
#: place correctly is what makes the cell a test of the layout rather than of its origin.
DEPOSIT_TOKEN = 5


@dataclass(frozen=True, slots=True)
class CarryCell:
    """One record, validated. ``seed`` is the draw's random seed, stated by the record, so a failure reproduces from the
    cell alone and a derived cell can change it."""

    name: str
    seed: int
    widths: tuple[int, ...]
    dv: int
    tokens: int
    draw: str
    k_tok: int | None
    cohort: int | None
    support: float
    backing: str
    state: str
    tier: str
    #: ``((axis, value), ...)`` on every axis of `REGIME_AXES`; None for a degenerate draw
    regime: tuple[tuple[str, str], ...] | None

    @property
    def D(self) -> int:
        return len(self.widths)

    @property
    def N(self) -> int:
        n = 1
        for width in self.widths:
            n *= width
        return n

    def declared_sparsity(self) -> tuple[tuple[bool, bool], ...]:
        """Per level, ``(read sparse, write sparse)``: the support the draw DECLARES, which a kernel may skip on. A dense,
        degenerate or corner draw declares nothing; the alternation declares the read side sparse at the odd levels and
        the write side at the even ones; ``both`` declares level zero sparse on both sides. A declaration may understate
        what the draw carries (a tied draw's zeros are undeclared) and never overstates it."""
        if self.draw not in ("alt", "both", "cohort") or self.k_tok is None:
            return ((False, False),) * self.D
        if self.draw == "both":
            return tuple((level == 0, level == 0) for level in range(self.D))
        return tuple((bool(level % 2), not level % 2) for level in range(self.D))


def carry_cell(name: str, **params) -> CarryCell:
    """A carry cell's data provider (`carry.json` names it): the record's parameters, validated."""
    from .regimes import REGIME_AXES

    regime = params["regime"]
    cell = CarryCell(name=name, seed=params["seed"], widths=tuple(params["widths"]), dv=params["dv"], tokens=params["tokens"],
                     draw=params["draw"], k_tok=params["k_tok"], cohort=params["cohort"], support=params["support"],
                     backing=params["backing"], state=params["state"], tier=params["tier"],
                     regime=None if regime is None else tuple((axis, regime[axis]) for axis in REGIME_AXES
                                                              if axis in regime))
    unknown = sorted(set(params) - {f.name for f in dataclasses.fields(CarryCell)})
    if unknown:
        raise ValueError(f"{name}: undeclared field(s) {unknown}")
    if (cell.draw in DEGENERATE) != (regime is None):
        raise ValueError(f"{name}: a {cell.draw} draw {'declares no' if cell.draw in DEGENERATE else 'states its'} regime")
    if regime is not None:
        if set(regime) != set(REGIME_AXES):
            raise ValueError(f"{name}: the regime states {sorted(regime)}, not every axis of {list(REGIME_AXES)}")
        for axis, value in regime.items():
            if value not in REGIME_AXES[axis]:
                raise ValueError(f"{name}: {value!r} is not a value of the {axis} axis {REGIME_AXES[axis]}")
    for field, allowed in (("draw", DRAWS), ("backing", BACKINGS), ("state", STATES), ("tier", TIERS)):
        if getattr(cell, field) not in allowed:
            raise ValueError(f"{name}: {field} {getattr(cell, field)!r} is not one of {allowed}")
    if (cell.draw == "cohort") != (cell.cohort is not None):
        raise ValueError(f"{name}: a cohort draw carries a cohort and nothing else does")
    if cell.draw in COUNTED and cell.k_tok is None:
        raise ValueError(f"{name}: a sparse draw states its k_tok")
    if cell.draw not in COUNTED and cell.k_tok is not None:
        raise ValueError(f"{name}: a {cell.draw} draw has no k_tok -- its support is stated by the draw itself, not by "
                         "a count")
    return cell


# ------------------------------------------------------------------ the draw

def simplex(shape, k_tok, gen, device, live=None):
    """A row-normalized draw with exactly ``k_tok`` nonzeros per row, or dense.

    ``live`` truncates a row to its FIRST ``live`` digits -- STRUCTURED support, as against ``k_tok``'s unstructured
    one. Unstructured sparsity at long ``L`` still reaches every page, so only a truncation can produce a page the
    routing never touches (the idle-resident cell). It is applied after both draws so a cell's RNG stream does not depend
    on it.
    """
    import torch

    x = torch.rand(shape, device=device, dtype=torch.float64, generator=gen)
    if k_tok is not None and k_tok < shape[-1]:
        keep = torch.zeros(shape, device=device, dtype=torch.float64)
        idx = torch.argsort(torch.rand(shape, device=device, generator=gen), dim=-1)[..., :k_tok]
        keep.scatter_(-1, idx, 1.0)
        x = x * keep
    if live is not None and live < shape[-1]:
        x[..., live:] = 0.0
    return x / x.sum(-1, keepdim=True)


def clustered(shape, k_tok, cohort, gen, device):
    """A simplex draw whose support is ONE contiguous digit window per run of ``cohort`` consecutive tokens -- the
    structure a whole-window skip exists to exploit, and the one that leaves an owner a handful of live tokens a
    window."""
    import torch

    B, T, H, width = shape
    x = torch.rand(shape, device=device, dtype=torch.float64, generator=gen)
    starts = torch.randint(0, width, (B, T // cohort, 1, 1), device=device, generator=gen)
    idx = (starts.expand(B, T // cohort, cohort, 1).reshape(B, T, 1, 1)
           + torch.arange(k_tok, device=device).view(1, 1, 1, k_tok)) % width
    keep = torch.zeros(shape, device=device, dtype=torch.float64)
    keep.scatter_(-1, idx.expand(B, T, H, k_tok), 1.0)
    x = x * keep
    return x / x.sum(-1, keepdim=True)


@dataclass(frozen=True, slots=True)
class RealizedCell:
    """A drawn cell: the bf16 operands a kernel gets, and their fp64 doubles.

    A reference reads the ROUNDED operands, never the raw fp64 draws: comparing an fp64 oracle of the fp64 draws against
    a kernel fed their bf16 images would measure the cast, which is not what is under test.
    """

    cell: CarryCell
    read: tuple
    write: tuple
    gain: object
    v: object
    #: ``[BH, N, DV + 1]`` fp32, CANONICAL leaf order (level 0 the most significant digit): the entry state a
    #: ``carried`` cell binds (value columns ``0.1 * N(0, 1)``, the mass column their magnitudes), or None. Drawn after
    #: every operand, so binding one moves no other tensor of the cell.
    entry: object = None

    def doubles(self):
        return (tuple(x.double() for x in self.read), tuple(x.double() for x in self.write),
                self.gain.double(), self.v.double())


def _degenerate(cell, shape, gen, device):
    """The two draws with NO support to sample: nothing live, and exactly one deposit.

    ``dead`` puts zero amplitude on every digit of every level on both sides, so no token reads and no token writes --
    the recurrence's identity, whose whole content is that the state comes out as it went in. ``deposit`` is the
    single-tile point of the fold: one token, one-hot on digit zero of every write level, so exactly one leaf receives
    exactly one deposit, and the read side is dead so the readout is the zero it would be against an entry state a
    single window never reads.
    """
    import torch

    B, T, H = shape
    zeros = tuple(torch.zeros(B, T, H, w, device=device, dtype=torch.float64) for w in cell.widths)
    write = zeros
    if cell.draw == "deposit":
        write = tuple(torch.zeros(B, T, H, w, device=device, dtype=torch.float64) for w in cell.widths)
        for level in write:
            level[:, DEPOSIT_TOKEN, :, 0] = 1.0
    gain = torch.rand(B, T, H, device=device, dtype=torch.float64, generator=gen) + 0.5
    v = torch.randn(B, T, H, cell.dv, device=device, dtype=torch.float64, generator=gen)
    return RealizedCell(cell=cell, read=tuple(x.to(torch.bfloat16) for x in zeros),
                        write=tuple(x.to(torch.bfloat16) for x in write), gain=gain.to(torch.bfloat16),
                        v=v.to(torch.bfloat16))


def _masked(shape, mask, gen, device):
    """A simplex draw confined to the digits ``mask`` marks (a ``[..., width]`` 0/1 tensor broadcast over ``shape``)."""
    import torch

    x = torch.rand(shape, device=device, dtype=torch.float64, generator=gen) * mask
    return x / x.sum(-1, keepdim=True)


def _one_hot(shape, gen, device):
    import torch

    digit = torch.randint(0, shape[-1], shape[:-1] + (1,), device=device, generator=gen)
    return torch.zeros(shape, device=device, dtype=torch.float64).scatter_(-1, digit, 1.0)


def _corner(cell, shape, gen, device):
    """``(read, write)`` for a CORNER draw -- a named region of the distribution a random draw does not reach.

    ``tied``: one ``k_tok`` draw per level is both sides. ``anti``: level zero reads its first half of digits and writes
    its second half, so no token reads a leaf it writes and every read leaf is cold. ``cold``: level zero's write side is
    confined to its leading ``support`` fraction under dense reads. ``readonly``: every token writes leaf zero and nothing
    else. ``concentrated``: every row carries 0.9 of its mass on one digit. ``onehot``: one live digit per row on both
    sides. ``flip``: level zero's write support is its first half of digits in even windows (`WINDOW`) and its second
    half in odd ones, so which window last wrote a leaf changes exactly at a window line.
    """
    import torch

    def dense(width):
        return simplex(shape + (width,), None, gen, device)

    def half(width, second):
        mask = torch.zeros(width, device=device, dtype=torch.float64)
        mask[width // 2:] = float(second)
        mask[:width // 2] = float(not second)
        return mask

    widths = cell.widths
    if cell.draw == "tied":
        read = tuple(simplex(shape + (w,), cell.k_tok, gen, device) for w in widths)
        return read, read
    if cell.draw == "onehot":
        return (tuple(_one_hot(shape + (w,), gen, device) for w in widths),
                tuple(_one_hot(shape + (w,), gen, device) for w in widths))
    if cell.draw == "concentrated":
        def peaked(width):
            x = torch.rand(shape + (width,), device=device, dtype=torch.float64, generator=gen)
            return 0.1 * x / x.sum(-1, keepdim=True) + 0.9 * _one_hot(shape + (width,), gen, device)
        return tuple(peaked(w) for w in widths), tuple(peaked(w) for w in widths)
    read = [dense(w) for w in widths]
    write = [dense(w) for w in widths]
    w0 = widths[0]
    if cell.draw == "anti":
        read[0] = _masked(shape + (w0,), half(w0, second=False), gen, device)
        write[0] = _masked(shape + (w0,), half(w0, second=True), gen, device)
    elif cell.draw == "cold":
        mask = torch.zeros(w0, device=device, dtype=torch.float64)
        mask[:max(1, int(w0 * cell.support))] = 1.0
        write[0] = _masked(shape + (w0,), mask, gen, device)
    elif cell.draw == "readonly":
        write = [torch.zeros(shape + (w,), device=device, dtype=torch.float64) for w in widths]
        for level in write:
            level[..., 0] = 1.0
    elif cell.draw == "flip":
        odd = (torch.arange(shape[1], device=device) // WINDOW) % 2 == 1
        mask = torch.where(odd.view(1, -1, 1, 1), half(w0, second=True), half(w0, second=False))
        write[0] = _masked(shape + (w0,), mask, gen, device)
    return tuple(read), tuple(write)


def realize(cell: CarryCell, B: int = 1, H: int = 1, device: str = "cuda") -> RealizedCell:
    """Draw ``cell``: ``[B, L, H, B_l]`` per level, ``[B, L, H]`` gain, ``[B, L, H, DV]`` v, and a carried cell's entry.

    THE ALTERNATION (Blake, 2026-08-19): per level at most ONE sparse side, and the sparse side alternates -- EVEN levels
    write-sparse / read-dense, ODD levels read-sparse / write-dense -- so the two sides' clause sets are different level
    subsets. ``both`` is sparse on both sides at level zero and dense above it; the two sides are drawn INDEPENDENTLY
    there on purpose, because a kernel may never assume they agree.
    """
    import torch

    gen = torch.Generator(device=device).manual_seed(cell.seed)
    shape = (B, cell.tokens, H)
    live = tuple(max(1, int(w * cell.support)) for w in cell.widths)

    def draw(width, kt, lv):
        if kt is None or kt >= width:
            return simplex(shape + (width,), None, gen, device, live=lv)
        if cell.draw == "cohort":
            return clustered(shape + (width,), kt, cell.cohort, gen, device)
        return simplex(shape + (width,), kt, gen, device, live=lv)

    if cell.draw in DEGENERATE:
        drawn = _degenerate(cell, shape, gen, device)
    else:
        kt = cell.k_tok
        if cell.draw in ("dense", "alt", "both", "cohort"):
            if cell.draw == "dense":
                read_k = write_k = [None] * cell.D
            elif cell.draw == "both":
                read_k = write_k = [kt if level == 0 else None for level in range(cell.D)]
            else:
                read_k = [kt if level % 2 else None for level in range(cell.D)]
                write_k = [None if level % 2 else kt for level in range(cell.D)]
            read = tuple(draw(w, read_k[level], live[level]) for level, w in enumerate(cell.widths))
            write = tuple(draw(w, write_k[level], live[level]) for level, w in enumerate(cell.widths))
        else:
            read, write = _corner(cell, shape, gen, device)
        gain = torch.rand(*shape, device=device, dtype=torch.float64, generator=gen) + 0.5
        v = torch.randn(*shape, cell.dv, device=device, dtype=torch.float64, generator=gen)
        drawn = RealizedCell(cell=cell, read=tuple(x.to(torch.bfloat16) for x in read),
                             write=tuple(x.to(torch.bfloat16) for x in write), gain=gain.to(torch.bfloat16),
                             v=v.to(torch.bfloat16))
    if cell.state == "carried":
        entry = 0.1 * torch.randn(B * H, cell.N, cell.dv + 1, device=device, dtype=torch.float64, generator=gen)
        entry[..., cell.dv] = entry[..., cell.dv].abs()
        drawn = dataclasses.replace(drawn, entry=entry.to(torch.float32))
    if cell.regime is not None:
        from .regimes import prove

        prove(drawn)
    return drawn


__all__ = ["BACKINGS", "DEGENERATE", "DEPOSIT_TOKEN", "DRAWS", "STATES", "TIERS", "WINDOW", "CarryCell", "RealizedCell",
           "carry_cell", "clustered", "realize", "simplex"]
