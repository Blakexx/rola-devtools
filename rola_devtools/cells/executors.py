"""THE CELL NODE'S EXECUTOR: one cell's record, with the digest of the code that draws it.

It runs in the build system's own environment and reads this package, so a target that takes the cell keys on its record
AND on the drawing code: change a draw and every result over that cell re-keys, instead of a new sample landing on the
old record. The record is a DESCRIPTION -- the provider, the parameters and the seed -- and never the tensors: those are
drawn in the worker that times them, from the seed the record states, and each worker reports this digest back so a
checkout drawing with other code fails its entry instead of being compared.
"""
from __future__ import annotations

from functools import cache

from ..build import identity
from . import FILES, central


@cache
def draw_digest() -> str:
    """The digest of this package's own files: what a cell's data is drawn BY."""
    root = FILES[0].parent
    return identity.files(root, sorted(p.name for p in root.iterdir() if p.suffix in (".py", ".json")))


def record(ctx) -> dict:
    return {**central().cell(ctx.params["cell"]), "draw": draw_digest()}


__all__ = ["draw_digest", "record"]
