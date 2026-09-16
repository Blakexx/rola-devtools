"""THE DIFF CLIENT: two executors, the same cells, one comparison rule, and a verdict the build can fail on.

A third client of the build system beside timing and the store, and the same shape: declaration helpers
(`declare.py`), executors that run in the environments the declaration names (`executors.py`), and the comparison
rules themselves (`strategies.py`).

    left  = side(g, "left",  env=tip,      executor="tests.oracle.subjects:carry_kernel", cells=cell_nodes)
    right = side(g, "right", env=baseline, executor="tests.oracle.subjects:carry_kernel", cells=cell_nodes)
    gate  = diff(g, "carry-vs-baseline", left=left, right=right, strategy="bit-identical")

WHAT IT IS FOR. "The same function in two checkouts" and "the kernel against its fp64 reference" are the same
question asked twice: run both, compare per cell under a stated rule, and say whether the difference is the one that
was expected. A side is an ordinary target, so "against `HEAD~1`" is a second checkout the root already composes; the
cells are cell nodes, so both sides are drawn from one record by one draw digest.

THE RAW NEVER LEAVES THE WORKSPACE. Each side writes its tensors into its own node's workspace, which is the build
cache -- wipeable, and swept. What the diff node outputs, and what a store target beside it files, is the DIFF: per
cell and quantity the worst slot and where it is, what bound it, how many slots were compared and how many failed.
"""
from __future__ import annotations

from .declare import diff, side

__all__ = ["diff", "side"]
