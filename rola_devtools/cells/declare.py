"""THE CELL TARGETS: one node a cell, declared once a graph and taken as a data input by whatever runs on it.

    inputs = [cell(g, name) for name in names]      # in a declaration file

A cell is a node like any other: its output is the cell's record with the digest of the code that draws it
(`executors.record`), and a target that takes it keys on that output the way it keys on any dependency. Nothing else in
the build system knows what a cell is. The node is uncached because it is milliseconds and because its output must move
when the drawing code does.
"""
from __future__ import annotations

EXECUTOR = "rola_devtools.cells.executors:record"


def cell(g, name: str):
    """The build's node for `name`, declared on first ask and shared by every target that takes it -- UNSCOPED, because
    a cell is one input for the whole build and two checkouts' declarations must reach the same node."""
    existing = g.targets.get(f"cells/{name}")
    if existing is not None:
        return existing
    return g.unscoped().node(f"cells/{name}", executor=EXECUTOR, params={"cell": name}, cache=False)


def cells(g, names) -> list:
    return [cell(g, name) for name in names]


__all__ = ["EXECUTOR", "cell", "cells"]
