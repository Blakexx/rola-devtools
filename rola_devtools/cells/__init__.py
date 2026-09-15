"""THE CELL REGISTRY: every input a RoLA measurement or test runs on, named once, for every package.

A CELL is a data provider and its parameters. `data` names a `module:function` that, called with the cell's name and its
parameters, returns the cell's typed record; the record's own module realizes it into tensors in the worker that runs it,
and derives any seed from the cell, so a cell reproduces from its name alone. A registry file of cells names `data` once
for its records, a record may name its own, and every other field of a record except `name` is a parameter.

THE CENTRAL REGISTRY (`central()`, Blake 2026-09-15: "keeps one consistent source, and if a package doesnt work with one
of the central changes, then you just update that package") is this package's own files, one per kind of input:

- `carry.json` (`rola_devtools.cells.carry`): a routed recurrence's input -- routing amplitudes, gain, values, and the
  state the sequence enters with, including its backing -- with the draws and the regime each cell proves;
- `layer.json` (`rola_devtools.cells.layer`): a sequence layer's hidden states and values;
- `qkv.json` (`rola_devtools.cells.qkv`): attention's queries, keys and values.

A cell holds its input and never how a package runs it: a kernel's launch shape, RoLA's routing widths or gain, an
attention backend are an arm's parameters. A name is unique across every file loaded, so within one run a name is one
input for every package that reads it.

A POINT is a named group of cells by RUNNER and what the group holds equal: `holds`, the claim in words, recorded with
every result, and `equal`, the parameters every cell of the point must carry with one value, checked when the registry
loads.

    {"schema": 1, "data": "rola_devtools.cells.carry:carry_cell", "cells": [{"name": "flat-small-alt-k16", ...}]}
    {"schema": 1, "points": [{"name": "L256-N256-dv64", "holds": "...", "equal": ["tokens", "dv"],
                              "runners": {"rola": ["flat-small-alt-k16"], "attention": ["qkv-L256-dv64"]}}]}

Standard library only: a registry loads and checks without torch; realizing a cell imports torch where it is called.
"""
from __future__ import annotations

import importlib
import json
from functools import cache
from pathlib import Path

SCHEMA = 1
#: the central registry's files, one per kind of input
FILES = tuple(Path(__file__).with_name(name) for name in ("carry.json", "layer.json", "qkv.json"))


class Registry:
    """Cells and points from any number of registry files; a name is unique across all of them."""

    def __init__(self, cells: dict[str, dict], points: dict[str, dict]) -> None:
        self.cells, self.points = cells, points
        for point in points.values():
            self._resolve(point)

    @classmethod
    def load(cls, paths) -> Registry:
        cells: dict[str, dict] = {}
        points: dict[str, dict] = {}
        origin: dict[str, str] = {}
        for path in map(Path, paths):
            doc = json.loads(path.read_text())
            if doc.get("schema") != SCHEMA:
                raise ValueError(f"{path}: schema {doc.get('schema')!r}; this reader reads {SCHEMA}")
            if "cells" not in doc and "points" not in doc:
                raise ValueError(f"{path}: neither cells nor points")
            for record in doc.get("cells", []):
                name = record["name"]
                data = record.get("data", doc.get("data"))
                if not data or ":" not in data:
                    raise ValueError(f"{path}: cell {name} names no data provider (module:function)")
                cls._claim(origin, name, path)
                cells[name] = {"name": name, "data": data,
                               "params": {k: v for k, v in record.items() if k not in ("name", "data")}}
            for point in doc.get("points", []):
                cls._claim(origin, point["name"], path)
                points[point["name"]] = point
        return cls(cells, points)

    @staticmethod
    def _claim(origin: dict, name: str, path: Path) -> None:
        if name in origin:
            raise ValueError(f"{name} is named twice: {origin[name]} and {path}")
        origin[name] = str(path)

    def cell(self, name: str) -> dict:
        if name not in self.cells:
            raise KeyError(f"no cell {name!r} in the registry")
        return self.cells[name]

    def point(self, name: str) -> dict:
        """The point with its cells resolved to their records: `{name, holds, equal, runners: {runner: [record]}}`."""
        if name not in self.points:
            raise KeyError(f"no point {name!r} in the registry")
        return self._resolve(self.points[name])

    def adhoc(self, runners: dict[str, list[str]], holds: str = "", equal: tuple[str, ...] = ()) -> dict:
        """An unregistered point, resolved and checked like a registered one; its name says it is not one."""
        return self._resolve({"name": "adhoc:" + ",".join(f"{r}={'+'.join(c)}" for r, c in sorted(runners.items())),
                              "holds": holds, "equal": list(equal), "runners": runners})

    def _resolve(self, point: dict) -> dict:
        name, runners = point["name"], point.get("runners") or {}
        if not runners or any(not cells for cells in runners.values()):
            raise ValueError(f"point {name}: every runner it names needs at least one cell, got {runners}")
        missing = sorted({c for cells in runners.values() for c in cells} - set(self.cells))
        if missing:
            raise KeyError(f"point {name} names cells no registry holds: {missing}")
        resolved = {runner: [self.cells[c] for c in cells] for runner, cells in runners.items()}
        for field in point.get("equal", []):
            values = {c["name"]: c["params"].get(field) for records in resolved.values() for c in records}
            if len({json.dumps(v, sort_keys=True) for v in values.values()}) != 1 or None in values.values():
                raise ValueError(f"point {name} holds {field} equal, but its cells carry {values}")
        return {"name": name, "holds": point.get("holds", ""), "equal": list(point.get("equal", [])),
                "runners": resolved}


def build(record: dict):
    """The data of a cell: its provider called with its name and parameters."""
    module, _, function = record["data"].partition(":")
    return getattr(importlib.import_module(module), function)(record["name"], **record["params"])


@cache
def central() -> Registry:
    """The central registry: every cell of `FILES`, read once a process."""
    return Registry.load(FILES)


def cell(name: str):
    """A central cell's typed record (`CarryCell`, `LayerCell`, `QKVCell`) by name."""
    return build(central().cell(name))
