"""CELLS AND POINTS: what a comparison runs, each named once and referenced by name.

A CELL is a data provider and its constructor parameters. `data` names a `module:function` that, called with the cell's
name and its parameters, returns the data an arm runs on: a description of operands, a fixture, the inputs of a layer.
It builds nothing heavy itself (an arm builds its tensors from the data, in its own worker) and derives any seed from the
name, so a cell reproduces from its name alone. A registry file of cells names `data` once for its records, a record may
name its own, and every other field of a record except `name` is a parameter.

A POINT is a named group of cells by RUNNER -- rola, attention, a fla layer -- and what the group holds equal: `holds`,
the claim in words, recorded with every result, and `equal`, the parameters every cell of the point must carry with one
value, checked when the registry loads. A runner receives the data of each cell the point sends it and builds arms from
it or refuses; nothing but the point matches a cell to a runner.

    {"schema": 1, "data": "benchmarks.cells:carry_cell", "cells": [{"name": "flat-small-alt-k16", "tokens": 256, ...}]}
    {"schema": 1, "points": [{"name": "L256-N256-dv64", "holds": "...", "equal": ["tokens", "dv"],
                              "runners": {"rola": ["flat-small-alt-k16"], "attention": ["attn-L256-dv64"]}}]}

Standard library only: a registry loads and checks without torch; `build` imports the data provider where it is called.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

SCHEMA = 1


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
