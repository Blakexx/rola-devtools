"""THE CELL REGISTRY: every input a RoLA measurement or test runs on, named once, for every package.

A CELL is a data provider and its parameters. `data` names a `module:function` that, called with the cell's name and its
parameters, returns the cell's typed record; the record's own module realizes it into tensors in the worker that runs it,
from the seed the cell states, so a cell reproduces from its record alone. A registry file of cells names `data` once
for its records, a record may name its own, and every other field of a record except `name` is a parameter.

THE CENTRAL REGISTRY (`central()`, Blake 2026-09-15: "keeps one consistent source, and if a package doesnt work with one
of the central changes, then you just update that package") is this package's own files, one per kind of input:

- `carry.json` (`rola_devtools.cells.carry`): a routed recurrence's input -- routing amplitudes, gain, values, and the
  state the sequence enters with, including its backing -- with the draws and the regime each cell proves;
- `layer.json` (`rola_devtools.cells.layer`): a sequence layer's hidden states and values;
- `qkv.json` (`rola_devtools.cells.qkv`): attention's queries, keys and values.

A cell holds its input and never how a package runs it: a kernel's launch shape, RoLA's routing widths or gain, an
attention backend are an arm's parameters. A name is unique across every file loaded, so within one run a name is one
input for every package that reads it. Every cell states the seed its data is drawn from.

BASES AND DERIVED CELLS. `bases.json` is the base registry: a BASE is a fragment -- any subset of a cell's fields (a
shape, a draw, a length) -- and never a cell: no key, no draw, no regime proof, no selection, and its name reaches no
record. A cell is DIRECT (a complete record) or DERIVED: `"from": ["flagship-shape", "alt-k4-draw"]` merges those bases
in order, later over earlier, its own fields last, and must come out a complete, valid cell. `"vary": {"seed": [1, 2]}`
with a name pattern (`"name": "flagship-alt-k4-seed{seed}"`) makes one cell per value (several vary fields, their
product). A derivation names bases only, never cells. `derive(name, bases, **fields)` is the same for code that
generates cells, and `Registry.derived` records each derived cell's bases, so a reading can group them.

    {"schema": 1, "data": "rola_devtools.cells.carry:carry_cell", "cells": [{"name": "flat-small-alt-k16", ...}]}

Which cells launch together, and what they hold equal, is a declaration's (a root's cell lists), never the registry's.

Standard library only: a registry loads and checks without torch; realizing a cell imports torch where it is called.
"""
from __future__ import annotations

import importlib
import json
from functools import cache
from itertools import product
from pathlib import Path

SCHEMA = 1
#: the central registry's files, one per kind of input
FILES = tuple(Path(__file__).with_name(name)
              for name in ("bases.json", "carry.json", "layer.json", "producer.json", "qkv.json"))


class Registry:
    """Bases and cells from any number of registry files. A cell name is unique across every file, and so is a base
    name; a cell may be DIRECT (a complete record) or DERIVED from bases (`from`, merged in order)."""

    def __init__(self, cells: dict[str, dict], bases: dict[str, dict] | None = None,
                 derived: dict[str, tuple[str, ...]] | None = None) -> None:
        self.cells, self.bases = cells, bases or {}
        self.derived = derived or {}

    @classmethod
    def load(cls, paths) -> Registry:
        docs = []
        for path in map(Path, paths):
            doc = json.loads(path.read_text())
            if doc.get("schema") != SCHEMA:
                raise ValueError(f"{path}: schema {doc.get('schema')!r}; this reader reads {SCHEMA}")
            if not {"cells", "bases"} & set(doc):
                raise ValueError(f"{path}: neither bases nor cells")
            docs.append((path, doc))
        registry = cls({})
        base_origin: dict[str, str] = {}
        for path, doc in docs:
            for base in doc.get("bases", []):
                cls._claim(base_origin, base["name"], path)
                if "from" in base or "vary" in base:
                    raise ValueError(f"{path}: base {base['name']} derives; a base is a fragment, never derived")
                registry.bases[base["name"]] = {k: v for k, v in base.items() if k != "name"}
        origin: dict[str, str] = {}
        for path, doc in docs:
            for record in doc.get("cells", []):
                for name, data, params, bases in registry._expand(record, doc.get("data"), path):
                    cls._claim(origin, name, path)
                    registry.cells[name] = {"name": name, "data": data, "params": params}
                    if bases:
                        registry.derived[name] = bases
        return registry

    def _expand(self, record: dict, default_data: str | None, where) -> list[tuple[str, str, dict, tuple[str, ...]]]:
        """One record as the cells it declares: its bases merged in order, its own fields over them, and one cell per
        combination of its `vary` values, named by formatting its name with them."""
        bases = record.get("from", [])
        if isinstance(bases, str) or not isinstance(bases, list):
            raise ValueError(f"{where}: cell {record['name']}: from is a list of bases, got {bases!r}")
        unknown = [b for b in bases if b not in self.bases]
        if unknown:
            raise KeyError(f"{where}: cell {record['name']} derives from {unknown}, which the base registry does not hold")
        merged: dict = {}
        for base in bases:
            merged.update(self.bases[base])
        merged.update({k: v for k, v in record.items() if k not in ("name", "from", "vary")})
        data = merged.pop("data", None) or default_data
        if not data or ":" not in data:
            raise ValueError(f"{where}: cell {record['name']} names no data provider (module:function)")
        vary = record.get("vary", {})
        combos = [dict(zip(sorted(vary), values, strict=True)) for values in product(*(vary[k] for k in sorted(vary)))]
        out = []
        for combo in combos or [{}]:
            if combo and any("{" + field + "}" not in record["name"] for field in combo):
                raise ValueError(f"{where}: cell {record['name']} varies {sorted(vary)}; its name names each ({{field}})")
            out.append((record["name"].format(**combo), data, {**merged, **combo}, tuple(bases)))
        return out

    def derive(self, name: str, bases=(), **fields) -> dict:
        """A derived cell declared by code: `bases` merged in order, `fields` over them, added to this registry."""
        ((cell_name, data, params, from_),) = self._expand({"name": name, "from": list(bases), **fields}, None, "derive")
        if cell_name in self.cells:
            raise ValueError(f"{cell_name} is named twice")
        self.cells[cell_name] = {"name": cell_name, "data": data, "params": params}
        if from_:
            self.derived[cell_name] = from_
        return self.cells[cell_name]

    @staticmethod
    def _claim(origin: dict, name: str, path: Path) -> None:
        if name in origin:
            raise ValueError(f"{name} is named twice: {origin[name]} and {path}")
        origin[name] = str(path)

    def cell(self, name: str) -> dict:
        if name not in self.cells:
            raise KeyError(f"no cell {name!r} in the registry")
        return self.cells[name]


def build(record: dict):
    """The data of a cell: its provider called with its name and parameters."""
    module, _, function = record["data"].partition(":")
    return getattr(importlib.import_module(module), function)(record["name"], **record["params"])


@cache
def central() -> Registry:
    """The central registry: every cell of `FILES`, read once a process."""
    return Registry.load(FILES)


def derive(name: str, bases=(), **fields) -> dict:
    """A derived cell added to the central registry by code (a declaration generating cells): see `Registry.derive`."""
    return central().derive(name, bases, **fields)


def cell(name: str):
    """A central cell's typed record (`CarryCell`, `LayerCell`, `QKVCell`) by name."""
    return build(central().cell(name))
