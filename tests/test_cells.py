"""The registry: cells are data providers with parameters, points group cells by runner, names are unique, references
resolve, and a point's `equal` claim is checked when it loads."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from rola_devtools.cells import Registry, build

HERE = Path(__file__).resolve().parent


def write(directory: Path, name: str, doc: dict) -> Path:
    path = directory / name
    path.write_text(json.dumps(doc))
    return path


CELLS = {"schema": 1, "data": "fake_provider:tokens", "note": "documentation is not a cell",
         "cells": [{"name": "t8", "tokens": 8}, {"name": "t16", "tokens": 16, "scale": 2.0},
                   {"name": "other", "data": "fake_provider:tokens", "tokens": 8}]}


class Registries(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_cell_is_its_data_provider_and_every_other_field_is_a_parameter(self):
        registry = Registry.load([write(self.dir, "cells.json", CELLS)])
        self.assertEqual(registry.cell("t16"), {"name": "t16", "data": "fake_provider:tokens",
                                                "params": {"tokens": 16, "scale": 2.0}})
        sys.path.insert(0, str(HERE))
        try:
            self.assertEqual(build(registry.cell("t16")).scale, 2.0)
        finally:
            sys.path.remove(str(HERE))

    def test_a_point_resolves_its_cells_by_runner_and_checks_what_it_holds_equal(self):
        points = {"schema": 1, "points": [
            {"name": "eight", "holds": "8 tokens", "equal": ["tokens"], "runners": {"rola": ["t8"], "other": ["other"]}},
            {"name": "mixed", "equal": ["tokens"], "runners": {"rola": ["t8", "t16"]}}]}
        cells = write(self.dir, "cells.json", CELLS)
        with self.assertRaisesRegex(ValueError, "mixed holds tokens equal"):
            Registry.load([cells, write(self.dir, "points.json", points)])
        points["points"].pop()
        registry = Registry.load([cells, write(self.dir, "points.json", points)])
        eight = registry.point("eight")
        self.assertEqual([c["name"] for c in eight["runners"]["other"]], ["other"])
        self.assertEqual(eight["holds"], "8 tokens")

    def test_what_a_registry_cannot_mean_is_refused(self):
        cells = write(self.dir, "cells.json", CELLS)
        with self.assertRaisesRegex(ValueError, "named twice"):
            Registry.load([cells, write(self.dir, "again.json", CELLS)])
        with self.assertRaisesRegex(KeyError, "no registry holds"):
            Registry.load([cells, write(self.dir, "p.json", {"schema": 1, "points": [
                {"name": "p", "runners": {"rola": ["absent"]}}]})])
        with self.assertRaisesRegex(ValueError, "at least one cell"):
            Registry.load([cells, write(self.dir, "p.json", {"schema": 1, "points": [{"name": "p", "runners": {"r": []}}]})])
        with self.assertRaisesRegex(ValueError, "names no data provider"):
            Registry.load([write(self.dir, "bare.json", {"schema": 1, "cells": [{"name": "x"}]})])
        with self.assertRaisesRegex(ValueError, "schema"):
            Registry.load([write(self.dir, "old.json", {"cells": []})])
        self.assertTrue(Registry.load([cells]).adhoc({"rola": ["t8"]})["name"].startswith("adhoc:"))


if __name__ == "__main__":
    os.chdir(HERE)
    unittest.main()
