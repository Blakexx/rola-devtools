"""The registry: cells are data providers with parameters, names are unique, a file of neither bases nor cells is refused,
and derived cells merge their bases in order."""
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

    def test_what_a_registry_cannot_mean_is_refused(self):
        cells = write(self.dir, "cells.json", CELLS)
        with self.assertRaisesRegex(ValueError, "named twice"):
            Registry.load([cells, write(self.dir, "again.json", CELLS)])
        with self.assertRaisesRegex(ValueError, "neither bases nor cells"):
            Registry.load([write(self.dir, "p.json", {"schema": 1, "points": []})])
        with self.assertRaisesRegex(ValueError, "names no data provider"):
            Registry.load([write(self.dir, "bare.json", {"schema": 1, "cells": [{"name": "x"}]})])
        with self.assertRaisesRegex(ValueError, "schema"):
            Registry.load([write(self.dir, "old.json", {"cells": []})])


class Derived(unittest.TestCase):
    BASES = {"schema": 1, "bases": [{"name": "small", "data": "fake_provider:tokens", "tokens": 8, "scale": 1.0},
                                    {"name": "wide", "scale": 4.0}]}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def load(self, *cells_docs):
        paths = [write(self.dir, "bases.json", self.BASES)]
        paths += [write(self.dir, f"cells{i}.json", doc) for i, doc in enumerate(cells_docs)]
        return Registry.load(paths)

    def test_a_derived_cell_merges_its_bases_in_order_and_its_own_fields_last(self):
        registry = self.load({"schema": 1, "cells": [{"name": "d", "from": ["small", "wide"], "tokens": 16},
                                                     {"name": "direct", "data": "fake_provider:tokens", "tokens": 2}]})
        self.assertEqual(registry.cell("d"), {"name": "d", "data": "fake_provider:tokens",
                                              "params": {"tokens": 16, "scale": 4.0}})
        self.assertEqual((registry.derived["d"], "direct" in registry.derived), (("small", "wide"), False))
        self.assertNotIn("small", registry.cells)

    def test_vary_makes_one_named_cell_per_combination(self):
        registry = self.load({"schema": 1, "cells": [{"name": "s{seed}-t{tokens}", "from": ["small"],
                                                     "vary": {"seed": [1, 2], "tokens": [8, 16]}}]})
        self.assertEqual(sorted(registry.cells), ["s1-t16", "s1-t8", "s2-t16", "s2-t8"])
        self.assertEqual(registry.cell("s2-t16")["params"], {"tokens": 16, "scale": 1.0, "seed": 2})

    def test_what_a_derivation_cannot_mean_is_refused(self):
        with self.assertRaisesRegex(KeyError, "base registry does not hold"):
            self.load({"schema": 1, "cells": [{"name": "d", "from": ["absent"]}]})
        with self.assertRaisesRegex(ValueError, "from is a list"):
            self.load({"schema": 1, "cells": [{"name": "d", "from": "small"}]})
        with self.assertRaisesRegex(ValueError, "names each"):
            self.load({"schema": 1, "cells": [{"name": "d", "from": ["small"], "vary": {"seed": [1, 2]}}]})
        with self.assertRaisesRegex(ValueError, "never derived"):
            Registry.load([write(self.dir, "b.json", {"schema": 1, "bases": [{"name": "x", "from": ["y"]}]})])

    def test_code_derives_a_cell_into_a_registry(self):
        registry = self.load({"schema": 1, "cells": []})
        record = registry.derive("generated", ["small"], seed=9)
        self.assertEqual((record["params"], registry.derived["generated"]), ({"tokens": 8, "scale": 1.0, "seed": 9},
                                                                            ("small",)))
        with self.assertRaises(ValueError):
            registry.derive("generated", ["small"])


if __name__ == "__main__":
    os.chdir(HERE)
    unittest.main()
