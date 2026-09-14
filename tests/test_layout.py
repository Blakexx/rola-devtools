"""The checkout is linked onto every RoLA venv's path (a `.pth` naming its root), so any importable directory at its root
shadows that name for every consumer: a `tests` package here once hid rola's own `tests.oracle` and `tests.integration`
from pytest in every venv. The root may expose exactly one package, `rola_devtools`."""
from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Layout(unittest.TestCase):
    def test_the_root_exposes_only_rola_devtools(self):
        packages = sorted(p.name for p in ROOT.iterdir() if p.is_dir() and (p / "__init__.py").exists())
        modules = sorted(p.name for p in ROOT.glob("*.py"))
        self.assertEqual((packages, modules), (["rola_devtools"], []))


if __name__ == "__main__":
    unittest.main()
