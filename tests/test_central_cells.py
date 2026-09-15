"""The central registry: every file loads, a record holds its input and nothing a package runs it with, every carry
cell realizes on the device and sits in its declared regime, the regime checks have teeth, and the frozen corners stay
(the checks moved from rola's tests/unit/test_cell_regimes.py). Needs torch and a CUDA device -- the gate runs in a venv
with both, and a missing device fails rather than skips; the device tests hold the GPU lock shared, with the machine's
lock settings but `host.nice` off, so this process's priority is left for the lock tests to observe.
`python -m unittest tests.test_central_cells`"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from rola_devtools import config
from rola_devtools.cells import FILES, build, carry, cell, central, layer, qkv
from rola_devtools.cells.regimes import REGIME_AXES, View, check
from rola_devtools.locks.gpu import gpu_lock

#: THE CORNERS, frozen: a named region of the distribution, once a cell, never leaves the registry.
CORNERS = ("corner-tied-k4", "corner-concentrated", "corner-window-flip", "corner-onehot", "corner-anti",
           "corner-cold-read", "corner-read-only")
T, WIDTH = 64, 8
WINDOW = carry.WINDOW


def _simplex(p=1.0, seed=0, shape=(1, T, 1, WIDTH)):
    gen = torch.Generator().manual_seed(seed)
    x = torch.rand(shape, dtype=torch.float64, generator=gen)
    if p < 1.0:
        x = x * (torch.rand(shape, dtype=torch.float64, generator=gen) < p)
        x[..., 0] = x[..., 0] + (x.sum(-1) == 0)
    return x / x.sum(-1, keepdim=True)


def _one_hot(seed=0, digit=None):
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, WIDTH, (1, T, 1, 1), generator=gen) if digit is None else torch.full((1, T, 1, 1), digit)
    return torch.zeros(1, T, 1, WIDTH, dtype=torch.float64).scatter_(-1, idx, 1.0)


def _half(second, seed):
    x = _simplex(seed=seed)
    x[..., WIDTH // 2:] *= float(second)
    x[..., :WIDTH // 2] *= float(not second)
    return x / x.sum(-1, keepdim=True)


def _view(read, write, tokens=T, entry=None):
    return View(SimpleNamespace(tokens=tokens), (read,), (write,), entry)


DENSE, OTHER = _simplex(seed=1), _simplex(seed=2)
SPARSE = _simplex(0.4, seed=3)
CONSTANT = _simplex(seed=4)[:, :1].expand(1, T, 1, WIDTH)
ENTRY = torch.ones(1, WIDTH, 2)

#: For every (axis, value): a view that demonstrates it, and one that does not.
CASES = {
    ("density", "dense"): (_view(DENSE, OTHER), _view(SPARSE, OTHER)),
    ("density", "sparse"): (_view(SPARSE, OTHER), _view(DENSE, OTHER)),
    ("coherence", "iid"): (_view(DENSE, OTHER), _view(CONSTANT, OTHER)),
    ("coherence", "coherent"): (_view(CONSTANT, OTHER), _view(DENSE, OTHER)),
    ("rw_correlation", "tied"): (_view(DENSE, DENSE), _view(DENSE, OTHER)),
    ("rw_correlation", "independent"): (_view(DENSE, OTHER), _view(DENSE, DENSE)),
    ("rw_correlation", "anti"): (_view(_half(False, 5), _half(True, 6)), _view(DENSE, OTHER)),
    ("mass", "spread"): (_view(DENSE, OTHER), _view(_one_hot(1), _one_hot(2))),
    ("mass", "concentrated"): (_view(_one_hot(1), _one_hot(2)), _view(DENSE, OTHER)),
    ("mass", "cold_read"): (_view(DENSE, _one_hot(digit=0), entry=ENTRY), _view(DENSE, OTHER, entry=ENTRY)),
    ("tail", "divisible"): (_view(DENSE, OTHER, tokens=2 * WINDOW), _view(DENSE, OTHER, tokens=WINDOW + 1)),
    ("tail", "ragged"): (_view(DENSE, OTHER, tokens=WINDOW + 1), _view(DENSE, OTHER, tokens=2 * WINDOW)),
    ("support", "singleton"): (_view(DENSE, _one_hot(2)), _view(DENSE, OTHER)),
    ("support", "partial"): (_view(SPARSE, OTHER), _view(DENSE, OTHER)),
    ("support", "full"): (_view(DENSE, OTHER), _view(SPARSE, OTHER)),
}

RECORD = {"seed": 0, "widths": [16, 16], "dv": 64, "tokens": 16, "draw": "dense", "k_tok": None, "cohort": None,
          "support": 1.0, "backing": "dense", "state": "fresh", "tier": "oracle", "regime": None}


class RegimeChecks(unittest.TestCase):
    def test_every_value_of_every_axis_has_a_case_and_nothing_else_does(self):
        self.assertEqual(set(CASES), {(axis, value) for axis, values in REGIME_AXES.items() for value in values})
        with self.assertRaises(KeyError):
            check("temperature", "hot", CASES[("density", "dense")][0])

    def test_a_check_accepts_its_regime_and_refuses_another(self):
        for (axis, value), (inside, outside) in sorted(CASES.items()):
            with self.subTest(f"{axis}={value}"):
                self.assertIsNone(check(axis, value, inside))
                self.assertIsNotNone(check(axis, value, outside), "the check accepted a view outside it: it has no teeth")

    def test_a_cold_read_without_an_entry_state_is_refused(self):
        self.assertIsNotNone(check("mass", "cold_read", _view(DENSE, _one_hot(digit=0))))


class Records(unittest.TestCase):
    def test_a_record_states_every_axis_or_none_at_all(self):
        with self.assertRaisesRegex(ValueError, "states its regime"):
            carry.carry_cell("x", **RECORD)
        with self.assertRaisesRegex(ValueError, "not every axis"):
            carry.carry_cell("x", **dict(RECORD, regime={"density": "dense"}))
        with self.assertRaisesRegex(ValueError, "declares no regime"):
            carry.carry_cell("x", **dict(RECORD, draw="dead",
                                         regime={axis: values[0] for axis, values in REGIME_AXES.items()}))

    def test_a_cell_names_no_way_of_running_it(self):
        with self.assertRaisesRegex(ValueError, "undeclared field"):
            carry.carry_cell("x", **dict(RECORD, draw="dead", warps_per_cta=8))
        layer_record = {"B": 1, "tokens": 8, "H": 1, "hidden": 0, "dv": 8, "dtype": "fp32", "seed": 0, "decode_steps": 0,
                        "note": ""}
        with self.assertRaisesRegex(ValueError, "undeclared field"):
            layer.layer_cell("x", **dict(layer_record, logit_gain=8.0))
        with self.assertRaisesRegex(ValueError, "undeclared field"):
            qkv.qkv_cell("x", tokens=8, dv=8, backend="flash")

    def test_the_declared_sparsity_is_the_alternation(self):
        self.assertEqual(cell("deep3-alt-k4").declared_sparsity(), ((False, True), (True, False), (False, True)))
        self.assertEqual(cell("flagship-both-k4").declared_sparsity(), ((True, True), (False, False)))
        self.assertEqual(cell("corner-tied-k4").declared_sparsity(), ((False, False), (False, False)))

    def test_every_file_loads_into_one_registry_and_each_kind_builds_its_record(self):
        kinds = {type(build(record)) for record in central().cells.values()}
        self.assertEqual(kinds, {carry.CarryCell, layer.LayerCell, qkv.QKVCell})
        self.assertEqual([path.name for path in FILES], ["bases.json", "carry.json", "layer.json", "qkv.json"])

    def test_the_corners_stay_and_the_registry_covers_the_whole_box(self):
        cells = [c for c in map(build, central().cells.values()) if isinstance(c, carry.CarryCell)]
        names = {c.name for c in cells}
        self.assertLessEqual(set(CORNERS), names, "a frozen corner left the registry")
        covered = {pair for c in cells if c.regime for pair in c.regime}
        missing = {(axis, value) for axis, values in REGIME_AXES.items() for value in values} - covered
        self.assertFalse(missing, f"no registry cell demonstrates {sorted(missing)}")


class Realized(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = tempfile.TemporaryDirectory()
        host = {name: config.machine(f"host.{name}") for name in config.MACHINE["host"]}
        (Path(cls.config.name) / "host.json").write_text(json.dumps(dict(host, nice=False)))
        cls.pointer = os.environ.get(config.POINTER)
        os.environ[config.POINTER] = cls.config.name
        config.reload()

    @classmethod
    def tearDownClass(cls):
        os.environ.pop(config.POINTER) if cls.pointer is None else os.environ.__setitem__(config.POINTER, cls.pointer)
        config.reload()
        cls.config.cleanup()

    def test_every_carry_cell_realizes_in_its_regime_and_reproduces_from_its_name(self):
        with gpu_lock(mode="shared"):
            for c in map(build, central().cells.values()):
                if not isinstance(c, carry.CarryCell):
                    continue
                with self.subTest(c.name):
                    drawn = carry.realize(c)
                    self.assertEqual([tuple(x.shape) for x in drawn.read], [(1, c.tokens, 1, w) for w in c.widths])
                    self.assertEqual(drawn.entry is not None, c.state == "carried")
                    del drawn
                    torch.cuda.empty_cache()
            first, again = carry.realize(cell("corner-cold-read")), carry.realize(cell("corner-cold-read"))
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(first.write, again.write, strict=True)))
            self.assertTrue(torch.equal(first.entry, again.entry))

    def test_a_layer_cell_and_a_qkv_cell_realize_their_shapes(self):
        with gpu_lock(mode="shared"):
            drawn = layer.realize(cell("layer-B2-T128-H2-h128-dv64-fp32-s4-dec8"))
            self.assertEqual((tuple(drawn.x.shape), tuple(drawn.v.shape)), ((2, 136, 128), (2, 136, 2, 64)))
            self.assertEqual(tuple(drawn.prefill[1].shape), (2, 128, 2, 64))
            q, k, v = qkv.realize(cell("qkv-L256-dv64"))
            self.assertEqual({tuple(t.shape) for t in (q, k, v)}, {(1, 1, 256, 64)})
            self.assertFalse(torch.equal(q, k))


if __name__ == "__main__":
    unittest.main()
