"""THE DIFF CLIENT: the three comparison rules on planted numbers, and the node they hang on -- a side per executor
over shared cell nodes, the claim (`expect`) it keeps or breaks, the two things a broken claim can be (a build failure
or a recorded one), a cell neither side produced counting as a comparison that did NOT happen, and the raw tensors
staying in the workspace. `python -m unittest tests.test_diff`"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from rola_devtools import config
from rola_devtools.build.cache import Cache
from rola_devtools.build.declare import Env, Graph
from rola_devtools.build.runtime import Runtime
from rola_devtools.build.scheduler import build
from rola_devtools.cells.declare import cells as cell_nodes
from rola_devtools.diff import diff, side
from rola_devtools.diff.strategies import bit_identical, per_slot, support_equal

HERE = Path(__file__).resolve().parent


class Rules(unittest.TestCase):
    """The rules alone, on planted numbers: no build, no cells."""

    def test_bit_identity_admits_no_tolerance_and_names_the_worst_slot(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        self.assertTrue(bit_identical(a, a.clone())["same"])
        moved = a.clone()
        moved[2] += 1e-3
        out = bit_identical(a, moved)
        self.assertEqual((out["same"], out["at"], out["failed"]), (False, [2], 1))
        self.assertFalse(bit_identical(a, a.double())["same"])  #: a dtype change is a difference

    def test_the_per_slot_rule_is_the_smallest_clause_and_the_message_names_what_bound_it(self):
        reference = torch.tensor([1e-4, 1.0, 100.0], dtype=torch.float64)
        clauses = [[1e-2, 1e-6]]
        #: inside every clause: 1% of the slot, or the floor below it
        self.assertTrue(per_slot(reference * 1.005, reference, clauses=clauses)["same"])
        out = per_slot(reference + torch.tensor([0.0, 0.0, 2.0]), reference, clauses=clauses)
        self.assertEqual((out["same"], out["at"]), (False, [2]))
        self.assertEqual(out["bound_by"], "clause (0.01, 1e-06)")
        #: two clauses, and the ABSOLUTE one binds the big slot
        out = per_slot(reference + torch.tensor([0.0, 0.0, 0.5]), reference, clauses=[[1e-2, 1e-6], [0.0, 1e-1]])
        self.assertEqual((out["same"], out["bound_by"]), (False, "clause (0.0, 0.1)"))

    def test_a_multilinear_output_adds_its_envelope_and_refuses_to_be_graded_without_one(self):
        reference = torch.tensor([1.0], dtype=torch.float64)
        envelope = torch.tensor([0.5], dtype=torch.float64)
        out = per_slot(reference + 9e-3, reference, clauses=[[1e-2, 0.0]], rtol=1e-2, envelope=envelope)
        self.assertEqual((out["same"], out["bound_by"]), (False, "the envelope"))
        with self.assertRaises(ValueError):
            per_slot(reference, reference, clauses=[[1e-2, 0.0]], rtol=1e-2)

    def test_support_equality_is_about_which_slots_survive_and_then_about_their_values(self):
        a = torch.tensor([1.0, 0.0, 3.0], dtype=torch.float64)
        self.assertTrue(support_equal(a, a.clone())["same"])
        moved = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        self.assertEqual(support_equal(moved, a)["why"], "the support moved")
        #: the support held, so the values on it are graded and nothing off it is
        out = support_equal(torch.tensor([1.0, 0.0, 3.5], dtype=torch.float64), a, rtol=1e-2)
        self.assertEqual((out["same"], out["at"]), (False, [2]))


class Node(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "config").mkdir()
        (self.root / "config" / "host.json").write_text(json.dumps(
            {"gpu_lock": str(self.root / "gpu.lock"), "lock_dir": str(self.root), "nice": False}))
        self.saved = os.environ.get(config.POINTER)
        os.environ[config.POINTER] = str(self.root / "config")
        config.reload()

    def tearDown(self):
        os.environ.pop(config.POINTER, None) if self.saved is None else os.environ.__setitem__(config.POINTER,
                                                                                               self.saved)
        config.reload()
        self._tmp.cleanup()

    def env(self):
        return Env("fake", sys.executable, str(HERE),
                   {"PYTHONPATH": os.pathsep.join([str(HERE.parent), str(HERE)]),
})

    def go(self, roots):
        with Runtime(HERE) as runtime:
            return {r.label: r for r in build(roots, cache=Cache(self.root / "cache"), runtime=runtime,
                                              log=lambda _l: None)}

    def sides(self, g, right_params, *, left_params=None, executor="fake_sides:flat", names=("corner-anti", "corner-onehot")):
        nodes = cell_nodes(g, names)
        left = side(g, "left", env=self.env(), executor=executor, cells=nodes, params=left_params or {}, holds={})
        right = side(g, "right", env=self.env(), executor=executor, cells=nodes, params=right_params, holds={})
        return left, right

    def test_two_sides_that_agree_hold_the_claim_and_the_raw_stays_in_the_workspace(self):
        g = Graph()
        left, right = self.sides(g, {})
        out = self.go([diff(g, "same", left=left, right=right, strategy="bit-identical")])
        self.assertEqual(out["same"].status, "ran")
        self.assertTrue(out["same"].output["held"])
        self.assertEqual(out["same"].output["compared"], 2)
        #: THE RAW STAYS PUT: the tensors are a file of the side's workspace, and the diff's own output carries
        #: verdicts about them and no numbers from them
        self.assertTrue((Path(out["left"].dir) / "corner-anti.pt").is_file())
        self.assertEqual(sorted(out["same"].output["cells"]["corner-anti"]["quantities"]), ["state", "y"])
        self.assertNotIn("file", json.dumps(out["same"].output["cells"]))

    def test_a_difference_fails_the_build_by_default_and_the_message_names_the_worst_slot(self):
        g = Graph()
        left, right = self.sides(g, {"bump": 0.5, "where": 3})
        out = self.go([diff(g, "moved", left=left, right=right, strategy="bit-identical")])
        self.assertEqual(out["moved"].status, "failed")
        self.assertIn("cells differ under bit-identical", out["moved"].detail)
        self.assertIn('"at": [3]', out["moved"].detail)

    def test_on_difference_record_keeps_the_build_going_and_the_diff_is_the_targets_own_outcome(self):
        g = Graph()
        left, right = self.sides(g, {"bump": 0.5})
        out = self.go([diff(g, "moved", left=left, right=right, strategy="bit-identical", on_difference="record")])
        #: the build does not stop and the target RAN: the broken claim is its own recorded outcome, which is what a
        #: dependent (a store beside it, a group over it) reads -- the same shape as an instrument's failed cell
        self.assertEqual(out["moved"].status, "ran")
        self.assertEqual(out["moved"].output["status"], "failed")
        self.assertFalse(out["moved"].output["held"])
        self.assertEqual(out["moved"].output["differing"], ["corner-anti", "corner-onehot"])

    def test_expecting_a_difference_is_the_non_vacuity_half_and_agreement_breaks_that_claim(self):
        g = Graph()
        left, right = self.sides(g, {"bump": 1.0})
        seen = self.go([diff(g, "mutant", left=left, right=right, strategy="bit-identical", expect="different")])
        self.assertTrue(seen["mutant"].output["held"])

        g2 = Graph()
        left2, right2 = self.sides(g2, {})
        blind = self.go([diff(g2, "mutant", left=left2, right=right2, strategy="bit-identical", expect="different",
                              on_difference="record")])
        self.assertFalse(blind["mutant"].output["held"])
        self.assertIn("the comparison cannot see what it was built to see", blind["mutant"].output["detail"])

    def test_a_cell_a_side_could_not_produce_is_not_a_comparison_that_passed(self):
        g = Graph()
        nodes = cell_nodes(g, ["corner-anti"])
        left = side(g, "left", env=self.env(), executor="fake_sides:flat", cells=nodes, holds={})
        right = side(g, "right", env=self.env(), executor="fake_sides:refuses", cells=nodes, holds={})
        out = self.go([diff(g, "half", left=left, right=right, strategy="bit-identical", on_difference="record")])
        self.assertEqual(out["right"].status, "ran")  #: producing nothing is the CELL's failure, not the side's
        self.assertFalse(out["half"].output["held"])
        self.assertEqual(out["half"].output["unusable"], ["corner-anti"])
        self.assertIn("produced no comparison", out["half"].output["detail"])

    def test_the_per_slot_rule_reaches_the_node_and_a_per_quantity_rule_overrides_it(self):
        g = Graph()
        left, right = self.sides(g, {}, left_params={"bump": 2e-3, "where": 0})
        #: y's slots are 1.0, so a 2e-3 bump is inside a 1% clause and outside a 1e-4 one
        loose = diff(g, "loose", left=left, right=right, strategy="per-slot", params={"clauses": [[1e-2, 0.0]]})
        self.assertTrue(self.go([loose])["loose"].output["held"])

        g2 = Graph()
        left2, right2 = self.sides(g2, {}, left_params={"bump": 2e-3, "where": 0})
        tight = diff(g2, "tight", left=left2, right=right2, strategy="per-slot", on_difference="record",
                     params={"clauses": [[1e-2, 0.0]], "per_quantity": {"y": {"clauses": [[1e-4, 0.0]]}}})
        out = self.go([tight])["tight"].output
        self.assertFalse(out["held"])
        self.assertTrue(out["cells"]["corner-anti"]["quantities"]["state"]["same"])  #: the default rule still holds elsewhere
        self.assertEqual(out["cells"]["corner-anti"]["quantities"]["y"]["bound_by"], "clause (0.0001, 0.0)")


if __name__ == "__main__":
    unittest.main()
