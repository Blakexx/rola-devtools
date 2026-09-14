"""The interleaving driver: every rep calls every row (an arm on a cell) once in a fresh order, samples are the arm's
own, workers are per environment, a runner lists or refuses cells, and what the method forbids is refused."""
from __future__ import annotations

import os
import unittest
from collections import Counter
from pathlib import Path

from rola_devtools.cells import Registry
from rola_devtools.interleave import WARMUP_FLOOR, ArmSpec, accepts, interleave, null_gate

HERE = str(Path(__file__).resolve().parent)
ENV = {"PYTHONPATH": os.pathsep.join([str(Path(HERE).parent), os.environ.get("PYTHONPATH", "")])}


def cell(name, tokens, **params):
    return {"name": name, "data": "fake_provider:tokens", "params": {"tokens": tokens, **params}}


REGISTRY = Registry({c["name"]: c for c in (cell("t64", 64), cell("t8", 8), cell("t16", 16), cell("huge", 4096))}, {})


def point(**runners):
    return REGISTRY.adhoc({runner: names.split(",") for runner, names in runners.items()})


def spec(label, arm, **kw):
    return ArmSpec(label, "fake_provider:arms", arm, cwd=HERE, env=ENV, **kw)


class Interleave(unittest.TestCase):
    def test_every_rep_calls_every_row_once_and_ratios_pair_within_a_rep(self):
        result = interleave(point(fast="t64", slow="t64"), [spec("fast", "fast"), spec("slow", "slow")], rounds=3, reps=5)
        order = result["order"]
        self.assertEqual(len(order), 2 * 3 * 5)
        self.assertTrue(all(sorted(order[i:i + 2]) == ["fast|t64", "slow|t64"] for i in range(0, len(order), 2)))
        self.assertEqual(Counter(order[0::2])["fast|t64"] + Counter(order[0::2])["slow|t64"], 15)
        self.assertGreater(len({tuple(order[i:i + 2]) for i in range(0, len(order), 2)}), 1)
        slow = result["arms"][1]
        self.assertEqual(slow["paired"][0]["ratio_median"], 2.0)
        self.assertEqual(slow["paired"][0]["round_diffs_ms"], [1.0, 1.0, 1.0])
        self.assertEqual((result["arms"][0]["median_ms"], result["instrument"]), (1.0, "fixed"))
        self.assertEqual((slow["cell"], slow["built"]["tokens"]), ("t64", 64))

    def test_an_arm_runs_on_every_cell_the_point_sends_its_runner_and_pairs_within_a_cell(self):
        result = interleave(point(rola="t8,t16", attention="t64"),
                            [spec("master", "slow", runner="rola"), spec("tip", "fast", runner="rola"),
                             spec("attention", "slow")], rounds=1, reps=1, reference="master")
        rows = {row["row"]: row for row in result["arms"]}
        self.assertEqual(list(rows), ["master|t8", "master|t16", "tip|t8", "tip|t16", "attention|t64"])
        self.assertEqual([row["built"]["tokens"] for row in result["arms"]], [8, 16, 8, 16, 64])
        self.assertEqual([p["reference"] for p in rows["tip|t16"]["paired"]], ["master|t16"])
        self.assertEqual(rows["tip|t16"]["paired"][0]["ratio_median"], 0.5)
        self.assertEqual([p["reference"] for p in rows["attention|t64"]["paired"]], ["master|t8", "master|t16"])
        self.assertEqual(rows["master|t8"]["paired"], [])
        one = interleave(point(rola="t8,t16"), [spec("tip", "fast", runner="rola")], rounds=1, reps=1,
                         reference="tip|t8")
        self.assertEqual([p["reference"] for p in one["arms"][1]["paired"]], ["tip|t8"])
        with self.assertRaisesRegex(ValueError, "sends no cell to runner 'attention'"):
            interleave(point(rola="t8"), [spec("attention", "fast")])
        with self.assertRaisesRegex(ValueError, "neither a label nor a row"):
            interleave(point(rola="t8"), [spec("tip", "fast", runner="rola")], reference="nobody")

    def test_one_environment_shares_a_worker_and_a_named_worker_is_its_own_process(self):
        shared = interleave(point(a="t8", b="t8"), [spec("a", "fast"), spec("b", "slow")], rounds=1, reps=1)
        self.assertEqual(shared["arms"][0]["built"]["pid"], shared["arms"][1]["built"]["pid"])
        split = interleave(point(a="t8", b="t8"), [spec("a", "fast"), spec("b", "slow", worker="second")], rounds=1,
                           reps=1)
        self.assertNotEqual(split["arms"][0]["built"]["pid"], split["arms"][1]["built"]["pid"])

    def test_the_null_gate_holds_for_one_arm_in_two_workers(self):
        result = null_gate(spec("fast", "fast"), point(fast="t8"), rounds=2, reps=3)
        self.assertTrue(result["holds"])
        self.assertNotEqual(result["arms"][0]["built"]["pid"], result["arms"][1]["built"]["pid"])

    def test_a_runner_lists_the_arms_it_accepts_and_names_the_cells_it_refuses(self):
        listed = accepts(spec("rola", "fast"), [REGISTRY.cell("t8"), REGISTRY.cell("huge")])
        self.assertIn("slow", listed["t8"]["arms"])
        self.assertIn("carries no arm for 4096 tokens", listed["huge"]["refused"])

    def test_a_runner_printing_to_stdout_does_not_corrupt_the_protocol(self):
        result = interleave(point(noisy="t8", fast="t8"), [spec("noisy", "noisy"), spec("fast", "fast")], rounds=1,
                            reps=1)
        self.assertEqual(result["arms"][0]["ms"], [1.0])

    def test_what_the_method_forbids_is_refused(self):
        p = point(a="t8", b="t8", u="t8", broken="t8")
        with self.assertRaisesRegex(ValueError, "floor"):
            interleave(p, [spec("a", "fast")], warmup=WARMUP_FLOOR - 1)
        with self.assertRaisesRegex(ValueError, "odd"):
            interleave(p, [spec("a", "fast")], reps=4)
        with self.assertRaisesRegex(ValueError, "one stopwatch"):
            interleave(p, [spec("a", "fast"), spec("b", "other_clock")], rounds=1, reps=1)
        with self.assertRaisesRegex(RuntimeError, r"no arm \['missing'\] for cell t8"):
            interleave(p, [spec("a", "missing")], rounds=1, reps=1)
        with self.assertRaisesRegex(RuntimeError, "carries no such kernel"):
            interleave(p, [spec("a", "fast"), spec("u", "unbuildable")], rounds=1, reps=1)
        with self.assertRaisesRegex(RuntimeError, "carries no arm for 4096 tokens"):
            interleave(point(a="huge"), [spec("a", "fast")], rounds=1, reps=1)
        with self.assertRaisesRegex(RuntimeError, "broken: fake_provider:arms refused call"):
            interleave(p, [spec("broken", "broken")], rounds=1, reps=1)
        with self.assertRaisesRegex(ValueError, "unique"):
            interleave(p, [spec("a", "fast"), spec("a", "slow")])


if __name__ == "__main__":
    unittest.main()
