"""The interleaving driver: every rep calls every arm once in a fresh order, samples are the arm's own, workers are
per environment, and what the method forbids is refused."""
from __future__ import annotations

import os
import unittest
from collections import Counter
from pathlib import Path

from rola_devtools.interleave import WARMUP_FLOOR, ArmSpec, interleave, null_gate

HERE = str(Path(__file__).resolve().parent)
ENV = {"PYTHONPATH": os.pathsep.join([str(Path(HERE).parent), os.environ.get("PYTHONPATH", "")])}


def spec(label, arm, **kw):
    return ArmSpec(label, "fake_provider:arms", arm, cwd=HERE, env=ENV, **kw)


class Interleave(unittest.TestCase):
    def test_every_rep_calls_every_arm_once_and_ratios_pair_within_a_rep(self):
        result = interleave({"tokens": 64}, [spec("fast", "fast"), spec("slow", "slow")], matching="test", rounds=3,
                            reps=5)
        order = result["order"]
        self.assertEqual(len(order), 2 * 3 * 5)
        self.assertTrue(all(sorted(order[i:i + 2]) == ["fast", "slow"] for i in range(0, len(order), 2)))
        self.assertEqual(Counter(order[0::2])["fast"] + Counter(order[0::2])["slow"], 15)
        self.assertGreater(len({tuple(order[i:i + 2]) for i in range(0, len(order), 2)}), 1)
        slow = result["arms"][1]
        self.assertEqual(slow["paired"]["ratio_median"], 2.0)
        self.assertEqual(slow["paired"]["round_diffs_ms"], [1.0, 1.0, 1.0])
        self.assertEqual((result["arms"][0]["median_ms"], result["instrument"]), (1.0, "fixed"))
        self.assertEqual(slow["cell"]["tokens"], 64)

    def test_one_environment_shares_a_worker_and_a_named_worker_is_its_own_process(self):
        shared = interleave({"tokens": 8}, [spec("a", "fast"), spec("b", "slow")], matching="test", rounds=1, reps=1)
        self.assertEqual(shared["arms"][0]["cell"]["pid"], shared["arms"][1]["cell"]["pid"])
        split = interleave({"tokens": 8}, [spec("a", "fast"), spec("b", "slow", worker="second")], matching="test",
                           rounds=1, reps=1)
        self.assertNotEqual(split["arms"][0]["cell"]["pid"], split["arms"][1]["cell"]["pid"])

    def test_the_null_gate_holds_for_one_arm_in_two_workers(self):
        result = null_gate(spec("fast", "fast"), {"tokens": 8}, rounds=2, reps=3)
        self.assertTrue(result["holds"])
        self.assertNotEqual(result["arms"][0]["cell"]["pid"], result["arms"][1]["cell"]["pid"])

    def test_a_provider_printing_to_stdout_does_not_corrupt_the_protocol(self):
        result = interleave({"tokens": 8}, [spec("noisy", "noisy"), spec("fast", "fast")], matching="test", rounds=1,
                            reps=1)
        self.assertEqual(result["arms"][0]["ms"], [1.0])

    def test_what_the_method_forbids_is_refused(self):
        with self.assertRaisesRegex(ValueError, "floor"):
            interleave({"tokens": 8}, [spec("a", "fast")], matching="test", warmup=WARMUP_FLOOR - 1)
        with self.assertRaisesRegex(ValueError, "odd"):
            interleave({"tokens": 8}, [spec("a", "fast")], matching="test", reps=4)
        with self.assertRaisesRegex(ValueError, "one stopwatch"):
            interleave({"tokens": 8}, [spec("a", "fast"), spec("b", "other_clock")], matching="test", rounds=1, reps=1)
        with self.assertRaisesRegex(RuntimeError, r"no arm \['missing'\]"):
            interleave({"tokens": 8}, [spec("a", "missing")], matching="test", rounds=1, reps=1)
        with self.assertRaisesRegex(RuntimeError, "carries no such kernel"):
            interleave({"tokens": 8}, [spec("a", "fast"), spec("u", "unbuildable")], matching="test", rounds=1, reps=1)
        with self.assertRaisesRegex(RuntimeError, "broken: fake_provider:arms refused call"):
            interleave({"tokens": 8}, [spec("broken", "broken")], matching="test", rounds=1, reps=1)
        with self.assertRaisesRegex(ValueError, "unique"):
            interleave({"tokens": 8}, [spec("a", "fast"), spec("a", "slow")], matching="test")


if __name__ == "__main__":
    unittest.main()
