"""The verdict: the paired test, the persistence rule and the three gates together. Every check plants the violation its
gate exists to catch and requires the gate to fire; a gate that cannot be shown failing is treated as absent."""
from __future__ import annotations

import unittest

from rola_devtools.verdict import (
    ALPHA,
    MIN_ROUNDS,
    _exact_two_sided_p,
    _normal_two_sided_p,
    _signed_rank_statistic,
    classify,
    classify_flags,
    paired_verdict,
)


class PairedTest(unittest.TestCase):
    def test_a_planted_shift_is_detected_and_an_unshifted_pair_is_cleared(self):
        shifted = paired_verdict([0.05] * MIN_ROUNDS)
        self.assertEqual(shifted["verdict"], "b_slower")
        self.assertTrue(shifted["p"] <= ALPHA and shifted["exact"])
        self.assertEqual(paired_verdict([0.05, -0.05] * (MIN_ROUNDS // 2))["verdict"], "same")
        self.assertEqual(paired_verdict([-0.05] * MIN_ROUNDS)["verdict"], "b_faster")

    def test_the_round_floor_is_derived_from_alpha_and_is_binding(self):
        self.assertLessEqual(paired_verdict([0.05] * MIN_ROUNDS)["p"], ALPHA)
        short = [0.05] * (MIN_ROUNDS - 1)
        self.assertGreater(_exact_two_sided_p(*_signed_rank_statistic(short)), ALPHA)
        self.assertEqual(paired_verdict(short)["verdict"], "insufficient_data")

    def test_a_shift_smaller_than_the_noise_is_not_called(self):
        diffs = [0.01, -0.4, 0.35, -0.30, 0.28, -0.25, 0.22, -0.18, 0.30]
        self.assertEqual(paired_verdict(diffs)["verdict"], "same")

    def test_the_exact_and_approximate_tails_agree_where_they_meet(self):
        diffs = [0.3, -0.1, 0.25, 0.4, -0.05, 0.2, 0.35, 0.15, -0.2, 0.45, 0.05, 0.5, -0.15, 0.28, 0.33]
        w, ranks = _signed_rank_statistic(diffs)
        self.assertAlmostEqual(paired_verdict(diffs)["p"], _normal_two_sided_p(w, ranks), delta=0.02)


class Persistence(unittest.TestCase):
    def test_one_unlucky_run_is_not_a_regression_and_two_are(self):
        self.assertEqual(classify_flags([False, False, False, True]), "no_regression")
        self.assertEqual(classify_flags([False, False, True, True]), "regression")

    def test_a_healed_run_is_suspicious_and_a_short_series_undecided(self):
        self.assertEqual(classify_flags([True, True, True, False, False]), "suspicious")
        self.assertEqual(classify_flags([True, True]), "insufficient_data")


BASELINE = [[1.00, 1.01, 0.99, 1.00]] * 5
SLOW = [1.60, 1.61, 1.59, 1.60]


class ThreeGates(unittest.TestCase):
    def test_all_three_gates_must_fire(self):
        got = classify(BASELINE, [SLOW, SLOW], paired_diffs=[0.6] * MIN_ROUNDS)
        self.assertEqual(got["verdict"], "regression")
        self.assertGreater(got["median_ms"], got["limit_ms"])

    def test_one_slow_run_is_flagged_not_confirmed(self):
        self.assertEqual(classify(BASELINE, [SLOW], paired_diffs=[0.6] * MIN_ROUNDS)["verdict"], "flagged_not_confirmed")

    def test_an_effect_without_significance_is_flagged_not_confirmed(self):
        got = classify(BASELINE, [SLOW, SLOW], paired_diffs=[0.6, -0.6] * (MIN_ROUNDS // 2))
        self.assertEqual(got["verdict"], "flagged_not_confirmed")

    def test_an_unchanged_kernel_passes_and_a_short_baseline_is_undecided(self):
        self.assertEqual(classify(BASELINE, [[1.00, 1.00, 1.01, 0.99]], paired_diffs=[0.0] * 8)["verdict"],
                         "no_regression")
        self.assertEqual(classify(BASELINE[:2], [SLOW])["verdict"], "insufficient_data")

    def test_a_spread_of_zero_is_floored_at_the_stopwatch_resolution(self):
        ticks = [[1.0, 1.0, 1.0625]] * 5
        got = classify(ticks, [[1.0625, 1.0625, 1.0]] * 2, paired_diffs=[0.0625] * MIN_ROUNDS)
        self.assertEqual((got["verdict"], got["resolution_ms"], got["floored_at_resolution"]), ("no_regression", 0.0625, True))
        slow = classify(ticks, [[1.5, 1.5, 1.5625]] * 2, paired_diffs=[0.5] * MIN_ROUNDS)
        self.assertEqual(slow["verdict"], "regression")


if __name__ == "__main__":
    unittest.main()
