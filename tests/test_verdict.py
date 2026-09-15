"""The verdict: the paired test, the persistence rule, one session's paired quantities and the three gates together. Every
check plants the violation its gate exists to catch and requires the gate to fire; a gate that cannot be shown failing is
treated as absent."""
from __future__ import annotations

import unittest

from rola_devtools.verdict import (
    ALPHA,
    MIN_ROUNDS,
    _exact_two_sided_p,
    _normal_two_sided_p,
    _signed_rank_statistic,
    classify,
    paired_verdict,
    persistence,
    session,
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
    def test_one_unlucky_session_is_not_a_regression_and_two_are(self):
        self.assertEqual(persistence([False, False, False, True]), "no_regression")
        self.assertEqual(persistence([False, True, True]), "regression")
        self.assertEqual(persistence([True, True]), "regression")

    def test_a_healed_run_is_suspicious_and_one_session_undecided(self):
        self.assertEqual(persistence([True, True, False]), "suspicious")
        self.assertEqual(persistence([True]), "insufficient_data")


def rounds(ms: float, n: int = MIN_ROUNDS, jitter: float = 0.01) -> list[list[float]]:
    """`n` rounds of three reps around `ms`, each round's reps spread by `jitter` and the rounds drifting together, as
    one session's host drift moves both arms."""
    return [[ms * (1 + 0.05 * (i % 3)) * f for f in (1 - jitter, 1, 1 + jitter)] for i in range(n)]


def paired(candidate_ms: float, reference_ms: float = 1.0, **kw) -> dict:
    return session(rounds(candidate_ms, **kw), rounds(reference_ms, **kw))


class OneSession(unittest.TestCase):
    def test_drift_both_arms_share_cancels_in_the_ratio(self):
        got = paired(1.0)
        self.assertEqual((got["ratio"], got["ratio_iqr"], got["violation"]), (1.0, 0.0, False))
        self.assertEqual(got["diffs_ms"], [0.0] * MIN_ROUNDS)

    def test_a_slower_candidate_violates_its_own_limit(self):
        got = paired(1.1)
        self.assertAlmostEqual(got["ratio"], 1.1)
        self.assertTrue(got["violation"])

    def test_a_spread_of_zero_is_floored_at_the_stopwatch_resolution(self):
        ticks = [[1.0, 1.0, 1.0625]] * MIN_ROUNDS
        got = session([[1.0625, 1.0625, 1.0]] * MIN_ROUNDS, ticks)
        self.assertEqual((got["ratio"], got["resolution"], got["floored_at_resolution"], got["violation"]),
                         (1.0625, 0.0625, True, False))
        self.assertTrue(session([[1.5, 1.5, 1.5625]] * MIN_ROUNDS, ticks)["violation"])

    def test_arms_of_different_rounds_or_an_unresolved_reference_are_refused(self):
        with self.assertRaises(ValueError):
            session(rounds(1.0, n=8), rounds(1.0, n=7))
        with self.assertRaises(ValueError):
            session(rounds(1.0), [[0.0, 0.0, 0.0]] * MIN_ROUNDS)


class ThreeGates(unittest.TestCase):
    def test_all_three_gates_must_fire(self):
        got = classify([paired(1.0), paired(1.6), paired(1.6)])
        self.assertEqual(got["verdict"], "regression")
        self.assertGreater(got["ratio"], got["limit"])
        self.assertEqual(got["significance"]["verdict"], "b_slower")

    def test_one_slow_session_is_flagged_not_confirmed(self):
        self.assertEqual(classify([paired(1.0), paired(1.6)])["verdict"], "flagged_not_confirmed")

    def test_an_effect_without_significance_is_flagged_not_confirmed(self):
        #: slower by a fifth on six rounds and faster by half on two: the median ratio is far over its limit, but the two
        #: large negative differences leave the paired signs insignificant
        reps = (0, 0.0001, 0.0002)
        mixed = session([[1.2 + e for e in reps]] * 6 + [[0.5 + e for e in reps]] * 2, [[1.0 + e for e in reps]] * 8)
        self.assertTrue(mixed["violation"])
        self.assertEqual(classify([mixed, mixed])["verdict"], "flagged_not_confirmed")

    def test_an_unchanged_kernel_passes_and_too_few_rounds_are_undecided(self):
        self.assertEqual(classify([paired(1.0), paired(1.0)])["verdict"], "no_regression")
        self.assertEqual(classify([paired(1.6, n=MIN_ROUNDS - 1)] * 2)["verdict"], "insufficient_data")

    def test_a_faster_candidate_is_no_regression(self):
        self.assertEqual(classify([paired(0.7), paired(0.7)])["verdict"], "no_regression")


if __name__ == "__main__":
    unittest.main()
