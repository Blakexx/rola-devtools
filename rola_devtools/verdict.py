"""THE VERDICT: whether a candidate's timing is a regression against its baseline, from samples alone.

Three gates, and a regression needs all three:

1. EFFECT SIZE -- the candidate's median above :func:`threshold_ms`, which is the baseline's own median plus three sigmas
   of its own spread (sigma from the interquartile range). Never a flat percentage.
2. SIGNIFICANCE -- :func:`paired_verdict`: an exact Wilcoxon signed-rank test over the per-round differences, candidate
   minus baseline, that one interleaved session produced. Paired, because the interleaving paid for the pairing; exact,
   because the round counts are single digits.
3. PERSISTENCE -- :func:`classify_flags` over the baseline's own violations followed by the candidate's runs in order:
   a regression is a trailing run of at least two violations. One unlucky run is not one.

Anything short of all three is reported as what it is (`flagged_not_confirmed`, `suspicious`, `insufficient_data`,
`no_regression`), never as a pass. The functions take plain numbers; which samples count as a baseline is a query over
stored records, and belongs to the store (`rola_results`).
"""
from __future__ import annotations

import math
import statistics
from itertools import product

#: The paired test's alpha, stated once.
ALPHA = 0.01
#: DERIVED FROM ALPHA, not chosen: the smallest two-sided exact p reachable with n non-zero differences is 2 / 2**n (every
#: sign the same way), so below ceil(log2(2 / ALPHA)) rounds no outcome can be significant and the test is undefined.
MIN_ROUNDS = math.ceil(math.log2(2 / ALPHA))
#: Above this many non-zero differences the exact enumeration's 2**n terms give way to the normal approximation, which is
#: accurate there to well past the third decimal of p.
EXACT_MAX_N = 20
#: The one free choice of the effect-size gate: a one-sided three-sigma alarm, about 0.1 % false alarms per cell per run.
ALARM_SIGMA = 3.0
#: Q3 - Q1 of a normal sample, in sigmas: what turns a measured interquartile range into a sigma.
IQR_PER_SIGMA = 1.349
#: The fewest baseline sessions a threshold is derived from.
MIN_BASELINE = 3


def iqr(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[int(0.75 * len(ordered))] - ordered[int(0.25 * len(ordered))]


def threshold_ms(median_ms: float, iqr_ms: float) -> float:
    """The latency above which a candidate is over the line: derived from the baseline's own median and spread."""
    return median_ms + ALARM_SIGMA * iqr_ms / IQR_PER_SIGMA


def _signed_rank_statistic(diffs: list[float]) -> tuple[float, list[float]]:
    """W+ (the sum of the ranks of the positive differences) and the ranks. Zero differences are discarded, Wilcoxon's
    own treatment; ties among the absolute differences take mid-ranks."""
    nonzero = [d for d in diffs if d != 0]
    order = sorted(range(len(nonzero)), key=lambda i: abs(nonzero[i]))
    ranks = [0.0] * len(nonzero)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and abs(nonzero[order[j + 1]]) == abs(nonzero[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return sum(r for r, d in zip(ranks, nonzero, strict=True) if d > 0), ranks


def _exact_two_sided_p(w_plus: float, ranks: list[float]) -> float:
    """P(|W+ - mean| >= |observed - mean|) with each sign a fair coin, enumerated over the actual ranks (ties included)."""
    mean = sum(ranks) / 2
    observed = abs(w_plus - mean)
    hits = sum(abs(sum(r for r, take in zip(ranks, signs, strict=True) if take) - mean) >= observed - 1e-12
               for signs in product((0, 1), repeat=len(ranks)))
    return hits / 2 ** len(ranks)


def _normal_two_sided_p(w_plus: float, ranks: list[float]) -> float:
    """The large-sample approximation, with the tie correction the mid-ranks require."""
    n = len(ranks)
    counts: dict[float, int] = {}
    for rank in ranks:
        counts[rank] = counts.get(rank, 0) + 1
    var = n * (n + 1) * (2 * n + 1) / 24 - sum(t ** 3 - t for t in counts.values()) / 48
    if var <= 0:
        return 1.0
    return math.erfc(abs((w_plus - n * (n + 1) / 4) / math.sqrt(var)) / math.sqrt(2))


def paired_verdict(diffs: list[float], *, alpha: float = ALPHA) -> dict:
    """Whether per-round differences (candidate minus baseline) are distinguishable from zero.

    `verdict` is `insufficient_data` (fewer than MIN_ROUNDS: the design cannot decide), `same`, `b_slower` or `b_faster`,
    with the p-value and the n it was computed on.
    """
    diffs = list(diffs)
    if len(diffs) < MIN_ROUNDS:
        return {"verdict": "insufficient_data", "p": None, "n": len(diffs),
                "reason": f"{len(diffs)} rounds < MIN_ROUNDS={MIN_ROUNDS}"}
    w_plus, ranks = _signed_rank_statistic(diffs)
    if not ranks:
        return {"verdict": "same", "p": 1.0, "n": 0, "reason": "every paired difference was exactly zero"}
    exact = len(ranks) <= EXACT_MAX_N
    p = _exact_two_sided_p(w_plus, ranks) if exact else _normal_two_sided_p(w_plus, ranks)
    verdict = "same" if p > alpha else ("b_slower" if sum(diffs) > 0 else "b_faster")
    return {"verdict": verdict, "p": p, "n": len(ranks), "w_plus": w_plus, "exact": exact}


def classify_flags(violations: list[bool]) -> str:
    """The persistence rule over violations in order, the last the run being judged: `regression` for a trailing run of
    at least two, `suspicious` for a run of at least three that has stopped, `insufficient_data` below three points,
    otherwise `no_regression`."""
    flags = [bool(v) for v in violations]
    if len(flags) < 3:
        return "insufficient_data"
    trailing = 0
    for flag in reversed(flags):
        if not flag:
            break
        trailing += 1
    if trailing >= 2:
        return "regression"
    run = best = 0
    for flag in flags:
        run = run + 1 if flag else 0
        best = max(best, run)
    return "suspicious" if best >= 3 else "no_regression"


def classify(baseline: list[list[float]], runs: list[list[float]], *, paired_diffs: list[float] | None = None) -> dict:
    """THE FLAGGING RULE over samples: `baseline` is the baseline's sessions (each a list of samples, oldest first), from
    which the threshold is derived; `runs` are the candidate's sessions in order, the last the one judged; `paired_diffs`
    are that last session's per-round differences, candidate minus baseline."""
    if len(baseline) < MIN_BASELINE or not runs:
        return {"verdict": "insufficient_data", "n_baseline": len(baseline), "n_runs": len(runs),
                "reason": f"fewer than {MIN_BASELINE} baseline sessions, or no run to judge"}
    medians = [statistics.median(samples) for samples in baseline]
    base_median, base_iqr = statistics.median(medians), iqr(medians)
    limit = threshold_ms(base_median, base_iqr)
    run_medians = [statistics.median(samples) for samples in runs]
    got = run_medians[-1]
    persistence = classify_flags([m > limit for m in medians] + [m > limit for m in run_medians])
    significance = (paired_verdict(paired_diffs) if paired_diffs is not None
                    else {"verdict": "insufficient_data", "p": None, "reason": "no paired rounds for the judged run"})
    if got > limit:
        verdict = ("regression" if significance["verdict"] == "b_slower" and persistence == "regression"
                   else "flagged_not_confirmed")
    else:
        verdict = "suspicious" if persistence == "suspicious" else "no_regression"
    return {"verdict": verdict, "median_ms": got, "limit_ms": limit, "baseline_median_ms": base_median,
            "baseline_iqr_ms": base_iqr, "n_baseline": len(baseline), "n_runs": len(runs), "persistence": persistence,
            "significance": significance}
