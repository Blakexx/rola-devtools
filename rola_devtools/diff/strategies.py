"""THE COMPARISON RULES, each a named function of two tensors and its own parameters.

A strategy answers one question about one quantity: are these the same, under this rule? It returns a plain-JSON
verdict -- whether they matched, how many slots were compared, how many failed, the worst one and where it is, and
what bound it -- and never raises on a difference: what a difference MEANS is the diff node's business
(`declare.diff`'s `expect` and `on_difference`), not the rule's.

    bit-identical   every slot equal, byte for byte. The rule for a change that claims to have moved no number: an
                    addressing change, a rename, a host refactor. No tolerance, because a tolerance would let exactly
                    the difference the claim denies through.
    per-slot        THE ORACLE RULE (`tests/oracle/fixtures.py` states it for the batteries, this mirrors it): a slot
                    of size `s` may be off by `max(r*s, a)` under each clause `(r, a)` of its output kind, and by
                    `rtol * envelope` where the output is multilinear; the allowance is the SMALLEST of those terms,
                    so every clause holds and the tightest one binds. A slot whose allowance is zero must be exact,
                    and a slot that is not finite on either side fails.
    support-equal   the same slots are nonzero on both sides, and the values on that shared support are within a
                    tolerance. The rule for a sparse producer, where WHICH slots survive is the claim and the surviving
                    values are ordinary arithmetic.

Every rule reads both sides in fp64: a comparison in the operand's own precision grades the comparison's rounding
along with the kernel's.
"""
from __future__ import annotations

import math

import torch


def _worst(err, allowance, bad, shape):
    """The failing slot with the largest error against its allowance, as JSON: a verdict names the worst one, because
    a count alone cannot be acted on."""
    ratio = torch.where(err == 0, torch.zeros_like(err), err / allowance).nan_to_num(nan=math.inf)
    i = int(torch.argmax(ratio.flatten()))
    return {"at": [int(x) for x in torch.unravel_index(torch.tensor(i), shape)],
            "error": float(err.flatten()[i]), "allowance": float(allowance.flatten()[i]),
            "ratio": float(ratio.flatten()[i]), "failed": int(bad.sum()), "slots": int(err.numel())}


def _shape_verdict(a, b):
    return {"same": False, "why": "shape", "left_shape": list(a.shape), "right_shape": list(b.shape),
            "slots": int(a.numel()), "failed": int(a.numel())}


def bit_identical(a, b, **_params) -> dict:
    """Byte for byte, in the dtype the sides produced -- NOT in fp64, because casting is what would hide a difference
    of the last bit, which is the only kind this rule exists to catch."""
    if a.shape != b.shape or a.dtype != b.dtype:
        out = _shape_verdict(a, b)
        out["why"] = "shape" if a.shape != b.shape else "dtype"
        return {**out, "left_dtype": str(a.dtype), "right_dtype": str(b.dtype)}
    if torch.equal(a, b):
        return {"same": True, "slots": int(a.numel()), "failed": 0}
    err = (a.double() - b.double()).abs()
    bad = err != 0
    return {"same": False, "why": "not bit-identical", **_worst(err, torch.zeros_like(err), bad, a.shape)}


def per_slot(a, b, *, clauses, rtol=None, envelope=None, derived_atol=0.0, **_params) -> dict:
    """`b` IS THE REFERENCE: the allowance is stated in terms of the reference's own slot sizes, so the two sides are
    not interchangeable here the way they are under bit-identity."""
    if a.shape != b.shape:
        return _shape_verdict(a, b)
    a, b = a.double(), b.double()
    s = b.abs()
    terms = [torch.clamp(float(r) * s, min=float(at) + float(derived_atol)) for r, at in clauses]
    if rtol is not None:
        if envelope is None:
            raise ValueError("a multilinear output states an envelope, and this comparison was given none")
        terms.append(float(rtol) * envelope.double())
    if not terms:
        raise ValueError("a per-slot comparison needs at least one clause; none was declared")
    allowance, binding = torch.stack(terms).min(dim=0)
    err = (a - b).abs()
    bad = ~(err <= allowance)
    if not bool(bad.any()):
        return {"same": True, "slots": int(err.numel()), "failed": 0,
                "worst_ratio": float(torch.where(err == 0, torch.zeros_like(err), err / allowance)
                                     .nan_to_num(nan=0.0).max())}
    worst = _worst(err, allowance, bad, a.shape)
    term = int(binding.flatten()[int(torch.argmax(torch.where(err == 0, torch.zeros_like(err), err / allowance)
                                                  .nan_to_num(nan=math.inf).flatten()))])
    worst["bound_by"] = "the envelope" if term == len(clauses) else f"clause {tuple(clauses[term])}"
    return {"same": False, "why": "outside the allowance", **worst}


def support_equal(a, b, *, rtol=0.0, atol=0.0, **_params) -> dict:
    """The SUPPORT is the claim: the same slots are nonzero on both sides. Values are then compared on that shared
    support alone, because a slot one side zeroed has no value to compare."""
    if a.shape != b.shape:
        return _shape_verdict(a, b)
    a, b = a.double(), b.double()
    left, right = a != 0, b != 0
    moved = int((left ^ right).sum())
    if moved:
        i = int(torch.argmax((left ^ right).flatten().to(torch.int8)))
        return {"same": False, "why": "the support moved", "slots": int(a.numel()), "failed": moved,
                "at": [int(x) for x in torch.unravel_index(torch.tensor(i), a.shape)],
                "left": float(a.flatten()[i]), "right": float(b.flatten()[i])}
    allowance = torch.clamp(float(rtol) * b.abs(), min=float(atol))
    err = (a - b).abs() * left
    bad = ~(err <= allowance)
    if not bool(bad.any()):
        return {"same": True, "slots": int(a.numel()), "failed": 0, "support": int(left.sum())}
    return {"same": False, "why": "the support held and a value did not", **_worst(err, allowance, bad, a.shape)}


STRATEGIES = {"bit-identical": bit_identical, "per-slot": per_slot, "support-equal": support_equal}


def compare(strategy: str, a, b, params: dict) -> dict:
    rule = STRATEGIES.get(strategy)
    if rule is None:
        raise ValueError(f"no comparison strategy {strategy!r}; the rules are {sorted(STRATEGIES)}")
    return rule(a, b, **params)


__all__ = ["STRATEGIES", "bit_identical", "compare", "per_slot", "support_equal"]
