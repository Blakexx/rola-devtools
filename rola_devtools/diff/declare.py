"""THE DIFF TARGETS, declared: helpers a declaration file calls on its graph (`rola_devtools.build.declare.Graph`)."""
from __future__ import annotations

EXECUTORS = "rola_devtools.diff.executors:"
#: what a diff may be told to do about a difference, and what it may be told to expect
OUTCOMES = ("fail", "record")
EXPECTATIONS = ("same", "different")


def side(g, name: str, *, env, executor: str, cells, params: dict | None = None, deps: dict | None = None,
         code: dict | None = None, holds: dict | None = None):
    """ONE SIDE of a diff: `executor` (a `module:function` taking a cell record and returning named tensors) run in
    `env` over every cell TARGET of `cells`, its tensors written into this node's workspace.

    The side is an ordinary cached target keyed like any other -- on its code, its environment's checkout and each
    cell's record -- so a side that has already run for these cells and this code is not run again, and the two sides
    of a diff between one checkout and itself are one job.
    """
    return g.node(name, executor=EXECUTORS + "produce", env=env, deps=deps or {}, inputs=cells,
                  params={"executor": executor, "params": params or {}}, code=code,
                  holds=holds if holds is not None else {"gpu": "all"})


def diff(g, name: str, *, left, right, strategy: str, params: dict | None = None, expect: str = "same",
         on_difference: str = "fail", env=None, cache: bool = True):
    """THE DIFF of two sides over their cells, under one named `strategy` (`strategies.py`).

    `expect` is the claim: "same" is a gate on an invariant (a refactor that changed no number, a kernel inside its
    oracle's band), "different" is the NON-VACUITY half -- a mutant that must be seen, which is what a planted error
    asserts. `on_difference` picks what a violated claim is: a BUILD FAILURE that stops the build ("fail"), or the
    target's own recorded outcome that dependents read and the build survives ("record"), the same two-way choice
    every other target makes.

    The comparison runs in `env` -- by default the left side's, because reading the tensors takes the same torch that
    wrote them, and the build system's own environment has none.
    """
    if expect not in EXPECTATIONS or on_difference not in OUTCOMES:
        raise ValueError(f"{name}: expect is one of {EXPECTATIONS} and on_difference one of {OUTCOMES}; "
                         f"got {expect!r} and {on_difference!r}")
    return g.node(name, executor=EXECUTORS + "compare", env=env if env is not None else left.env,
                  deps={"left": left, "right": right}, cache=cache,
                  params={"strategy": strategy, "params": params or {}, "expect": expect,
                          "on_difference": on_difference})


__all__ = ["EXPECTATIONS", "OUTCOMES", "diff", "side"]
