"""THE DIFF TARGETS, declared: helpers a declaration file calls on its graph (`rola_devtools.build.declare.Graph`)."""
from __future__ import annotations

EXECUTORS = "rola_devtools.diff.executors:"
#: what a diff may be told to do about a difference, and what it may be told to expect
OUTCOMES = ("fail", "record")
EXPECTATIONS = ("same", "different")


def side(g, name: str, *, env, executor: str, cells, params: dict | None = None, deps: dict | None = None,
         code: dict | None = None, holds: dict | None = None, binds: str | None = None):
    """ONE SIDE of a diff: `executor` (a `module:function` taking a cell record and returning named tensors) run in
    `env` over every cell TARGET of `cells`, its tensors written into this node's workspace.

    `binds` NAMES THE LIBRARY THIS SIDE IS SUPPOSED TO BE RUNNING, and the executor proves it did. This is not
    ceremony: an editable install resolves its package through the install's path entry, which names the CANONICAL
    checkout, so a side run out of another worktree can import the very tree it is being compared against -- measured,
    not hypothetical. Both sides then agree perfectly, having compared one tree with itself. The side records where its
    library actually resolved and refuses to run when that is outside its own environment's directory, and the
    comparison refuses two sides that resolved to the same file.

    The side is an ordinary cached target keyed like any other -- on its code, its environment's checkout and each
    cell's record -- so a side that has already run for these cells and this code is not run again, and the two sides
    of a diff between one checkout and itself are one job.
    """
    #: A SIDE HOLDS THE WHOLE HOST BUDGET UNLESS TOLD OTHERWISE. A reference in fp64 over a cell is CPU work that
    #: takes every core it is given, and the budget is cores minus the headroom that keeps the shell alive; a side
    #: that held nothing took every core and froze the host (2026-09-15). A kernel side declares `{"gpu": "all"}`.
    return g.node(name, executor=EXECUTORS + "produce", env=env, deps=deps or {}, inputs=cells,
                  params={"executor": executor, "params": params or {}, "binds": binds}, code=code,
                  holds=holds if holds is not None else {"host_cpu": "all"})


def diff(g, name: str, *, left, right, strategy: str, params: dict | None = None, expect: str = "same",
         on_difference: str = "fail", minimum: int = 1, env=None, cache: bool = True):
    """THE DIFF of two sides over their cells, under one named `strategy` (`strategies.py`).

    `expect` is the claim: "same" is a gate on an invariant (a refactor that changed no number, a kernel inside its
    oracle's band), "different" is the NON-VACUITY half -- a mutant that must be seen, which is what a planted error
    asserts. `on_difference` picks what a violated claim is: a BUILD FAILURE that stops the build ("fail"), or the
    target's own recorded outcome that dependents read and the build survives ("record"), the same two-way choice
    every other target makes.

    `minimum` IS THE COUNT: how many quantities this comparison must actually have compared for its verdict to mean
    anything. A gate that cannot run has to say so rather than report a verdict over whatever survived, so falling
    short of it is a refusal and never a pass -- whatever `expect` and `on_difference` say.

    The comparison runs in `env` -- by default the left side's, because reading the tensors takes the same torch that
    wrote them, and the build system's own environment has none.
    """
    if expect not in EXPECTATIONS or on_difference not in OUTCOMES:
        raise ValueError(f"{name}: expect is one of {EXPECTATIONS} and on_difference one of {OUTCOMES}; "
                         f"got {expect!r} and {on_difference!r}")
    if minimum < 1:
        raise ValueError(f"{name}: a comparison that may compare nothing is not a comparison; minimum={minimum}")
    return g.node(name, executor=EXECUTORS + "compare", env=env if env is not None else left.env,
                  deps={"left": left, "right": right}, cache=cache,
                  params={"strategy": strategy, "params": params or {}, "expect": expect,
                          "on_difference": on_difference, "minimum": minimum})


__all__ = ["EXPECTATIONS", "OUTCOMES", "diff", "side"]
