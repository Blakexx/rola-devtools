"""THE INTERLEAVING DRIVER: every library's arms on the cells of one point, timed one call at a time.

A POINT (`rola_devtools.cells`) is a named group of cells by runner and what the group holds equal. A RUNNER is a
library's `module:function`: called with the data a cell builds, it returns a builder per arm it accepts for that cell,
or raises to refuse it. Each builder returns an :class:`Arm`: what the runner built and the call that runs one timed unit.
A comparison names its arms (`ArmSpec`); each runs on every cell the point sends its runner, and a result row -- one arm
on one cell -- keeps the cell's name and what the runner built from it.

The driver knows no library. It starts one WORKER process per runner environment (a python, a directory, an environment;
two builds of one library are two workers), prepares every row, warms each past the floor, and then, for every rep of
every round, calls each row once in a fresh random order. The samples of one rep are adjacent in time, so a drift step
lands on every row alike; the paired statistic is the median of the per-rep ratios to a reference row, and the per-round
differences are what a significance test reads. Timing happens inside the worker, by the arm's own stopwatch, so the
pipe between processes is never in a sample; one comparison uses one stopwatch. `accepts` asks a runner what it makes of
cells without building anything, which is how a suite plans.

    from rola_devtools.cells import Registry
    from rola_devtools.interleave import ArmSpec, interleave
    point = Registry.load(["cells.json", "points.json"]).point("L4096-N4096-dv64")
    result = interleave(point, [ArmSpec("rola", "bench.provider:arms", "prefill_op", python=venv_a),
                                ArmSpec("attention", "attention:arms", "flash")], reference="attention")
"""
from .arm import Arm
from .driver import WARMUP_FLOOR, ArmSpec, accepts, interleave, null_gate

__all__ = ["WARMUP_FLOOR", "Arm", "ArmSpec", "accepts", "interleave", "null_gate"]
