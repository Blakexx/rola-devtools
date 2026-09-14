"""THE INTERLEAVING DRIVER: every library's arms at one comparison point, timed one call at a time.

A COMPARISON is a point -- what is held equal across the libraries (the tokens, the value width, the dtype, the
capacity) -- and the matching rule that makes it fair, named. Each library realizes the point as its OWN native cell
through a PROVIDER, `module:function`, called with the point and returning its arms by name (:class:`Arm`): the cell it
built and the call that runs one timed unit. Arms are never asked to share a cell; each result row keeps the point and
its cell.

The driver knows no library. It starts one WORKER process per provider environment (a python, a directory, an
environment; two builds of one library are two workers), prepares every arm, warms each past the floor, and then, for
every rep of every round, calls each arm once in a fresh random order. The samples of one rep are adjacent in time, so a
drift step lands on every arm alike; the paired statistic is the median of the per-rep ratios to a reference arm, and
the per-round differences are what a significance test reads. Timing happens inside the worker, by the arm's own
stopwatch, so the pipe between processes is never in a sample; one comparison uses one stopwatch.

    from rola_devtools.interleave import ArmSpec, interleave
    result = interleave({"tokens": 4096, "d_v": 64}, [ArmSpec("rola", "bench.arms:carry", "prefill", python=venv_a),
                                                      ArmSpec("attention", "bench.arms:attention", "flash")],
                        matching="capacity at N = L", reference="attention")
"""
from .arm import Arm
from .driver import WARMUP_FLOOR, ArmSpec, interleave, null_gate

__all__ = ["WARMUP_FLOOR", "Arm", "ArmSpec", "interleave", "null_gate"]
