"""The one type a provider's builders return: an arm."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Arm:
    """One thing a runner times on one cell.

    `cell` is the runner's own JSON description of what it built from the cell's data. `call` runs one timed unit and
    returns its elapsed milliseconds, measured by the stopwatch `instrument` names, the device synchronized on both sides.
    `outside_allocator`, when the arm holds device memory its framework's caching allocator does not see (a state mapped
    through the driver's virtual memory API), returns those bytes now; a memory measurement adds them to the allocator's
    peak, which cannot count them.
    """

    cell: dict
    call: Callable[[], float]
    instrument: str
    outside_allocator: Callable[[], int] | None = None
