"""The one type a provider's builders return: an arm."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Arm:
    """One thing a library times at a comparison point.

    `cell` is the library's own JSON description of what it built for the point. `call` runs one timed unit and returns
    its elapsed milliseconds, measured by the stopwatch `instrument` names, the device synchronized on both sides.
    """

    cell: dict
    call: Callable[[], float]
    instrument: str
