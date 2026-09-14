"""A provider for the driver's tests: arms with fixed elapsed times, so the protocol and order are checked exactly."""
from __future__ import annotations

import os

from rola_devtools.interleave import Arm


def arms(point: dict) -> dict:
    scale = float(point.get("scale", 1.0))

    def noisy():
        print("a provider printing to stdout must not corrupt the protocol")
        return 1.0 * scale

    built = {
        "fast": lambda: Arm(cell={"pid": os.getpid(), "tokens": point["tokens"]}, call=lambda: 1.0 * scale,
                            instrument="fixed"),
        "slow": lambda: Arm(cell={"pid": os.getpid(), "tokens": point["tokens"]}, call=lambda: 2.0 * scale,
                            instrument="fixed"),
        "noisy": lambda: Arm(cell={"pid": os.getpid()}, call=noisy, instrument="fixed"),
        "other_clock": lambda: Arm(cell={}, call=lambda: 1.0, instrument="another"),
        "broken": lambda: Arm(cell={}, call=lambda: 1 / 0, instrument="fixed"),
        "unbuildable": lambda: (_ for _ in ()).throw(RuntimeError("this binary carries no such kernel")),
    }
    return built
