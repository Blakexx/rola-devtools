"""A provider for the driver's tests: arms with fixed elapsed times, so the protocol and order are checked exactly."""
from __future__ import annotations

import os

from rola_devtools.interleave import Arm


def arms(point: dict) -> dict:
    scale = float(point.get("scale", 1.0))

    def noisy():
        print("a provider printing to stdout must not corrupt the protocol")
        return 1.0 * scale

    return {
        "fast": Arm(cell={"pid": os.getpid(), "tokens": point["tokens"]}, call=lambda: 1.0 * scale, instrument="fixed"),
        "slow": Arm(cell={"pid": os.getpid(), "tokens": point["tokens"]}, call=lambda: 2.0 * scale, instrument="fixed"),
        "noisy": Arm(cell={"pid": os.getpid()}, call=noisy, instrument="fixed"),
        "other_clock": Arm(cell={}, call=lambda: 1.0, instrument="another"),
        "broken": Arm(cell={}, call=lambda: 1 / 0, instrument="fixed"),
    }
