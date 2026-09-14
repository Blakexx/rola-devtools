"""A data provider and a runner for the driver's tests: arms with fixed elapsed times, so the protocol and order are
checked exactly."""
from __future__ import annotations

import os
from dataclasses import dataclass

from rola_devtools.interleave import Arm


@dataclass(frozen=True)
class Tokens:
    name: str
    tokens: int
    scale: float = 1.0


def tokens(name: str, **params) -> Tokens:
    return Tokens(name, **params)


def arms(data: Tokens) -> dict:
    if not isinstance(data, Tokens):
        raise TypeError(f"this runner takes Tokens, got {type(data).__name__}")
    if data.tokens > 1024:
        raise ValueError(f"{data.name}: this binary carries no arm for {data.tokens} tokens")
    scale = data.scale

    def noisy():
        print("a runner printing to stdout must not corrupt the protocol")
        return 1.0 * scale

    def cell():
        return {"pid": os.getpid(), "tokens": data.tokens}

    return {
        "fast": lambda: Arm(cell=cell(), call=lambda: 1.0 * scale, instrument="fixed"),
        "slow": lambda: Arm(cell=cell(), call=lambda: 2.0 * scale, instrument="fixed"),
        "noisy": lambda: Arm(cell=cell(), call=noisy, instrument="fixed"),
        "other_clock": lambda: Arm(cell={}, call=lambda: 1.0, instrument="another"),
        "broken": lambda: Arm(cell={}, call=lambda: 1 / 0, instrument="fixed"),
        "unbuildable": lambda: (_ for _ in ()).throw(RuntimeError("this binary carries no such kernel")),
    }
