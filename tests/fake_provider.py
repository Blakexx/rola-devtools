"""A data provider for the tests: a cell of `tokens` tokens and a scale, drawn from nothing."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tokens:
    name: str
    tokens: int
    scale: float = 1.0


def tokens(name: str, **params) -> Tokens:
    return Tokens(name, **params)
