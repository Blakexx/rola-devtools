"""WHAT AN EXECUTOR IS GIVEN: one target's run, as its worker hands it over.

    def compile_kernel(ctx: Context) -> dict:      # the JSON output dependents read and keys hash
        ...

`workspace` is the target's own directory for this run: raw files go there, and a cached target's workspace is kept
with its output. `inputs` are the central cells' records the target takes; `deps` maps each role to what that
dependency produced (its output, its directory, its key and status). `run` is the build's run id.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Dep:
    label: str
    status: str
    key: str | None
    output: object
    dir: str | None


@dataclass(frozen=True)
class Context:
    label: str
    run: str
    workspace: Path
    params: dict
    inputs: tuple[dict, ...]
    deps: dict[str, Dep]

    @classmethod
    def from_json(cls, doc: dict) -> Context:
        return cls(doc["label"], doc["run"], Path(doc["workspace"]), doc["params"], tuple(doc["inputs"]),
                   {role: Dep(**dep) for role, dep in doc["deps"].items()})


__all__ = ["Context", "Dep"]
