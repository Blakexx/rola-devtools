"""THE SCHEDULER: a build's targets in dependency order -- keyed, taken from the cache or run, held, reported.

    results = build(roots, cache=Cache(root), runtime=Runtime(dir))

Every target the roots reach runs after its dependencies, one at a time. A target's KEY is sha256 over its SEMANTICS --
its executor and parameters, the digest of the code it declares, and each dependency's key and output digest -- a data
input is a dependency too, so a cell node's record and the digest of the code that drew it reach the key that way -- the
first 24 hex digits. Labels,
environments' paths and run ids never enter it, so two checkouts of one tree reach one key; an output's `local` field
(paths and other machine-local facts a dependent needs) reaches dependents and is left out of the digest.

For each target, in order:

- a GROUP (no executor) is `ok` when every member is, else `blocked`;
- after a BUILD FAILURE, only `always_run` targets still run; the rest are `skipped`;
- a target whose dependency failed, was blocked or skipped is `blocked`, unless it is `always_run` (it then runs and
  reads each dependency's status);
- a CACHED target whose key the cache holds is `cached` (a `verify` function first confirms its output still holds on
  this machine, or it runs again);
- under `dry`, what would run is `pending` (with no key while a dependency is itself pending);
- otherwise the target runs: its resources are held (`holds`), its executor runs in its environment's worker with a
  fresh workspace, and its output is cached when the target caches. An exception is a build failure: `failed`, and the
  build stops scheduling. What a target reports as a failure inside its own output (a timing entry that could not run)
  is its domain's, not the build's.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass

from . import identity, resources
from .cache import Cache
from .declare import Target
from .runtime import Runtime

OK = ("ran", "cached", "ok")


@dataclass
class Result:
    label: str
    status: str
    key: str | None = None
    output: object = None
    dir: str | None = None
    wall_s: float = 0.0
    detail: str = ""


def key(semantics: dict) -> str:
    return hashlib.sha256(json.dumps(semantics, sort_keys=True).encode()).hexdigest()[:24]


def order(roots) -> list[Target]:
    """Every target the roots reach, each after its dependencies."""
    out: list[Target] = []
    state: dict[str, str] = {}

    def visit(target: Target) -> None:
        if state.get(target.label) == "done":
            return
        if state.get(target.label) == "open":
            raise ValueError(f"a dependency cycle through {target.label}")
        state[target.label] = "open"
        for dep in (*target.deps.values(), *target.inputs):
            visit(dep)
        state[target.label] = "done"
        out.append(target)

    for root in roots:
        visit(root)
    return out


def select(roots, *, only=(), skip=()) -> list[Target]:
    """THE ROOTS A BUILD RUNS, pruned by label pattern -- the build system's one selection mechanism, and it knows
    nothing of what a label means. `only`: the reachable targets matching any pattern become the roots, so what they
    need comes along and nothing else does. `skip`: a matching target is dropped, and so is everything that needs it,
    because a target cannot run on a dependency the build was told not to make. Patterns are `fnmatch` globs."""
    reachable = order(roots)
    if only:
        chosen = [t for t in reachable if any(fnmatch.fnmatchcase(t.label, p) for p in only)]
        if not chosen:
            raise ValueError(f"no target of {[r.label for r in roots]} matches {list(only)}")
        reachable = order(chosen)
    if not skip:
        return reachable if only else list(roots)
    dropped = {t.label for t in reachable if any(fnmatch.fnmatchcase(t.label, p) for p in skip)}
    for t in reachable:  # dependency order: a dropped dependency drops its dependents in one pass
        if t.label not in dropped and any(d.label in dropped for d in (*t.deps.values(), *t.inputs)):
            dropped.add(t.label)
    kept = [t for t in reachable if t.label not in dropped]
    needed = {d.label for t in kept for d in (*t.deps.values(), *t.inputs)}
    return [t for t in kept if t.label not in needed]


def _code(target: Target, cwd: str) -> str | None:
    spec = target.code
    if spec is None:
        return None
    if "files" in spec:
        return identity.files(cwd, spec["files"])
    return identity.code(cwd, spec["entry"], tuple(spec.get("roots", (".",))), tuple(spec.get("data", ())))


def _json_digest(value) -> str:
    """The digest of an output as its dependents' keys read it: its `local` field (machine-local facts -- paths, the
    environment an entry runs in) reaches dependents but never a key."""
    if isinstance(value, dict):
        value = {k: v for k, v in value.items() if k != "local"}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def run_id() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + secrets.token_hex(3)


def build(roots, *, cache: Cache, runtime: Runtime, dry: bool = False, force: bool = False,
          log: Callable[[str], None] = print, run: str | None = None) -> list[Result]:
    run = run or run_id()
    results: dict[str, Result] = {}
    stopped = False

    def done(result: Result) -> Result:
        results[result.label] = result
        log(f"{result.status:8s} {result.label}" + (f" [{result.key[:12]}]" if result.key else "")
            + (f" {result.wall_s:.1f}s" if result.status in ("ran", "failed") else "")
            + (f": {result.detail[-300:]}" if result.detail else ""))
        return result

    for target in order(roots):
        deps = {role: results[dep.label] for role, dep in target.deps.items()}
        inputs = [results[dep.label] for dep in target.inputs]
        deps.update({f"input {i}": result for i, result in enumerate(inputs)})
        if target.group:
            ok = all(d.status in OK for d in deps.values())
            pending = any(d.status == "pending" for d in deps.values())
            done(Result(target.label, "pending" if pending and dry else ("ok" if ok else "blocked"),
                        output={role: d.status for role, d in deps.items()}))
            continue
        if stopped and not target.always_run:
            done(Result(target.label, "skipped", detail="the build failed"))
            continue
        waiting = [d.label for d in deps.values() if d.status == "pending"]
        if waiting:
            done(Result(target.label, "pending", detail=f"keyed once {', '.join(waiting)} has run"))
            continue
        unusable = [d.label for d in deps.values() if d.status not in OK]
        if unusable and not target.always_run:
            done(Result(target.label, "blocked", detail=f"no usable result for {', '.join(unusable)}"))
            continue
        env = runtime.env(target)
        semantics = {"executor": target.executor, "params": target.params, "code": _code(target, env.cwd),
                     "deps": {role: {"key": d.key, "output": _json_digest(d.output)} for role, d in sorted(deps.items())
                              if d.status in OK}}
        k = key(semantics)
        if target.cache and not force:
            hit = cache.get(k)
            if hit is not None and (target.verify is None or dry or runtime.verify(target, hit["output"])):
                done(Result(target.label, "cached", k, hit["output"], hit["dir"]))
                continue
        if dry:
            done(Result(target.label, "pending", k))
            continue
        workspace = cache.workspace(run, target.label)
        context = {"label": target.label, "run": run, "workspace": str(workspace), "params": target.params,
                   "settings": target.settings,
                   "inputs": [result.output for result in inputs],
                   "deps": {role: {"label": d.label, "status": d.status, "key": d.key, "output": d.output, "dir": d.dir}
                            for role, d in deps.items() if not role.startswith("input ")}}
        (workspace / "semantics.json").write_text(json.dumps(semantics, indent=1, sort_keys=True))
        started = time.time()
        try:
            with resources.hold(target.holds) as markers:
                output = runtime.run(target, context, markers)
        except Exception as ex:  # noqa: BLE001 -- a build failure: reported, and the build stops scheduling
            stopped = True
            (workspace / "error.txt").write_text(f"{type(ex).__name__}: {ex}\n")
            done(Result(target.label, "failed", k, wall_s=time.time() - started,
                        detail=str(ex).strip().splitlines()[-1] if str(ex).strip() else type(ex).__name__))
            continue
        if target.cache and isinstance(output, dict) and "local" in output:
            stopped = True
            done(Result(target.label, "failed", k, detail="a cached output carries local facts, which no key covers: "
                                                          "declare the target cache=False"))
            continue
        kept = cache.put(k, semantics, output, workspace) if target.cache else str(workspace)
        done(Result(target.label, "ran", k, output, kept, time.time() - started))
    return list(results.values())


def summary(results: list[Result]) -> str:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return ", ".join(f"{n} {status}" for status, n in sorted(counts.items()))


__all__ = ["OK", "Result", "build", "key", "order", "run_id", "summary"]
