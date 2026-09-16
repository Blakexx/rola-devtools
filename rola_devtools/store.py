"""STORING A TARGET'S RESULT in rola-results: a target of its own, run in the build system's environment.

    store(g, "store-L1024", source=session, location="timing/session")

The record's semantics are the source target's (its `semantics.json`), so one measured configuration keeps one record
and every run appends a sample, stamped with the run id; the sample's output is the source's file (`output["file"]`),
with the rest of the source's output kept on the sample as its `summary` (each cell's or member's status), or else the
output itself; its provenance is each checkout the source ran in. rola_results is imported here, in the build system's
environment, and nowhere a kernel runs.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path


def store(g, name: str, *, source, location: str, root: str | None = None, cache: bool = False):
    """`root` is the rola-results backend directory; None is its own records -- a SETTING, not a parameter, because
    the same record written to two checkouts of the results repository is the same record and keying the path would
    make it two. `cache`: skip storing a result already stored (for an analysis the build caches, whose key the store's
    key follows); a measurement stores every run."""
    #: WHERE THE SOURCE RAN is provenance, so it rides as a setting: a cached source cannot carry a `local` field, and
    #: its checkout is known at declaration -- its environment's directory -- without keying a path
    ran_in = None if source.env is None else {"label": source.env.label, "cwd": source.env.cwd}
    return g.node(name, executor="rola_devtools.store:put", deps={"source": source},
                  params={"location": location}, settings={"root": root, "source": ran_in}, cache=cache)


def put(ctx) -> dict:
    from rola_results import ROOT, Store, checkout

    source = ctx.deps["source"]
    semantics = json.loads((Path(source.dir) / "semantics.json").read_text())
    output = {k: v for k, v in source.output.items() if k != "local"}
    envs = source.output.get("local", {}).get("envs", {}) if isinstance(source.output, dict) else {}
    checkouts = {owner: checkout(env["cwd"]) for owner, env in sorted(envs.items())}
    ran_in = ctx.settings.get("source")
    if not checkouts and ran_in:
        checkouts = {ran_in["label"]: checkout(ran_in["cwd"])}
    provenance = {"run": ctx.run, "checkouts": checkouts}
    store = Store(ctx.params["location"], ctx.settings["root"] or ROOT)
    if "file" in output:
        copy = ctx.workspace / output["file"]
        shutil.copyfile(Path(source.dir) / output["file"], copy)
        #: the rest of the target's output (each cell's or member's status, a gate's finding) rides on the sample
        summary = {k: v for k, v in output.items() if k != "file"}
        sample = store.put(semantics, output_file=copy, provenance=provenance, run=ctx.run, summary=summary)
    else:
        sample = store.put(semantics, output=output, provenance=provenance, run=ctx.run)
    return {"location": ctx.params["location"], "sample": sample["n"], "run": ctx.run}


__all__ = ["put", "store"]
