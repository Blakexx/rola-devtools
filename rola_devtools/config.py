"""THE DEV CONFIG READER: every value that depends on the machine, read from one directory of JSON files.

The directory is `~/.config/rola/`, or the directory `ROLA_DEV_CONFIG` names (a container, a test). It holds one file a
section. A SCHEMA declares each section's keys with their type, default and meaning: a key a file holds that its
section does not declare is refused, so a typo fails loudly instead of silently taking a default, and a missing file or
key takes the declared default.

`MACHINE` is the part every RoLA repository shares -- the host's locks and budget (`host`) and its clock (`clock`) --
and `machine(key)` reads it; the locks read nothing else. A repository with sections of its own reads its whole schema
with `read(schema, owns_directory=True)`, which also refuses a file no section declares (rola's `tools/dev_config.py`).

A setting is never read from the process environment; `POINTER` is the one variable read here, and it names a
directory, never a value.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

POINTER = "ROLA_DEV_CONFIG"


@dataclass(frozen=True)
class Key:
    kind: str  #: path, str, int, float, bool or argv; a trailing ? admits null
    default: Any
    doc: str


MACHINE: dict[str, dict[str, Key]] = {
    "host": {
        "nvcc_threads": Key("int", 2, "nvcc -t for each compile job"),
        "build_jobs": Key("int?", None, "ninja jobs a build asks for; null derives it from nvcc_threads and memory"),
        "budget_slots": Key("int?", None, "the machine-wide compute slots every build and census compile shares; "
                            "null derives cores - 2 bounded by memory (rola_devtools.locks.host)"),
        "lock_dir": Key("path", "/tmp", "where the host budget's slot locks live; host and containers must share it"),
        "gpu_lock": Key("path", "/tmp/rola_gpu.lock", "the GPU lock file; host and containers must share it"),
        "gpu_shared_slots": Key("int", 2, "correctness runs that may hold the GPU lock shared at once"),
        "nice": Key("bool", True, "locked processes lower their own CPU and IO priority"),
        "locks_trace": Key("bool", False, "print every lock acquire and release to stderr"),
        "scratch": Key("path", "/tmp/rola", "the tools' scratch root; never read back as a record"),
        "build_cache": Key("path", "~/.cache/rola/build", "the build system's cache of completed targets and run "
                           "workspaces (rola_devtools.build); wipeable"),
        "tools_dir": Key("path", "~/.local/share/rola/tools", "where rola's `tools/dev.py init` installs pinned tools"),
        "windows_system32": Key("path?", None, "WSL only: the Windows System32 directory as mounted in Linux"),
        "wsl_lib": Key("path?", None, "WSL only: the Windows driver's Linux libraries (libdxcore, libcuda); the dev "
                       "container mounts their directory's parent read-only (the loader opens the driver store beside "
                       "lib/), without which CUDA fails to initialize inside it"),
    },
    "clock": {
        "ghz": Key("float?", None, "the SM clock the harness locks, in GHz; null means this host runs unlocked"),
        "lock": Key("argv?", None, "the command that locks the clock"),
        "unlock": Key("argv?", None, "the command that releases it"),
    },
}


def directory() -> Path:
    pointer = os.environ.get(POINTER)
    return Path(pointer).expanduser() if pointer else Path.home() / ".config" / "rola"


def _check(section: str, name: str, key: Key, value: Any) -> Any:
    kind, optional = key.kind.rstrip("?"), key.kind.endswith("?")
    where = f"{directory() / (section + '.json')}: {name}"
    if value is None:
        if optional:
            return None
        raise SystemExit(f"{where} may not be null ({key.doc})")
    ok = {"path": isinstance(value, str), "str": isinstance(value, str), "int": isinstance(value, int)
          and not isinstance(value, bool), "float": isinstance(value, (int, float)) and not isinstance(value, bool),
          "bool": isinstance(value, bool), "argv": isinstance(value, list) and all(isinstance(x, str) for x in value)}
    if not ok[kind]:
        raise SystemExit(f"{where} must be {kind}, got {value!r} ({key.doc})")
    return str(Path(value).expanduser()) if kind == "path" else value


def read(schema: dict[str, dict[str, Key]], *, owns_directory: bool) -> dict[str, dict[str, tuple[Any, str]]]:
    """Every section's every key as `(value, source)`: the file it was read from, or "default". `owns_directory`: the
    schema declares every file the directory may hold, so an undeclared one is refused."""
    root = directory()
    if owns_directory and root.exists():
        stray = sorted(p.name for p in root.glob("*.json") if p.stem not in schema)
        if stray:
            raise SystemExit(f"{root}: undeclared config file(s) {stray}; declared: {sorted(schema)}")
    out: dict[str, dict[str, tuple[Any, str]]] = {}
    for section, keys in schema.items():
        path = root / f"{section}.json"
        doc = json.loads(path.read_text()) if path.exists() else {}
        unknown = sorted(set(doc) - set(keys))
        if unknown:
            raise SystemExit(f"{path}: undeclared key(s) {unknown}; declared: {sorted(keys)}")
        out[section] = {name: (_check(section, name, key, doc[name]), str(path)) if name in doc
                        else (_check(section, name, key, key.default), "default") for name, key in keys.items()}
    return out


@cache
def _machine() -> dict[str, dict[str, tuple[Any, str]]]:
    return read(MACHINE, owns_directory=False)


def machine(dotted: str) -> Any:
    """One value of the shared machine sections (`host.gpu_lock`, `clock.ghz`)."""
    section, name = dotted.split(".", 1)
    return _machine()[section][name][0]


def reload() -> None:
    """Forget what was read: a process that repoints `ROLA_DEV_CONFIG` reads the new directory next."""
    _machine.cache_clear()


__all__ = ["MACHINE", "POINTER", "Key", "directory", "machine", "read", "reload"]
