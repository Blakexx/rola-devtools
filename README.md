# rola-devtools

The development tools every RoLA repository shares. Standard library only, so every venv, CI job and container can
carry it. It is a development dependency: nothing a user installs imports it.

| Module | What it is |
|---|---|
| `rola_devtools.mirror` | the public mirror's export, run by each repository's commit gate and its mirror workflow |
| `rola_devtools.cells` | the central cell registry: every input a RoLA measurement or test runs on (carry, layer, QKV), its draw and the regime it proves, named once; and points (cells grouped by runner, and what they hold equal) |
| `rola_devtools.interleave` | the interleaving driver: every runner's arms on the cells of one point, timed one call at a time |
| `rola_devtools.verdict` | whether a candidate's timing is a regression against its baseline: effect size, paired significance, persistence |
| `rola_devtools.build` | the build system: nodes keyed by their content and their dependencies' outputs; what is stored is skipped, the rest runs in dependency order |
| `rola_devtools.measure` | the measurement service: packages register units (arms, instruments, builds, clock readers), the service composes them with the central cells into build nodes and sessions and runs them, one worker per instance |
| `rola_devtools.process` | `subprocess.run` for a command that starts processes of its own: a timeout or an interrupt stops the whole tree |
| `rola_devtools.config` | the dev config reader: one directory of JSON files a section, every key declared; the machine's `host` and `clock` sections every repository shares |
| `rola_devtools.locks` | the machine's locks: the GPU lock (exclusive or shared), the host-compute budget and named file locks, the SM clock lock |

## The public mirror

Each RoLA project lives in two places: a private development repository, where every branch is pushed, and a public
repository holding what has been published. A push to the development repository's default branch publishes.

**What ships.** `.github/mirror/declarations.json` in the repository declares every tracked path. `ships` and `private`
list path prefixes, and a path takes the longest entry it falls under, so a directory can ship while one file inside it
stays private. A tracked path under no entry refuses the export, and `python -m rola_devtools.mirror --check` refuses it
at commit time: a new file never ships, and never silently stays behind, until someone declares it. The list names what
ships rather than what hides, because a list of hidden things fails open on the file nobody anticipated.

**The tripwire.** The export is HEAD's shipping files, written by `git archive` (committed bytes only). It is refused if
any file is an agent or scratch file at any depth (`CLAUDE.md`, `.claude/`, `SCRATCHPAD.md`) or holds a private-shaped
string: a home directory, the private scratch repository's name, a personal email address, a cloud project or bucket
identifier, or a credential (GitHub, Hugging Face, Anthropic and AWS token shapes, a private key block). `allow` names
shapes a repository's published files may carry, each with its reason; a credential can never be allowed. Every pattern
is spelled so that the tool's own text does not match it.

**The snapshot.** Each repository's `.github/workflows/mirror.yml` installs this package at a pinned commit, runs the
export on the pushed commit, commits the export on top of the public repository's `main` naming the development commit
it came from, and pushes with the development repository's `MIRROR_KEY` secret (the private half of a write deploy key
on the public repository).

```bash
python -m rola_devtools.mirror /tmp/export   # the export of HEAD, with any refusal
python -m rola_devtools.mirror --check       # every tracked path declared
python -m rola_devtools.mirror --public      # the public repository
```

## The interleaving driver

A **comparison** is a point (what is held equal across the libraries: tokens, value width, dtype, capacity) and the
matching rule that makes it fair, named. Each library realizes the point as its own native cell through a **provider**,
`module:function`, which takes the point and returns a builder per arm name; the worker builds only the arms a
comparison asks for, so an arm the library cannot build refuses by name without blocking the others. An `Arm` is the
cell the library built and a call that runs one timed unit, returning its elapsed milliseconds by the stopwatch the arm
names, the device synchronized. Arms are never asked to share a cell; every result row keeps the point and its own cell.

The driver knows no library. It starts one worker process per provider environment (a python, a directory, an
environment: two builds of one library are two workers), prepares every arm, warms each past the floor of 10 launches,
and then calls each arm once per rep in a fresh random order. The samples of one rep are adjacent in time, so a drift
step lands on every arm alike. The paired statistic is the median of the per-rep ratios to a reference arm; the
per-round differences are what a significance test reads. Timing happens inside the worker, so the pipe between
processes is never in a sample, and one comparison uses one stopwatch. `null_gate` runs one arm in two workers: a
comparison across workers is trusted once its per-rep ratios put one inside their interquartile range.

```python
from rola_devtools.interleave import ArmSpec, interleave

result = interleave(
    {"tokens": 4096, "d_v": 64},
    [ArmSpec("rola", "bench.arms:carry", "prefill", python="/path/to/venv-a/bin/python", cwd="/path/to/rola-a"),
     ArmSpec("attention", "bench.arms:attention", "flash")],
    matching="capacity at N = L", reference="attention", hold=gpu_and_clock_lock)
```

## The verdict

`rola_devtools.verdict.classify(baseline, runs, paired_diffs=...)` calls a candidate's latest run a regression only when
three gates fire: its median is above the baseline's own median plus three sigmas of the baseline's spread (sigma from
the interquartile range, never a flat percentage); an exact Wilcoxon signed-rank test over that session's per-round
differences (candidate minus baseline, alpha 0.01, at least 8 rounds, the floor the alpha itself sets) says it is slower;
and the violation persists, a trailing run of at least two over the baseline's sessions followed by the candidate's runs.
Anything less is reported as what it is: `flagged_not_confirmed`, `suspicious`, `insufficient_data` or `no_regression`.
It takes plain samples; which stored sessions are the baseline is a query over the store (`python -m rola_results
verdict`).

## The central cell registry

A cell is an input, named once for every package: `rola_devtools/cells/carry.json` (a routed recurrence's routing
amplitudes, gain, values and the state it enters with, including the state's backing), `layer.json` (a layer's hidden
states and values) and `qkv.json` (attention's queries, keys and values). A record holds the input and never how a
package runs it -- a kernel's launch shape, RoLA's routing widths or gain, an attention backend are an arm's -- so two
arms on one cell read the same tensors, and when the registry changes, a package that no longer fits it is what changes.

```python
from rola_devtools.cells import cell, carry

drawn = carry.realize(cell("flagship-alt-k4"))   # bf16 read/write levels, gain, v; the entry state of a carried cell
```

Every carry cell but the two degenerate ones declares where in the routing distribution it sits on six axes (density,
coherence, read/write correlation, mass, tail, support; `rola_devtools.cells.regimes`), and `realize` proves the draw is
there before returning it. A cell's seed comes from its name, so a failure reproduces from the name alone. Loading and
checking the registry needs only the standard library; realizing a cell needs torch.

## The build system and the measurement service

`rola_devtools.build` is generic. A node is an id, its semantics (a JSON object: everything its result depends on
besides its dependencies), a store location and its dependencies by role; its key is sha256 over the semantics and each
dependency's key and output digest, so ids, labels and paths never move a record. `run(nodes, store, execute)` skips
what the store holds, runs the rest in dependency order through the caller's executor, stores outputs, refusals and
failures, and blocks the dependents of a refusal or a failure. A node marked local (a build) counts only while its output
is still on the machine.

`rola_devtools.measure` is the service the packages register with. An owner's registry returns `Registration`s of four
unit kinds -- `Arm` (timed on a cell), `Instrument` (exclusive), `Build`, `ClockReader` -- and each unit declares which
cells it accepts and what its result depends on. The service describes every instance in its own environment (one
worker an instance for the whole run), composes each selected unit with each cell it accepts into a node, gives every
arm a memory node, and interleaves arms in sessions: every instance's arms of a session on its cells, set up behind a
barrier, warmed, then called in a fresh random order each rep, with each member paired to the reference instance's arm
on its cell. Setups, executes and sessions run under the GPU lock with the host's clock proven; posts write results
through a handle after the device is released.

```bash
python -m rola_devtools.measure plan benchmarks.registry:registry --cells flagship-dense --units carry.phases
python -m rola_devtools.measure run  benchmarks.registry:registry --session carry_forward@flagship-dense,flagship-alt-k4
```

## The dev config and the machine's locks

Every value that depends on the machine lives in one directory of JSON files, one a section: `~/.config/rola/`, or the
directory `ROLA_DEV_CONFIG` names (a container, a test). A schema declares each section's keys with a kind, a default and
a meaning; a key a file holds that its section does not declare is refused, so a typo fails instead of silently taking
a default. `rola_devtools.config.MACHINE` declares the sections every repository shares -- `host` (lock paths, the
compute budget, niceness) and `clock` (the SM clock the harness locks) -- and `machine("host.gpu_lock")` reads one value.
A repository with sections of its own (rola's toolchain, store and workspace) reads its whole schema with
`read(schema, owns_directory=True)`, which also refuses a file no section declares.

The locks read only those sections, so one host and its containers share one of each:

```python
from rola_devtools.locks.gpu import gpu_lock
from rola_devtools.locks import host

with gpu_lock():                      # a measurement: the device to itself (mode="shared" for a correctness run)
    ...
with host.acquire(4, label="tidy"):   # up to 4 host-compute slots, at least one
    ...
```

`python -m rola_devtools.locks.host [--slots N] [--exclusive] -- <command...>` runs a command under the budget. A nested
acquire in the same process or a child is a no-op, so an outer tool that locks can start an inner tool that locks.
`rola_devtools.locks.clock.engage(read_ghz)` locks the clock the host declares, proves it with the device's own reading,
and unlocks on every exit.

## Tests and the gate

```bash
python -m unittest discover -s tests -t .
```

The commit gate (`.githooks/pre-commit` -> `.pre-commit-config.yaml`) runs ruff, its blank-line family, the tests and
the mirror's declaration check, from the venv the checkout names (`git config rola.venv`).

## Publishing

Developed in the private `rola-devtools-dev`, published to the public `rola-devtools` by its own mirror.

## License

Apache-2.0.
