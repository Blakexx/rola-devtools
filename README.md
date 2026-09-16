# rola-devtools

The development tools every RoLA repository shares. Standard library only, so every venv, CI job and container can
carry it. It is a development dependency: nothing a user installs imports it.

| Module | What it is |
|---|---|
| `rola_devtools.mirror` | the public mirror's export, run by each repository's commit gate and its mirror workflow |
| `rola_devtools.cells` | the central cell registry: every input a RoLA measurement or test runs on (carry, layer, QKV), its draw, its seed and the regime it proves, named once; bases and the cells derived from them; each cell declarable as a build NODE whose output is its record |
| `rola_devtools.build` | the declared build system: targets declared by composable functions in files, keyed by what they read, cached or run in dependency order, holding the machine's resources, each in its own environment's worker |
| `rola_devtools.timing` | the timing system, as targets: a server of entry workers, registrations of timed callables on cells, interleaved clock-locked sessions, memory passes, null gates |
| `rola_devtools.diff` | the diff targets: two executors run over the same cells in the environments they name, compared under one rule (bit-identity, the per-slot clause rule, support equality), with a claim the build can fail on -- a difference expected or refused |
| `rola_devtools.store` | the store target: a result written through rola-results as a run-stamped sample of its source's record |
| `rola_devtools.verdict` | whether a candidate's timing is a regression against a reference timed in the same sessions: effect size, paired significance, persistence |
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

## The declared build system

A DECLARATION FILE is plain Python, loaded by path, whose functions declare TARGETS on a graph. A target is an executor
(`module:function`) that runs in an environment (`Env`: a python, a directory, variables; none is the build system's
own), the targets it reads by role, the central cells it takes, its parameters, the resources it holds while it runs,
whether it caches, whether it runs after a failure (`always_run`), a function that confirms a cached result still holds
on this machine (`verify`), and the code its result depends on. A group is a target with no executor. Labels are scoped
(`g.scoped("tip")`), so one function declares one checkout's targets under any label, and a root composes declaration
files from several checkouts:

```python
def root(g, target):
    rola = load(Path(target) / "declare.py")
    tip = rola["declare"](g.scoped("tip"), rola["checkout"](target, python=..., label="tip"), cells=["flagship-dense"])
    return {"all": g.group("all", [tip["binary"], *tip["instruments"].values()])}
```

A target's KEY is sha256 over its executor, parameters, declared code digest, and each dependency's key and output
digest -- a DATA INPUT is a dependency too, so a cell node's record and the digest of the code that drew it reach the key
that way; labels, paths and run ids never enter it. The
scheduler takes the targets in dependency order: a cached target whose key the build cache holds (`host.build_cache`,
wipeable) is skipped, the rest run with their resources held -- `{"gpu": "all"}`, `{"gpu": 1}`, `{"host_cpu": 8}`,
`{"clock": 1}`, counting semaphores over the machine's own locks, so processes outside the build are excluded too. An
exception is a BUILD failure: the build stops scheduling and only `always_run` targets still run. What a target records
as failed inside its own output (a timing entry whose kernel is not built) is its domain's, and the build goes on.

```bash
python -m rola_devtools.build plan declare.py:all
python -m rola_devtools.build run  declare.py:all --arg cells=flagship-dense,flagship-alt-k4
```

## The timing system

Timing is a client of the build system (`rola_devtools.timing`). `start_timing_server` starts a pool of entry workers,
one per checkout environment; `register_timing` is one target registering a checkout's timed callable on every cell it
takes, as ENTRIES; `measure_timing` is a session over the entries of any registrations: every entry set up behind a
barrier, warmed past a floor of 10 calls, then called one at a time in a fresh random order each rep, with the entry's
untimed reset before every call, under the GPU and clock locks with the clock read through a registered reader before
and after. The session keeps every sample in the order taken, with its round, rep and position; which entry is the
reference is chosen when the samples are read. An entry that cannot set up is recorded as that member's failure; two
stopwatches or a clock off the lock fail the build. `measure_memory` takes each entry alone, and `measure_null_gate`
times one registration's entries against copies of themselves in second workers, finding a worker's bias.
`rola_devtools.store.store` writes a target's result through rola-results as a run-stamped sample of the record its
source's semantics key.

## The diff

"The same function in two checkouts" and "the kernel against its fp64 reference" are the same question asked twice, so
`rola_devtools.diff` asks it once. `side` is one side of a comparison: an executor run in the environment it names over
every cell TARGET it takes, writing its tensors into its own node's workspace. `diff` takes two sides and one named
STRATEGY -- `bit-identical` (a change that claims to have moved no number), `per-slot` (the oracle's clause rule: a slot
of size `s` may be off by `max(r*s, a)` under each `(r, a)` clause of its output kind, and by `rtol * envelope` where
the output is multilinear, the smallest term binding), `support-equal` (which slots survive is the claim, then their
values) -- and states what it EXPECTS: `same` is a gate on an invariant, `different` is the non-vacuity half, where a
mutant that goes unseen is the failure. `on_difference` picks what a broken claim is, the same two-way choice every
other target makes: a build failure, or the target's own recorded outcome.

The raw never leaves the workspace. A side's tensors live in the build cache, which is wipeable and swept; what the
diff node outputs, and what a store target beside it would file, is the DIFF -- per cell and quantity the worst slot,
where it is, what bound it, and how many slots were compared. A cell either side could not produce is not a comparison
that passed: it is counted as unusable and the claim does not hold, because a gate that goes green on a cell neither
side ran measured nothing.

## The verdict

Timing is comparable only within the session that interleaved it, so the verdict reads within-session quantities.
`rola_devtools.verdict.session(candidate, reference)` takes one session's two arms, each its samples by round, and gives
per round each arm's median, their ratio and their difference. `classify(sessions)` calls the last of a candidate's
sessions against one reference a regression only when three gates fire: the session's median ratio is above one plus
three sigmas of its per-round ratios' own spread (sigma from the interquartile range, never below what the stopwatch
resolves, never a flat percentage); an exact Wilcoxon signed-rank test over its per-round differences (alpha 0.01, at
least 8 rounds, the floor the alpha itself sets) says the candidate is slower; and the violation persists, a trailing run
of at least two sessions over their own limits. Anything less is reported as what it is: `flagged_not_confirmed`,
`suspicious`, `insufficient_data` or `no_regression`. It takes plain samples; which stored sessions and which reference
are read is a query over the store (`python -m rola_results verdict --reference LABEL`).

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
there before returning it. Every cell states the seed its data is drawn from, so a failure reproduces from the record.
A base (`bases.json`) is a fragment, never a cell; a derived cell merges the bases its `from` lists in order, and `vary`
with a name pattern makes one cell per value. Loading and checking the registry needs only the standard library;
realizing a cell needs torch.

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
