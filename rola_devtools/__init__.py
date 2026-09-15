"""rola_devtools: the development tools every RoLA repository shares, standard library only.

- `rola_devtools.mirror`: the public mirror's export (every tracked path declared ships or private, a tripwire over
  private-shaped content), run by each repository's commit gate and its mirror workflow.
- `rola_devtools.cells`: the central cell registry -- every input (carry, layer, QKV) with its draw, seed and regime
  proof, named once for every package -- with bases and derived cells.
- `rola_devtools.build`: the declared build system -- targets declared in files, keyed, cached, held and run.
- `rola_devtools.timing`: the timing system as targets -- interleaved clock-locked sessions, memory passes, null gates.
- `rola_devtools.store`: the store target, a result written through rola-results.
- `rola_devtools.verdict`: whether a candidate's timing is a regression against a reference timed in the same sessions.
- `rola_devtools.process`: a command stopped whole, profiler and measured process included.
- `rola_devtools.config`: the dev config reader, and the machine sections (`host`, `clock`) every repository shares.
- `rola_devtools.locks`: the GPU lock, the host-compute budget and file locks, the clock lock.
"""
