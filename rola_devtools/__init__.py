"""rola_devtools: the development tools every RoLA repository shares, standard library only.

- `rola_devtools.mirror`: the public mirror's export (every tracked path declared ships or private, a tripwire over
  private-shaped content), run by each repository's commit gate and its mirror workflow.
- `rola_devtools.cells`: the central cell registry -- every input (carry, layer, QKV) with its draw and regime proof,
  named once for every package -- and points (cells grouped by runner).
- `rola_devtools.interleave`: the interleaving driver: every runner's arms on the cells of one point, timed one call at
  a time in a fresh random order per rep, across worker processes when the arms cannot share one.
- `rola_devtools.verdict`: whether a candidate's timing is a regression against its baseline.
- `rola_devtools.process`: a command stopped whole, profiler and measured process included.
- `rola_devtools.build`: the build system -- nodes keyed by content, stored results skipped, the rest run in order.
- `rola_devtools.measure`: the measurement service packages register arms, instruments and builds with.
- `rola_devtools.config`: the dev config reader, and the machine sections (`host`, `clock`) every repository shares.
- `rola_devtools.locks`: the GPU lock, the host-compute budget and file locks, the clock lock.
"""
