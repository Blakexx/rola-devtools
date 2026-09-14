"""rola_devtools: the development tools every RoLA repository shares, standard library only.

- `rola_devtools.mirror`: the public mirror's export (every tracked path declared ships or private, a tripwire over
  private-shaped content), run by each repository's commit gate and its mirror workflow.
- `rola_devtools.cells`: cells (a data provider and its parameters) and points (cells grouped by runner), named once.
- `rola_devtools.interleave`: the interleaving driver: every runner's arms on the cells of one point, timed one call at
  a time in a fresh random order per rep, across worker processes when the arms cannot share one.
- `rola_devtools.verdict`: whether a candidate's timing is a regression against its baseline.
- `rola_devtools.process`: a command stopped whole, profiler and measured process included.
"""
