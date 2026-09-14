"""rola_devtools: the development tools every RoLA repository shares, standard library only.

- `rola_devtools.mirror`: the public mirror's export (every tracked path declared ships or private, a tripwire over
  private-shaped content), run by each repository's commit gate and its mirror workflow.
- `rola_devtools.interleave`: the interleaving driver: arms a library exposes at a comparison point, timed one call at a
  time in a fresh random order per rep, across worker processes when the arms cannot share one.
"""
