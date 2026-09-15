"""THE DECLARED BUILD SYSTEM: targets declared by composable functions in files, keyed, cached, held and run.

`declare` (targets, groups, scoped labels, declaration files loaded by path), `scheduler` (keys, dependency order, the
cache, build and domain failures), `runtime` and `worker` (one worker per environment, a protocol version), `resources`
(requirements as counting semaphores over the machine's locks), `cache` and `context` (what an executor is given), and
`identity` (the code digests a target declares). `python -m rola_devtools.build plan|run FILE:TARGET` runs one.
"""
