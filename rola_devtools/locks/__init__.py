"""The machine's locks, one of each for every RoLA repository and container on a host, configured by the dev config's
`host` and `clock` sections (`rola_devtools.config`):

- `rola_devtools.locks.gpu`: `gpu_lock(mode=exclusive|shared)`, the device held by a measurement or shared by correctness
  runs;
- `rola_devtools.locks.host`: the host-compute budget (`acquire`), named file locks (`file_lock`), and the CLI that runs a
  command under the budget;
- `rola_devtools.locks.clock`: the SM clock lock the host declares, engaged and proven before a measurement.
"""
