"""THE GRAPH'S GENERIC RULES, every RoLA repository's: a check over the tree's tracked files (`tree_test`), a check of the
commit (`checkout_test`), a tree tool as a test (`tool_test`) or as a run target (`tool_run`), a package's part of the
tree (`tree_part`), and the pytest tiers (`pytest_tier`, `device_lane`). A workspace's environment is its own (the
environment extension's `python` tag), so each rule takes it as an argument: `venv`, the environment's store path from
its `flags.bzl`, and `python`, its interpreter's label. A workspace wraps these once with its own environment.
"""

load("@bazel_skylib//lib:shell.bzl", "shell")

_TREE_TEST = """\
#!/bin/bash
# a check over the tree's tracked files (rola_devtools' tree_test): the runfiles copied into a real tree first, so a
# tool that resolves a path -- the runfiles are links into the checkout -- reads the check's inputs and nothing else
set -euo pipefail
tree="$TEST_TMPDIR/tree"
cp -rL "$TEST_SRCDIR/_main" "$tree"
rm -f "$tree/$TEST_BINARY"
cd "$tree"
#: no machine reaches a check over files: the dev config is the defaults
unset ROLA_DEV_CONFIG
export PATH={venv}/bin:/usr/bin:/bin RUFF_NO_CACHE=true
{git}exec {cmd}
"""

#: the copy as a git index of exactly its files: a check that asks git for the tracked files gets the check's inputs
_TREE_GIT = "git init -q && git add -A\n"

def _tree_test_impl(ctx):
    script = ctx.actions.declare_file(ctx.label.name + ".sh")
    body = _TREE_TEST.format(venv = shell.quote(ctx.attr.venv), git = _TREE_GIT if ctx.attr.git else "",
                             cmd = " ".join([shell.quote(w) for w in ctx.attr.cmd]))
    ctx.actions.write(script, body, is_executable = True)
    return [DefaultInfo(executable = script, runfiles = ctx.runfiles(files = ctx.files.data))]

tree_test = rule(
    implementation = _tree_test_impl,
    test = True,
    attrs = {
        "cmd": attr.string_list(mandatory = True, doc = "the check's command, run from the copy's root"),
        "data": attr.label_list(allow_files = True, doc = "the files it reads: the tree, or the part of it the check reads"),
        "git": attr.bool(default = False, doc = "the copy a git index of its files, for a check that asks git what is tracked"),
        "venv": attr.string(mandatory = True, doc = "the environment's store venv, whose tools are on PATH"),
    },
    doc = "A check over the tree's tracked files, the environment's tools on PATH: one test, cached on what it reads.",
)

_TOOL_TEST = """\
#!/bin/bash
# a tree tool as a test (rola_devtools' tool_test): run on the environment's base interpreter, standard library only,
# with its arguments' paths in the runfiles
set -euo pipefail
exec {python} {tool} {args}
"""

def _tool_test_impl(ctx):
    script = ctx.actions.declare_file(ctx.label.name + ".sh")
    args = [ctx.expand_location(a, ctx.attr.data) for a in ctx.attr.args_]
    ctx.actions.write(script, _TOOL_TEST.format(
        python = shell.quote(ctx.file.python.short_path),
        tool = shell.quote(ctx.file.tool.short_path),
        args = " ".join(args),
    ), is_executable = True)
    runfiles = ctx.runfiles(files = ctx.files.data + [ctx.file.tool, ctx.file.python])
    return [DefaultInfo(executable = script, runfiles = runfiles)]

tool_test = rule(
    implementation = _tool_test_impl,
    test = True,
    attrs = {
        "tool": attr.label(mandatory = True, allow_single_file = [".py"], doc = "the tree's tool, standard library only"),
        "args_": attr.string_list(doc = "its arguments, `$(rootpath ...)` and `$(rootpaths ...)` of `data` expanded"),
        "data": attr.label_list(allow_files = True, doc = "what its arguments name"),
        "python": attr.label(mandatory = True, allow_single_file = True, cfg = "exec", doc = "the environment's interpreter"),
    },
    doc = "A check that is one run of a tree tool over files the graph names.",
)

_CHECKOUT_TEST = """\
#!/bin/bash
# a check of the commit, run in the checkout it is made from (rola_devtools' checkout_test): never cached; the
# machine's dev config is the one the front door names (ROLA_DEV_CONFIG)
set -euo pipefail
{resolve}cd "$(dirname "$(readlink -f "$TEST_SRCDIR/_main/MODULE.bazel")")"
export PATH={venv}/bin:/usr/bin:/bin RUFF_NO_CACHE=true
exec {cmd} "$@"
"""

def _checkout_test_impl(ctx):
    resolve, words = [], [shell.quote(w) for w in ctx.attr.cmd]
    tools = []
    for i, (target, flag) in enumerate(ctx.attr.tools.items()):
        found = target[DefaultInfo].files.to_list()
        if len(found) != 1:
            fail("%s: %s names %d files, not one" % (ctx.label, target.label, len(found)))
        tools += found
        resolve.append('tool%d="$(readlink -f "$TEST_SRCDIR/_main/%s")"\n' % (i, found[0].short_path))
        words += [flag, '"$tool%d"' % i]
    script = ctx.actions.declare_file(ctx.label.name + ".sh")
    ctx.actions.write(script, _CHECKOUT_TEST.format(resolve = "".join(resolve), venv = shell.quote(ctx.attr.venv),
                                                    cmd = " ".join(words)), is_executable = True)
    return [DefaultInfo(executable = script, runfiles = ctx.runfiles(files = [ctx.file.anchor] + tools))]

_checkout_test = rule(
    implementation = _checkout_test_impl,
    test = True,
    attrs = {
        "cmd": attr.string_list(mandatory = True, doc = "the check's command, run from the checkout's root"),
        "tools": attr.label_keyed_string_dict(allow_files = True, doc = "graph outputs it reads, each by the flag that names its path"),
        "anchor": attr.label(mandatory = True, allow_single_file = True, doc = "the workspace's MODULE.bazel"),
        "venv": attr.string(mandatory = True, doc = "the environment's store venv, whose tools are on PATH"),
    },
)

def checkout_test(name, cmd, **kwargs):
    """A check of the commit rather than of a tree -- what git knows (HEAD, the index, pinned commits): run in the
    checkout, outside the sandbox, never cached, since its input is the repository's state and not a file set."""
    _checkout_test(name = name, cmd = cmd, tags = ["external", "local"], **kwargs)

_TOOL_RUN = """\
#!/bin/bash
# a tree tool as a run target (rola_devtools' tool_run): in the checkout, on the checkout's own venv (`rola.venv`, the
# git config key its venv writer records), so it imports the checkout, its siblings and the environment -- or, for a
# tool that needs only the environment, on the environment's own interpreter
set -euo pipefail
checkout="${{BUILD_WORKSPACE_DIRECTORY:?run through bazel run}}"
{venv}cd "$checkout"
exec "$venv/bin/python" {what} "$@"
"""

_CHECKOUT_VENV = """venv="$(git -C "$checkout" config --get rola.venv || true)"
if [ ! -x "$venv/bin/python" ]; then
  echo "$checkout has no venv of its own (git config rola.venv)" >&2
  exit 1
fi
"""

def _tool_run_impl(ctx):
    if bool(ctx.attr.tool) == bool(ctx.attr.module):
        fail("%s: a tool run names its script (`tool`) or its module (`module`): exactly one" % ctx.label)
    what = shell.quote(ctx.attr.tool) if ctx.attr.tool else "-m " + shell.quote(ctx.attr.module)
    script = ctx.actions.declare_file(ctx.label.name + ".sh")
    venv = ("venv=%s\n" % shell.quote(ctx.attr.venv)) if ctx.attr.environment else _CHECKOUT_VENV
    ctx.actions.write(script, _TOOL_RUN.format(venv = venv, what = what), is_executable = True)
    return [DefaultInfo(executable = script)]

tool_run = rule(
    implementation = _tool_run_impl,
    executable = True,
    attrs = {
        "tool": attr.string(doc = "the tool's script, from the checkout's root (`tools/lab.py`)"),
        "module": attr.string(doc = "or the module it runs (`rola_devtools.build`)"),
        "environment": attr.bool(doc = "run on the environment's interpreter, for a tool that imports nothing of the " +
                                       "checkout's siblings (CI has no venv)"),
        "venv": attr.string(mandatory = True, doc = "the environment's store venv"),
    },
    doc = "A tree tool run by `bazel run` in the checkout, on its own venv or the environment's interpreter: its " +
          "arguments follow `--`.",
)

def tree_part(tracked, parts = {}):
    """This package's part of the tree (`tracked`, the tree repository's TRACKED): `tracked`, every tracked file it
    holds; `tracked_python`, its Python files; and each of `parts` (name -> a directory prefix), the tracked files under
    it -- so a check's data names the files it reads."""
    files = tracked[native.package_name()]
    native.filegroup(name = "tracked", srcs = files, visibility = ["//visibility:public"])
    native.filegroup(name = "tracked_python", srcs = [f for f in files if f.endswith(".py")], visibility = ["//visibility:public"])
    for name, prefix in parts.items():
        native.filegroup(name = name, srcs = [f for f in files if f.startswith(prefix)], visibility = ["//visibility:public"])

_TIER = """\
#!/bin/bash
# a test tier or device lane (rola_devtools' pytest_tier, device_lane): its inputs copied into a real tree, the tree's
# packages importable from it
set -euo pipefail
tree="$TEST_TMPDIR/tree"
cp -rL "$TEST_SRCDIR/_main" "$tree"
rm -f "$tree/$TEST_BINARY"
cd "$tree"
{machine}export PATH={venv}/bin:/usr/bin:/bin PYTHONPATH="$tree" PYTHONDONTWRITEBYTECODE=1
{toolkit}{run}"""

#: the graph's copy of a pinned CUDA toolkit, the tree its nvcc runs in, handed to the tests (`ROLA_CUDA_HOME`)
_TOOLKIT = 'export ROLA_CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$TEST_SRCDIR/_main/{nvcc}")")")"\n'

#: a pytest tier's run: Bazel's filter (`--test_filter`, pytest's `-k`), its shards (rola_devtools.graph.pytest_shards),
#: its arguments (`--test_arg`) and its test report
_PYTEST_RUN = """\
only=()
if [ -n "${{TESTBRIDGE_TEST_ONLY:-}}" ]; then only=(-k "$TESTBRIDGE_TEST_ONLY"); fi
#: Bazel's sharding protocol: this shard's index and count, and the file that says the test honours them; under a
#: filter the first shard runs the whole selection and the rest none, so a filter that matches nothing still fails
shard=()
if [ -n "${{TEST_TOTAL_SHARDS:-}}" ]; then
  touch "$TEST_SHARD_STATUS_FILE"
  if [ "${{#only[@]}}" -eq 0 ]; then
    shard=(--bazel-shard "$TEST_SHARD_INDEX/$TEST_TOTAL_SHARDS")
  elif [ "$TEST_SHARD_INDEX" != 0 ]; then
    exit 0
  fi
fi
exec {venv}/bin/python -m pytest -p no:cacheprovider -p rola_devtools.graph.pytest_shards -q -rfE \\
  --junitxml="$XML_OUTPUT_FILE" {args} "${{only[@]}}" "${{shard[@]}}" "$@"
"""

#: a tier with no device sees none; every tier reads the machine's dev config (its locks)
_NO_DEVICE = 'export CUDA_VISIBLE_DEVICES=""\n'

def _tier_impl(ctx):
    script = ctx.actions.declare_file(ctx.label.name + ".sh")
    venv = shell.quote(ctx.attr.venv)
    if bool(ctx.attr.cmd) == bool(ctx.attr.tests) or (ctx.attr.cmd and ctx.attr.marks):
        fail("%s: a pytest tier names its paths (`tests`, with `marks`), a lane its command (`cmd`): exactly one" % ctx.label)
    #: a lane reaches a tool by its path alone: a `cmd` word naming it in braces (`{compute-sanitizer}`) is that path
    paths = {}
    for target, binary in ctx.attr.tools.items():
        found = [f for f in target.files.to_list() if f.basename == binary]
        if len(found) != 1:
            fail("%s carries %d files named %s" % (target.label, len(found), binary))
        paths[binary] = found[0]
    if ctx.attr.cmd:
        braced = [w[1:-1] for w in ctx.attr.cmd if len(w) > 2 and w[0] + w[-1] == "{}"]
        unknown = [name for name in braced if name not in paths]
        unused = [name for name in paths if name not in braced]
        if unknown or unused:
            fail("%s: the command's braced words %s and its tools %s must name each other" % (ctx.label, braced, sorted(paths)))
        words = [('"$TEST_SRCDIR/_main/%s"' % paths[w[1:-1]].short_path) if len(w) > 2 and w[0] + w[-1] == "{}" else shell.quote(w) for w in ctx.attr.cmd]
        run = "exec %s \"$@\"\n" % " ".join(words)
    else:
        words = list(ctx.attr.tests) + (["-m", ctx.attr.marks] if ctx.attr.marks else [])
        run = _PYTEST_RUN.format(venv = venv, args = " ".join([shell.quote(w) for w in words]))
    carried = [target.files for target in ctx.attr.tools]
    toolkit = _TOOLKIT.format(nvcc = ctx.file.nvcc.short_path) if ctx.attr.nvcc else ""
    body = _TIER.format(venv = venv, machine = "" if ctx.attr.devices else _NO_DEVICE, toolkit = toolkit, run = run)
    ctx.actions.write(script, body, is_executable = True)
    #: a device tier's pass is its devices': their identity is one of its inputs
    identity = [ctx.file.devices] if ctx.attr.devices else []
    files = depset(ctx.files.data + ctx.files.toolkit + identity, transitive = carried)
    return [DefaultInfo(executable = script, runfiles = ctx.runfiles(transitive_files = files))]

_tier_test = rule(
    implementation = _tier_impl,
    test = True,
    attrs = {
        "tests": attr.string_list(doc = "a pytest tier's paths, as pytest takes them"),
        "marks": attr.string(doc = "its marker expression (`-m`); none keeps the project's default selection"),
        "cmd": attr.string_list(doc = "a lane's command, in place of pytest, run from the copy's root; a word `{name}` " +
                                      "is the path of the tool binary of that name"),
        "tools": attr.label_keyed_string_dict(allow_files = True, doc = "the command's tools: each label's binary, by its name"),
        "data": attr.label_list(allow_files = True, doc = "what it reads: the tree and what the graph builds for it"),
        "venv": attr.string(mandatory = True, doc = "the environment's store venv"),
        "toolkit": attr.label(doc = "a pinned CUDA toolkit the tests compile, assemble and disassemble with: an input, " +
                                    "so a new one reruns them"),
        "nvcc": attr.label(allow_single_file = True, doc = "the toolkit's nvcc, by which the tests find it"),
        "devices": attr.label(allow_single_file = True, doc = "a device tier's: the devices' identity (`host_gpu`), " +
                                                              "an input of its pass; the tier sees the device"),
    },
)

def pytest_tier(name, devices = None, tags = [], **kwargs):
    """A test tier as one target, outside the sandbox (`no-sandbox`): it takes the machine's locks itself -- a device
    tier the GPU lock (`host.gpu_lock`), a sharded tier a host slot a shard (`host.lock_dir`) -- and both live in `/tmp`
    by default, which a sandboxed test sees as its own. A device tier (`devices`, the devices' identity) runs alone
    (`exclusive`)."""
    _tier_test(name = name, devices = devices, tags = tags + ["no-sandbox"] + (["exclusive"] if devices else []), **kwargs)

def device_lane(name, cmd, devices, tags = [], **kwargs):
    """A device lane: a command over the tree on the machine's device, as a device tier runs (alone, outside the sandbox,
    keyed on the devices), never part of a suite a commit runs (`manual`)."""
    _tier_test(name = name, cmd = cmd, devices = devices, tags = tags + ["no-sandbox", "exclusive", "manual"], **kwargs)
