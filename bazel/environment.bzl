"""THE GRAPH'S ENVIRONMENT, every RoLA repository's: a module extension whose tags make the repositories a workspace's
graph runs on -- a Python environment built from a lock into the machine's shared store (`python`), a release pinned
by digest (`pinned`), the host C++ compiler's and the machine's devices' identities (`host_cc`, `host_gpu`), a pinned
CUDA toolkit (`cuda_toolkit`) and the checkout's tracked files (`tree`). The root module names each repository and
hands it the workspace's own pins and files; nothing here names a workspace. Docs: README.md, the build graph.

    environment = use_extension("@rola_devtools//bazel:environment.bzl", "environment")
    environment.pinned(name = "uv", pin = "//tools:uv_pin.json", build = UV_BUILD)
    environment.python(name = "env", lock = "//:requirements.lock", uv = "@uv//:uv", uv_pin = "//tools:uv_pin.json")
    environment.tree(name = "tree")
    use_repo(environment, "env", "tree", "uv")
"""

def _store(rctx):
    store = rctx.getenv("ROLA_GRAPH_STORE")
    if not store:
        fail("ROLA_GRAPH_STORE is unset: run Bazel through its front door, `bazel` (rola_devtools.graph.front_door)")
    return store

_PYTHON_BUILD = """\
package(default_visibility = ["//visibility:public"])

#: the interpreter the graph's actions run: the environment's base Python (a link it resolves to find its standard
#: library), standard library only
filegroup(name = "python", srcs = ["bin/python3"])
"""

def _build(rctx, default):
    """The repository's BUILD file: `build_file`'s text, else `build`'s, else `default`."""
    if rctx.attr.build_file:
        return rctx.read(rctx.attr.build_file)
    return rctx.attr.build or default

def _python_impl(rctx):
    pin = json.decode(rctx.read(rctx.attr.uv_pin))
    index = rctx.attr.index
    if rctx.attr.index_from:
        index = json.decode(rctx.read(rctx.attr.index_from))[rctx.attr.index_field]
    #: the environment's inputs: a change to any of them is a new environment
    facts_inputs = ([rctx.attr.post] if rctx.attr.post else []) + rctx.attr.post_args
    for source in [rctx.attr.lock, rctx.attr.make_env] + rctx.attr.watch + facts_inputs + ([rctx.attr.index_from] if rctx.attr.index_from else []):
        rctx.watch(rctx.path(source))
    for source in rctx.attr.watch_trees:
        rctx.watch_tree(rctx.path(source).dirname)
    packages = []
    if rctx.attr.devtools:
        #: rola_devtools from this module's own source -- the pinned commit's, or the machine's checkout overriding it --
        #: staged as what its install reads, so the environment's key is the package's and not the repository's tests
        devtools = rctx.path(rctx.attr._devtools).dirname
        rctx.watch_tree(devtools.get_child("rola_devtools"))
        staged = rctx.path("devtools_package")
        rctx.execute(["rm", "-rf", str(staged)])
        rctx.execute(["mkdir", "-p", str(staged)])
        for name in _DEVTOOLS_INSTALLS:
            source = devtools.get_child(name)
            if source.exists:
                rctx.watch_tree(source) if source.is_dir else rctx.watch(source)
                copied = rctx.execute(["cp", "-rL", str(source), str(staged.get_child(name))])
                if copied.return_code != 0:
                    fail("staging %s: %s" % (source, copied.stderr))
        packages.append(str(staged))
    packages += rctx.attr.packages
    args = [
        rctx.which("python3"),
        rctx.path(rctx.attr.make_env),
        "--store",
        _store(rctx),
        "--uv",
        str(rctx.path(rctx.attr.uv)),
        "--python",
        pin["python"],
        "--lock",
        str(rctx.path(rctx.attr.lock)),
        "--index",
        index,
    ]
    for package in packages:
        args += ["--package", package]
    built = rctx.execute(args, timeout = 7200, quiet = False)
    if built.return_code != 0:
        fail("the environment's store build failed (rola_devtools.graph.make_env):\n" + built.stderr)
    paths = json.decode(built.stdout.strip().splitlines()[-1])
    rctx.symlink(paths["site"], "site-packages")
    rctx.symlink(paths["include"], "python_include")
    rctx.symlink(paths["base"] + "/bin/python3", "bin/python3")
    facts = ""
    if rctx.attr.post:
        environment = {}
        for name in rctx.attr.post_env:
            value = rctx.getenv(name)
            if not value:
                fail("%s is unset: run Bazel through its front door, `bazel` (rola_devtools.graph.front_door)" % name)
            environment[name] = value
        post = rctx.execute([paths["python"], rctx.path(rctx.attr.post)] + [str(rctx.path(f)) for f in rctx.attr.post_args],
                            timeout = 600, environment = environment)
        if post.return_code != 0:
            fail("the environment's facts (%s) failed:\n%s" % (rctx.attr.post, post.stderr))
        facts = post.stdout
    #: the environment's own interpreter and tools by their store path: the path carries the environment's key, so a
    #: command naming it is keyed by the environment
    rctx.file("flags.bzl", facts + "VENV = %s\n" % json.encode(paths["venv"]))
    rctx.file("BUILD.bazel", _build(rctx, _PYTHON_BUILD))

#: what rola_devtools' install reads of its repository: the project file, the files it names, the package
_DEVTOOLS_INSTALLS = ("pyproject.toml", "LICENSE", "README.md", "rola_devtools")

_python_attrs = {
    "lock": attr.label(mandatory = True, doc = "the hash-pinned lock (`uv pip compile --generate-hashes`)"),
    "uv": attr.label(mandatory = True, allow_single_file = True, doc = "the pinned uv the store is built with"),
    "uv_pin": attr.label(mandatory = True, doc = "uv's pin, whose `python` names the interpreter uv installs"),
    "index": attr.string(default = "", doc = "an extra index the lock's packages come from (a torch index)"),
    "index_from": attr.label(allow_single_file = True, doc = "or a JSON file whose `index_field` names that index"),
    "index_field": attr.string(default = "torch_index", doc = "the field of `index_from` that names the index"),
    "devtools": attr.bool(default = True, doc = "rola_devtools from this module's source, after the lock"),
    "packages": attr.string_list(doc = "requirements pinned by commit (`name @ git+URL@COMMIT`), after the lock"),
    "post": attr.label(doc = "a script the environment's interpreter runs, its output the facts flags.bzl opens with"),
    "post_args": attr.label_list(allow_files = True, doc = "the script's arguments, each a file's path"),
    "post_env": attr.string_list(doc = "the repository environment variables the script reads"),
    "watch": attr.label_list(allow_files = True, doc = "the further files whose change is a new environment"),
    "watch_trees": attr.label_list(allow_files = True, doc = "files whose whole directory's change is a new environment"),
    "build": attr.string(doc = "the repository's BUILD file (the environment's interpreter alone by default)"),
    "build_file": attr.label(allow_single_file = True, doc = "or a file holding it"),
}

_python_repo = repository_rule(
    implementation = _python_impl,
    attrs = _python_attrs | {
        "make_env": attr.label(default = "//rola_devtools/graph:make_env.py"),
        "_devtools": attr.label(default = "//:pyproject.toml"),
    },
    doc = "A Python environment: a lock built once a key into the shared store, and `flags.bzl` with its venv's path.",
)

def _pinned_impl(rctx):
    pin = json.decode(rctx.read(rctx.attr.pin))
    asset = pin["release_asset"]
    #: an archive unpacks from its pin's folder (`archive_root`) -- a wheel (a zip) names its root; a bare binary lands
    #: at its `binary_path`
    wheel = asset.endswith(".whl")
    #: the release URL, then its mirrors: the same asset through the host's API, which serves it as octet-stream
    urls = [pin["release_url"]] + pin.get("mirror_urls", [])
    headers = {"Accept": ["application/octet-stream"]}
    if pin.get("asset_kind") == "binary":
        rctx.download(url = urls, sha256 = pin["asset_sha256"], output = pin["binary_path"], executable = True, headers = headers)
    else:
        rctx.download_and_extract(
            url = urls,
            sha256 = pin["asset_sha256"],
            type = "zip" if wheel else "tar.gz",
            stripPrefix = pin["archive_root"],
            headers = headers,
        )
    if "binary_sha256" in pin:
        binary = str(rctx.path(pin["binary_path"]))
        digest = rctx.execute(["sha256sum", binary]).stdout.split(" ")[0]
        if digest != pin["binary_sha256"]:
            fail("%s: %s's sha256 is %s, not the pin's %s" % (pin["name"], pin["binary_path"], digest, pin["binary_sha256"]))
        rctx.execute(["chmod", "0755", binary])
    #: the repository's BUILD file names the binary by its pin
    rctx.file("BUILD.bazel", _build(rctx, "").replace("{binary}", pin["binary_path"]))

_pinned_attrs = {
    "pin": attr.label(mandatory = True, doc = "the pin: its release, digest and folder"),
    "build": attr.string(doc = "the repository's BUILD file, `{binary}` the pin's binary path"),
    "build_file": attr.label(allow_single_file = True, doc = "or a file holding it"),
}

_pinned_repo = repository_rule(
    implementation = _pinned_impl,
    attrs = _pinned_attrs,
    doc = "A pinned release, downloaded by its pin's digest.",
)

def _host_cc_impl(rctx):
    cc = rctx.which("gcc")
    if not cc:
        fail("no gcc on PATH: the host C++ compiler nvcc drives is the machine's")
    tools = {
        "gcc": cc,
        "g++": rctx.which("g++"),
        "cc1plus": rctx.path(rctx.execute([cc, "-print-prog-name=cc1plus"]).stdout.strip()),
    }
    parts = []
    for tool, path in tools.items():
        rctx.watch(path)
        digest = rctx.execute(["sha256sum", str(path)]).stdout.split(" ")[0]
        parts.append("# %s %s %s" % (tool, path, digest))
    version = rctx.execute([cc, "--version"]).stdout.splitlines()[0]
    rctx.file("identity.txt", "# THE HOST C++ COMPILER'S IDENTITY, an input of every compile that no compile prunes\n# %s\n%s\n" % (version, "\n".join(parts)))
    rctx.file("BUILD.bazel", 'exports_files(["identity.txt"], visibility = ["//visibility:public"])\n')

_host_cc_repo = repository_rule(
    implementation = _host_cc_impl,
    doc = "The host C++ compiler's version and digests, as a file every compile takes as an input (never a header, " +
          "which a compile drops from its key when it does not read it).",
)

def _host_gpu_impl(rctx):
    #: only a device tier reads this repository, and a device tier on a machine that cannot name its devices refuses
    smi = rctx.which("nvidia-smi")
    if not smi:
        fail("no nvidia-smi on PATH: a device test tier is keyed on the machine's devices -- load the NVIDIA driver, " +
             "whose nvidia-smi names them, or run a tier that needs none")
    #: read again when the driver moves: its tools move with it
    rctx.watch(smi)
    found = rctx.execute([smi, "--query-gpu=index,name,uuid,driver_version,compute_cap,memory.total", "--format=csv,noheader"])
    if found.return_code != 0 or not found.stdout.strip():
        fail("nvidia-smi names no device (exit %d): %s -- a device test tier needs one" % (found.return_code, found.stderr))
    rctx.file("identity.txt", "# THE MACHINE'S DEVICES, an input of every device test tier\n" + found.stdout)
    rctx.file("BUILD.bazel", 'exports_files(["identity.txt"], visibility = ["//visibility:public"])\n')

_host_gpu_repo = repository_rule(
    implementation = _host_gpu_impl,
    #: read at every fetch, so a device swapped under one driver is the key's too
    local = True,
    doc = "The machine's devices (each one's name, UUID, driver and compute capability), an input of every device " +
          "test tier: a pass is the device's, never replayed for another.",
)

def _component(rctx, root, entry, output, license):
    """One redistributable component by the manifest's digest, its archive's own folder stripped, its LICENSE kept at
    `license` (every component carries one at the same path)."""
    folder = entry["relative_path"].rsplit("/", 1)[1].removesuffix(".tar.xz")
    rctx.download_and_extract(
        url = root + "/" + entry["relative_path"],
        sha256 = entry["sha256"],
        stripPrefix = folder,
        #: a rename comes before the folder is stripped, so the new name is under the folder too
        rename_files = {folder + "/LICENSE": folder + "/" + license},
        output = output,
    )

def _cuda_toolkit_impl(rctx):
    pin = json.decode(rctx.read(rctx.attr.pin))
    rctx.download(url = pin["manifest_url"], sha256 = pin["manifest_sha256"], output = "redistrib.json")
    redist = json.decode(rctx.read("redistrib.json"))
    root = pin["manifest_url"].rsplit("/", 1)[0]
    #: every component in one tree, as the machine's own installer lays it out, then the pin's links
    for component in pin["components"]:
        _component(rctx, root, redist[component]["linux-x86_64"], ".", "licenses/%s/LICENSE" % component)
    for link, target in pin["links"].items():
        rctx.execute(["ln", "-sfn", target, link])
    #: further components each in a folder of its own, the BUILD file's to expose (their headers, their tools)
    for component, folder in rctx.attr.beside.items():
        _component(rctx, root, redist[component]["linux-x86_64"], folder, "LICENSE")
    for path in rctx.attr.delete:
        rctx.delete(path)
    rctx.file("BUILD.bazel", _build(rctx, ""))

_cuda_toolkit_attrs = {
    "pin": attr.label(mandatory = True, doc = "the redistributable manifest by its digest, the components and links"),
    "beside": attr.string_dict(doc = "further components, each into the folder named"),
    "delete": attr.string_list(doc = "paths removed after the unpack"),
    "build": attr.string(doc = "the repository's BUILD file"),
    "build_file": attr.label(allow_single_file = True, doc = "or a file holding it"),
}

_cuda_toolkit_repo = repository_rule(
    implementation = _cuda_toolkit_impl,
    attrs = _cuda_toolkit_attrs,
    doc = "A pinned CUDA toolkit: NVIDIA's redistributable manifest by its digest, each component by the manifest's.",
)

def _tree_impl(rctx):
    root = str(rctx.workspace_root)
    git = rctx.which("git")
    #: the commit gate's tree is the commit's index (a partial commit's is not the checkout's): the hook hands it in
    index_file = rctx.getenv("GIT_INDEX_FILE")
    env = {"GIT_INDEX_FILE": index_file} if index_file else {}
    index = rctx.execute([git, "-C", root, "rev-parse", "--path-format=absolute", "--git-path", "index"], environment = env).stdout.strip()
    #: re-read when the index moves: a file added, removed or staged
    rctx.watch(index)
    listed = rctx.execute([git, "-C", root, "ls-files", "-z"], environment = env)
    if listed.return_code != 0:
        fail("git ls-files failed in %s: %s" % (root, listed.stderr))
    files = [f for f in listed.stdout.split("\0") if f]
    packages = sorted([f.rsplit("/", 1)[0] if "/" in f else "" for f in files if f.rsplit("/", 1)[-1] == "BUILD.bazel"])
    tracked = {pkg: [] for pkg in packages}
    for f in files:
        owner = ""
        for pkg in packages:
            if pkg and f.startswith(pkg + "/") and len(pkg) > len(owner):
                owner = pkg
        tracked[owner].append(f[len(owner) + 1:] if owner else f)
    rctx.file("tree.bzl", "\"\"\"GENERATED from `git ls-files`: the checkout's tracked files, by the package that holds them.\"\"\"\n\nTRACKED = %s\n" % json.encode_indent(tracked, indent = "    "))
    rctx.file("BUILD.bazel", 'exports_files(["tree.bzl"], visibility = ["//visibility:public"])\n')

_tree_repo = repository_rule(
    implementation = _tree_impl,
    doc = "The checkout's tracked files (`git ls-files`), by package: the tree the lints read, re-read when the index moves.",
)

def _named(attrs):
    return attrs | {"name": attr.string(mandatory = True, doc = "the repository's name")}

_TAGS = {
    "python": (_python_repo, _named(_python_attrs)),
    "pinned": (_pinned_repo, _named(_pinned_attrs)),
    "host_cc": (_host_cc_repo, _named({})),
    "host_gpu": (_host_gpu_repo, _named({})),
    "cuda_toolkit": (_cuda_toolkit_repo, _named(_cuda_toolkit_attrs)),
    "tree": (_tree_repo, _named({})),
}

def _environment_impl(mctx):
    for mod in mctx.modules:
        if not mod.is_root:
            continue
        for kind, (repo, attrs) in _TAGS.items():
            for tag in getattr(mod.tags, kind):
                repo(**{name: getattr(tag, name) for name in attrs})

environment = module_extension(
    implementation = _environment_impl,
    tag_classes = {kind: tag_class(attrs = attrs) for kind, (_repo, attrs) in _TAGS.items()},
    doc = "The graph's environment, made from the root module's tags: its Python environments, pinned releases, the " +
          "host compiler's and devices' identities, a pinned CUDA toolkit and the checkout's tracked files.",
)
