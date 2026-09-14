"""One command, so there is no per-OS path to get wrong.

`.venv/bin/python scripts/see.py` is three platform assumptions in one string:
that the interpreter lives at that path, that the working directory is the
repo root, and that `scripts/` shipped at all. An installed console script has
none of those problems -- `flybrain see` is the same on every machine.

Commands split in two. The handful implemented here are the SDK itself:
setup, diagnostics, data, and the bundled circuits. The rest are the
experiments, which live in `flybrain.apps` and are run unchanged, so they
still work as `python -m flybrain.apps.see` too.
"""

from __future__ import annotations

import argparse
import runpy
import sys

# command name -> (module in flybrain.apps, one-line description)
APPS = {
    "see": ("see", "render a stimulus onto the measured retina"),
    "react": ("react", "show the fly something, read what its neurons do"),
    "remember": ("remember", "what the circuit can hold on to, and how long"),
    "pong": ("pong", "play pong against the dopamine rule"),
    "train-brain": ("train_brain", "train a circuit on input/output pairs"),
    "train-fly": ("train_fly", "condition an odour, save the checkpoint"),
    "teach-odor": ("teach_odor", "differential conditioning, good vs bad odour"),
    "forage": ("forage", "hunt by smell in an open arena"),
    "maze-forage": ("maze_forage", "solve a maze by smell"),
    "walk": ("walk", "connectome VNC in closed loop with the body"),
    "walk-cpg": ("walk_cpg", "watch the fly walk, driven by the CPG"),
    "build-maze": ("build_maze", "generate a maze and drop the fly in it"),
    "train-gait": ("train_gait", "optimise the gait (differential evolution)"),
    "tune-gait": ("tune_gait", "optimise the gait (CMA-ES, with gearing)"),
    "simulate-optomotor": ("simulate_optomotor", "drive T4, watch the circuit"),
    "simulate-vnc": ("simulate_vnc", "drive a descending neuron, look for gait"),
    "fetch-optomotor": ("fetch_optomotor", "extract the optomotor circuit"),
    "fetch-mushroom-body": ("fetch_mushroom_body", "extract the mushroom body"),
    "check-env": ("check_env", "full end-to-end check (needs every extra)"),
}

BUILTIN = {
    "doctor": "what this machine has, and what it is missing",
    "info": "package, data, and bundled-circuit status",
    "circuits": "list the circuits that ship with this install",
    "fetch": "download the connectome tables (1.1 GB)",
    "bundle": "rebuild a bundled circuit from the full tables",
    "demo": "a 30-second tour that needs no download",
}


def _usage() -> str:
    lines = ["flybrain <command> [options]", "", "setup and data:"]
    for name, about in BUILTIN.items():
        lines.append(f"  {name:20s} {about}")
    lines += ["", "experiments:"]
    for name, (_, about) in APPS.items():
        lines.append(f"  {name:20s} {about}")
    lines += ["", "any command takes --help for its own options."]
    return "\n".join(lines)


def cmd_doctor(argv) -> int:
    """Report the environment without needing any optional dependency."""
    from . import _deps, _platform, circuits, data

    print(f"platform   {_platform.describe()}")
    print(f"data dir   {data.config.dataset_dir()}")
    print(f"cache root {_platform.cache_root()}")

    print("\noptional dependencies")
    for module, present in _deps.installed().items():
        extra, _ = _deps.EXTRAS[module]
        mark = "x" if present else " "
        print(f"  [{mark}] {module:12s} flybrain[{extra}]")

    print()
    print(circuits.describe())

    missing = data.missing_core()
    print(f"\nconnectome tables: "
          f"{'all present' if not missing else f'{len(missing)} missing'}")
    if missing:
        print("  run `flybrain fetch` for the full dataset, or just use the "
              "bundled circuits above.")

    if not _platform.has_compiler() and _deps.have("brian2"):
        print("\nwarning: brian2 is installed but no C compiler was found.")
        print("  Brian2 will fall back to numpy codegen, which is much slower.")
        if _platform.WINDOWS:
            print("  Install Microsoft C++ Build Tools to get the fast path.")
    return 0


def cmd_info(argv) -> int:
    from . import circuits, data

    print(f"flybrain {_version()}")
    manifest = circuits.manifest()
    if manifest:
        print("\nbundled circuits")
        for name, meta in manifest.items():
            print(f"  {name:14s} {meta['neurons']:>7,} neurons  "
                  f"{meta['synapses']:>9,} synapses  "
                  f"{meta['bytes']/1e6:5.1f} MB  {meta['dataset']}")
    print()
    print(data.status())
    return 0


def cmd_circuits(argv) -> int:
    from . import circuits

    print(circuits.describe())
    return 0


def cmd_fetch(argv) -> int:
    from . import data

    ap = argparse.ArgumentParser(prog="flybrain fetch")
    ap.add_argument("--all", action="store_true",
                    help="also the synapse-level tables (~24 GB total)")
    ap.add_argument("--status", action="store_true", help="show what is here")
    args = ap.parse_args(argv)

    if args.status:
        print(data.status())
        return 0

    names = list(data.ALL if args.all else data.CORE)
    for name in names:
        data.fetch(name)
    print("\n" + data.status())
    return 0


def cmd_bundle(argv) -> int:
    """Regenerate the shipped circuits from the full tables.

    Only needed by whoever builds the wheel -- the npz files are the product
    of this command, and once they exist nobody else has to download anything.
    """
    from . import circuits, data

    ap = argparse.ArgumentParser(prog="flybrain bundle")
    ap.add_argument("names", nargs="*", default=None,
                    help=f"which to build; default all of {list(circuits.CATALOG)}")
    args = ap.parse_args(argv)

    if data.missing_core():
        print("bundling needs the full tables; fetching them first")
        data.fetch_core()

    targets = args.names or list(circuits.CATALOG)
    for name in targets:
        if name not in circuits.CATALOG:
            raise SystemExit(f"unknown circuit {name!r}; "
                             f"known: {list(circuits.CATALOG)}")
        print(f"building {name} ...", flush=True)
        circuit = circuits.rebuild(name)
        path = circuits.write_npz(circuit, circuits.BUNDLE_DIR / f"{name}.npz")
        print(f"  {circuit.summary()}  ->  {path.name} "
              f"({path.stat().st_size/1e6:.1f} MB)")

    for side in ("R", "L"):
        try:
            from . import vision

            ret = vision.retina(side=side)
        except Exception as exc:                       # noqa: BLE001
            print(f"  skipped retina {side}: {exc}")
            continue
        path = circuits.write_retina(
            ret, circuits.BUNDLE_DIR / f"retina_{side}.npz")
        print(f"  retina {side}: {ret.n} columns -> {path.name}")

    circuits.write_manifest()
    print(f"\nmanifest written to {circuits.MANIFEST}")
    return 0


def cmd_demo(argv) -> int:
    """Prove the install works, using only the bundled data and numpy."""
    import numpy as np

    from . import circuits

    print("loading the mushroom body from the bundle (no download) ...")
    mb = circuits.mushroom_body()
    print(f"  {mb.summary()}")
    print(f"  dataset {mb.dataset or 'male-cns-v1.0'}, "
          f"topology fingerprint {mb.fingerprint()}")

    print("\n  largest cell-type populations")
    for name, count in mb.type_counts(limit=6):
        print(f"    {name:12s} {count:>6,}")

    inhibitory = int((mb.weight < 0).sum())
    print(f"\n  {mb.n_synapses:,} synapses, "
          f"{inhibitory:,} inhibitory ({100*inhibitory/mb.n_synapses:.1f}%)")

    indeg = np.bincount(mb.post, minlength=mb.n)
    print(f"  in-degree: median {int(np.median(indeg))}, max {int(indeg.max())}")

    print("\nthat ran with numpy alone. Next:")
    print("  pip install 'flybrain[torch]'   then  flybrain train-brain")
    print("  pip install 'flybrain[body]'    then  flybrain walk-cpg")
    return 0


BUILTIN_FUNCS = {
    "doctor": cmd_doctor,
    "info": cmd_info,
    "circuits": cmd_circuits,
    "fetch": cmd_fetch,
    "bundle": cmd_bundle,
    "demo": cmd_demo,
}


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("flybrain")
    except Exception:                                   # noqa: BLE001
        return "0.1.0 (not installed)"


def _run_app(module: str, argv: list[str], command: str) -> int:
    """Run an experiment module as if it had been invoked directly.

    `runpy` with `run_name="__main__"` means the scripts keep their existing
    `if __name__ == "__main__"` blocks and their own argparse, so nothing had
    to be rewritten to make them dispatchable -- and they still run standalone.
    """
    saved = sys.argv
    sys.argv = [f"flybrain {command}", *argv]
    try:
        runpy.run_module(f"flybrain.apps.{module}", run_name="__main__",
                         alter_sys=True)
    except SystemExit as exc:
        # `raise SystemExit("message")` is how these scripts report a bad
        # setup, so code is often a string rather than a status number.
        if exc.code is None or isinstance(exc.code, int):
            return exc.code or 0
        print(exc.code, file=sys.stderr)
        return 1
    except ImportError as exc:
        print(f"{command} needs an optional dependency:\n  {exc}",
              file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    finally:
        sys.argv = saved
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_usage())
        return 0
    if argv[0] in ("-V", "--version"):
        print(_version())
        return 0

    command, rest = argv[0], argv[1:]

    if command in BUILTIN_FUNCS:
        return BUILTIN_FUNCS[command](rest)
    if command in APPS:
        return _run_app(APPS[command][0], rest, command)

    near = [c for c in (*BUILTIN, *APPS) if c.startswith(command[:3])]
    print(f"unknown command {command!r}", file=sys.stderr)
    if near:
        print(f"did you mean: {', '.join(near)}", file=sys.stderr)
    print(f"\n{_usage()}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
