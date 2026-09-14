#!/usr/bin/env python
"""Verify the scaffold works end to end.

Checks package versions, builds the fly body and steps physics, confirms the
sensory readouts are live, and lists neuPrint datasets if a token is set.

    flybrain check-env

This is the deep check, and it needs `flybrain[body]` to say anything about
the body. For the shallow one that runs on a bare core install and tells you
what you are missing, use `flybrain doctor`.
"""

import os
import sys

from flybrain import _deps, _platform

_platform.setup_rendering()


def check_versions() -> None:
    """Report versions, tolerating the ones this install deliberately omits."""
    import importlib.metadata as md

    print(f"platform  {_platform.describe()}\n")
    print("packages")
    for pkg in ["numpy", "pandas", "pyarrow", "torch", "brian2", "mujoco",
                "flygym", "neuprint-python"]:
        try:
            print(f"  {pkg:16s} {md.version(pkg)}")
        except md.PackageNotFoundError:
            print(f"  {pkg:16s} -- not installed")


def check_body() -> bool:
    """Build the fly, step physics, and read the sensory channels."""
    print("\nbody")
    if not _deps.have("flygym"):
        print("  skipped -- pip install 'flybrain[body]'")
        return False

    from flybrain.body import FLY_NAME, build_simulation

    sim, _ = build_simulation()
    for _ in range(200):
        sim.step()

    angles = sim.get_joint_angles(FLY_NAME)
    vision = sim.get_ommatidia_readouts(FLY_NAME)

    print(f"  stepped 200 frames to t={sim.time:.4f}s (timestep {sim.timestep}s)")
    print(f"  joint angles  {angles.shape}  actuated leg DOFs")
    print(f"  ommatidia     {vision.shape}  (eyes, facets, channels)")
    return True


def check_circuits() -> None:
    """Load a bundled circuit. Works on a core install, offline."""
    from flybrain import circuits

    print("\ncircuits")
    for name, present in circuits.available().items():
        if not present:
            print(f"  {name:14s} not bundled")
            continue
        circuit = circuits.load(name)
        print(f"  {name:14s} {circuit.summary()}")


def check_neuprint() -> None:
    """List datasets on the server. Needs a token; skipped without one."""
    from flybrain import config

    print("\nneuprint")
    tok = os.environ.get("NEUPRINT_TOKEN", "").strip()
    if not tok or tok == "paste_your_token_here":
        print("  no NEUPRINT_TOKEN set -- skipping (the bulk tables need none)")
        return
    if not _deps.have("neuprint"):
        print("  skipped -- pip install 'flybrain[neuprint]'")
        return

    from neuprint import Client

    client = Client(config.NEUPRINT_SERVER, token=tok)
    datasets = client.fetch_datasets()
    print(f"  connected to {config.NEUPRINT_SERVER}, {len(datasets)} datasets:")
    for name in sorted(datasets):
        print(f"    {name}")
    print("\n  Set NEUPRINT_DATASET=<name> to pick one.")


def main(argv=None) -> int:
    check_versions()
    check_circuits()
    check_body()
    check_neuprint()
    print("\nok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
