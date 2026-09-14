#!/usr/bin/env python
"""Verify the scaffold works end to end.

Checks package versions, builds the fly body and steps physics, confirms the
sensory readouts are live, and lists neuPrint datasets if a token is set.

    .venv/bin/python scripts/check_env.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("MUJOCO_GL", "glfw")


def check_versions() -> None:
    import importlib.metadata as md

    print("packages")
    for pkg in ["brian2", "neuprint-python", "mujoco", "flygym", "numpy", "pandas"]:
        print(f"  {pkg:16s} {md.version(pkg)}")


def check_body() -> None:
    """Build the fly, step physics, and read the sensory channels."""
    from flybrain.body import FLY_NAME, build_simulation

    print("\nbody")
    sim, _ = build_simulation()
    for _ in range(200):
        sim.step()

    angles = sim.get_joint_angles(FLY_NAME)
    vision = sim.get_ommatidia_readouts(FLY_NAME)

    print(f"  stepped 200 frames to t={sim.time:.4f}s (timestep {sim.timestep}s)")
    print(f"  joint angles  {angles.shape}  actuated leg DOFs")
    print(f"  ommatidia     {vision.shape}  (eyes, facets, channels)")


def check_neuprint() -> None:
    """List datasets on the server. Needs a token; skipped without one."""
    from flybrain import config

    print("\nneuprint")
    tok = os.environ.get("NEUPRINT_TOKEN", "").strip()
    if not tok or tok == "paste_your_token_here":
        print("  no NEUPRINT_TOKEN in .env -- skipping (see .env.example)")
        return

    from neuprint import Client

    client = Client(config.NEUPRINT_SERVER, token=tok)
    datasets = client.fetch_datasets()
    print(f"  connected to {config.NEUPRINT_SERVER}, {len(datasets)} datasets:")
    for name in sorted(datasets):
        print(f"    {name}")
    print("\n  Set NEUPRINT_DATASET=<name> in .env to pick one.")


if __name__ == "__main__":
    check_versions()
    check_body()
    check_neuprint()
    print("\nok")
