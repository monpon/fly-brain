#!/usr/bin/env python
"""Watch the fly walk, driven by the CPG. Opens a live window by default.

    .venv/bin/python scripts/walk_cpg.py                 # flat ground
    .venv/bin/python scripts/walk_cpg.py --maze          # in the maze
    .venv/bin/python scripts/walk_cpg.py --turn 0.5      # steer right
    .venv/bin/python scripts/walk_cpg.py --no-view       # headless
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("MUJOCO_GL", "glfw")

import warnings
warnings.filterwarnings("ignore", message=".*Wayland.*")

import numpy as np
from flygym import Simulation
from flygym.compose import FlatGroundWorld
from flygym.utils.math import Rotation3D

from flybrain import maze as M
from flybrain.body import build_fly
from flybrain.cpg import TETRAPOD, TRIPOD, CPGWalker, GaitParams
from flybrain.view import Viewer


def main(a):
    world = FlatGroundWorld()
    spawn = np.array([0.0, 0.0, 1.2])
    if a.maze:
        grid = M.open_ends(M.generate(6, 6, seed=7))
        M.add_to_world(world, grid, M.MazeSpec())
        x, y = M.corridor_centre(grid, 0, 0, M.MazeSpec())
        spawn = np.array([x, y, 1.2])

    fly = build_fly(vision=False, adhesion=not a.maze)
    world.add_fly(fly, spawn, Rotation3D("quat", [1.0, 0.0, 0.0, 0.0]),
                  add_ground_contact_sensors=False)
    sim = Simulation(world)
    sim.reset()

    trained = Path("output/gait.json")
    if trained.exists() and not a.hand_tuned:
        import json
        saved = json.loads(trained.read_text())
        params = GaitParams(**saved["params"])
        print(f"using trained gait from {trained} "
              f"({saved['metrics']['speed']:.2f} mm/s in training)")
    else:
        params = GaitParams(frequency=a.freq, protraction=a.stride,
                            levation=a.lift, duty=a.duty)
        print("using hand-tuned gait")
    gait = TETRAPOD if a.gait == "tetrapod" else TRIPOD

    with Viewer(sim, enabled=not a.no_view, speed=a.speed,
                distance=40.0 if a.maze else 12.0) as viewer:
        walker = CPGWalker(sim, fly, params, gait, viewer)
        print(f"{a.gait} gait, {a.freq:.0f} Hz, drive {a.drive}, turn {a.turn}")
        if viewer.handle is not None:
            print("live window open -- close it to stop early")
        walker.run(a.ms, drive=a.drive, turn=a.turn)

    start, end = walker.trace[0], walker.trace[-1]
    secs = a.ms / 1000.0
    print(f"\nstart ({start[0]:6.2f}, {start[1]:6.2f})  "
          f"end ({end[0]:6.2f}, {end[1]:6.2f}) mm")
    print(f"displacement {walker.displacement():.2f} mm "
          f"over {secs:.1f}s = {walker.displacement()/secs:.1f} mm/s")
    print(f"path length  {walker.path_length():.2f} mm")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", type=float, default=3000)
    ap.add_argument("--freq", type=float, default=14.0)
    ap.add_argument("--stride", type=float, default=0.30)
    ap.add_argument("--lift", type=float, default=0.15)
    ap.add_argument("--drive", type=float, default=1.0)
    ap.add_argument("--turn", type=float, default=0.0)
    ap.add_argument("--duty", type=float, default=0.80)
    ap.add_argument("--gait", choices=["tripod", "tetrapod"], default="tripod")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--maze", action="store_true")
    ap.add_argument("--no-view", action="store_true")
    ap.add_argument("--hand-tuned", action="store_true",
                    help="ignore output/gait.json")
    main(ap.parse_args())
