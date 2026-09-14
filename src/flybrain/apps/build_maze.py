#!/usr/bin/env python
"""Generate a maze, drop the fly at the entrance, and render it.

    flybrain build-maze
    flybrain build-maze --rows 8 --cols 8 --seed 3
"""

import argparse
import os
import sys
from pathlib import Path

from flybrain import _platform

_platform.setup_rendering()


import numpy as np
from flygym import Simulation
from flygym.compose import FlatGroundWorld
from flygym.utils.math import Rotation3D

from flybrain import maze as M
from flybrain.body import FLY_NAME, add_wall_proxies, build_fly


def main(rows: int, cols: int, seed: int, steps: int, out: Path) -> None:
    grid = M.open_ends(M.generate(rows, cols, seed=seed))
    spec = M.MazeSpec()
    print(M.as_text(grid))

    world = FlatGroundWorld()
    info = M.add_to_world(world, grid, spec)
    print(f"\n{info['n_cells']} wall cells -> {info['n_geoms']} merged geoms")
    print(f"maze {info['width_mm']:.0f} x {info['depth_mm']:.0f} mm, "
          f"{spec.cell_mm:.0f} mm corridors")

    # Overhead camera, framed to the whole maze.
    cx, cy = info["width_mm"] / 2, info["depth_mm"] / 2
    world.mjcf_root.worldbody.add_camera(
        name="overhead",
        pos=[cx, cy, max(info["width_mm"], info["depth_mm"]) * 1.15],
        quat=[1, 0, 0, 0],
        fovy=60,
    )

    # Adhesion stays on: in flygym 2.1 it is also what pulls the head and
    # antennal segments into the model, and the wall proxies live on those.
    fly = build_fly(vision=True, adhesion=True)
    add_wall_proxies(fly)
    start_x, start_y = M.corridor_centre(grid, 0, 0, spec)
    world.add_fly(
        fly,
        np.array([start_x, start_y, 2.0]),
        Rotation3D("quat", [1.0, 0.0, 0.0, 0.0]),
        add_ground_contact_sensors=False,
    )

    sim = Simulation(world)
    sim.reset()
    print(f"fly spawned at corridor (0,0) = ({start_x:.1f}, {start_y:.1f}) mm")

    for _ in range(steps):
        sim.step()

    pos = sim.get_body_positions(FLY_NAME)[0]
    print(f"after {steps} steps (t={sim.time:.3f}s): "
          f"thorax at ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.2f}) mm")

    # Wall contacts are read straight off the MuJoCo contact list -- see
    # docs/KNOWN_ISSUES.md for why the flygym helper reports zero here.

    import mujoco
    renderer = mujoco.Renderer(sim.mj_model, 900, 900)
    renderer.update_scene(sim.mj_data, camera="overhead")
    out.parent.mkdir(parents=True, exist_ok=True)
    import PIL.Image
    PIL.Image.fromarray(renderer.render()).save(out)
    print(f"rendered -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--out", type=Path, default=Path("output/maze.png"))
    a = ap.parse_args()
    main(a.rows, a.cols, a.seed, a.steps, a.out)
