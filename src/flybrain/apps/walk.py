#!/usr/bin/env python
"""Run the connectome VNC in closed loop with the body. No CPG.

    flybrain walk                    # DNp09, flat ground
    flybrain walk --ms 2000 --maze
"""

import argparse
import os
import sys
import time
from pathlib import Path

from flybrain import _platform

_platform.setup_rendering()


import numpy as np
from flygym import Simulation
from flygym.compose import FlatGroundWorld
from flygym.utils.math import Rotation3D

from flybrain import maze as M, vnc
from flybrain.body import build_fly
from flybrain.embodied import EmbodiedFly, EmbodiedParams
from flybrain.network import use_fast_codegen
from flybrain.view import Viewer


def main(a):
    print("extracting VNC from connectome...")
    t0 = time.time()
    d = vnc.extract()
    W, neurons = d["W"], d["neurons"]
    print(f"  {W.shape[0]} neurons, {int((W != 0).sum())} connections "
          f"({time.time() - t0:.1f}s)")

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

    use_fast_codegen()
    params = EmbodiedParams(normalize_input_mv=a.norm, joint_gain=a.gain)
    viewer = Viewer(sim, enabled=not a.no_view, speed=a.speed,
                    distance=40.0 if a.maze else 12.0)
    ef = EmbodiedFly(sim, fly, W, neurons, params, viewer=viewer)
    n = ef.command_descending(a.stim, a.rate)
    if viewer.handle is not None:
        print("  live window open -- close it to stop early")

    print(f"  {len(ef.motor.index)} motor neurons -> {len(ef.dof_index)} joints")
    print(f"  {len(ef.sensory.index)} proprioceptors "
          f"({dict(zip(*np.unique(ef.sensory.modality, return_counts=True)))})")
    print(f"\ndriving {n} {a.stim} at {a.rate:.0f} Hz, "
          f"norm {a.norm}, joint gain {a.gain}")

    t0 = time.time()
    ef.run(a.ms)
    wall = time.time() - t0
    viewer.close()

    act = np.asarray(ef.group.a[:])
    print(f"\nran {a.ms:.0f} ms biological in {wall:.1f}s "
          f"({wall / (a.ms / 1000):.0f}x slower than real time)")
    print(f"{len(ef.spikes.i)} spikes, "
          f"mean muscle activation {act[ef.motor.index].mean():.3f}")

    start, end = ef.trace[0], ef.trace[-1]
    print(f"\nthorax start ({start[0]:.2f}, {start[1]:.2f}, {start[2]:.2f}) mm")
    print(f"thorax end   ({end[0]:.2f}, {end[1]:.2f}, {end[2]:.2f}) mm")
    print(f"displacement {ef.displacement():.2f} mm")

    path = np.array(ef.trace)
    dist = float(np.abs(np.diff(path[:, :2], axis=0)).sum())
    print(f"path length  {dist:.2f} mm   height drift {np.ptp(path[:,2]):.2f} mm")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stim", default="DNp09")
    ap.add_argument("--rate", type=float, default=100.0)
    ap.add_argument("--norm", type=float, default=100.0)
    ap.add_argument("--gain", type=float, default=0.35)
    ap.add_argument("--ms", type=float, default=1000.0)
    ap.add_argument("--maze", action="store_true")
    ap.add_argument("--no-view", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0)
    main(ap.parse_args())
