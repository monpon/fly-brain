#!/usr/bin/env python
"""Put the fly in a maze with something good at the far end and let it walk.

The fly is given no map, no coordinates and no plan. It has:

  * two antennae reading the concentration of an odour that travels down the
    corridors (`odor_field.MazeOdorSource` -- the field is a geodesic, so it
    does not leak through walls);
  * a mushroom body that has been taught what that odour is worth
    (`scripts/teach_odor.py`);
  * antennal contact sensing, which turns it away from walls it runs into.

    .venv/bin/python scripts/maze_forage.py
    .venv/bin/python scripts/maze_forage.py --rows 5 --cols 5 --seed 3
    .venv/bin/python scripts/maze_forage.py --no-view --ms 90000
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("MUJOCO_GL", "glfw")

import warnings
warnings.filterwarnings("ignore", message=".*Wayland.*")

import mujoco
import numpy as np
from flygym import Simulation
from flygym.compose import ActuatorType, FlatGroundWorld
from flygym.utils.math import Rotation3D

from flybrain import checkpoint, maze as M, mushroom_body as MB
from flybrain.antennae import ReflexParams, WallReflex
from flybrain.body import (FLY_NAME, add_wall_proxies, build_fly,
                           joint_dof_index, neutral_targets)
from flybrain.cpg import CPG, GaitParams
from flybrain.learning import FlyBrain
from flybrain.odor_field import MazeOdorSource
from flybrain.olfaction import Nose
from flybrain.olfactory_brain import BrainNavParams, BrainNavigator
from flybrain.view import Viewer


def main(a):
    gait_file = Path("output/gait.json")
    gait = GaitParams(**json.loads(gait_file.read_text())["params"]) \
        if gait_file.exists() else GaitParams()

    grid = M.generate(a.rows, a.cols, seed=a.seed)
    spec = M.MazeSpec(cell_mm=a.cell, wall_mm=a.wall, height_mm=5.0)
    start = M.corridor_centre(grid, 0, 0, spec)
    goal = M.corridor_centre(grid, a.rows - 1, a.cols - 1, spec)

    world = FlatGroundWorld()
    info = M.add_to_world(world, grid, spec)
    world.mjcf_root.worldbody.add_geom(
        name="odor_source", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[1.5],
        pos=[goal[0], goal[1], 1.5], rgba=[0.3, 0.85, 0.3, 0.7],
    )
    world.mjcf_root.worldbody.add_camera(
        name="overhead", pos=[info["width_mm"] / 2, info["depth_mm"] / 2,
                              max(info["width_mm"], info["depth_mm"]) * 1.15],
        quat=[1, 0, 0, 0], fovy=60,
    )

    source = MazeOdorSource.build(
        grid, spec, (goal[0], goal[1], 0.0),
        res_mm=a.res, clearance_mm=a.clearance,
        decay_mm=a.decay, odor=a.odor,
    )
    route = source.path_length((start[0], start[1]))
    straight = float(np.hypot(goal[0] - start[0], goal[1] - start[1]))

    print(M.as_text(grid))
    print(f"\nmaze {info['width_mm']:.0f} x {info['depth_mm']:.0f} mm, "
          f"{spec.cell_mm:.0f} mm corridors, {info['n_geoms']} wall geoms")
    print(f"route through the maze {route:.0f} mm "
          f"(straight line would be {straight:.0f} mm)")
    print(f"odour at the exit: concentration {source.concentration([[start[0], start[1], 0]])[0]:.5f} "
          f"at the entrance, 1.0 at the source")

    # Adhesion stays on. It is also, in flygym 2.1, what pulls the head and
    # antennal segments into the model at all -- with `adhesion=False` the
    # body tree stops at the thorax and there is nothing to feel walls with.
    fly = build_fly(vision=False, adhesion=True)
    add_wall_proxies(fly)
    world.add_fly(fly, np.array([start[0], start[1], 2.0]),
                  Rotation3D("quat", [1.0, 0.0, 0.0, 0.0]),
                  add_ground_contact_sensors=False)

    sim = Simulation(world)
    sim.reset()
    index = joint_dof_index(fly)
    neutral = neutral_targets(fly)
    cpg = CPG(gait)
    nose = Nose(sim)
    reflex = WallReflex(sim, FLY_NAME, ReflexParams(turn=a.wall_turn))

    mb = MB.load_cache()
    brain = FlyBrain(mb)
    if not a.untrained and Path(a.trained).exists():
        checkpoint.load(a.trained, brain)
        print(f"controller: mushroom body, loaded {a.trained}")
    else:
        print("controller: mushroom body, untrained")
    nav = BrainNavigator(
        brain, mb, [a.odor], seed=a.seed,
        params=BrainNavParams(turn_slowdown=a.slowdown, cast_turn=0.5),
    )
    print(f"learned valence: {nav.report()}\n")

    # get_body_positions is ordered by get_bodysegs_order(), which is *not*
    # the model's body order -- looking the thorax up in the model's body list
    # silently returns a tarsus.
    thorax = [s.name for s in fly.get_bodysegs_order()].index("c_thorax")
    for _ in range(500):
        sim.step()

    viewer = Viewer(sim, enabled=not a.no_view, speed=a.speed, distance=45.0)
    if viewer.handle is not None:
        print("live window open -- close it to stop early\n")

    inner = max(1, int(round(0.001 / sim.timestep)))
    best = np.inf
    reached = None
    touches = 0
    was_touching = False
    stall_ref = sim.get_body_positions(FLY_NAME)[thorax][:2].copy()
    stall_t = 0
    escape = 0.0
    trail = []

    for step in range(int(a.ms)):
        samples = nose.sample_each([source])
        turn, drive = nav.update(samples, 1.0)

        # Mechanosensation preempts olfaction: no gradient tells you about a
        # wall. The escape is a last resort for the case the antennae miss.
        wall = reflex.override(1.0)
        if escape > 0.0:
            escape -= 1.0
            turn, drive = np.sign(escape_sign), 0.3
        elif wall is not None:
            turn, drive = wall
        if reflex.touching and not was_touching:
            touches += 1
        was_touching = reflex.touching

        cpg.step(0.001, drive=drive, turn=turn)
        targets = neutral.copy()
        for key, value in cpg.joint_targets().items():
            for slot in index.get(key, []):
                targets[slot] += value
        sim.set_actuator_inputs(FLY_NAME, ActuatorType.POSITION, targets)
        for _ in range(inner):
            sim.step()

        pos = sim.get_body_positions(FLY_NAME)[thorax]
        if step % 20 == 0:
            trail.append(pos[:2].copy())

        # Stall: no net progress for a while means wedged in a corner with
        # both antennae just short of the wall.
        stall_t += 1
        if stall_t >= a.stall_ms:
            moved = float(np.linalg.norm(pos[:2] - stall_ref))
            if moved < a.stall_mm and escape <= 0.0:
                escape = a.escape_ms
                escape_sign = 1.0 if (step // 1000) % 2 == 0 else -1.0
            stall_ref = pos[:2].copy()
            stall_t = 0

        remaining = source.path_length(pos[:2])
        best = min(best, remaining)
        if remaining < a.radius and reached is None:
            reached = step
            print(f"*** reached the {a.odor} at {step/1000:.1f} s ***")
            break
        if step % 5000 == 0 and step:
            print(f"  t={step/1000:5.1f}s  {remaining:6.1f} mm of corridor "
                  f"left (best {best:.1f})  {touches} wall contacts")

        if viewer.handle is not None and not viewer.sync(sim.time):
            break
    viewer.close()

    pos = sim.get_body_positions(FLY_NAME)[thorax]
    walked = float(np.linalg.norm(np.diff(np.array(trail), axis=0),
                                  axis=1).sum()) if len(trail) > 1 else 0.0
    print(f"\nstarted {route:.0f} mm from the source along the corridors")
    print(f"closest approach {best:.1f} mm   final {source.path_length(pos[:2]):.1f} mm")
    print(f"walked {walked:.0f} mm of path, {touches} wall contacts")
    print("solved the maze" if reached is not None else "did not reach the source")

    if a.trail:
        field = np.where(np.isfinite(source.distance),
                         np.exp(-source.distance / source.decay_mm), np.nan)
        print("trail -> " + M.plot_run(grid, spec, trail, goal, a.trail,
                                       field=field, res_mm=a.res))

    if a.render:
        renderer = mujoco.Renderer(sim.mj_model, 900, 900)
        renderer.update_scene(sim.mj_data, camera="overhead")
        import PIL.Image
        Path(a.render).parent.mkdir(parents=True, exist_ok=True)
        PIL.Image.fromarray(renderer.render()).save(a.render)
        print(f"rendered -> {a.render}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cell", type=float, default=20.0,
                    help="corridor width in mm; the fly needs room to turn")
    ap.add_argument("--wall", type=float, default=2.0)
    ap.add_argument("--odor", default="vinegar")
    ap.add_argument("--decay", type=float, default=80.0,
                    help="odour length constant along the corridors")
    ap.add_argument("--res", type=float, default=1.0)
    ap.add_argument("--clearance", type=float, default=2.0)
    ap.add_argument("--radius", type=float, default=6.0,
                    help="corridor distance counting as arrival")
    ap.add_argument("--wall-turn", type=float, default=0.9)
    ap.add_argument("--slowdown", type=float, default=0.5)
    ap.add_argument("--stall-ms", type=int, default=2000)
    ap.add_argument("--stall-mm", type=float, default=3.0)
    ap.add_argument("--escape-ms", type=float, default=900.0)
    ap.add_argument("--ms", type=float, default=60000)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--no-view", action="store_true")
    ap.add_argument("--untrained", action="store_true")
    ap.add_argument("--trained", default="output/odor_trained.flyckpt")
    ap.add_argument("--render", default="output/maze_run.png")
    ap.add_argument("--trail", default="output/maze_trail.png",
                    help="overhead plot of the odour field and the path taken")
    main(ap.parse_args())
