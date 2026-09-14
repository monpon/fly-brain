#!/usr/bin/env python
"""Let the fly hunt by smell, and act on what it has learned to like.

No target coordinates reach the controller. It gets the concentration of each
odour at its two antennae, and the mushroom body's learned verdict on what
each smell is worth. With a rewarded odour and a punished one in the same
arena, that is enough to walk toward one and away from the other.

    flybrain forage                  # good + bad source
    flybrain forage --untrained      # before conditioning
    flybrain forage --no-bad         # single source
    flybrain forage --distance 60 --ms 20000 --no-view

Train the preference first with flybrain teach-odor.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from flybrain import _platform

_platform.setup_rendering()


import mujoco
import numpy as np
from flygym import Simulation
from flygym.compose import ActuatorType, FlatGroundWorld
from flygym.utils.math import Rotation3D

from flybrain.body import FLY_NAME, build_fly, joint_dof_index, neutral_targets
from flybrain.cpg import CPG, GaitParams
from flybrain.learning import FlyBrain
from flybrain.navigate import NavParams, OdorNavigator
from flybrain.olfactory_brain import BrainNavParams, BrainNavigator
from flybrain import checkpoint, config, mushroom_body as MB
from flybrain.olfaction import Nose, OdorSource
from flybrain.view import Viewer


def place(distance: float, bearing_deg: float) -> np.ndarray:
    angle = np.radians(bearing_deg)
    return np.array([distance * np.cos(angle), distance * np.sin(angle), 0.0])


def main(a):
    gait_file = config.gait_file()
    params = GaitParams(**json.loads(gait_file.read_text())["params"]) \
        if gait_file.exists() else GaitParams()

    good_at = place(a.distance, a.bearing)
    sources = [OdorSource(good_at, strength=1.0, decay_mm=a.decay, odor=a.good)]
    if not a.no_bad:
        bad_at = place(a.bad_distance or a.distance, a.bad_bearing)
        sources.append(
            OdorSource(bad_at, strength=1.0, decay_mm=a.decay, odor=a.bad)
        )

    world = FlatGroundWorld()
    colors = [[0.3, 0.8, 0.3, 0.6], [0.7, 0.2, 0.8, 0.6]]
    for i, src in enumerate(sources):
        world.mjcf_root.worldbody.add_geom(
            name=f"odor_{i}", type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[a.radius],
            pos=[float(src.position[0]), float(src.position[1]), a.radius],
            rgba=colors[i % len(colors)],
        )
    fly = build_fly(vision=False, adhesion=True)
    world.add_fly(fly, np.array([0.0, 0.0, 1.2]),
                  Rotation3D("quat", [1.0, 0.0, 0.0, 0.0]),
                  add_ground_contact_sensors=False)

    sim = Simulation(world)
    sim.reset()
    index = joint_dof_index(fly)
    neutral = neutral_targets(fly)
    cpg = CPG(params)
    nose = Nose(sim)

    if a.tumble:
        nav = OdorNavigator(NavParams(), seed=a.seed)
        print("controller: run-and-tumble (hand-written)")
    else:
        mb = MB.load_cache()
        brain = FlyBrain(mb)
        if not a.untrained and Path(a.trained).exists():
            checkpoint.load(a.trained, brain)
            print(f"controller: mushroom body, loaded {a.trained}")
        elif a.untrained:
            print("controller: mushroom body, UNTRAINED (nothing learned yet)")
        else:
            print(f"controller: mushroom body, no checkpoint at {a.trained}")
        nav = BrainNavigator(brain, mb, [s.odor for s in sources],
                             params=BrainNavParams(valence_scale=a.valence_scale,
                                                   noise=a.noise,
                                                   noise_tail=a.noise_tail),
                             seed=a.seed)
        print(f"learned valence: {nav.report()}")

    # get_body_positions is ordered by get_bodysegs_order(), not by model
    # body id; the two disagree and the model-order lookup returns a tarsus.
    thorax = [s.name for s in fly.get_bodysegs_order()].index("c_thorax")

    for _ in range(500):
        sim.step()

    for src, tag in zip(sources, ("green", "purple")):
        d = np.linalg.norm(src.position[:2])
        print(f"{tag:7s} {src.odor:9s} {d:.0f} mm away at "
              f"{np.degrees(np.arctan2(src.position[1], src.position[0])):.0f} deg")
    print()

    viewer = Viewer(sim, enabled=not a.no_view, speed=a.speed, distance=60.0)
    if viewer.handle is not None:
        print("live window open -- close it to stop early\n")

    best = np.full(len(sources), np.inf)
    found_at = None
    states = {}
    track = []
    inner = max(1, int(round(0.001 / sim.timestep)))

    for step in range(int(a.ms)):
        samples = nose.sample_each(sources)
        if a.tumble:
            turn, drive = nav.update(*samples.sum(axis=0), 1.0)
        else:
            turn, drive = nav.update(samples, 1.0)
        states[nav.state] = states.get(nav.state, 0) + 1

        cpg.step(0.001, drive=drive, turn=turn)
        targets = neutral.copy()
        for key, value in cpg.joint_targets().items():
            for slot in index.get(key, []):
                targets[slot] += value
        sim.set_actuator_inputs(FLY_NAME, ActuatorType.POSITION, targets)
        for _ in range(inner):
            sim.step()

        P = sim.get_body_positions(FLY_NAME)
        pos = P[thorax]
        # Distance from the *nearest body segment*, not from the thorax. The
        # thorax centre sits about 1.5 mm behind the antennae, so an arrival
        # threshold measured from it is arbitrary by roughly that much -- and
        # the earlier version of this script measured it from a tarsus, which
        # swings a further millimetre ahead during the swing phase.
        dist = np.array([
            float(np.linalg.norm(P[:, :2] - s.position[:2], axis=1).min())
            for s in sources
        ])
        best = np.minimum(best, dist)
        if step % 100 == 0:
            track.append(dist.copy())
        if dist[0] < a.radius + 0.5 and found_at is None:
            found_at = step
            print(f"*** reached the {a.good} source at {step/1000:.2f} s ***")
            break

        if viewer.handle is not None and not viewer.sync(sim.time):
            break
    viewer.close()

    pos = sim.get_body_positions(FLY_NAME)[thorax]
    total = sum(states.values()) or 1
    print()
    for i, src in enumerate(sources):
        final = float(np.linalg.norm(pos[:2] - src.position[:2]))
        label = "good" if i == 0 else "bad "
        print(f"{label} ({src.odor:9s})  closest {best[i]:5.1f} mm   "
              f"final {final:5.1f} mm   started {np.linalg.norm(src.position[:2]):.0f}")
    print("behaviour: " + "  ".join(
        f"{k} {100*v/total:.0f}%" for k, v in states.items()))
    print("found the good source" if found_at is not None
          else "did not reach the good source")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--good", default="vinegar")
    ap.add_argument("--bad", default="geosmin")
    ap.add_argument("--distance", type=float, default=40.0)
    ap.add_argument("--bearing", type=float, default=25.0)
    ap.add_argument("--bad-distance", type=float, default=None)
    ap.add_argument("--bad-bearing", type=float, default=-25.0)
    ap.add_argument("--no-bad", action="store_true",
                    help="single source, no aversive odour")
    ap.add_argument("--decay", type=float, default=25.0)
    ap.add_argument("--radius", type=float, default=2.0)
    ap.add_argument("--ms", type=float, default=15000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--no-view", action="store_true")
    ap.add_argument("--untrained", action="store_true",
                    help="run the naive brain, before conditioning")
    ap.add_argument("--tumble", action="store_true",
                    help="use the hand-written run-and-tumble controller")
    ap.add_argument("--noise", type=float, default=0.03,
                    help="spontaneous turn variability")
    ap.add_argument("--noise-tail", type=float, default=3.0,
                    help="Student-t df; 30+ is Gaussian")
    ap.add_argument("--valence-scale", type=float, default=0.002,
                    help="valence at which the drive to act saturates")
    ap.add_argument("--trained", default="output/odor_trained.flyckpt")
    main(ap.parse_args())
