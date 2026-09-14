"""Score a gait, so it can be optimised instead of hand-tuned.

Walking is a control problem with a measurable objective, which makes it a
good target for black-box optimisation: propose gait parameters, simulate a
few seconds, score the result. This is how the real fly's gait was tuned too,
by evolution against survival, with a considerably larger compute budget.

The fitness deliberately penalises instability rather than rewarding raw
distance. A fly that lunges forward while rolling 50 degrees is not walking,
and without the penalties the optimiser finds exactly that.
"""

from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("MUJOCO_GL", "glfw")

# First pass used much weaker penalties and the optimiser exploited them: it
# found a gait with 3 degrees of leg lift that shuffles at 12 mm/s while rolling
# 47 degrees. Technically optimal, visibly awful. Smoothness now costs enough
# that scraping along is not worth it.
# Second pass: the winner of the first one steps at 24 Hz with a duty factor
# of 0.51 and rocks the body through **62 degrees of roll** while covering
# 15.4 mm/s. It is fast and it looks awful, because rolling is free distance
# under the old weights -- 62 degrees only cost 15.6 of a 23.3 fitness. Real
# Drosophila walking at this speed steps at 10-16 Hz and rolls a few degrees.
# Roll, pitch and bounce now cost enough that the optimiser cannot buy speed
# with them, and `speed_std` penalises the lurching that makes a gait look
# wrong even when the average velocity is fine.
FORWARD_WEIGHT = 1.0
YAW_PENALTY = 0.20      # per degree of heading drift
ROLL_PENALTY = 0.60     # per degree of roll excursion
LATERAL_PENALTY = 1.50  # per mm of sideways drift
HEIGHT_PENALTY = 8.0    # per mm of body-height standard deviation
PITCH_PENALTY = 0.40    # per degree of pitch excursion
SPEED_STD_PENALTY = 0.25  # per mm/s of stride-to-stride speed variation

# Shape of the step, stated directly instead of hoped for. Penalties are
# hinges: no cost while the foot stays inside a fly-like envelope, steep
# outside it, so the optimiser cannot trade a good-looking step for speed.
LIFT_TARGET_MM = 0.30     # vertical foot travel a real fly does not exceed
STRIDE_TARGET_MM = 1.40   # fore-aft travel it should at least reach
LIFT_PENALTY = 40.0       # per mm of excess lift
SHORT_STRIDE_PENALTY = 12.0  # per mm of missing stride


def evaluate(params, *, ms: float = 2000, seed: int = 0,
             settle_steps: int = 500, detail: bool = False,
             kp: float = 50.0):
    """Simulate one gait and return its fitness (higher is better)."""
    import mujoco
    from flygym import Simulation
    from flygym.compose import ActuatorType, FlatGroundWorld
    from flygym.utils.math import Rotation3D

    from .body import build_fly, joint_dof_index, neutral_targets
    from .cpg import CPG

    world = FlatGroundWorld()
    fly = build_fly(vision=False, adhesion=True, kp=kp)
    world.add_fly(fly, np.array([0.0, 0.0, 1.2]),
                  Rotation3D("quat", [1.0, 0.0, 0.0, 0.0]),
                  add_ground_contact_sensors=False)
    sim = Simulation(world)
    sim.reset()

    index = joint_dof_index(fly)
    neutral = neutral_targets(fly)
    cpg = CPG(params)

    # get_body_positions / get_body_rotations are indexed by the fly's own
    # body-segment order, NOT by model body id. Looking these up in the
    # model's name list returns leg segments -- which is what this file did,
    # so every "body roll" and "body height" it ever reported was the roll and
    # height of a tarsus. See docs/KNOWN_ISSUES.md.
    order = [seg.name for seg in fly.get_bodysegs_order()]
    thorax = order.index("c_thorax")
    front = [order.index(f"{l}_coxa") for l in ("lf", "rf")]
    hind = [order.index(f"{l}_coxa") for l in ("lh", "rh")]
    feet = [order.index(f"{l}_tarsus5")
            for l in ("lf", "lm", "lh", "rf", "rm", "rh")]

    for _ in range(settle_steps):
        sim.step()

    pos = sim.get_body_positions("fly")
    start = pos[thorax].copy()
    forward = pos[front].mean(axis=0) - pos[hind].mean(axis=0)
    forward[2] = 0.0
    forward /= np.linalg.norm(forward) + 1e-9
    lateral = np.array([-forward[1], forward[0], 0.0])

    heights, quats, track, foot_track = [], [], [], []
    inner = max(1, int(round(0.001 / sim.timestep)))
    for _ in range(int(ms)):
        cpg.step(0.001, drive=1.0, turn=0.0)
        targets = neutral.copy()
        for key, value in cpg.joint_targets().items():
            for slot in index.get(key, []):
                targets[slot] += value
        sim.set_actuator_inputs("fly", ActuatorType.POSITION, targets)
        for _ in range(inner):
            sim.step()
        p = sim.get_body_positions("fly")
        heights.append(p[thorax][2])
        track.append(p[thorax][:2].copy())
        rel = p[feet] - p[thorax]
        foot_track.append(np.stack([rel @ forward, rel[:, 2]], axis=1))
        quats.append(np.asarray(sim.get_body_rotations("fly")[thorax]).copy())

    end = sim.get_body_positions("fly")[thorax]
    delta = end - start
    Q = np.array(quats)
    w, x, y, z = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]
    yaw = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    roll = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))

    # Instantaneous speed, smoothed over 20 ms so this measures lurching
    # between strides rather than the within-stride oscillation every legged
    # gait has.
    xy = np.asarray(track)
    win = 20
    if len(xy) > 2 * win:
        step = np.linalg.norm(xy[win:] - xy[:-win], axis=1) / (win / 1000.0)
        speed_std = float(np.std(step))
    else:
        speed_std = 0.0

    # How the foot actually moves: fore-aft travel against vertical travel,
    # per leg, in millimetres. This is the quantity a gait is judged on by
    # eye. A real fly strides 1.5-2.5 mm and lifts 0.1-0.3 mm, a ratio near
    # 10:1; a gait that lifts almost as much as it advances looks like
    # marching on the spot however fast it happens to travel.
    F = np.asarray(foot_track)                      # (t, 6 legs, 2)
    stride_mm = float(np.ptp(F[:, :, 0], axis=0).mean())
    lift_mm = float(np.ptp(F[:, :, 1], axis=0).mean())

    metrics = {
        "stride_mm": stride_mm,
        "lift_mm": lift_mm,
        "foot_ratio": stride_mm / max(lift_mm, 1e-6),
        "speed_std": speed_std,
        "forward": float(delta @ forward),
        "lateral": float(abs(delta @ lateral)),
        "yaw": float(abs(yaw[-1] - yaw[0])),
        "roll": float(np.ptp(roll)),
        "height_std": float(np.std(heights)),
        "pitch": float(np.ptp(pitch)),
    }
    metrics["speed"] = metrics["forward"] / (ms / 1000.0)
    metrics["fitness"] = (
        FORWARD_WEIGHT * metrics["forward"]
        - YAW_PENALTY * metrics["yaw"]
        - ROLL_PENALTY * metrics["roll"]
        - LATERAL_PENALTY * metrics["lateral"]
        - HEIGHT_PENALTY * metrics["height_std"]
        - PITCH_PENALTY * metrics["pitch"]
        - SPEED_STD_PENALTY * metrics["speed_std"]
        - LIFT_PENALTY * max(0.0, lift_mm - LIFT_TARGET_MM)
        - SHORT_STRIDE_PENALTY * max(0.0, STRIDE_TARGET_MM - stride_mm)
    )
    return metrics if detail else metrics["fitness"]
