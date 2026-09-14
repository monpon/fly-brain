"""Inverse kinematics for the legs, so the foot follows a path we choose.

Driving each joint with its own sine wave makes the foot trace whatever curve
those happen to produce -- an arc, not a flat stroke. The foot then lifts or
digs in mid-stance, which is why longer strides made the fly roll harder and
move *slower*: measured 96 degrees of roll at 0.8 rad stride.

This module inverts the problem. The gait specifies where the foot should be;
the joint angles are solved for.

Rather than deriving forward kinematics from the model's joint conventions --
which differ per leg and are easy to get subtly wrong -- the mapping is
measured. `mj_forward` evaluates kinematics without stepping physics, so a
grid of joint offsets can be swept in milliseconds and then inverted by
nearest-neighbour lookup with local refinement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")


@dataclass
class LegChart:
    """Measured map from joint offsets to foot position, for one leg."""

    leg: str
    offsets: np.ndarray      # (n, 2) femur_pitch, tibia_pitch offsets
    feet: np.ndarray         # (n, 2) foot (fore, vert) relative to the coxa
    home: np.ndarray         # foot position at zero offset

    def solve(self, fore: float, vert: float) -> tuple[float, float]:
        """Joint offsets that put the foot nearest the requested position."""
        target = np.array([fore, vert])
        d = np.linalg.norm(self.feet - target, axis=1)
        return tuple(self.offsets[int(np.argmin(d))])

    def reach(self) -> dict:
        return {"fore": (float(self.feet[:, 0].min()), float(self.feet[:, 0].max())),
                "vert": (float(self.feet[:, 1].min()), float(self.feet[:, 1].max()))}


def chart_legs(sim, fly, *, grid: int = 41, span: float = 1.2) -> dict[str, LegChart]:
    """Measure foot position over a grid of femur/tibia offsets, per leg.

    Uses `mujoco.mj_forward`, which resolves body positions from joint angles
    without advancing the simulation, so this costs milliseconds and cannot
    be corrupted by the fly falling over while we measure.
    """
    import mujoco

    from .body import joint_dof_index, neutral_targets

    model, data = sim.mj_model, sim.mj_data
    index = joint_dof_index(fly)
    neutral = neutral_targets(fly)

    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
             for i in range(model.nbody)]

    # Actuator slot -> qpos address. Go through the DOF definitions rather
    # than actuator_trnid, because the adhesion actuators transmit to bodies
    # and would index the joint table out of range.
    from flygym.compose import ActuatorType

    order = fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
    qadr = {}
    for slot, dof in enumerate(order):
        joint = fly.jointdof_to_mjcfjoint[dof]
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint.name)
        if jid >= 0:
            qadr[slot] = model.jnt_qposadr[jid]

    saved = data.qpos.copy()
    charts = {}
    steps = np.linspace(-span, span, grid)

    for leg in LEGS:
        femur_slots = index.get(f"{leg}/femur_pitch", [])
        tibia_slots = index.get(f"{leg}/tibia_pitch", [])
        coxa_id = names.index(f"fly/{leg}_coxa")
        foot_id = names.index(f"fly/{leg}_tarsus5")
        if not femur_slots or not tibia_slots:
            continue

        offsets, feet = [], []
        for df in steps:
            for dt in steps:
                data.qpos[:] = saved
                for slot in femur_slots:
                    if slot in qadr:
                        data.qpos[qadr[slot]] = neutral[slot] + df
                for slot in tibia_slots:
                    if slot in qadr:
                        data.qpos[qadr[slot]] = neutral[slot] + dt
                mujoco.mj_forward(model, data)
                rel = data.xpos[foot_id] - data.xpos[coxa_id]
                offsets.append((df, dt))
                feet.append((rel[0], rel[2]))     # fore-aft and vertical

        offsets = np.array(offsets)
        feet = np.array(feet)
        home = feet[np.argmin(np.linalg.norm(offsets, axis=1))]
        charts[leg] = LegChart(leg, offsets, feet, home)

    data.qpos[:] = saved
    mujoco.mj_forward(model, data)
    return charts


def calibrate_gains(sim, fly, *, delta: float = 0.30) -> dict[str, dict]:
    """Measure how far each leg's foot moves per radian of joint command.

    `coxa_roll` sweeps the foot fore-aft and `coxa_yaw` lifts it, but the
    gearing differs per leg pair because their neutral coxa angles differ
    (0.58 rad front, 1.52 middle, 2.34 hind). Commanding the same angle to all
    six therefore moves the hind foot less than half as far as the middle one.

    Measuring mm-per-radian lets the gait be specified in millimetres, so every
    leg takes the same size step regardless of its geometry.
    """
    import mujoco

    from flygym.compose import ActuatorType

    from .body import joint_dof_index, neutral_targets

    model, data = sim.mj_model, sim.mj_data
    index = joint_dof_index(fly)
    neutral = neutral_targets(fly)
    order = fly.get_actuated_jointdofs_order(ActuatorType.POSITION)

    qadr = {}
    for slot, dof in enumerate(order):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                fly.jointdof_to_mjcfjoint[dof].name)
        if jid >= 0:
            qadr[slot] = model.jnt_qposadr[jid]

    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
             for i in range(model.nbody)]
    saved = data.qpos.copy()

    def foot_at(leg, key, offset):
        data.qpos[:] = saved
        for slot in index.get(f"{leg}/{key}", []):
            if slot in qadr:
                data.qpos[qadr[slot]] = neutral[slot] + offset
        mujoco.mj_forward(model, data)
        return (data.xpos[names.index(f"fly/{leg}_tarsus5")]
                - data.xpos[names.index(f"fly/{leg}_coxa")]).copy()

    gains = {}
    for leg in LEGS:
        a, b = foot_at(leg, "coxa_roll", -delta), foot_at(leg, "coxa_roll", delta)
        fore_per_rad = (b[0] - a[0]) / (2 * delta)
        a, b = foot_at(leg, "coxa_yaw", -delta), foot_at(leg, "coxa_yaw", delta)
        vert_per_rad = (b[2] - a[2]) / (2 * delta)
        gains[leg] = {"fore_per_rad": float(fore_per_rad),
                      "vert_per_rad": float(vert_per_rad)}

    data.qpos[:] = saved
    mujoco.mj_forward(model, data)
    return gains
