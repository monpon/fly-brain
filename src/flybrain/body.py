"""Build the NeuroMechFly body and world.

FlyGym 2.x uses a composition API: a bare `NeuroMechFly()` has a skeleton but
no joints, actuators, or sensors until you add them. This module wires up a
standard walking-and-seeing fly so the rest of the project has one entry point.
"""

import numpy as np
from flygym import Simulation
from flygym.anatomy import ActuatedDOFPreset, AxisOrder, JointPreset, Skeleton
from flygym.compose import (
    ActuatorType,
    FlatGroundWorld,
    KinematicPosePreset,
    NeuroMechFly,
)
from flygym.utils.math import Rotation3D

FLY_NAME = "fly"

# Leg coxae are the ground-contact subtree roots FlyGym sensorizes by default,
# and that path fails to compile on flygym 2.1.0 + mujoco 3.9.0. We build the
# contact *geoms* (so physics is unaffected) but skip the *sensors*; read foot
# contact off the MuJoCo contact list instead. See docs/KNOWN_ISSUES.md.
GROUND_CONTACT_SENSORS = False


def build_fly(*, vision: bool = True, adhesion: bool = True,
              kp: float = 50.0) -> NeuroMechFly:
    """A fly with actuated legs, and optionally eyes and tarsal adhesion.

    `kp` is the position-actuator gain. FlyGym's default leaves the actuators
    too weak to move the legs against joint stiffness: a commanded 0.5 rad
    step produced only 0.037 rad of motion. At kp=50 tracking is ~84%.
    """
    fly = NeuroMechFly(name=FLY_NAME)

    skeleton = Skeleton(
        axis_order=AxisOrder.YAW_PITCH_ROLL,
        joint_preset=JointPreset.LEGS_ONLY,
    )
    fly.add_joints(skeleton, KinematicPosePreset.NEUTRAL)

    dofs = skeleton.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ONLY)
    fly.add_actuators(dofs, ActuatorType.POSITION, KinematicPosePreset.NEUTRAL,
                      kp=kp)

    if vision:
        fly.add_vision()
    if adhesion:
        fly.add_leg_adhesion()

    return fly


def build_simulation(
    fly: NeuroMechFly | None = None,
    *,
    spawn_height: float = 0.5,
) -> tuple[Simulation, NeuroMechFly]:
    """Put a fly on flat ground and return a ready-to-step simulation."""
    fly = fly if fly is not None else build_fly()

    world = FlatGroundWorld()
    world.add_fly(
        fly,
        np.array([0.0, 0.0, spawn_height]),
        Rotation3D("quat", [1.0, 0.0, 0.0, 0.0]),
        add_ground_contact_sensors=GROUND_CONTACT_SENSORS,
    )

    sim = Simulation(world)
    sim.reset()
    return sim, fly


LEG_PREFIXES = ("lf", "lm", "lh", "rf", "rm", "rh")
_SEGMENT_TO_JOINT = {
    "coxa": "coxa", "trochanterfemur": "femur",
    "tibia": "tibia", "tarsus1": "tarsus",
}


def joint_dof_index(fly, actuator_type=None) -> dict[str, list[int]]:
    """Map "lf/tibia_pitch" style keys onto actuator slots.

    Both the connectome controller and the CPG address joints by this name, so
    they share one mapping and stay swappable.
    """
    from flygym.compose import ActuatorType

    actuator_type = actuator_type or ActuatorType.POSITION
    out: dict[str, list[int]] = {}
    for i, dof in enumerate(fly.get_actuated_jointdofs_order(actuator_type)):
        child, parent, axis = dof.child.name, dof.parent.name, dof.axis.value
        leg = child[:2] if child[:2] in LEG_PREFIXES else parent[:2]
        if leg not in LEG_PREFIXES:
            continue
        seg = _SEGMENT_TO_JOINT.get(child.split("_", 1)[-1])
        if seg:
            out.setdefault(f"{leg}/{seg}_{axis}", []).append(i)
    return out


def neutral_targets(fly, actuator_type=None):
    """Neutral actuator inputs, in the order the actuators expect."""
    import numpy as np
    from flygym.compose import ActuatorType

    actuator_type = actuator_type or ActuatorType.POSITION
    table = fly.jointdof_to_neutralaction_by_type[actuator_type]
    order = fly.get_actuated_jointdofs_order(actuator_type)
    return np.array([float(table[d]) for d in order], dtype=np.float64)


# -- collision proxies ----------------------------------------------------
# The NeuroMechFly body is made of meshes, and in this flygym/mujoco pairing
# those meshes generate contacts against the ground plane (through the 55
# explicit <pair> elements flygym emits) but against nothing else. Setting
# contype/conaffinity on them changes nothing: a fly dropped onto a maze wall
# falls straight through it and stands inside the wall, and `mj_geomDistance`
# reports the overlap the whole time. So the mesh-versus-box collider never
# runs for these geoms.
#
# Rather than fight that, the fly gets a handful of invisible primitive geoms
# -- spheres, which collide with boxes perfectly well. They carry no mass
# (`density=0`) so the dynamics the gait was tuned against are untouched, and
# their contype/conaffinity is chosen to match walls and nothing else, so the
# fly still cannot collide with itself.
#
# The antennal proxies are the wall *sensor*: they belong to the funiculus
# bodies, so their contacts show up in `get_bodysegment_contact_forces` for
# those segments, which is what `antennae.WallReflex` reads.
PROXY_CONTYPE, PROXY_CONAFFINITY = 2, 1

# Radii are chosen so the antennal proxies stick out furthest: standing, the
# funiculus proxies reach 1.54 mm forward of the fly's origin against the
# thorax's 1.28, so a wall is felt before the body hits it. That ordering is
# the whole point -- the reflex fires on the sensor, not on the collision.
WALL_PROXIES = {
    "c_thorax": (0.70, (0.0, 0.0, 0.0)),
    "c_head": (0.55, (0.0, 0.0, 0.0)),
    "c_abdomen3": (0.60, (0.0, 0.0, 0.0)),
    "c_abdomen5": (0.50, (0.0, 0.0, 0.0)),
    "l_funiculus": (0.45, (0.0, 0.0, 0.0)),
    "r_funiculus": (0.45, (0.0, 0.0, 0.0)),
}


def add_wall_proxies(fly, proxies: dict | None = None) -> int:
    """Give the fly massless primitive geoms that can hit a wall."""
    import mujoco

    proxies = proxies or WALL_PROXIES
    by_name = {seg.name: body for seg, body in fly.bodyseg_to_mjcfbody.items()}
    added = 0
    for name, (radius, pos) in proxies.items():
        body = by_name.get(name)
        if body is None:
            continue
        body.add_geom(
            name=f"wallproxy_{name}",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[radius, 0.0, 0.0],
            pos=list(pos),
            rgba=[1.0, 0.2, 0.2, 0.0],   # invisible
            density=0.0,
            contype=PROXY_CONTYPE,
            conaffinity=PROXY_CONAFFINITY,
        )
        added += 1
    return added
