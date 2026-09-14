"""Central pattern generator standing in for VNC rhythm generation.

The connectome cannot be made to oscillate from wiring alone (see
docs/FINDINGS.md), so the rhythm is supplied here instead. This is not a
biological cheat: pattern generation really is a property of the ventral nerve
cord, and the brain really does control walking through low-dimensional
descending commands rather than by specifying joint trajectories.

What stays connectome-derived: the descending command that sets drive and
turning, the motor-neuron-to-joint mapping, and the proprioceptive feedback.
What is hand-written: the phase relationships between the six legs.

Six coupled Kuramoto oscillators, one per leg:

    dtheta_i/dt = omega + sum_j K_ij sin(theta_j - theta_i - phi_ij)

The matrix of offsets phi_ij *is* the gait. Coupling makes it self-correcting:
perturb one leg and the sine terms pull it back into formation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")

# The three leg pairs are geared differently. Neutral coxa_roll is 0.58 rad on
# the front leg, 1.52 on the middle and 2.34 on the hind, so an identical joint
# command sweeps the hind foot only 0.26 mm against the middle leg's 0.65 mm.
# These factors, measured from foot excursion during walking, equalise the
# stride so all six legs contribute.
# Measured per-leg gearing. Front, middle and hind coxae sit at very
# different neutral angles (0.58, 1.52, 2.34 rad), so an identical joint
# command sweeps the hind foot less than half as far as the middle one.
# These factors equalise the legs: all six contribute and speed roughly
# doubles to ~15 mm/s, at the cost of roll (25 -> 62 deg) and lateral drift
# (0.06 -> 3.2 mm). They now live in GaitParams so the optimiser can tune
# them; these dicts are kept only as the source of the defaults.
STRIDE_GAIN = {"lf": 1.65, "lm": 1.00, "lh": 2.50,
               "rf": 1.65, "rm": 1.00, "rh": 2.50}
LIFT_GAIN = {"lf": 1.50, "lm": 1.00, "lh": 2.35,
             "rf": 1.50, "rm": 1.00, "rh": 2.35}

# Tripod: front and hind of one side step with the middle leg of the other.
TRIPOD = {"lf": 0.0, "lh": 0.0, "rm": 0.0,
          "rf": np.pi, "rh": np.pi, "lm": np.pi}

# Tetrapod: a travelling wave, four legs down at any moment.
TETRAPOD = {"lf": 0.0, "lm": 2 * np.pi / 3, "lh": 4 * np.pi / 3,
            "rf": np.pi, "rm": np.pi + 2 * np.pi / 3, "rh": np.pi + 4 * np.pi / 3}


@dataclass
class GaitParams:
    frequency: float = 14.0     # Hz, fly stepping is roughly 5-20 Hz
    coupling: float = 8.0       # rad/s, how hard legs pull into formation

    # Joint amplitudes, radians. Named for what the muscle actually does.
    # Stride amplitude has a narrow usable window. Past ~0.35 rad the feet
    # stop gripping through the sweep and the fly is dragged *backwards*:
    # measured -4.17 mm forward at protraction 0.6.
    protraction: float = 0.30   # coxa pitch: swing the leg fore and aft
    levation: float = 0.15      # femur pitch: lift the leg during swing
    extension: float = 0.30     # tibia pitch: extend into stance
    abduction: float = 0.0      # coxa roll: symmetric, see joint_targets()

    duty: float = 0.80          # fraction of the cycle spent in stance
    stance_direction: float = -1.0   # flip if the fly walks backwards
    # Stride asymmetry turns out to be a poor steering mechanism on this body:
    # it does nothing until the inner legs reverse, then the fly pivots on the
    # spot. Measured -1.7 deg at turn 0.5 but -233 deg at turn 1.0. Offsetting
    # the sweep *centre* instead gives a clean proportional response
    # (+84 to -78 deg across the command range) while forward travel holds up.
    turn_gain: float = 0.0      # stride asymmetry per unit turn command
    turn_bias: float = 0.25     # rad of sweep-centre offset per unit turn
    amplitude_floor: float = 0.45  # stride length retained at zero drive

    # Per-leg gearing, previously hard-coded module constants. They multiply
    # the sweep and lift commands, so they are the strongest lever on body
    # roll there is -- and the optimiser could not see them. Middle legs are
    # the reference at 1.0; only front and hind are free.
    stride_front: float = 1.65
    stride_hind: float = 2.50
    lift_front: float = 1.50
    lift_hind: float = 2.35

    # Arc compensation. Sweeping `coxa_roll` alone swings the foot on a
    # circle about the coxa, so the further it sweeps the higher it rises at
    # the ends of the stroke: measured, a sweep that grows from 0.91 to
    # 1.31 mm of fore-aft travel drags vertical travel from 0.24 to 0.53 mm.
    # Fore-aft and lift are therefore not independent knobs, and no amount of
    # reducing `levation` flattens the stroke. This subtracts the geometric
    # rise -- proportional to the square of the sweep -- from the lift command,
    # which is the cheap version of solving the leg's inverse kinematics.
    arc_comp: float = 0.0


class CPG:
    """Six coupled oscillators driving joint targets."""

    def __init__(self, params: GaitParams | None = None, gait=TRIPOD):
        self.p = params or GaitParams()
        self.offsets = np.array([gait[leg] for leg in LEGS])
        self.phase = self.offsets.copy()
        self.drive = 1.0
        self.turn = 0.0

    def step(self, dt_s: float, drive: float = 1.0, turn: float = 0.0):
        """Advance the oscillators. `drive` scales frequency, `turn` steers."""
        p = self.p
        omega = 2 * np.pi * p.frequency * max(drive, 0.0)

        # Kuramoto coupling toward the target phase offsets.
        diff = (self.phase[None, :] - self.phase[:, None]
                - (self.offsets[None, :] - self.offsets[:, None]))
        coupling = p.coupling * np.sin(diff).sum(axis=1) / len(LEGS)

        self.phase = (self.phase + dt_s * (omega + coupling)) % (2 * np.pi)
        self.turn = turn
        self.drive = drive

    def joint_targets(self) -> dict[str, float]:
        """Per-joint angular offsets from the neutral pose.

        Stance occupies `duty` of the cycle: the foot is planted and sweeps
        backward, propelling the body. Swing is the remainder: the leg lifts,
        swings forward, and sets down.
        """
        p = self.p
        out = {}
        for i, leg in enumerate(LEGS):
            phase = self.phase[i] / (2 * np.pi)          # 0..1
            side = 1.0 if leg[0] == "l" else -1.0

            # Stride shrinks on the inside of a turn, and may go negative so
            # the inner legs step backwards and the fly pivots. Clipping this
            # at zero left a dead zone: only saturated turn commands steered
            # at all, so fine corrections did nothing.
            stride = 1.0 - np.clip(side * self.turn * p.turn_gain, -1.8, 1.8)

            if phase < p.duty:
                # Stance: foot planted, sweeping front to back. Getting this
                # sign backwards makes the fly moonwalk.
                s = phase / p.duty
                fore_aft = (0.5 - s) * 2.0
                lift = 0.0
            else:
                # Swing: foot lifts and returns to the front.
                s = (phase - p.duty) / (1.0 - p.duty)
                fore_aft = (s - 0.5) * 2.0
                lift = np.sin(np.pi * s)

            fore_aft *= p.stance_direction

            # Measured with mj_forward: coxa_roll moves the foot fore-aft with
            # almost no vertical coupling, coxa_yaw moves it vertically with
            # none fore-aft. The per-leg gains equalise front/middle/hind,
            # whose gearing differs by up to 1.7x.
            # A constant sweep-centre offset, opposite on each side, adds a
            # steady turning moment on top of the stride asymmetry.
            bias = side * self.turn * p.turn_bias
            # Stride amplitude only partly follows the drive. Frequency
            # already scales with it, so making amplitude scale too made speed
            # go as drive squared: at the 0.35 drive the wall reflex uses, the
            # fly was managing 12% of its top speed and scrabbling on the
            # spot. Real flies change step frequency far more than step length
            # when they slow down.
            amplitude = p.amplitude_floor + (1.0 - p.amplitude_floor) * self.drive
            stride_gain = {"f": p.stride_front, "m": 1.0,
                           "h": p.stride_hind}[leg[1]]
            lift_gain = {"f": p.lift_front, "m": 1.0, "h": p.lift_hind}[leg[1]]
            sweep = fore_aft * p.protraction * stride * amplitude * stride_gain
            out[f"{leg}/coxa_roll"] = sweep + bias
            out[f"{leg}/coxa_yaw"] = (lift * p.levation * lift_gain
                                      - p.arc_comp * 0.5 * sweep * sweep)
            out[f"{leg}/femur_pitch"] = p.abduction
            out[f"{leg}/tibia_pitch"] = p.extension * (1.0 - 0.5 * lift)
        return out


class CPGWalker:
    """Drives a NeuroMechFly body from the CPG, with an optional viewer."""

    def __init__(self, sim, fly, params: GaitParams | None = None,
                 gait=TRIPOD, viewer=None):
        from flygym.compose import ActuatorType

        from .body import joint_dof_index, neutral_targets

        self.sim, self.fly = sim, fly
        self.actuator_type = ActuatorType.POSITION
        self.dof_index = joint_dof_index(fly, self.actuator_type)
        self.neutral = neutral_targets(fly, self.actuator_type)
        self.cpg = CPG(params, gait)
        self.viewer = viewer
        self.trace = []

    def run(self, duration_ms: float, *, drive: float = 1.0, turn: float = 0.0,
            control_dt_ms: float = 1.0):
        from .body import FLY_NAME

        steps = max(1, int(round(control_dt_ms / (self.sim.timestep * 1000))))
        for _ in range(int(duration_ms / control_dt_ms)):
            self.cpg.step(control_dt_ms / 1000.0, drive=drive, turn=turn)

            targets = self.neutral.copy()
            for key, value in self.cpg.joint_targets().items():
                for slot in self.dof_index.get(key, []):
                    targets[slot] += value
            self.sim.set_actuator_inputs(FLY_NAME, self.actuator_type, targets)

            for _ in range(steps):
                self.sim.step()

            self.trace.append(self.sim.get_body_positions(FLY_NAME)[1].copy())
            if self.viewer is not None and not self.viewer.sync(self.sim.time):
                break

    def displacement(self) -> float:
        if len(self.trace) < 2:
            return 0.0
        return float(np.linalg.norm(self.trace[-1][:2] - self.trace[0][:2]))

    def path_length(self) -> float:
        p = np.array(self.trace)
        return float(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).sum())
