"""Closed loop: connectome VNC <-> physics body, with no pattern generator.

    body  --proprioception-->  vnc_sensory
                                    |
                               VNC intrinsic  <-- descending command
                                    |
                                motor neurons
                                    |
    body  <----muscle activation----+

Everything the legs do is produced by spiking neurons read out of
male-cns v1.0. There is no oscillator, no gait table, no scripted trajectory.

Biological choices, all of which are assumptions the connectome does not make
for us:

* **Muscle low-pass.** A motor neuron spike does not move a joint; it releases
  calcium into a muscle that contracts and relaxes over tens of milliseconds.
  Each motor neuron therefore carries an activation variable that jumps on
  every spike and decays with `tau_muscle`. This is the single most important
  piece of realism here -- without it, joint commands are raw spike noise.

* **Antagonist pairs.** Muscles pull, they do not push. Net joint drive is
  extensor activation minus flexor activation, which is how an insect joint
  is actually controlled.

* **Equilibrium-point control.** Activation sets a *target angle* offset from
  the neutral pose rather than a raw torque. Insect muscle has high intrinsic
  stiffness, so commanding an equilibrium the limb springs toward is closer to
  the biology than commanding torque, and it does not require a stabilising
  controller the fly does not have.

* **Proprioceptive encoding.** Femoral chordotonal organ neurons report
  femur-tibia angle, hair plates report joints approaching their limits, and
  leg mechanosensors report ground contact load.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from brian2 import (Hz, Network, NeuronGroup, PoissonGroup, SpikeMonitor,
                    Synapses, defaultclock, mV, ms, network_operation, second)

from . import local_data as L, vnc
from .body import FLY_NAME

LEG_NERVE_TO_POSITION = {"ProLN": "f", "MesoLN": "m", "MetaLN": "h"}
PROPRIOCEPTIVE = {"chordotonal organ", "hair plate", "leg"}


@dataclass
class EmbodiedParams:
    # neuron
    v_rest: float = -52.0
    v_reset: float = -52.0
    v_threshold: float = -45.0
    tau_m: float = 20.0
    refractory: float = 2.2
    epsp_per_synapse: float = 0.275
    normalize_input_mv: float | None = 100.0

    # synaptic and intrinsic dynamics. Instantaneous voltage jumps give an
    # oscillator no delay structure to form around, and no adaptation means no
    # half-centre mechanism. These are the ingredients rhythm is built from.
    biophysical: bool = True
    tau_exc: float = 5.0          # ms, nicotinic acetylcholine
    tau_inh: float = 10.0         # ms, GABA-A / GluCl
    tau_adapt: float = 100.0      # ms, spike-frequency adaptation
    adapt_mv: float = 0.6         # mV of after-hyperpolarisation per spike

    # muscle
    tau_muscle: float = 30.0      # ms, calcium/contraction time constant
    joint_gain: float = 0.35      # radians of target offset per unit activation
    max_offset: float = 0.6       # radians, clamp on commanded deviation

    # proprioception
    sensory_drive_mv: float = 8.0  # peak depolarisation from a saturated sensor

    # loop
    control_dt: float = 1.0       # ms, sensorimotor update period
    timestep: float = 0.1         # ms, integration step


@dataclass
class SensoryMap:
    index: np.ndarray            # row in the weight matrix
    leg: np.ndarray              # "lf" ... "rh"
    modality: np.ndarray         # "angle" | "limit" | "load"


def _infer_side(neurons: pd.DataFrame, W: np.ndarray,
                motor_idx: np.ndarray, motor_side: np.ndarray) -> np.ndarray:
    """Side for sensory neurons, from annotation where present.

    Only ~18% of leg sensory neurons carry a somaSide (their somata sit in the
    periphery, outside the imaged volume). For the rest we take the side of the
    motor neurons they most strongly drive, which is the functionally relevant
    definition anyway.
    """
    sides = neurons["somaSide"].astype(str).str.lower().to_numpy()
    out = np.array(["?"] * len(neurons), dtype="<U1")
    out[sides == "l"] = "l"
    out[sides == "r"] = "r"

    unknown = np.flatnonzero(out == "?")
    if len(unknown) and len(motor_idx):
        left = motor_idx[motor_side == "l"]
        right = motor_idx[motor_side == "r"]
        reach = np.abs(W[unknown]) @ np.eye(W.shape[1])[:, :0] if False else None
        l_drive = np.abs(W[np.ix_(unknown, left)]).sum(axis=1)
        r_drive = np.abs(W[np.ix_(unknown, right)]).sum(axis=1)
        out[unknown] = np.where(l_drive >= r_drive, "l", "r")
    return out


def sensory_map(neurons: pd.DataFrame, W: np.ndarray,
                motor: vnc.MotorMap) -> SensoryMap:
    """Resolve leg proprioceptors onto (leg, modality)."""
    sc = neurons["superclass"].astype(str)
    nerve = neurons["entryNerve"].astype(str)
    sub = neurons["subclass"].astype(str)

    mask = (sc == "vnc_sensory") & nerve.isin(LEG_NERVE_TO_POSITION) \
        & sub.isin(PROPRIOCEPTIVE)
    idx = np.flatnonzero(mask.to_numpy())
    sel = neurons.iloc[idx]

    motor_side = np.array([j[0] for j in motor.joint])
    side = _infer_side(sel, W, motor.index, motor_side)

    pos = sel["entryNerve"].astype(str).map(LEG_NERVE_TO_POSITION).to_numpy()
    leg = np.char.add(side, pos.astype(str))

    modality = np.where(
        sel["subclass"].astype(str).to_numpy() == "chordotonal organ", "angle",
        np.where(sel["subclass"].astype(str).to_numpy() == "hair plate",
                 "limit", "load"))

    return SensoryMap(index=idx, leg=leg, modality=modality)


class EmbodiedFly:
    """A connectome VNC wired to a NeuroMechFly body, stepped in lockstep."""

    def __init__(self, sim, fly, W, neurons, params: EmbodiedParams | None = None,
                 viewer=None):
        from flygym.compose import ActuatorType

        self.p = params or EmbodiedParams()
        self.sim, self.fly, self.viewer = sim, fly, viewer
        self._stop = False
        self.W, self.neurons = W, neurons
        self.motor = vnc.motor_map(neurons)
        self.sensory = sensory_map(neurons, W, self.motor)

        self.actuator_type = ActuatorType.POSITION
        self.dof_order = fly.get_actuated_jointdofs_order(self.actuator_type)
        self.neutral = self._neutral_targets()
        self._build_dof_index()
        self._build_network()
        self.trace = []

    # ---------- body side ----------

    def _neutral_targets(self) -> np.ndarray:
        table = self.fly.jointdof_to_neutralaction_by_type[self.actuator_type]
        return np.array([float(table[d]) for d in self.dof_order], dtype=np.float64)

    def _build_dof_index(self):
        """Map each 'lf/tibia_pitch' key onto its actuator slot."""
        joint_of = {}
        for i, d in enumerate(self.dof_order):
            child = d.child.name              # e.g. lf_tibia
            parent = d.parent.name
            axis = d.axis.value               # pitch / roll / yaw
            legs = ("lf", "lm", "lh", "rf", "rm", "rh")
            leg = child[:2] if child[:2] in legs else parent[:2]
            seg = child.split("_", 1)[-1]
            key = None
            if seg == "coxa":
                key = f"coxa_{axis}"
            elif seg == "trochanterfemur":
                key = f"femur_{axis}"
            elif seg == "tibia":
                key = f"tibia_{axis}"
            elif seg == "tarsus1":
                key = f"tarsus_{axis}"
            if key:
                joint_of.setdefault(f"{leg}/{key}", []).append(i)
        self.dof_index = joint_of

    # ---------- neural side ----------

    def _build_network(self):
        p = self.p
        defaultclock.dt = p.timestep * ms

        if p.biophysical:
            eqs = """
            dv/dt = (v_rest - v + drive + I_e + I_i - w_adapt) / tau_m : volt (unless refractory)
            dI_e/dt = -I_e / tau_exc : volt
            dI_i/dt = -I_i / tau_inh : volt
            dw_adapt/dt = -w_adapt / tau_adapt : volt
            da/dt = -a / tau_muscle : 1
            drive : volt
            """
            reset = f"v = v_reset; a += 1; w_adapt += {p.adapt_mv} * mV"
        else:
            eqs = """
            dv/dt = (v_rest - v + drive) / tau_m : volt (unless refractory)
            da/dt = -a / tau_muscle : 1
            drive : volt
            """
            reset = "v = v_reset; a += 1"

        self.group = NeuronGroup(
            self.W.shape[0], eqs,
            threshold="v > v_threshold",
            reset=reset,
            refractory=p.refractory * ms,
            method="euler",
            namespace={
                "v_rest": p.v_rest * mV, "v_reset": p.v_reset * mV,
                "v_threshold": p.v_threshold * mV, "tau_m": p.tau_m * ms,
                "tau_muscle": p.tau_muscle * ms,
                "tau_exc": p.tau_exc * ms, "tau_inh": p.tau_inh * ms,
                "tau_adapt": p.tau_adapt * ms, "mV": mV,
            },
            name="vnc",
        )
        self.group.v = p.v_rest * mV

        weights = self.W.astype(np.float64) * p.epsp_per_synapse
        if p.normalize_input_mv is not None:
            total = np.abs(weights).sum(axis=0)
            scale = np.divide(p.normalize_input_mv, total,
                              out=np.ones_like(total), where=total > 0)
            weights *= scale[np.newaxis, :]

        pre, post = np.nonzero(weights)
        vals = weights[pre, post]
        if p.biophysical:
            # Excitatory and inhibitory input are filtered separately, with
            # their own transmitter time constants, instead of both landing as
            # an instantaneous step on the membrane.
            exc = vals > 0
            syn_e = Synapses(self.group, self.group, "w : volt",
                             on_pre="I_e_post += w", name="exc")
            syn_e.connect(i=pre[exc], j=post[exc])
            syn_e.w = vals[exc] * mV
            syn_i = Synapses(self.group, self.group, "w : volt",
                             on_pre="I_i_post += w", name="inh")
            syn_i.connect(i=pre[~exc], j=post[~exc])
            syn_i.w = vals[~exc] * mV
            syn = [syn_e, syn_i]
        else:
            s1 = Synapses(self.group, self.group, "w : volt", on_pre="v_post += w")
            s1.connect(i=pre, j=post)
            s1.w = vals * mV
            syn = [s1]

        self.spikes = SpikeMonitor(self.group)
        self.net = Network(self.group, *syn, self.spikes)

        @network_operation(dt=p.control_dt * ms)
        def sensorimotor():
            self._motor_to_body()
            self._step_body()
            self._body_to_sensory()

        self.net.add(sensorimotor)

    def command_descending(self, dn_type: str, rate: float, drive: float = 20.0):
        """Poisson drive onto a named descending neuron type."""
        targets = np.flatnonzero(
            self.neurons["type"].astype(str).to_numpy() == dn_type)
        if targets.size == 0:
            raise SystemExit(f"no descending neurons of type {dn_type}")
        src = PoissonGroup(len(targets), rate * Hz)
        inj = Synapses(src, self.group, on_pre=f"v_post += {drive} * mV")
        inj.connect(i=np.arange(len(targets)), j=targets)
        self.net.add(src, inj)
        return targets.size

    # ---------- the loop ----------

    def _motor_to_body(self):
        """Muscle activations -> joint equilibrium targets."""
        act = np.asarray(self.group.a[:])
        targets = self.neutral.copy()

        for key, slots in self.dof_index.items():
            mask = self.motor.joint == key
            if not mask.any():
                continue
            drive = float((act[self.motor.index[mask]] * self.motor.sign[mask]).sum())
            drive /= max(mask.sum(), 1)
            offset = np.clip(drive * self.p.joint_gain,
                             -self.p.max_offset, self.p.max_offset)
            for slot in slots:
                targets[slot] += offset

        self.sim.set_actuator_inputs(FLY_NAME, self.actuator_type, targets)

    def _step_body(self):
        steps = max(1, int(round(self.p.control_dt / self.p.timestep)))
        for _ in range(steps):
            self.sim.step()

    def _body_to_sensory(self):
        """Joint angles and ground contact -> proprioceptor depolarisation."""
        angles = np.asarray(self.sim.get_joint_angles(FLY_NAME))
        drive = np.zeros(self.W.shape[0], dtype=np.float64)

        # Normalised femur-tibia angle per leg, and contact load per leg.
        legs = ("lf", "lm", "lh", "rf", "rm", "rh")
        angle_by_leg, limit_by_leg = {}, {}
        for leg in legs:
            slots = self.dof_index.get(f"{leg}/tibia_pitch", [])
            if slots:
                a = angles[slots[0]]
                angle_by_leg[leg] = float(np.tanh(a))
                limit_by_leg[leg] = float(min(1.0, abs(a) / 1.5))

        segs = [f"{leg}_tarsus5" for leg in legs]
        try:
            forces = self.sim.get_bodysegment_contact_forces(
                FLY_NAME, segs, ground_only=False)
            load = np.linalg.norm(np.asarray(forces), axis=1)
            load = load / (load.max() + 1e-9)
        except Exception:
            load = np.zeros(len(legs))
        load_by_leg = dict(zip(legs, load))

        s = self.sensory
        for modality, table in (("angle", angle_by_leg),
                                ("limit", limit_by_leg),
                                ("load", load_by_leg)):
            m = s.modality == modality
            if not m.any():
                continue
            vals = np.array([table.get(l, 0.0) for l in s.leg[m]])
            drive[s.index[m]] = np.abs(vals) * self.p.sensory_drive_mv

        self.group.drive = drive * mV

        self.trace.append(self.sim.get_body_positions(FLY_NAME)[1].copy())
        if self.viewer is not None and not self.viewer.sync(self.sim.time):
            self._stop = True

    def run(self, duration_ms: float):
        self.net.run(duration_ms * ms)

    def displacement(self) -> float:
        if len(self.trace) < 2:
            return 0.0
        return float(np.linalg.norm(self.trace[-1][:2] - self.trace[0][:2]))
