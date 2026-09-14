"""Drive the fly's joints from connectome motor neurons, with no hand-written CPG.

The brain does not connect to muscles. In male-cns v1.0, 121,567 brain
intrinsic neurons make 80 synapses onto the 708 motor neurons -- annotation
noise. Everything the brain sends to the legs passes through 1,314 descending
neurons, and motor neurons receive roughly ten times more synaptic input from
the 13,161 VNC intrinsic neurons than from descending commands.

So the pattern generator is the VNC, and this module builds it from the
connectome rather than substituting coupled oscillators for it:

    descending (1,314) -> VNC intrinsic (13,161) -> motor (708) -> joints

Motor neurons are annotated by *muscle* ("Ti flexor MN"), while MuJoCo is
actuated by *joint*. Muscles come in antagonist pairs, so the mapping is
many-to-one and signed: net joint command = extensors - flexors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import local_data as L

# Which joint each muscle pulls on, and in which direction.
# +1 extends / elevates / promotes, -1 flexes / depresses / remotes.
MUSCLE_TO_JOINT: dict[str, tuple[str, float]] = {
    # femur-tibia joint
    "Ti extensor MN":                   ("tibia_pitch", +1.0),
    "Ti flexor MN":                     ("tibia_pitch", -1.0),
    "Acc. ti flexor MN":                ("tibia_pitch", -1.0),
    # trochanter-femur joint
    "Tr extensor MN":                   ("femur_pitch", +1.0),
    "Tr flexor MN":                     ("femur_pitch", -1.0),
    "Acc. tr flexor MN":                ("femur_pitch", -1.0),
    "Fe reductor MN":                   ("femur_roll",  -1.0),
    # thorax-coxa: rotation, pro/remotion, ad/abduction
    "Sternal anterior rotator MN":      ("coxa_yaw",    +1.0),
    "Sternal posterior rotator MN":     ("coxa_yaw",    -1.0),
    "Tergopleural/Pleural promotor MN": ("coxa_pitch",  +1.0),
    "Pleural remotor/abductor MN":      ("coxa_pitch",  -1.0),
    "Sternal adductor MN":              ("coxa_roll",   +1.0),
    "Sternotrochanter MN":              ("coxa_roll",   -1.0),
    "Tergotr. MN":                      ("coxa_roll",   -1.0),
    # tarsus
    "Ta levator MN":                    ("tarsus_pitch", +1.0),
    "Ta depressor MN":                  ("tarsus_pitch", -1.0),
    "ltm MN":                           ("tarsus_pitch", -1.0),
    "ltm1-tibia MN":                    ("tarsus_pitch", -1.0),
    "ltm2-femur MN":                    ("tarsus_pitch", -1.0),
}

# subclass -> leg position; somaSide -> body side. Together these give the
# NeuroMechFly segment prefix (lf, lm, lh, rf, rm, rh).
_POSITION = {"fl": "f", "ml": "m", "hl": "h"}

JOINT_KEYS = ["coxa_yaw", "coxa_pitch", "coxa_roll",
              "femur_pitch", "femur_roll", "tibia_pitch", "tarsus_pitch"]


@dataclass
class MotorMap:
    """Which motor neurons drive which joint, and with what sign."""

    neurons: pd.DataFrame          # one row per mapped motor neuron
    index: np.ndarray              # row in the network weight matrix
    joint: np.ndarray              # "lf/tibia_pitch" style key
    sign: np.ndarray               # +1 extensor, -1 flexor

    def joints(self) -> list[str]:
        return sorted(set(self.joint))

    def command(self, rates: np.ndarray) -> dict[str, float]:
        """Net signed drive per joint from motor neuron firing rates."""
        out = {}
        for key in self.joints():
            mask = self.joint == key
            out[key] = float((rates[self.index[mask]] * self.sign[mask]).sum())
        return out


def extract(include_brain: bool = False, sensory: bool = True) -> dict:
    """Pull the descending -> VNC -> motor subnetwork out of the connectome.

    `sensory` includes the 6,370 vnc_sensory neurons. Without them the loop is
    open: the VNC has no way to know what the body is doing.
    """
    ann = L.annotations()
    sc = ann["superclass"].astype(str)

    wanted = ["descending_neuron", "vnc_intrinsic", "vnc_motor"]
    if sensory:
        wanted += ["vnc_sensory"]
    if include_brain:
        wanted += ["cb_intrinsic"]
    sub = ann[sc.isin(wanted)].reset_index(drop=True)

    W, neurons = _matrix(sub)
    return {"W": W, "neurons": neurons}


def _matrix(neurons: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Signed weight matrix over an arbitrary neuron selection."""
    body_ids = neurons["bodyId"].tolist()
    index = {b: i for i, b in enumerate(body_ids)}

    sub = L.weights_between(body_ids)
    signs = L._signs(neurons["bodyId"])

    W = np.zeros((len(body_ids), len(body_ids)), dtype=np.float32)
    rows = sub["body_pre"].map(index).to_numpy()
    cols = sub["body_post"].map(index).to_numpy()
    W[rows, cols] = sub["weight"].to_numpy(np.float32) * signs[rows]
    return W, neurons


def motor_map(neurons: pd.DataFrame) -> MotorMap:
    """Resolve leg motor neurons onto NeuroMechFly joint DOFs."""
    rows, idx, joints, signs = [], [], [], []

    for i, row in neurons.iterrows():
        if str(row.get("superclass")) != "vnc_motor":
            continue
        pos = _POSITION.get(str(row.get("subclass")))
        side = str(row.get("somaSide", "")).lower()
        mapping = MUSCLE_TO_JOINT.get(str(row.get("type")))
        if pos is None or side not in ("l", "r") or mapping is None:
            continue

        joint_key, sign = mapping
        rows.append(row)
        idx.append(i)
        joints.append(f"{side}{pos}/{joint_key}")
        signs.append(sign)

    return MotorMap(
        neurons=pd.DataFrame(rows).reset_index(drop=True),
        index=np.array(idx, dtype=int),
        joint=np.array(joints),
        sign=np.array(signs, dtype=np.float32),
    )
