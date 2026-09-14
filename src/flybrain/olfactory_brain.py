"""Steer from the connectome olfactory circuit instead of a hand-written law.

A proportional controller on the antennal difference works, but it is an
engineer's answer: the fly already has the machinery. Odorant receptor neurons
feed the antennal lobe, projection neurons carry 53 glomerular channels into
the mushroom body, Kenyon cells recode them sparsely, and MBONs report a
behavioural valence -- approach or avoid -- that dopamine can retune.

The projection neurons are cleanly lateralised in male-cns v1.0: 139 on the
left, 137 on the right, each covering essentially the full glomerular set. So
the obvious plan is to run the two antennae through the circuit *separately*
and compare the outputs, getting both of a controller's requirements -- what to
do, and which way to go -- out of one structure. That plan does not survive
contact with the circuit.

Measured: the mushroom body is **completely concentration-invariant**. Driving
the same glomeruli from PN rate 1 to 300 leaves 202 Kenyon cells active and the
valence at +0.003712, unchanged to six decimals. Change *which* glomeruli fire
and the valence moves (0.0031 / 0.0020 / 0.0011). That is the circuit doing its
job -- the sparse KC code with APL feedback normalises intensity away so the
animal recognises a smell however faint it is -- but it means the mushroom body
cannot possibly tell you which antenna smells more. It has thrown that away.

So the labour is divided the way it is in the animal:

* **Mushroom body -> whether.** A learned, concentration-invariant valence:
  is this odour worth approaching? Training flips it, and that is the part the
  61,210 plastic KC->MBON synapses control.
* **Antennal lobe -> which way.** The raw left/right difference, before the
  Kenyon cells discard it.

The valence gates the direction. Turn toward the stronger antenna when the
valence is positive, away from it when negative -- so a single rule covers
approach and avoidance, and what the fly has *learned* decides which it does.

Two odours at once
------------------
Because the valence is learned per odour and the direction is read per
glomerular channel, nothing about this is specific to a single smell. Give the
fly a rewarded odour and a punished one in the same arena and each channel
contributes a turn weighted by its own learned sign: toward the good source,
away from the bad one, summed. Approach and avoidance are the same equation
with opposite signs, which is what makes a depression-only learning rule
enough to produce both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import odors as ODORS


@dataclass
class BrainNavParams:
    turn_gain: float = 120.0       # bilateral difference -> steering command
    drive: float = 1.0             # forward drive while a smell is present
    casting_drive: float = 0.7     # forward drive with nothing to smell
    cast_turn: float = 0.6         # sweep amplitude when searching
    cast_period_ms: float = 900.0
    lost_level: float = 1e-4       # below this, assume no odour
    pn_gain: float = 30.0          # concentration -> projection neuron rate
    valence_scale: float = 0.002   # valence at which the drive to act saturates
    noise: float = 0.03
    # A sharp turn has to be a slow turn on this body: at 15 mm/s the turning
    # circle is ~23 mm across, wider than a corridor. Real flies also drop
    # their walking speed sharply during saccadic turns.
    turn_slowdown: float = 0.0     # fraction of drive given up at full turn


class BrainNavigator:
    """Steers from mushroom-body valence and antennal-lobe laterality."""

    def __init__(self, fly, mb, odor_names: list[str] | None = None,
                 params: BrainNavParams | None = None, seed: int = 0):
        self.p = params or BrainNavParams()
        self.fly = fly
        self.mb = mb
        self.rng = np.random.default_rng(seed)
        self.odor_names = list(odor_names or ["vinegar"])

        # What each odour is worth. Evaluated once, because the mushroom body
        # is concentration-invariant: the valence of a smell does not change
        # as the fly walks toward it, only the antennal difference does. Call
        # `revalue()` after any training to refresh these.
        self.valences = np.zeros(len(self.odor_names))
        self.revalue()

        self.cast_timer = 0.0
        self.cast_sign = 1.0
        self.state = "search"
        self.valence = 0.0

    def revalue(self) -> np.ndarray:
        """Re-read the mushroom body's verdict on each odour."""
        self.valences = np.array([
            self.fly.respond(ODORS.pn_vector(self.mb, name, 1.0, self.p.pn_gain))
            for name in self.odor_names
        ])
        return self.valences

    def report(self) -> str:
        return "   ".join(
            f"{n} {v:+.5f} ({'approach' if v >= 0 else 'avoid'})"
            for n, v in zip(self.odor_names, self.valences)
        )

    def update(self, samples, dt_ms: float) -> tuple[float, float]:
        """`samples` is (n_odours, 2): left and right concentration of each."""
        p = self.p
        samples = np.atleast_2d(np.asarray(samples, dtype=float))
        left = samples[:, 0]
        right = samples[:, 1]
        total = float(samples.sum())

        self.cast_timer += dt_ms
        if self.cast_timer >= p.cast_period_ms:
            self.cast_timer = 0.0
            self.cast_sign = -self.cast_sign

        if total < p.lost_level * 2:
            self.state = "search"
            self.valence = 0.0
            return float(np.clip(p.cast_turn * self.cast_sign, -1, 1)), \
                p.casting_drive

        # Antennal lobe: which side is each channel on? This is the
        # information the Kenyon cells discard, so it is read before them,
        # and read per odour because the glomeruli stay separate.
        bilateral = (left - right) / total

        # Mushroom body: what is each smell worth? Positive turn steers right,
        # so a stronger *left* antenna needs a negative turn to approach it --
        # and a positive one to flee it. One equation, both behaviours.
        appetitive = np.tanh(self.valences / max(p.valence_scale, 1e-9))
        turn = -p.turn_gain * float(appetitive @ bilateral)
        turn += self.rng.normal(0.0, p.noise)

        share = (left + right) / total
        self.valence = float(appetitive @ share)
        self.state = "approach" if self.valence >= 0 else "avoid"
        turn = float(np.clip(turn, -1, 1))
        drive = p.drive * (1.0 - p.turn_slowdown * abs(turn))
        return turn, drive
