"""Odour-guided search, built from what walking flies actually do.

A naive controller would steer up the concentration gradient. Real flies do
not, and cannot: their antennae are 0.216 mm apart, so the left-right
difference across a shallow gradient is nearly noise. What Drosophila actually
use is a combination of three things:

* **Osmotropotaxis** -- a weak bias toward whichever antenna reads higher.
  Real, measurable, but not sufficient on its own.
* **Klinokinesis** -- turning rate depends on whether odour is *rising or
  falling over time*, not on its absolute level. This is the dominant
  mechanism. Concentration increasing means the last few steps were good, so
  keep going; decreasing means turn and search.
* **Noise** -- real trajectories wander. Some randomness is not a defect here,
  it is what makes search work when the signal is momentarily absent.

The result is surge-and-cast: run when the trail is warming, sweep when it is
not. It looks nothing like gradient descent and it is far more robust to the
intermittent, patchy signals a real animal gets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NavParams:
    """Run-and-tumble, the strategy that actually converges here.

    Osmotropotaxis is hopeless at this scale. With the antennae 0.216 mm apart
    and the odour decaying over 25 mm, the left-right difference 13 mm from the
    source is about 0.004 of normalised signal against a 0.12 turn-noise floor
    -- thirty times below noise. So all the information is temporal, and the
    controller has to act on whether the smell is getting better or worse.

    Run while it improves. When it worsens, stop and turn through a large,
    randomly-sized angle, then run again. Slowing down during the turn is
    essential: at full speed the turning circle is ~21 mm across, as wide as
    the odour field itself, and the fly sails past the source.
    """

    run_drive: float = 1.0          # forward drive while the trail improves
    tumble_drive: float = 0.10      # near-stationary while reorienting
    tumble_turn: float = 1.0        # full steering authority during a tumble
    # A tumble must be able to reverse the fly. At ~75 deg/s of turn
    # authority these durations span roughly 30-180 degrees. Shorter tumbles
    # were the bug: the fly could only nudge its heading by 10-40 deg, so it
    # walked straight past the source while its bearing swept 25 -> 156 deg.
    tumble_min_ms: float = 400.0    # shortest reorientation
    tumble_max_ms: float = 2400.0   # longest reorientation
    tau_ms: float = 80.0            # smoothing on the concentration estimate
    run_min_ms: float = 450.0       # commit to a run before judging it
    run_noise: float = 0.05         # slight wander while running
    bilateral_gain: float = 2.0     # weak, but free
    lost_level: float = 1e-4        # below this, assume no odour at all

    # Flies slow down as odour intensifies, and they have to: at 15 mm/s the
    # fly covers ~7 mm between decisions, so it steps straight over a source
    # only a few mm wide. Scaling speed down near the strongest smell yet
    # encountered turns a fast search into a fine one.
    slow_near_source: float = 0.85  # how much to slow at peak odour (0 = none)
    slow_scale: float = 0.75        # concentration at which slowing saturates
    lost_tumble_ms: float = 600.0   # reorient this often when nothing is smelt


class OdorNavigator:
    """Turns antennal concentrations into (turn, drive) commands."""

    def __init__(self, params: NavParams | None = None, seed: int = 0):
        self.p = params or NavParams()
        self.rng = np.random.default_rng(seed)
        self.smoothed = None
        self.derivative = 0.0
        self.state = "run"
        self.tumble_left = 0.0
        self.tumble_sign = 1.0
        self.run_left = 0.0
        self.peak = 0.0

    def _start_run(self):
        """Commit to running for a while before judging the trail again.

        The smoothed concentration lags by roughly tau, so sampling the
        derivative the instant a tumble ends just re-reads the old decline and
        tumbles again. Measured 88% of time spent tumbling on the spot.
        """
        self.state = "run"
        self.run_left = self.p.run_min_ms

    def _start_tumble(self):
        p = self.p
        self.state = "tumble"
        self.tumble_left = float(self.rng.uniform(p.tumble_min_ms,
                                                 p.tumble_max_ms))
        self.tumble_sign = 1.0 if self.rng.random() < 0.5 else -1.0

    def update(self, left: float, right: float,
               dt_ms: float) -> tuple[float, float]:
        p = self.p
        total = 0.5 * (left + right)

        if self.smoothed is None:
            self.smoothed = total
        previous = self.smoothed
        self.smoothed += (dt_ms / max(p.tau_ms, 1e-6)) * (total - self.smoothed)
        self.derivative = (self.smoothed - previous) / max(dt_ms, 1e-6)

        self.peak = max(self.peak, self.smoothed)
        denom = left + right
        bilateral = -(left - right) / denom if denom > 1e-12 else 0.0

        # Finish any tumble already under way before reassessing.
        if self.state == "tumble":
            self.tumble_left -= dt_ms
            if self.tumble_left <= 0.0:
                self._start_run()
            else:
                return (float(np.clip(p.tumble_turn * self.tumble_sign,
                                      -1.0, 1.0)),
                        p.tumble_drive)

        if self.smoothed < p.lost_level:
            # Nothing to smell: run, reorienting periodically.
            self.state = "run"
            if self.rng.random() < dt_ms / p.lost_tumble_ms:
                self._start_tumble()
            turn = self.rng.normal(0.0, p.run_noise)
            return float(np.clip(turn, -1.0, 1.0)), p.run_drive

        self.run_left = max(0.0, self.run_left - dt_ms)
        if self.derivative < 0.0 and self.run_left <= 0.0:
            # Getting worse, and we have given the run a fair chance: reorient.
            self._start_tumble()
            return (float(np.clip(p.tumble_turn * self.tumble_sign, -1.0, 1.0)),
                    p.tumble_drive)

        # Getting better: hold course, but ease off the throttle as the smell
        # approaches the strongest yet found.
        self.state = "run"
        turn = p.bilateral_gain * bilateral + self.rng.normal(0.0, p.run_noise)
        return float(np.clip(turn, -1.0, 1.0)), self._drive()

    def _drive(self) -> float:
        """Throttle back as the absolute smell strengthens.

        Scaling against the running peak does not work: the peak is refreshed
        every step of an approach, so closeness pins at 1.0 and the fly crawls
        the whole way in. Absolute concentration is the right signal, and the
        fly is nearest the source exactly when it is largest.
        """
        p = self.p
        closeness = min(1.0, self.smoothed / max(p.slow_scale, 1e-9))
        return float(p.run_drive * (1.0 - p.slow_near_source * closeness) + 0.08)
