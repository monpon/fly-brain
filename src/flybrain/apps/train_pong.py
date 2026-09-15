#!/usr/bin/env python
"""Train a fly to intercept, by reward and punishment alone.

A ball machine, not a rally. Balls are served at random angles from the far
side; the fly either gets in the way or does not. Hitting one drives the
reward dopaminergic neurons, missing drives the punishment ones. Nothing else
teaches it -- no gradient, no correct answer, no supervised warm-up.

    flybrain train-pong                    # 600 balls, report every 10
    flybrain train-pong --balls 2000
    flybrain train-pong --explore 0        # no variability: watch it not learn
    flybrain train-pong --save output/drill.flyckpt

What the fly knows
------------------
Three quantities, one place code each over a disjoint third of the projection
neurons: **where the ball is**, **how fast it is moving across**, and **where
its own paddle is**. That is the whole sensory world.

This is deliberately harder than the encoding in `pong.py`, which hands the
fly `ball_y - paddle_y` already subtracted and so gives away the answer. Here
the two positions arrive on separate channels and the relation between them
has to be found. Kenyon cells are the natural place for that to happen: each
samples a handful of projection neurons, so a cell that happens to sample
"ball high" and "paddle low" together fires for exactly the conjunction that
should drive an upward move.

Ball *x* is not supplied. With a constant approach speed it only says how
long is left, and the decision -- which way to go -- does not depend on it.
Adding it is one more channel if you want to test that.

Why this should work when the rally version did not
---------------------------------------------------
One ball is one episode: ~150 decisions, then an outcome, then the eligibility
trace is cleared so nothing bleeds into the next. Credit stays inside the
episode that earned it.
"""

import argparse
import sys

import numpy as np

from flybrain import checkpoint, mushroom_body as MB
from flybrain.apps.pong import (BALL, H, MARGIN, MOVES, PAD_H, W, place_code,
                                wilson)
from flybrain.learning import FlyBrain, LearningParams

ACTIONS = ("UP", "STAY", "DOWN")


class Fly:
    """The paddle, and the brain driving it."""

    def __init__(self, a):
        self.mb = MB.load_cache()
        self.brain = FlyBrain(self.mb, n_actions=3, params=LearningParams(
            learning_rate=a.learning_rate,
            trace_tau=a.trace_tau,
            trace_normalize=True,
            lateral=a.lateral,
            bidirectional=a.bidirectional,
            reverse_pairing=a.reverse_pairing,
            readout_rate=a.readout_rate,
            gain_ceiling=a.gain_ceiling,
            potentiation_rate=a.potentiation_rate,
            da_trace_tau=a.da_trace_tau,
        ))
        self.n_pn = self.mb.n("PN")
        self.rng = np.random.default_rng(a.seed)
        self.speed = a.fly_speed
        self.pn_gain = a.pn_gain
        self.explore0, self.explore_final = a.explore, a.explore_final
        self.halflife = a.explore_halflife
        self.max_vy = a.ball_speed * np.sin(a.max_angle)
        self.y = H / 2
        self.hits = self.misses = 0
        self.last_pn = None
        self.last_action = 1
        self.last_drives = np.zeros(3, dtype=np.float32)
        self.decisions = self.moved = self.toward = 0

    # -- sensing ----------------------------------------------------------

    def sense(self, ball_y: float, ball_vy: float) -> np.ndarray:
        """Ball position, ball velocity, own position. Three channels."""
        n = self.n_pn
        third = n // 3
        span_y = (H - 2 * BALL) / 2
        pad_span = (H - PAD_H) / 2
        return (
            place_code(n, (ball_y - H / 2) / span_y, 0, third, 5.0,
                       self.pn_gain)
            + place_code(n, float(np.clip(ball_vy / max(self.max_vy, 1e-6),
                                          -1, 1)),
                         third, 2 * third, 5.0, self.pn_gain)
            + place_code(n, (self.y - H / 2) / pad_span, 2 * third, n, 5.0,
                         self.pn_gain)
        )

    # -- acting -----------------------------------------------------------

    def explore_now(self) -> float:
        if self.halflife <= 0:
            return self.explore0
        balls = self.hits + self.misses
        frac = 1.0 / (1.0 + balls / self.halflife)
        return self.explore_final + (self.explore0 - self.explore_final) * frac

    def step(self, ball_y: float, ball_vy: float) -> None:
        pn = self.sense(ball_y, ball_vy)
        # Tag what is active now without changing it; the outcome arrives when
        # the ball finally gets here.
        self.brain.learn(pn, update=False)
        action, drives = self.brain.choose(pn, explore=self.explore_now(),
                                           rng=self.rng)

        self.decisions += 1
        if MOVES[3][action] != 0.0:
            self.moved += 1
            if np.sign(MOVES[3][action]) == np.sign(ball_y - self.y):
                self.toward += 1

        self.last_pn, self.last_action, self.last_drives = pn, action, drives
        self.y = float(np.clip(self.y + MOVES[3][action] * self.speed,
                               PAD_H / 2, H - PAD_H / 2))

    def reinforce(self, hit: bool) -> None:
        """Dopamine on an intercept, shock on a miss. The only teacher."""
        if self.last_pn is not None:
            self.brain.reinforce(self.last_pn, self.last_action,
                                 +1.0 if hit else -1.0)
        if hit:
            self.hits += 1
        else:
            self.misses += 1
        # One ball is one episode. Eligibility must not survive it, or the
        # next ball's outcome lands partly on this ball's decisions.
        self.brain.reset_trace()


class BallMachine:
    """Serves balls at random angles across the court. No opponent."""

    def __init__(self, a, rng):
        self.a = a
        self.rng = rng
        self.paddle_x = W - MARGIN
        self.serve()

    def serve(self) -> None:
        self.bx = float(MARGIN)
        self.by = float(self.rng.uniform(BALL, H - BALL))
        angle = float(self.rng.uniform(-self.a.max_angle, self.a.max_angle))
        self.vx = self.a.ball_speed * np.cos(angle)
        self.vy = self.a.ball_speed * np.sin(angle)

    def step(self, fly: Fly) -> bool | None:
        """Advance one frame. Returns True/False on a resolved ball."""
        fly.step(self.by, self.vy)

        self.bx += self.vx
        self.by += self.vy
        if self.by < BALL or self.by > H - BALL:
            self.vy = -self.vy
            self.by = float(np.clip(self.by, BALL, H - BALL))

        if self.bx >= self.paddle_x - BALL:
            hit = abs(self.by - fly.y) < PAD_H / 2 + BALL
            fly.reinforce(hit)
            self.serve()
            return hit
        return None


def shuffle_null(actions: np.ndarray, correct: np.ndarray, rng,
                 reps: int = 200) -> tuple[float, float, float]:
    """Chance level that respects both marginals.

    `1/n_actions` is the wrong null here and flatters the result: STAY is
    almost never the right answer, so the task is effectively a two-way choice
    over near-even labels and any policy that moves both ways scores about
    50% knowing nothing. Shuffling the pairing keeps both distributions and
    destroys only the mapping, which is the thing being tested.
    """
    scores = [float((actions == rng.permutation(correct)).mean())
              for _ in range(reps)]
    return float(np.mean(scores)), *np.percentile(scores, [2.5, 97.5])


def main(a) -> int:
    fly = Fly(a)
    machine = BallMachine(a, np.random.default_rng(a.seed + 1))

    print(f"fresh fly: {fly.mb.n('KC')} Kenyon cells, {fly.n_pn} PN channels "
          f"(ball y | ball vy | paddle y)")
    print(f"3 actions {ACTIONS}, reward on intercept, shock on miss")
    print(f"{a.balls} balls, angles up to {np.degrees(a.max_angle):.0f} deg, "
          f"explore {a.explore} -> {a.explore_final}\n")
    print(f"{'balls':>6s} {'last 10':>8s} {'overall':>8s} {'toward':>7s} "
          f"{'depressed':>10s} {'explore':>8s}")

    block_hits = 0
    frames = 0
    limit = a.balls * 5000
    while (fly.hits + fly.misses) < a.balls and frames < limit:
        outcome = machine.step(fly)
        frames += 1
        if outcome is None:
            continue
        block_hits += int(outcome)
        seen = fly.hits + fly.misses
        if seen % a.every == 0:
            toward = 100 * fly.toward / max(fly.moved, 1)
            print(f"{seen:6d} {100*block_hits/a.every:7.0f}% "
                  f"{100*fly.hits/seen:7.1f}% {toward:6.1f}% "
                  f"{100*fly.brain.depressed_fraction():9.1f}% "
                  f"{fly.explore_now():8.3f}")
            block_hits = 0

    seen = fly.hits + fly.misses
    lo, hi = wilson(fly.hits, seen)
    print(f"\n{fly.hits} / {seen} = {100*fly.hits/seen:.1f}% "
          f"(95% CI {100*lo:.1f}-{100*hi:.1f}%)")

    # Did the policy itself learn, independent of how the balls happened to
    # fall? Scored greedily on fresh positions, against the honest null.
    rng = np.random.default_rng(a.seed + 99)
    acts, correct = [], []
    for _ in range(3000):
        fly.y = float(rng.uniform(PAD_H / 2, H - PAD_H / 2))
        by = float(rng.uniform(BALL, H - BALL))
        vy = float(rng.uniform(-fly.max_vy, fly.max_vy))
        acts.append(fly.brain.greedy(fly.sense(by, vy)))
        gap = by - fly.y
        correct.append(1 if abs(gap) < fly.speed else (0 if gap < 0 else 2))
    acts, correct = np.array(acts), np.array(correct)
    agree = float((acts == correct).mean())
    null, nlo, nhi = shuffle_null(acts, correct, np.random.default_rng(3))

    print(f"\npolicy vs a tracking rule: {100*agree:.1f}%")
    print(f"  shuffle null {100*null:.1f}% (95% {100*nlo:.1f}-{100*nhi:.1f})")
    mix = np.bincount(acts, minlength=3) / len(acts)
    print(f"  action mix  " + "  ".join(f"{ACTIONS[i]} {100*m:.1f}%"
                                        for i, m in enumerate(mix)))
    if a.save:
        print(f"\nsaved -> {checkpoint.save(a.save, fly.brain, task='train-pong', notes=f'{fly.hits}/{seen}')}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--balls", type=int, default=600)
    ap.add_argument("--every", type=int, default=10,
                    help="report accuracy every N balls")
    ap.add_argument("--max-angle", type=float, default=0.9,
                    help="serve angle limit in radians")
    ap.add_argument("--explore", type=float, default=0.6)
    ap.add_argument("--explore-final", type=float, default=0.05)
    ap.add_argument("--explore-halflife", type=float, default=150.0,
                    metavar="BALLS")
    ap.add_argument("--learning-rate", type=float, default=0.35)
    ap.add_argument("--trace-tau", type=float, default=40.0,
                    help="eligibility decay in frames; a ball is ~150")
    ap.add_argument("--pn-gain", type=float, default=30.0)
    ap.add_argument("--fly-speed", type=float, default=7.0)
    ap.add_argument("--ball-speed", type=float, default=5.0)
    # -- plasticity mechanisms, all off by default so the baseline stands --
    ap.add_argument("--lateral", type=float, default=0.0,
                    help="MBON->MBON lateral interaction strength (0 = off)")
    ap.add_argument("--bidirectional", action="store_true",
                    help="allow potentiation as well as depression")
    ap.add_argument("--gain-ceiling", type=float, default=1.0,
                    help="how far above anatomy a synapse may be potentiated")
    ap.add_argument("--potentiation-rate", type=float, default=0.0,
                    help="0 uses --learning-rate")
    ap.add_argument("--readout-rate", type=float, default=0.0,
                    help="learn which MBONs drive which action (0 = frozen, "
                         "the arbitrary index-order split)")
    ap.add_argument("--reverse-pairing", action="store_true",
                    help="order-dependent potentiation (relief learning); "
                         "writes noise in a task with no post-outcome cue")
    ap.add_argument("--da-trace-tau", type=float, default=8.0,
                    help="frames dopamine stays available to potentiate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=None)
    sys.exit(main(ap.parse_args()) or 0)
