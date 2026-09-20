"""Teach the fly to name where the ball will arrive, from any point in flight.

Every earlier attempt at prediction measured the *hit rate*, which mixes two
different failures: guessing the wrong place, and failing to walk there in
time. This measures the guess alone.

The task is a classification. At an arbitrary moment during the ball's
flight the fly sees where the ball is, how fast it is rising or falling, and
how far it still has to travel; it names one of `bins` slices of the court.
The right answer is where the ball actually arrives, which the animal can
simply watch happen -- no physics is handed to it, and the label is available
for every frame of the flight rather than once per ball.

That is the shape this circuit is good at: one stimulus, one choice among N,
one outcome, immediately. `flybrain teach` solves it at 100% for six odours
and 68% for thirty-two.

It works, and not well enough to catch a ball. Naming the right bin runs at
40.7% against a 30.7% always-one-bin baseline over five seeds, and the
mid-flight estimate genuinely sharpens as the ball closes -- 116 px of error
at 94% of the flight remaining, down to 77 px at 37%. That is real
extrapolation rather than pattern matching. But it plateaus at 77 px against
a 48 px catch window, so most of the sharpening never crosses the threshold
that decides whether a ball comes back.

What the limit is *not*, each measured rather than argued:

    credit assignment   immediate per-decision credit instead of one
                        terminal outcome changes nothing
    association capacity bidirectional plasticity doubles capacity on
                        arbitrary patterns and moves this task 2 points
    input resolution    every reallocation of levels across the three
                        channels scored worse than the uniform split
    readout precision   finer bins raise precision and lose accuracy at the
                        same rate

The likeliest reading is that a smooth regression is the wrong job for this
circuit. The mushroom body maps a pattern to a channel; interception wants a
continuous function of three variables, and overlap between neighbouring
place codes is doing more of that work than the plasticity is -- which is
why replacing them with orthogonal patterns made it worse, not better.

So this reports three numbers together:

    chance     1/bins, picking blindly, plus the always-one-bin floor
    fly        what the mushroom body learns
    ceiling    the best any lookup table over the same quantised addresses
               could do, measured on the same data

The gap between fly and ceiling is a learning failure. The gap between
ceiling and 100% is an encoding failure. Without both, a mediocre score
cannot be attributed to either -- and the ceiling has to be computed on the
distribution the task will actually present, which is a mistake made once
here already.
"""

from __future__ import annotations

import argparse
import dataclasses
import types

import numpy as np

from flybrain.apps.pong import (BALL, H, MARGIN, PAD_H, W, Aimer, Court,
                                landing_of, wilson)
from flybrain.learning import FlyBrain


def phase_of(y0: float, vy: float, t: float) -> tuple[float, float]:
    """Ball height and vertical velocity t frames after (y0, vy).

    Perfectly elastic walls turn a straight line into a triangle wave, so
    both come out of one phase calculation. Getting the *velocity* from that
    phase is the part that has to be right: computing bounces as
    `abs(raw - BALL) // span` says zero bounces for a ball that has gone
    below the floor, so vy keeps its old sign and the fly is shown a
    (height, velocity) pair that does not imply the label it is taught. That
    reads as below-chance performance and looks exactly like a circuit that
    cannot learn.
    """
    lo, span = BALL, H - 2 * BALL
    phase = (y0 + vy * t - lo) % (2 * span)
    if phase <= span:
        return lo + phase, vy
    return lo + 2 * span - phase, -vy


class Flight:
    """One serve, and the truth about where it lands.

    Serve geometry matches `aim_drill` so results stay comparable: the same
    steepness clip, and the ball may be seen anywhere in the left half of the
    court rather than only at the moment of release.
    """

    def __init__(self, rng, speed: float):
        self.y0 = float(rng.uniform(BALL, H - BALL))
        angle = float(rng.uniform(-0.5, 0.5)) * 1.6
        angle = float(np.clip(angle, -np.arcsin(Court.MAX_STEEPNESS),
                              np.arcsin(Court.MAX_STEEPNESS)))
        self.vx = speed * float(np.cos(angle))
        self.vy = speed * float(np.sin(angle))
        self.x0 = float(MARGIN)
        self.x_end = float(W - MARGIN - BALL)
        self.duration = (self.x_end - self.x0) / self.vx
        # The label comes from pong's own physics, not a reimplementation.
        self.landing = landing_of(self.y0, self.vy, self.vx, self.x0)

    def at(self, t: float) -> tuple[float, float, float]:
        y, vy = phase_of(self.y0, self.vy, t)
        return y, vy, self.x0 + self.vx * t


def _levels(args):
    """Per-channel resolution, or the single value if none was given."""
    if args.levels_vel or args.levels_pos or args.levels_togo:
        return (args.levels_pos or args.levels,
                args.levels_vel or args.levels,
                args.levels_togo or args.levels)
    return args.levels


def make_aimer(args) -> Aimer:
    """An Aimer with the encoding pong uses, so numbers stay comparable."""
    fly = Aimer(types.SimpleNamespace(
        bins=args.bins, levels=_levels(args),
        learning_rate=args.learning_rate,
        seed=args.seed, fly_speed=7.0, pn_gain=args.pn_gain,
        explore=args.explore, explore_final=args.explore,
        explore_halflife=0.0, smooth=1.0))
    if args.bidirectional:
        # Depression-only is a one-way budget: gains fall into
        # [gain_floor, 1] and cannot be re-spent, so each new association
        # partly erases an older one. Measured on arbitrary sparse patterns,
        # that caps 6-channel accuracy at 45.8% for 8 associations; letting
        # the rule push gains back up takes the same circuit to 91.7%, and
        # the *depressed* fraction goes down rather than up -- it stores more
        # by storing reversibly, not by writing more.
        #
        # LearningParams is frozen, so this rebuilds the brain rather than
        # mutating it. Nothing has been learned yet, so nothing is lost.
        params = dataclasses.replace(
            fly.brain.params, bidirectional=True,
            gain_ceiling=args.gain_ceiling,
            potentiation_rate=args.potentiation_rate)
        fly.brain = FlyBrain(fly.mb, params, n_actions=args.bins)
    return fly


def moments(flight: Flight, per_ball: int, rng) -> list:
    """When during the flight to ask, stratified over the whole flight.

    Never a fixed t=0: that pins ball_x to the serve point, leaving the
    "how far to go" channel constant and the task degenerate.
    """
    span = flight.duration * 0.95
    if per_ball <= 1:
        return [float(rng.uniform(0.0, span))]
    edges = np.linspace(0.0, span, per_ball + 1)
    return [float(rng.uniform(edges[i], edges[i + 1]))
            for i in range(per_ball)]


def evaluate(fly, args, seed: int) -> dict:
    """Greedy, no learning, unseen serves."""
    rng = np.random.default_rng(seed)
    catch = PAD_H / 2 + BALL
    ok = win = n = 0
    per_bin = np.zeros(args.bins, dtype=np.int64)
    by_address: dict = {}
    for _ in range(args.eval_balls):
        flight = Flight(rng, args.ball_speed)
        want = fly.bin_of(flight.landing)
        for t in moments(flight, args.per_ball, rng):
            y, vy, x = flight.at(t)
            pn = fly.sense(y, vy, x, args.ball_speed)
            action = int(np.argmax(fly.brain.drives(pn)))
            ok += int(action == want)
            win += int(abs(fly.bin_centre(action) - flight.landing) < catch)
            per_bin[action] += 1
            n += 1
            by_address.setdefault(pn.tobytes(), []).append(want)

    # Ceiling: the best a lookup table over these addresses could score. When
    # two situations with different answers share an address, no amount of
    # learning can separate them, and this is where that shows up.
    best = sum(int(np.bincount(np.asarray(v)).max())
               for v in by_address.values())
    # Chance is only 1/bins if the landing distribution is flat. The honest
    # floor is the best a fly could do by always naming the same bin.
    labels = (np.concatenate([np.asarray(v) for v in by_address.values()])
              if by_address else np.zeros(1, dtype=int))
    counts = np.bincount(labels, minlength=args.bins)
    return {"ok": ok, "win": win, "n": n, "per_bin": per_bin,
            "addresses": len(by_address), "ceiling": best / max(n, 1),
            "majority": float(counts.max()) / max(counts.sum(), 1),
            "labels": counts}


def train(fly, args) -> None:
    rng = np.random.default_rng(args.seed)
    catch = PAD_H / 2 + BALL
    print(f"{'balls':>7s} {'exact':>8s} {'window':>8s} {'depressed':>10s}")
    ok = win = seen = 0
    step = max(1, args.balls // 10)
    for ball in range(1, args.balls + 1):
        flight = Flight(rng, args.ball_speed)
        want = fly.bin_of(flight.landing)
        for t in moments(flight, args.per_ball, rng):
            y, vy, x = flight.at(t)
            pn = fly.sense(y, vy, x, args.ball_speed)
            fly.brain.reset_trace()
            action, _ = fly.brain.choose(pn, explore=args.explore,
                                         rng=fly.rng)
            # Immediate, per-decision credit: was *this* guess right? Not
            # whether the ball was eventually caught, which is what spread a
            # single outcome over fifty decisions and taught nothing.
            fly.brain.reset_trace()
            fly.brain.reinforce(pn, want, +1.0)
            if action != want:
                fly.brain.reset_trace()
                fly.brain.reinforce(pn, action, -1.0)
            fly.brain.reset_trace()
            ok += int(action == want)
            win += int(abs(fly.bin_centre(action) - flight.landing) < catch)
            seen += 1
        if ball % step == 0:
            print(f"{ball:7d} {100 * ok / max(seen, 1):7.1f}% "
                  f"{100 * win / max(seen, 1):7.1f}% "
                  f"{100 * fly.brain.depressed_fraction():9.1f}%")
            ok = win = seen = 0


def report(res: dict, args) -> None:
    n = max(res["n"], 1)
    lo, hi = wilson(res["ok"], n)
    print(f"\ngreedy on {args.eval_balls} unseen serves, {n} decisions")
    print(f"  exact bin          {100 * res['ok'] / n:5.1f}%  "
          f"(95% CI {100 * lo:.1f}-{100 * hi:.1f}%)")
    print(f"  within catch window{100 * res['win'] / n:5.1f}%")
    print(f"  chance             {100 / args.bins:5.1f}%  "
          f"(always one bin: {100 * res['majority']:.1f}%)")
    print(f"  landing spread     " +
          " ".join(f"{100 * v / max(res['labels'].sum(), 1):.0f}%"
                   for v in res["labels"]))
    print(f"  encoding ceiling   {100 * res['ceiling']:5.1f}%  "
          f"({res['addresses']} distinct addresses used)")
    frac = (res["ok"] / n) / max(res["ceiling"], 1e-9)
    print(f"  -> the fly reaches {100 * frac:.0f}% of what its encoding "
          f"allows")
    used = int((res["per_bin"] > 0).sum())
    print(f"  bins used {used}/{args.bins}: " +
          " ".join(f"{100 * v / n:.0f}%" for v in res["per_bin"]))
    if used == 1:
        print("  WARNING: one bin for everything -- the readout is stuck, "
              "not trained")


def run(args) -> int:
    catch = PAD_H / 2 + BALL
    lv = _levels(args)
    shown = (f"levels pos/vel/togo {lv[0]}/{lv[1]}/{lv[2]}"
             if isinstance(lv, tuple) else f"{lv or 'continuous'} levels")
    print(f"{args.bins} bins, {shown}, {args.balls} balls x "
          f"{args.per_ball} moments")
    print(f"a bin is {(H - PAD_H) / args.bins:.0f} px wide, the catch window "
          f"is {2 * catch:.0f} px\n")
    fly = make_aimer(args)
    train(fly, args)
    report(evaluate(fly, args, args.seed + 9991), args)
    return 0


def sweep(args) -> int:
    """Encoding sweep: where does the ceiling stop being the problem?"""
    print(f"{'levels':>7s} {'bins':>6s} {'addr':>6s} {'fly':>7s} "
          f"{'ceiling':>8s} {'window':>8s} {'of ceiling':>11s}")
    for levels in (3, 4, 6, 8):
        for bins in (4, 6, 8):
            a = types.SimpleNamespace(**vars(args))
            a.levels, a.bins = levels, bins
            fly = make_aimer(a)
            train_quiet(fly, a)
            r = evaluate(fly, a, a.seed + 9991)
            n = max(r["n"], 1)
            print(f"{levels:7d} {bins:6d} {r['addresses']:6d} "
                  f"{100 * r['ok'] / n:6.1f}% {100 * r['ceiling']:7.1f}% "
                  f"{100 * r['win'] / n:7.1f}% "
                  f"{100 * (r['ok'] / n) / max(r['ceiling'], 1e-9):10.0f}%")
    return 0


def train_quiet(fly, args) -> None:
    rng = np.random.default_rng(args.seed)
    for _ in range(args.balls):
        flight = Flight(rng, args.ball_speed)
        want = fly.bin_of(flight.landing)
        for t in moments(flight, args.per_ball, rng):
            y, vy, x = flight.at(t)
            pn = fly.sense(y, vy, x, args.ball_speed)
            fly.brain.reset_trace()
            action, _ = fly.brain.choose(pn, explore=args.explore,
                                         rng=fly.rng)
            fly.brain.reset_trace()
            fly.brain.reinforce(pn, want, +1.0)
            if action != want:
                fly.brain.reset_trace()
                fly.brain.reinforce(pn, action, -1.0)
            fly.brain.reset_trace()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--bins", type=int, default=6,
                   help="slices of the court the fly chooses between")
    p.add_argument("--levels", type=int, default=4,
                   help="quantisation per input channel; levels^3 addresses")
    p.add_argument("--balls", type=int, default=400)
    p.add_argument("--per-ball", type=int, default=8,
                   help="moments during each flight to ask about")
    p.add_argument("--eval-balls", type=int, default=200)
    p.add_argument("--explore", type=float, default=0.2)
    p.add_argument("--learning-rate", type=float, default=0.02)
    p.add_argument("--pn-gain", type=float, default=1.0)
    p.add_argument("--ball-speed", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--levels-pos", type=int, default=0,
                   help="resolution for ball height; 0 uses --levels")
    p.add_argument("--levels-vel", type=int, default=0,
                   help="resolution for vertical velocity. This is the one "
                        "that matters: the landing point is fold(y + vy*t) "
                        "with t about 150 frames, so a step of vy moves the "
                        "answer ~150x further than a step of y")
    p.add_argument("--levels-togo", type=int, default=0,
                   help="resolution for distance still to travel")
    p.add_argument("--bidirectional", action="store_true",
                   help="let the rule potentiate as well as depress, so the "
                        "plastic budget is reusable")
    p.add_argument("--gain-ceiling", type=float, default=2.0)
    p.add_argument("--potentiation-rate", type=float, default=0.0,
                   help="0 uses the learning rate")
    p.add_argument("--sweep", action="store_true",
                   help="sweep levels and bins instead of one run")
    args = p.parse_args(argv)
    return sweep(args) if args.sweep else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
