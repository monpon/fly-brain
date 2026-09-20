"""The aiming task, hundreds of flies at once on the GPU.

`train_pong_gpu` only ever wired the tracking and `--predict` drills into the
batched backend. The *aimer* -- see the ball once, name one of `bins` places
to stand -- stayed on the numpy path, one fly at a time, which made every
comparison between aiming variants a sequence of 80-second single-seed runs.

That was expensive twice over. Slow, and worse, single-seed: the same
configuration measured 25.0% and 48.8% on different seeds, so point estimates
from one fly were noise and several conclusions drawn from them were wrong.
Running the flies as a batch fixes the statistics and the runtime together,
which is the whole reason `batched.py` exists.

The task is `aim_drill`, vectorised:

    serve            random height and angle, closed-form landing point
    sense            three place-coded channels: height, vertical velocity,
                     distance still to travel, optionally quantised
    choose           one of `bins` slices of the court
    reinforce        reward and punishment, on one of two criteria

`--reward catch` is the criterion that rewards any choice that would have
caught the ball rather than only the exact bin, since with bins narrower than
the catch window an adjacent bin often still catches and the exact-bin rule
punishes it for doing so.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from flybrain import mushroom_body as MB
from flybrain.apps.pong import BALL, H, MARGIN, PAD_H, W
from flybrain.batched import BatchedBrain, pick_device
from flybrain.learning import LearningParams

MAX_STEEPNESS = 0.85
CATCH = PAD_H / 2 + BALL


def serve(torch, n, rng_gen, device, speed, x_max):
    """(N,) serves: height, vertical and horizontal velocity, and ball x."""
    y = torch.rand(n, generator=rng_gen, device=device) * (H - 2 * BALL) + BALL
    u = torch.rand(n, generator=rng_gen, device=device) - 0.5
    angle = (u * 1.6).clamp(-float(np.arcsin(MAX_STEEPNESS)),
                            float(np.arcsin(MAX_STEEPNESS)))
    vx = speed * torch.cos(angle)
    vy = speed * torch.sin(angle)
    x = (torch.rand(n, generator=rng_gen, device=device)
         * (x_max - MARGIN) + MARGIN)
    return y, vy, vx, x


def landing(torch, y, vy, vx, x):
    """Closed-form arrival height: extrapolate, then fold the walls in."""
    span = H - 2 * BALL
    t = (W - MARGIN - BALL - x) / vx.clamp(min=1e-6)
    raw = (y + vy * t - BALL) % (2 * span)
    folded = torch.where(raw > span, 2 * span - raw, raw)
    return folded + BALL


def place_code(torch, value, n_pn, lo, hi, width, peak):
    """(N,) values -> (N, n_pn) Gaussian bumps over channels [lo, hi)."""
    span = hi - lo
    centre = (value * 0.5 + 0.5) * (span - 1)
    idx = torch.arange(span, device=value.device, dtype=value.dtype)
    bump = peak * torch.exp(-0.5 * ((idx.unsqueeze(0)
                                     - centre.unsqueeze(1)) / width) ** 2)
    out = torch.zeros(value.shape[0], n_pn, device=value.device,
                      dtype=value.dtype)
    out[:, lo:hi] = bump
    return out


def snap(value, levels):
    """Quantise a [-1, 1] value to `levels` steps, as Aimer._snap does."""
    if not levels:
        return value
    idx = ((value + 1) / 2 * levels).floor().clamp(0, levels - 1)
    return (idx + 0.5) / levels * 2 - 1


def sense(torch, y, vy, x, n_pn, speed, levels, pn_gain):
    """The Aimer's three channels, batched."""
    third = n_pn // 3
    span = (H - 2 * BALL) / 2
    pos = ((y - H / 2) / span).clamp(-1.0, 1.0)
    vel = (vy / max(speed, 1e-6)).clamp(-1.0, 1.0)
    togo = (((x - MARGIN) / max(W - 2 * MARGIN, 1.0)) * 2 - 1).clamp(-1.0, 1.0)
    pos, vel, togo = (snap(v, levels) for v in (pos, vel, togo))
    return (place_code(torch, pos, n_pn, 0, third, 5.0, pn_gain)
            + place_code(torch, vel, n_pn, third, 2 * third, 5.0, pn_gain)
            + place_code(torch, togo, n_pn, 2 * third, n_pn, 5.0, pn_gain))


def bin_centres(torch, bins, device, dtype):
    lo, hi = PAD_H / 2, H - PAD_H / 2
    k = torch.arange(bins, device=device, dtype=dtype)
    return lo + (k + 0.5) / bins * (hi - lo)


def bin_of(torch, y, bins):
    lo, hi = PAD_H / 2, H - PAD_H / 2
    frac = (y - lo) / max(hi - lo, 1e-9)
    return (frac * bins).floor().clamp(0, bins - 1).long()


def run(a) -> int:
    device = pick_device(a.device)
    mb = MB.load_cache()
    brain = BatchedBrain(mb, a.flies, n_actions=a.bins, device=device,
                         seed=a.seed,
                         params=LearningParams(learning_rate=a.learning_rate,
                                               trace_tau=1.0))
    torch = brain.torch
    gen = torch.Generator(device=device)
    gen.manual_seed(a.seed + 4242)
    n_pn = mb.n("PN")
    centres = bin_centres(torch, a.bins, device, torch.float32)
    x_max = (W - MARGIN - BALL) if a.reaim else W / 2

    print(f"{a.flies} flies, {a.bins} bins of {(H - PAD_H) / a.bins:.1f} px, "
          f"{a.levels or 'continuous'} levels, reward={a.reward}")
    print(f"catch window {2 * CATCH:.0f} px, ball seen in "
          f"{'the whole court' if a.reaim else 'the left half'}")
    print(f"device: {device}\n")
    print(f"{'serves':>8s} {'in window':>10s} {'median err':>11s} "
          f"{'depressed':>10s}")

    t0 = time.perf_counter()
    block_win, block_err, block_n = 0.0, [], 0
    step = max(1, a.balls // 10)
    for i in range(1, a.balls + 1):
        y, vy, vx, x = serve(torch, a.flies, gen, device, a.ball_speed, x_max)
        land = landing(torch, y, vy, vx, x)
        want = bin_of(torch, land, a.bins)

        pn = sense(torch, y, vy, x, n_pn, a.ball_speed, a.levels, a.pn_gain)
        brain.reset_trace()
        _, _, drives = brain.observe(pn)
        actions, _ = brain.choose(drives, a.explore)

        err = (centres[actions] - land).abs()
        caught = err < CATCH
        active = torch.ones(a.flies, dtype=torch.bool, device=device)

        if a.reward == "catch":
            # Reward the choice it made when that choice would have worked;
            # otherwise teach a bin that would have, and discourage this one.
            brain.reset_trace()
            brain.observe(pn)
            brain.reinforce(torch.where(caught, actions, want),
                            torch.ones(a.flies, device=device), active)
            wrong = ~caught & (actions != want)
            if bool(wrong.any()):
                brain.reset_trace()
                brain.observe(pn)
                brain.reinforce(actions,
                                -torch.ones(a.flies, device=device), wrong)
        else:
            brain.reset_trace()
            brain.observe(pn)
            brain.reinforce(want, torch.ones(a.flies, device=device), active)
            wrong = actions != want
            if bool(wrong.any()):
                brain.reset_trace()
                brain.observe(pn)
                brain.reinforce(actions,
                                -torch.ones(a.flies, device=device), wrong)
        brain.reset_trace()

        block_win += float(caught.float().sum())
        block_err.append(err)
        block_n += a.flies
        if i % step == 0:
            med = float(torch.cat(block_err).median())
            print(f"{i:8d} {100 * block_win / block_n:9.1f}% {med:10.1f} px "
                  f"{100 * float(brain.depressed_fraction().mean()):9.1f}%")
            block_win, block_err, block_n = 0.0, [], 0

    # -- evaluation: greedy, no learning, fresh serves --------------------
    egen = torch.Generator(device=device)
    egen.manual_seed(a.seed + 99991)
    win = torch.zeros(a.flies, device=device)
    errs = []
    for _ in range(a.eval_balls):
        y, vy, vx, x = serve(torch, a.flies, egen, device, a.ball_speed, x_max)
        land = landing(torch, y, vy, vx, x)
        pn = sense(torch, y, vy, x, n_pn, a.ball_speed, a.levels, a.pn_gain)
        brain.reset_trace()
        _, _, drives = brain.observe(pn)
        actions = drives.argmax(dim=1)
        err = (centres[actions] - land).abs()
        win += (err < CATCH).float()
        errs.append(err)
    brain.reset_trace()

    rate = (win / a.eval_balls).cpu().numpy()
    med = float(torch.cat(errs).median())
    el = time.perf_counter() - t0
    print(f"\n{a.eval_balls} unseen serves per fly, greedy")
    print(f"  in the catch window  {100 * rate.mean():.1f}% "
          f"+/- {100 * rate.std(ddof=1) / np.sqrt(len(rate)):.1f} (SEM over "
          f"{a.flies} flies)")
    print(f"  spread across flies  {100 * rate.min():.1f}% to "
          f"{100 * rate.max():.1f}%")
    print(f"  median aim error     {med:.1f} px")
    print(f"  a parked paddle scores about 21%")
    print(f"\n{a.flies * (a.balls + a.eval_balls):,} fly-trials in {el:.0f}s "
          f"({a.flies * (a.balls + a.eval_balls) / max(el, 1e-9):,.0f}/s)")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--flies", type=int, default=128)
    p.add_argument("--bins", type=int, default=8)
    p.add_argument("--levels", type=int, default=4)
    p.add_argument("--balls", type=int, default=4000,
                   help="training serves per fly")
    p.add_argument("--eval-balls", type=int, default=200)
    p.add_argument("--reward", choices=("exact", "catch"), default="catch",
                   help="'catch' rewards any aim that would have caught the "
                        "ball, 'exact' only the correct bin")
    p.add_argument("--reaim", action="store_true",
                   help="train on the ball anywhere in the court, which is "
                        "what re-aiming every frame actually presents")
    p.add_argument("--explore", type=float, default=0.35)
    p.add_argument("--learning-rate", type=float, default=0.02)
    p.add_argument("--pn-gain", type=float, default=1.0)
    p.add_argument("--ball-speed", type=float, default=5.0)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"),
                   default="auto")
    p.add_argument("--seed", type=int, default=1)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
