#!/usr/bin/env python
"""The interception drill, run on hundreds of flies at once.

Same task as `flybrain train-pong`: balls at random angles, dopamine on an
intercept, shock on a miss, three place-coded channels for ball position,
ball velocity and paddle position. The difference is that this runs N
independent flies simultaneously on the GPU.

    flybrain train-pong-gpu                      # 128 flies, 600 balls each
    flybrain train-pong-gpu --flies 256 --balls 2000
    flybrain train-pong-gpu --device cpu         # same code, no card

Why this exists is only half about speed. Every interval in
docs/FINDINGS.md so far came from a single fly with a single seed, which is
why 28.8% against 26.7% could not be resolved however many balls were thrown
at it -- more balls tightens the estimate of *that fly's* luck, not of the
mechanism. N flies are N seeds, so the spread across flies is the error bar
that was actually wanted.

Nothing leaves the GPU inside the loop. The physics is a handful of `(N,)`
tensors updated in place, and balls are resolved by masking rather than by
indexing, because asking which flies finished would force a host-device
sync every frame and hand back the speedup this file exists for.
"""

import argparse
import sys
import time

import numpy as np

from flybrain import mushroom_body as MB
from flybrain._deps import require
from flybrain.apps.pong import BALL, H, MARGIN, PAD_H, W, wilson
from flybrain.batched import BatchedBrain, pick_device
from flybrain.learning import LearningParams

ACTIONS = ("UP", "STAY", "DOWN")
MOVE = (-1.0, 0.0, 1.0)


class Drill:
    """N flies, N balls, all resident on the device."""

    def __init__(self, a):
        self.torch = require("torch")
        torch = self.torch
        self.a = a
        self.device = pick_device(a.device)
        mb = MB.load_cache()

        # The eligibility trace has to match the length of a trial. With
        # `trace_normalize`, one `observe` contributes 1-exp(-1/tau) of the
        # Kenyon-cell magnitude -- at tau=40 that is 2.5%, which is fine when
        # the trace has ~157 frames to build before the outcome lands, and
        # useless when every frame is its own trial and the trace is cleared
        # straight afterwards. The first dense run had `depressed` stuck at
        # 1.2% against the sparse run's 32.7%: depression was running about
        # 40x too weak and the fly was not learning at all.
        #
        # `flybrain teach`, which reaches 100% under exactly this per-trial
        # structure, uses tau=1. Dense mode does the same unless told otherwise.
        trace_tau = a.trace_tau
        if trace_tau is None:
            trace_tau = 1.0 if a.dense else 40.0

        self.brain = BatchedBrain(
            mb, a.flies, n_actions=3, device=self.device, seed=a.seed,
            params=LearningParams(
                learning_rate=a.learning_rate,
                trace_tau=trace_tau,
                trace_normalize=True,
                lateral=a.lateral,
                bidirectional=a.bidirectional,
                gain_ceiling=a.gain_ceiling,
                readout_rate=a.readout_rate,
            ),
        )
        self.n_pn = mb.n("PN")
        self.third = self.n_pn // 3
        N = a.flies
        d = self.device

        self.gen = torch.Generator(device=d)
        self.gen.manual_seed(a.seed + 1)
        self.y = torch.full((N,), H / 2.0, device=d)
        self.hits = torch.zeros(N, device=d)
        self.balls = torch.zeros(N, device=d)
        self.max_vy = a.ball_speed * float(np.sin(a.max_angle))
        self.paddle_x = W - MARGIN

        self.bx = torch.empty(N, device=d)
        self.by = torch.empty(N, device=d)
        self.vx = torch.empty(N, device=d)
        self.vy = torch.empty(N, device=d)
        self._serve(torch.ones(N, device=d, dtype=torch.bool))

    def _rand(self, n):
        return self.torch.rand(n, generator=self.gen, device=self.device)

    def _serve(self, who) -> None:
        """Re-serve only the flies in `who`, without indexing."""
        torch = self.torch
        N = self.a.flies
        angle = (self._rand(N) * 2 - 1) * self.a.max_angle
        new_by = BALL + self._rand(N) * (H - 2 * BALL)
        keep = (~who).float()
        take = who.float()
        self.bx = self.bx * keep + take * float(MARGIN)
        self.by = self.by * keep + take * new_by
        self.vx = self.vx * keep + take * (self.a.ball_speed * torch.cos(angle))
        self.vy = self.vy * keep + take * (self.a.ball_speed * torch.sin(angle))

    def _place_code(self, value, width: float):
        """Gaussian bump over a third of the PNs. `value` is (N,) in [-1, 1]."""
        torch = self.torch
        span = self.third
        idx = torch.arange(span, device=self.device).unsqueeze(0)
        centre = ((value.clamp(-1, 1) * 0.5 + 0.5) * (span - 1)).unsqueeze(1)
        return self.a.pn_gain * torch.exp(-0.5 * ((idx - centre) / width) ** 2)

    def sense(self):
        """Ball position, ball velocity, paddle position. Three channels."""
        torch = self.torch
        span_y = (H - 2 * BALL) / 2
        pad_span = (H - PAD_H) / 2
        parts = [
            self._place_code((self.by - H / 2) / span_y, 5.0),
            self._place_code(self.vy / max(self.max_vy, 1e-6), 5.0),
            self._place_code((self.y - H / 2) / pad_span, 5.0),
        ]
        pn = torch.zeros(self.a.flies, self.n_pn, device=self.device)
        for i, part in enumerate(parts):
            pn[:, i * self.third:(i + 1) * self.third] = part
        return pn

    def step(self, explore: float):
        """One frame for every fly. Returns the mask of resolved balls."""
        torch = self.torch
        _, _, drives = self.brain.observe(self.sense())
        actions, _ = self.brain.choose(drives, explore)

        gap_before = (self.by - self.y).abs()
        move = torch.tensor(MOVE, device=self.device)[actions]
        self.y = (self.y + move * self.a.fly_speed).clamp(PAD_H / 2,
                                                          H - PAD_H / 2)

        if self.a.dense:
            # Every frame is its own trial. The move either closed the gap to
            # the ball or opened it, and that is the outcome -- immediately,
            # about this decision, with nothing to assign credit across.
            #
            # This is the regime `flybrain teach` reaches 100% in. It is also
            # not the experimenter handing over the answer: a real fly gets
            # graded sensory consequences continuously, not one bit when the
            # ball finally arrives. What is supplied here is feedback the
            # animal plausibly has, not a label it could not obtain.
            #
            # STAY leaves the gap unchanged, so it earns nothing either way
            # and is masked out rather than being scored as a failure.
            delta = gap_before - (self.by - self.y).abs()
            outcome = torch.sign(delta)
            self.brain.reinforce(actions, outcome, outcome != 0)
            self.brain.reset_trace()

        self.bx = self.bx + self.vx
        self.by = self.by + self.vy
        bounced = (self.by < BALL) | (self.by > H - BALL)
        self.vy = torch.where(bounced, -self.vy, self.vy)
        self.by = self.by.clamp(BALL, H - BALL)

        arrived = self.bx >= self.paddle_x - BALL
        if not bool(arrived.any()):
            return arrived, actions
        hit = ((self.by - self.y).abs() < PAD_H / 2 + BALL) & arrived
        if not self.a.dense:
            # Sparse: the only teaching signal is the outcome of the whole
            # flight. In dense mode the per-frame reinforcement has already
            # done the teaching, and adding a terminal bit on top would just
            # attribute the entire ball to its last frame.
            outcomes = torch.where(hit, torch.ones_like(self.y),
                                   -torch.ones_like(self.y))
            self.brain.reinforce(actions, outcomes, arrived)
        self.brain.reset_trace(arrived)
        self.hits = self.hits + hit.float()
        self.balls = self.balls + arrived.float()
        self._serve(arrived)
        return arrived, actions


def main(a) -> int:
    torch = require("torch")
    drill = Drill(a)
    dev = drill.device
    print(f"{a.flies} flies on {dev}, {a.balls} balls each "
          f"({a.flies * a.balls:,} balls total)")
    print(f"3 actions {ACTIONS}, reward on intercept, shock on miss")
    print(f"explore {a.explore} -> {a.explore_final}, "
          f"halflife {a.explore_halflife} balls\n")
    print(f"{'balls':>6s} {'hit rate':>10s} {'across flies':>16s} "
          f"{'depressed':>10s} {'explore':>8s} {'elapsed':>9s}")

    t0 = time.perf_counter()
    frames = 0
    next_report = a.every
    limit = a.balls * 600
    while frames < limit:
        done = float(drill.balls.min().item()) if frames % 64 == 0 else None
        if done is not None and done >= a.balls:
            break
        balls_seen = drill.balls.mean().item()
        frac = 1.0 / (1.0 + balls_seen / max(a.explore_halflife, 1e-9))
        explore = a.explore_final + (a.explore - a.explore_final) * frac
        drill.step(explore)
        frames += 1

        if balls_seen >= next_report:
            rates = (drill.hits / drill.balls.clamp(min=1)).cpu().numpy()
            dep = drill.brain.depressed_fraction().mean().item()
            print(f"{int(balls_seen):6d} {100*rates.mean():9.1f}% "
                  f"{'sd ' + format(100*rates.std(), '.1f'):>16s} "
                  f"{100*dep:9.1f}% {explore:8.3f} "
                  f"{time.perf_counter()-t0:8.0f}s")
            next_report += a.every

    rates = (drill.hits / drill.balls.clamp(min=1)).cpu().numpy()
    total_hits = int(drill.hits.sum().item())
    total_balls = int(drill.balls.sum().item())
    lo, hi = wilson(total_hits, total_balls)
    elapsed = time.perf_counter() - t0

    print(f"\n{total_hits:,} / {total_balls:,} = "
          f"{100*total_hits/total_balls:.1f}% "
          f"(95% CI {100*lo:.1f}-{100*hi:.1f}%)")
    print(f"across {a.flies} flies: mean {100*rates.mean():.1f}%, "
          f"sd {100*rates.std():.1f}, "
          f"range {100*rates.min():.1f}-{100*rates.max():.1f}%")
    # The spread across independent flies is the honest error bar: it is the
    # one that includes the variation a single seed cannot show you.
    sem = rates.std() / np.sqrt(len(rates))
    print(f"standard error of the mean across flies: {100*sem:.2f} points")
    print(f"\n{frames:,} frames in {elapsed:.0f}s "
          f"({a.flies * frames / max(elapsed, 1e-9):,.0f} fly-frames/s)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--flies", type=int, default=128)
    ap.add_argument("--balls", type=int, default=600)
    ap.add_argument("--every", type=int, default=50)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--max-angle", type=float, default=0.9)
    ap.add_argument("--explore", type=float, default=0.6)
    ap.add_argument("--explore-final", type=float, default=0.05)
    ap.add_argument("--explore-halflife", type=float, default=150.0)
    ap.add_argument("--learning-rate", type=float, default=0.35)
    ap.add_argument("--trace-tau", type=float, default=None,
                    help="eligibility decay in frames; defaults to 1 with "
                         "--dense (each frame is a trial) and 40 without")
    ap.add_argument("--lateral", type=float, default=0.0)
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--gain-ceiling", type=float, default=1.0)
    ap.add_argument("--dense", action="store_true",
                    help="reinforce every frame on whether the move closed "
                         "the gap, instead of once per ball on the outcome")
    ap.add_argument("--readout-rate", type=float, default=0.0)
    ap.add_argument("--pn-gain", type=float, default=30.0)
    ap.add_argument("--fly-speed", type=float, default=7.0)
    ap.add_argument("--ball-speed", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    sys.exit(main(ap.parse_args()) or 0)
