#!/usr/bin/env python
"""Teach interception by shaping, the way an animal is actually trained.

Every attempt so far dropped a naive fly into the finished task and hoped
reward would find its way back through ~150 decisions. It does not: the drill
plateaus at 28.8% against a 20.9% floor, and three connectome-derived
plasticity mechanisms -- lateral inhibition, bidirectional plasticity, a
learnable output map -- all failed to move it.

The circuit is not the problem. Supervised single decisions reach 97% policy
agreement in about a hundred trials and 83.7% at pong, through the same
wiring and the same frozen readout. What is missing is the teaching signal:
one bit per ~150 decisions at a ~21% base rate, where most balls return the
same answer whatever the fly did.

Nobody trains an animal that way. They shape it -- start where the
contingency is obvious and tighten it only once the animal has the idea. The
knobs here are the **paddle** and the **ball speed**:

    rung        0      1      2      3      4
    paddle    240    180    140    105     78  px
    speed     3.0    3.5    4.0    4.5    5.0  px/frame
    base rate 56%    43%    34%    27%    21%

Serves stay uniform across the full court at every rung, and the sensory
encoding never changes. Only the slack does. That matters more than it
sounds: the first version of this file shortened the approach and served the
ball near the paddle, which correlated the two channels the fly is supposed
to relate -- 0.85 on the easiest rung against 0.00 at evaluation -- so the
easy rung was not a sub-problem of the hard one but a different problem, and
what the fly learned there was wrong everywhere else.

Each fly carries its own rung and advances on a criterion, not a schedule:
beat your own rung's base rate by `--margin` on a running average and you
move up. Flies that are struggling stay where the contingency is still
legible instead of being dragged into a task they cannot learn from. That is
the entire point of shaping, and it is only practical because the batched
backend runs hundreds of flies at once -- each one is its own curriculum.

The number that matters is printed at the end: every fly re-tested at **full
difficulty** with exploration off, which is the only figure comparable to the
28.8% the unshaped drill reaches.
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

MOVE = (-1.0, 0.0, 1.0)

# (paddle height, ball speed). Both make the task easier without touching the
# statistics the fly has to learn.
#
# The first version of this ladder shortened the approach and served the ball
# near the paddle, and it failed for a reason worth recording. The fly gets
# ball position and paddle position on *separate* channels and has to
# discover the relation between them. Serving relative to the paddle
# correlates those channels -- measured at 0.85 on the easiest rung against
# 0.00 at evaluation -- so there was almost no variance in the gap to learn
# from, and the policy that worked there ("the ball is roughly where I am,
# barely move") is precisely wrong once the correlation goes away. The easy
# rung was not a sub-problem of the hard one, it was a different problem.
#
# A bigger paddle and a slower ball change only how much slack the fly has.
# Serves stay uniform across the full court at every rung, so
# corr(ball_y, paddle_y) is zero throughout and the early task really is the
# final task with more room for error.
# (paddle height, act every k frames).
#
# Ball speed was the wrong second axis and the reason v2 only matched the
# unshaped baseline instead of beating it. Slowing the ball makes the flight
# *longer*, so the easiest rung had 262 decisions between an action and its
# outcome against 157 at the full task -- the curriculum was making credit
# assignment harder at exactly the point it was supposed to be easiest.
#
# Decision frequency is the honest knob. The paddle still moves every frame,
# holding whatever command it was last given, so nothing about the physics or
# the reachability changes; the fly simply commits to a direction for k frames
# instead of reconsidering every one. At k=16 an outcome is about eight
# decisions away, which is inside the eligibility trace and close to the
# single-decision task that already trains to 97%.
#
# This is also the architecture the rest of the project argues for. `cpg.py`
# and `vnc.py` both make the case that the brain issues a low-dimensional
# descending command and the ventral nerve cord executes it -- it does not
# specify a new motor state sixty times a second.
LADDER = (
    (240.0, 16),   # ~8 decisions per ball, base rate ~56%
    (180.0, 8),    # ~17, ~43%
    (140.0, 4),    # ~34, ~34%
    (105.0, 2),    # ~67, ~27%
    (78.0, 1),     # ~134, ~21% -- the full task, identical to `train-pong`
)

# Sensing and the paddle's range of movement both stay pinned to the real
# geometry at every rung. Only the catch width and the ball speed change.
# Scaling the encoding with the paddle would reintroduce exactly the kind of
# distribution shift this ladder exists to avoid.
FULL_PAD = 78.0


class ShapedDrill:
    """N flies, each on its own rung of the curriculum."""

    def __init__(self, a):
        self.torch = require("torch")
        torch = self.torch
        self.a = a
        self.device = pick_device(a.device)
        mb = MB.load_cache()
        d = self.device
        N = a.flies

        self.brain = BatchedBrain(
            mb, N, n_actions=3, device=d, seed=a.seed,
            params=LearningParams(
                learning_rate=a.learning_rate,
                trace_tau=a.trace_tau,
                trace_normalize=True,
            ),
        )
        self.n_pn = mb.n("PN")
        self.third = self.n_pn // 3
        self.pad_h = torch.tensor([s[0] for s in LADDER], device=d)
        self.every_k = torch.tensor([s[1] for s in LADDER], device=d,
                                    dtype=torch.long)
        self.max_stage = len(LADDER) - 1
        # The command currently being held, and how long until the fly is
        # allowed to reconsider it.
        self.held = torch.ones(N, device=d, dtype=torch.long)     # STAY
        self.countdown = torch.zeros(N, device=d, dtype=torch.long)

        # Promotion is judged against each rung's own base rate, not against a
        # single number. A flat threshold means completely different things up
        # the ladder: 55% is below rung 0's 56% chance rate, so a fly is
        # promoted for being lucky, and it is far above rung 4's 21%, so no
        # fly is ever promoted at all. The bar is "meaningfully better than
        # luck, here", which is the same demand at every rung.
        base = torch.tensor(
            [min(1.0, (p + 2 * BALL) / (H - 2 * BALL)) for p, _ in LADDER],
            device=d)
        self.bar = (base + a.margin).clamp(max=0.95)
        self.paddle_x = W - MARGIN
        self.max_vy = a.ball_speed * float(np.sin(a.max_angle))

        self.gen = torch.Generator(device=d)
        self.gen.manual_seed(a.seed + 1)

        self.stage = torch.zeros(N, device=d, dtype=torch.long)
        self.y = torch.full((N,), H / 2.0, device=d)
        self.hits = torch.zeros(N, device=d)
        self.balls = torch.zeros(N, device=d)
        # Running average of recent outcomes, per fly. This is what the
        # promotion criterion reads, so a fly is judged on how it is doing
        # now rather than on everything it has ever done.
        self.recent = torch.zeros(N, device=d)
        self.at_stage = torch.zeros(N, device=d)

        self.bx = torch.empty(N, device=d)
        self.by = torch.empty(N, device=d)
        self.vx = torch.empty(N, device=d)
        self.vy = torch.empty(N, device=d)
        self._serve(torch.ones(N, device=d, dtype=torch.bool))

    def _rand(self, n):
        return self.torch.rand(n, generator=self.gen, device=self.device)

    def _serve(self, who) -> None:
        torch = self.torch
        N = self.a.flies
        angle = (self._rand(N) * 2 - 1) * self.a.max_angle
        speed = self.a.ball_speed
        # Uniform across the full court at every rung, so ball position and
        # paddle position stay independent and the fly is always learning the
        # same relation.
        target = BALL + self._rand(N) * (H - 2 * BALL)

        keep, take = (~who).float(), who.float()
        self.bx = self.bx * keep + take * float(MARGIN)
        self.by = self.by * keep + take * target
        self.vx = self.vx * keep + take * (speed * torch.cos(angle))
        self.vy = self.vy * keep + take * (speed * torch.sin(angle))
        # A new ball gets a fresh decision rather than inheriting whatever
        # command the last one ended on.
        self.countdown = torch.where(who, torch.zeros_like(self.countdown),
                                     self.countdown)

    def _place_code(self, value, width: float = 5.0):
        torch = self.torch
        span = self.third
        idx = torch.arange(span, device=self.device).unsqueeze(0)
        centre = ((value.clamp(-1, 1) * 0.5 + 0.5) * (span - 1)).unsqueeze(1)
        return self.a.pn_gain * torch.exp(-0.5 * ((idx - centre) / width) ** 2)

    def sense(self):
        torch = self.torch
        parts = [
            self._place_code((self.by - H / 2) / ((H - 2 * BALL) / 2)),
            self._place_code(self.vy / max(self.max_vy, 1e-6)),
            self._place_code((self.y - H / 2) / ((H - PAD_H) / 2)),
        ]
        pn = torch.zeros(self.a.flies, self.n_pn, device=self.device)
        for i, part in enumerate(parts):
            pn[:, i * self.third:(i + 1) * self.third] = part
        return pn

    def step(self, explore: float, *, learn: bool = True):
        torch = self.torch
        _, _, drives = self.brain.observe(self.sense())
        fresh, _ = self.brain.choose(drives, explore)

        # A fly only reconsiders when its countdown runs out; otherwise it
        # keeps executing the command it already gave. The eligibility trace
        # still advances every frame, because the Kenyon cells are responding
        # to the world either way -- it is the *decision* that is sparser, not
        # the sensing.
        due = self.countdown <= 0
        actions = torch.where(due, fresh, self.held)
        self.held = actions
        self.countdown = torch.where(due, self.every_k[self.stage],
                                     self.countdown) - 1

        move = torch.tensor(MOVE, device=self.device)[actions]
        self.y = (self.y + move * self.a.fly_speed).clamp(PAD_H / 2,
                                                          H - PAD_H / 2)

        self.bx = self.bx + self.vx
        self.by = self.by + self.vy
        bounced = (self.by < BALL) | (self.by > H - BALL)
        self.vy = torch.where(bounced, -self.vy, self.vy)
        self.by = self.by.clamp(BALL, H - BALL)

        arrived = self.bx >= self.paddle_x - BALL
        if not bool(arrived.any()):
            return arrived
        # The rung's paddle, not the real one. This is the only place the
        # curriculum touches the physics: a wider catch window means more of
        # the fly's attempts are rewarded, and nothing else changes.
        half = self.pad_h[self.stage] / 2 + BALL
        hit = ((self.by - self.y).abs() < half) & arrived

        if learn:
            outcomes = torch.where(hit, torch.ones_like(self.y),
                                   -torch.ones_like(self.y))
            self.brain.reinforce(actions, outcomes, arrived)
            self.brain.reset_trace(arrived)
            self._promote(hit, arrived)

        self.hits = self.hits + hit.float()
        self.balls = self.balls + arrived.float()
        self._serve(arrived)
        return arrived

    def _promote(self, hit, arrived) -> None:
        """Advance the flies that have got the idea at their current rung."""
        torch = self.torch
        a = self.a
        w = a.window
        mask = arrived.float()
        self.recent = torch.where(
            arrived, self.recent * (1 - 1 / w) + hit.float() * (1 / w),
            self.recent)
        self.at_stage = self.at_stage + mask

        ready = (arrived & (self.recent >= self.bar[self.stage])
                 & (self.at_stage >= a.min_balls)
                 & (self.stage < self.max_stage))
        if bool(ready.any()):
            self.stage = self.stage + ready.long()
            # Fresh start at the new rung: the running average described the
            # old difficulty and would otherwise promote the fly straight
            # through several stages on one good streak.
            self.recent = torch.where(ready, torch.zeros_like(self.recent),
                                      self.recent)
            self.at_stage = torch.where(ready, torch.zeros_like(self.at_stage),
                                        self.at_stage)

    def evaluate(self, balls: int, explore: float = 0.0):
        """Re-test every fly at full difficulty, learning off."""
        torch = self.torch
        self.stage = torch.full_like(self.stage, self.max_stage)
        self.hits.zero_()
        self.balls.zero_()
        self.y.fill_(H / 2.0)
        self._serve(torch.ones_like(self.stage, dtype=torch.bool))
        frames = 0
        while float(self.balls.min().item()) < balls and frames < balls * 600:
            self.step(explore, learn=False)
            frames += 1
        return (self.hits / self.balls.clamp(min=1)).cpu().numpy()


    def policy_probe(self, trials: int = 3000):
        """Is the learned policy a tracking rule, on states it never chose?

        Behaviour during play only visits the states the fly steers itself
        into, so a good score there can mean a good policy or a lucky orbit.
        This samples ball and paddle positions uniformly instead, and scores
        the greedy action against "move toward the ball". Chance is obtained
        by shuffling the pairing, because STAY is almost never correct and
        `1/n_actions` badly flatters the result.
        """
        torch = self.torch
        N = self.a.flies
        acts, want = [], []
        for _ in range(trials // N + 1):
            self.y = BALL + self._rand(N) * (H - 2 * BALL)
            self.by = BALL + self._rand(N) * (H - 2 * BALL)
            self.vy = (self._rand(N) * 2 - 1) * self.max_vy
            _, _, drives = self.brain.observe(self.sense())
            gap = self.by - self.y
            correct = torch.where(gap.abs() < self.a.fly_speed,
                                  torch.ones_like(gap),
                                  torch.where(gap < 0, torch.zeros_like(gap),
                                              torch.full_like(gap, 2.0)))
            acts.append(drives.argmax(dim=1).cpu().numpy())
            want.append(correct.long().cpu().numpy())
        acts = np.concatenate(acts)
        want = np.concatenate(want)
        rng = np.random.default_rng(0)
        null = [float((acts == rng.permutation(want)).mean())
                for _ in range(200)]
        return {
            "agreement": float((acts == want).mean()),
            "null": float(np.mean(null)),
            "null_hi": float(np.percentile(null, 97.5)),
            "mix": np.bincount(acts, minlength=3) / len(acts),
        }


def main(a) -> int:
    torch = require("torch")
    drill = ShapedDrill(a)
    print(f"{a.flies} flies on {drill.device}, shaping over "
          f"{len(LADDER)} stages")
    frames_per_ball = (W - 2 * MARGIN) / (a.ball_speed * 0.85)
    print("  rung   paddle   act every   base rate   decisions/ball")
    for i, (pad, every) in enumerate(LADDER):
        base = min(1.0, (pad + 2 * BALL) / (H - 2 * BALL))
        tag = "   <- the full task" if pad <= FULL_PAD and every == 1 else ""
        print(f"  {i:4d} {pad:8.0f} {every:11d} {100*base:10.0f}% "
              f"{frames_per_ball / every:15.0f}{tag}")
    print(f"  promote at base rate + {a.margin:.0%} over a {a.window}-ball average, "
          f"minimum {a.min_balls} balls per stage\n")
    print(f"{'balls':>6s} {'stage':>22s} {'mean':>6s} {'hit rate':>9s} "
          f"{'explore':>8s} {'elapsed':>8s}")

    t0 = time.perf_counter()
    frames, next_report = 0, a.every
    while frames < a.balls * 600:
        seen = drill.balls.mean().item()
        if seen >= a.balls:
            break
        frac = 1.0 / (1.0 + seen / max(a.explore_halflife, 1e-9))
        explore = a.explore_final + (a.explore - a.explore_final) * frac
        drill.step(explore)
        frames += 1

        if seen >= next_report:
            counts = torch.bincount(drill.stage,
                                    minlength=len(LADDER)).cpu().numpy()
            rate = (drill.hits.sum() / drill.balls.sum().clamp(min=1)).item()
            hist = " ".join(f"{c:3d}" for c in counts)
            print(f"{int(seen):6d} {hist:>22s} "
                  f"{drill.stage.float().mean().item():6.2f} "
                  f"{100*rate:8.1f}% {explore:8.3f} "
                  f"{time.perf_counter()-t0:7.0f}s")
            next_report += a.every

    train_time = time.perf_counter() - t0
    counts = torch.bincount(drill.stage, minlength=len(LADDER)).cpu().numpy()
    print(f"\nafter training: stage histogram {counts.tolist()}, "
          f"{int(counts[-1])} of {a.flies} flies reached the full task")

    print(f"\nre-testing all {a.flies} flies at FULL difficulty, "
          f"exploration off, no learning ...")
    rates = drill.evaluate(a.eval_balls)
    hits = int(drill.hits.sum().item())
    total = int(drill.balls.sum().item())
    lo, hi = wilson(hits, total)
    sem = rates.std() / np.sqrt(len(rates))

    print(f"\n{hits:,} / {total:,} = {100*hits/total:.1f}% "
          f"(95% CI {100*lo:.1f}-{100*hi:.1f}%)")
    print(f"across flies: mean {100*rates.mean():.1f}%, "
          f"sd {100*rates.std():.1f}, sem {100*sem:.2f}, "
          f"range {100*rates.min():.1f}-{100*rates.max():.1f}%")
    probe = drill.policy_probe()
    print(f"\npolicy vs a tracking rule: {100*probe['agreement']:.1f}%  "
          f"(shuffle null {100*probe['null']:.1f}%, "
          f"95% up to {100*probe['null_hi']:.1f}%)")
    print(f"  action mix  UP {100*probe['mix'][0]:.1f}%  "
          f"STAY {100*probe['mix'][1]:.1f}%  DOWN {100*probe['mix'][2]:.1f}%")

    print(f"\nreference points at full difficulty:")
    print(f"  geometric floor (paddle parked)      20.9%")
    print(f"  random actions                       21.7%")
    print(f"  unshaped drill, 1000 balls           28.8%")
    print(f"  supervised warm-start (not reward)   83.7%")
    print(f"\n{frames:,} training frames in {train_time:.0f}s")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--flies", type=int, default=128)
    ap.add_argument("--balls", type=int, default=1500,
                    help="training balls per fly, across all stages")
    ap.add_argument("--eval-balls", type=int, default=200)
    ap.add_argument("--every", type=int, default=100)
    ap.add_argument("--margin", type=float, default=0.12,
                    help="how far above a rung's own base rate a fly must "
                         "score before it moves up")
    ap.add_argument("--window", type=float, default=20.0,
                    help="balls in the running average behind the criterion")
    ap.add_argument("--min-balls", type=float, default=30.0)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--max-angle", type=float, default=0.9)
    ap.add_argument("--explore", type=float, default=0.6)
    ap.add_argument("--explore-final", type=float, default=0.05)
    ap.add_argument("--explore-halflife", type=float, default=300.0)
    ap.add_argument("--learning-rate", type=float, default=0.35)
    ap.add_argument("--trace-tau", type=float, default=40.0)
    ap.add_argument("--pn-gain", type=float, default=30.0)
    ap.add_argument("--fly-speed", type=float, default=7.0)
    ap.add_argument("--ball-speed", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    sys.exit(main(ap.parse_args()) or 0)
