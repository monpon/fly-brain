#!/usr/bin/env python
"""Play pong against a fly.

You are the left paddle (mouse, or up/down arrows). The fly is the right one.
It is taught only by the dopamine rule: reward when it intercepts, punishment
when it misses. No gradients, no labels, no target action -- just "that went
well" and "that did not".

It does not work yet, and the window is honest about that. At 1,000 balls
against a returner that never misses:

    actions              hit rate    95% CI       chance    delta
    2  (UP/DOWN)         21.0%       18.6-23.6    20.0%     +1.0
    3  (UP/HOLD/DOWN)    21.4%       19.0-24.0    21.0%     +0.4

Both sit on their own chance baseline, neither trends across blocks, and the
depressed-synapse fraction plateaus near 22% either way -- the fly spends its
whole plastic budget and learns nothing.

The fly makes about 107 decisions per rally and receives one bit at the end
of it. The same two-channel setup learns a single-decision task to 100% (see
docs/FINDINGS.md), so the machinery works and something about spanning time
does not.

Two candidate explanations have now been tested and neither survives.

**Not the paddle's inability to hold position.** It could only ever move, up
or down, every frame. `--actions 3` gives it a HOLD and buys 0.4 points, a
fifth of a standard error. Worth knowing, because "it cannot stand still" is
the first thing anyone notices watching it play.

**Not the length of the eligibility trace**, which was the obvious suspect
and the one this file used to assert. Sweeping `--trace-tau` from 2 to 300
frames with `--trace-normalize` -- 150x, from 32 ms to well past a whole
rally -- moves the hit rate not at all:

    trace_tau      2      25     107     300     (chance 21.0)
    hit rate    20.6    21.9    20.4    21.2
    depressed   15.8    17.7    19.2    20.4

The last row is the control that makes this a result rather than a null: the
trace demonstrably reaches further back and tags monotonically more synapses.
The plasticity changes. The behaviour does not.

So a trace that spans the rally is necessary and nowhere near sufficient.
The likely reason is that it credits all ~107 decisions roughly equally, and
half of them were wrong even in a rally that was won -- lengthening the trace
turns no signal into a very noisy one rather than into a useful one. There is
no baseline, nothing that encodes "better than expected", so the variance is
redistributed rather than reduced.

Note also that `_trace` is never cleared between balls here, so credit bleeds
across the serve at any tau. That is worth fixing before trusting a future
sweep, though it cannot explain a flat result at tau=2.

The experiment that would bisect what is left: dense per-frame reinforcement,
judging each move by whether it closed the gap. That removes the temporal
problem entirely rather than trying to reach across it. If the fly learns,
credit assignment really was the whole story and the question becomes how to
get that signal from something the animal could plausibly have. If it still
does not learn, the problem was never temporal -- it is that a sparse KC code
over two positional bumps cannot express the policy, or that the arbitrary
MBON-to-action split cannot read it out -- and no amount of better credit
will help.

    flybrain pong
    flybrain pong --load output/pong.flyckpt   # a taught fly
    flybrain pong --explore 0 --load output/pong.flyckpt
    flybrain pong --actions 2                  # no HOLD, the original

Press `s` to save the fly's memory, `r` to reset the score, `q` to quit.

Measuring it
------------
`--benchmark N` plays N balls headless against a returner that never misses,
and reports a Wilson interval. This exists because the physics used to be
inside the tkinter callback, so the only way to get a number was to sit and
play -- and because 500 balls cannot answer the question being asked of them:
at ~20% the standard error is 1.75 points, so 19.6 against 18.9 is a fifth of
a standard error. Resolving a three-point effect needs a few thousand balls.

    flybrain pong --benchmark 2000
    flybrain pong --benchmark 2000 --actions 2
    flybrain pong --benchmark 2000 --random-policy   # the chance baseline

The chance rate has to be recomputed per action count rather than carried
over, since a uniform policy over three actions is not the same coin as one
over two. Measured at 300 balls: 20.0% for two actions, 21.0% for three.

How the fly sees the ball
------------------------
Not through the eyes. `vision.py` maps images onto real retinotopic neurons,
but `learning.py` -- the dopamine rule, the only trainer here -- is wired to
the olfactory pathway, PN -> KC -> MBON. So the game state is written into the
projection neuron vector as a place code. That is a stand-in for a sensory
pathway, not a model of one.

Two frames go in: where the ball is now, and where it was last frame, as two
bumps over separate halves of the projection neuron population. Velocity is
not supplied -- the fly has to derive it, and it can, because Kenyon cells
form conjunctions of their inputs and a conjunction of "here now" with "there
before" is a motion detector. That is the same computation T4 and T5 perform
in the optic lobe.

Two channels come out. The mushroom body readout is one column per action
over a disjoint group of MBONs, and reinforcement is gated to the chosen
channel's compartments -- see `learning.action_readout`. Without that gating
dopamine floods every compartment equally, the update does not depend on what
the fly did, and no choice is learnable. Measured: 50% with global dopamine,
100% with it gated.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from flybrain import checkpoint, mushroom_body as MB
from flybrain.learning import FlyBrain, LearningParams

W, H = 720, 460
PAD_H, PAD_W = 78, 10
BALL = 9
MARGIN = 26
DEFAULT_SAVE = "output/pong.flyckpt"

# What each action channel does to the paddle, indexed by action. Two actions
# is up/down and nothing else: the paddle is forced to jump every single frame
# and cannot hold a position even when it is already in the right place. It
# can still *approximate* holding, because sensing is egocentric -- overshoot
# and the ball appears on the other side, so the drive flips -- but it pays
# for that with a permanent dither and spends half its decisions maintaining
# it. Three actions give it somewhere to stand.
MOVES = {
    2: (-1.0, +1.0),
    3: (-1.0, 0.0, +1.0),
}

ACTION_NAMES = {
    2: ("UP", "DOWN"),
    3: ("UP", "HOLD", "DOWN"),
}


def place_code(n: int, value: float, lo: int, hi: int, width: float,
               peak: float) -> np.ndarray:
    """A Gaussian bump over channels [lo, hi), centred by `value` in [-1, 1]."""
    out = np.zeros(n, dtype=np.float32)
    span = hi - lo
    centre = (value * 0.5 + 0.5) * (span - 1)
    idx = np.arange(span)
    out[lo:hi] = peak * np.exp(-0.5 * ((idx - centre) / width) ** 2)
    return out


class Fly:
    """The right-hand paddle, and the brain driving it."""

    def __init__(self, a):
        self.mb = MB.load_cache()
        # Separate behavioural channels, not a sign. With --actions 3 the
        # middle one is "hold", which costs nothing structurally: MBONs are
        # split into that many disjoint readout groups either way.
        self.n_actions = getattr(a, "actions", 2)
        self.moves = MOVES[self.n_actions]
        # trace_tau is in frames here, because `learn` is called once per
        # frame. The default of 2 is therefore ~32 ms at 16 ms/frame, against
        # a rally of ~107 decisions -- the trace is three orders of magnitude
        # too short to reach the moves that placed the paddle. Real Drosophila
        # eligibility traces run to seconds, so spanning a rally is the
        # biologically faithful setting, not a convenient one.
        self.brain = FlyBrain(self.mb, n_actions=self.n_actions,
                              params=LearningParams(
                                  learning_rate=a.learning_rate,
                                  trace_tau=a.trace_tau,
                                  trace_normalize=a.trace_normalize))
        self.random_policy = getattr(a, "random_policy", False)
        self.n_pn = self.mb.n("PN")
        self.rng = np.random.default_rng(a.seed)
        self.explore = a.explore
        self.explore_final = getattr(a, "explore_final", a.explore)
        self.explore_halflife = getattr(a, "explore_halflife", 0.0)
        self.explore_absolute = getattr(a, "explore_absolute", False)
        self.decisions = self.greedy_agree = self.moved = self.toward = 0
        self.pn_gain = a.pn_gain
        self.speed = a.fly_speed
        self.y = H / 2
        self.hits = self.misses = 0
        self.last_pn = None
        self.last_action = 0
        self.last_drives = np.zeros(self.n_actions, dtype=np.float32)
        self.prev_ball_y = H / 2
        if a.load and Path(a.load).exists():
            print(checkpoint.load(a.load, self.brain))

    def sense(self, ball_y: float, prev_ball_y: float) -> np.ndarray:
        """Two frames of the ball, relative to the paddle, as PN activity.

        Both bumps are offsets from the paddle rather than absolute positions,
        so "the ball is above me" means the same thing wherever on the court
        the rally is happening -- the fly learns one rule instead of one per
        location.
        """
        half = self.n_pn // 2
        now = np.clip((ball_y - self.y) / (H / 2), -1.0, 1.0)
        before = np.clip((prev_ball_y - self.y) / (H / 2), -1.0, 1.0)
        return (place_code(self.n_pn, now, 0, half, 6.0, self.pn_gain)
                + place_code(self.n_pn, before, half, self.n_pn, 6.0,
                             self.pn_gain))

    def correct_action(self, ball_y: float) -> int:
        """What a fly that simply tracked the ball would do.

        Only used for the supervised warm-up and for scoring how non-random
        the fly is. Never available to it during play.
        """
        gap = ball_y - self.y
        if self.n_actions == 3 and abs(gap) < self.speed:
            return 1                      # already there: HOLD
        return 0 if gap < 0 else self.n_actions - 1

    def explore_now(self) -> float:
        """Annealed exploration.

        Held fixed, exploration is a permanent ceiling on how decisive the
        animal can be. Decaying it with experience is the standard remedy and
        the one a shaping protocol implies anyway: vary a lot while you know
        nothing, commit once you do.
        """
        if self.explore_halflife <= 0:
            return self.explore
        balls = self.hits + self.misses
        frac = 1.0 / (1.0 + balls / self.explore_halflife)
        return self.explore_final + (self.explore - self.explore_final) * frac

    def step(self, ball_y: float) -> None:
        pn = self.sense(ball_y, self.prev_ball_y)
        self.prev_ball_y = ball_y
        # Tag the synapses active now, without changing them: the eligibility
        # trace is what lets reinforcement arrive later, when the ball finally
        # gets here, and still find the right ones.
        self.brain.learn(pn, update=False)
        if self.random_policy:
            # The chance baseline, which has to be recomputed per action set:
            # a uniform 3-way policy does not miss as often as a uniform 2-way
            # one, because standing still beats thrashing.
            action = int(self.rng.integers(self.n_actions))
            drives = np.zeros(self.n_actions, dtype=np.float32)
        else:
            action, drives = self.brain.choose(
                pn, explore=self.explore_now(), rng=self.rng,
                absolute=self.explore_absolute)

        # How non-random is it? Two separate questions, both worth asking.
        # `agree` is whether exploration overrode the policy -- how much of
        # the behaviour is the fly at all. `toward` is whether the policy is
        # any good: of the decisions that move, how many move at the ball.
        # Chance is 1/n_actions for the first and 50% for the second.
        self.decisions += 1
        if action == int(np.argmax(drives)):
            self.greedy_agree += 1
        if self.moves[action] != 0.0:
            self.moved += 1
            if np.sign(self.moves[action]) == np.sign(ball_y - self.y):
                self.toward += 1

        self.last_pn, self.last_action, self.last_drives = pn, action, drives
        move = self.moves[action] * self.speed
        self.y = float(np.clip(self.y + move, PAD_H / 2, H - PAD_H / 2))

    def reinforce(self, outcome: float) -> None:
        """+1 when it intercepted, -1 when it missed."""
        if self.last_pn is None:
            return
        self.brain.reinforce(self.last_pn, self.last_action, outcome)
        if outcome > 0:
            self.hits += 1
        else:
            self.misses += 1


class Court:
    """Ball, paddles and the rules, with no window attached.

    The physics used to live inside the tkinter callback, which meant the only
    way to get a number out of this game was to sit and play it. Five hundred
    balls of that is an afternoon, and five hundred balls is not enough to
    resolve a few points of hit rate anyway -- at ~19% the standard error is
    1.75 points, so the published 19.6-versus-18.9 is well inside the noise.
    Separating the court from the canvas is what makes the measurement
    possible at all.
    """

    def __init__(self, a, fly, rng):
        self.a = a
        self.fly = fly
        # Seeded, so a benchmark is reproducible. This used the global
        # np.random before, which meant two runs of the same configuration
        # were not comparable.
        self.rng = rng
        self.human_y = H / 2
        # Named, because an index got these backwards once: each side's
        # number was counting its own misses.
        self.points = {"human": 0, "fly": 0}
        self.serve(+1)

    def serve(self, direction):
        self.bx = W / 2
        self.by = float(self.rng.uniform(H * 0.25, H * 0.75))
        speed = self.a.ball_speed
        angle = float(self.rng.uniform(-0.5, 0.5))
        self.vx = direction * speed
        self.vy = speed * np.sin(angle) * 1.6

    def step(self, human_y: float) -> None:
        """One frame: the fly decides, the ball moves, contacts are resolved."""
        self.human_y = human_y
        self.fly.step(self.by)

        self.bx += self.vx
        self.by += self.vy
        if self.by < BALL or self.by > H - BALL:
            self.vy = -self.vy
            self.by = float(np.clip(self.by, BALL, H - BALL))

        # Fly's side.
        fx = W - MARGIN
        if self.vx > 0 and self.bx >= fx - BALL:
            if abs(self.by - self.fly.y) < PAD_H / 2 + BALL:
                self.vx = -abs(self.vx)
                self.vy += (self.by - self.fly.y) * 0.09
                self.fly.reinforce(+1.0)
            else:
                self.fly.reinforce(-1.0)
                self.points["human"] += 1      # the fly missed: your point
                self.serve(+1)

        # Your side.
        hx = MARGIN
        if self.vx < 0 and self.bx <= hx + BALL:
            if abs(self.by - self.human_y) < PAD_H / 2 + BALL:
                self.vx = abs(self.vx)
                self.vy += (self.by - self.human_y) * 0.09
            else:
                self.points["fly"] += 1        # you missed: the fly's point
                self.serve(-1)


class Game:
    def __init__(self, root, a):
        import tkinter as tk

        self.a = a
        self.fly = Fly(a)
        if a.pretrain:
            print(f"warming up on {a.pretrain} single decisions ...",
                  flush=True)
            pretrain(self.fly, a.pretrain, np.random.default_rng(a.seed + 7))
            report = policy_report(self.fly, np.random.default_rng(a.seed + 99))
            print(f"policy agrees with tracking on "
                  f"{100*report['agreement']:.1f}% of positions "
                  f"(chance {100*report['chance']:.1f}%)", flush=True)
        self.court = Court(a, self.fly, np.random.default_rng(a.seed))
        self.human_y = H / 2
        self.canvas = tk.Canvas(root, width=W, height=H, bg="#12131a",
                                highlightthickness=0)
        self.canvas.pack()
        self.status = tk.Label(root, text="", font=("monospace", 10),
                               bg="#12131a", fg="#9aa0b5", anchor="w")
        self.status.pack(fill="x")
        root.bind("<Motion>", self.on_mouse)
        root.bind("<Up>", lambda e: self.nudge(-26))
        root.bind("<Down>", lambda e: self.nudge(26))
        root.bind("q", lambda e: root.destroy())
        root.bind("r", lambda e: self.reset_score())
        root.bind("s", lambda e: self.save())
        self.root = root
        self.tick()

    def on_mouse(self, e):
        self.human_y = float(np.clip(e.y, PAD_H / 2, H - PAD_H / 2))

    def nudge(self, dy):
        self.human_y = float(np.clip(self.human_y + dy, PAD_H / 2,
                                     H - PAD_H / 2))

    def reset_score(self):
        self.court.points = {"human": 0, "fly": 0}
        self.fly.hits = self.fly.misses = 0

    def save(self):
        path = checkpoint.save(self.a.save or DEFAULT_SAVE, self.fly.brain,
                               task="pong",
                               notes=f"{self.fly.hits} hits, "
                                     f"{self.fly.misses} misses")
        print(f"saved -> {path}")

    def tick(self):
        self.court.step(self.human_y)
        self.draw()
        self.root.after(self.a.frame_ms, self.tick)

    def draw(self):
        c = self.canvas
        c.delete("all")
        c.create_line(W / 2, 0, W / 2, H, fill="#262a38", dash=(6, 8))
        c.create_rectangle(MARGIN - PAD_W, self.human_y - PAD_H / 2,
                           MARGIN, self.human_y + PAD_H / 2,
                           fill="#5ac8fa", width=0)
        c.create_rectangle(W - MARGIN, self.fly.y - PAD_H / 2,
                           W - MARGIN + PAD_W, self.fly.y + PAD_H / 2,
                           fill="#ffd166", width=0)
        c.create_oval(self.court.bx - BALL, self.court.by - BALL,
                      self.court.bx + BALL, self.court.by + BALL,
                      fill="#f4f4f6", width=0)
        c.create_text(W / 2 - 44, 30, text=str(self.court.points["human"]),
                      fill="#5ac8fa", font=("monospace", 26))
        c.create_text(W / 2 + 44, 30, text=str(self.court.points["fly"]),
                      fill="#ffd166", font=("monospace", 26))
        c.create_text(W / 2 - 44, 56, text="you", fill="#3d6b85",
                      font=("monospace", 9))
        c.create_text(W / 2 + 44, 56, text="fly", fill="#8a7038",
                      font=("monospace", 9))

        b = self.fly.brain
        seen = self.fly.hits + self.fly.misses
        rate = 100.0 * self.fly.hits / seen if seen else 0.0
        names = ACTION_NAMES[self.fly.n_actions]
        drives = " ".join(f"{d:+.4f}" for d in self.fly.last_drives)
        self.status.config(
            text=(f" fly: {self.fly.hits} hits / {seen} balls ({rate:.0f}%)   "
                  f"{names[self.fly.last_action]:4s} "
                  f"[{drives}]   "
                  f"trials {b.state.trials}   "
                  f"depressed {100*b.depressed_fraction():.1f}%   "
                  f"explore {self.fly.explore:.1f}      "
                  f"[s] save  [r] reset  [q] quit"))


def pretrain(fly, trials: int, rng) -> None:
    """Teach "the ball is over there, move that way" as a single decision.

    The evidence this is worth trying: the same machinery learns a
    single-decision task to 100% (docs/FINDINGS.md), and "ball above me ->
    UP" *is* a single-decision task. Pong only makes it hard by putting 107
    of them between the decision and the consequence.

    So the temporal problem is removed rather than solved. Each trial is one
    ball position, one correct move, reinforcement immediately, and the trace
    cleared in between so nothing bleeds across. This is shaping, which is how
    an animal would actually be trained, and it is a clean diagnostic either
    way: a fly that plays well afterwards proves the policy is representable
    through this readout and the failure is purely in the credit path; one
    that still plays badly says per-frame ball-tracking was never enough and
    the framing needs rethinking.
    """
    for _ in range(trials):
        fly.y = float(rng.uniform(PAD_H / 2, H - PAD_H / 2))
        ball_y = float(rng.uniform(BALL, H - BALL))
        prev = float(np.clip(ball_y - rng.uniform(-12, 12), BALL, H - BALL))
        pn = fly.sense(ball_y, prev)
        fly.brain.reset_trace()
        fly.brain.reinforce(pn, fly.correct_action(ball_y), +1.0)
    fly.brain.reset_trace()
    fly.y = H / 2
    fly.prev_ball_y = H / 2


def policy_report(fly, rng, trials: int = 2000) -> dict:
    """Is the fly's *policy* non-random, independent of how it plays?

    Scored greedily, with exploration off, on ball positions drawn fresh --
    so this measures what the fly has learned rather than what the dice did
    to it during a rally.

    The chance level is obtained by shuffling, not assumed. `1/n_actions` is
    the intuitive answer and it is wrong by a wide margin: HOLD is the correct
    move only ~3% of the time, so the task is effectively a two-way choice
    over near-even labels, and a policy that moves both ways scores about 48%
    while knowing nothing at all. Quoting 33.3% turns a +11 point effect into
    an apparent +26. Permuting the pairing keeps both marginal distributions
    and destroys only the input-to-action mapping, which is the thing being
    measured.
    """
    actions, correct = [], []
    for _ in range(trials):
        fly.y = float(rng.uniform(PAD_H / 2, H - PAD_H / 2))
        ball_y = float(rng.uniform(BALL, H - BALL))
        prev = float(np.clip(ball_y - rng.uniform(-12, 12), BALL, H - BALL))
        actions.append(fly.brain.greedy(fly.sense(ball_y, prev)))
        correct.append(fly.correct_action(ball_y))
    fly.y = H / 2
    fly.prev_ball_y = H / 2

    actions, correct = np.array(actions), np.array(correct)
    shuffle_rng = np.random.default_rng(0)
    null = [float((actions == shuffle_rng.permutation(correct)).mean())
            for _ in range(200)]
    return {
        "agreement": float((actions == correct).mean()),
        "chance": float(np.mean(null)),
        "chance_lo": float(np.percentile(null, 2.5)),
        "chance_hi": float(np.percentile(null, 97.5)),
        "uniform_chance": 1.0 / fly.n_actions,
    }


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval, which behaves at small n where normal does not."""
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def benchmark(a) -> int:
    """Play the fly against a perfect returner, headless, and report.

    The opponent never misses, so every rally comes back and the fly sees the
    maximum number of balls per unit of simulation. The quantity of interest
    is the fly's hit rate on balls that reach it, which this maximises the
    rate of collecting.
    """
    fly = Fly(a)
    rng = np.random.default_rng(a.seed)
    court = Court(a, fly, rng)

    label = "random policy" if a.random_policy else f"explore {a.explore}"
    if not a.random_policy and a.explore_halflife > 0:
        label += f" -> {a.explore_final} (halflife {a.explore_halflife} balls)"
    if a.explore_absolute:
        label += ", absolute"
    print(f"{a.actions} actions ({'/'.join(ACTION_NAMES[a.actions])}), "
          f"{label}, seed {a.seed}")

    before = policy_report(fly, np.random.default_rng(a.seed + 99))
    print(f"policy before: {100*before['agreement']:.1f}% agreement with "
          f"tracking (chance {100*before['chance']:.1f}%)")

    if a.pretrain:
        pretrain(fly, a.pretrain, np.random.default_rng(a.seed + 7))
        mid = policy_report(fly, np.random.default_rng(a.seed + 99))
        print(f"after {a.pretrain} supervised trials: "
              f"{100*mid['agreement']:.1f}% agreement")

    print(f"target {a.benchmark} balls\n")
    print(f"{'balls':>7s} {'block':>8s} {'overall':>9s} {'depressed':>10s}")

    block = max(1, a.benchmark // 10)
    seen = last_seen = last_hits = 0
    frames = 0
    max_frames = a.benchmark * 4000        # generous; rallies are ~137 frames

    while seen < a.benchmark and frames < max_frames:
        # A returner that tracks the ball perfectly, so rallies never end on
        # its side and the fly keeps getting served to.
        court.step(float(np.clip(court.by, PAD_H / 2, H - PAD_H / 2)))
        frames += 1
        seen = fly.hits + fly.misses
        if seen >= last_seen + block:
            b_hits = fly.hits - last_hits
            b_seen = seen - last_seen
            print(f"{seen:7d} {100*b_hits/b_seen:7.1f}% "
                  f"{100*fly.hits/seen:8.1f}% "
                  f"{100*fly.brain.depressed_fraction():9.1f}%")
            last_seen, last_hits = seen, fly.hits

    lo, hi = wilson(fly.hits, seen)
    print(f"\n{fly.hits} / {seen} balls = {100*fly.hits/seen:.1f}% "
          f"(95% CI {100*lo:.1f}-{100*hi:.1f}%)")
    print(f"{frames} frames, {frames/max(seen,1):.0f} decisions per ball")

    # The question this run exists to answer.
    after = policy_report(fly, np.random.default_rng(a.seed + 99))
    d = fly.decisions or 1
    print(f"\nis it random?")
    print(f"  policy agrees with tracking   {100*after['agreement']:5.1f}%  "
          f"(shuffle null {100*after['chance']:.1f}%, "
          f"95% {100*after['chance_lo']:.1f}-{100*after['chance_hi']:.1f})")
    # A different question with a different null: how often the dice left the
    # policy's own choice standing. Picking uniformly would agree 1/n of the
    # time, so that is the right baseline here even though it is the wrong one
    # above.
    print(f"  exploration left policy alone {100*fly.greedy_agree/d:5.1f}%  "
          f"(uniform null {100*after['uniform_chance']:.1f}%)")
    if fly.moved:
        print(f"  moves that went at the ball   "
              f"{100*fly.toward/fly.moved:5.1f}%  (chance 50.0%)")
    print(f"  final exploration             {fly.explore_now():.3f}")
    if a.save and not a.random_policy:
        print(f"saved -> {checkpoint.save(a.save, fly.brain, task='pong', notes=f'{fly.hits}/{seen}')}")
    return 0


def main(a):
    if a.benchmark:
        return benchmark(a)

    import tkinter as tk

    root = tk.Tk()
    root.title("pong against a fly")
    root.configure(bg="#12131a")
    Game(root, a)
    root.mainloop()
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", default=None, help="a .flyckpt to start from")
    # No default path: a benchmark that silently overwrote the fly you had
    # been teaching all afternoon would be a poor trade for a convenience.
    ap.add_argument("--save", default=None,
                    help="where `s` writes, and where --benchmark saves if "
                         f"given at all (default: {DEFAULT_SAVE}, benchmark "
                         "saves nothing)")
    # Was 2.0, which put the noise at twice the entire spread of the drives
    # and made the action essentially a coin flip. 0.05 measured best: a
    # little variability beats none (83.7% against 78.8% greedy over 600
    # balls), presumably by breaking tracking errors a deterministic policy
    # would otherwise repeat.
    ap.add_argument("--explore", type=float, default=0.05,
                    help="behavioural variability; 0 to just perform")
    ap.add_argument("--learning-rate", type=float, default=0.35)
    ap.add_argument("--pn-gain", type=float, default=30.0)
    ap.add_argument("--fly-speed", type=float, default=7.0)
    ap.add_argument("--ball-speed", type=float, default=5.0)
    ap.add_argument("--frame-ms", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--actions", type=int, choices=(2, 3), default=3,
                    help="3 adds HOLD, so the paddle can stand still")
    ap.add_argument("--benchmark", type=int, default=0, metavar="BALLS",
                    help="headless self-play for BALLS balls, then report")
    ap.add_argument("--explore-final", type=float, default=None,
                    help="anneal exploration toward this (default: no anneal)")
    ap.add_argument("--explore-halflife", type=float, default=0.0,
                    metavar="BALLS",
                    help="balls over which exploration halves toward the final")
    ap.add_argument("--explore-absolute", action="store_true",
                    help="fixed-magnitude noise instead of noise scaled to the "
                         "drive spread, which pins signal-to-noise forever")
    ap.add_argument("--pretrain", type=int, default=0, metavar="TRIALS",
                    help="supervised single-decision warm-up before playing")
    ap.add_argument("--trace-tau", type=float, default=2.0,
                    help="eligibility decay in frames; a rally is ~107")
    ap.add_argument("--trace-normalize", action="store_true",
                    help="leaky average rather than sum, so --trace-tau does "
                         "not also scale the learning rate")
    ap.add_argument("--random-policy", action="store_true",
                    help="uniform random actions: the chance baseline, which "
                         "differs per action count")
    args = ap.parse_args()
    if args.explore_final is None:
        args.explore_final = args.explore
    sys.exit(main(args) or 0)
