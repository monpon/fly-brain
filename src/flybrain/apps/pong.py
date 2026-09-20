#!/usr/bin/env python
"""Play pong against a fly.

You are the left paddle (mouse, or up/down arrows). The fly is the right one.
It is taught only by the dopamine rule: reward when it intercepts, punishment
when it misses. No gradients, no labels, no target action -- just "that went
well" and "that did not".

It works, and how it got there is the interesting part.

    flybrain pong --drill 90        # dense reinforcement first, then play
    flybrain pong                   # sparse reward only: near chance
    flybrain pong --no-brain        # hide the live circuit panel
    flybrain pong --actions 2       # the original two-channel version

Press `s` to save the fly's memory, `r` to reset the score, `q` to quit.

Measured against a returner that never misses, 200 balls, seed 1:

    teaching signal                       hit rate
    random actions                           22.5%
    sparse: one bit per rally                92.0%
    dense: was that move toward the ball    100.0%

Sparse reward works *here*, and that is worth being precise about, because
this file spent most of its history claiming the opposite. It was never a
fact about sparse reward -- it was the parked-paddle bug below. With the
readout fixed, one bit per rally is enough for this encoding.

Where sparse reward genuinely fails is `flybrain train-pong`, which gives the
fly ball position and paddle position on *separate* channels and makes it
discover the relation. There the same rule plateaus at 28.8% against a 19.0%
frozen-gain control, and stays there through MBON->MBON lateral inhibition,
bidirectional plasticity, a learnable output map, and three curricula --
while dense feedback reaches 74.1%. See docs/FINDINGS.md.

So the contrast is not "dense beats sparse" flatly. How much the *structure*
of the feedback matters depends on how much work the encoding has already
done: this file hands over `ball_y - paddle_y` already subtracted, which is
most of the problem solved before the circuit sees it.

The bug underneath all of it
----------------------------
For most of this project's history the fly was not learning slowly, it was
**parked**. `action_readout` sliced a signed vector into contiguous groups,
one per action, and an arbitrary slice has an arbitrary net sum -- a constant
offset per channel. The largest offset won `argmax` regardless of the
stimulus: measured, two actions gave UP on 100.0% of frames and three gave
HOLD on 100.0%. The paddle sat against a wall, and its hit rate was the
geometric odds of a ball arriving in a stationary 96-pixel window, 20.9%.
Every measurement was that constant, which is why nothing ever moved it.

Centring each channel fixed it. Untrained agreement with a tracking rule went
2.2% to 57.6%, and pong went ~21% to 68.7% with no training at all.

What the fly ends up doing
--------------------------
Tracking, not prediction, and the numbers are blunt about it: the paddle sits
6.1 px from where the ball *is* and 137.4 px from where it will *arrive*,
with a lead fraction of -0.04. That is the correct strategy here -- the
paddle moves 7 px per frame against a ball whose vertical speed tops out at
4.25, so following always catches up. It also could not do otherwise: ball
*x* is never supplied, so time-to-arrival is not derivable and the intercept
is uncomputable from what it is given.

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

Three channels come out, one per action. The mushroom body readout gives each
a disjoint group of MBONs, and reinforcement is gated to the chosen channel's
compartments -- see `learning.action_readout`. Without that gating dopamine
floods every compartment equally, the update does not depend on what the fly
did, and no choice is learnable. Measured: 50% with global dopamine, 100%
with it gated.

Which MBON drives which action is *not* in the connectome. The split is
`array_split(arange(97), 3)` -- extraction order -- and it is incoherent: 17
MBON cell types land in more than one channel, so biologically identical
neurons are assigned to opposing commands. It is a placeholder for a
behavioural mapping that would have to come from experiment, and it is the
least faithful part of this file.
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


def action_names(n: int) -> tuple:
    """Channel labels for any count. `--bins` makes the channels court
    positions rather than moves, so they are numbered."""
    return ACTION_NAMES.get(n) or tuple(f"b{i}" for i in range(n))


def place_code(n: int, value: float, lo: int, hi: int, width: float,
               peak: float) -> np.ndarray:
    """A Gaussian bump over channels [lo, hi), centred by `value` in [-1, 1]."""
    out = np.zeros(n, dtype=np.float32)
    span = hi - lo
    centre = (value * 0.5 + 0.5) * (span - 1)
    idx = np.arange(span)
    out[lo:hi] = peak * np.exp(-0.5 * ((idx - centre) / width) ** 2)
    return out


class Aimer:
    """A fly that decides *where to stand*, once per ball.

    Everything else in this file asks the fly for a move every frame -- up,
    down or hold -- and reinforces whether that move closed a gap. The result
    is a follower, and measurably so: the paddle sits 6.1 px from where the
    ball is and 137.4 px from where it will arrive.

    This asks a different question. The instant the ball starts coming back,
    the fly sees its position, its velocity and how far it has to travel, and
    picks **one** of `bins` places to stand. Then it goes there. No further
    decisions, nothing to follow.

    That turns interception into the shape of task this circuit is good at:
    one stimulus, one choice among N, one outcome, immediately. It is the
    structure `flybrain teach` solves at 100% for six odours and 68% for
    thirty-two -- a classification, where the classes are court positions and
    the label is where the ball actually lands.

    Each output channel is a slice of the court, so `argmax` over the readout
    is a place code for "stand here". The mushroom body is being used exactly
    as it is in the animal -- sparse recoding, then a choice among behavioural
    channels -- with the channels meaning positions instead of odour valences.
    """

    def __init__(self, a):
        self.mb = MB.load_cache()
        self.bins = int(a.bins)
        self.brain = FlyBrain(self.mb, n_actions=self.bins,
                              params=LearningParams(
                                  learning_rate=a.learning_rate,
                                  trace_tau=1.0))
        self.n_pn = self.mb.n("PN")
        self.rng = np.random.default_rng(a.seed)
        self.speed = a.fly_speed
        self.pn_gain = a.pn_gain
        self.explore0, self.explore_final = a.explore, a.explore_final
        self.halflife = a.explore_halflife
        self.y = H / 2
        self.target = H / 2
        self.hits = self.misses = 0
        self.last_pn = None
        self.last_action = self.bins // 2
        self.last_drives = np.zeros(self.bins, dtype=np.float32)
        self.decisions = self.moved = self.toward = 0
        self.n_actions = self.bins
        lv = getattr(a, "levels", 0)
        self.levels = (tuple(int(v) for v in lv)
                       if isinstance(lv, (tuple, list)) else int(lv))
        self.smooth = float(getattr(a, "smooth", 1.0))
        # Reward any choice that would have caught the ball, rather than only
        # the exact bin. With 5 bins of 76 px and a 96 px catch window an
        # adjacent bin frequently still catches, and the exact-bin rule
        # punishes it for doing so. It also gives most situations more than
        # one acceptable answer, which is the one lever that relieves the
        # association-capacity limit rather than trading against it.
        self.tolerant = bool(getattr(a, "reward_catch", False))
        # For diagnostics: which bins get used, and how far off each aim was.
        self.aim_log = []
        self.aim_err = []
        self.moves = None

    # -- the court, in bins -------------------------------------------------

    def bin_of(self, y: float) -> int:
        """Which slice of the court a height falls in."""
        lo, hi = PAD_H / 2, H - PAD_H / 2
        frac = (float(y) - lo) / max(hi - lo, 1e-9)
        return int(np.clip(frac * self.bins, 0, self.bins - 1))

    def bin_centre(self, index: int) -> float:
        lo, hi = PAD_H / 2, H - PAD_H / 2
        return lo + (index + 0.5) / self.bins * (hi - lo)

    # -- sensing ------------------------------------------------------------

    def _levels(self) -> tuple:
        """Resolution per channel: height, vertical velocity, distance to go.

        A single number for all three spends resolution where it does not
        buy anything. The landing point is fold(y + vy * t) with t about 150
        frames, so height enters with coefficient 1 and velocity enters
        multiplied by the time of flight -- one step of vy displaces the
        answer ~150x further than one step of y. Measured on the encoding
        ceiling, (pos 4, vel 8, togo 6) reaches 61.9% from 72 addresses while
        a uniform (4, 4, 4) reaches 49.8% from 32, and the reverse skew
        (pos 48, vel 2, togo 2) collapses to 37.1%.
        """
        lv = self.levels
        if isinstance(lv, (tuple, list)):
            return tuple(int(v) for v in lv)
        return (int(lv), int(lv), int(lv))

    def _snap(self, v: float, levels: int | None = None) -> float:
        """Quantise a [-1, 1] value to `levels` steps."""
        n = int(levels if levels is not None else self._levels()[0])
        if n <= 0:
            return v
        idx = int(np.clip((v + 1) / 2 * n, 0, n - 1))
        return (idx + 0.5) / n * 2 - 1

    def sense(self, ball_y: float, ball_vy: float, ball_x: float,
              max_vy: float) -> np.ndarray:
        """One snapshot: where the ball is, where it is going, how long it has.

        Absolute positions, because where the ball ends up depends on where it
        bounces off the walls and the walls are at fixed heights. There is no
        tracking shortcut available here anyway -- by the time the answer is
        known the decision has long been made.
        """
        n = self.n_pn
        third = n // 3
        span = (H - 2 * BALL) / 2
        pos = np.clip((ball_y - H / 2) / span, -1.0, 1.0)
        vel = np.clip(ball_vy / max(max_vy, 1e-6), -1.0, 1.0)
        reach = max(W - 2 * MARGIN, 1.0)
        togo = np.clip((ball_x - MARGIN) / reach * 2 - 1, -1.0, 1.0)

        if self.levels:
            # Coarsen the world before the mushroom body sees it.
            #
            # A Kenyon cell fires for a *conjunction* of projection neurons,
            # so learning "this situation -> stand there" means having a cell
            # tuned to each combination worth distinguishing. With 92 bins per
            # channel and three channels that is ~778,000 combinations against
            # 4,064 Kenyon cells, and almost nothing gets covered -- measured,
            # chance. Quantising to `levels` per channel gives levels^3
            # combinations, and at 4 levels (64 of them) the same circuit
            # reaches 45.9% against a 21.8% baseline.
            #
            # This is the mushroom body used as what it is: a lookup table
            # with a few thousand entries, not a function approximator.
            lp, lv, lt = self._levels()
            pos, vel, togo = (self._snap(v, n) for v, n in
                              ((pos, lp), (vel, lv), (togo, lt)))
        return (place_code(n, pos, 0, third, 5.0, self.pn_gain)
                + place_code(n, vel, third, 2 * third, 5.0, self.pn_gain)
                + place_code(n, togo, 2 * third, n, 5.0, self.pn_gain))

    def explore_now(self) -> float:
        if self.halflife <= 0:
            return self.explore0
        balls = self.hits + self.misses
        frac = 1.0 / (1.0 + balls / self.halflife)
        return self.explore_final + (self.explore0 - self.explore_final) * frac

    # -- deciding and moving ------------------------------------------------

    def aim(self, ball_y: float, ball_vy: float, ball_x: float,
            max_vy: float) -> float:
        """Commit to a place to stand. Called once, when the ball turns."""
        pn = self.sense(ball_y, ball_vy, ball_x, max_vy)
        self.brain.reset_trace()
        action, drives = self.brain.choose(pn, explore=self.explore_now(),
                                           rng=self.rng)
        self.last_pn, self.last_action, self.last_drives = pn, action, drives
        aimed = self.bin_centre(action)

        # Re-deciding every frame makes the goal jitter between adjacent bins,
        # and `approach` only walks at `fly_speed`, so the paddle chases a
        # moving target and never arrives. Measured: the aim was just as good
        # (32.5% of choices inside the catch window against 33.8% for
        # deciding once) while the hit rate fell from 33.8% to 27.8% -- all of
        # the loss was execution, none of it decision.
        #
        # A low-pass on the target keeps the benefit of re-deciding without
        # the chase. `smooth` of 1.0 is the raw choice.
        if self.smooth >= 1.0 or self.target is None:
            self.target = aimed
        else:
            self.target += self.smooth * (aimed - self.target)
        self.decisions += 1
        return self.target

    def approach(self) -> None:
        """Walk toward the chosen spot. No decision is taken here."""
        delta = self.target - self.y
        step = float(np.clip(delta, -self.speed, self.speed))
        self.y = float(np.clip(self.y + step, PAD_H / 2, H - PAD_H / 2))

    def replay_aim(self, landed_at: float, flight) -> None:
        """Teach every decision from the flight against the observed landing.

        One ball, many labelled trials. Each remembered (pattern, choice) pair
        is replayed as its own trial with the trace cleared between, which is
        the regime this rule works in.
        """
        want = self.bin_of(landed_at)
        catch = PAD_H / 2 + BALL
        for pn, action in flight:
            if self.tolerant and abs(self.bin_centre(action)
                                     - landed_at) < catch:
                self.brain.reset_trace()
                self.brain.reinforce(pn, action, +1.0)
                continue
            self.brain.reset_trace()
            self.brain.reinforce(pn, want, +1.0)
            if action != want:
                self.brain.reset_trace()
                self.brain.reinforce(pn, action, -1.0)
        self.brain.reset_trace()

    def reinforce_aim(self, landed_at: float, *, learn: bool = True) -> None:
        """Teach it where the ball went.

        The landing point is *observed*, not computed -- the ball arrives
        somewhere and that is a fact the animal can see. So the target the fly
        should have chosen is available without anyone modelling the physics,
        and the whole flight collapses to a single labelled trial.
        """
        if learn and self.last_pn is not None:
            would_catch = (abs(self.bin_centre(self.last_action) - landed_at)
                           < PAD_H / 2 + BALL)
            if self.tolerant and would_catch:
                # The choice it made would have caught the ball, so reinforce
                # *that*, not whichever bin centre happens to be nearest. The
                # exact-bin rule punishes a working choice for being
                # approximate, which is teaching it to avoid success.
                self.brain.reset_trace()
                self.brain.reinforce(self.last_pn, self.last_action, +1.0)
                self.brain.reset_trace()
            else:
                want = self.bin_of(landed_at)
                self.brain.reset_trace()
                # Reward the bin that was right. With a dozen or more
                # channels, punishing whatever was chosen teaches almost
                # nothing -- most guesses are wrong early and "not that one"
                # barely narrows it.
                self.brain.reinforce(self.last_pn, want, +1.0)
                if self.last_action != want:
                    self.brain.reset_trace()
                    self.brain.reinforce(self.last_pn, self.last_action, -1.0)
                self.brain.reset_trace()

        if self.last_pn is not None:
            self.aim_log.append(int(self.last_action))
            self.aim_err.append(abs(self.bin_centre(self.last_action)
                                    - landed_at))
        if abs(landed_at - self.y) < PAD_H / 2 + BALL:
            self.hits += 1
        else:
            self.misses += 1


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
        # Hindsight uses the same four-channel absolute encoding as predict
        # mode -- it has to, since the intercept depends on absolute wall
        # positions -- but never sees a computed intercept. Same inputs, and
        # the only difference is where the teaching signal comes from.
        self.predict = getattr(a, "predict", False) or getattr(
            a, "hindsight", False)
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

    def sense_predict(self, ball_y: float, prev_ball_y: float,
                      ball_x: float) -> np.ndarray:
        """Four channels: ball now, ball a frame ago, own paddle, and how far
        the ball still has to travel.

        The last one is what makes prediction possible at all. Without it
        there is no time-to-arrival, so the intercept is not a function of the
        inputs and no reward can teach the fly to aim at one -- it can only
        learn to follow. Measured on the tracking version: 6.1 px from where
        the ball is, 137.4 px from where it arrives.

        Positions here are **absolute**, not offsets from the paddle, and that
        is not a detail. Where the ball ends up depends on where it bounces
        off the walls, and the walls are at fixed absolute heights -- so an
        egocentric code cannot express the intercept however much ball-x is
        added to it. Measured with the egocentric version: 40.0% against
        53.8% for plain tracking, worse than not trying.

        The cost is that the fly now has to learn the ball-minus-paddle
        relation itself, which the tracking encoding hands over for free.
        """
        n = self.n_pn
        quarter = n // 4
        span = (H - 2 * BALL) / 2
        now = np.clip((ball_y - H / 2) / span, -1.0, 1.0)
        before = np.clip((prev_ball_y - H / 2) / span, -1.0, 1.0)
        mine = np.clip((self.y - H / 2) / ((H - PAD_H) / 2), -1.0, 1.0)
        reach = max(W - 2 * MARGIN, 1.0)
        togo = np.clip((ball_x - MARGIN) / reach * 2 - 1, -1.0, 1.0)
        return (place_code(n, now, 0, quarter, 5.0, self.pn_gain)
                + place_code(n, before, quarter, 2 * quarter, 5.0,
                             self.pn_gain)
                + place_code(n, mine, 2 * quarter, 3 * quarter, 5.0,
                             self.pn_gain)
                + place_code(n, togo, 3 * quarter, n, 5.0, self.pn_gain))

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

    def step(self, ball_y: float, ball_x: float | None = None) -> None:
        pn = (self.sense_predict(ball_y, self.prev_ball_y, ball_x)
              if self.predict and ball_x is not None
              else self.sense(ball_y, self.prev_ball_y))
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

    def reinforce(self, outcome: float, *, learn: bool = True) -> None:
        """+1 when it intercepted, -1 when it missed.

        `learn=False` keeps the score but skips the synaptic update, for when
        a denser signal is already doing the teaching.
        """
        if self.last_pn is None:
            return
        if learn:
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
        # Decisions taken during the current flight, awaiting the outcome.
        self.flight = []
        # Whether the fly has already chosen a spot for this incoming ball.
        self.committed = False
        # Decisions taken during the current inbound flight (bins mode).
        self.aim_flight = []
        self.serve(+1)

    def serve(self, direction):
        self.bx = W / 2
        self.by = float(self.rng.uniform(H * 0.25, H * 0.75))
        speed = self.a.ball_speed
        # Decompose the speed rather than setting vx to it and adding vy on
        # top: the old version served at up to 1.28x `ball_speed`, so the
        # opening shot was faster than anything the paddles could return it
        # at once rebounds started conserving speed.
        angle = float(self.rng.uniform(-0.5, 0.5)) * 1.6
        angle = float(np.clip(angle, -np.arcsin(self.MAX_STEEPNESS),
                              np.arcsin(self.MAX_STEEPNESS)))
        self.vx = direction * speed * float(np.cos(angle))
        self.vy = speed * float(np.sin(angle))

    # No steeper than this off a paddle, as a fraction of the ball's speed.
    # Below it the ball would cross the court so slowly that a rally stops
    # being a rally.
    MAX_STEEPNESS = 0.85

    def _advance_ball(self) -> None:
        """Ball physics, shared by both control modes."""
        self.bx += self.vx
        self.by += self.vy
        if self.by < BALL or self.by > H - BALL:
            self.vy = -self.vy
            self.by = float(np.clip(self.by, BALL, H - BALL))

    def _resolve_aim(self, *, learn: bool = True) -> None:
        """Contacts, for the one-decision-per-ball mode."""
        fx = W - MARGIN
        if self.vx > 0 and self.bx >= fx - BALL:
            landed = float(self.by)
            caught = abs(landed - self.fly.y) < PAD_H / 2 + BALL
            if learn and self.aim_flight:
                # Hindsight: the landing point is now a fact, so every
                # decision taken during the flight can be graded against it.
                # The last one is scored by `reinforce_aim` below, which also
                # keeps the hit counters.
                self.fly.replay_aim(landed, self.aim_flight[:-1])
            self.aim_flight.clear()
            self.fly.reinforce_aim(landed, learn=learn)
            if caught:
                self._rebound(self.fly.y, -1.0)
            else:
                self.points["human"] += 1
                self.serve(+1)
            self.committed = False

        hx = MARGIN
        if self.vx < 0 and self.bx <= hx + BALL:
            if abs(self.by - self.human_y) < PAD_H / 2 + BALL:
                self._rebound(self.human_y, +1.0)
            else:
                self.points["fly"] += 1
                self.serve(-1)

    def _replay(self, landed_at: float) -> None:
        """Grade the whole flight now that the answer has arrived.

        This is the difference between *computing* the intercept and
        *observing* it. Everywhere else in this file the target comes from
        `intercept()` -- a closed-form extrapolation done in Python, outside
        the circuit, which is really my solution being distilled into the fly.
        Here nothing is computed. The ball lands somewhere, that somewhere is
        a fact the animal can see, and every decision taken during the flight
        is graded against it after the fact.

        It is also a much richer signal than the sparse reward that plateaued.
        A hit-or-miss bit is one number shared across ~150 decisions; this is
        a *target*, so every decision gets its own sign. Same eligibility
        machinery, far more information per ball.

        Each remembered frame is replayed as its own trial, with the trace
        cleared between, because that is the regime this rule works in --
        `flybrain teach` reaches 100% exactly this way.
        """
        for pn, action, y_before, y_after in self.flight:
            closed = abs(y_before - landed_at) - abs(y_after - landed_at)
            if closed == 0.0:
                continue
            self.fly.brain.reset_trace()
            self.fly.brain.reinforce(pn, action, float(np.sign(closed)))
        self.fly.brain.reset_trace()
        self.flight.clear()

    def intercept(self) -> float:
        """Where the ball will reach the fly's paddle, with wall bounces.

        Closed form: extrapolate to the paddle plane, then fold back into the
        court with a triangle wave, which is what perfectly elastic walls do
        to a straight line.

        Only meaningful while the ball is inbound. When it is heading for the
        human the fly has nothing to aim at, so the target falls back to the
        ball itself and the reward is about staying with it.

        In tracking mode this returns the ball's current y, so the reward is
        unchanged and the two modes share one code path.
        """
        if not self.a.predict or self.a.hindsight or self.vx <= 0:
            return float(self.by)
        span = H - 2 * BALL
        t = (W - MARGIN - BALL - self.bx) / max(self.vx, 1e-6)
        y = (self.by + self.vy * t - BALL) % (2 * span)
        if y > span:
            y = 2 * span - y
        return float(y + BALL)

    def _rebound(self, paddle_y: float, direction: float) -> None:
        """Send the ball back, with english, at a constant speed.

        Where you hit the ball on the paddle sets the angle -- that is the
        part worth keeping. What the old version also did was *add* to `vy`
        without renormalising, so every hit made the ball faster: measured
        over a long rally it reached 3.2x the serve speed and |vy| of 15.4
        against a ball 18 px across. At that point it moves nearly its own
        diameter per frame, the single bounce test per frame stops being
        enough, and it jitters along the top and bottom walls -- 73 frames in
        6,000 pinned against one. That reads as the window glitching, and it
        is really the physics running away.

        Speed is now conserved: english changes the direction only.
        """
        speed = float(np.hypot(self.vx, self.vy)) or self.a.ball_speed
        vy = self.vy + (self.by - paddle_y) * 0.09
        # Keep enough horizontal motion that the ball actually crosses.
        vy = float(np.clip(vy, -self.MAX_STEEPNESS * speed,
                           self.MAX_STEEPNESS * speed))
        vx = direction * np.sqrt(max(speed ** 2 - vy ** 2, 1e-9))
        self.vx, self.vy = float(vx), float(vy)

    def step(self, human_y: float, *, dense: bool = False,
             learn: bool = True) -> None:
        """One frame: the fly decides, the ball moves, contacts are resolved.

        `learn=False` freezes the synapses. Evaluation needs it: the benchmark
        used to keep applying sparse terminal reward while measuring, and
        sparse reward does not merely fail to teach a drilled fly, it takes it
        apart -- 100% at ball 40 down to 40% by ball 200. Every "after the
        drill" number measured that way was a policy being destroyed as it was
        read.
        """
        self.human_y = human_y

        if self.a.bins:
            # One decision per ball, taken the instant it turns toward the
            # fly, then a walk to the chosen spot.
            if self.vx > 0:
                # Re-deciding every frame does two things. It refines the
                # target as the ball approaches -- distance-to-travel shrinks,
                # so the estimate gets easier -- and it turns one labelled
                # trial per ball into ~136 of them, which is the difference
                # between the 33.4% a single trial per ball reached and
                # whatever this does.
                #
                # Worth being honest that it also erodes the claim: as the
                # ball nears the paddle the intercept converges on the ball's
                # current position, so a continuously re-aiming fly ends up
                # doing something close to tracking by the end of the flight.
                if self.a.reaim or not self.committed:
                    self.fly.aim(self.by, self.vy, self.bx, self.a.ball_speed)
                    if learn and self.fly.last_pn is not None:
                        self.aim_flight.append((self.fly.last_pn,
                                                self.fly.last_action))
                self.committed = True
            else:
                self.committed = False
            self.fly.approach()
            self._advance_ball()
            self._resolve_aim(learn=learn)
            return

        target = self.intercept()
        gap_before = abs(target - self.fly.y)
        y_before = self.fly.y
        self.fly.step(self.by, self.bx)

        if (learn and self.a.hindsight and self.vx > 0
                and self.fly.last_pn is not None):
            # Remember the decision. Nothing is taught yet -- at this point in
            # the flight nobody knows where the ball is going, the fly least
            # of all.
            self.flight.append((self.fly.last_pn, self.fly.last_action,
                                y_before, self.fly.y))

        if learn and dense and not self.a.hindsight:
            # Every frame is its own trial: the move either closed the gap to
            # the ball or opened it, and that is the outcome, immediately.
            #
            # This is what took the batched drill from 28% to 74%. Sparse
            # reward -- one bit when the ball finally arrives, ~137 decisions
            # later -- never beat 29% under six different plasticity
            # mechanisms and four curricula. The circuit was never the limit;
            # the feedback was.
            #
            # STAY does not move the paddle, so it earns nothing either way
            # rather than being scored as a failure.
            outcome = float(np.sign(gap_before - abs(target - self.fly.y)))
            # In predict mode, say nothing while the ball is heading away.
            # There is no intercept to aim at then, so the target falls back
            # to the ball itself and the fly would be taught to track for half
            # of every rally and to anticipate for the other half. Measured
            # with that contradiction in place: 31.7% against 76.7% for plain
            # tracking. The ball machine in `train-pong-gpu` never hits this,
            # because its ball is always inbound.
            if self.a.predict and self.vx <= 0:
                outcome = 0.0
            if outcome != 0.0 and self.fly.last_pn is not None:
                self.fly.brain.reinforce(self.fly.last_pn,
                                         self.fly.last_action, outcome)
            self.fly.brain.reset_trace()

        self.bx += self.vx
        self.by += self.vy
        if self.by < BALL or self.by > H - BALL:
            self.vy = -self.vy
            self.by = float(np.clip(self.by, BALL, H - BALL))

        # Fly's side.
        fx = W - MARGIN
        if self.vx > 0 and self.bx >= fx - BALL:
            if learn and self.a.hindsight:
                self._replay(float(self.by))
            caught = abs(self.by - self.fly.y) < PAD_H / 2 + BALL
            # In dense mode the per-frame signal has already done the
            # teaching; a terminal bit on top would credit the whole rally to
            # its final frame. The hit still counts for the score either way.
            if caught:
                self._rebound(self.fly.y, -1.0)
                self.fly.reinforce(+1.0, learn=learn and not dense)
            else:
                self.fly.reinforce(-1.0, learn=learn and not dense)
                self.points["human"] += 1      # the fly missed: your point
                self.serve(+1)

        # Your side.
        hx = MARGIN
        if self.vx < 0 and self.bx <= hx + BALL:
            if abs(self.by - self.human_y) < PAD_H / 2 + BALL:
                self._rebound(self.human_y, +1.0)
            else:
                self.points["fly"] += 1        # you missed: the fly's point
                self.serve(-1)


class BrainPanel:
    """A live readout of the circuit, beside the court.

    Five rows, following the signal from the world to the muscles:

        PN      what the fly is being shown, as projection neuron drive
        KC      the sparse code -- ~5% of 4,064 Kenyon cells, the rest dark
        MBON    output cells, coloured by the action channel each belongs to
        drives  the three numbers the choice is an argmax over
        gains   how far each MBON's KC synapses have been depressed

    The last row is the memory. It starts uniform and darkens where dopamine
    has written something, so you can watch learning happen rather than infer
    it from the score.

    Kenyon cells go through a `PhotoImage` rather than 4,064 canvas items:
    one `put` of a whole image is a few hundred microseconds, where four
    thousand `itemconfig` calls would blow the 16 ms frame budget on its own.
    """

    W = 300
    KC_SIDE = 64                      # 64x64 = 4,096 cells, enough for 4,064

    def __init__(self, parent, fly, tk):
        self.tk = tk
        self.fly = fly
        self.canvas = tk.Canvas(parent, width=self.W, height=H,
                                bg="#0d0e14", highlightthickness=0)
        self.canvas.pack(side="left", fill="y")
        c = self.canvas

        def label(y, text):
            c.create_text(10, y, text=text, anchor="w", fill="#5c6478",
                          font=("monospace", 8))

        # -- projection neurons ------------------------------------------
        label(14, "PROJECTION NEURONS  what it sees")
        # Downsampled 3:1. The input is two smooth Gaussian bumps, so 92 bars
        # look identical to 276 and cost a third as many canvas updates --
        # and canvas updates, not arithmetic, are what this panel spends its
        # frame budget on.
        self.pn_group = 3
        self.pn_bars = self._strip(c, 22, 44, fly.n_pn // self.pn_group,
                                   "#5ac8fa")

        # -- kenyon cells --------------------------------------------------
        label(80, f"KENYON CELLS  {fly.brain.n_kc} cells, ~5% active")
        side = self.KC_SIDE
        self.kc_img = tk.PhotoImage(width=side, height=side)
        c.create_image(10, 88, image=self.kc_img, anchor="nw")
        self.kc_dark = "{" + " ".join(["#14161f"] * side) + "}"
        self.kc_img.put(" ".join([self.kc_dark] * side))

        # -- MBONs ---------------------------------------------------------
        label(174, "MBONs  output cells, by action channel")
        n_mbon = fly.brain.n_mbon
        groups = np.array_split(np.arange(n_mbon), fly.n_actions)
        self.mbon_colour = np.empty(n_mbon, dtype=object)
        # Cycled, since --bins gives as many channels as court slices.
        palette = ("#ff6b6b", "#9aa0b5", "#6bcB77")
        for k, idx in enumerate(groups):
            self.mbon_colour[idx] = palette[k % len(palette)]
        self.mbon_bars = self._strip(c, 182, 50, n_mbon, "#9aa0b5")
        # Channel colours are fixed by the readout groups, so set them once
        # rather than re-issuing 97 itemconfig calls every frame.
        for bar, colour in zip(self.mbon_bars, self.mbon_colour):
            c.itemconfig(bar, fill=colour)

        # -- drives ---------------------------------------------------------
        label(248, "DRIVES  the choice is an argmax of these")
        self.drive_bars = []
        self.drive_text = []
        for k in range(fly.n_actions):
            y = 256 + k * 22
            self.drive_bars.append(
                c.create_rectangle(70, y, 70, y + 14,
                                   fill=palette[k % len(palette)],
                                   width=0))
            c.create_text(10, y + 7, text=action_names(fly.n_actions)[k],
                          anchor="w", fill="#9aa0b5", font=("monospace", 8))
            self.drive_text.append(
                c.create_text(self.W - 10, y + 7, text="", anchor="e",
                              fill="#5c6478", font=("monospace", 8)))
        c.create_line(70, 252, 70, 256 + fly.n_actions * 22,
                      fill="#262a38")

        # -- learned gains ---------------------------------------------------
        label(340, "MEMORY  KC->MBON gain per output cell")
        self.gain_bars = self._strip(c, 348, 46, n_mbon, "#ffd166")
        self.note = c.create_text(10, H - 14, text="", anchor="w",
                                  fill="#5c6478", font=("monospace", 8))

    def _strip(self, c, top: int, height: int, n: int, colour: str):
        """`n` vertical bars filling the panel width, created once."""
        bars = []
        x0, span = 10, self.W - 20
        w = span / n
        for i in range(n):
            x = x0 + i * w
            bars.append(c.create_rectangle(x, top + height, x + max(w, 1.0),
                                           top + height, fill=colour,
                                           width=0))
        return bars

    def _fill(self, bars, values, top: int, height: int):
        peak = float(np.max(values)) if len(values) else 0.0
        if peak <= 0:
            peak = 1.0
        c = self.canvas
        tops = top + height - np.asarray(values, dtype=np.float64) / peak * height
        base = top + height
        for bar, y in zip(bars, tops):
            x0, _, x1, _ = c.coords(bar)
            c.coords(bar, x0, float(y), x1, base)

    def update(self) -> None:
        fly = self.fly
        if fly.last_pn is None:
            return
        brain = fly.brain
        # Recomputed rather than threaded through `Fly.step`, which costs a
        # second forward pass (~0.4 ms) and keeps the panel from reaching
        # into the middle of the model.
        kc = brain.kenyon_cells(fly.last_pn)
        mbon = brain.mbon_rates(kc)

        pn = np.asarray(fly.last_pn, dtype=np.float64)
        keep = (len(pn) // self.pn_group) * self.pn_group
        self._fill(self.pn_bars,
                   pn[:keep].reshape(-1, self.pn_group).max(axis=1), 22, 44)
        self._fill(self.mbon_bars, np.abs(mbon), 182, 50)

        # Kenyon cells: active ones lit, laid out row-major.
        side = self.KC_SIDE
        grid = np.zeros(side * side, dtype=np.float32)
        grid[:len(kc)] = kc
        lit = grid > 0
        rows = []
        for r in range(side):
            row = lit[r * side:(r + 1) * side]
            rows.append("{" + " ".join(
                "#ffd166" if v else "#14161f" for v in row) + "}")
        self.kc_img.put(" ".join(rows))

        drives = np.asarray(fly.last_drives, dtype=np.float64)
        span = float(np.abs(drives).max()) or 1e-9
        for k, bar in enumerate(self.drive_bars):
            width = drives[k] / span * (self.W - 90)
            y0 = 256 + k * 22
            self.canvas.coords(bar, 70, y0, 70 + width, y0 + 14)
            self.canvas.itemconfig(
                self.drive_text[k],
                text=("<-- chosen" if k == fly.last_action else ""))

        gains = np.bincount(brain.post_idx, weights=brain.state.gain,
                            minlength=brain.n_mbon)
        counts = np.bincount(brain.post_idx, minlength=brain.n_mbon)
        mean = np.divide(gains, counts, out=np.ones_like(gains),
                         where=counts > 0)
        self._fill(self.gain_bars, mean, 348, 46)
        self.canvas.itemconfig(
            self.note,
            text=f"{int(lit.sum()):4d} KC active   "
                 f"{100*brain.depressed_fraction():.1f}% depressed")


class Game:
    def __init__(self, root, a):
        import tkinter as tk

        self.a = a
        self.fly = Aimer(a) if a.bins else Fly(a)
        if a.drill:
            # The aimer trains on synthetic serves with no flight to
            # simulate, which is both faster and the right shape of trial.
            rate = (aim_drill(a, self.fly, a.drill) if a.bins
                    else dense_drill(a, self.fly, a.drill))
            baseline = "21.8%" if a.bins else "22.5%"
            print(f"drill finished at {100*rate:.1f}% "
                  f"(random scores {baseline} here)", flush=True)
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
        # Court and brain panel side by side in their own row, so the status
        # bar below still spans the full width.
        row = tk.Frame(root, bg="#12131a")
        row.pack()
        self.canvas = tk.Canvas(row, width=W, height=H, bg="#12131a",
                                highlightthickness=0)
        self.canvas.pack(side="left")
        self.panel = (BrainPanel(row, self.fly, tk) if a.brain else None)
        # Belt and braces against the window resizing itself: the label is
        # given a fixed character width so it never asks for more room, and
        # the toplevel is pinned and made non-resizable so it cannot grant it
        # even if something else changes size.
        self.status = tk.Label(root, text="", font=("monospace", 10),
                               bg="#12131a", fg="#9aa0b5", anchor="w",
                               width=120)
        self.status.pack(fill="x")
        root.resizable(False, False)
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
        # A drilled fly keeps being taught the way it was drilled. Switching
        # back to one bit per rally does not merely stop the learning, it
        # actively undoes it: benchmarked straight after a drill, the hit rate
        # fell from 100% at ball 40 to 40% by ball 200 under terminal reward.
        # Sparse feedback is worse than none once there is a good policy to
        # wreck.
        self.court.step(self.human_y, dense=bool(self.a.drill))
        # The panel refreshes on alternate frames. It costs about as much as
        # everything else in a tick put together, and a readout at 30 Hz is
        # indistinguishable from one at 60 -- whereas overrunning the 16 ms
        # budget makes the ball itself stutter, which is not.
        self.frame = getattr(self, "frame", 0) + 1
        if self.panel is not None and self.frame % 2 == 0:
            self.panel.update()
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
        names = action_names(self.fly.n_actions)
        drives = " ".join(f"{d:+.4f}" for d in self.fly.last_drives)
        # Every field is fixed-width. A Label sizes itself to its text and the
        # toplevel sizes itself to the Label, so a status line that grows from
        # "1 hits / 1 balls" to "123 hits / 456 balls" resizes the whole
        # window mid-rally. Padding the numbers keeps the string length
        # constant no matter what the values do.
        self.status.config(
            text=(f" fly: {self.fly.hits:4d} hits /{seen:5d} balls "
                  f"({rate:5.1f}%)   "
                  f"{names[self.fly.last_action]:4s} "
                  f"[{drives}]   "
                  f"trials {b.state.trials:6d}   "
                  f"depressed {100*b.depressed_fraction():5.1f}%   "
                  f"explore {self.fly.explore_now():5.3f}      "
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


def landing_of(ball_y: float, vy: float, vx: float, ball_x: float) -> float:
    """Where a ball launched from here will reach the fly's paddle.

    Closed form: extrapolate to the paddle plane, then fold back into the
    court with a triangle wave, which is what perfectly elastic walls do to a
    straight line. Used only to *generate* training balls quickly -- in the
    game the landing point is observed, not computed.
    """
    span = H - 2 * BALL
    t = (W - MARGIN - BALL - ball_x) / max(vx, 1e-6)
    y = (ball_y + vy * t - BALL) % (2 * span)
    if y > span:
        y = 2 * span - y
    return float(y + BALL)


def aim_drill(a, fly, balls: int, *, quiet: bool = False) -> float:
    """Teach the aimer on synthetic serves, without simulating the flight.

    The flight is ~136 frames of physics that produce exactly one learning
    trial, so running it is pure overhead for training. Sampling ball states
    directly gives the same trials thousands of times faster, and the task is
    then literally `flybrain teach`: one stimulus, one choice among N, one
    outcome, immediately -- the regime this circuit solves at 100%.

    It is harder than `teach` in one way that matters. There the stimuli are
    six fixed odours; here the input is continuous in three dimensions, so the
    fly is learning a *mapping* rather than memorising a handful of patterns.
    That needs far more trials.
    """
    rng = np.random.default_rng(a.seed + 11)
    block = max(1, balls // 10)
    right = 0
    log = []
    if not quiet:
        print(f"drilling {balls} synthetic serves, one decision each "
              f"({fly.bins} bins, {fly.levels or 'continuous'} input levels)")
        print(f"{'serves':>7s} {'in window':>10s} {'median err':>11s} "
              f"{'explore':>8s}")

    for i in range(1, balls + 1):
        ball_y = float(rng.uniform(BALL, H - BALL))
        angle = float(rng.uniform(-0.5, 0.5)) * 1.6
        angle = float(np.clip(angle, -np.arcsin(Court.MAX_STEEPNESS),
                              np.arcsin(Court.MAX_STEEPNESS)))
        vx = a.ball_speed * float(np.cos(angle))
        vy = a.ball_speed * float(np.sin(angle))
        # Where along the flight the fly is asked has to match how it will
        # be used. Committing once means deciding the moment the ball turns,
        # so the left half is the whole world. `--reaim` asks again on every
        # frame, and then the ball is seen right across the court -- training
        # only on the left half leaves every late-flight decision out of
        # distribution, which is part of why re-aiming measured worse than
        # committing once rather than better.
        x_max = (W - MARGIN - BALL) if getattr(a, "reaim", False) else W / 2
        ball_x = float(rng.uniform(MARGIN, x_max))
        landed = landing_of(ball_y, vy, vx, ball_x)

        fly.y = fly.bin_centre(fly.bin_of(rng.uniform(PAD_H / 2,
                                                     H - PAD_H / 2)))
        fly.aim(ball_y, vy, ball_x, a.ball_speed)
        err = abs(fly.bin_centre(fly.last_action) - landed)
        log.append(err)
        right += int(err < PAD_H / 2 + BALL)
        fly.reinforce_aim(landed)

        if not quiet and i % block == 0:
            recent = np.array(log[-block:])
            print(f"{i:7d} {100*(recent < PAD_H/2 + BALL).mean():9.1f}% "
                  f"{np.median(recent):10.1f} px {fly.explore_now():8.3f}")

    fly.hits = fly.misses = 0
    fly.aim_log.clear()
    fly.aim_err.clear()
    fly.y = H / 2
    return right / max(balls, 1)


def dense_drill(a, fly, balls: int, *, quiet: bool = False) -> float:
    """Train against a ball machine with per-frame reinforcement.

    The fly plays a returner that never misses, and is reinforced every frame
    on whether its move closed the gap to the ball rather than once per rally
    on whether it caught it. Measured on the batched backend, that is the
    difference between 28% and 74%.

    Nothing here is supervision: the fly is never told which way to move, only
    whether what it did helped. A real animal has exactly that -- continuous
    sensory consequences -- and does not wait for a terminal verdict.
    """
    court = Court(a, fly, np.random.default_rng(a.seed + 3))
    start = fly.hits + fly.misses
    block = max(1, balls // 8)
    last = (fly.hits, start)
    if not quiet:
        print(f"drilling {balls} balls with per-frame reinforcement ...")
        print(f"{'balls':>7s} {'block':>8s} {'overall':>9s} {'depressed':>10s}")

    frames = 0
    while (fly.hits + fly.misses) - start < balls and frames < balls * 4000:
        # A returner that tracks perfectly, so every rally comes back and the
        # fly sees the maximum number of balls per second of simulation.
        court.step(float(np.clip(court.by, PAD_H / 2, H - PAD_H / 2)),
                   dense=True)
        frames += 1
        seen = fly.hits + fly.misses
        if not quiet and seen - last[1] >= block:
            print(f"{seen - start:7d} "
                  f"{100*(fly.hits-last[0])/max(seen-last[1],1):7.0f}% "
                  f"{100*fly.hits/max(seen,1):8.1f}% "
                  f"{100*fly.brain.depressed_fraction():9.1f}%")
            last = (fly.hits, seen)

    rate = fly.hits / max(fly.hits + fly.misses, 1)
    fly.hits = fly.misses = 0          # the window starts from a clean score
    fly.y = H / 2
    fly.prev_ball_y = H / 2
    fly.brain.reset_trace()
    return rate


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
    fly = Aimer(a) if a.bins else Fly(a)
    rng = np.random.default_rng(a.seed)
    court = Court(a, fly, rng)

    label = "random policy" if a.random_policy else f"explore {a.explore}"
    if not a.random_policy and a.explore_halflife > 0:
        label += f" -> {a.explore_final} (halflife {a.explore_halflife} balls)"

    if a.bins:
        # A place code over the court, not a set of moves, so the
        # tracking-agreement probe does not apply -- there is nothing to track.
        print(f"{a.bins} bins of {(H - PAD_H) / a.bins:.0f} px, one decision "
              f"per ball, {label}, seed {a.seed}")
        bin_px = (H - PAD_H) / a.bins
        window = PAD_H + 2 * BALL
        wider = "wider than" if bin_px > window else "narrower than"
        print(f"a bin is {bin_px:.0f} px, {wider} the paddle's "
              f"{window:.0f} px catch window")
    else:
        if a.explore_absolute:
            label += ", absolute"
        print(f"{a.actions} actions ({'/'.join(ACTION_NAMES[a.actions])}), "
              f"{label}, seed {a.seed}")
        before = policy_report(fly, np.random.default_rng(a.seed + 99))
        print(f"policy before: {100*before['agreement']:.1f}% agreement with "
              f"tracking (chance {100*before['chance']:.1f}%)")

    if a.drill:
        if a.bins:
            rate = aim_drill(a, fly, a.drill)
            print(f"\nafter {a.drill} drilled serves: "
                  f"{100*rate:.1f}% landed in the catch window\n")
        else:
            rate = dense_drill(a, fly, a.drill)
            mid = policy_report(fly, np.random.default_rng(a.seed + 99))
            print(f"after {a.drill} drilled balls: {100*rate:.1f}% during the "
                  f"drill, policy now {100*mid['agreement']:.1f}%")

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
        # After a drill, freeze. Otherwise the benchmark's own sparse terminal
        # reward unpicks what the drill built while the number is being read,
        # and the result describes the destruction rather than the policy.
        court.step(float(np.clip(court.by, PAD_H / 2, H - PAD_H / 2)),
                   learn=not a.drill)
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

    if a.bins:
        # For a place-code readout the question is different: how close does
        # the chosen bin land to where the ball actually did?
        print(f"\naiming accuracy over the last balls")
        print(f"  bins used: {len(set(fly.aim_log))} of {a.bins}")
        if fly.aim_err:
            err = np.array(fly.aim_err)
            print(f"  median |chosen - actual| {np.median(err):5.1f} px  "
                  f"(catch window {PAD_H / 2 + BALL:.0f} px, "
                  f"court {H - PAD_H:.0f} px)")
            print(f"  within the window        "
                  f"{100 * (err < PAD_H / 2 + BALL).mean():5.1f}%")
        print(f"  final exploration        {fly.explore_now():.3f}")
        if a.save:
            print(f"saved -> {checkpoint.save(a.save, fly.brain, task='pong-aim', notes=f'{fly.hits}/{seen}')}")
        return 0

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
    ap.add_argument("--smooth", type=float, default=1.0,
                    help="low-pass on the target when re-aiming; 1.0 is raw, "
                         "0.2 keeps the paddle from chasing a jittering goal")
    ap.add_argument("--reaim", action="store_true",
                    help="re-decide the target every frame instead of once "
                         "per ball, and learn from every decision in the "
                         "flight once the landing point is known")
    ap.add_argument("--levels", type=int, default=0, metavar="K",
                    help="quantise each input channel to K steps before the "
                         "mushroom body sees it. K^3 combinations must fit in "
                         "~4000 Kenyon cells; 4 works, continuous does not")
    ap.add_argument("--bins", type=int, default=0, metavar="N",
                    help="one decision per ball: the fly sees the ball turn, "
                         "picks one of N places to stand, and goes there. "
                         "Try 12")
    ap.add_argument("--levels-pos", type=int, default=0, metavar="K",
                    help="resolution for ball height alone; 0 uses --levels")
    ap.add_argument("--levels-vel", type=int, default=0, metavar="K",
                    help="resolution for vertical velocity alone. Buys no "
                         "update rate under --reaim: velocity sits still "
                         "between wall bounces, so finer steps only add "
                         "states the flight never visits")
    ap.add_argument("--levels-togo", type=int, default=0, metavar="K",
                    help="resolution for distance still to travel. The only "
                         "channel that moves through a flight, so it alone "
                         "sets how often --reaim can change its mind: 4 "
                         "gives 2.4 decisions/sec, 12 gives 5.7")
    ap.add_argument("--reward-catch", action="store_true",
                    help="with --bins: reward any aim that would have caught "
                         "the ball, not only the exact bin. An adjacent bin "
                         "often still catches, and the exact-bin rule "
                         "punishes it for doing so. Measured: no significant "
                         "effect either way at 5 bins, where only 1.2 bins "
                         "catch a typical ball")
    ap.add_argument("--hindsight", action="store_true",
                    help="learn from where the ball actually landed, graded "
                         "after the fact. No intercept is computed -- the "
                         "target is observed, so the prediction is the "
                         "circuit's rather than the experimenter's")
    ap.add_argument("--predict", action="store_true",
                    help="aim at where the ball will arrive instead of where "
                         "it is: adds a ball-x channel and rewards progress "
                         "toward the computed intercept. Pair with a low "
                         "--fly-speed so tracking cannot keep up")
    ap.add_argument("--no-brain", dest="brain", action="store_false",
                    help="hide the live circuit panel beside the court")
    ap.add_argument("--drill", type=int, default=0, metavar="BALLS",
                    help="dense per-frame reinforcement against a ball "
                         "machine before playing. Reward and shock only, no "
                         "supervision")
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
    # Aimer takes either one number or a (height, velocity, distance) triple.
    # The CLI offers both; fold the second into the first here so nothing
    # downstream has to know which was given.
    _per = (args.levels_pos, args.levels_vel, args.levels_togo)
    if any(_per):
        args.levels = tuple(v or args.levels for v in _per)
    sys.exit(main(args) or 0)
