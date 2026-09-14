#!/usr/bin/env python
"""Play pong against a fly.

You are the left paddle (mouse, or up/down arrows). The fly is the right one.
It is taught only by the dopamine rule: reward when it intercepts, punishment
when it misses. No gradients, no labels, no target action -- just "that went
well" and "that did not".

It does not work yet, and the window is honest about that. Measured over 500
balls: 20.0% in the first five blocks, 19.6% in the last, against a chance
rate of 18.9%. No trend. The depressed-synapse fraction meanwhile climbs from
10.7% to 20.5% and plateaus -- the fly spends its whole plastic budget and
learns nothing.

The reason is temporal credit assignment, not a bug in the rule or in the
signs. The fly makes about 137 decisions per rally and receives one bit at
the end of it, which is attributed to the last frame. `trace_tau` is 2 calls
against 137 per rally, so the eligibility trace cannot reach back to the
decisions that actually placed the paddle. The same two-channel setup learns
a single-decision task to 100% (see docs/FINDINGS.md), which is what isolates
this to credit assignment over time rather than to the machinery.

What is untested: dense per-frame reinforcement (judge each move by whether
it closed the gap), and feeding a longer history of positions and velocities
so the fly can see where the ball is going. Both are plausible fixes and
neither has been measured.

    .venv/bin/python scripts/pong.py
    .venv/bin/python scripts/pong.py --load output/pong.flyckpt   # a taught fly
    .venv/bin/python scripts/pong.py --explore 0 --load output/pong.flyckpt

Press `s` to save the fly's memory, `r` to reset the score, `q` to quit.

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import tkinter as tk

import numpy as np

from flybrain import checkpoint, mushroom_body as MB
from flybrain.learning import FlyBrain, LearningParams

W, H = 720, 460
PAD_H, PAD_W = 78, 10
BALL = 9
MARGIN = 26


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
        # UP and DOWN as separate behavioural channels, not a sign.
        self.brain = FlyBrain(self.mb, n_actions=2, params=LearningParams(
            learning_rate=a.learning_rate))
        self.n_pn = self.mb.n("PN")
        self.rng = np.random.default_rng(a.seed)
        self.explore = a.explore
        self.pn_gain = a.pn_gain
        self.speed = a.fly_speed
        self.y = H / 2
        self.hits = self.misses = 0
        self.last_pn = None
        self.last_action = 0
        self.last_drives = np.zeros(2, dtype=np.float32)
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

    def step(self, ball_y: float) -> None:
        pn = self.sense(ball_y, self.prev_ball_y)
        self.prev_ball_y = ball_y
        # Tag the synapses active now, without changing them: the eligibility
        # trace is what lets reinforcement arrive later, when the ball finally
        # gets here, and still find the right ones.
        self.brain.learn(pn, update=False)
        action, drives = self.brain.choose(pn, explore=self.explore,
                                           rng=self.rng)
        self.last_pn, self.last_action, self.last_drives = pn, action, drives
        # channel 0 is up, channel 1 is down
        move = -self.speed if action == 0 else self.speed
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


class Game:
    def __init__(self, root, a):
        self.a = a
        self.fly = Fly(a)
        self.human_y = H / 2
        # Named, because an index got these backwards once: each side's
        # number was counting its own misses.
        self.points = {"human": 0, "fly": 0}
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
        self.serve(+1)
        self.tick()

    def on_mouse(self, e):
        self.human_y = float(np.clip(e.y, PAD_H / 2, H - PAD_H / 2))

    def nudge(self, dy):
        self.human_y = float(np.clip(self.human_y + dy, PAD_H / 2,
                                     H - PAD_H / 2))

    def reset_score(self):
        self.points = {"human": 0, "fly": 0}
        self.fly.hits = self.fly.misses = 0

    def save(self):
        path = checkpoint.save(self.a.save, self.fly.brain, task="pong",
                               notes=f"{self.fly.hits} hits, "
                                     f"{self.fly.misses} misses")
        print(f"saved -> {path}")

    def serve(self, direction):
        self.bx, self.by = W / 2, self.rng_y()
        speed = self.a.ball_speed
        angle = np.random.uniform(-0.5, 0.5)
        self.vx = direction * speed
        self.vy = speed * np.sin(angle) * 1.6

    def rng_y(self):
        return float(np.random.uniform(H * 0.25, H * 0.75))

    def tick(self):
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
        c.create_oval(self.bx - BALL, self.by - BALL,
                      self.bx + BALL, self.by + BALL, fill="#f4f4f6", width=0)
        c.create_text(W / 2 - 44, 30, text=str(self.points["human"]),
                      fill="#5ac8fa", font=("monospace", 26))
        c.create_text(W / 2 + 44, 30, text=str(self.points["fly"]),
                      fill="#ffd166", font=("monospace", 26))
        c.create_text(W / 2 - 44, 56, text="you", fill="#3d6b85",
                      font=("monospace", 9))
        c.create_text(W / 2 + 44, 56, text="fly", fill="#8a7038",
                      font=("monospace", 9))

        b = self.fly.brain
        seen = self.fly.hits + self.fly.misses
        rate = 100.0 * self.fly.hits / seen if seen else 0.0
        self.status.config(
            text=(f" fly: {self.fly.hits} hits / {seen} balls ({rate:.0f}%)   "
                  f"{'UP  ' if self.fly.last_action == 0 else 'DOWN'} "
                  f"[{self.fly.last_drives[0]:+.5f} {self.fly.last_drives[1]:+.5f}]   "
                  f"trials {b.state.trials}   "
                  f"depressed {100*b.depressed_fraction():.1f}%   "
                  f"explore {self.fly.explore:.1f}      "
                  f"[s] save  [r] reset  [q] quit"))


def main(a):
    root = tk.Tk()
    root.title("pong against a fly")
    root.configure(bg="#12131a")
    Game(root, a)
    root.mainloop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", default=None, help="a .flyckpt to start from")
    ap.add_argument("--save", default="output/pong.flyckpt")
    ap.add_argument("--explore", type=float, default=2.0,
                    help="behavioural variability; 0 to just perform")
    ap.add_argument("--learning-rate", type=float, default=0.35)
    ap.add_argument("--pn-gain", type=float, default=30.0)
    ap.add_argument("--fly-speed", type=float, default=7.0)
    ap.add_argument("--ball-speed", type=float, default=5.0)
    ap.add_argument("--frame-ms", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
