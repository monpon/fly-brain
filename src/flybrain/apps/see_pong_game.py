"""The court for `see_pong`: a ball, two paddles, and a fly that looks.

Kept separate from the circuit so the visual pathway can be measured without
importing tkinter, which is also how `--benchmark` runs headless.

The fly's paddle is driven entirely by `Eye`: a stimulus is drawn where the
ball is, pushed through the connectome circuit, and the readout decoded to an
elevation the paddle walks toward. Nothing about the ball's velocity or its
future is supplied, and there is no mushroom body and no plasticity anywhere
in it.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from flybrain.apps.pong import BALL, H, MARGIN, PAD_H, PAD_W, W
from flybrain.apps.see_pong import (CATCH, FRONTS, MAX_STEEPNESS,
                                    Eye, Tracker)


class Court:
    """Ball physics and two paddles. No brain in here."""

    def __init__(self, rng, ball_speed=5.0, pad_speed=7.0):
        self.rng = rng
        self.ball_speed = ball_speed
        self.pad_speed = pad_speed
        self.human_y = self.fly_y = H / 2
        self.fly_target = H / 2
        self.points = {"you": 0, "fly": 0}
        # Fly-side encounters, resolved where the contact is resolved. The
        # first version of the benchmark counted *frames* on which the ball
        # was near and inbound -- about eleven per approach -- so forty
        # "balls" were four, and the hit test ran after the ball had already
        # gone past.
        self.fly_hits = self.fly_misses = 0
        # Hit/miss per ball, so a session can be checked for drift rather
        # than summarised by a single rate that averages a decline away.
        self.outcomes = []
        self.serve()

    def serve(self, direction=+1):
        self.bx, self.by = W / 2, float(self.rng.uniform(H * 0.25, H * 0.75))
        angle = float(np.clip(self.rng.uniform(-0.5, 0.5) * 1.6,
                              -np.arcsin(MAX_STEEPNESS),
                              np.arcsin(MAX_STEEPNESS)))
        # Decomposed, so the serve is never faster than a rebound.
        self.vx = direction * self.ball_speed * float(np.cos(angle))
        self.vy = self.ball_speed * float(np.sin(angle))

    def _rebound(self, paddle_y, direction):
        """Reflect off a paddle, conserving speed.

        Adding vy on top of an unchanged vx made the ball accelerate down the
        rally until it was faster than either paddle could reach -- 3.2x the
        serve speed, and it stuck to the walls. Decomposing keeps |v| fixed.
        """
        offset = (self.by - paddle_y) / (PAD_H / 2)
        angle = float(np.arcsin(np.clip(offset, -1, 1) * MAX_STEEPNESS))
        speed = float(np.hypot(self.vx, self.vy))
        self.vx = direction * speed * float(np.cos(angle))
        self.vy = speed * float(np.sin(angle))

    def step(self):
        """One frame. Returns 'you', 'fly' or None if the point continues."""
        # The fly walks toward whatever its eye last decoded.
        delta = self.fly_target - self.fly_y
        self.fly_y = float(np.clip(
            self.fly_y + np.clip(delta, -self.pad_speed, self.pad_speed),
            PAD_H / 2, H - PAD_H / 2))

        self.bx += self.vx
        self.by += self.vy
        if self.by < BALL or self.by > H - BALL:
            self.vy = -self.vy
            self.by = float(np.clip(self.by, BALL, H - BALL))

        if self.bx <= MARGIN + PAD_W + BALL and self.vx < 0:
            if abs(self.by - self.human_y) < CATCH:
                self._rebound(self.human_y, +1)
                self.bx = MARGIN + PAD_W + BALL
            elif self.bx < MARGIN - BALL:
                self.points["fly"] += 1
                self.serve(+1)
                return "fly"
        if self.bx >= W - MARGIN - PAD_W - BALL and self.vx > 0:
            if abs(self.by - self.fly_y) < CATCH:
                self.fly_hits += 1
                self.outcomes.append(1)
                self._rebound(self.fly_y, -1)
                self.bx = W - MARGIN - PAD_W - BALL
            elif self.bx > W - MARGIN + BALL:
                self.fly_misses += 1
                self.outcomes.append(0)
                self.points["you"] += 1
                self.serve(+1)
                return "you"
        return None


def collect(eye, args, balls, tracker=None):
    """Play `balls` balls and record (readout, true ball_y) on inbound frames.

    With `tracker=None` the fly's paddle is driven by the true ball position,
    which is the only option before a decode exists. With a tracker it is
    driven by the decode, which is the point: the states a policy encounters
    depend on the policy, so a decode fitted against a perfect paddle has
    never seen what its own mistakes produce.
    """
    court = Court(np.random.default_rng(args.seed + 5 + int(balls)),
                  args.ball_speed, args.pad_speed)
    returner = np.random.default_rng(args.seed + 77 + int(balls))
    if tracker is not None:
        tracker.reset(H / 2)
    else:
        eye.reset()
    F, Y, G = [], [], []
    frame, last = 0, 0
    seen = 0
    while court.fly_hits + court.fly_misses < balls:
        # Whatever the game will do between balls, do it here too. Carrying
        # state through calibration and clearing it in play (or the reverse)
        # is the same regime mismatch as every other one in this file.
        now_balls = court.fly_hits + court.fly_misses
        if args.reset_on_serve and now_balls != seen:
            if tracker is not None:
                tracker.reset(court.fly_target)
            else:
                eye.reset()
            seen = now_balls
        # Return the ball from a *varied* contact point, not dead centre.
        # A perfect tracker meets every ball at its paddle midpoint, so the
        # rebound angle is ~0 and the ball comes back nearly horizontal and
        # almost never reaches a wall. A human returns it off centre, which
        # produces steep balls and wall bounces -- trajectories a decode
        # calibrated against a perfect returner has never seen. That is
        # exactly where play was observed to fall apart.
        if court.vx < 0:
            court.human_y = float(np.clip(
                court.by + returner.uniform(-PAD_H / 2, PAD_H / 2) * 0.9,
                PAD_H / 2, H - PAD_H / 2))
        if court.vx > 0 and frame % args.brain_every == 0:
            v = eye.look(court.by, court.bx) if tracker is None else None
            if tracker is None:
                court.fly_target = float(np.clip(court.by, PAD_H / 2,
                                                 H - PAD_H / 2))
            else:
                # The tracker looks for us, so read the same glance it used.
                court.fly_target = tracker.update(court.by, court.bx,
                                                  frame - last)
                v = (eye.net.rate(eye.state)[0, eye.read]
                     .detach().cpu().numpy())
            last = frame
            F.append(v)
            Y.append(court.by)
            # One group per ball, so cross-validation holds out whole balls
            # rather than frames sitting beside their own neighbours.
            G.append(court.fly_hits + court.fly_misses)
        court.step()
        frame += 1
    return np.stack(F), np.array(Y), np.array(G)


def calibrate_iteratively(eye, args):
    """Calibrate, play, refit on what playing actually produced, repeat.

    Every regime mismatch in this project came from fitting in one regime and
    running in another, and this is the last and subtlest of them. A decode
    fitted while a perfect paddle returns every ball scores 99% on held-out
    balls and 128 px in play, because once the *decode* drives the paddle it
    sits off centre, rebounds steepen, and the circuit meets trajectories the
    calibration never held. The training distribution depends on the policy,
    so it has to be collected with the policy in the loop.
    """
    F, Y, G = collect(eye, args, args.cal_balls)
    hit, med = eye.fit_and_score(F, Y, groups=G,
                                 label="perfect-paddle rallies")
    tracker = Tracker(eye, args.ball_speed, window=args.median)
    for rnd in range(1, args.rounds):
        f2, y2, g2 = collect(eye, args, args.cal_balls, tracker=tracker)
        F = np.concatenate([F, f2])
        Y = np.concatenate([Y, y2])
        G = np.concatenate([G, g2 + G.max() + 1])
        hit, med = eye.fit_and_score(F, Y, groups=G,
                                     label=f"round {rnd + 1}, with the "
                                           f"decode driving")
    return F, Y, G


def benchmark(eye, args):
    """Headless: how many balls does the eye-driven paddle return?

    The near side is a wall that always returns, so every ball comes back and
    the fly is the only thing being scored. Balls are counted where the
    contact is resolved, not by watching the ball approach.
    """
    def run(driven):
        court = Court(np.random.default_rng(args.seed), args.ball_speed,
                      args.pad_speed)
        returner = np.random.default_rng(args.seed + 1234)
        tracker = Tracker(eye, args.ball_speed, window=args.median)
        if driven:
            tracker.reset(H / 2)
        errs = []
        frame = 0
        last_look = 0
        seen_balls = 0
        while court.fly_hits + court.fly_misses < args.benchmark:
            # Optionally clear the circuit between balls. The membrane state
            # is otherwise carried for a whole session, and a recurrent net
            # with feedback can drift somewhere the decode was never fitted
            # -- which would look exactly like play decaying over time.
            now_balls = court.fly_hits + court.fly_misses
            if driven and args.reset_on_serve and now_balls != seen_balls:
                tracker.reset(court.fly_target)
                seen_balls = now_balls
            # Only look while the ball is coming this way. The calibration
            # set is one crossing per flight, all inbound, so decoding an
            # outbound ball asks the circuit about states it was never fitted
            # on -- which read as 10 px on held-out calibration frames and
            # 96 px in play. It is also what the animal would do.
            if (driven and court.vx > 0
                    and frame % args.brain_every == 0):
                court.fly_target = tracker.update(court.by, court.bx,
                                                  frame - last_look)
                last_look = frame
                errs.append(abs(tracker.last_raw - court.by))
            elif not driven:
                court.fly_target = H / 2
            if court.vx < 0:
                court.human_y = float(np.clip(
                    court.by + returner.uniform(-PAD_H / 2, PAD_H / 2) * 0.9,
                    PAD_H / 2, H - PAD_H / 2))
            court.step()
            frame += 1
        return court, errs, frame

    t0 = time.perf_counter()
    court, errs, frames = run(True)
    blocks = getattr(court, "block_log", [])
    seen = court.fly_hits + court.fly_misses
    el = time.perf_counter() - t0
    parked, _, _ = run(False)

    print()
    print(f"  eye-driven paddle  {court.fly_hits}/{seen} = "
          f"{100 * court.fly_hits / seen:.1f}%")
    print(f"  paddle parked      {parked.fly_hits}/"
          f"{parked.fly_hits + parked.fly_misses} = "
          f"{100 * parked.fly_hits / max(parked.fly_hits + parked.fly_misses, 1):.1f}%"
          f"   (same serves)")
    print(f"  median |estimate - ball_y|  {np.median(errs):.1f} px")
    print(f"  90th percentile             {np.percentile(errs, 90):.1f} px")
    print(f"  {len(errs)} looks over {frames} frames in {el:.0f}s = "
          f"{len(errs) / max(el, 1e-9):.0f} decodes/sec")
    # Does it hold up over a session? With the paddle sitting on the
    # estimate, a wrong estimate sets a steep rebound, a steep ball spends
    # longer near the walls, and the walls are where the decode is worst --
    # a loop that would show up as decline rather than noise.
    out = np.array(court.outcomes, dtype=float)
    if len(out) >= 8:
        chunks = np.array_split(out, 4)
        print("  by quarter of the session: " + "  ".join(
            f"{100 * c.mean():.0f}%" for c in chunks))
    return 0


def _heat(mag):
    """Dim blue to warm orange, so an active cell reads at a glance."""
    lo = (0x2a, 0x33, 0x50)
    hi = (0xff, 0x9f, 0x43)
    return "#%02x%02x%02x" % tuple(
        int(a + (b - a) * mag) for a, b in zip(lo, hi))


def play(eye, args):
    import tkinter as tk

    root = tk.Tk()
    root.title("pong against a fly that can see")
    root.configure(bg="#12131a")
    root.resizable(False, False)
    panel_h = 0 if args.no_brain else 132
    canvas = tk.Canvas(root, width=W, height=H + panel_h, bg="#0c0d12",
                       highlightthickness=0)
    canvas.pack()
    status = tk.Label(root, font=("Consolas", 10), fg="#9aa0b5",
                      bg="#12131a", width=110, anchor="w")
    status.pack(fill="x")

    court = Court(np.random.default_rng(args.seed), args.ball_speed,
                  args.pad_speed)
    tracker = Tracker(eye, args.ball_speed, window=args.median)
    tracker.reset(H / 2)
    held = {"up": False, "down": False}
    for key, which in (("Up", "up"), ("w", "up"), ("W", "up"),
                       ("Down", "down"), ("s", "down"), ("S", "down")):
        root.bind(f"<KeyPress-{key}>",
                  lambda e, w=which: held.__setitem__(w, True))
        root.bind(f"<KeyRelease-{key}>",
                  lambda e, w=which: held.__setitem__(w, False))
    root.bind("<Escape>", lambda e: root.destroy())

    # The readout, drawn cell by cell. Colours are set once and only the
    # fill is updated per frame; rebuilding 143 items every frame was what
    # made the earlier brain panel miss its budget.
    bars = []
    if panel_h:
        canvas.create_line(0, H, W, H, fill="#1c2030")
        n_cells = len(eye.read)
        bw = (W - 24) / n_cells
        for i in range(n_cells):
            x0 = 12 + i * bw
            bars.append(canvas.create_rectangle(
                x0, H + 96, x0 + max(bw - 1, 1), H + 96,
                fill="#2a3350", outline=""))
        canvas.create_text(12, H + 16, anchor="w", fill="#6a7086",
                           font=("Consolas", 9),
                           text=f"{eye.readout}  {n_cells} cells  "
                                f"(what the fly sees)")
        canvas.create_line(12, H + 118, W - 12, H + 118, fill="#1c2030")
        canvas.create_text(12, H + 128, anchor="w", fill="#6a7086",
                           font=("Consolas", 8),
                           text="court height:  white = ball    "
                                "orange = where the fly thinks it is")
    truth_m = canvas.create_rectangle(0, 0, 0, 0, fill="#f2f4ff", outline="")
    guess_m = canvas.create_rectangle(0, 0, 0, 0, fill="#ff9f43", outline="")
    ball = canvas.create_oval(0, 0, 0, 0, fill="#f2f4ff", outline="")
    you = canvas.create_rectangle(0, 0, 0, 0, fill="#5c7cfa", outline="")
    fly = canvas.create_rectangle(0, 0, 0, 0, fill="#ff9f43", outline="")
    guess = canvas.create_line(0, 0, 0, 0, fill="#2d3350", width=2)
    state = {"frame": 0, "looks": 0, "t0": time.perf_counter(),
             "err": 0.0, "rate": 0.0, "last": 0,
             # Physics runs on wall-clock, not on ticks. The eye only decodes
             # while the ball is inbound, a decode costs ~30 ms against a
             # 16 ms budget, and `after()` counts from when the tick
             # *finishes* -- so the ball visibly slowed down on its way in,
             # which is to say it slowed down exactly when the fly was
             # thinking. Accumulating elapsed time and stepping the owed
             # number of frames keeps its speed constant however long a
             # glance takes.
             "clock": time.perf_counter(), "owed": 0.0, "fps": 0.0,
             "balls": 0}

    def tick():
        s = state
        # The decode is fitted under a specific between-ball policy, so the
        # game has to use the same one. Calibrating with resets and playing
        # without them is why the paddle stopped responding.
        balls_now = court.fly_hits + court.fly_misses
        if args.reset_on_serve and balls_now != s["balls"]:
            tracker.reset(court.fly_target)
            s["balls"] = balls_now
        # Your paddle keeps a human speed whatever the fly's is set to.
        if held["up"]:
            court.human_y = max(PAD_H / 2, court.human_y - 7.0)
        if held["down"]:
            court.human_y = min(H - PAD_H / 2, court.human_y + 7.0)

        # The eye looks as often as the circuit allows, not once per frame,
        # and only while the ball is inbound -- see the note in `benchmark`.
        if court.vx > 0 and s["frame"] % args.brain_every == 0:
            court.fly_target = tracker.update(court.by, court.bx,
                                              s["frame"] - s["last"])
            s["last"] = s["frame"]
            s["looks"] += 1
            s["err"] = abs(tracker.last_raw - court.by)
            el = time.perf_counter() - s["t0"]
            if el > 0.5:
                s["rate"] = s["looks"] / el

        now = time.perf_counter()
        budget = args.frame_ms / 1000.0
        s["owed"] += now - s["clock"]
        s["clock"] = now
        # Catch up on the frames the glance cost, but never more than a few:
        # each step moves the ball its full distance, and stepping too many
        # at once would let it pass through a paddle between checks.
        steps = min(int(s["owed"] / budget), 2) or 1
        for _ in range(steps):
            court.step()
            s["frame"] += 1
        s["owed"] = max(0.0, s["owed"] - steps * budget)
        s["fps"] = 1.0 / max(now - s.get("prev", now - budget), 1e-6)
        s["prev"] = now

        canvas.coords(ball, court.bx - BALL, court.by - BALL,
                      court.bx + BALL, court.by + BALL)
        canvas.coords(you, MARGIN, court.human_y - PAD_H / 2,
                      MARGIN + PAD_W, court.human_y + PAD_H / 2)
        canvas.coords(fly, W - MARGIN - PAD_W, court.fly_y - PAD_H / 2,
                      W - MARGIN, court.fly_y + PAD_H / 2)
        # Where the fly thinks the ball is -- the whole of its opinion.
        canvas.coords(guess, W - MARGIN - PAD_W - 26, court.fly_target,
                      W - MARGIN - PAD_W - 6, court.fly_target)
        if (panel_h and tracker.last_vec is not None
                and s["frame"] % args.panel_every == 0):
            v = tracker.last_vec
            hi = float(np.abs(v).max()) or 1.0
            for i, item in enumerate(bars):
                mag = abs(float(v[i])) / hi
                x0, _, x1, _ = canvas.coords(item)
                canvas.coords(item, x0, H + 96 - 72 * mag, x1, H + 96)
                canvas.itemconfig(item, fill=_heat(mag))
            # Belief against truth, on one axis, so the error is visible.
            tx = 12 + (W - 24) * (court.by - BALL) / (H - 2 * BALL)
            gx = 12 + (W - 24) * (court.fly_target - BALL) / (H - 2 * BALL)
            canvas.coords(truth_m, tx - 2, H + 112, tx + 2, H + 124)
            canvas.coords(guess_m, gx - 2, H + 112, gx + 2, H + 124)
        hit = court.fly_hits + court.fly_misses
        rate = 100.0 * court.fly_hits / hit if hit else 0.0
        status.config(text=(
            f" you {court.points['you']:2d}  fly {court.points['fly']:2d}   "
            f"|  {eye.readout} {len(eye.read):3d} cells   "
            f"decode {s['rate']:5.1f}/s   "
            f"error {s['err']:5.1f} px   "
            f"returned {rate:4.0f}% of {hit:3d}   "
            f"(esc quits)"))
        # Schedule from the time this tick started, so a slow glance eats
        # into the wait rather than being added on top of it.
        spent = (time.perf_counter() - now) * 1000.0
        root.after(max(1, int(args.frame_ms - spent)), tick)

    tick()
    root.mainloop()
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--front", choices=tuple(FRONTS), default="minimal",
                   help="visual front end. All three decode equally well; "
                        "minimal is 10,827 neurons and the fastest")
    p.add_argument("--readout", default="LC11",
                   help="cell type the paddle is driven from")
    p.add_argument("--steps", type=int, default=4,
                   help="integration steps per look")
    p.add_argument("--radius", type=float, default=40.0,
                   help="ball radius on the retina, in frame pixels")
    p.add_argument("--brain-every", type=int, default=2, metavar="N",
                   help="look every N display frames; the paddle servos "
                        "toward the last estimate in between")
    p.add_argument("--frame-ms", type=int, default=16)
    p.add_argument("--ball-speed", type=float, default=5.0)
    p.add_argument("--pad-speed", type=float, default=7.0, metavar="PX",
                   help="how fast the fly's paddle travels, in px per frame. "
                        "A large value makes it jump straight to whatever the "
                        "eye decoded, which removes travel time and leaves "
                        "the decode as the only thing being measured")
    p.add_argument("--benchmark", type=int, default=0, metavar="BALLS",
                   help="headless, report the hit rate, then exit")
    p.add_argument("--rounds", type=int, default=3, metavar="N",
                   help="calibration rounds. Round 1 uses a perfect paddle; "
                        "later rounds collect what the decode itself runs "
                        "into, which is the distribution that matters")
    p.add_argument("--cal-balls", type=int, default=30, metavar="N",
                   help="balls per calibration round")
    p.add_argument("--median", type=int, default=3, metavar="N",
                   help="median over the last N glances before the target "
                        "moves; rejects a single bad reading")
    p.add_argument("--calibrate", choices=("rally", "flights"),
                   default="rally",
                   help="'rally' fits the decode on frames from the game "
                        "itself, which is the distribution it will run in")
    p.add_argument("--decode", default="output/see_pong_decode.npz",
                   metavar="PATH",
                   help="where the fitted decode is cached. Reused when it "
                        "matches this circuit, so calibration runs once")
    p.add_argument("--recalibrate", action="store_true",
                   help="ignore any cached decode and fit a fresh one")
    p.add_argument("--reset-on-serve", action="store_true",
                   help="clear the circuit between balls instead of carrying "
                        "membrane state across a whole session")
    p.add_argument("--panel-every", type=int, default=6, metavar="N",
                   help="redraw the readout panel every N frames. It is 143 "
                        "canvas items; redrawing every frame costs more than "
                        "the circuit does")
    p.add_argument("--no-brain", action="store_true",
                   help="hide the readout panel under the court")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"),
                   default="auto")
    a = p.parse_args(argv)

    print(f"building the visual pathway ({a.front} front end) ...",
          flush=True)
    eye = Eye(a.front, a.readout, a.steps, a.radius, a.device)
    print(f"  {eye.summary()}", flush=True)
    # Conditions that shape the calibration distribution, not just the
    # circuit. A decode fitted with a teleporting paddle is not the same
    # decode as one fitted with a paddle that has to walk.
    cond = {"pad_speed": a.pad_speed, "ball_speed": a.ball_speed,
            "brain_every": a.brain_every, "rounds": a.rounds,
            "reset_on_serve": float(bool(a.reset_on_serve))}
    if not a.recalibrate and eye.load(a.decode, conditions=cond):
        pass
    else:
        print("calibrating the decode ...", flush=True)
        if a.calibrate == "rally":
            F, Y, G = calibrate_iteratively(eye, a)
        else:
            cal = eye.calibrate()
            F, Y, G = cal["F"], cal["Y"], None
        saved = eye.save(a.decode, F, Y, G, conditions=cond)
        print(f"  saved to {saved} -- reused next time unless the circuit "
              f"changes or --recalibrate is given")
    return benchmark(eye, a) if a.benchmark else play(eye, a)


if __name__ == "__main__":
    raise SystemExit(main())
