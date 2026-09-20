"""Pong against a fly that actually looks at the ball.

Every other pong in this package hands the fly numbers. `pong.py` place-codes
ball_y, ball_vy, paddle_y and ball_x into 276 *olfactory* projection neurons,
and the mushroom body learns an odour-to-action association that we relabel
as playing. It works, and the fly is not seeing anything.

This one has no mushroom body in it. A visual stimulus goes into the
photoreceptor-mapped input of a connectome circuit, and the paddle is driven
by what comes out of the lobula columnar cells:

    ball on screen -> retinotopic columns -> lamina -> medulla -> LC11
      -> decoded elevation -> paddle

LC11 is the small-object detector, and it is the best localiser in the
circuit: 143 cells place a held-out ball accurately enough to intercept 84.4%
of the time, matching all 1,310 descending neurons and beating the 3,206-cell
LC population.

Two things are worth being plain about.

The decode is a kernel regression fitted offline on a calibration grid, so
the *information* is the connectome's and the *extraction* is not the fly's.
Making it the fly's own is the obvious next step -- `BrainNet` is
differentiable, and `LearningParams.readout_rate` gives a reward-driven
output map -- and neither is done here.

And the circuit does not run at 60 fps. It decodes as often as it can and the
paddle servos toward the most recent estimate, which is the arrangement a
real animal has anyway: sensing slower than its own movement. The status line
reports the true decode rate rather than hiding it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from flybrain.apps.pong import BALL, H, MARGIN, PAD_H, W

CATCH = PAD_H / 2 + BALL
MAX_STEEPNESS = 0.85

# Front ends, to be traded off against frame rate. LC11's measured inputs are
# T2/T2a/T3 and the Tm/TmY arrays; the medulla types beyond those cost frames
# without obviously earning accuracy, which is what `--front` exists to test.
FRONTS = {
    "full": ["L1", "L2", "L3", "L5", "Mi1", "Mi4", "Mi9", "Tm1", "Tm2",
             "Tm3", "Tm4", "Tm6", "Tm9", "Tm20", "Tm5Y", "TmY3", "TmY18",
             "TmY5a", "Y3", "T2", "T2a", "T3", "T4[a-d]", "T5[a-d]"],
    "trimmed": ["L1", "L2", "L3", "L5", "T2", "T2a", "T3", "Tm3", "Tm4",
                "Tm5Y", "TmY3"],
    "minimal": ["L1", "L2", "L3", "L5", "T2", "T3"],
}


class Eye:
    """A connectome circuit with a stimulus in one end and a decode out."""

    def __init__(self, front, readout, steps, radius, device="auto"):
        import torch

        from flybrain import circuits, trainable, vision as V

        self.torch, self.V = torch, V
        self.circuit = circuits.from_types(FRONTS[front] + [readout],
                                           ["L1", "L2", "L3", "L5"],
                                           [readout])
        types = self.circuit.types.astype(str)
        self.eyes = V.Binocular(self.circuit)
        self.fh, self.fw = self.eyes.frame_size()
        self.device = trainable.pick_device(device)
        self.net = trainable.BrainNet(self.circuit, steps=steps, tau=2.0,
                                      seed=0, device=self.device)
        self.w = (self.net.w_anat * torch.exp(self.net.log_gain)).detach()
        self.bias = self.net.bias.detach()
        self.in_gain = self.net.in_gain.detach()
        self.k = self.net._k()
        self.steps = steps
        self.radius = radius
        self.read = torch.tensor(np.nonzero(types == readout)[0],
                                 device=self.device)
        self.readout = readout
        # Membrane state persists between looks, as in the animal.
        self.state = torch.zeros(1, self.circuit.n, device=self.device)
        self._decode = None

    def summary(self):
        return (f"{self.circuit.summary()}, {len(self.read)} "
                f"{self.readout} cells, {self.device}")

    def _inject(self, ball_y, ball_x):
        torch = self.torch
        fy = self.fh * (0.15 + 0.7 * (ball_y - BALL) / (H - 2 * BALL))
        fx = self.fw * (0.2 + 0.6 * (ball_x - MARGIN) / (W - 2 * MARGIN))
        img = self.V.disc((self.fh, self.fw), radius_px=self.radius,
                          centre=(fy, fx), value=0.0, background=0.5)
        u = torch.tensor(self.eyes(img, img)[None, :], dtype=torch.float32,
                         device=self.device)
        return torch.zeros(1, self.circuit.n, device=self.device).index_add(
            1, self.net.in_idx, u * self.in_gain)

    def look(self, ball_y, ball_x):
        """One glance: advance the circuit, return the readout rates."""
        torch = self.torch
        inj = self._inject(ball_y, ball_x)
        for _ in range(self.steps):
            r = self.net.rate(self.state)
            drive = torch.zeros(1, self.circuit.n,
                                device=self.device).index_add(
                1, self.net.post, r[:, self.net.pre] * self.w)
            self.state = self.state + self.k * (
                -self.state + drive + self.bias + inj)
        return self.net.rate(self.state)[0, self.read].detach().cpu().numpy()

    def reset(self):
        self.state = self.torch.zeros(1, self.circuit.n, device=self.device)

    # -- the decode -------------------------------------------------------

    @staticmethod
    def _sq(A, B):
        return (A * A).sum(1)[:, None] + (B * B).sum(1)[None] - 2 * A @ B.T

    def _fit(self, F, Y, lam=1e-3, gamma=1.0):
        mu, sd = F.mean(0), F.std(0).mean() + 1e-12
        A = (F - mu) / sd
        d = self._sq(A, A)
        g = gamma / max(np.median(d[d > 0]), 1e-12)
        alpha = np.linalg.solve(np.exp(-g * d) + lam * np.eye(len(Y)),
                                Y - Y.mean())
        self._decode = (mu, sd, g, A, alpha, float(Y.mean()))

    def estimate(self, vec):
        mu, sd, g, A, alpha, ybar = self._decode
        B = ((vec - mu) / sd)[None, :]
        out = np.exp(-g * self._sq(B, A)) @ alpha + ybar
        # Reshaped rather than float()-ed: NumPy 2 refuses float() on a
        # 1-element 1-D array, which threw away a finished 6-minute run once.
        return float(np.asarray(out).reshape(-1)[0])

    def _score(self, F, Y, groups=None, folds=5):
        """Held-out accuracy, as would-the-paddle-have-caught-it.

        Not rms: the error distribution has a tight centre and a heavy tail,
        so squaring reports the tail and hides the centre. Measured that way
        this circuit read as a hopeless 63 px floor; measured as hits it is
        84%.

        And split by **group**, not by frame. Consecutive frames of a rally
        are nearly identical, so a random frame split leaves a near-duplicate
        of every held-out sample in the training set -- which reported 99.9%
        and 0.0 px median for a decode that was looking up its own neighbour.
        Passing one group per ball makes the held-out balls genuinely unseen.
        """
        rng = np.random.default_rng(0)
        if groups is None:
            order = rng.permutation(len(Y))
            splits = np.array_split(order, folds)
        else:
            uniq = rng.permutation(np.unique(groups))
            splits = [np.nonzero(np.isin(groups, chunk))[0]
                      for chunk in np.array_split(uniq, folds)]
        errs = np.zeros(len(Y))
        keep = self._decode
        for te in splits:
            tr = np.setdiff1d(np.arange(len(Y)), te)
            if len(tr) < 5 or len(te) == 0:
                continue
            self._fit(F[tr], Y[tr])
            errs[te] = [abs(self.estimate(F[i]) - Y[i]) for i in te]
        self._decode = keep
        return 100 * float((errs < CATCH).mean()), float(np.median(errs))

    def calibrate(self, flights=16, every=3, quiet=False):
        """Fit elevation from the readout, on a ball that is moving.

        Calibrating on static stimuli settled from rest and then running with
        persistent membrane state is a regime mismatch, and it is expensive:
        it read 11.6 px on the grid and 104 px in play. So the calibration
        set is collected the way the loop will actually run -- a ball
        crossing the court, state carried from frame to frame, sampled every
        few frames.
        """
        rng = np.random.default_rng(0)
        span = H - 2 * BALL
        F, Y = [], []
        self.reset()
        for _ in range(flights):
            y0 = float(rng.uniform(BALL, H - BALL))
            ang = float(np.clip(rng.uniform(-0.5, 0.5) * 1.6,
                                -np.arcsin(MAX_STEEPNESS),
                                np.arcsin(MAX_STEEPNESS)))
            vy, vx = 5.0 * np.sin(ang), 5.0 * np.cos(ang)
            x, t = float(MARGIN), 0.0
            i = 0
            while x < W - MARGIN - BALL:
                phase = (y0 + vy * t - BALL) % (2 * span)
                y = BALL + (phase if phase <= span else 2 * span - phase)
                v = self.look(y, x)
                if i % every == 0:
                    F.append(v)
                    Y.append(y)
                i += 1
                t += 1.0
                x = MARGIN + vx * t
        F, Y = np.stack(F), np.array(Y)
        self.fit_and_score(F, Y, label=f"{flights} synthetic flights",
                           quiet=quiet)
        return {"F": F, "Y": Y}

    def fit_and_score(self, F, Y, groups=None, label="", quiet=False,
                      cap=450):
        """Fit the decode and report held-out accuracy.

        `cap` subsamples the training set. Every estimate costs a kernel row
        against all of it, so 2,490 rally frames dropped the decode from 34
        to 6 per second -- and a few hundred well-spread frames carry the
        same information as thousands of near-duplicates.
        """
        if cap and len(Y) > cap:
            keep = np.linspace(0, len(Y) - 1, cap).astype(int)
            F, Y = F[keep], Y[keep]
            groups = None if groups is None else groups[keep]
        self._fit(F, Y)
        hit, med = self._score(F, Y, groups=groups)
        if not quiet:
            where = f" from {label}" if label else ""
            print(f"  decode fitted on {len(Y)} frames{where}: {hit:.1f}% "
                  f"of held-out balls would be caught, median error "
                  f"{med:.1f} px")
        return hit, med


    # -- saving the decode ------------------------------------------------

    def fingerprint(self, conditions=None):
        """What the decode was fitted against.

        The circuit half is obvious: change the front end, the readout, the
        integration steps or the stimulus size and the readout vectors mean
        something else.

        The conditions half is the one that bites. A decode is fitted on the
        distribution the game produced while collecting it, and paddle speed,
        ball speed and glance interval all change that distribution --
        rebound angles most of all. Reusing a decode across a change in those
        is the same regime mismatch that has cost this project four separate
        results, so they belong in the key rather than in a comment.
        """
        key = {
            "readout": self.readout,
            "cells": len(self.read),
            "neurons": self.circuit.n,
            "synapses": self.circuit.n_synapses,
            "steps": self.steps,
            "radius": self.radius,
        }
        if conditions:
            key.update({f"cond_{k}": v for k, v in conditions.items()})
        return key

    def save(self, path, F, Y, groups=None, conditions=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = self.fingerprint(conditions)
        np.savez_compressed(
            path, F=F, Y=Y,
            groups=np.array([] if groups is None else groups),
            readout=np.array(meta["readout"]),
            cells=np.array(meta["cells"]),
            neurons=np.array(meta["neurons"]),
            synapses=np.array(meta["synapses"]),
            steps=np.array(meta["steps"]),
            radius=np.array(meta["radius"]),
            cond_keys=np.array(sorted(k for k in meta if
                                      k.startswith("cond_"))),
            cond_vals=np.array([float(meta[k]) for k in
                                sorted(k for k in meta
                                       if k.startswith("cond_"))]),
        )
        return path

    def load(self, path, conditions=None, quiet=False):
        """Restore a saved decode, or return False if it does not match."""
        path = Path(path)
        if not path.exists():
            return False
        with np.load(path, allow_pickle=False) as z:
            want = self.fingerprint(conditions)
            got = {
                "readout": str(z["readout"]),
                "cells": int(z["cells"]),
                "neurons": int(z["neurons"]),
                "synapses": int(z["synapses"]),
                "steps": int(z["steps"]),
                "radius": float(z["radius"]),
            }
            if "cond_keys" in z:
                got.update(dict(zip([str(k) for k in z["cond_keys"]],
                                    [float(v) for v in z["cond_vals"]])))
            if got != want:
                differs = [k for k in want if got[k] != want[k]]
                print(f"  cached decode does not match this circuit "
                      f"({', '.join(differs)}); recalibrating")
                return False
            F, Y = z["F"], z["Y"]
            groups = z["groups"]
            groups = groups if groups.size else None
        # Through the same capped path the decode was fitted with. Fitting
        # the full saved set instead gives a *different* decode from the one
        # that was measured -- 74.7% against the 82.7% reported when it was
        # built -- and a slower one, since every estimate costs a kernel row
        # against the whole training set.
        hit, med = self.fit_and_score(F, Y, groups=groups, quiet=True)
        if not quiet:
            print(f"  loaded decode from {path.name}: {len(Y)} frames saved, "
                  f"{hit:.1f}% of held-out balls would be caught, median "
                  f"error {med:.1f} px")
        return True


class Tracker:
    """Turns glances into a paddle target, refusing impossible jumps.

    The decode is accurate for most glances and occasionally wrong by a lot:
    10 px median against 119 px at the 90th percentile. Those outliers are
    isolated rather than drift, and they are *provably* wrong without knowing
    the answer -- a ball cannot move further between two glances than its own
    speed allows. So the target is only ever allowed to move as fast as the
    ball could have, which passes real motion through untouched and clips a
    150 px jump down to something harmless.

    A short median over recent glances does the same job from the other side,
    rejecting a single bad reading before it reaches the target at all.
    """

    def __init__(self, eye, ball_speed, window=3, slack=1.6, margin=6.0):
        self.eye = eye
        self.ball_speed = ball_speed
        self.window = max(1, int(window))
        self.slack = slack
        self.margin = margin
        self.recent = []
        self.target = None
        self.last_raw = None
        # The glance itself, kept so a display can show what the cells did
        # rather than only what the decode concluded.
        self.last_vec = None

    def reset(self, target=None):
        self.recent.clear()
        self.target = target
        self.eye.reset()

    def update(self, ball_y, ball_x, gap_frames=1):
        """One glance. Returns the paddle target."""
        self.last_vec = self.eye.look(ball_y, ball_x)
        raw = self.eye.estimate(self.last_vec)
        self.last_raw = raw
        self.recent.append(raw)
        if len(self.recent) > self.window:
            self.recent.pop(0)
        proposal = float(np.median(self.recent))

        if self.target is None:
            self.target = proposal
            return self.target
        # The ball's vertical speed cannot exceed its total speed, so this is
        # the furthest it could have moved since the last glance.
        reach = self.ball_speed * max(gap_frames, 1) * self.slack + self.margin
        self.target += float(np.clip(proposal - self.target, -reach, reach))
        self.target = float(np.clip(self.target, PAD_H / 2, H - PAD_H / 2))
        return self.target
