#!/usr/bin/env python
"""What the circuit can hold on to, and for how long.

Three separate senses of "memory", measured rather than asserted:

  capacity   how many pattern->value associations the mushroom body keeps
  persist    train on images, save, reload into a fresh net, recall them
  attractor  does activity outlive the stimulus that caused it

    .venv/bin/python scripts/remember.py                 # all three
    .venv/bin/python scripts/remember.py capacity
    .venv/bin/python scripts/remember.py attractor
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from flybrain import mushroom_body as MB, trainable as T, vision as V

STORE = Path("output/seen_memory.npz")


def capacity(a):
    """Synaptic memory: how many associations survive, and how noisy a cue."""
    c = T.from_mushroom_body(MB.load_cache())
    n_in, n_out = len(c.input_idx), len(c.output_idx)
    print(f"mushroom body: {c.n} neurons, {n_in} PN in, {n_out} MBON out")
    print(f"\n{'patterns':>9s} {'clean':>7s} {'10% noise':>10s} {'30% noise':>10s}")
    for K in a.sizes:
        rng = np.random.default_rng(a.seed)
        X = np.zeros((K, n_in), np.float32)
        for i in range(K):
            X[i, rng.choice(n_in, max(1, int(0.15 * n_in)), replace=False)] = 1.0
        Y = np.zeros((K, n_out), np.float32)
        Y[np.arange(K), np.arange(K) % n_out] = 1.0

        net = T.BrainNet(c, steps=a.steps, seed=a.seed,
                         device=T.pick_device(a.device))
        net.fit(X, Y, epochs=a.epochs, lr=a.lr, batch=32, loss="ce",
                report_every=0)
        row = [net.accuracy(X, Y)]
        for lvl in (0.1, 0.3):
            Xn = np.clip(X + lvl * rng.standard_normal(X.shape).astype(np.float32),
                         0, None)
            row.append(net.accuracy(Xn, Y))
        print(f"{K:9d} {100*row[0]:6.1f}% {100*row[1]:9.1f}% {100*row[2]:9.1f}%")
    print("\nClean recall does not break; recall from a degraded cue does. Past\n"
          "roughly 150 patterns the memories start interfering. Note that with\n"
          f"only {n_out} MBONs, beyond {n_out} patterns several must share an output.")


def _scenes(eyes):
    shape = eyes.frame_size()
    h, w = shape
    out = [V.disc(shape, min(shape) / 5)]
    im = V.flash(shape, 1.0); im[:h // 2] = 0.0; out.append(im)
    im = V.flash(shape, 1.0); im[:, :w // 2] = 0.0; out.append(im)
    out.append(V.flash(shape, 0.15))
    return np.stack(out)


def persist(a):
    """Does a visual memory survive being written out and read back?"""
    c = T.build(["L1", "L2", "L5", "Mi1", "Tm1", "Tm2", "Tm9"],
                ["L1", "L2", "L5"], ["Tm1", "Tm2", "Tm9"])
    eyes = V.Binocular(c)
    print(eyes.summary())
    odd = eyes.unbalanced_types()
    if odd:
        print(f"  warning: {odd} are addressed on one eye only")

    rng = np.random.default_rng(a.seed)
    imgs = _scenes(eyes)
    X0 = np.stack([eyes(im) for im in imgs])
    K = len(X0)
    Y = np.zeros((K, len(c.output_idx)), np.float32)
    Y[np.arange(K), np.arange(K)] = 1.0
    Xtr = np.concatenate([
        np.clip(X0 + 0.05 * rng.standard_normal(X0.shape).astype(np.float32), 0, None)
        for _ in range(12)])
    Ytr = np.tile(Y, (12, 1))

    dev = T.pick_device(a.device)
    net = T.BrainNet(c, steps=a.steps, seed=a.seed, device=dev)
    net.fit(Xtr, Ytr, epochs=a.epochs, lr=a.lr, batch=16, loss="ce",
            report_every=0)
    print(f"\nafter training      : {100*net.accuracy(X0, Y):5.1f}%")
    STORE.parent.mkdir(parents=True, exist_ok=True)
    net.save(STORE)

    fresh = T.BrainNet(c, steps=a.steps, seed=a.seed + 1, device=dev)
    print(f"fresh net, no memory: {100*fresh.accuracy(X0, Y):5.1f}%")
    fresh.load(STORE)
    print(f"memory restored     : {100*fresh.accuracy(X0, Y):5.1f}%")
    for lvl in (0.1, 0.3):
        Xn = np.clip(X0 + lvl * rng.standard_normal(X0.shape).astype(np.float32),
                     0, None)
        print(f"  recall at {int(lvl*100):2d}% noise: {100*fresh.accuracy(Xn, Y):5.1f}%")
    print(f"\nsaved to {STORE}. The gains are keyed to connectome body ids, so\n"
          "the file survives re-extracting the circuit.")


def attractor(a):
    """Activity memory: poke the heading ring, then take the input away."""
    import torch

    c = T.build(["EPG", "PEN.*", "PEG", "Delta7", "ER.*"], ["EPG"], ["EPG"])
    print(c.summary())
    n = c.n

    for rate in ("softplus", "saturating"):
        net = T.BrainNet(c, steps=1, tau=a.tau, seed=a.seed, device="cpu",
                         rate=rate)
        epg = torch.as_tensor(c.input_idx, dtype=torch.long)
        w0 = net.w_anat * torch.exp(net.log_gain)
        print(f"\n{rate}:")
        print(f"  {'gain':>5s} {'during':>8s} {'t=+10':>8s} {'t=+100':>8s}  verdict")
        for g in a.gains:
            w = w0 * g
            stim = torch.zeros(1, n)
            stim[0, epg[:8]] = 5.0
            x = torch.zeros(1, n)
            r = net.rate(x)
            k = net.dt / net.tau
            tr = []
            with torch.no_grad():
                for t in range(a.hold + 100):
                    inject = stim if t < a.hold else torch.zeros(1, n)
                    drive = torch.zeros(1, n).index_add(
                        1, net.post, r[:, net.pre] * w)
                    x = x + k * (-x + drive + net.bias + inject)
                    r = net.rate(x)
                    tr.append(float(r[0, epg].mean()))
            tr = np.array(tr)
            rest = float(net.rate(torch.zeros(1)).item())
            if not np.isfinite(tr[-1]) or tr[-1] > 1e4:
                verdict = "runaway"
            elif tr[-1] > rest * 1.05:
                verdict = "PERSISTS"
            else:
                verdict = "decays"
            print(f"  {g:5.1f} {tr[a.hold-1]:8.3f} {tr[a.hold+9]:8.3f} "
                  f"{tr[-1]:8.3f}  {verdict}")
    print("\nUnbounded rates give no stable high state: the ring decays or\n"
          "explodes with nothing between. Bound them and the same wiring holds\n"
          "its activity with no input -- but only one state, a single global\n"
          "pattern rather than a heading. A bump needs the gains trained, not\n"
          "just scaled.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", nargs="?", default="all",
                    choices=("all", "capacity", "persist", "attractor"))
    ap.add_argument("--sizes", type=int, nargs="*",
                    default=[6, 25, 50, 97, 150, 250])
    ap.add_argument("--gains", type=float, nargs="*",
                    default=[2.0, 3.0, 4.0, 6.0, 8.0])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--hold", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    for name in (["capacity", "persist", "attractor"]
                 if a.experiment == "all" else [a.experiment]):
        print(f"\n{'=' * 68}\n{name}\n{'=' * 68}")
        globals()[name](a)
