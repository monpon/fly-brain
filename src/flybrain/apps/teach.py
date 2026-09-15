#!/usr/bin/env python
"""What *can* reward and punishment teach this circuit?

Pong says what it cannot do. Four curricula and three plasticity mechanisms
all plateau around 10 points above chance, while the same wiring reaches 97%
from a supervised signal -- so the limit is the teaching signal, not the
circuit. That answer is only half useful without the other half: where does
reward-driven learning actually work?

This is the task the dopamine rule was built for. One stimulus, one choice,
one outcome, immediately:

    show an odour  ->  fly picks an action  ->  right: dopamine
                                                wrong: shock

No trajectory, no delay, nothing to assign credit across. Everything pong
made hard is simply absent, which is what makes this the clean measurement of
what the rule can hold.

    flybrain teach                       # 6 odours -> 3 actions
    flybrain teach --stimuli 12 --actions 3
    flybrain teach --random              # arbitrary patterns, not real odours
    flybrain teach --capacity            # sweep until it breaks

With `--odors` (the default) the stimuli are real glomerular activation
patterns from `odors.py` -- vinegar, geosmin, CO2 and the rest, each a
published response map rather than a random vector. Teaching the fly that
geosmin means "go left" is the same experiment as teaching it that geosmin
means "run away", which is the assay this circuit is famous for.
"""

import argparse
import sys

import numpy as np

from flybrain import mushroom_body as MB, odors
from flybrain._deps import require
from flybrain.batched import BatchedBrain, pick_device
from flybrain.learning import LearningParams


def build_stimuli(mb, n: int, use_odors: bool, rng) -> tuple[np.ndarray, list]:
    """(n, n_pn) activation patterns, and what to call them."""
    n_pn = mb.n("PN")
    if use_odors:
        names = list(odors.ODORS)[:n]
        if len(names) >= n:
            X = np.stack([odors.pn_vector(mb, name, 1.0, 1.0)
                          for name in names]).astype(np.float32)
            return X, names
        print(f"only {len(names)} named odours available; "
              f"falling back to random patterns")
    X = np.zeros((n, n_pn), dtype=np.float32)
    active = max(1, int(0.15 * n_pn))
    for i in range(n):
        X[i, rng.choice(n_pn, active, replace=False)] = 1.0
    return X, [f"pattern {i}" for i in range(n)]


def run(a, n_stimuli: int, quiet: bool = False) -> dict:
    torch = require("torch")
    device = pick_device(a.device)
    mb = MB.load_cache()
    rng = np.random.default_rng(a.seed)

    X, names = build_stimuli(mb, n_stimuli, not a.random, rng)
    n_pn = X.shape[1]
    # Scale to the drive the circuit expects, the same way the pong encoding
    # does; the raw odour vectors are unit-ish and would barely move the KCs.
    X = X / max(X.max(), 1e-9) * a.pn_gain
    correct = np.arange(n_stimuli) % a.actions

    brain = BatchedBrain(
        mb, a.flies, n_actions=a.actions, device=device, seed=a.seed,
        params=LearningParams(learning_rate=a.learning_rate,
                              trace_tau=1.0, trace_normalize=True),
    )
    Xt = torch.tensor(X, device=device)
    correct_t = torch.tensor(correct, device=device, dtype=torch.long)
    gen = torch.Generator(device=device)
    gen.manual_seed(a.seed + 5)

    N = a.flies
    curve = []
    block_right = torch.zeros(N, device=device)
    for trial in range(1, a.trials + 1):
        which = torch.randint(n_stimuli, (N,), generator=gen, device=device)
        pn = Xt[which]
        # One decision per outcome: nothing to carry credit across, so the
        # eligibility trace never has to reach past the trial that earned it.
        brain.reset_trace()
        _, _, drives = brain.observe(pn)
        frac = 1.0 / (1.0 + trial / max(a.explore_halflife, 1e-9))
        explore = a.explore_final + (a.explore - a.explore_final) * frac
        actions, _ = brain.choose(drives, explore)

        want = correct_t[which]
        right = (actions == want)
        outcomes = torch.where(right, torch.ones(N, device=device),
                               -torch.ones(N, device=device))
        brain.reinforce(actions, outcomes,
                        torch.ones(N, device=device, dtype=torch.bool))
        block_right += right.float()

        if trial % a.every == 0:
            acc = (block_right / a.every).cpu().numpy()
            curve.append((trial, acc.mean(), acc.std()))
            if not quiet:
                print(f"{trial:7d} {100*acc.mean():9.1f}% "
                      f"{'sd ' + format(100*acc.std(), '.1f'):>12s} "
                      f"{explore:9.3f} "
                      f"{100*brain.depressed_fraction().mean().item():9.1f}%")
            block_right.zero_()

    # Final test: greedy, every stimulus, no exploration and no learning.
    per_fly = torch.zeros(N, device=device)
    for i in range(n_stimuli):
        pn = Xt[i].unsqueeze(0).expand(N, -1)
        brain.reset_trace()
        _, _, drives = brain.observe(pn)
        per_fly += (drives.argmax(dim=1) == correct_t[i]).float()
    final = (per_fly / n_stimuli).cpu().numpy()
    return {"names": names, "final": final, "curve": curve,
            "chance": 1.0 / a.actions, "device": device}


def main(a) -> int:
    if a.capacity:
        print(f"{a.flies} flies, {a.actions} actions, {a.trials} trials each")
        print(f"how many stimulus->action associations survive?\n")
        print(f"{'stimuli':>8s} {'accuracy':>10s} {'sd':>7s} {'chance':>8s} "
              f"{'verdict':>10s}")
        for n in (2, 3, 4, 6, 8, 12, 16, 24, 32):
            r = run(a, n, quiet=True)
            acc, chance = r["final"].mean(), r["chance"]
            verdict = "learned" if acc > chance + 0.25 else (
                "partial" if acc > chance + 0.10 else "no")
            print(f"{n:8d} {100*acc:9.1f}% {100*r['final'].std():6.1f} "
                  f"{100*chance:7.0f}% {verdict:>10s}")
        return 0

    print(f"{a.flies} flies on {pick_device(a.device)}, "
          f"{a.stimuli} stimuli -> {a.actions} actions, "
          f"chance {100/a.actions:.0f}%")
    print(f"{'trial':>7s} {'accuracy':>10s} {'across flies':>12s} "
          f"{'explore':>9s} {'depressed':>10s}")
    r = run(a, a.stimuli)

    acc = r["final"]
    print(f"\nfinal (greedy, every stimulus): {100*acc.mean():.1f}% "
          f"(sd {100*acc.std():.1f}, chance {100*r['chance']:.0f}%)")
    print(f"  flies at 100%: {int((acc >= 0.999).sum())} of {len(acc)}")
    print(f"  worst fly {100*acc.min():.1f}%, best {100*acc.max():.1f}%")
    print(f"\nstimuli: {', '.join(r['names'])}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimuli", type=int, default=6)
    ap.add_argument("--actions", type=int, default=3)
    ap.add_argument("--trials", type=int, default=600)
    ap.add_argument("--flies", type=int, default=64)
    ap.add_argument("--every", type=int, default=100)
    ap.add_argument("--random", action="store_true",
                    help="arbitrary sparse patterns instead of real odours")
    ap.add_argument("--capacity", action="store_true",
                    help="sweep the number of stimuli until it breaks")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--learning-rate", type=float, default=0.35)
    ap.add_argument("--explore", type=float, default=1.0)
    ap.add_argument("--explore-final", type=float, default=0.02)
    ap.add_argument("--explore-halflife", type=float, default=100.0)
    ap.add_argument("--pn-gain", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    sys.exit(main(ap.parse_args()) or 0)
