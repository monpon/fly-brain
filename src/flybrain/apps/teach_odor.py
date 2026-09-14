#!/usr/bin/env python
"""Teach the fly that one real odour is good and another is bad.

Differential conditioning with reinforcement on both sides: vinegar arrives
with sugar (PAM dopamine), geosmin with shock (PPL1). Afterwards the fly's
valence for the rewarded odour is higher and the punished one has crossed into
negative -- which is what makes it walk toward one source and away from the
other in flybrain forage.

    flybrain teach-odor
    flybrain teach-odor --good banana --bad almond
    flybrain teach-odor --epochs 80

Writes output/odor_trained.flyckpt.
"""

import argparse
import sys
from pathlib import Path


from flybrain import checkpoint, mushroom_body as MB, odors
from flybrain.learning import FlyBrain, LearningParams
from flybrain.tasks import ValencePair

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "output" / "odor_trained.flyckpt"


def main(a) -> None:
    mb = MB.load_cache()
    fly = FlyBrain(mb, LearningParams(learning_rate=a.learning_rate,
                                      kc_sparsity=a.sparsity))
    task = ValencePair.build(mb, good=a.good, bad=a.bad,
                             reward=a.reward, shock=a.shock)

    print(mb.summary())
    print(f"{len(fly.state.gain)} plastic synapses, "
          f"fingerprint {mb.fingerprint()}\n")
    print(f"good  {a.good:10s} glomeruli {', '.join(odors.glomeruli_of(a.good))}")
    print(f"bad   {a.bad:10s} glomeruli {', '.join(odors.glomeruli_of(a.bad))}")
    print(f"Kenyon cell code overlap: {100 * task.overlap(fly):.1f}%\n")

    before = task.test(fly)
    print(f"before   {a.good} {before['v_good']:+.5f}   "
          f"{a.bad} {before['v_bad']:+.5f}")

    history = task.train(fly, epochs=a.epochs)
    for row in history:
        if row["epoch"] % max(1, a.epochs // 6) == 0 or row["epoch"] == 1:
            print(f"  epoch {row['epoch']:3d}   {a.good} {row['v_good']:+.5f}   "
                  f"{a.bad} {row['v_bad']:+.5f}   "
                  f"mean gain {row['mean_gain']:.4f}")

    after = task.test(fly)
    print(f"after    {a.good} {after['v_good']:+.5f}   "
          f"{a.bad} {after['v_bad']:+.5f}")
    print(f"\nseparation {before['separation']:+.5f} -> {after['separation']:+.5f}")

    # The sign is what the navigator acts on: it steers toward a positive
    # valence and away from a negative one. Attraction that never crosses zero
    # produces a fly that approaches both sources, just at different speeds.
    if after["v_bad"] >= 0:
        print("WARNING: the punished odour is still positive -- the fly will "
              "approach it. Train for more epochs or raise --shock.")
    else:
        print(f"{a.bad} crossed into avoidance; {a.good} strengthened "
              f"{after['v_good'] / max(before['v_good'], 1e-9):.2f}x")

    print(f"\n{100 * fly.depressed_fraction():.1f}% of plastic synapses "
          f"depressed below 0.9")
    ranked = sorted(fly.compartment_gains().items(), key=lambda kv: kv[1])[:6]
    print("most-depressed compartments")
    for name, gain in ranked:
        print(f"  {name:14s} {gain:.4f}")

    path = checkpoint.save(
        a.out, fly,
        task=f"valence pair: {a.good} rewarded, {a.bad} punished",
        notes=f"{a.epochs} epochs, separation {after['separation']:+.5f}",
    )
    print(f"\nsaved -> {path}  ({path.stat().st_size / 1e3:.0f} kB)")
    print("Now run: flybrain forage "
          f"--good {a.good} --bad {a.bad}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--good", default="vinegar", choices=sorted(odors.ODORS))
    ap.add_argument("--bad", default="geosmin", choices=sorted(odors.ODORS))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--reward", type=float, default=1.0)
    ap.add_argument("--shock", type=float, default=1.0)
    ap.add_argument("--learning-rate", type=float, default=0.35)
    ap.add_argument("--sparsity", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    main(ap.parse_args())
