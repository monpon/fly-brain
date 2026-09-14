#!/usr/bin/env python
"""Teach a fly to avoid an odour, then save what it learned.

Differential conditioning: one odour is paired with shock (driving PPL1
dopaminergic neurons), a second is presented alone. Dopamine depresses the
KC->MBON synapses carrying the punished odour, and the fly's valence for it
drops.

    flybrain train-fly
    flybrain train-fly --epochs 40 --seed 3
    flybrain train-fly --reward     # appetitive instead

Writes a portable checkpoint to output/trained_fly.flyckpt.
"""

import argparse
import sys
from pathlib import Path


from flybrain import checkpoint, mushroom_body as MB
from flybrain.learning import FlyBrain, LearningParams
from flybrain import tasks
from flybrain.tasks import OdorConditioning

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "output" / "trained_fly.flyckpt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--learning-rate", type=float, default=0.35)
    p.add_argument("--sparsity", type=float, default=0.05)
    p.add_argument("--shock", type=float, default=1.0)
    p.add_argument(
        "--reward", action="store_true",
        help="appetitive conditioning (PAM) instead of aversive (PPL1)",
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    mb = MB.load_cache()
    params = LearningParams(
        learning_rate=args.learning_rate, kc_sparsity=args.sparsity
    )
    fly = FlyBrain(mb, params)
    print(f"{mb.summary()}")
    print(f"{len(fly.state.gain)} plastic synapses, fingerprint {mb.fingerprint()}\n")

    task = OdorConditioning.build(
        mb,
        seed=args.seed,
        shock=0.0 if args.reward else args.shock,
        reward=args.shock if args.reward else 0.0,
    )
    kind = "appetitive (PAM/reward)" if args.reward else "aversive (PPL1/shock)"
    print(f"task: differential conditioning, {kind}")
    print(f"  KC code overlap between the two odours: "
          f"{100 * task.odor_overlap(fly):.1f}%")

    before = task.test(fly)
    print(f"\nbefore training   CS+ {before['v_plus']:+.5f}   "
          f"CS- {before['v_minus']:+.5f}   pref {before['preference']:+.5f}")

    print(f"\ntraining, {args.epochs} epochs")
    history = task.train(fly, epochs=args.epochs)
    for row in history:
        if row["epoch"] % max(1, args.epochs // 8) == 0 or row["epoch"] == 1:
            print(f"  epoch {row['epoch']:3d}   CS+ {row['v_plus']:+.5f}   "
                  f"CS- {row['v_minus']:+.5f}   mean gain {row['mean_gain']:.4f}")

    after = task.test(fly)
    print(f"\nafter training    CS+ {after['v_plus']:+.5f}   "
          f"CS- {after['v_minus']:+.5f}   pref {after['preference']:+.5f}")

    # The learning index is a shift between two timepoints, not a property of
    # either one on its own -- the two odours start with different valences.
    li = tasks.learning_index(before, after)
    print(f"\nlearning index {li:+.3f}  "
          f"({'aversive: moved away from CS+' if li < 0 else 'appetitive: moved toward CS+'})")
    print(f"{100 * fly.depressed_fraction():.1f}% of plastic synapses depressed "
          f"below 0.9, mean gain {fly.state.gain.mean():.4f}")

    compartments = fly.compartment_gains()
    ranked = sorted(compartments.items(), key=lambda kv: kv[1])[:8]
    print("\nmost-depressed compartments (mean KC->MBON gain)")
    for name, gain in ranked:
        print(f"  {name:14s} {gain:.4f}")

    path = checkpoint.save(
        args.out, fly,
        task=f"differential conditioning ({kind}), seed {args.seed}",
        notes=f"{args.epochs} epochs, LI {li:+.4f}",
    )
    print(f"\nsaved -> {path}  ({path.stat().st_size / 1e3:.0f} kB)")
    print("Load it onto another copy of the brain with flybrain transfer-fly")


if __name__ == "__main__":
    main()
