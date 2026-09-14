#!/usr/bin/env python
"""Train a connectome circuit to map inputs onto outputs. No body involved.

The wiring is the fly's and stays the fly's: no synapse is created, destroyed
or sign-flipped. Training sets one non-negative gain per existing synapse.

    flybrain train-brain                    # 6 odours -> 6 MBONs
    flybrain train-brain --task random --classes 10
    flybrain train-brain --data mine.npz    # your own X, Y
    flybrain train-brain --types 'KC.*' 'MBON.*' \
        --inputs 'KC.*' --outputs 'MBON.*'

`mine.npz` must hold `X` (trials, n_inputs) and `Y` (trials, n_outputs).
Run with --list-io first to see how many of each the circuit has.
"""

import argparse
import sys
from pathlib import Path


import numpy as np

from flybrain import circuits, mushroom_body as MB, odors, trainable

OUT = Path("output/trained_brain.npz")


def make_task(a, circuit):
    """Returns (X, Y, labels). Targets are one-hot over the output neurons."""
    n_in, n_out = len(circuit.input_idx), len(circuit.output_idx)
    rng = np.random.default_rng(a.seed)

    if a.data:
        d = np.load(a.data)
        X, Y = d["X"].astype(np.float32), d["Y"].astype(np.float32)
        if X.shape[1] != n_in or Y.shape[1] != n_out:
            raise SystemExit(
                f"data shape mismatch: circuit wants X (*, {n_in}) and "
                f"Y (*, {n_out}), got {X.shape} and {Y.shape}"
            )
        return X, Y, [f"trial {i}" for i in range(len(X))]

    if a.task == "odors":
        # Real glomerular patterns, one per named odour. The task is to make
        # each odour drive its own output neuron and no other -- categorising
        # smells, which is what this pathway does in the animal.
        mb = MB.load_cache()
        names = list(odors.ODORS)[: a.classes]
        X = np.stack([odors.pn_vector(mb, n, 1.0, 1.0) for n in names])
        labels = names
    else:
        # Sparse random input patterns, the standard capacity probe: how many
        # arbitrary associations will this wiring hold?
        names = [f"pattern {i}" for i in range(a.classes)]
        X = np.zeros((a.classes, n_in), dtype=np.float32)
        active = max(1, int(0.15 * n_in))
        for i in range(a.classes):
            X[i, rng.choice(n_in, active, replace=False)] = 1.0
        labels = names

    Y = np.zeros((len(X), n_out), dtype=np.float32)
    Y[np.arange(len(X)), np.arange(len(X)) % n_out] = 1.0
    return X.astype(np.float32), Y, labels


def main(a):
    if a.types:
        circuit = trainable.build(a.types, a.inputs, a.outputs)
    else:
        # The bundled copy is byte-for-byte what from_mushroom_body() builds,
        # so the default path needs no tables and no download.
        circuit = circuits.mushroom_body()

    print(f"circuit: {circuit.summary()}")
    print("  " + ", ".join(f"{t} x{c}" for t, c in circuit.type_counts(6)))
    print(f"  dataset {circuit.dataset}, topology {circuit.fingerprint()}")
    if a.list_io:
        return

    X, Y, labels = make_task(a, circuit)
    print(f"\ntask: {len(X)} patterns -> {Y.shape[1]} output neurons "
          f"({'your data' if a.data else a.task})")

    device = trainable.pick_device(a.device)
    net = trainable.BrainNet(circuit, steps=a.steps, tau=a.tau, seed=a.seed,
                             device=device)
    before = net.accuracy(X, Y)
    print(f"before training: accuracy {100 * before:5.1f}%  "
          f"(chance {100 / len(X):.1f}%)\n")

    net.fit(X, Y, epochs=a.epochs, lr=a.lr, batch=a.batch, loss=a.loss,
            anatomy_pull=a.anatomy_pull, report_every=a.report_every)

    after = net.accuracy(X, Y)
    print(f"\nafter training:  accuracy {100 * after:5.1f}%")

    out = net.predict(X)
    print("\nwhat each input now produces (winning output neuron, margin)")
    for i, name in enumerate(labels[:12]):
        order = np.argsort(-out[i])
        win, runner = order[0], order[1]
        mark = "ok " if win == int(np.argmax(Y[i])) else "MISS"
        print(f"  {mark} {name:12s} -> neuron {win:4d} "
              f"({circuit.types[circuit.output_idx[win]]:12s}) "
              f"margin {out[i, win] - out[i, runner]:+.4f}")

    # Robustness: the circuit should still answer correctly when the input is
    # noisy, or it has memorised rather than generalised.
    rng = np.random.default_rng(a.seed + 1)
    for noise in (0.1, 0.3):
        Xn = X + rng.normal(0, noise * max(X.max(), 1e-9), X.shape)
        print(f"accuracy with {noise:.0%} input noise: "
              f"{100 * net.accuracy(np.clip(Xn, 0, None), Y):5.1f}%")

    g = net.gains()
    print(f"\nsynaptic gains: mean {g.mean():.3f}, "
          f"{100 * (g < 0.9).mean():.1f}% weakened, "
          f"{100 * (g > 1.1).mean():.1f}% strengthened")
    print(f"saved -> {net.save(a.out)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", nargs="*", default=None,
                    help="cell-type regexes selecting the circuit")
    ap.add_argument("--inputs", nargs="*", default=None)
    ap.add_argument("--outputs", nargs="*", default=None)
    ap.add_argument("--task", choices=("odors", "random"), default="odors")
    ap.add_argument("--classes", type=int, default=6)
    ap.add_argument("--data", default=None, help="npz with X and Y")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--loss", choices=("mse", "ce"), default="ce")
    ap.add_argument("--anatomy-pull", type=float, default=1e-3)
    ap.add_argument("--report-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                    help="auto uses the GPU when one is usable")
    ap.add_argument("--list-io", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    main(ap.parse_args())
