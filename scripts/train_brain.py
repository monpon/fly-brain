#!/usr/bin/env python
"""Train a connectome circuit to map inputs onto outputs. No body involved.

The wiring is the fly's and stays the fly's: no synapse is created, destroyed
or sign-flipped. Training sets one non-negative gain per existing synapse.

    .venv/bin/python scripts/train_brain.py                    # 6 odours -> 6 MBONs
    .venv/bin/python scripts/train_brain.py --task random --classes 10
    .venv/bin/python scripts/train_brain.py --data mine.npz    # your own X, Y
    .venv/bin/python scripts/train_brain.py --types 'KC.*' 'MBON.*' \
        --inputs 'KC.*' --outputs 'MBON.*'

`mine.npz` must hold `X` (trials, n_inputs) and `Y` (trials, n_outputs).
Run with --list-io first to see how many of each the circuit has.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from flybrain import mushroom_body as MB, odors, trainable

OUT = Path("output/trained_brain.npz")


def pick_device(requested: str) -> str:
    """Resolve 'auto' to cuda when it is actually usable, else cpu.

    The forward pass is a gather, a multiply and a scatter-add over the
    synapse list, which is memory-bandwidth bound rather than compute bound --
    exactly the shape of problem a GPU wins on. But only for large circuits:
    the mushroom body is small enough that kernel launch overhead eats the
    gain, so 'auto' is not automatically the fast choice.

    Memory is the limit, not speed: autograd keeps every timestep, so VRAM
    goes as steps x batch x synapses. Halve --batch before giving up.
    """
    import torch

    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info()
        print(f"device: cuda -- {name}, {free / 2**30:.1f} of "
              f"{total / 2**30:.1f} GiB free")
        return "cuda"
    if requested == "cuda":
        raise SystemExit(
            "--device cuda, but torch.cuda.is_available() is False.\n"
            f"  torch {torch.__version__}, built against CUDA "
            f"{torch.version.cuda}\n"
            "  A '+cpu' build has no CUDA at all -- reinstall from "
            "https://download.pytorch.org/whl/cu124\n"
            "  If the build is right, check `nvidia-smi` finds the driver."
        )
    print("device: cpu (no CUDA available)")
    return "cpu"


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
        circuit = trainable.from_mushroom_body(MB.load_cache())

    print(f"circuit: {circuit.summary()}")
    print("  " + ", ".join(f"{t} x{c}" for t, c in circuit.type_counts(6)))
    print(f"  dataset {circuit.dataset}, topology {circuit.fingerprint()}")
    if a.list_io:
        return

    X, Y, labels = make_task(a, circuit)
    print(f"\ntask: {len(X)} patterns -> {Y.shape[1]} output neurons "
          f"({'your data' if a.data else a.task})")

    device = pick_device(a.device)
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
