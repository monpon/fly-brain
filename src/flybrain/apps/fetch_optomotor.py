#!/usr/bin/env python
"""Extract the optomotor circuit from the local male-CNS tables.

Wide-field visual motion -> direction-selective cells -> wide-field integrators
-> a descending steering command. No neuPrint token required; reads the Feather
files, downloading them on first use.

    flybrain fetch-optomotor
"""

import sys
import time
from pathlib import Path


import numpy as np

from flybrain import config, local_data as L

# Regexes, because type naming shifts between connectome releases.
CIRCUIT = {
    "Mi1": r"Mi1",          # excitatory input to T4
    "Mi4": r"Mi4",          # GABAergic input to T4
    "Mi9": r"Mi9",          # glutamatergic input to T4
    "T4": r"T4[a-d]",       # ON-edge direction-selective, one type per direction
    "T5": r"T5[a-d]",       # OFF-edge direction-selective
    "HS": r"HS[NES]",       # horizontal-system wide-field integrators
    "VS": r"VS",            # vertical-system integrator
    "DNa02": r"DNa02",      # descending steering command
}


def main() -> None:
    print(f"dataset: {config.dataset_dir().name}\n")

    print("cell types found")
    for label, pattern in CIRCUIT.items():
        hits = L.find_types(f"^{pattern}$")
        total = int(hits["n_neurons"].sum())
        names = ", ".join(hits["type"].astype(str)) or "(none)"
        print(f"  {label:6s} {total:6d} neurons   {names}")

    print("\nbuilding weight matrix...")
    t0 = time.time()
    W, neurons = L.circuit_weight_matrix(list(CIRCUIT.values()))
    print(f"  {W.shape[0]} neurons in {time.time() - t0:.1f}s")

    nonzero = int((W != 0).sum())
    print(f"  {nonzero} connections, {100 * nonzero / W.size:.4f}% dense")
    print(f"  excitatory {int((W > 0).sum())}, inhibitory {int((W < 0).sum())}")
    print(f"  {np.abs(W).sum():.0f} synapses total")

    print("\npathway synapse counts (signed)")
    idx = {
        label: neurons.index[
            neurons["type"].astype(str).str.fullmatch(pattern, na=False)
        ].to_numpy()
        for label, pattern in CIRCUIT.items()
    }
    for src, dst in [
        ("Mi1", "T4"), ("Mi4", "T4"), ("Mi9", "T4"),
        ("T4", "HS"), ("T5", "HS"), ("T4", "VS"), ("T5", "VS"),
        ("HS", "DNa02"), ("VS", "DNa02"),
    ]:
        total = W[np.ix_(idx[src], idx[dst])].sum()
        print(f"  {src:6s} -> {dst:6s} {total:+12.0f}")

    out = config.cache_dir() / "optomotor.npz"
    np.savez_compressed(
        out, W=W, body_ids=neurons["bodyId"].to_numpy(),
        types=neurons["type"].astype(str).to_numpy(),
    )
    print(f"\nsaved {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
