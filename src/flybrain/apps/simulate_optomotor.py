#!/usr/bin/env python
"""Stimulate direction-selective cells and watch the circuit respond.

Drives one T4 subtype (one preferred direction of ON-edge motion) with Poisson
input and measures the firing rate of every downstream cell type, including the
DNa02 steering command.

    flybrain simulate-optomotor            # T4a
    flybrain simulate-optomotor T4b 150    # type, Hz
    flybrain simulate-optomotor --raw       # no normalization
"""

import sys
import time
from pathlib import Path


import numpy as np
from brian2 import ms

from flybrain import config
from flybrain.network import LIFParams, add_stimulus, build, rates_by_type, use_fast_codegen

DURATION_MS = 1000.0
NORMALIZE_MV = 12.0
REPORT_TYPES = ["Mi1", "Mi4", "Mi9", "T4a", "T4b", "T4c", "T4d",
                "T5a", "T5b", "T5c", "T5d", "HSN", "HSE", "HSS", "VS", "DNa02"]


def main(stim_type: str, rate: float) -> None:
    path = config.cache_dir() / "optomotor.npz"
    if not path.exists():
        raise SystemExit("Run flybrain fetch-optomotor first.")

    data = np.load(path, allow_pickle=True)
    W, types = data["W"], data["types"].astype(str)
    print(f"loaded {W.shape[0]} neurons, {int((W != 0).sum())} connections")

    targets = np.flatnonzero(types == stim_type)
    if targets.size == 0:
        raise SystemExit(f"No neurons of type {stim_type}")

    use_fast_codegen()
    # 12 mV sits in the narrow band where the circuit is neither silent nor
    # saturated. See docs/FINDINGS.md -- this value is a free parameter the
    # connectome does not supply, not a measurement.
    normalize = None if "--raw" in sys.argv else NORMALIZE_MV
    params = LIFParams(normalize_input_mv=normalize)
    print("input normalization:",
          f"{normalize:.0f} mV total per cell" if normalize else "off (raw synapse counts)")
    net, neurons, spikes = build(W, params)
    add_stimulus(net, neurons, targets, rate)

    print(f"stimulating {targets.size} {stim_type} neurons at {rate:.0f} Hz "
          f"for {DURATION_MS:.0f} ms\n")

    t0 = time.time()
    net.run(DURATION_MS * ms)
    wall = time.time() - t0
    duration_s = DURATION_MS / 1000.0

    print(f"simulated {duration_s:.1f}s of biological time in {wall:.1f}s wall "
          f"({wall / duration_s:.0f}x slower than real time)")
    print(f"{len(spikes.i)} spikes total\n")

    rates = rates_by_type(spikes, types, duration_s)
    print(f"{'type':8s} {'n':>6s} {'rate (Hz)':>10s}")
    for t in REPORT_TYPES:
        if t in rates:
            n = int((types == t).sum())
            marker = "  <- stimulated" if t == stim_type else ""
            print(f"{t:8s} {n:6d} {rates[t]:10.2f}{marker}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    stim = args[0] if args else "T4a"
    hz = float(args[1]) if len(args) > 1 else 100.0
    main(stim, hz)
