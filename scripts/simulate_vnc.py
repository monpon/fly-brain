#!/usr/bin/env python
"""Drive the connectome VNC from a descending command and look for a gait.

No CPG. Stimulates a descending neuron type, runs the 15,183-neuron
descending -> VNC -> motor network as LIF, converts motor neuron firing into
per-joint commands, and tests whether the output is rhythmic.

    .venv/bin/python scripts/simulate_vnc.py                # DNp09, forward
    .venv/bin/python scripts/simulate_vnc.py MDN 150        # backward walking
    .venv/bin/python scripts/simulate_vnc.py DNp09 100 --norm 12
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from brian2 import ms

from flybrain import vnc
from flybrain.network import (LIFParams, add_stimulus, build, use_fast_codegen)

BIN_MS = 5.0


def dominant_frequency(signal: np.ndarray, bin_s: float) -> tuple[float, float]:
    """Peak frequency of a joint command trace, and its relative power.

    Fly leg stepping is roughly 5-20 Hz. A flat spectrum means no rhythm.
    """
    x = signal - signal.mean()
    if np.allclose(x, 0):
        return 0.0, 0.0
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), bin_s)
    band = (freqs > 1.0) & (freqs < 50.0)
    if not band.any() or spec[band].sum() == 0:
        return 0.0, 0.0
    peak = freqs[band][np.argmax(spec[band])]
    share = spec[band].max() / spec[band].sum()
    return float(peak), float(share)


def main(stim: str, rate: float, norm: float | None, duration_ms: float) -> None:
    print("building VNC network from connectome...")
    t0 = time.time()
    net_data = vnc.extract()
    W, neurons = net_data["W"], net_data["neurons"]
    mm = vnc.motor_map(neurons)
    print(f"  {W.shape[0]} neurons, {int((W != 0).sum())} connections "
          f"({time.time() - t0:.1f}s)")
    print(f"  {len(mm.index)} motor neurons -> {len(mm.joints())} joints")

    targets = np.flatnonzero(neurons["type"].astype(str) == stim)
    if targets.size == 0:
        raise SystemExit(f"No descending neurons of type {stim}")

    use_fast_codegen()
    net, group, spikes = build(W, LIFParams(normalize_input_mv=norm))
    add_stimulus(net, group, targets, rate)
    print(f"\ndriving {targets.size} {stim} neurons at {rate:.0f} Hz, "
          f"normalization {norm if norm else 'off'}")

    t0 = time.time()
    net.run(duration_ms * ms)
    print(f"ran {duration_ms:.0f} ms in {time.time() - t0:.1f}s, "
          f"{len(spikes.i)} spikes")

    # Bin motor neuron spikes and build per-joint command traces.
    n_bins = int(duration_ms / BIN_MS)
    times = np.asarray(spikes.t / ms)
    ids = np.asarray(spikes.i)
    bins = np.clip((times / BIN_MS).astype(int), 0, n_bins - 1)

    counts = np.zeros((W.shape[0], n_bins), dtype=np.float32)
    np.add.at(counts, (ids, bins), 1.0)
    rates = counts / (BIN_MS / 1000.0)

    print(f"\n{'joint':18s} {'mean':>9s} {'peak Hz':>9s} {'power share':>12s}")
    rhythmic = 0
    for key in mm.joints()[:14]:
        mask = mm.joint == key
        trace = (rates[mm.index[mask]] * mm.sign[mask, None]).sum(axis=0)
        freq, share = dominant_frequency(trace, BIN_MS / 1000.0)
        flag = ""
        if share > 0.25 and 3.0 < freq < 30.0:
            flag, rhythmic = "  <- rhythmic", rhythmic + 1
        print(f"{key:18s} {trace.mean():9.1f} {freq:9.1f} {share:12.3f}{flag}")

    motor_rate = rates[mm.index].mean()
    print(f"\nmean motor neuron rate {motor_rate:.1f} Hz")
    print(f"joints with a clear rhythm: {rhythmic} of {len(mm.joints())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stim", nargs="?", default="DNp09")
    ap.add_argument("rate", nargs="?", type=float, default=100.0)
    ap.add_argument("--norm", type=float, default=12.0)
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--ms", type=float, default=1000.0)
    a = ap.parse_args()
    main(a.stim, a.rate, None if a.raw else a.norm, a.ms)
