#!/usr/bin/env python
"""Train the gait: optimise CPG parameters against a walking fitness.

Black-box optimisation over the seven numbers that define the gait. Each
candidate is simulated for two seconds and scored on forward distance, minus
penalties for heading drift, body roll, sideways drift, and bouncing.

    flybrain train-gait                 # ~10 min on 8 cores
    flybrain train-gait --iters 40 --pop 12
    flybrain train-gait --show          # replay the best

Writes output/gait.json, which walk_cpg.py loads automatically.
"""

import argparse
import json
import sys
import time
from pathlib import Path


import numpy as np

from flybrain.cpg import GaitParams
from flybrain.fitness import evaluate

OUT = Path("output/gait.json")   # a retune always writes locally, not into the package

# name, low, high
SPACE = [
    ("frequency",   5.0,  18.0),   # real fly stepping is 5-20 Hz
    ("protraction", 0.10,  1.60),   # fore-aft sweep, on coxa_roll
    ("levation",    0.15,  0.45),   # lift, on coxa_yaw
    ("extension",   0.00,  0.60),
    ("duty",        0.62,  0.88),   # >=0.62 keeps a tripod planted
    ("coupling",    2.0,  20.0),
    ("abduction",  -0.30,  0.30),   # femur_pitch posture offset
]


def to_params(vector) -> GaitParams:
    kwargs = {name: float(v) for (name, _, _), v in zip(SPACE, vector)}
    return GaitParams(**kwargs)


def cost(vector) -> float:
    """Negative fitness, because scipy minimises."""
    try:
        return -float(evaluate(to_params(vector), ms=2000))
    except Exception:
        return 1e3          # a gait that crashes the simulator is a bad gait


def main(a):
    from scipy.optimize import differential_evolution

    if a.show:
        best = json.loads(OUT.read_text())
        m = evaluate(GaitParams(**best["params"]), ms=3000, detail=True)
        print(json.dumps(m, indent=2))
        return

    baseline = evaluate(GaitParams(), ms=2000, detail=True)
    print(f"baseline fitness {baseline['fitness']:.3f}  "
          f"speed {baseline['speed']:.2f} mm/s\n")

    history = []

    def report(xk, convergence=None):
        f = -cost(xk)
        history.append(f)
        print(f"  gen {len(history):3d}   best fitness {f:8.3f}")

    t0 = time.time()
    result = differential_evolution(
        cost,
        bounds=[(lo, hi) for _, lo, hi in SPACE],
        maxiter=a.iters, popsize=a.pop, tol=0.01,
        seed=a.seed, polish=False, workers=-1, updating="deferred",
        callback=report,
    )
    wall = time.time() - t0

    params = to_params(result.x)
    final = evaluate(params, ms=3000, detail=True)

    print(f"\noptimised in {wall/60:.1f} min, {result.nfev} simulations")
    print(f"{'metric':12s} {'baseline':>10s} {'trained':>10s}")
    for key in ("speed", "forward", "lateral", "yaw", "roll", "pitch", "fitness"):
        print(f"{key:12s} {baseline[key]:10.3f} {final[key]:10.3f}")

    print("\nparameters")
    for name, _, _ in SPACE:
        print(f"  {name:12s} {getattr(params, name):8.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "params": {n: float(getattr(params, n)) for n, _, _ in SPACE},
        "metrics": final,
        "baseline": baseline,
        "simulations": int(result.nfev),
    }, indent=2))
    print(f"\nsaved {OUT}  --  walk_cpg.py will use it automatically")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", action="store_true")
    main(ap.parse_args())
