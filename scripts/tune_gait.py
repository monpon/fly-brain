#!/usr/bin/env python
"""Tune the gait for smoothness, gearing included, with CMA-ES.

Two things changed from `train_gait.py`:

* **The per-leg gearing is in the search.** The front and hind stride and lift
  multipliers were module constants, so the previous optimiser could not see
  the strongest lever on body roll there is. A hind-leg lift of 2.35x levers
  the abdomen up and down once per step, and no amount of tuning the other
  seven numbers undoes that.
* **CMA-ES instead of differential evolution.** DE spent 2,200 simulations on
  seven parameters; CMA-ES adapts a covariance over the search space and
  converges on eleven in a few hundred. The bottleneck here was never
  throughput, it was sample efficiency.

    .venv/bin/python -u scripts/tune_gait.py
    .venv/bin/python -u scripts/tune_gait.py --gens 40 --workers 12

Writes output/gait.json.
"""

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from flybrain.cpg import GaitParams
from flybrain.fitness import evaluate

OUT = Path("output/gait.json")

# name, low, high, start
SPACE = [
    # Re-centred on the step *shape* the fly was getting wrong: it was
    # advancing the foot 0.84 mm per step while lifting it 0.37 mm, a ratio of
    # 2.2:1 where a real fly is nearer 10:1. Lift is now bounded an order of
    # magnitude lower and the sweep is free to grow.
    ("frequency",    6.0, 14.0, 11.0),   # real fly stepping, 5-20 Hz
    ("protraction",  0.25, 1.40, 0.70),  # fore-aft sweep, on coxa_roll
    ("levation",     0.01, 0.15, 0.05),  # lift, on coxa_yaw
    ("extension",    0.00, 0.40, 0.25),
    ("duty",         0.55, 0.80, 0.65),  # tripod needs a planted majority
    ("coupling",     4.0, 20.0, 10.0),
    ("abduction",   -0.20, 0.20, 0.0),
    ("stride_front", 0.80, 3.00, 1.60),
    ("stride_hind",  0.80, 3.00, 1.80),
    ("lift_front",   0.50, 2.00, 1.00),
    ("lift_hind",    0.50, 2.00, 1.00),
]

MS = 1500          # trial length during the search; 2 s bought nothing


def to_params(unit) -> GaitParams:
    """CMA-ES searches in [0, 1]^11; the bounds live here."""
    kwargs = {}
    for value, (name, lo, hi, _) in zip(unit, SPACE):
        kwargs[name] = float(lo + np.clip(value, 0.0, 1.0) * (hi - lo))
    return GaitParams(**kwargs)


def start_vector() -> np.ndarray:
    return np.array([(x0 - lo) / (hi - lo) for _, lo, hi, x0 in SPACE])


def cost(unit) -> float:
    try:
        return -float(evaluate(to_params(unit), ms=MS))
    except Exception:
        return 1e3


def main(a):
    import cma

    base = evaluate(GaitParams(), ms=MS, detail=True)
    print(f"defaults: fitness {base['fitness']:7.2f}  speed {base['speed']:5.2f} mm/s"
          f"  stride {base['stride_mm']:.2f} lift {base['lift_mm']:.2f} mm "
          f"({base['foot_ratio']:.1f}:1)", flush=True)

    if Path(a.compare).exists():
        old = GaitParams(**json.loads(Path(a.compare).read_text())["params"])
        m = evaluate(old, ms=MS, detail=True)
        print(f"previous: fitness {m['fitness']:7.2f}  speed {m['speed']:5.2f} mm/s"
              f"  stride {m['stride_mm']:.2f} lift {m['lift_mm']:.2f} mm "
              f"({m['foot_ratio']:.1f}:1)  roll {m['roll']:.1f} deg", flush=True)

    es = cma.CMAEvolutionStrategy(
        start_vector(), 0.25,
        {"bounds": [0.0, 1.0], "popsize": a.pop, "seed": a.seed,
         "maxiter": a.gens, "verbose": -9},
    )

    t0 = time.time()
    best, best_cost = None, np.inf
    with Pool(a.workers) as pool:
        while not es.stop():
            batch = es.ask()
            costs = pool.map(cost, batch)
            es.tell(batch, costs)
            i = int(np.argmin(costs))
            if costs[i] < best_cost:
                best_cost, best = costs[i], np.array(batch[i])
            m = evaluate(to_params(best), ms=MS, detail=True)
            print(f"  gen {es.countiter:3d}  fitness {-best_cost:7.2f}  "
                  f"speed {m['speed']:5.2f}  stride {m['stride_mm']:.2f}  "
                  f"lift {m['lift_mm']:.2f}  ratio {m['foot_ratio']:5.1f}:1  "
                  f"roll {m['roll']:5.1f}  ({time.time()-t0:5.0f}s)", flush=True)

    params = to_params(best)
    final = evaluate(params, ms=3000, detail=True)
    print(f"\n{es.countiter} generations, "
          f"{es.countiter * a.pop} simulations, {(time.time()-t0)/60:.1f} min")
    print(f"\n{'metric':12s} {'defaults':>10s} {'tuned':>10s}")
    for key in ("speed", "stride_mm", "lift_mm", "foot_ratio", "roll", "pitch",
                "lateral", "yaw", "height_std", "speed_std", "fitness"):
        print(f"{key:12s} {base[key]:10.3f} {final[key]:10.3f}")

    print("\nparameters")
    for name, _, _, _ in SPACE:
        print(f"  {name:14s} {getattr(params, name):8.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "params": {n: float(getattr(params, n)) for n, _, _, _ in SPACE},
        "metrics": final,
        "defaults": base,
        "simulations": int(es.countiter * a.pop),
        "note": "CMA-ES, gearing in the search, smoothness-weighted fitness",
    }, indent=2))
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--pop", type=int, default=14)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--compare", default="output/gait.json")
    main(ap.parse_args())
