"""Pull circuits out of neuPrint and turn them into weight matrices.

This is the *token* path, kept for exploratory queries. The bulk tables on the
storage volume cover the same ground without auth -- see `local_data.py`, which
is what the scripts actually use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# The optomotor response: wide-field visual motion drives a compensatory turn.
# Cell-type names differ between datasets, so these are regexes to search with
# rather than literal names -- run discover_types() to see what a dataset uses.
OPTOMOTOR_TYPES = {
    "T4": r"T4[a-d]?",          # ON-edge direction-selective, 4 directions
    "T5": r"T5[a-d]?",          # OFF-edge direction-selective
    "HS": r"HS[NES]?",          # horizontal system LPTCs (lobula plate)
    "VS": r"VS",                # vertical system LPTC (one type in male-cns)
    "DNa02": r"DNa02",          # descending neuron for steering
}


def discover_types(patterns: dict[str, str] | None = None) -> pd.DataFrame:
    """Find which cell types in the configured dataset match each pattern.

    Type naming is not consistent across connectome releases, so always run
    this before assuming a name exists.
    """
    from neuprint import NeuronCriteria as NC
    from neuprint import fetch_neurons

    patterns = patterns or OPTOMOTOR_TYPES
    config.client()

    rows = []
    for label, pattern in patterns.items():
        neurons, _ = fetch_neurons(NC(type=pattern, regex=True))
        if neurons.empty:
            rows.append({"group": label, "type": None, "n_neurons": 0})
            continue
        for type_name, group in neurons.groupby("type"):
            rows.append(
                {"group": label, "type": type_name, "n_neurons": len(group)}
            )
    return pd.DataFrame(rows).sort_values(["group", "type"]).reset_index(drop=True)


def fetch_circuit(types: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch neurons of the given types and the synapses between them.

    Returns (neurons, connections). `connections` has one row per
    bodyId_pre -> bodyId_post pair with a synapse `weight`.
    """
    from neuprint import NeuronCriteria as NC
    from neuprint import fetch_adjacencies

    config.client()
    criteria = NC(type=types, regex=True)
    neurons, connections = fetch_adjacencies(criteria, criteria)
    return neurons, connections


def to_weight_matrix(
    neurons: pd.DataFrame,
    connections: pd.DataFrame,
    *,
    scale: float = 1.0,
) -> tuple[np.ndarray, list[int]]:
    """Build a signed weight matrix from synapse counts.

    This is where anatomy becomes a model, and where the assumptions live:

      - weight is proportional to synapse count (no measured strengths exist)
      - sign comes from the predicted neurotransmitter, defaulting to
        excitatory when no prediction is available
      - every synapse of a given neuron is treated as equally effective,
        ignoring where on the dendrite it lands

    Returns (W, body_ids) where W[i, j] is the weight from body_ids[i] onto
    body_ids[j].
    """
    body_ids = sorted(neurons["bodyId"].tolist())
    index = {bid: i for i, bid in enumerate(body_ids)}

    inhibitory = {"gaba", "glutamate"}  # glutamate is mostly inhibitory in fly
    nt_col = next(
        (c for c in ("predictedNt", "consensusNt", "nt") if c in neurons.columns),
        None,
    )
    sign = {}
    for _, row in neurons.iterrows():
        nt = str(row[nt_col]).lower() if nt_col else ""
        sign[row["bodyId"]] = -1.0 if nt in inhibitory else 1.0

    W = np.zeros((len(body_ids), len(body_ids)), dtype=np.float32)
    for _, row in connections.iterrows():
        pre, post = row["bodyId_pre"], row["bodyId_post"]
        if pre in index and post in index:
            W[index[pre], index[post]] = scale * row["weight"] * sign.get(pre, 1.0)

    return W, body_ids


def save(name: str, **frames: pd.DataFrame) -> None:
    """Cache dataframes under data/cache/."""
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for key, frame in frames.items():
        path = config.CACHE_DIR / f"{name}_{key}.parquet"
        frame.to_parquet(path)
        print(f"  wrote {path.relative_to(config.PROJECT_ROOT)}  ({len(frame)} rows)")
