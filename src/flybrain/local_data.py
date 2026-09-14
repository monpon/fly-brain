"""Read the bulk male-CNS tables.

This is the no-token path: the same data neuPrint serves, as flat Feather
files. Slower to get started (a 1.1 GB download), but then everything is
local, offline, and reproducible.

The download happens by itself the first time a table is touched -- see
`data.py` -- into whichever directory `config.data_dir()` resolves to on this
OS. Nothing here assumes a particular mount point, and most of the package
never gets this far, because the circuits worth starting from are bundled.

Files, in `config.dataset_dir()`:
    connectome-weights-*.feather    synapse counts per connected pair
    body-annotations-*.feather      cell type / instance / soma side per body
    body-neurotransmitters-*.feather  predicted NT per body
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.compute as pc
import pyarrow.feather

from . import config

FILES = {
    "weights": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
}

# Glutamate is predominantly inhibitory in Drosophila (GluCl receptors).
INHIBITORY = {"gaba", "glutamate"}


def _path(key: str):
    """Path to one table, fetching it if this machine does not have it yet.

    Downloading on demand rather than failing with instructions is the whole
    difference between "clone it and play" and "clone it, read the README,
    find a 1.1 GB download, come back tomorrow".
    """
    from . import data

    return data.fetch(FILES[key])


@functools.lru_cache(maxsize=None)
def load(key: str) -> pd.DataFrame:
    """Load and cache one table as pandas.

    Do not use this for "weights" -- it has 151.8M rows and materializing it
    as a DataFrame costs ~9 GB. Use `weights_between()` instead, which filters
    in Arrow against a memory-mapped file and only converts the matching rows.
    """
    return pd.read_feather(_path(key))


def weights_between(body_ids) -> pd.DataFrame:
    """Rows of the connectome weights table with both ends inside `body_ids`.

    Filters on a memory-mapped Arrow table so peak memory stays proportional
    to the circuit, not to the 151.8M-row file.
    """
    table = pyarrow.feather.read_table(_path("weights"), memory_map=True)
    wanted = pyarrow.array(list(body_ids))
    mask = pc.and_(
        pc.is_in(table["body_pre"], value_set=wanted),
        pc.is_in(table["body_post"], value_set=wanted),
    )
    return table.filter(mask).to_pandas()


def annotations() -> pd.DataFrame:
    return load("annotations")


def weights() -> pd.DataFrame:
    """The full 151.8M-row table. Expensive -- prefer `weights_between()`."""
    return load("weights")


def neurotransmitters() -> pd.DataFrame:
    return load("neurotransmitters")


def find_types(pattern: str) -> pd.DataFrame:
    """Cell types whose name matches a regex, with neuron counts.

    Type naming varies between connectome releases -- always look before
    assuming a name exists.
    """
    ann = annotations()
    col = _type_column(ann)
    hits = ann[ann[col].astype(str).str.contains(pattern, regex=True, na=False)]
    return (
        hits.groupby(col).size().reset_index(name="n_neurons")
        .sort_values("n_neurons", ascending=False).reset_index(drop=True)
    )


def _type_column(frame: pd.DataFrame) -> str:
    for candidate in ("type", "cellType", "celltype", "instance"):
        if candidate in frame.columns:
            return candidate
    raise KeyError(f"No cell-type column found in: {list(frame.columns)}")


def _body_column(frame: pd.DataFrame) -> str:
    for candidate in ("bodyId", "body_id", "bodyid", "body"):
        if candidate in frame.columns:
            return candidate
    raise KeyError(f"No body-id column found in: {list(frame.columns)}")


def circuit_weight_matrix(
    types: list[str],
    *,
    regex: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Signed weight matrix for every neuron of the given cell types.

    This is where anatomy becomes a model, and where the assumptions live:
      - weight proportional to synapse count (no measured strengths exist)
      - sign from predicted neurotransmitter, excitatory when unknown
      - synapse location on the dendrite ignored

    Returns (W, neurons) where W[i, j] is the weight from neurons.iloc[i]
    onto neurons.iloc[j].
    """
    ann = annotations()
    type_col, body_col = _type_column(ann), _body_column(ann)

    pattern = "|".join(f"(?:{t})" for t in types) if regex else None
    if regex:
        mask = ann[type_col].astype(str).str.fullmatch(pattern, na=False)
    else:
        mask = ann[type_col].isin(types)
    neurons = ann[mask].reset_index(drop=True)
    if neurons.empty:
        raise SystemExit(f"No neurons matched {types}")

    index = {bid: i for i, bid in enumerate(neurons[body_col])}

    sub = weights_between(neurons[body_col].tolist())
    pre_col = next(c for c in sub.columns if "pre" in c.lower())
    post_col = next(c for c in sub.columns if "post" in c.lower())
    weight_col = next(
        c for c in sub.columns if c.lower() in ("weight", "count", "syn_count")
    )

    signs = _signs(neurons[body_col])
    W = np.zeros((len(neurons), len(neurons)), dtype=np.float32)
    rows = sub[pre_col].map(index).to_numpy()
    cols = sub[post_col].map(index).to_numpy()
    vals = sub[weight_col].to_numpy(dtype=np.float32)
    W[rows, cols] = vals * signs[rows]

    return W, neurons


def _signs(body_ids: pd.Series) -> np.ndarray:
    """+1 excitatory, -1 inhibitory, per neuron, from NT prediction."""
    nt = neurotransmitters()
    body_col = _body_column(nt)
    # Prefer the consensus call, then the per-body prediction. Do NOT match
    # loosely on "nt" -- that hits total_nt_predictions, which is a count.
    nt_col = next(
        (c for c in ("consensus_nt", "predicted_nt", "celltype_predicted_nt")
         if c in nt.columns),
        None,
    )
    if nt_col is None:
        return np.ones(len(body_ids), dtype=np.float32)

    lookup = dict(zip(nt[body_col], nt[nt_col].astype(str).str.lower()))
    return np.array(
        [-1.0 if lookup.get(b, "") in INHIBITORY else 1.0 for b in body_ids],
        dtype=np.float32,
    )
