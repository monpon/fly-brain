"""Extract the mushroom body -- the fly's learning circuit -- from male-CNS.

Everything else in this project treats the connectome as fixed wiring. The
mushroom body is where that stops being true: KC->MBON synapses are the site
of olfactory associative memory in Drosophila, and dopaminergic neurons
depress them when odour and reinforcement coincide.

The circuit, in the order signal flows through it:

    PN    olfactory projection neurons, one per glomerulus. The odour code.
    KC    Kenyon cells. ~4,000 of them, each sampling a handful of PNs, so
          any one odour activates a sparse, near-random ~5% subset.
    APL   a single giant GABAergic cell per hemisphere. Feedback inhibition
          onto all KCs -- this is what *enforces* the sparseness.
    MBON  output neurons. ~20 compartments, each reading a different slice of
          the KC population. Their relative activity is the learned valence.
    DAN   dopaminergic neurons. PAM (~316 cells) carry reward, PPL1 (~16)
          carry punishment. Each innervates one compartment.

The plastic synapse is KC->MBON, and the teaching signal is DAN activity in
the same compartment. That is the whole learning rule, and it is implemented
in `learning.py`.

Compartment structure is not read from a table -- there isn't one. It is
inferred from DAN->MBON anatomical overlap, on the grounds that a DAN and an
MBON that share a compartment have interdigitating arbors and therefore
register synapses between them. That is an assumption, and a load-bearing one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import local_data as L

# Cell-type regexes. Naming shifts between connectome releases, so these are
# patterns to search with -- run `flybrain.local_data.find_types` to check.
ROLE_PATTERNS = {
    # Uniglomerular olfactory PNs: <glomerulus>_<tract>PN.
    "PN": r"[A-Z0-9]+[a-z]*_(?:ad|l|lv|il|v|vl|vm)PN\d*",
    "KC": r"KC.*",
    "APL": r"APL",
    "MBON": r"MBON.*",
    "DAN": r"PAM\d*|PPL1\d*",
}

# Which DANs carry which sign of reinforcement. PAM neurons signal reward
# (sugar, water); PPL1 neurons signal punishment (shock, bitter). This is one
# of the better-established facts about the circuit.
REWARD_DAN = r"PAM\d*"
PUNISH_DAN = r"PPL1\d*"

ROLES = ("PN", "KC", "APL", "MBON", "DAN")


@dataclass
class MushroomBody:
    """The learning circuit, as arrays keyed by stable neuron identity.

    Every matrix is indexed by position within a role, but `body_ids` keeps
    the mapping back to connectome body IDs. That indirection is what makes a
    trained state portable: gains are saved against body IDs, never against
    row numbers, so a checkpoint survives re-extraction in a different order.
    """

    dataset: str
    body_ids: dict[str, np.ndarray]
    types: dict[str, np.ndarray]

    # Feedforward pathway. Dense float32; these are small enough (276x4064 and
    # 4064x97) that sparsity buys nothing.
    W_pn_kc: np.ndarray
    W_kc_mbon: np.ndarray

    # Sparse-coding loop.
    W_kc_apl: np.ndarray
    W_apl_kc: np.ndarray

    # Teaching signal: DAN -> MBON anatomical overlap, read as compartment
    # membership rather than as a synaptic drive.
    W_dan_mbon: np.ndarray

    # +1 reward (PAM), -1 punishment (PPL1), per DAN.
    dan_valence: np.ndarray

    # +1 excitatory, -1 inhibitory, per MBON, from predicted neurotransmitter.
    mbon_sign: np.ndarray

    glomeruli: np.ndarray = field(default_factory=lambda: np.array([]))

    def n(self, role: str) -> int:
        return len(self.body_ids[role])

    def summary(self) -> str:
        parts = [f"{role} {self.n(role)}" for role in ROLES]
        return f"{self.dataset}: " + ", ".join(parts)

    # -- edge views -------------------------------------------------------
    # The plastic synapses, as an explicit (pre_body, post_body, weight) edge
    # list. This is the unit a checkpoint is written against.

    def plastic_edges(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(pre_idx, post_idx, pre_body, post_body) for every KC->MBON synapse."""
        pre_idx, post_idx = np.nonzero(self.W_kc_mbon)
        return (
            pre_idx,
            post_idx,
            self.body_ids["KC"][pre_idx],
            self.body_ids["MBON"][post_idx],
        )

    def fingerprint(self) -> str:
        """Stable hash of the plastic topology.

        Two extractions of the same circuit hash identically regardless of row
        order; a different dataset, threshold, or type selection does not. A
        checkpoint records this so a mismatched load is a warning, not a
        silent garbage result.
        """
        import hashlib

        _, _, pre, post = self.plastic_edges()
        order = np.lexsort((post, pre))
        buf = np.stack([pre[order], post[order]]).astype(np.int64).tobytes()
        return hashlib.sha256(buf).hexdigest()[:16]


def _role_of(types: pd.Series) -> pd.Series:
    """Assign each neuron to a role, first pattern wins."""
    out = pd.Series(["", ] * len(types), index=types.index, dtype=object)
    for role in ROLES:
        pattern = ROLE_PATTERNS[role]
        hit = types.str.fullmatch(pattern, na=False) & (out == "")
        out[hit] = role
    return out


def _matrix(
    edges: pd.DataFrame,
    pre_index: dict[int, int],
    post_index: dict[int, int],
    *,
    signs: np.ndarray | None = None,
) -> np.ndarray:
    """Synapse-count matrix between two labelled neuron sets."""
    W = np.zeros((len(pre_index), len(post_index)), dtype=np.float32)
    if edges.empty:
        return W
    rows = edges["body_pre"].map(pre_index).to_numpy()
    cols = edges["body_post"].map(post_index).to_numpy()
    keep = ~(pd.isna(rows) | pd.isna(cols))
    rows, cols = rows[keep].astype(int), cols[keep].astype(int)
    vals = edges["weight"].to_numpy(dtype=np.float32)[keep]
    if signs is not None:
        vals = vals * signs[rows]
    np.add.at(W, (rows, cols), vals)
    return W


def extract(dataset: str | None = None) -> MushroomBody:
    """Pull the mushroom body out of the local male-CNS tables.

    Reads the memory-mapped weights table once and slices it five ways, which
    is much cheaper than five separate passes over 151.8M rows.
    """
    ann = L.annotations()
    types = ann["type"].astype(str)
    roles = _role_of(types)

    selected = ann[roles != ""].copy()
    selected["role"] = roles[roles != ""]

    body_ids: dict[str, np.ndarray] = {}
    type_names: dict[str, np.ndarray] = {}
    for role in ROLES:
        block = selected[selected["role"] == role].sort_values("bodyId")
        if block.empty:
            raise SystemExit(
                f"No neurons matched role {role} "
                f"(pattern {ROLE_PATTERNS[role]!r}) in this dataset."
            )
        body_ids[role] = block["bodyId"].to_numpy(dtype=np.int64)
        type_names[role] = block["type"].astype(str).to_numpy()

    index = {
        role: {int(b): i for i, b in enumerate(ids)}
        for role, ids in body_ids.items()
    }
    role_of_body = {
        int(b): role for role, ids in body_ids.items() for b in ids
    }

    # One filtered pass over the big table.
    all_ids = np.concatenate([body_ids[r] for r in ROLES])
    edges = L.weights_between(all_ids.tolist())
    edges["role_pre"] = edges["body_pre"].map(role_of_body)
    edges["role_post"] = edges["body_post"].map(role_of_body)

    def block(pre_role: str, post_role: str) -> pd.DataFrame:
        return edges[
            (edges["role_pre"] == pre_role) & (edges["role_post"] == post_role)
        ]

    kc_signs = L._signs(pd.Series(body_ids["KC"]))
    apl_signs = L._signs(pd.Series(body_ids["APL"]))

    mb = MushroomBody(
        dataset=dataset or L.config.dataset_dir().name,
        body_ids=body_ids,
        types=type_names,
        W_pn_kc=_matrix(block("PN", "KC"), index["PN"], index["KC"]),
        # KC->MBON is left unsigned: KCs are cholinergic, and the sign that
        # matters downstream is the MBON's own transmitter, not the KC's.
        W_kc_mbon=_matrix(block("KC", "MBON"), index["KC"], index["MBON"]),
        W_kc_apl=_matrix(block("KC", "APL"), index["KC"], index["APL"]),
        W_apl_kc=_matrix(block("APL", "KC"), index["APL"], index["KC"]),
        W_dan_mbon=_matrix(block("DAN", "MBON"), index["DAN"], index["MBON"]),
        dan_valence=_dan_valence(type_names["DAN"]),
        mbon_sign=L._signs(pd.Series(body_ids["MBON"])),
        glomeruli=np.array(
            [t.split("_")[0] for t in type_names["PN"]], dtype=object
        ),
    )
    del kc_signs, apl_signs  # signs enter via the model, not the anatomy
    return mb


def _dan_valence(dan_types: np.ndarray) -> np.ndarray:
    """+1 for reward DANs (PAM), -1 for punishment DANs (PPL1)."""
    out = np.zeros(len(dan_types), dtype=np.float32)
    for i, t in enumerate(dan_types):
        if re.fullmatch(REWARD_DAN, t):
            out[i] = 1.0
        elif re.fullmatch(PUNISH_DAN, t):
            out[i] = -1.0
    return out


# -- caching --------------------------------------------------------------
# Extraction costs a filtered pass over the 151.8M-row weights table. Cache it
# so training runs start instantly.

_MATRICES = (
    "W_pn_kc", "W_kc_mbon", "W_kc_apl", "W_apl_kc", "W_dan_mbon",
    "dan_valence", "mbon_sign",
)


def cache_path():
    from . import config

    return config.cache_dir() / "mushroom_body.npz"


def save_cache(mb: MushroomBody, path=None):
    path = path or cache_path()
    payload = {name: getattr(mb, name) for name in _MATRICES}
    for role in ROLES:
        payload[f"body_{role}"] = mb.body_ids[role]
        payload[f"type_{role}"] = mb.types[role].astype(str)
    payload["glomeruli"] = mb.glomeruli.astype(str)
    payload["dataset"] = np.array(mb.dataset)
    np.savez_compressed(path, **payload)
    return path


def load_cache(path=None) -> MushroomBody:
    path = path or cache_path()
    if not path.exists():
        raise SystemExit(
            f"No cached mushroom body at {path}\n"
            "Run: .venv/bin/python scripts/fetch_mushroom_body.py"
        )
    with np.load(path, allow_pickle=False) as data:
        return MushroomBody(
            dataset=str(data["dataset"]),
            body_ids={r: data[f"body_{r}"] for r in ROLES},
            types={r: data[f"type_{r}"] for r in ROLES},
            glomeruli=data["glomeruli"],
            **{name: data[name] for name in _MATRICES},
        )
