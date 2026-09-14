"""Save a trained fly, and load it onto another copy of the brain.

The problem
-----------
A trained state is a number per synapse. The naive way to store that is an
array in row order, which is worthless the moment anything about the
extraction changes -- a different sort, a different confidence threshold, a
newer connectome release -- because row 4,001 is no longer the same synapse.

So nothing here is stored positionally. Every gain is written against the pair
of connectome body IDs it belongs to, plus a fingerprint of the topology it
was trained on. Loading is an explicit join, and it reports exactly how much
of the memory found a home.

Two kinds of "another copy of the brain"
----------------------------------------
**Same animal, re-extracted** (`mode="body"`). Body IDs are the same physical
neurons, so the join is exact and the memory transfers completely. This is the
case that matters for reproducibility: train here, ship the file, rerun there.

**A different animal or dataset** (`mode="type"`). Body IDs mean nothing
across brains, so gains are pooled by cell-type pair instead. Be clear about
what survives: compartment-level memory does, odour-specific memory does not.
That is not a limitation of the file format -- Kenyon cell odour tuning is set
by developmentally stochastic wiring and genuinely differs between individual
flies. There is no fact of the matter about which KC in *this* fly corresponds
to a given KC in *that* one. You can transfer "compartment gamma-1 was
depressed"; you cannot transfer "the memory of pentyl acetate".
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .learning import FlyBrain, LearningParams, TrainedState

FORMAT = "flybrain-checkpoint"
FORMAT_VERSION = 1


@dataclass
class LoadReport:
    """What actually happened during a load. Always worth printing."""

    mode: str
    fingerprint_match: bool
    source_dataset: str
    target_dataset: str
    edges_in_checkpoint: int
    edges_in_target: int
    matched: int
    unmatched_in_target: int
    dropped_from_checkpoint: int
    mean_gain_transferred: float
    trials: int

    @property
    def coverage(self) -> float:
        """Fraction of the target's plastic synapses that got a saved gain."""
        return self.matched / max(self.edges_in_target, 1)

    def __str__(self) -> str:
        head = "identical topology" if self.fingerprint_match else "DIFFERENT topology"
        lines = [
            f"loaded {self.trials} trials of training ({self.mode}-keyed, {head})",
            f"  source  {self.source_dataset}  {self.edges_in_checkpoint} synapses",
            f"  target  {self.target_dataset}  {self.edges_in_target} synapses",
            f"  matched {self.matched}  ({100 * self.coverage:.1f}% of target)",
        ]
        if self.unmatched_in_target:
            lines.append(
                f"  {self.unmatched_in_target} target synapses had no saved "
                "gain, left at baseline 1.0"
            )
        if self.dropped_from_checkpoint:
            lines.append(
                f"  {self.dropped_from_checkpoint} saved gains had no target "
                "synapse, discarded"
            )
        lines.append(f"  mean transferred gain {self.mean_gain_transferred:.4f}")
        if not self.fingerprint_match and self.mode == "body":
            lines.append(
                "  NOTE: fingerprints differ -- this is a different extraction. "
                "Body-ID matching still holds where neurons are shared."
            )
        return "\n".join(lines)


def _pair_key(pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    """Pack a (pre, post) body-ID pair into one sortable int64 per synapse.

    Body IDs exceed 2^32 in some releases, so packing by bit-shift is not
    safe. Structured void view is exact and still sorts/joins in one pass.
    """
    stacked = np.empty(len(pre), dtype=[("pre", np.int64), ("post", np.int64)])
    stacked["pre"] = pre
    stacked["post"] = post
    return stacked


def save(
    path: str | Path,
    fly: FlyBrain,
    *,
    task: str | None = None,
    notes: str | None = None,
) -> Path:
    """Write a portable trained state.

    The file is a zip: a readable JSON manifest plus compressed arrays. You
    can `unzip -p file.flyckpt manifest.json` to see what a checkpoint is
    without loading the connectome.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    mb = fly.mb
    pre_type = mb.types["KC"][fly.pre_idx]
    post_type = mb.types["MBON"][fly.post_idx]

    manifest = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset": mb.dataset,
        "fingerprint": mb.fingerprint(),
        "task": task,
        "notes": notes,
        "trials": int(fly.state.trials),
        "params": asdict(fly.params),
        "counts": {role: int(mb.n(role)) for role in mb.body_ids},
        "plastic_synapses": int(len(fly.state.gain)),
        "stats": {
            "mean_gain": float(fly.state.gain.mean()),
            "min_gain": float(fly.state.gain.min()),
            "depressed_fraction": fly.depressed_fraction(),
        },
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, indent=2))

        arrays = io.BytesIO()
        np.savez_compressed(
            arrays,
            pre_body=fly.pre_body,
            post_body=fly.post_body,
            pre_type=pre_type.astype(str),
            post_type=post_type.astype(str),
            gain=fly.state.gain,
            mbon_body=mb.body_ids["MBON"],
            mbon_type=mb.types["MBON"].astype(str),
            readout=fly.state.readout,
        )
        z.writestr("state.npz", arrays.getvalue())

    path.write_bytes(buf.getvalue())
    return path


def inspect(path: str | Path) -> dict:
    """Read a checkpoint's manifest without touching the connectome."""
    with zipfile.ZipFile(Path(path)) as z:
        return json.loads(z.read("manifest.json"))


def _read(path: Path) -> tuple[dict, dict]:
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read("manifest.json"))
        if manifest.get("format") != FORMAT:
            raise SystemExit(f"{path.name} is not a {FORMAT} file")
        if manifest.get("format_version", 0) > FORMAT_VERSION:
            raise SystemExit(
                f"{path.name} is format version "
                f"{manifest['format_version']}, this code reads {FORMAT_VERSION}"
            )
        with np.load(io.BytesIO(z.read("state.npz")), allow_pickle=False) as data:
            state = {k: data[k] for k in data.files}
    return manifest, state


def load(
    path: str | Path,
    fly: FlyBrain,
    *,
    mode: str = "auto",
    restore_params: bool = False,
) -> LoadReport:
    """Load a trained state onto `fly`, matching by neuron identity.

    `mode="auto"` uses body-ID matching when the datasets agree and falls back
    to cell-type pooling when they do not. Pass `mode` explicitly to force one.
    Mutates `fly.state` in place and returns a report.
    """
    path = Path(path)
    manifest, saved = _read(path)

    same_dataset = manifest["dataset"] == fly.mb.dataset
    if mode == "auto":
        mode = "body" if same_dataset else "type"
    if mode not in ("body", "type"):
        raise ValueError("mode must be 'auto', 'body', or 'type'")

    gain = np.ones(len(fly.pre_idx), dtype=np.float32)

    if mode == "body":
        matched, dropped = _join_by_body(fly, saved, gain)
    else:
        matched, dropped = _join_by_type(fly, saved, gain)

    fly.state = TrainedState(
        gain=gain,
        readout=_load_readout(fly, saved),
        trials=int(manifest.get("trials", 0)),
    )
    fly._trace[:] = 0.0

    if restore_params:
        known = {f for f in LearningParams.__dataclass_fields__}
        fly.params = LearningParams(
            **{k: v for k, v in manifest.get("params", {}).items() if k in known}
        )

    transferred = gain[gain < 1.0]
    return LoadReport(
        mode=mode,
        fingerprint_match=manifest.get("fingerprint") == fly.mb.fingerprint(),
        source_dataset=manifest["dataset"],
        target_dataset=fly.mb.dataset,
        edges_in_checkpoint=int(len(saved["gain"])),
        edges_in_target=int(len(fly.pre_idx)),
        matched=matched,
        unmatched_in_target=int(len(fly.pre_idx) - matched),
        dropped_from_checkpoint=dropped,
        mean_gain_transferred=(
            float(transferred.mean()) if transferred.size else 1.0
        ),
        trials=int(manifest.get("trials", 0)),
    )


def _join_by_body(fly: FlyBrain, saved: dict, gain: np.ndarray) -> tuple[int, int]:
    """Exact (pre_body, post_body) join. Same animal, any extraction order."""
    src = _pair_key(saved["pre_body"], saved["post_body"])
    dst = _pair_key(fly.pre_body, fly.post_body)

    order = np.argsort(src, order=("pre", "post"))
    src_sorted = src[order]
    pos = np.searchsorted(src_sorted, dst, sorter=None)
    pos = np.clip(pos, 0, len(src_sorted) - 1)
    hit = src_sorted[pos] == dst

    gain[hit] = saved["gain"][order][pos[hit]]
    matched = int(hit.sum())
    return matched, int(len(src) - matched)


def _join_by_type(fly: FlyBrain, saved: dict, gain: np.ndarray) -> tuple[int, int]:
    """Pool gains by (KC type, MBON type). Cross-animal transfer.

    Compartment-level memory survives this; odour identity does not, because
    which individual KCs encode an odour is not conserved between flies.
    """
    src_pairs = np.char.add(
        np.char.add(saved["pre_type"].astype(str), ">"),
        saved["post_type"].astype(str),
    )
    keys, inverse = np.unique(src_pairs, return_inverse=True)
    sums = np.bincount(inverse, weights=saved["gain"], minlength=len(keys))
    counts = np.bincount(inverse, minlength=len(keys))
    pooled = dict(zip(keys, sums / np.maximum(counts, 1)))

    dst_pairs = np.char.add(
        np.char.add(fly.mb.types["KC"][fly.pre_idx].astype(str), ">"),
        fly.mb.types["MBON"][fly.post_idx].astype(str),
    )
    matched = 0
    for i, key in enumerate(dst_pairs):
        value = pooled.get(key)
        if value is not None:
            gain[i] = value
            matched += 1

    unused = len(set(keys) - set(dst_pairs.tolist()))
    return matched, unused


def _load_readout(fly: FlyBrain, saved: dict) -> np.ndarray:
    """Restore the MBON readout, joined on MBON body ID."""
    readout = fly.default_readout()
    lookup = dict(zip(saved["mbon_body"].tolist(), saved["readout"].tolist()))
    for i, body in enumerate(fly.mb.body_ids["MBON"]):
        value = lookup.get(int(body))
        if value is not None:
            readout[i] = value
    return readout
