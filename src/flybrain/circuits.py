"""Circuits you can have without downloading anything.

The barrier to playing with this was never the code, it was that nothing at
all worked until 1.1 GB of Feather tables had landed on a specific external
drive. That is a bad first five minutes.

So the circuits worth starting from ship *inside the package*, precomputed:

    mushroom_body()   4,161 neurons   olfactory learning, PN -> KC -> MBON
    optomotor()      18,926 neurons   T4/T5 -> HS/VS -> DNa02 steering
    lamina()                          early vision into direction selectivity

Each is an already-extracted `trainable.Circuit` stored as a compressed npz:
an edge list of int32 indices and float32 weights, which for the mushroom
body is about 700 KB. Cheap to ship, instant to load, works on a plane.

`from_types()` is the escape hatch. Ask for anything not bundled and it falls
through to the full tables, downloading them on first use. The bundled
circuits are a fast path, not a ceiling.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .trainable import Circuit

BUNDLE_DIR = Path(__file__).resolve().parent / "_bundled"
MANIFEST = BUNDLE_DIR / "manifest.json"

# Kept here rather than only in the manifest so `available()` can describe a
# circuit that has not been generated yet, and say how to generate it.
#
# The patterns are the *measured* type names in male-cns v1.0, which are not
# the ones you would guess: olfactory projection neurons are not called "PN",
# they are `<glomerulus>_<tract>PN` (DA1_lPN, VM5d_adPN, ...). Check a pattern
# against `local_data.find_types()` before adding an entry here.
CATALOG = {
    "mushroom_body": {
        # Built through `mushroom_body.extract()` rather than by regex, so the
        # bundled circuit is the same 276-in / 97-out object the rest of the
        # project and docs/FINDINGS.md refer to.
        "builder": "mushroom_body",
        "about": "olfactory learning: PN -> KC -> MBON, the plastic circuit",
    },
    "optomotor": {
        "types": ["T4[a-d]", "T5[a-d]", "Mi1", "Mi4", "Mi9",
                  "HS[NES]", "VS", "DNa02"],
        "inputs": ["T4[a-d]", "T5[a-d]"],
        "outputs": ["DNa02"],
        "about": "wide-field motion to a descending steering command",
    },
    "lamina": {
        # The circuit `see.py` and `Binocular` are demonstrated with.
        "types": ["L1", "L2", "L3", "L5", "T4[a-d]", "T5[a-d]"],
        "inputs": ["L1", "L2", "L3", "L5"],
        "outputs": ["T4[a-d]", "T5[a-d]"],
        "about": "early vision: lamina monopolars into direction selectivity",
    },
}


class NotBundled(KeyError):
    """A circuit name that did not ship with this install."""


def available() -> dict[str, bool]:
    """Every catalogued circuit, and whether its data shipped with the wheel."""
    return {name: (BUNDLE_DIR / f"{name}.npz").exists() for name in CATALOG}


def describe() -> str:
    lines = ["bundled circuits:"]
    for name, present in available().items():
        mark = "x" if present else " "
        lines.append(f"  [{mark}] {name:14s} {CATALOG[name]['about']}")
    if not all(available().values()):
        lines.append("")
        lines.append("  unbundled entries need the full tables; they will "
                     "download on first use.")
    return "\n".join(lines)


def load(name: str) -> Circuit:
    """Load a bundled circuit by name. Raises NotBundled if it did not ship."""
    path = BUNDLE_DIR / f"{name}.npz"
    if not path.exists():
        raise NotBundled(
            f"{name!r} is not bundled in this install.\n"
            f"Build it from the full tables with:  "
            f"flybrain fetch && flybrain bundle {name}"
        )
    return read_npz(path)


def read_npz(path: str | Path) -> Circuit:
    """A Circuit from the on-disk form. The inverse of `write_npz`."""
    with np.load(path, allow_pickle=False) as z:
        return Circuit(
            body_ids=z["body_ids"].astype(np.int64),
            types=z["types"].astype(str),
            pre=z["pre"].astype(np.int64),
            post=z["post"].astype(np.int64),
            weight=z["weight"].astype(np.float32),
            input_idx=z["input_idx"].astype(np.int64),
            output_idx=z["output_idx"].astype(np.int64),
            raw_weight=(z["raw_weight"].astype(np.float32)
                        if "raw_weight" in z else None),
            dataset=str(z["dataset"][()]) if "dataset" in z else "",
        )


def write_npz(circuit: Circuit, path: str | Path) -> Path:
    """Serialise a Circuit compactly.

    Edge indices are stored as int32 -- no circuit here has two billion
    neurons -- and the arrays compress well, because an edge list sorted by
    presynaptic index is highly repetitive. Body IDs stay int64: they are
    64-bit connectome identifiers, and truncating them would silently break
    every checkpoint, which keys its gains by exactly these numbers.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        body_ids=circuit.body_ids.astype(np.int64),
        types=circuit.types.astype(str),
        pre=circuit.pre.astype(np.int32),
        post=circuit.post.astype(np.int32),
        weight=circuit.weight.astype(np.float32),
        input_idx=circuit.input_idx.astype(np.int32),
        output_idx=circuit.output_idx.astype(np.int32),
        dataset=np.array(circuit.dataset),
    )
    if circuit.raw_weight is not None:
        payload["raw_weight"] = circuit.raw_weight.astype(np.float32)
    np.savez_compressed(path, **payload)
    return path


def from_types(types, inputs=None, outputs=None, *, budget: float = 1.0,
               regex: bool = True) -> Circuit:
    """Extract any circuit by cell type, downloading the tables if needed.

    This is the general path. It costs a 1.1 GB download the first time and is
    then local forever.
    """
    from . import data
    from .trainable import build

    if data.missing_core():
        print("this circuit is not bundled; fetching the connectome tables "
              "(1.1 GB, one time)", flush=True)
        data.fetch_core()

    types = [types] if isinstance(types, str) else list(types)
    return build(types, list(inputs or types), list(outputs or types),
                 budget=budget, regex=regex)


def rebuild(name: str) -> Circuit:
    """Extract a catalogued circuit from the full tables, ignoring the bundle.

    This is what `flybrain bundle` runs. Two shapes of entry: most are a set
    of cell-type patterns, but the mushroom body goes through its own
    extractor because it has roles (KC, APL, DAN) that a type regex does not
    capture.
    """
    spec = CATALOG[name]
    if spec.get("builder") == "mushroom_body":
        from . import data, mushroom_body as MB
        from .trainable import from_mushroom_body

        if data.missing_core():
            data.fetch_core()
        return from_mushroom_body(MB.extract())
    return from_types(spec["types"], spec["inputs"], spec["outputs"])


def _bundled_or_build(name: str) -> Circuit:
    try:
        return load(name)
    except NotBundled:
        return rebuild(name)


def mushroom_body() -> Circuit:
    """The olfactory learning circuit. Bundled: no download, no token."""
    return _bundled_or_build("mushroom_body")


def optomotor() -> Circuit:
    """Visual motion to a descending steering command. Bundled."""
    return _bundled_or_build("optomotor")


def lamina() -> Circuit:
    """Early vision, lamina through direction selectivity. Bundled."""
    return _bundled_or_build("lamina")


def retina(side: str = "R"):
    """The measured ommatidial lattice for one eye.

    Bundled separately from the circuits because it is a coordinate table
    rather than a wiring graph: the hex addresses the connectome assigns to
    each column.
    """
    from .vision import Retina

    path = BUNDLE_DIR / f"retina_{side.upper()}.npz"
    if not path.exists():
        from . import vision

        return vision.retina(side=side)
    with np.load(path, allow_pickle=False) as z:
        return Retina(
            side=str(z["side"][()]),
            body_ids=z["body_ids"].astype(np.int64),
            hex1=z["hex1"], hex2=z["hex2"], centers=z["centers"],
            kernel_size=int(z["kernel_size"][()]),
            type_name=str(z["type_name"][()]),
        )


def write_retina(ret, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, side=np.array(ret.side), body_ids=ret.body_ids.astype(np.int64),
        hex1=ret.hex1, hex2=ret.hex2, centers=ret.centers,
        kernel_size=np.array(ret.kernel_size),
        type_name=np.array(ret.type_name),
    )
    return path


def write_manifest() -> Path:
    """Record what is in the bundle, so an install can report its provenance."""
    entries = {}
    for name in CATALOG:
        path = BUNDLE_DIR / f"{name}.npz"
        if not path.exists():
            continue
        circuit = read_npz(path)
        entries[name] = {
            "neurons": circuit.n,
            "synapses": circuit.n_synapses,
            "dataset": circuit.dataset,
            "fingerprint": circuit.fingerprint(),
            "bytes": path.stat().st_size,
        }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return MANIFEST


def manifest() -> dict:
    if not MANIFEST.exists():
        return {}
    return json.loads(MANIFEST.read_text(encoding="utf-8"))
