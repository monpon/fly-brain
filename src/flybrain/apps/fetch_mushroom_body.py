#!/usr/bin/env python
"""Extract the mushroom body -- the learning circuit -- and cache it.

Olfactory PNs -> Kenyon cells -> MBONs, with dopaminergic neurons innervating
each compartment. The KC->MBON synapses are the plastic ones.

    flybrain fetch-mushroom-body
"""

import sys
import time
from pathlib import Path


import numpy as np

from flybrain import mushroom_body as MB

ROLE_NOTES = {
    "PN": "olfactory projection neurons (odour code)",
    "KC": "Kenyon cells (sparse expansion layer)",
    "APL": "giant GABAergic feedback (enforces sparseness)",
    "MBON": "output neurons (learned valence)",
    "DAN": "dopaminergic neurons (teaching signal)",
}


def main() -> None:
    print("extracting mushroom body...")
    t0 = time.time()
    mb = MB.extract()
    print(f"  {time.time() - t0:.1f}s\n")

    print(f"dataset: {mb.dataset}\n")
    print("populations")
    for role in MB.ROLES:
        n_types = len(set(mb.types[role]))
        print(
            f"  {role:5s} {mb.n(role):6d} neurons  {n_types:3d} types   "
            f"{ROLE_NOTES[role]}"
        )

    n_reward = int((mb.dan_valence > 0).sum())
    n_punish = int((mb.dan_valence < 0).sum())
    print(f"\n  of which {n_reward} reward DANs (PAM), {n_punish} punishment (PPL1)")

    print("\npathways (synapses)")
    for label, W in [
        ("PN   -> KC  ", mb.W_pn_kc),
        ("KC   -> MBON", mb.W_kc_mbon),
        ("KC   -> APL ", mb.W_kc_apl),
        ("APL  -> KC  ", mb.W_apl_kc),
        ("DAN  -> MBON", mb.W_dan_mbon),
    ]:
        edges = int((W != 0).sum())
        print(f"  {label}  {edges:8d} edges  {W.sum():10.0f} synapses")

    plastic = int((mb.W_kc_mbon != 0).sum())
    print(f"\n  {plastic} plastic KC->MBON synapses -- this is what gets trained")

    per_kc = (mb.W_pn_kc > 0).sum(axis=0)
    connected = per_kc[per_kc > 0]
    print(
        f"  each KC samples {connected.mean():.1f} PNs on average "
        f"(median {np.median(connected):.0f}), {len(connected)} KCs have PN input"
    )

    print(f"\nfingerprint: {mb.fingerprint()}")
    path = MB.save_cache(mb)
    print(f"cached -> {path}  ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
