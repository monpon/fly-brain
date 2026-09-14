"""Named odours, as glomerular activation patterns.

The connectome says which projection neuron belongs to which glomerulus. It
cannot say which *molecule* opens which receptor -- that comes from
physiology, from decades of single-sensillum recordings and calcium imaging.
So this file is the seam where measured biology is bolted onto measured
anatomy, and every entry here is a citation rather than a computation.

The patterns are deliberately coarse: the strongest-responding glomeruli for
each odour, at roughly their relative response amplitude. A real odour at a
real concentration recruits a long tail of weakly-activated channels, and
that tail matters for discrimination between similar molecules. It does not
matter for the question this project asks, which is whether a fly can learn
that one smell is worth walking toward and another is worth walking away
from.

Sources: Hallem & Carlson 2006 (receptor-odorant map); Stensmyr et al. 2012
(geosmin -> DA2, a dedicated labelled line for microbial spoilage); Suh et al.
2004 (CO2 -> V, the avoidance channel); Semmelhack & Wang 2009 (apple cider
vinegar -> DM1/VA2 attraction); Kurtovic et al. 2007 (cVA -> DA1).
"""

from __future__ import annotations

import numpy as np

# glomerulus -> relative response. Values are ratios, not absolute rates; the
# navigator scales the whole vector by concentration.
ODORS: dict[str, dict[str, float]] = {
    # Apple cider vinegar: the standard attractive stimulus in walking
    # assays. DM1 and VA2 carry most of the attraction on their own.
    "vinegar": {"DM1": 1.0, "VA2": 0.9, "DM4": 0.7, "DC2": 0.5, "VM2": 0.5},

    # Geosmin: produced by harmful moulds, detected by a single glomerulus,
    # and innately aversive even to a starving fly. The cleanest labelled
    # line in the olfactory system.
    "geosmin": {"DA2": 1.0},

    # CO2: the stress-odour channel. Aversive to walking flies.
    "co2": {"V": 1.0},

    # Ethyl butyrate: ripe fruit ester, broadly attractive, hits a wide set
    # of channels -- the opposite coding strategy to geosmin.
    "banana": {"DM2": 1.0, "VM2": 0.8, "DM1": 0.6, "VA2": 0.5,
               "DM5": 0.4, "VM3": 0.4},

    # Benzaldehyde: bitter-almond, mildly aversive, overlaps a little with
    # the fruit esters. Useful as a hard discrimination against "banana".
    "almond": {"DM5": 1.0, "DL5": 0.8, "DM3": 0.5, "DM2": 0.4},

    # cVA, the male pheromone. Not food, included because DA1 is the most
    # famous glomerulus in the fly and it makes a good control.
    "pheromone": {"DA1": 1.0},
}


def glomeruli_of(name: str) -> list[str]:
    try:
        return list(ODORS[name])
    except KeyError:
        raise SystemExit(
            f"unknown odour {name!r}; known: {', '.join(sorted(ODORS))}"
        ) from None


def pn_vector(mb, name: str, concentration: float = 1.0,
              gain: float = 30.0) -> np.ndarray:
    """Projection-neuron activity for an odour at a given concentration.

    Every PN of a responsive glomerulus gets the same rate, which is the
    approximation that hurts most: sister PNs of one glomerulus have
    correlated but not identical responses, and the differences carry timing
    information. They are averaged away here.
    """
    glom = np.asarray(mb.glomeruli).astype(str)
    pn = np.zeros(len(glom), dtype=np.float32)
    for g, level in ODORS[name].items():
        pn[glom == g] = gain * level * concentration
    if not pn.any():
        raise SystemExit(
            f"odour {name!r} activates no glomerulus present in this dataset"
        )
    return pn


def mask_of(mb, name: str) -> np.ndarray:
    """Boolean mask over PNs: which ones this odour drives at all."""
    glom = np.asarray(mb.glomeruli).astype(str)
    return np.isin(glom, glomeruli_of(name))
