"""Odour sources and antennal sampling.

FlyGym 2.1 exposes no olfaction -- "odor" survives only in a legacy config
file -- so the field is computed here and sampled at the real olfactory
organs. In Drosophila the odorant receptor neurons sit on the third antennal
segment (the funiculus) and the maxillary palp; this samples the funiculi,
which the model has as `l_funiculus` and `r_funiculus`.

The two antennae are **0.216 mm apart**. That is a very short baseline for
comparing concentrations, and it is precisely why real flies cannot navigate
by left-right comparison alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class OdorSource:
    """A point source with exponentially decaying concentration."""

    position: np.ndarray
    strength: float = 1.0
    decay_mm: float = 12.0      # length constant of the falloff
    odor: str = "vinegar"       # which molecule, i.e. which glomeruli it hits

    def concentration(self, points: np.ndarray) -> np.ndarray:
        """Concentration at one or more world positions."""
        points = np.atleast_2d(points)
        r = np.linalg.norm(points[:, :2] - np.asarray(self.position)[:2], axis=1)
        return self.strength * np.exp(-r / self.decay_mm)


class Nose:
    """Samples odour at the fly's two antennae."""

    def __init__(self, sim, fly_name: str = "fly"):
        import mujoco

        self.sim = sim
        names = [mujoco.mj_id2name(sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
                 for i in range(sim.mj_model.nbody)]
        self.left = names.index(f"{fly_name}/l_funiculus")
        self.right = names.index(f"{fly_name}/r_funiculus")
        self.fly_name = fly_name

    def sample(self, sources: list[OdorSource]) -> tuple[float, float]:
        """Concentration at the left and right funiculus."""
        pos = self.sim.get_body_positions(self.fly_name)
        points = np.stack([pos[self.left], pos[self.right]])
        total = np.zeros(2)
        for src in sources:
            total += src.concentration(points)
        return float(total[0]), float(total[1])

    def sample_each(self, sources: list[OdorSource]) -> np.ndarray:
        """(n_sources, 2) concentrations, left and right, kept separate.

        Separate because the antennal lobe keeps them separate: each
        glomerulus is a private channel from one receptor type, so a blend of
        two odours is not one number at the antenna, it is a pattern. Summing
        here would throw away exactly the thing that lets a fly approach one
        smell while avoiding another arriving from a different direction.
        """
        pos = self.sim.get_body_positions(self.fly_name)
        points = np.stack([pos[self.left], pos[self.right]])
        return np.stack([src.concentration(points) for src in sources])
