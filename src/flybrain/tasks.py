"""Things to teach a fly, and the interface for teaching it something else.

The built-in task is differential olfactory conditioning, because it is the
assay the mushroom body literature is built on: present odour A with shock,
present odour B without, then measure whether the fly avoids A relative to B.
The readout is a learning index, which is what the behavioural papers report,
so the model's output is directly comparable to published numbers.

`Environment` is the seam for everything else. Anything that can turn a state
into a PN activation vector and a reinforcement signal can train this brain --
including a physics simulation. `flygym`'s odour sensors return exactly the
kind of vector `observe()` is expected to produce.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Environment(Protocol):
    """Minimal interface for anything that can train a FlyBrain.

    Deliberately not gym.Env: the fly does not emit an action in a discrete
    space, it emits a scalar valence, and reinforcement splits into reward and
    punishment because they recruit anatomically distinct dopaminergic
    populations. Collapsing those into one signed number would throw away the
    structure the circuit is built around.
    """

    n_glomeruli: int

    def reset(self) -> np.ndarray:
        """Start an episode, return the initial PN activation."""
        ...

    def step(self, valence: float) -> tuple[np.ndarray, float, float, bool]:
        """Act on the fly's valence.

        Returns (pn_activation, reward, punishment, done), where reward and
        punishment are non-negative and drive PAM and PPL1 respectively.
        """
        ...


def make_odor(
    n_pn: int,
    glomeruli: np.ndarray,
    *,
    n_active: int = 12,
    seed: int = 0,
) -> np.ndarray:
    """A synthetic odour: activity across a random subset of glomeruli.

    Real odours activate a sparse, overlapping set of glomeruli with graded
    intensity. This reproduces the statistics, not any particular molecule --
    there is no odour-to-glomerulus map in the connectome, that comes from
    physiology.

    PNs sharing a glomerulus are co-activated, which matters: they are the
    reason KC input is correlated rather than independent.
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(glomeruli)
    chosen = rng.choice(unique, size=min(n_active, len(unique)), replace=False)
    strength = dict(zip(chosen, rng.uniform(0.5, 1.0, size=len(chosen))))

    pn = np.zeros(n_pn, dtype=np.float32)
    for i, g in enumerate(glomeruli):
        pn[i] = strength.get(g, 0.0)
    return pn


@dataclass
class OdorConditioning:
    """Differential conditioning: one odour punished, one not.

    This is the Tully-Quinn paradigm as run on a model. `cs_plus` is paired
    with shock, `cs_minus` is presented alone, and a trained fly should
    develop a more negative valence for the former.
    """

    cs_plus: np.ndarray
    cs_minus: np.ndarray
    shock: float = 1.0
    reward: float = 0.0

    @classmethod
    def build(
        cls,
        mb,
        *,
        seed: int = 0,
        n_active: int = 12,
        overlap: bool = False,
        **kwargs,
    ) -> "OdorConditioning":
        """Two odours drawn for a given mushroom body."""
        n_pn = mb.n("PN")
        plus = make_odor(n_pn, mb.glomeruli, n_active=n_active, seed=seed)
        minus = make_odor(
            n_pn, mb.glomeruli, n_active=n_active, seed=seed if overlap else seed + 977
        )
        return cls(cs_plus=plus, cs_minus=minus, **kwargs)

    def odor_overlap(self, fly) -> float:
        """Fraction of active Kenyon cells the two odours share.

        The thing that actually determines whether the task is learnable. KC
        codes for distinct odours overlap little by design, which is what lets
        a depression-only rule store two memories without interference.
        """
        a = fly.kenyon_cells(self.cs_plus) > 0
        b = fly.kenyon_cells(self.cs_minus) > 0
        union = (a | b).sum()
        return float((a & b).sum() / union) if union else 0.0

    def train(
        self, fly, *, epochs: int = 20, interleave: bool = True
    ) -> list[dict]:
        """Run the conditioning protocol.

        One epoch is a paired CS+ presentation and an unpaired CS-. The trace
        is cleared between them so eligibility from one odour cannot be
        captured by the other's (absent) reinforcement.
        """
        history = []
        for epoch in range(epochs):
            fly._trace[:] = 0.0
            paired = fly.learn(
                self.cs_plus, punishment=self.shock, reward=self.reward
            )
            fly._trace[:] = 0.0
            unpaired = fly.learn(self.cs_minus)

            history.append(
                {
                    "epoch": epoch + 1,
                    "v_plus": paired["valence"],
                    "v_minus": unpaired["valence"],
                    "mean_gain": paired.get("mean_gain", 1.0),
                }
            )
            if not interleave:
                continue
        return history

    def test(self, fly) -> dict[str, float]:
        """Measure what was learned, without changing it."""
        v_plus = fly.respond(self.cs_plus)
        v_minus = fly.respond(self.cs_minus)
        return {
            "v_plus": v_plus,
            "v_minus": v_minus,
            "preference": preference(v_plus, v_minus),
        }


def preference(v_plus: float, v_minus: float) -> float:
    """Raw valence difference between the two odours, CS+ minus CS-."""
    return float(v_plus - v_minus)


def learning_index(before: dict, after: dict) -> float:
    """How far conditioning moved the fly's preference, as a signed fraction.

    Measured as a *shift*, not an absolute preference, because two odours
    almost never start out equally attractive: they recruit different Kenyon
    cells, so they drive different compartments before any training happens.
    An absolute index would mostly report that initial asymmetry.

    Negative means the fly moved against the punished odour -- successful
    aversive conditioning. Positive means it moved toward it, which is what
    appetitive training should produce.
    """
    shift = preference(after["v_plus"], after["v_minus"]) - preference(
        before["v_plus"], before["v_minus"]
    )
    scale = abs(before["v_plus"]) + abs(before["v_minus"])
    return float(shift / scale) if scale > 1e-12 else 0.0


@dataclass
class ValencePair:
    """Teach opposite signs to two odours at once.

    Differential conditioning pairs one odour with reinforcement and shows the
    other alone. This is the two-sided version: the good odour arrives with
    sugar (PAM dopamine), the bad one with shock (PPL1). Both are still pure
    depression -- nothing is ever potentiated -- and the opposite behavioural
    outcomes come from the fact that reward and punishment DANs innervate
    different compartments, whose MBONs push behaviour opposite ways.

    Unlike `OdorConditioning`, the odours here are real glomerular patterns
    from `odors.py` rather than random draws, because the point is to have the
    fly act on them in a physics simulation where the sources smell of
    something specific.
    """

    good: np.ndarray
    bad: np.ndarray
    reward: float = 1.0
    shock: float = 1.0

    @classmethod
    def build(cls, mb, good: str = "vinegar", bad: str = "geosmin",
              pn_gain: float = 30.0, **kwargs) -> "ValencePair":
        from . import odors

        return cls(
            good=odors.pn_vector(mb, good, 1.0, pn_gain),
            bad=odors.pn_vector(mb, bad, 1.0, pn_gain),
            **kwargs,
        )

    def overlap(self, fly) -> float:
        a = fly.kenyon_cells(self.good) > 0
        b = fly.kenyon_cells(self.bad) > 0
        union = (a | b).sum()
        return float((a & b).sum() / union) if union else 0.0

    def train(self, fly, *, epochs: int = 40) -> list[dict]:
        history = []
        for epoch in range(epochs):
            # The trace is cleared between odours so eligibility tagged by one
            # cannot be captured by the other's dopamine -- that leak is what
            # would make the fly learn to avoid both.
            fly._trace[:] = 0.0
            rewarded = fly.learn(self.good, reward=self.reward)
            fly._trace[:] = 0.0
            punished = fly.learn(self.bad, punishment=self.shock)
            history.append({
                "epoch": epoch + 1,
                "v_good": rewarded["valence"],
                "v_bad": punished["valence"],
                "mean_gain": punished.get("mean_gain", 1.0),
            })
        return history

    def test(self, fly) -> dict[str, float]:
        good, bad = fly.respond(self.good), fly.respond(self.bad)
        return {"v_good": good, "v_bad": bad, "separation": good - bad}
