"""A trainable fly: dopamine-gated plasticity at the KC->MBON synapse.

This is a rate model, not a spiking one, and that is a deliberate trade. The
Brian2 network in `network.py` answers "what does this circuit do to a pulse
of input"; training answers "what does this circuit do after a thousand
trials", and a thousand trials of 4,000 LIF neurons is hours of CPU where this
is seconds. The plasticity rule is the part that has to be right.

The rule
--------
Associative memory in the fly is *depression* of KC->MBON synapses when
Kenyon cell activity coincides with dopamine in the same compartment:

    dg_ij = -lr * e_i * DA_j * g_ij

`g_ij` is a multiplicative gain on the anatomical weight, starting at 1.0 and
falling toward 0. `e_i` is a presynaptic eligibility trace -- KC activity that
decays over a few hundred ms, which is what lets reinforcement arrive *after*
the odour has gone. `DA_j` is dopamine in the compartment MBON j reads from.

Note what is absent: no error signal, no gradient, no backprop. Dopamine does
not say "you were wrong by this much", it says "something good/bad happened
here, now". The valence falls out of *which* compartments get depressed, since
each MBON pushes behaviour a different way.

Storing gains multiplicatively rather than as absolute weights is what makes a
trained memory portable. A gain of 0.4 means "this synapse was depressed to
40% of whatever it anatomically was", which stays meaningful when the same
memory is dropped onto a different copy of the brain whose raw synapse counts
differ. See `checkpoint.py`.

What is assumed, and is not in the connectome
---------------------------------------------
  - KC sparseness is imposed (k-winners-take-all standing in for APL feedback)
    rather than emerging from the dynamics
  - every MBON's total input is rescaled to a fixed budget, for the reason
    documented in docs/FINDINGS.md: raw synapse counts span three orders of
    magnitude and a single scale cannot serve all of them
  - compartment membership is inferred from DAN->MBON overlap
  - the mapping from MBON rates to a behavioural valence is a free readout,
    initialised from neurotransmitter sign
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .mushroom_body import MushroomBody


@dataclass(frozen=True)
class LearningParams:
    """Plasticity and coding parameters. None of these are in the EM."""

    kc_sparsity: float = 0.05      # fraction of KCs active per odour (APL)
    kc_gain: float = 1.0           # scale on PN->KC drive before the k-WTA
    mbon_input_budget: float = 1.0  # total |KC->MBON| weight per MBON
    dan_input_budget: float = 1.0  # total DAN->MBON coupling per MBON

    learning_rate: float = 0.35    # depression per unit KC x DA coincidence
    trace_tau: float = 2.0         # eligibility decay, in trials
    recovery: float = 0.002        # drift of gain back toward 1.0 per trial
    gain_floor: float = 0.05       # synapses are depressed, never abolished

    def __post_init__(self) -> None:
        if not 0.0 < self.kc_sparsity <= 1.0:
            raise ValueError("kc_sparsity must be in (0, 1]")


@dataclass
class TrainedState:
    """Everything a fly learns. This is what gets saved and transferred.

    `gain` is per KC->MBON synapse, aligned to `FlyBrain.pre_body` /
    `post_body`. `readout` is per MBON. Both are keyed to body IDs by the
    owning FlyBrain, never to row positions.
    """

    gain: np.ndarray
    readout: np.ndarray
    trials: int = 0

    def copy(self) -> "TrainedState":
        return TrainedState(self.gain.copy(), self.readout.copy(), self.trials)


class FlyBrain:
    """A mushroom body you can teach.

    >>> mb = mushroom_body.extract()
    >>> fly = FlyBrain(mb)
    >>> fly.respond(odor)                     # valence before training
    >>> fly.learn(odor, punishment=1.0)       # pair odour with shock
    >>> fly.respond(odor)                     # more negative
    """

    def __init__(
        self,
        mb: MushroomBody,
        params: LearningParams | None = None,
        state: TrainedState | None = None,
    ) -> None:
        self.mb = mb
        self.params = params or LearningParams()

        pre_idx, post_idx, pre_body, post_body = mb.plastic_edges()
        self.pre_idx = pre_idx
        self.post_idx = post_idx
        self.pre_body = pre_body
        self.post_body = post_body
        self.n_mbon = mb.n("MBON")
        self.n_kc = mb.n("KC")

        # Anatomical weight per plastic synapse, rescaled so every MBON
        # integrates the same total drive. Without this the few MBONs with
        # tens of thousands of KC inputs swamp everything else.
        w = mb.W_kc_mbon[pre_idx, post_idx].astype(np.float64)
        total_in = np.bincount(post_idx, weights=np.abs(w), minlength=self.n_mbon)
        scale = np.divide(
            self.params.mbon_input_budget, total_in,
            out=np.ones_like(total_in), where=total_in > 0,
        )
        self.w_anat = (w * scale[post_idx]).astype(np.float32)

        # PN -> KC drive, likewise normalised per Kenyon cell.
        pn_kc = mb.W_pn_kc.astype(np.float64)
        kc_in = pn_kc.sum(axis=0)
        kc_scale = np.divide(
            1.0, kc_in, out=np.zeros_like(kc_in), where=kc_in > 0
        )
        self.w_pn_kc = (pn_kc * kc_scale[np.newaxis, :]).astype(np.float32)

        # DAN -> MBON overlap, read as compartment membership.
        dan_mbon = mb.W_dan_mbon.astype(np.float64)
        da_in = dan_mbon.sum(axis=0)
        da_scale = np.divide(
            self.params.dan_input_budget, da_in,
            out=np.zeros_like(da_in), where=da_in > 0,
        )
        self.w_dan_mbon = (dan_mbon * da_scale[np.newaxis, :]).astype(np.float32)

        self.state = state or self.fresh_state()
        self._trace = np.zeros(self.n_kc, dtype=np.float32)

    # -- initial state ----------------------------------------------------

    def fresh_state(self) -> TrainedState:
        """An untrained fly: every synapse at full anatomical strength."""
        return TrainedState(
            gain=np.ones(len(self.pre_idx), dtype=np.float32),
            readout=self.default_readout(),
            trials=0,
        )

    def default_readout(self) -> np.ndarray:
        """Map MBON rates to a behavioural valence.

        An MBON's behavioural sign opposes its compartment's dopamine: the
        compartments innervated by punishment DANs (PPL1) carry approach
        drive, and the ones innervated by reward DANs (PAM) carry avoidance.
        That opposition is what makes a depression-only rule useful in both
        directions -- shock depresses approach, sugar depresses avoidance
        (Aso et al. 2014).

        So the readout is a compartment's punishment share minus its reward
        share, which the connectome does supply. This is inferred structure,
        not measurement: real MBON valences are established one compartment at
        a time by behavioural experiment. It is saved with the checkpoint so
        it can be swapped for measured values without retraining synapses.
        """
        punish = self.dopamine(self.dan_activity(punishment=1.0))
        reward = self.dopamine(self.dan_activity(reward=1.0))
        readout = (punish - reward).astype(np.float32)
        scale = np.abs(readout).sum()
        return readout / scale if scale > 0 else readout

    def reset_state(self) -> None:
        self.state = self.fresh_state()
        self._trace[:] = 0.0

    # -- forward ----------------------------------------------------------

    def kenyon_cells(self, pn: np.ndarray) -> np.ndarray:
        """Sparse Kenyon cell code for a PN activity vector.

        APL feedback inhibition is modelled as a k-winners-take-all: whatever
        threshold leaves `kc_sparsity` of the population above it. That is
        what APL achieves and not how it achieves it -- the real cell is a
        graded, delayed, distributed inhibition loop.
        """
        drive = (pn.astype(np.float32) @ self.w_pn_kc) * self.params.kc_gain
        k = max(1, int(round(self.params.kc_sparsity * self.n_kc)))
        if k >= self.n_kc:
            return np.maximum(drive, 0.0)

        cut = np.partition(drive, -k)[-k]
        out = np.where(drive >= cut, drive - cut, 0.0).astype(np.float32)
        peak = out.max()
        return out / peak if peak > 0 else out

    def mbon_rates(self, kc: np.ndarray) -> np.ndarray:
        """MBON output, with learning applied."""
        contrib = kc[self.pre_idx] * self.w_anat * self.state.gain
        return np.bincount(
            self.post_idx, weights=contrib, minlength=self.n_mbon
        ).astype(np.float32)

    def dopamine(self, dan: np.ndarray) -> np.ndarray:
        """Dopamine level per MBON compartment, from DAN activity."""
        return np.maximum(dan.astype(np.float32), 0.0) @ self.w_dan_mbon

    def valence(self, mbon: np.ndarray) -> float:
        """Scalar approach(+)/avoid(-) drive. Positive means approach."""
        return float(mbon @ self.state.readout)

    def respond(self, pn: np.ndarray) -> float:
        """Present an odour, read the behavioural valence. No learning."""
        return self.valence(self.mbon_rates(self.kenyon_cells(pn)))

    # -- learning ---------------------------------------------------------

    def dan_activity(
        self, *, reward: float = 0.0, punishment: float = 0.0
    ) -> np.ndarray:
        """Turn a scalar reinforcement into activity across the DAN population.

        Reward recruits PAM neurons, punishment recruits PPL1. Both are
        positive activity -- dopamine depresses either way, and the sign of
        the behavioural outcome comes from which compartments they sit in.
        """
        valence = self.mb.dan_valence
        act = np.zeros(len(valence), dtype=np.float32)
        act[valence > 0] = reward
        act[valence < 0] = punishment
        return np.maximum(act, 0.0)

    def learn(
        self,
        pn: np.ndarray,
        *,
        reward: float = 0.0,
        punishment: float = 0.0,
        update: bool = True,
    ) -> dict[str, float]:
        """One conditioning trial: present an odour with reinforcement.

        Returns the trial's observables so a training loop can report without
        re-running the forward pass.
        """
        p = self.params
        kc = self.kenyon_cells(pn)

        # Eligibility: KC activity persists past odour offset, so dopamine
        # arriving a few seconds later still finds the right synapses tagged.
        decay = float(np.exp(-1.0 / max(p.trace_tau, 1e-6)))
        self._trace = self._trace * decay + kc

        mbon = self.mbon_rates(kc)
        out = {
            "valence": self.valence(mbon),
            "mbon_total": float(mbon.sum()),
            "kc_active": int((kc > 0).sum()),
        }
        if not update:
            return out

        da = self.dopamine(self.dan_activity(reward=reward, punishment=punishment))

        # The rule. Multiplicative, so a synapse approaches the floor
        # asymptotically instead of crossing it.
        coincidence = self._trace[self.pre_idx] * da[self.post_idx]
        self.state.gain -= p.learning_rate * coincidence * self.state.gain

        # Forgetting: untouched synapses drift back toward baseline.
        self.state.gain += p.recovery * (1.0 - self.state.gain)
        np.clip(self.state.gain, p.gain_floor, 1.0, out=self.state.gain)

        self.state.trials += 1
        out["dopamine"] = float(da.sum())
        out["mean_gain"] = float(self.state.gain.mean())
        return out

    # -- diagnostics ------------------------------------------------------

    def depressed_fraction(self, threshold: float = 0.9) -> float:
        """Share of plastic synapses meaningfully below baseline."""
        return float((self.state.gain < threshold).mean())

    def compartment_gains(self) -> dict[str, float]:
        """Mean gain per MBON type -- which compartments hold the memory."""
        out: dict[str, list[float]] = {}
        totals = np.bincount(
            self.post_idx, weights=self.state.gain, minlength=self.n_mbon
        )
        counts = np.bincount(self.post_idx, minlength=self.n_mbon)
        mean = np.divide(totals, counts, out=np.ones_like(totals), where=counts > 0)
        for t, m, c in zip(self.mb.types["MBON"], mean, counts):
            if c:
                out.setdefault(str(t), []).append(float(m))
        return {k: float(np.mean(v)) for k, v in sorted(out.items())}

    def with_params(self, **kwargs) -> "FlyBrain":
        """A copy of this fly with some parameters changed."""
        return FlyBrain(self.mb, replace(self.params, **kwargs), self.state.copy())
