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
        n_actions: int = 1,
    ) -> None:
        self.mb = mb
        self.params = params or LearningParams()
        # 1 keeps the scalar approach/avoid readout every other caller uses.
        # More than 1 gives separate behavioural channels -- see
        # `action_readout` for why they must read disjoint MBONs.
        self.n_actions = int(n_actions)

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
        n = getattr(self, "n_actions", 1)
        readout = (self.default_readout() if n == 1
                   else self.action_readout(n))
        return TrainedState(
            gain=np.ones(len(self.pre_idx), dtype=np.float32),
            readout=readout,
            trials=0,
        )

    def action_readout(self, n_actions: int) -> np.ndarray:
        """One readout column per action, over disjoint MBON groups.

        The scalar `default_readout` gives approach/avoid, which is a two-way
        choice only through its sign. Real behavioural channels are separate
        populations: different MBONs drive different actions. So each action
        gets its own slice of the MBON population, and its drive is that
        slice's rates against the default valence weighting.

        Disjoint is what makes the channels independently trainable -- the
        plastic gains are shared, so overlapping readouts would move together
        and the fly could never prefer one action over another.

        Which MBON drives which action is not in the connectome. Splitting the
        population in order is a placeholder for a behavioural mapping that
        would have to come from experiment.
        """
        base = self.default_readout()
        out = np.zeros((len(base), n_actions), dtype=np.float32)
        groups = np.array_split(np.arange(len(base)), n_actions)
        for k, idx in enumerate(groups):
            out[idx, k] = base[idx]
        return out

    def drives(self, pn: np.ndarray) -> np.ndarray:
        """Per-action drive for a stimulus. Length `n_actions`."""
        mbon = self.mbon_rates(self.kenyon_cells(pn))
        r = self.state.readout
        return (mbon @ r) if r.ndim == 2 else np.array([self.valence(mbon)],
                                                       dtype=np.float32)

    def choose(self, pn: np.ndarray, *, explore: float = 0.0, rng=None,
               tail: float = 3.0) -> tuple[int, np.ndarray]:
        """Pick an action. Returns (index, the drives it chose between).

        Exploration is heavy-tailed and scaled to the spread of the drives, so
        it stays meaningful whatever magnitude the readout happens to produce.
        """
        d = self.drives(pn)
        if explore > 0.0 and len(d) > 1:
            rng = rng if rng is not None else np.random.default_rng()
            scale = np.sqrt(tail / (tail - 2.0)) if tail > 2.0 else 1.0
            spread = float(np.ptp(d)) or float(np.abs(d).mean()) or 1e-6
            d = d + rng.standard_t(tail, size=len(d)) / scale * explore * spread
        return int(np.argmax(d)), d

    def reinforce(self, pn: np.ndarray, action: int, outcome: float,
                  *, strength: float = 1.0) -> dict[str, float]:
        """Reward or punish the action that was actually taken.

        Dopamine is not action-specific -- it floods a compartment, and every
        MBON reading that compartment is affected. What makes this train a
        *choice* is that each action reads a disjoint MBON group, so the same
        dopamine moves the chosen channel's drive without dragging the others
        with it.
        """
        magnitude = min(abs(float(outcome)), 1.0) * strength
        if magnitude <= 0.0:
            return self.learn(pn, update=False)
        r = self.state.readout
        mask = None
        if r.ndim == 2:
            mask = (r[:, int(action)] != 0).astype(np.float32)
        if outcome > 0:
            return self.learn(pn, reward=magnitude, mbon_mask=mask)
        return self.learn(pn, punishment=magnitude, mbon_mask=mask)

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
        """Scalar approach(+)/avoid(-) drive. Positive means approach.

        With several action channels there is no single valence, so this
        reports the net across them -- enough for logging and for the
        diagnostics `learn` returns, but `drives` is what selects an action.
        """
        r = self.state.readout
        if r.ndim == 2:
            return float((mbon @ r).sum())
        return float(mbon @ r)

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
        mbon_mask: np.ndarray | None = None,
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
        # Dopamine is compartment-specific in the animal: a DAN floods the
        # compartment it innervates and no other. `mbon_mask` restricts it
        # further, to the compartments read by one behavioural channel, which
        # is what makes an update depend on the action that was taken. Without
        # it every action produces the same synaptic change and no choice can
        # ever be learned.
        if mbon_mask is not None:
            da = da * np.asarray(mbon_mask, dtype=np.float32)

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

    def act(self, pn: np.ndarray, *, explore: float = 0.0,
            rng=None, tail: float = 3.0) -> float:
        """Choose an action from a stimulus. Returns a signed drive.

        The readout is deterministic -- a fly's behavioural variability comes
        from premotor circuits, not from reading its memories badly -- so the
        variability is added here, at action selection.

        `explore` is in units of the valence itself, so it scales with
        whatever the readout happens to produce. Heavy-tailed for the reason
        in `olfactory_brain.BrainNavParams`: occasional large excursions
        explore far better than jitter of the same size.
        """
        valence = self.respond(pn)
        if explore <= 0.0:
            return valence
        rng = rng if rng is not None else np.random.default_rng()
        scale = np.sqrt(tail / (tail - 2.0)) if tail > 2.0 else 1.0
        kick = float(rng.standard_t(tail) / scale)
        return valence + kick * explore * max(abs(valence), 1e-6)

    def learn_operant(
        self,
        pn: np.ndarray,
        action: float,
        outcome: float,
        *,
        strength: float = 1.0,
    ) -> dict[str, float]:
        """Reinforce the action the fly actually took.

        `learn` is classical: it tags whatever the fly was *seeing* when
        reinforcement arrived. That is right for "this odour predicts sugar"
        and wrong the moment behaviour is what earned the outcome, because
        exploration can make the action disagree with the valence that
        produced it. Reward such a trial as if the valence had chosen it and
        you reinforce the direction the fly did *not* take -- exploration then
        actively teaches the wrong thing.

        Binding credit to the action is one sign:

            action > 0, outcome > 0  -> push valence up   (reward, PAM)
            action > 0, outcome < 0  -> push valence down (punish, PPL1)
            action < 0, outcome > 0  -> push valence down (punish)
            action < 0, outcome < 0  -> push valence up   (reward)

        so the reinforcement is `sign(action * outcome)`, and depression-only
        plasticity still moves it both ways because reward and punishment
        depress opposite compartments.

        `action` is the signed drive actually executed, `outcome` is positive
        for success. Magnitudes scale the update, so a near-miss teaches less
        than a clean hit.
        """
        drive = float(action) * float(outcome)
        magnitude = min(abs(drive), 1.0) * strength
        if magnitude <= 0.0:
            return self.learn(pn, update=False)
        if drive > 0:
            return self.learn(pn, reward=magnitude)
        return self.learn(pn, punishment=magnitude)

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
