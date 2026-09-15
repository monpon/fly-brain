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

    # The eligibility trace is a running *sum* of KC activity, so its steady
    # state scales with `trace_tau`: doubling the time constant roughly
    # doubles the trace, and therefore the size of every depression step. The
    # time constant and the learning rate are not independent knobs, which
    # makes "lengthen the trace" impossible to test in isolation -- raise
    # trace_tau far enough and every synapse hits gain_floor on the first
    # reinforced trial, which reads as "a longer trace is worse" when what
    # actually happened is a 50x learning rate.
    #
    # Normalising makes it a leaky average instead, so trace_tau changes only
    # how far back the trace reaches. Off by default because every measured
    # result in docs/FINDINGS.md was produced with the sum, and those numbers
    # should keep meaning what they say.
    trace_normalize: bool = False

    # -- lateral interaction between MBONs --------------------------------
    # Strength of MBON -> MBON feedback, and how many damped iterations to
    # settle it. 0 reproduces the purely feedforward model exactly.
    lateral: float = 0.0
    lateral_steps: int = 3

    # -- bidirectional plasticity -----------------------------------------
    # The rule is depression-only by default: gains start at 1.0, dopamine
    # drives them down, and `gain_floor` stops them at 0.05. That gives the
    # animal a *finite plastic budget* -- it can only ever subtract, and in
    # the pong drill the depressed fraction climbs past 35% and keeps going,
    # which is the budget visibly running out.
    #
    # `bidirectional` makes each outcome depress its own compartments and
    # potentiate the *opposite valence's*, both against the same eligibility
    # trace. Reward therefore restores what punishment previously spent, and
    # the budget stops being one-way.
    #
    # Getting this wrong is instructive and was measured. The first attempt
    # potentiated the *same* compartments the same dopamine had just
    # depressed, which is not a second direction of learning -- it is an
    # eraser. The depressed fraction fell from 32.7% to 0.9% and the drill
    # dropped from 28.8% to 22.0%, barely above the geometric floor. A rule
    # that writes and unwrites the same synapse in one step holds nothing.
    bidirectional: bool = False
    potentiation_rate: float = 0.0     # 0 means "same as learning_rate"
    gain_ceiling: float = 1.0          # raise above 1.0 to allow strengthening

    # The order-dependent rule, kept separate because it is real biology that
    # this particular task cannot use. In the animal, dopamine *followed by*
    # Kenyon cell activity potentiates -- which is how an odour arriving after
    # shock offset becomes attractive (relief learning). Here the only
    # stimulus after an outcome is the next ball, which has nothing to do with
    # what earned it, so the pairing writes noise. Worth having for
    # conditioning protocols where something meaningful follows the
    # reinforcement.
    reverse_pairing: bool = False
    da_trace_tau: float = 8.0          # frames dopamine stays available

    # -- a learnable output map -------------------------------------------
    # Which MBON drives which action is not in the connectome, so the model
    # assigns it: `np.array_split(arange(97), n_actions)`, contiguous groups
    # in extraction order. That is not a behavioural mapping, it is an array
    # index, and it is incoherent -- 17 MBON cell types end up split across
    # more than one channel, so biologically identical neurons are assigned
    # to opposing commands.
    #
    # Until now the fly could not do anything about that. `learn` writes only
    # `state.gain`; the readout is built once and never touched, so the animal
    # can change which Kenyon cells drive which MBONs but not which MBONs mean
    # "up". It has to route information *into* a scrambled output map instead
    # of fixing the map.
    #
    # `readout_rate` makes that map plastic under the same three-factor rule
    # as everything else: MBON active, action taken, outcome good -> that MBON
    # drives that action more. Reward-modulated Hebbian learning at the output
    # synapse, which is ordinary biology. 0 keeps the frozen map, so the
    # ablation is a flag away.
    readout_rate: float = 0.0
    readout_trace_tau: float = 0.0     # 0 follows trace_tau

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

        # MBON -> MBON, signed by the *presynaptic* MBON's transmitter, which
        # is how real neurons work: a cell releases the same thing at all its
        # terminals. Normalised per postsynaptic cell so the recurrent input
        # is comparable to the feedforward drive rather than swamping it.
        self.w_mbon_mbon = self._recurrent(getattr(mb, "W_mbon_mbon", None))

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
        # Lingering dopamine per compartment, for the reverse pairing.
        self._da_trace = np.zeros(self.n_mbon, dtype=np.float32)
        # MBON activity eligibility, for learning the output map. The rule
        # needs to know which output cells were active when the action was
        # chosen, which is a different quantity from which Kenyon cells were.
        self._mbon_trace = np.zeros(self.n_mbon, dtype=np.float32)
        # Each channel's initial total drive, so learning can redistribute
        # weight within a channel without changing how loud that channel is
        # relative to the others. Channels that drift apart in scale are the
        # bug that parked the paddle against a wall.
        r = self.state.readout
        self._readout_norm = (np.abs(r).sum(axis=0) if r.ndim == 2
                              else np.array([np.abs(r).sum()]))

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
            column = base[idx]
            # Centre it. `base` is punish-minus-reward and carries both signs,
            # so an arbitrary contiguous slice of it has an arbitrary net sum
            # -- and that sum is a constant offset on the channel's drive.
            # Uncentred, the channel with the largest offset wins `argmax`
            # almost regardless of the stimulus: measured 96.5% of decisions
            # going to one action and 0% to another, with agreement against a
            # tracking policy at 2.2% where chance is 33.3%. The policy was
            # not noisy, it was constant, and no amount of plasticity fixed it
            # because depression moves gains by a few percent while the offset
            # is structural.
            #
            # Centred, a uniform MBON response scores zero on every channel,
            # so the comparison is about which *pattern* of MBON activity a
            # stimulus produces, which is the thing that carries information.
            out[idx, k] = column - column.mean()
        return out

    def drives(self, pn: np.ndarray) -> np.ndarray:
        """Per-action drive for a stimulus. Length `n_actions`."""
        mbon = self.mbon_rates(self.kenyon_cells(pn))
        r = self.state.readout
        return (mbon @ r) if r.ndim == 2 else np.array([self.valence(mbon)],
                                                       dtype=np.float32)

    def choose(self, pn: np.ndarray, *, explore: float = 0.0, rng=None,
               tail: float = 3.0, absolute: bool = False
               ) -> tuple[int, np.ndarray]:
        """Pick an action. Returns (index, the drives it chose between).

        Exploration is heavy-tailed. By default it is scaled to the spread of
        the drives, so it stays meaningful whatever magnitude the readout
        happens to produce -- but that convenience has a sting. Scaling the
        noise to the signal pins the signal-to-noise ratio at about
        `1/explore` *forever*: as learning separates the drives the noise
        grows in exact proportion, and the animal can never become decisive
        no matter how much it has learned. At the pong default of explore=2.0
        the noise is twice the entire spread of the drives, which is to say
        the action is noise.

        `absolute=True` drops the scaling. Noise is then a fixed magnitude
        the learned drives can outgrow, so decisiveness becomes a consequence
        of having learned something rather than a parameter.
        """
        d = self.drives(pn)
        if explore <= 0.0 or len(d) < 2:
            return int(np.argmax(d)), d

        rng = rng if rng is not None else np.random.default_rng()
        scale = np.sqrt(tail / (tail - 2.0)) if tail > 2.0 else 1.0
        kick = rng.standard_t(tail, size=len(d)) / scale * explore
        if not absolute:
            kick = kick * (float(np.ptp(d)) or float(np.abs(d).mean()) or 1e-6)
        # The drives returned are the *clean* ones. The caller wants to know
        # what the policy thinks, not what the dice did to it -- otherwise
        # "did exploration override the policy" is unanswerable, because the
        # chosen action is the argmax of the noised vector by construction.
        return int(np.argmax(d + kick)), d

    def greedy(self, pn: np.ndarray) -> int:
        """The action the policy would take with no exploration at all."""
        return int(np.argmax(self.drives(pn)))

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
            column = r[:, int(action)]
            if self.params.readout_rate > 0.0:
                # A learned output map goes dense -- every MBON acquires some
                # weight on every action, which is the point, since that is
                # how the fly reassigns a cell from one command to another.
                # But the dopamine mask was binary, `readout != 0`, so the
                # moment the map densifies the mask covers all 97 compartments
                # and dopamine floods the lot. Measured: support went 33 -> 97
                # within 150 balls. Global dopamine makes the update identical
                # whatever the fly did, and no choice is learnable -- 50%
                # against 100% on the single-decision task.
                #
                # Graded instead of binary. A compartment receives dopamine in
                # proportion to how much it actually drives the chosen action,
                # which keeps the update action-specific however dense the map
                # becomes, and is closer to the biology than a hard set
                # membership was.
                peak = float(np.abs(column).max())
                mask = (np.abs(column) / peak).astype(np.float32) if peak > 0 \
                    else np.zeros_like(column, dtype=np.float32)
            else:
                mask = (column != 0).astype(np.float32)
        if outcome > 0:
            result = self.learn(pn, reward=magnitude, mbon_mask=mask)
        else:
            result = self.learn(pn, punishment=magnitude, mbon_mask=mask)

        if self.params.readout_rate > 0.0:
            self.learn_readout(int(action), 1.0 if outcome > 0 else -1.0,
                               strength=magnitude)
        return result

    def learn_readout(self, action: int, outcome: float, *,
                      strength: float = 1.0) -> None:
        """Teach the fly which output neurons to drive.

        Three factors, the same shape as the rule at KC->MBON: an MBON was
        active (`_mbon_trace`), an action was taken (`action`), and it went
        well or badly (`outcome`). Coincide them and that MBON's contribution
        to that action moves accordingly.

        Two constraints keep this from destroying the thing it is fixing.

        The column is re-centred after every update. A constant offset on a
        channel is precisely the bug that pinned the paddle to a wall for the
        whole of this project's history, and a freely drifting readout would
        reintroduce it within a few dozen trials.

        The column is renormalised to the total drive it started with, so
        learning redistributes weight *within* a channel rather than making
        one channel louder than the others. Without it the winning action
        wins harder every time it wins, which is a positive feedback loop and
        not a policy.
        """
        r = self.state.readout
        if r.ndim != 2 or not (0 <= action < r.shape[1]):
            return
        column = r[:, action].astype(np.float32)
        column = column + (self.params.readout_rate * outcome * strength
                           * self._mbon_trace)
        column -= column.mean()
        total = float(np.abs(column).sum())
        if total > 0:
            column *= float(self._readout_norm[action]) / total
        r[:, action] = column

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

    def _recurrent(self, W) -> np.ndarray | None:
        """Sign and normalise an MBON -> MBON matrix, or None if absent.

        Returns None rather than zeros when the matrix is missing, so a
        mushroom body cached before these pathways were extracted keeps the
        old feedforward behaviour instead of silently pretending the
        recurrence is there and empty.
        """
        if W is None or np.size(W) == 0 or W.shape != (self.n_mbon,
                                                       self.n_mbon):
            return None
        signs = np.asarray(self.mb.mbon_sign, dtype=np.float32)
        if signs.size != self.n_mbon:
            signs = np.ones(self.n_mbon, dtype=np.float32)

        W = np.asarray(W, dtype=np.float64) * signs[:, np.newaxis]
        np.fill_diagonal(W, 0.0)              # a cell does not inhibit itself
        total = np.abs(W).sum(axis=0)
        scale = np.divide(1.0, total, out=np.zeros_like(total),
                          where=total > 0)
        return (W * scale[np.newaxis, :]).astype(np.float32)

    def reset_state(self) -> None:
        self.state = self.fresh_state()
        self._trace[:] = 0.0
        self._mbon_trace[:] = 0.0
        self._da_trace[:] = 0.0

    def reset_trace(self) -> None:
        """Forget what is currently tagged, keeping everything learned.

        Eligibility is a claim about what just happened, so it should not
        survive into an unrelated episode. Without this, credit for one pong
        rally lands partly on the decisions of the one before it, and a
        single-trial association is contaminated by the trial before that.

        The *dopamine* trace is deliberately left alone. It is not a claim
        about the past, it is neuromodulator that has been released and has
        not yet cleared, and it does not know where an episode boundary is.
        Clearing it here would also make potentiation impossible in any task
        that reinforces at the end of an episode: the dopamine would be wiped
        in the same breath that deposited it, and the reverse pairing would
        never have any Kenyon cell activity to meet. It decays on its own,
        over `da_trace_tau` frames.
        """
        self._trace[:] = 0.0
        self._mbon_trace[:] = 0.0

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
        """MBON output, with learning applied, then lateral interaction.

        The feedforward part is the KC drive each MBON collects. What follows
        is 26,259 synapses of MBON -> MBON that the earlier model ignored.

        They matter for a specific reason. When MBONs are split into disjoint
        behavioural channels, each channel's drive floats on its own constant
        offset, and whichever offset is largest wins the choice regardless of
        the stimulus -- which is exactly the failure documented in
        docs/FINDINGS.md, where the paddle parked against a wall. Centring the
        readout by hand fixes it. Lateral inhibition fixes it the way the
        animal does: channels suppress one another, so what survives is the
        *contrast* between them rather than their absolute level.

        Solved by a few damped iterations rather than a matrix inverse. The
        loop is inhibition-dominated and converges quickly; the damping is
        there because a strongly recurrent inhibitory network with no time
        constant can oscillate between iterations.
        """
        contrib = kc[self.pre_idx] * self.w_anat * self.state.gain
        raw = np.bincount(
            self.post_idx, weights=contrib, minlength=self.n_mbon
        ).astype(np.float32)

        if self.w_mbon_mbon is None or self.params.lateral <= 0.0:
            return raw

        rate = raw
        for _ in range(self.params.lateral_steps):
            fed = rate @ self.w_mbon_mbon
            rate = np.maximum(raw + self.params.lateral * fed, 0.0)
        return rate.astype(np.float32)

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
        self._trace = (self._trace * decay + kc * (1.0 - decay)
                       if p.trace_normalize
                       else self._trace * decay + kc)

        mbon = self.mbon_rates(kc)
        if p.readout_rate > 0.0:
            # Same shape of eligibility as the Kenyon cells get, one layer
            # further out: which output cells were active when the choice was
            # made, still tagged when the outcome lands.
            r_decay = float(np.exp(-1.0 / max(p.readout_trace_tau
                                              or p.trace_tau, 1e-6)))
            self._mbon_trace = self._mbon_trace * r_decay + mbon * (1 - r_decay)

        out = {
            "valence": self.valence(mbon),
            "mbon_total": float(mbon.sum()),
            "kc_active": int((kc > 0).sum()),
        }
        if p.reverse_pairing:
            # Kenyon cells active *now* against dopamine that arrived earlier
            # and has not yet decayed. Runs on every call, not only reinforced
            # ones: this direction needs the cells to fire *after* the
            # dopamine, so it belongs with the frames that follow an outcome.
            if self._da_trace.any():
                rate = p.potentiation_rate or p.learning_rate
                revived = kc[self.pre_idx] * self._da_trace[self.post_idx]
                headroom = p.gain_ceiling - self.state.gain
                self.state.gain += rate * revived * headroom
                np.clip(self.state.gain, p.gain_floor,
                        max(p.gain_ceiling, 1.0), out=self.state.gain)
            self._da_trace *= float(np.exp(-1.0 / max(p.da_trace_tau, 1e-6)))

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

        # Depression. Multiplicative, so a synapse approaches the floor
        # asymptotically instead of crossing it. `_trace` is Kenyon cell
        # activity that happened *before* this dopamine, which is the forward
        # pairing that depresses.
        coincidence = self._trace[self.pre_idx] * da[self.post_idx]
        self.state.gain -= p.learning_rate * coincidence * self.state.gain

        if p.bidirectional:
            # Opponent potentiation. The same eligibility trace -- the same
            # stimulus that just earned the outcome -- but paired with the
            # dopamine of the *opposite* valence, so a reward strengthens in
            # the punishment compartments and a punishment strengthens in the
            # reward ones.
            #
            # This is what makes the budget renewable instead of one-way.
            # Depression alone can only subtract, and the drill spends it:
            # 35% of synapses depressed and still climbing. Here a later
            # reward can lift back what an earlier punishment pushed down, so
            # the same synapse can be used again rather than being consumed.
            # Crucially it does not touch the compartments this outcome just
            # depressed, which is the difference between a second direction
            # of learning and an eraser.
            opponent = self.dopamine(self.dan_activity(
                reward=punishment, punishment=reward))
            if mbon_mask is not None:
                opponent = opponent * np.asarray(mbon_mask, dtype=np.float32)
            revived = self._trace[self.pre_idx] * opponent[self.post_idx]
            rate = p.potentiation_rate or p.learning_rate
            self.state.gain += rate * revived * (p.gain_ceiling
                                                 - self.state.gain)

        # Forgetting: untouched synapses drift back toward baseline.
        self.state.gain += p.recovery * (1.0 - self.state.gain)
        np.clip(self.state.gain, p.gain_floor, max(p.gain_ceiling, 1.0),
                out=self.state.gain)

        if p.reverse_pairing:
            # This event's dopamine joins the pool the next few frames of
            # Kenyon cell activity will be potentiated against.
            self._da_trace = self._da_trace + da

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
