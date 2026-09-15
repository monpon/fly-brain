"""Many flies at once, on the GPU.

`learning.py` simulates one fly in numpy, and one fly is far too little work
to fill a graphics card: the PN->KC matmul is 2.2 MFLOP, which an RTX 3050
would finish in less time than it takes to launch the kernel. Measured, a
single fly on the GPU is 2.2x -- not worth the rewrite.

Flies are independent, though, so the batch dimension is free parallelism.
The same benchmark at batch 256:

    batch     ms/frame   ms per fly per frame   vs numpy
        1        0.521                 0.5214       2.2x
       16        1.073                 0.0670      16.7x
       64        3.238                 0.0506      22.2x
      256       11.451                 0.0447      25.1x

Per-frame cost rises 22x for 256x the work, which is the shape you get when
the card was idle to begin with.

The speed is only half the reason. Every confidence interval in
docs/FINDINGS.md so far came from a *single seed*, which is why 28.8% against
26.7% has been unresolvable no matter how many balls were run. Two hundred
and fifty-six flies is 256 seeds, and the error bars finally mean something.

What this is not
----------------
Not a replacement for `learning.py`. That stays the reference implementation:
it is readable, it has no dependency on torch, and `check_parity` exists to
prove this file still agrees with it. When the two disagree, the numpy one is
right until shown otherwise.
"""

from __future__ import annotations

import numpy as np

from ._deps import require
from .learning import FlyBrain, LearningParams
from .mushroom_body import MushroomBody


def pick_device(requested: str = "auto") -> str:
    torch = require("torch")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


class BatchedBrain:
    """`n_flies` independent mushroom bodies sharing one anatomy.

    The connectome is identical across the batch -- same neurons, same
    synapses, same normalisation -- so the weights are stored once and the
    per-fly state is what carries the batch dimension:

        gain    (N, n_synapses)   what each fly has learned
        readout (N, n_mbon, n_actions)
        trace   (N, n_kc)         eligibility

    At 256 flies the gains are 63 MB, which is the only allocation big enough
    to think about.
    """

    def __init__(self, mb: MushroomBody, n_flies: int, *, n_actions: int = 3,
                 params: LearningParams | None = None, device: str = "auto",
                 seed: int = 0) -> None:
        self.torch = require("torch")
        torch = self.torch
        self.params = params or LearningParams()
        self.n_flies = int(n_flies)
        self.n_actions = int(n_actions)
        self.device = pick_device(device)

        # Build one CPU fly and borrow its arrays. Every normalisation
        # decision -- per-MBON input budget, per-KC drive scaling, the signed
        # and centred readout -- then lives in exactly one place, and this
        # class cannot quietly drift away from the reference.
        ref = FlyBrain(mb, self.params, n_actions=n_actions)
        self.ref = ref
        self.mb = mb
        self.n_kc = ref.n_kc
        self.n_mbon = ref.n_mbon
        self.n_syn = len(ref.pre_idx)
        self.k = max(1, int(round(self.params.kc_sparsity * self.n_kc)))

        d = self.device
        t = lambda a, dt=torch.float32: torch.tensor(np.asarray(a), device=d,
                                                     dtype=dt)
        self.w_pn_kc = t(ref.w_pn_kc)
        self.w_anat = t(ref.w_anat)
        self.w_dan_mbon = t(ref.w_dan_mbon)
        self.pre = t(ref.pre_idx, torch.long)
        self.post = t(ref.post_idx, torch.long)
        self.dan_valence = t(mb.dan_valence)
        self.lateral_w = (t(ref.w_mbon_mbon)
                          if ref.w_mbon_mbon is not None else None)

        N = self.n_flies
        self.gain = torch.ones(N, self.n_syn, device=d)
        self.trace = torch.zeros(N, self.n_kc, device=d)
        self.mbon_trace = torch.zeros(N, self.n_mbon, device=d)
        self.readout = t(ref.state.readout).unsqueeze(0).repeat(N, 1, 1)
        self.readout_norm = self.readout.abs().sum(dim=1)     # (N, n_actions)
        self.trials = torch.zeros(N, device=d, dtype=torch.long)

        self.gen = torch.Generator(device=d)
        self.gen.manual_seed(int(seed))

    # -- forward ----------------------------------------------------------

    def kenyon_cells(self, pn):
        """Sparse code, k-winners-take-all standing in for APL feedback."""
        torch = self.torch
        drive = pn @ self.w_pn_kc
        cut = torch.topk(drive, self.k, dim=1).values[:, -1:]
        out = torch.clamp(drive - cut, min=0.0)
        peak = out.max(dim=1, keepdim=True).values
        return torch.where(peak > 0, out / peak.clamp(min=1e-12), out)

    def mbon_rates(self, kc):
        """KC drive into MBONs, then lateral interaction if enabled."""
        torch = self.torch
        contrib = kc[:, self.pre] * self.w_anat * self.gain
        raw = torch.zeros(kc.shape[0], self.n_mbon, device=self.device)
        raw.index_add_(1, self.post, contrib)
        if self.lateral_w is None or self.params.lateral <= 0.0:
            return raw
        rate = raw
        for _ in range(self.params.lateral_steps):
            rate = torch.clamp(raw + self.params.lateral
                               * (rate @ self.lateral_w), min=0.0)
        return rate

    def drives(self, mbon):
        """Per-action drive. (N, n_mbon) x (N, n_mbon, n_actions)."""
        return self.torch.einsum("nm,nma->na", mbon, self.readout)

    def observe(self, pn):
        """One frame of sensing. Updates eligibility, returns the drives."""
        p = self.params
        kc = self.kenyon_cells(pn)
        decay = float(np.exp(-1.0 / max(p.trace_tau, 1e-6)))
        self.trace = (self.trace * decay + kc * (1.0 - decay)
                      if p.trace_normalize else self.trace * decay + kc)
        mbon = self.mbon_rates(kc)
        if p.readout_rate > 0.0:
            r_decay = float(np.exp(-1.0 / max(p.readout_trace_tau
                                              or p.trace_tau, 1e-6)))
            self.mbon_trace = (self.mbon_trace * r_decay
                               + mbon * (1.0 - r_decay))
        return kc, mbon, self.drives(mbon)

    def choose(self, drives, explore: float, tail: float = 3.0):
        """Argmax with heavy-tailed exploration. Returns (N,) actions."""
        torch = self.torch
        if explore <= 0.0:
            return drives.argmax(dim=1), drives
        # StudentT for the same reason the scalar version uses it: occasional
        # large excursions explore better than jitter of the same variance.
        chi = torch.distributions.Chi2(torch.tensor(float(tail), device=self.device))
        z = torch.randn(drives.shape, generator=self.gen, device=self.device)
        u = chi.sample(drives.shape).clamp(min=1e-6)
        kick = z / torch.sqrt(u / tail)
        scale = np.sqrt(tail / (tail - 2.0)) if tail > 2.0 else 1.0
        spread = (drives.max(dim=1, keepdim=True).values
                  - drives.min(dim=1, keepdim=True).values)
        spread = torch.where(spread > 0, spread,
                             drives.abs().mean(dim=1, keepdim=True)
                             .clamp(min=1e-6))
        return (drives + kick / scale * explore * spread).argmax(dim=1), drives

    # -- plasticity -------------------------------------------------------

    def _dopamine(self, reward, punishment):
        """(N,) scalars -> (N, n_mbon) compartment dopamine."""
        torch = self.torch
        act = torch.zeros(self.n_flies, len(self.dan_valence),
                          device=self.device)
        pos, neg = self.dan_valence > 0, self.dan_valence < 0
        act[:, pos] = reward.unsqueeze(1)
        act[:, neg] = punishment.unsqueeze(1)
        return torch.clamp(act, min=0.0) @ self.w_dan_mbon

    def reinforce(self, actions, outcomes, active):
        """Reward or punish, per fly. `active` selects which flies resolved.

        Everything is computed for the whole batch and then multiplied by the
        `active` mask rather than indexed. Masking keeps every tensor the same
        shape on every frame, which avoids re-allocating and avoids a
        host-device sync to find out which flies are involved -- and that sync
        is exactly what would give back the speedup this file exists for.
        """
        torch = self.torch
        p = self.params
        act = active.float().unsqueeze(1)                      # (N, 1)

        mag = outcomes.abs().clamp(max=1.0) * active.float()
        reward = torch.where(outcomes > 0, mag, torch.zeros_like(mag))
        punish = torch.where(outcomes <= 0, mag, torch.zeros_like(mag))

        # Dopamine, restricted to the compartments the chosen action reads.
        column = torch.gather(
            self.readout, 2, actions.view(-1, 1, 1).expand(-1, self.n_mbon, 1)
        ).squeeze(2)                                           # (N, n_mbon)
        if p.readout_rate > 0.0:
            peak = column.abs().max(dim=1, keepdim=True).values.clamp(min=1e-12)
            mask = column.abs() / peak
        else:
            mask = (column != 0).float()

        da = self._dopamine(reward, punish) * mask * act
        self.gain = self.gain - (p.learning_rate
                                 * self.trace[:, self.pre]
                                 * da[:, self.post] * self.gain)

        if p.bidirectional:
            opp = self._dopamine(punish, reward) * mask * act
            rate = p.potentiation_rate or p.learning_rate
            self.gain = self.gain + (rate * self.trace[:, self.pre]
                                     * opp[:, self.post]
                                     * (p.gain_ceiling - self.gain))

        self.gain = self.gain + p.recovery * (1.0 - self.gain) * act
        self.gain = self.gain.clamp(p.gain_floor, max(p.gain_ceiling, 1.0))

        if p.readout_rate > 0.0:
            self._learn_readout(actions, outcomes, mag, active)

        self.trials = self.trials + active.long()

    def _learn_readout(self, actions, outcomes, mag, active):
        """Reward-modulated Hebbian update on the output map, per fly."""
        torch = self.torch
        sign = torch.where(outcomes > 0, torch.ones_like(mag),
                           -torch.ones_like(mag)) * mag
        idx = actions.view(-1, 1, 1).expand(-1, self.n_mbon, 1)
        column = torch.gather(self.readout, 2, idx).squeeze(2)
        column = column + (self.params.readout_rate
                           * sign.unsqueeze(1) * self.mbon_trace)
        column = column - column.mean(dim=1, keepdim=True)
        total = column.abs().sum(dim=1, keepdim=True).clamp(min=1e-12)
        target = torch.gather(self.readout_norm, 1,
                              actions.view(-1, 1))                # (N, 1)
        column = column * (target / total)
        keep = active.float().unsqueeze(1)
        merged = torch.gather(self.readout, 2, idx).squeeze(2)
        self.readout.scatter_(
            2, idx, (keep * column + (1 - keep) * merged).unsqueeze(2))

    def reset_trace(self, active=None) -> None:
        """Clear eligibility, for the flies whose episode just ended."""
        if active is None:
            self.trace.zero_()
            self.mbon_trace.zero_()
            return
        keep = (~active).float().unsqueeze(1)
        self.trace = self.trace * keep
        self.mbon_trace = self.mbon_trace * keep

    # -- diagnostics ------------------------------------------------------

    def depressed_fraction(self, threshold: float = 0.9):
        return (self.gain < threshold).float().mean(dim=1)

    def numpy_gain(self, fly: int = 0) -> np.ndarray:
        return self.gain[fly].detach().cpu().numpy()


def check_parity(mb: MushroomBody, *, trials: int = 12, seed: int = 0,
                 params: LearningParams | None = None) -> dict:
    """Does the batched backend still agree with the numpy reference?

    Runs the same stimuli through both with exploration off, and compares the
    drives and the learned gains. The batched path has to reproduce the one
    that every published number came from, or the speedup is worthless.
    """
    params = params or LearningParams()
    rng = np.random.default_rng(seed)
    n_pn = mb.n("PN")
    stimuli = []
    for _ in range(trials):
        pn = np.zeros(n_pn, dtype=np.float32)
        pn[rng.choice(n_pn, 40, replace=False)] = 30.0
        stimuli.append(pn)

    cpu = FlyBrain(mb, params, n_actions=3)
    gpu = BatchedBrain(mb, 4, n_actions=3, params=params, seed=seed)
    torch = gpu.torch

    drive_err = 0.0
    for i, pn in enumerate(stimuli):
        mbon_c = cpu.mbon_rates(cpu.kenyon_cells(pn))
        d_c = mbon_c @ cpu.state.readout

        batch = torch.tensor(np.tile(pn, (4, 1)), device=gpu.device)
        _, _, d_g = gpu.observe(batch)
        d_g = d_g[0].detach().cpu().numpy()
        drive_err = max(drive_err, float(np.abs(d_c - d_g).max()))

        action = int(np.argmax(d_c))
        outcome = 1.0 if i % 2 == 0 else -1.0
        # One trace update per trial on each side. `cpu.reinforce` calls
        # `learn`, which advances the trace itself, so an explicit
        # `learn(update=False)` here would advance it twice and the two
        # backends would disagree for a reason that is not a real difference.
        cpu.reinforce(pn, action, outcome)
        cpu.reset_trace()

        gpu.reinforce(
            torch.full((4,), action, device=gpu.device, dtype=torch.long),
            torch.full((4,), outcome, device=gpu.device),
            torch.ones(4, device=gpu.device, dtype=torch.bool),
        )
        gpu.reset_trace()

    gain_err = float(np.abs(cpu.state.gain - gpu.numpy_gain(0)).max())
    return {
        "device": gpu.device,
        "max_drive_error": drive_err,
        "max_gain_error": gain_err,
        "cpu_mean_gain": float(cpu.state.gain.mean()),
        "gpu_mean_gain": float(gpu.numpy_gain(0).mean()),
    }
