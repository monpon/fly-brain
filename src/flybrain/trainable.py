"""A connectome circuit you can train on arbitrary input/output pairs.

Everything else in this project asks what the fly's wiring *does*. This asks
what it can be *taught* to do: give it input patterns and the outputs you want
back, and fit the synaptic strengths until it produces them.

What is fixed, and what is learned
----------------------------------
The connectome supplies the topology and the signs, and neither is touched by
training:

* **Which neurons connect to which.** No synapse is added or removed. If the
  EM volume shows no synapse from A to B, this network has none either, and
  gradient descent cannot invent one.
* **Excitatory or inhibitory.** Taken from the predicted neurotransmitter of
  the *presynaptic* cell, because that is how real neurons work -- a neuron
  releases the same transmitter at all its terminals. A cell cannot excite one
  target and inhibit another, and training cannot flip a sign.

What training changes is one non-negative **gain per synapse**, multiplying
the anatomical weight. A gain of 0.4 means "this synapse ended up at 40% of
the strength its synapse count implies". That is the same representation
`learning.py` uses for dopamine-gated plasticity, for the same reason: it
stays meaningful across a re-extraction of the circuit, so a trained state is
portable (see `checkpoint.py`).

This is a real constraint, not a decoration. An unconstrained network of the
same neuron count would be a plain recurrent net and would fit far more
tasks. What it buys is that whatever the trained circuit does, it does over
the fly's own wiring diagram, and you can ask afterwards which anatomically
identified synapses carry the solution.

The dynamics
------------
Leaky rate units, run for `steps` timesteps per trial:

    x <- x + (dt/tau) * (-x + W_eff^T r + b + input)
    r <- softplus(x)

Rates, not spikes. `network.py` has the Brian2 spiking version; it is the
right tool for "what does this circuit do to a pulse" and the wrong one for
"fit a thousand trials", which is what this file is for.

No claim is made that gradient descent is biological. It is not. Dopamine
does not compute a gradient, and `learning.py` implements what the fly
actually does. This is an engineering tool for asking what the wiring is
*capable* of, which is a different and also interesting question.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Circuit:
    """A subcircuit: who is wired to whom, with signs, plus its input/output.

    Stored as an edge list rather than a dense matrix -- the mushroom body
    alone is 4,161 neurons and 61,210 synapses, which is 0.35% dense. Dense
    would be 69 MB of mostly zeros and a matmul that is almost all waste.
    """

    body_ids: np.ndarray          # (n,) connectome body IDs, stable identity
    types: np.ndarray             # (n,) cell-type names
    pre: np.ndarray               # (e,) index of presynaptic neuron
    post: np.ndarray              # (e,) index of postsynaptic neuron
    weight: np.ndarray            # (e,) signed, normalised anatomical weight
    input_idx: np.ndarray         # neurons the stimulus is injected into
    output_idx: np.ndarray        # neurons the answer is read from
    dataset: str = ""
    raw_weight: np.ndarray = field(default=None, repr=False)

    @property
    def n(self) -> int:
        return len(self.body_ids)

    @property
    def n_synapses(self) -> int:
        return len(self.pre)

    def summary(self) -> str:
        return (f"{self.n} neurons, {self.n_synapses} synapses, "
                f"{len(self.input_idx)} in, {len(self.output_idx)} out")

    def type_counts(self, limit: int = 8) -> list[tuple[str, int]]:
        names, counts = np.unique(self.types.astype(str), return_counts=True)
        order = np.argsort(-counts)
        return [(str(names[i]), int(counts[i])) for i in order[:limit]]

    def fingerprint(self) -> str:
        """Stable hash of the topology, so a checkpoint can refuse a mismatch."""
        import hashlib

        pre_b = self.body_ids[self.pre]
        post_b = self.body_ids[self.post]
        order = np.lexsort((post_b, pre_b))
        buf = np.stack([pre_b[order], post_b[order]]).astype(np.int64).tobytes()
        return hashlib.sha256(buf).hexdigest()[:16]


def _normalise(pre, post, weight, n, budget: float = 1.0):
    """Scale each neuron's incoming weights to a fixed total.

    Raw synapse counts span three orders of magnitude, and without this the
    handful of neurons with tens of thousands of inputs saturate immediately
    while everything else sits silent -- the same problem documented for MBONs
    in docs/FINDINGS.md. Normalising costs the model the absolute scale of the
    anatomy and buys it a conditioned optimisation problem.
    """
    total = np.bincount(post, weights=np.abs(weight), minlength=n)
    scale = np.divide(budget, total, out=np.ones_like(total), where=total > 0)
    return (weight * scale[post]).astype(np.float32)


def build(types: list[str], inputs: list[str], outputs: list[str],
          *, budget: float = 1.0, regex: bool = True) -> Circuit:
    """Extract a circuit by cell type, and mark its input and output cells.

    `types` selects the neurons; `inputs` and `outputs` are patterns matched
    against those neurons' types. Input and output populations may overlap the
    hidden ones -- in the fly they often do, since most neurons are both.
    """
    import pandas as pd

    from . import local_data as L

    W, neurons = L.circuit_weight_matrix(types, regex=regex)
    body_col = L._body_column(neurons)
    type_col = L._type_column(neurons)
    body_ids = neurons[body_col].to_numpy(dtype=np.int64)
    type_names = neurons[type_col].astype(str).to_numpy()

    pre, post = np.nonzero(W)
    weight = W[pre, post].astype(np.float32)

    def match(patterns) -> np.ndarray:
        pat = "|".join(f"(?:{p})" for p in patterns)
        hit = pd.Series(type_names).str.fullmatch(pat, na=False).to_numpy()
        if not hit.any():
            raise SystemExit(
                f"nothing in this circuit matches {patterns}; "
                f"types present: {sorted(set(type_names))[:12]}"
            )
        return np.nonzero(hit)[0]

    return Circuit(
        body_ids=body_ids,
        types=type_names,
        pre=pre.astype(np.int64),
        post=post.astype(np.int64),
        weight=_normalise(pre, post, weight, len(body_ids), budget),
        raw_weight=weight,
        input_idx=match(inputs),
        output_idx=match(outputs),
        dataset=L.config.dataset_dir().name,
    )


def from_mushroom_body(mb, *, budget: float = 1.0) -> Circuit:
    """The olfactory learning pathway as a trainable circuit: PN -> KC -> MBON.

    Convenient because it is already cached (`scripts/fetch_mushroom_body.py`)
    and because it is the one circuit here whose *biological* learning rule is
    also implemented, in `learning.py`. Training the same wiring both ways --
    dopamine-gated depression against gradient descent -- is a comparison
    worth being able to make.
    """
    body_ids = np.concatenate([mb.body_ids["PN"], mb.body_ids["KC"],
                               mb.body_ids["MBON"]])
    types = np.concatenate([mb.types["PN"], mb.types["KC"], mb.types["MBON"]])
    n_pn, n_kc = mb.n("PN"), mb.n("KC")

    blocks = [(mb.W_pn_kc, 0, n_pn), (mb.W_kc_mbon, n_pn, n_pn + n_kc)]
    pre_list, post_list, w_list = [], [], []
    for W, row0, col0 in blocks:
        r, c = np.nonzero(W)
        pre_list.append(r + row0)
        post_list.append(c + col0)
        w_list.append(W[r, c])

    pre = np.concatenate(pre_list).astype(np.int64)
    post = np.concatenate(post_list).astype(np.int64)
    weight = np.concatenate(w_list).astype(np.float32)
    n = len(body_ids)

    return Circuit(
        body_ids=body_ids, types=types, pre=pre, post=post,
        weight=_normalise(pre, post, weight, n, budget), raw_weight=weight,
        input_idx=np.arange(n_pn),
        output_idx=np.arange(n_pn + n_kc, n),
        dataset=mb.dataset,
    )


def pick_device(requested: str) -> str:
    """Resolve 'auto' to cuda when it is actually usable, else cpu.

    The forward pass is a gather, a multiply and a scatter-add over the
    synapse list, which is memory-bandwidth bound rather than compute bound --
    exactly the shape of problem a GPU wins on. But only for large circuits:
    the mushroom body is small enough that kernel launch overhead eats the
    gain, so 'auto' is not automatically the fast choice.

    Memory is the limit, not speed: autograd keeps every timestep, so VRAM
    goes as steps x batch x synapses. Halve --batch before giving up.
    """
    import torch

    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info()
        print(f"device: cuda -- {name}, {free / 2**30:.1f} of "
              f"{total / 2**30:.1f} GiB free")
        return "cuda"
    if requested == "cuda":
        raise SystemExit(
            "--device cuda, but torch.cuda.is_available() is False.\n"
            f"  torch {torch.__version__}, built against CUDA "
            f"{torch.version.cuda}\n"
            "  A '+cpu' build has no CUDA at all -- reinstall from "
            "https://download.pytorch.org/whl/cu124\n"
            "  If the build is right, check `nvidia-smi` finds the driver."
        )
    print("device: cpu (no CUDA available)")
    return "cpu"


# -- the trainable network ------------------------------------------------


class BrainNet:
    """Leaky rate units on a fixed connectome, with one gain per synapse.

    >>> net = BrainNet(circuit)
    >>> net.fit(X, Y, epochs=200)
    >>> net.predict(X)

    `X` is (trials, n_inputs) activation injected into the input neurons;
    `Y` is (trials, n_outputs) the rates you want out of the output neurons.

    Memory scales as steps x batch x synapses, because autograd keeps every
    timestep. 12 steps x 32 trials x 82k synapses is about 126 MB; raising
    either past that is what will exhaust RAM first.
    """

    def __init__(self, circuit: Circuit, *, steps: int = 12, tau: float = 2.0,
                 dt: float = 1.0, seed: int = 0, device: str = "cpu",
                 rate: str = "softplus", rmax: float = 5.0):
        import torch

        self.circuit = circuit
        self.steps = steps
        self.tau = tau
        self.dt = dt
        self.rate_name = rate
        self.rmax = rmax
        self.device = torch.device(device)
        torch.manual_seed(seed)

        c = circuit
        self.pre = torch.as_tensor(c.pre, dtype=torch.long, device=self.device)
        self.post = torch.as_tensor(c.post, dtype=torch.long, device=self.device)
        self.w_anat = torch.as_tensor(c.weight, dtype=torch.float32,
                                      device=self.device)
        self.in_idx = torch.as_tensor(c.input_idx, dtype=torch.long,
                                      device=self.device)
        self.out_idx = torch.as_tensor(c.output_idx, dtype=torch.long,
                                       device=self.device)

        # The learned quantities. `log_gain` keeps the gain positive, so a
        # synapse can be silenced but never reversed: the presynaptic
        # neuron's transmitter decides the sign, not the optimiser.
        self.log_gain = torch.zeros(c.n_synapses, device=self.device,
                                    requires_grad=True)
        self.bias = torch.zeros(c.n, device=self.device, requires_grad=True)
        self.in_gain = torch.ones(len(c.input_idx), device=self.device,
                                  requires_grad=True)

    # -- forward ----------------------------------------------------------

    def rate(self, v):
        """Voltage to firing rate.

        `softplus` is unbounded, which is fine for feedforward mapping and
        fatal for memory: with no ceiling there is no high state to settle
        into, so a recurrent circuit can only decay or run away. Sweeping a
        global gain on the EPG ring bears that out -- it decays at 2.5 and
        explodes at 4.0, with no stable window between, the same silent-to-
        runaway transition `docs/FINDINGS.md` reports for the spiking model.

        `saturating` bounds the rate at `rmax`, as a real neuron is bounded.
        The same ring then holds its activity indefinitely after the stimulus
        is removed. Use it whenever the question involves persistence.
        """
        import torch
        import torch.nn.functional as F

        if self.rate_name == "saturating":
            return self.rmax * torch.sigmoid(v)
        return F.softplus(v)

    def parameters(self):
        return [self.log_gain, self.bias, self.in_gain]

    def _tensor(self, a):
        """Coerce to a float32 tensor on this net's device.

        Accepts arrays *and* tensors that are already on the device: `fit`
        moves X and Y to the GPU once and then hands those same tensors to
        `accuracy`, and routing a CUDA tensor back through numpy raises
        rather than copying.
        """
        import torch

        if torch.is_tensor(a):
            return a.to(device=self.device, dtype=torch.float32)
        return torch.as_tensor(np.asarray(a, dtype=np.float32),
                               device=self.device)

    def gains(self) -> np.ndarray:
        import torch

        with torch.no_grad():
            return torch.exp(self.log_gain).cpu().numpy()

    def forward(self, u, *, trace: bool = False):
        import torch
        import torch.nn.functional as F

        batch = u.shape[0]
        n = self.circuit.n
        w = self.w_anat * torch.exp(self.log_gain)

        inject = torch.zeros(batch, n, device=self.device)
        inject = inject.index_add(1, self.in_idx, u * self.in_gain)

        x = torch.zeros(batch, n, device=self.device)
        r = self.rate(x)
        history = []
        k = self.dt / self.tau
        for _ in range(self.steps):
            contrib = r[:, self.pre] * w
            drive = torch.zeros(batch, n, device=self.device)
            drive = drive.index_add(1, self.post, contrib)
            x = x + k * (-x + drive + self.bias + inject)
            r = self.rate(x)
            if trace:
                history.append(r[:, self.out_idx].detach().cpu().numpy())

        out = r[:, self.out_idx]
        return (out, np.asarray(history)) if trace else out

    def response(self, u, *, keep: str = "all") -> np.ndarray:
        """Every neuron's rate at every timestep, for one stimulus.

        `forward` reports only the output population, which answers "what did
        the circuit decide". This answers the prior question: what does the
        wiring *do* when you show it something. No training is involved --
        with gains at 1.0 the network is the anatomy, run forward.

        Returns (steps, n_neurons), or (steps,) reduced if `keep` is "mean".
        """
        import torch
        import torch.nn.functional as F

        x = self._tensor(u)
        if x.ndim == 1:
            x = x[None]
        with torch.no_grad():
            n = self.circuit.n
            w = self.w_anat * torch.exp(self.log_gain)
            inject = torch.zeros(x.shape[0], n, device=self.device)
            inject = inject.index_add(1, self.in_idx, x * self.in_gain)

            v = torch.zeros(x.shape[0], n, device=self.device)
            r = self.rate(v)
            k = self.dt / self.tau
            history = []
            for _ in range(self.steps):
                drive = torch.zeros(x.shape[0], n, device=self.device)
                drive = drive.index_add(1, self.post, r[:, self.pre] * w)
                v = v + k * (-v + drive + self.bias + inject)
                r = self.rate(v)
                history.append(r.mean(0).cpu().numpy() if keep == "all"
                               else float(r.mean()))
        return np.asarray(history)

    def response_by_type(self, u, *, baseline=None, limit: int = 0):
        """Per-cell-type response, strongest first.

        `baseline` is an optional second stimulus to subtract, which is what
        makes the number interpretable: the raw rate of a cell type mostly
        reflects how many inputs it has, whereas the *change* between two
        stimuli reflects what it is tuned to. Pass a blank field to get the
        response to the stimulus rather than to light in general.
        """
        rates = self.response(u)[-1]
        if baseline is not None:
            rates = rates - self.response(baseline)[-1]

        types = self.circuit.types.astype(str)
        order = np.argsort(types)
        names, starts = np.unique(types[order], return_index=True)
        groups = np.split(order, starts[1:])
        rows = [(str(nm), float(rates[g].mean()), int(len(g)))
                for nm, g in zip(names, groups)]
        rows.sort(key=lambda r: -abs(r[1]))
        return rows[:limit] if limit else rows

    def predict(self, X) -> np.ndarray:
        import torch

        with torch.no_grad():
            return self.forward(self._tensor(X)).cpu().numpy()

    # -- training ---------------------------------------------------------

    def fit(self, X, Y, *, epochs: int = 200, lr: float = 0.05,
            batch: int = 32, loss: str = "mse", anatomy_pull: float = 1e-3,
            report_every: int = 20, callback=None) -> list[dict]:
        """Fit synaptic gains so the circuit maps X onto Y.

        `anatomy_pull` is an L2 pull on log_gain toward zero, i.e. toward the
        anatomical strength. Without it the optimiser is free to wander
        arbitrarily far from the measured synapse counts, and the resulting
        circuit is a network that happens to share the fly's topology rather
        than a plausible tuning of the fly's own connectivity.
        """
        import torch
        import torch.nn.functional as F

        X, Y = self._tensor(X), self._tensor(Y)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        n_trials = X.shape[0]
        history = []

        for epoch in range(1, epochs + 1):
            order = torch.randperm(n_trials, device=self.device)
            total, seen = 0.0, 0
            for start in range(0, n_trials, batch):
                idx = order[start:start + batch]
                out = self.forward(X[idx])
                if loss == "ce":
                    err = F.cross_entropy(out, Y[idx].argmax(dim=1))
                else:
                    err = F.mse_loss(out, Y[idx])
                reg = anatomy_pull * (self.log_gain ** 2).mean()
                opt.zero_grad()
                (err + reg).backward()
                opt.step()
                total += float(err.detach()) * len(idx)
                seen += len(idx)

            row = {"epoch": epoch, "loss": total / max(seen, 1)}
            row["accuracy"] = self.accuracy(X, Y)
            history.append(row)
            if callback is not None:
                callback(row)
            elif report_every and (epoch % report_every == 0 or epoch == 1):
                print(f"  epoch {epoch:4d}   loss {row['loss']:.5f}   "
                      f"accuracy {100 * row['accuracy']:5.1f}%   "
                      f"mean gain {self.gains().mean():.3f}", flush=True)
        return history

    def accuracy(self, X, Y) -> float:
        """Fraction of trials whose largest output is the intended one.

        Only meaningful when the targets are one-per-trial patterns; for
        regression targets read the loss instead.
        """
        import torch

        with torch.no_grad():
            out = self.forward(self._tensor(X))
            return float((out.argmax(1) == self._tensor(Y).argmax(1))
                         .float().mean())

    # -- persistence ------------------------------------------------------

    def save(self, path) -> str:
        import torch

        with torch.no_grad():
            np.savez_compressed(
                path,
                log_gain=self.log_gain.cpu().numpy(),
                bias=self.bias.cpu().numpy(),
                in_gain=self.in_gain.cpu().numpy(),
                pre_body=self.circuit.body_ids[self.circuit.pre],
                post_body=self.circuit.body_ids[self.circuit.post],
                # Per-neuron state is keyed by body id too, so a memory can be
                # slotted into a *different* circuit -- a bigger one that
                # contains these cells, or the same cells re-extracted in
                # another order. Without these the arrays are positional and
                # only reload into an identically shaped network.
                bias_body=self.circuit.body_ids,
                in_body=self.circuit.body_ids[self.circuit.input_idx],
                out_body=self.circuit.body_ids[self.circuit.output_idx],
                fingerprint=np.array(self.circuit.fingerprint()),
                dataset=np.array(self.circuit.dataset),
                # The dynamics are part of the trained state: the gains were
                # fit assuming these, so restoring gains without them gives a
                # network that has the memory and does not reproduce it.
                steps=np.array(self.steps), tau=np.array(self.tau),
                dt=np.array(self.dt), rate=np.array(self.rate_name),
                rmax=np.array(self.rmax),
            )
        return str(path)

    def load(self, path, *, strict: bool = False) -> dict:
        """Load a trained state, matching everything by connectome body ID.

        Nothing is positional. Synapses join on (pre body id, post body id),
        per-neuron bias and per-input gain join on body id. A memory state is
        therefore portable: it can be loaded into the same circuit
        re-extracted in a different order, into a larger circuit that contains
        these cells, or into a circuit that only partly overlaps. Whatever is
        not in the file keeps its anatomical value, so a partial memory is a
        partial update rather than an error.

        `strict=True` refuses anything but an exact topology match, for when
        silent partial loading would be a bug rather than a feature.

        Returns the coverage, which is worth looking at: 12% matched means
        you loaded a mushroom-body memory into an optic-lobe circuit.
        """
        import torch

        data = np.load(path, allow_pickle=False)

        # Body ids are only meaningful inside one connectome release. Loading
        # across releases matches almost nothing and looks like an empty file
        # rather than the version mismatch it is, so say so.
        theirs = str(data["dataset"]) if "dataset" in data else ""
        if theirs and theirs != self.circuit.dataset:
            raise SystemExit(
                f"{path} was trained on {theirs}, this circuit is "
                f"{self.circuit.dataset}. Body ids do not carry across "
                "connectome releases, so nothing would match. Re-extract the "
                "circuit from the same release, or retrain."
            )
        same = str(data["fingerprint"]) == self.circuit.fingerprint()
        if strict and not same:
            raise SystemExit(
                f"{path} was trained on a different topology "
                f"({data['fingerprint']} vs {self.circuit.fingerprint()}); "
                "pass strict=False to load what overlaps"
            )

        def join(saved_ids, mine_ids, values, fallback):
            """Values from the file, indexed by body id, defaulting to `fallback`."""
            lookup = {int(b): i for i, b in enumerate(saved_ids)}
            rows = np.array([lookup.get(int(b), -1) for b in mine_ids])
            hit = rows >= 0
            out = np.full(len(mine_ids), fallback, dtype=np.float32)
            out[hit] = values[rows[hit]]
            return out, hit

        pair = lambda a, b: (a.astype(np.int64) * (10 ** 10)
                             + b.astype(np.int64))
        syn, hit_syn = join(
            pair(data["pre_body"], data["post_body"]),
            pair(self.circuit.body_ids[self.circuit.pre],
                 self.circuit.body_ids[self.circuit.post]),
            data["log_gain"], 0.0)

        # Older files stored bias and in_gain positionally.
        if "bias_body" in data:
            bias, hit_b = join(data["bias_body"], self.circuit.body_ids,
                               data["bias"], 0.0)
            ingain, _ = join(data["in_body"],
                             self.circuit.body_ids[self.circuit.input_idx],
                             data["in_gain"], 1.0)
        elif len(data["bias"]) == self.circuit.n:
            bias, hit_b = data["bias"], np.ones(self.circuit.n, bool)
            ingain = data["in_gain"]
        else:
            raise SystemExit(
                f"{path} predates body-id keying and its shapes do not match "
                f"this circuit ({len(data['bias'])} vs {self.circuit.n} "
                "neurons). Re-save it from the circuit it was trained on."
            )

        with torch.no_grad():
            self.log_gain.copy_(torch.as_tensor(syn, device=self.device))
            self.bias.copy_(torch.as_tensor(np.asarray(bias, np.float32),
                                            device=self.device))
            self.in_gain.copy_(torch.as_tensor(np.asarray(ingain, np.float32),
                                               device=self.device))

        # Restore the dynamics the gains were fit under, unless the caller
        # deliberately set something else.
        changed = {}
        for attr, field, default in (("steps", "steps", None),
                                     ("tau", "tau", None),
                                     ("dt", "dt", 1.0),
                                     ("rate_name", "rate", "softplus"),
                                     ("rmax", "rmax", 5.0)):
            if field not in data:
                continue
            want = data[field].item() if data[field].ndim == 0 else data[field]
            want = str(want) if field == "rate" else type(default or 0)(want) \
                if default is not None else want
            have = getattr(self, attr)
            if want != have:
                changed[attr] = (have, want)
            setattr(self, attr, want)

        # Whether the target can actually *use* the memory. Gains transfer by
        # body id regardless, but if the input or output population differs
        # you have the memory and no interface to it: the same vector no
        # longer means the same thing.
        io = {}
        for field, idx in (("in_body", self.circuit.input_idx),
                           ("out_body", self.circuit.output_idx)):
            if field in data:
                mine = set(int(b) for b in self.circuit.body_ids[idx])
                theirs = set(int(b) for b in data[field])
                io[field[:-5]] = (len(mine & theirs) / max(len(theirs), 1)
                                  if theirs else 0.0)

        return {
            "synapses_matched": int(hit_syn.sum()),
            "synapses_total": int(len(hit_syn)),
            "coverage": float(hit_syn.mean()),
            "neurons_matched": int(np.asarray(hit_b).sum()),
            "fingerprint_match": same,
            "dynamics_changed": changed,
            "input_match": io.get("in"),
            "output_match": io.get("out"),
        }
