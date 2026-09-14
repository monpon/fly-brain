# Simulating a Fly Brain

Orientation notes: what the connectome data is, how to turn it into a running
simulation, and what questions that simulation can actually answer.

```bash
pip install -e .
flybrain demo
```

```python
import flybrain as fb

mb = fb.circuits.mushroom_body()     # bundled: no download, no token, offline
print(mb.summary())                  # 4,161 neurons, 61,210 synapses, 276 in, 97 out

net = fb.BrainNet(mb)                # pip install -e '.[torch]'
net.fit(X, Y)
```

Runs on Windows, macOS and Linux. `flybrain doctor` says what your machine has.

---

## 1. The dataset

**Male CNS Connectome** (Janelia FlyEM + Cambridge Connectomics + Google Research)
<https://www.janelia.org/project-team/flyem/male-cns-connectome>

The first finished connectome of an entire *male* Drosophila central nervous
system: brain, optic lobes, and ventral nerve cord (VNC), fully proofread and
annotated. Published in *Cell* (Sep 2026); preprint v2 on bioRxiv (Oct 2025).
Licensed CC-BY 4.0.

**Why it matters beyond "another connectome":**

- **No seam.** Prior datasets were partial. FlyWire and the hemibrain are
  brain-only; MANC is VNC-only. Simulating a full sensorimotor loop meant
  stitching two different animals together by matching descending neurons on
  morphology — and the seam sat exactly where the interesting computation
  happens. Here the descending neurons are physically continuous in one volume.
- **It's male.** 262 sex-specific and 114 sexually dimorphic cell types (~5% of
  the central brain). Sex-specific neurons concentrate in higher-order centers;
  sensory and motor regions are largely shared. This enables the first
  controlled male/female comparison of whole-CNS wiring.

**Access**

| Tool | Use |
|---|---|
| [neuprint.janelia.org](https://neuprint.janelia.org) | Web UI, connectivity queries |
| `neuprint-python` / `malecns` (R) | Programmatic access |
| Cell Type Explorer | Browse connectivity by neuron type |
| Clio | Annotations |
| Neuroglancer | EM volume + mesh visualization |
| NeuronBridge | Light-microscopy matching |

Connectivity and annotation **tables** are a few hundred MB — fine locally.
Raw **EM volumes** are tens of TB — leave those on the server.

---

## 2. How to simulate it

### The core idea (and its honest caveat)

A connectome is **anatomy, not dynamics**. It tells you who connects to whom
and with how many synapses. It does not tell you synaptic strength, sign, time
constants, or any cellular biophysics.

To make it run you add three assumptions:

1. Every neuron is a **leaky integrate-and-fire** (LIF) unit
2. Synaptic **weight ∝ synapse count**
3. Synaptic **sign** comes from the predicted neurotransmitter

Crude — but sufficient. Shiu et al. showed that stimulating sugar-sensing
neurons in such a model activates proboscis-extension motor neurons on the
correct timescale, and that bitter neurons suppress it. Real behavior out of a
static wiring diagram plus three assumptions.

### The three layers

**Layer 1 — Whole-brain spiking model**
[`philshiu/Drosophila_brain_model`](https://github.com/philshiu/Drosophila_brain_model)
Brian2, ~130k LIF neurons. Name neurons by ID, drive them at a chosen rate
(simulating optogenetics), silence others, read out spike trains brain-wide.
Pure CPU with C++ codegen. Minimally maintained (~2023, built on FlyWire's
female brain) — expect to patch it. **Best starting point.**

**Layer 2 — Trainable visual system**
[`TuragaLab/flyvis`](https://github.com/TuragaLab/flyvis)
PyTorch, connectome-constrained, *differentiable*. Trainable on optic-flow
tasks; predicts real recorded activity (Nature 2024). Choose this if you want
to do ML with the connectome rather than just poke it. Wants a working GPU.

**Layer 3 — A body**
[`NeLy-EPFL/flygym`](https://github.com/NeLy-EPFL/flygym) (NeuroMechFly v2)
MuJoCo digital twin that walks rough terrain, climbs, sees through ommatidia,
and smells. ~2x real-time on CPU, ~60x on GPU. Turns "neurons spiked" into
"the fly walked toward the odor."

**Not solved:** Layer 1 driving Layer 3 — a full connectome-derived brain
closing the loop through a physical body in real time. That's an open research
problem, not a package. Current practice is connectome models for the
interesting stage (e.g. descending command neurons) plus hand-written
controllers elsewhere.

### Install

Windows, macOS and Linux, Python 3.10 or newer:

```bash
pip install -e .
flybrain demo
```

That is the whole first run. No token, no download, no compiler, nothing to
configure — `demo` loads the mushroom body out of the package and prints what
it is made of, because the circuits worth starting from **ship inside the
wheel** as precomputed edge lists.

The core install is numpy, pandas and pyarrow. Everything expensive is an
extra, so a machine without a GPU or a C++ compiler still gets a working
install instead of a failed one:

```bash
pip install -e '.[torch]'     # training, memory, checkpoints
pip install -e '.[body]'      # MuJoCo physics, walking, mazes
pip install -e '.[spiking]'   # Brian2 LIF simulation
pip install -e '.[all]'       # everything
```

Reach for something an extra provides and you get the install command, not an
`ImportError`. `flybrain doctor` reports what this machine has, where its data
directory is, and whether Brian2 will find a compiler.

```bash
flybrain doctor
flybrain circuits             # what shipped in the bundle
flybrain fetch                # the full 1.1 GB tables, only if you want them
```

The bulk tables are optional and download on first use into the OS cache
directory (`%LOCALAPPDATA%\flybrain\Cache`, `~/Library/Caches/flybrain`, or
`$XDG_CACHE_HOME/flybrain`). Override with `FLYBRAIN_DATA_DIR` to put them on
another volume. `.env.example` lists every knob; all of them are optional.

Everything under `output/` is regenerated and untracked, except
`output/gait.json` — the tuned gait, ~10 minutes of CMA-ES to rebuild. A copy
also ships in the package, so the fly walks properly on a fresh install; a
local `output/gait.json` overrides it.

### Reproducing the measured numbers

`requirements.txt` pins the exact environment every figure in
[docs/FINDINGS.md](docs/FINDINGS.md) came from — Python 3.12 on Linux with
CUDA, on an i7-11800H and an RTX 3050 Mobile (4 GB). Use it when you are
checking a published result, not to install the package:

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt
```

Python 3.14 is still too new for the Brian2 / torch / MuJoCo wheels.

---

## 3. What you can do with it

### Perturb and observe
Activate or silence any named neuron or cell type and watch the consequences
propagate brain-wide. This is an optogenetics experiment with no fly, no rig,
and no waiting — the whole point of having the model.

### Trace causation through a full loop
With the seam gone, you can follow sensory input → central decision →
descending command → VNC motor circuit → motor neuron as one continuous
simulated path.

### Open questions worth attacking

**Courtship song, end to end** *(recommended first real experiment)*
P1/pC1 (male-specific courtship command) → pIP10 (descending) → wing motor
circuitry (hg1–hg4, ps1) → wing. Male-specific, so it could not have been
simulated before now. Output is quantitative: pulse trains at ~35 ms
inter-pulse intervals alternating with sine song, with decades of behavioral
data to check against. A few thousand neurons — runs on CPU in seconds, so you
can iterate on parameters all afternoon.

*Expect it to partially fail, and that's the value.* Rhythm generation likely
depends on adaptation currents, rebound, and intrinsic resonance that
LIF-plus-synapse-counts discards. Predicted outcome: correct routing
(P1 → pIP10 → motor neurons activate), wrong timing. A clean "connectivity is
sufficient for routing but not for rhythm" localizes exactly what the
connectome alone cannot tell you — on a circuit where the right answer is
already known.

**The matched-pair experiment**
Run identical dynamics on male and female connectomes for a shared circuit.
With the model held fixed, every behavioral difference is attributable to
wiring alone. No matched whole-CNS pair existed before this dataset.

**Vision all the way down**
flyvis stops at optic lobe output — it predicts neural activity, not behavior.
The male CNS includes optic lobes *and* VNC, so photoreceptors → leg motor
neurons is now one continuous graph. Nobody has run a visual stimulus in one
end and looked for a motor pattern at the other.

---

## Simulation

18,926 neurons and 349,867 connections run as leaky integrate-and-fire in
Brian2, at roughly **2x slower than real time** on CPU. Driving T4a (one
direction of ON-edge motion) at 100 Hz:

| type | n | rate (Hz) |
|---|---:|---:|
| T4a *(stimulated)* | 1684 | 82.2 |
| T4b / T4c / T4d | ~1700 ea | ~0.1 |
| T5a | 1664 | 23.9 |
| HSN / HSE / HSS | 2 ea | 46–54 |
| VS | 18 | 3.2 |
| DNa02 | 2 | 54.0 |

Signal reaches the descending steering command, the other three T4 direction
channels stay silent (no crosstalk), and the horizontal system responds while
the vertical system stays quiet — all consistent with horizontal motion driving
a yaw correction.

**But those numbers depend on a parameter the connectome does not provide.**
Raw synapse counts saturate HS and DNa02 at the refractory ceiling; the circuit
only behaves physiologically inside a narrow band of synaptic scaling, and
flips from silent to runaway across less than a factor of three.
See **[docs/FINDINGS.md](docs/FINDINGS.md)** — that result, not the table above,
is the actual finding.


## Training the brain on its own, without a body

The connectome as a trainable network: give it input patterns and the outputs
you want, and it fits the synaptic strengths. No physics, no flygym, no body.

```bash
flybrain train-brain                 # 6 real odours -> 6 MBONs
flybrain train-brain --task random --classes 10
flybrain train-brain --data mine.npz # your own X and Y
flybrain train-brain --list-io       # how many in/out?
```

`mine.npz` holds `X` of shape (trials, n_inputs) and `Y` of shape
(trials, n_outputs). For the default mushroom-body circuit that is 276 inputs
(one per olfactory projection neuron) and 97 outputs (one per MBON).

Any set of cell types can be the circuit:

```bash
flybrain train-brain \
    --types 'T4[a-d]' 'T5[a-d]' 'HS[NES]?' 'VS' 'DNa02' \
    --inputs 'T4[a-d]' 'T5[a-d]' --outputs 'DNa02'
```

The wiring stays the fly's: no synapse is added, removed, or sign-flipped.

Training runs on the GPU when one is available (`--device auto`, the default;
force it either way with `--device cpu|cuda`). Measured on an RTX 3050 Mobile
against an i7-11800H, that is **8.9x** on the mushroom body and **12.3x** on a
383,079-synapse visual circuit, with results agreeing to ~6 significant
figures on both devices. VRAM
is the limit rather than speed -- autograd retains every timestep, so memory
goes as `steps x batch x synapses`; halve `--batch` before assuming the card
is too small. See [docs/FINDINGS.md](docs/FINDINGS.md) for the numbers and
[docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) if `nvidia-smi` cannot find the
driver.
Training sets one non-negative gain per existing synapse. See
`docs/FINDINGS.md` for what it learns and how well.

## Seeing

`vision.py` renders an image onto the retina the connectome actually
describes: 892 hex columns on the right eye, 875 on the left, each a named
cell with a body ID.

```bash
flybrain see                    # a looming disc
flybrain see --stimulus drifting
flybrain see --image photo.png  # your own picture
```

```python
from flybrain import trainable as T, vision as V

circuit = T.build(["L1", "L2", "L3", "L5", "T4.*", "T5.*"],
                  ["L1", "L2", "L3", "L5"], ["T4.*", "T5.*"])
eyes = V.Binocular(circuit)
X = eyes(V.disc(eyes.frame_size(), radius_px=40))   # -> (n_inputs,)
```

Trained on object position with balanced inputs it reaches 66.7% on held-out
positions against 33% chance -- it separates left cleanly but cannot tell
centre from right. On looming versus receding versus static it also reaches
66.7%, which there is the ceiling: those are the same frames in opposite
order, so one frame can only identify the static case. `BrainNet` takes one
vector per trial -- motion needs sequences.

Watch the eye balance. `L3`, `C2` and `Tm4` carry column addresses on the
right eye only, so including them hands the model a lateralisation cue that
looks like retinotopy and is not. `Binocular.summary()` warns; see
[docs/FINDINGS.md](docs/FINDINGS.md).

To see what the wiring does with no training at all:

```bash
flybrain react --stimulus looming
```

Photoreceptors are histaminergic, so the lamina inverts: L1 and L2 depolarize
to *darkness*. `vision.py` applies that sign flip, because no photoreceptor in
this volume carries a column address to route it through. See
[docs/FINDINGS.md](docs/FINDINGS.md).

## Remembering

```bash
flybrain remember            # all three experiments
flybrain remember capacity   # how many associations
flybrain remember attractor  # does activity outlive input
```

Synaptic memory works and persists: train on images, `save()`, load into a
network built fresh from the anatomy, and it recalls them. About 150
associations survive a degraded cue; clean recall has no ceiling we found.

Activity memory needs `BrainNet(rate="saturating")`. With unbounded rates the
heading ring only decays or explodes; bounded, it holds its state with no
input -- though only one state, not a heading.

Memory states are portable. `save()`/`load()` key everything to connectome
body ids, and the dynamics travel with the gains, so a file trained on one
machine loads into a brain rebuilt from scratch on another and recalls
exactly. `load()` reports coverage and whether the target's input and output
populations match -- gains transfer regardless, but a circuit with different
inputs has the memory and no interface to it.

Nothing temporal can be held yet, because `forward` applies the same input at
every timestep. See [docs/FINDINGS.md](docs/FINDINGS.md).

## Playing pong against it

```bash
flybrain pong
```

You are the left paddle, the fly is the right one, and it is taught only by
the dopamine rule -- reward on intercept, punishment on miss. It does not work
yet: 19.6% against a chance rate of 18.9% over 500 balls. A rally is ~137
decisions yielding one bit at the end, and the eligibility trace cannot reach
back that far. The same machinery learns a single-decision task to 100%. See
[docs/FINDINGS.md](docs/FINDINGS.md).

## Next steps

1. Map `get_ommatidia_readouts()` (2 x 721 x 2) onto retinotopic T4/T5 input.
   The transduction function is an open modelling decision, not a given, and
   without it direction selectivity cannot be tested at all.
2. Map DNa02 firing onto a turning command in flygym
3. Close the loop: visual motion in, compensatory turn out
4. Compare against classical baselines (Bug algorithms, infotaxis) under
   degraded sensing — apparently nobody has done this
