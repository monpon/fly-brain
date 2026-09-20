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
neurons is now one continuous graph.

That experiment has now been run here, and it **works well enough to play
against**. `flybrain see-pong` puts a ball on a retina and drives a paddle
from what comes out of the lobula: **93% of balls returned**, against 16% for
a paddle that never moves, with the fly's estimate landing 10.9 px from the
ball on a 460 px court.

    ball on screen → retinotopic columns → lamina → medulla → LC11
      → decoded elevation → paddle

There is no mushroom body in it, no plasticity, and no coordinates handed
over — only 10,827 connectome neurons and the 143 LC11 cells that report
where the ball is. For comparison, the mushroom-body fly in the same game
returns ~95% while being *told* `ball_y` through olfactory projection
neurons.

Getting there took eight measurements that were each wrong in the same way,
and they are worth more than the result. The first was the metric: every
early number was root-mean-square error, and the error distribution has a
tight centre with a heavy tail — 7.8 px median, 119 px at the 90th
percentile. Rms squares the errors, so it reported the tail and hid the
centre, and a working readout read as a hopeless ~60 px floor across four
different pathways.

The rest were all one mistake: **fit in one regime, run in another.** A
decode fitted on stimuli settled from rest read 906 px in a loop that carries
membrane state. One fitted on single flights read 10 px held out and 96 px in
rallies. One fitted while a perfect paddle returned every ball read 99% and
128 px, because a decode changes the trajectories it then has to handle.
Cross-validating by frame instead of by ball reported 99.9% for a decode
reading its own neighbour. Carrying membrane state across a session the
calibration never spanned turned 90% into 8% by the fourth quarter. And
calibrating against an opponent that returned every ball dead centre removed
wall bounces from the training set entirely — found by a human playing it,
the one opponent that had not been simulated.

What survives from the negatives: the optic *flow* system really is the wrong
readout — VS pools ~892 retinotopic columns onto 34 cells and its elevation
response inverts across azimuth, which is a sign error rather than a
magnitude one. The motion pathway really does carry no motion, across an 18×
range of membrane time constants, because a leaky sum cannot form the
Reichardt product. And LC11 really is the best localiser in the circuit — 143
cells matching all 1,310 descending neurons and beating the 3,206-cell LC
population, which is independent evidence that these type labels carry
functional content. See [docs/FINDINGS.md](docs/FINDINGS.md).

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
flybrain pong --drill 90      # 90 balls of training, then the window opens
```

You are the left paddle, the fly is the right one, and it is taught only by
the dopamine rule -- no gradients, no labels, never told which way to move. A
live panel beside the court shows the circuit working: what the projection
neurons see, which Kenyon cells are firing, what the MBON channels are
saying, and where dopamine has written into the KC->MBON gains.

Against a returner that never misses, 200 balls:

| teaching signal | hit rate |
|---|---:|
| random actions | 22.5% |
| sparse: one bit per rally, ~137 decisions later | 92.0% |
| **dense: was that move toward the ball** | **100.0%** |

Measured in continuous rallies, where a ball only reaches the fly if it
returned the last one — so the sample leans on its own success. The same
conditioning inflated this project's aiming numbers by ~17 points, and the
flight-matched re-measurement of these figures is still outstanding. See
[docs/FINDINGS.md](docs/FINDINGS.md).

For most of this project's history this read ~21% and the notes said sparse
reward could not work. That was wrong, and it was a bug rather than a
finding: an uncentred readout gave one action a constant advantage, so the
paddle sat **parked** against a wall and every measurement was the geometric
odds of a ball arriving in a stationary 96-pixel window, `96/460`.

How much the feedback's *structure* matters turns out to depend on how much
the encoding has already done. `pong` hands the fly `ball_y − paddle_y`
already subtracted. `flybrain train-pong` does not — it supplies ball
position and paddle position on separate channels and makes the circuit
discover the relation. There sparse reward plateaus at **28.8%** against a
19.0% frozen-gain control, and stays there through MBON→MBON lateral
inhibition, bidirectional plasticity, a learnable output map and three
curricula, while dense feedback reaches **74.1%**.

What it learns is *tracking*, not prediction: the paddle sits 6 px from where
the ball is and 137 px from where it will arrive. That is the right strategy
here, since it moves faster than the ball drifts — and it could not do
otherwise, because ball *x* is never supplied, so time-to-arrival is not
derivable.

What reinforcement teaches well is single decisions. `flybrain teach` maps
six real odours onto three actions at **100%, on all 64 flies**, and still
holds 32 arbitrary associations at twice chance.

`flybrain intercept` asks the harder question — shown the ball at an
arbitrary moment in its flight, name the slice of court it will arrive in —
and separates the guess from the execution, which every hit rate confounds.
The guess is real: **40.7%** against a 30.7% always-one-bin baseline over
five seeds, and the estimate sharpens as the ball closes, from 116 px of
error to 77 px. It is also not enough, because 77 px exceeds the 48 px catch
window. Neither credit assignment, association capacity, input resolution nor
readout precision is the limit — each was tested and none moved it.

`flybrain train-aim-gpu` runs that task for 128 flies at once, which matters
more than speed: the same configuration spans 25–48% across single seeds, so
point estimates from one fly are noise. See
[docs/FINDINGS.md](docs/FINDINGS.md).

### ...and against one that can actually see it

Every pong above hands the fly numbers: `pong.py` place-codes ball_y, ball_vy,
paddle_y and ball_x into 276 *olfactory* projection neurons, and the mushroom
body learns an odour-to-action association we relabel as playing. It works,
and the fly is not seeing anything.

```bash
flybrain see-pong --pad-speed 9999 --reset-on-serve --frame-ms 20
```

This one has no mushroom body in it. A disc is drawn where the ball is, fed
into the photoreceptor-mapped input of a connectome circuit, and the paddle
goes wherever 143 LC11 cells say the ball is — **93% of balls returned**
against 16% for a parked paddle, holding flat across a session. A panel under
the court shows all 143 cells live, with the fly's estimate against the
ball's true height.

The flags are load-bearing rather than decoration. `--reset-on-serve` clears
the circuit between balls: without it the membrane state drifts beyond
anything calibration covered and play collapses from ~90% to 8% over a
session. `--pad-speed` frees the paddle from its 7 px/frame walk, worth ~10
points. `--frame-ms 20` gives the frame a budget a ~28 ms glance fits inside.

Calibration is cached in `output/`, keyed on the circuit *and* the conditions
it was fitted under — paddle speed, ball speed, glance interval, reset policy
— so changing any of them refits rather than silently reusing a decode from a
different regime. That key exists because getting it wrong is the single most
expensive mistake in this file's history.

## Next steps

1. Give the neuron model a multiplicative operation. Everything that failed
   here needs one: direction selectivity needs the delayed and undelayed arms
   multiplied, interception needs velocity times time-of-flight, and reading
   position out of a place code needs a quotient. `x ← x + (dt/tau)(−x + Wr +
   b + u)` with a monotone nonlinearity is a leaky *sum*. This is one line of
   the model and it is load-bearing for every negative result above.
2. Model presynaptic plasticity. Dopamine acts on Kenyon cell *terminals* in
   the animal, so it scales a KC's output to every MBON at once rather than a
   per-(KC, MBON) gain — a different interference structure, and 223,045
   synapses of DAN→KC that are not modelled at all.
3. Make the decode the fly's own. `see-pong` proves the information is in the
   connectome, but a kernel regression fitted offline is doing the reading.
   `BrainNet` is differentiable and `LearningParams.readout_rate` gives a
   reward-driven output map — either would move the extraction inside the
   animal, which is the difference between "the wiring carries this" and "the
   fly can use it".
4. Compare against classical baselines (Bug algorithms, infotaxis) under
   degraded sensing — apparently nobody has done this
