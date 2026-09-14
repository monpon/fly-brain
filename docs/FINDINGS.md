# Findings

## The connectome routes signal correctly. It does not set the operating point.

Extracting the optomotor pathway from `male-cns:v1.0` and running it as a
leaky integrate-and-fire network reproduces the right *anatomy* immediately:
signal flows Mi1/Mi4/Mi9 -> T4 -> HS/VS -> DNa02, with Mi4 (GABAergic) and
Mi9 (glutamatergic) correctly inhibitory against Mi1's excitation. None of that
was tuned.

The dynamics are a different story.

### Raw synapse counts saturate the network

Using Shiu et al.'s parameterisation directly — every anatomical synapse worth
a fixed 0.275 mV EPSP — stimulating T4a at 100 Hz gives:

| type | rate |
|---|---:|
| T4a (stimulated) | 93 Hz |
| HSE | 433 Hz |
| DNa02 | 433 Hz |

433 Hz is the ceiling imposed by the 2.2 ms refractory period (455 Hz). Those
cells are not computing; they are pinned.

The reason is a three-order-of-magnitude spread in input counts:

| type | synapses in, per cell | mV if all fire | x threshold |
|---|---:|---:|---:|
| Mi1 | 8 | 2.1 | 0.3 |
| T5a | 25 | 6.8 | 1.0 |
| DNa02 | 68 | 18.8 | 2.7 |
| T4a | 130 | 35.7 | 5.1 |
| VS | 8,793 | 2,418 | 346 |
| HSE | 15,548 | 4,276 | **611** |

A single fixed per-synapse EPSP cannot serve both Mi1 and HSE. Wide-field
integrators collect thousands of inputs by design, so any uniform weighting
either starves the small cells or saturates the large ones.

### Homeostatic scaling helps, but the window is narrow

Rescaling each postsynaptic cell's weights so its total input magnitude equals
a fixed budget is the standard fix. Sweeping that budget, with T4a driven at
100 Hz throughout:

| budget (mV) | T4a | T5a | HSE | VS | DNa02 |
|---:|---:|---:|---:|---:|---:|
| raw | 93 | 1 | 433 | 332 | 433 |
| 8 | 82 | 1 | 0 | 0 | 0 |
| 10 | 82 | 6 | 22 | 0 | 22 |
| 12 | 82 | 23 | 48 | 2 | 55 |
| 14 | 82 | 47 | 77 | 15 | 78 |
| 16 | 84 | 73 | 108 | 28 | 117 |
| 20 | 91 | 141 | 167 | 114 | 165 |
| 28 | 134 | 248 | 257 | 243 | 257 |

Below 8 mV the circuit is silent. Above ~20 mV it is saturating. Physiological
rates occur only around 10–14 mV, and the transition from dead to runaway spans
less than a factor of three.

**That budget is a free parameter the connectome does not supply.** The wiring
diagram fixes which neurons talk to which and how often, and that turns out to
be enough to get signal to the right place — but not enough to determine
whether the circuit does anything sensible when it gets there. The operating
point is chosen by the modeller.

This is the quantitative form of the prediction in the README: connectivity is
sufficient for routing, insufficient for dynamics.

### Caveats that matter

- **HS and VS cells do not spike in reality.** They are graded-potential
  neurons. Modelling them as LIF units is wrong in kind, not just in degree,
  and their firing rates above should be read as "how hard is this cell being
  driven", not as a prediction.
- **This is a subcircuit.** Eight cell types were extracted from a network of
  211,577 bodies. Inhibitory partners outside the selection are missing, so the
  isolated circuit has less restraint than the real one — part of the
  saturation is an artifact of the cut.
- **Synapses are instantaneous voltage jumps.** No conductance dynamics, no
  synaptic time constants, no short-term plasticity, no adaptation currents.
- **Direction selectivity is untested.** All T4 subtypes are driven identically
  because there is no spatial visual input yet. Testing the actual optomotor
  computation requires mapping ommatidia onto retinotopic T4 inputs.

---

## The brain does not connect to the joints

Direct test on `male-cns:v1.0`, counting synapses between superclasses:

| pathway | synapses | connections |
|---|---:|---:|
| brain intrinsic (121,567 cells) -> motor | **80** | 8 |
| brain intrinsic -> descending | 2,745,199 | 303,972 |
| descending (1,314) -> motor | 229,565 | 17,645 |
| VNC intrinsic (13,161) -> motor | 2,214,610 | 155,068 |

Eighty synapses between a 121,567-neuron brain and 708 motor neurons is
annotation noise. The brain reaches the legs through a **1,314-cell descending
bottleneck**, and motor neurons draw roughly **ten times more input from VNC
circuits than from descending commands**. Whatever generates a gait lives in
the VNC, not the brain.

## A connectome VNC routes commands but does not produce a gait

`src/flybrain/vnc.py` builds the real pathway with no hand-written oscillator:

    descending (1,314) -> VNC intrinsic (13,161) -> motor (708) -> joints

15,183 neurons, 2,386,069 connections, and unlike the optomotor circuit it is
close to balanced: 1,236,816 excitatory against 1,149,253 inhibitory.

Motor neurons are annotated by *muscle* ("Ti flexor MN"), MuJoCo is actuated by
*joint*, so the mapping is many-to-one and signed — net command = extensors
minus flexors. 328 motor neurons resolve onto **42 of the 66 actuated DOFs**,
which is 7 joints x 6 legs. The 24 that go unmapped are exactly the distal
tarsus chain, 4 per leg, which in a real fly has no dedicated motor neurons and
is driven passively by the long tendon muscle. The map is anatomically
complete, not truncated.

### Driving it

Stimulating DNp09, the forward-walking command, at 100 Hz:

| synaptic budget | total spikes | network Hz | motor Hz |
|---:|---:|---:|---:|
| 12 mV | 81 | 0.01 | 0.00 |
| 25 mV | 90 | 0.01 | 0.00 |
| 50 mV | 1,813 | 0.24 | 0.00 |
| 100 mV | 174,352 | 22.97 | 7.11 |

At 100 mV motor neurons fire at plausible rates and individual joints carry
real signed commands. Then the spectrum:

| joint | mean command | peak Hz | power share |
|---|---:|---:|---:|
| lf/coxa_yaw | -81 | 2 | 0.115 |
| lf/femur_pitch | -181 | 32 | 0.110 |
| lh/coxa_pitch | -33 | 27 | 0.084 |
| lh/coxa_yaw | +96 | 35 | 0.089 |
| lh/tibia_pitch | -34 | 3 | 0.137 |

**Rhythmic joints: 0 of 42.** Power share sits at 0.06–0.14, meaning a flat
spectrum, and the peak frequencies scatter randomly (2, 3, 25, 27, 32, 35,
41 Hz) with no consistency across the six legs. That is noise, not a gait.

So the descending command reaches the right motor neurons, through the right
circuit, at the right rates — and nothing walks. Same failure as the optomotor
circuit, one level deeper: **connectivity determines routing, not dynamics.**

### What is missing, specifically

- **Synaptic time constants.** Synapses here are instantaneous voltage jumps.
  An oscillator needs delay structure to oscillate around; there is none.
- **Intrinsic cellular dynamics.** Adaptation currents, post-inhibitory
  rebound, intrinsic bursting — the ingredients half-centre oscillators are
  built from, all discarded by LIF.
- **Proprioceptive feedback.** This one is our omission, not the model's. The
  VNC was run open-loop, with no body. Real leg coordination depends on load
  and position feedback, and the dataset contains **6,370 `vnc_sensory`
  neurons** we never connected to anything.

That last point is a testable hypothesis rather than a caveat: the gait may be
a property of the **brain-body loop**, not of the neural circuit alone. We have
both halves — a connectome VNC with a sensory population, and a physics body
that reports joint angles and contact forces. Closing that loop is the
experiment.

---

## Closed loop: connectome VNC <-> body, no pattern generator

`src/flybrain/embodied.py` wires the whole thing together:

    body --proprioception--> vnc_sensory (1,273 leg proprioceptors)
                                  |
                            VNC intrinsic  <-- descending command
                                  |
                             motor neurons (328 mapped)
                                  |
    body <---muscle activation----+

21,553 neurons, 2,982,086 connections, stepped in lockstep with MuJoCo at a
1 kHz sensorimotor update. Runs at **11x slower than real time** on CPU.

Proprioceptors resolved from the connectome by entry nerve and subclass:
370 chordotonal (femur-tibia angle), 60 hair plate (joint limits), 843 leg
mechanosensory (ground load).

Biological choices layered on the anatomy, since the connectome supplies none
of them: motor neuron spikes drive a muscle activation variable with a 30 ms
decay (calcium/contraction low-pass); antagonist muscles subtract; activation
sets an equilibrium *target angle* rather than a raw torque, because insect
muscle is intrinsically stiff and the fly has no stabilising controller.

### It does not walk

| descending | budget | gain | muscle act. | displacement | path |
|---|---:|---:|---:|---:|---:|
| DNp09 | 100 | 0.35 | 0.170 | 0.10 mm | 0.16 mm |
| DNp09 | 200 | 1.00 | 0.571 | 0.09 mm | 0.20 mm |
| MDN | 100 | 0.35 | 0.187 | 0.09 mm | 0.15 mm |
| MDN | 200 | 1.00 | 0.837 | 0.12 mm | 0.16 mm |
| DNa02 | 100 | 0.35 | 0.167 | 0.11 mm | 0.15 mm |
| DNa02 | 200 | 1.00 | 0.715 | 0.11 mm | 0.20 mm |

Three different descending commands, muscle activation varied roughly
**fivefold**, and displacement never leaves 0.09-0.12 mm. The fly settles onto
its legs and stays there.

### Why: not co-contraction, but incoherence

The obvious explanation is that flexors and extensors fire together and cancel.
**That is not what happens.** Measuring activation per antagonist group:

| joint | extensor | flexor | net | net/total |
|---|---:|---:|---:|---:|
| lf/coxa_pitch | 0.827 | 0.038 | +0.790 | 0.91 |
| lf/femur_pitch | 0.376 | 1.231 | -0.856 | 0.53 |
| lh/coxa_pitch | 0.000 | 1.699 | -1.699 | 1.00 |
| lf/tibia_pitch | 0.038 | 0.284 | -0.246 | 0.77 |

Net commands are large, strongly signed, and the mean temporal variation of the
net joint command relative to activation is **0.91** — these joints are being
driven hard and they are changing constantly.

So the motor system is neither silent nor jammed. It is **incoherent**. Each of
the 42 joints receives a big, fluctuating, essentially independent command, and
six legs pushing in uncorrelated directions produce no net displacement.

This sharpens the earlier spectral result rather than repeating it. Walking is
not about whether motor neurons fire, or how hard. It is about *phase* — which
leg lifts while which others bear load. That is precisely the structure a
connectome plus LIF does not supply, and closing the proprioceptive loop did not
conjure it: adding 1,273 real sensory neurons reporting real joint angles and
real contact forces changed nothing measurable.

The negative result is now three-deep and consistent: connectivity determines
**routing**, not dynamics, not rhythm, and not coordination.

---

## Correction: two bugs confounded the locomotion results above

Everything in the previous section that concerns *movement* was measured on a
body that could not move. Two separate defects:

**1. Wrong body tracked.** `Simulation.get_body_positions()` returns all 69
model bodies, and index 0 is the MuJoCo **world** body, permanently at the
origin. The thorax is index 1. Every displacement figure above was reading a
body that is fixed by definition.

**2. Position actuators had no usable gain.** FlyGym's default leaves them far
too weak to move legs against joint stiffness. Measured: commanding +0.5 rad on
all 66 joints produced **0.037 rad** of actual motion, a tracking error of
0.477. The fly could not execute any motor command, from any controller.

Setting `kp=50` in `build_fly()` raises tracking to ~84%.

### What changes, and what does not

With the body fixed, the hand-written CPG walks: **7.3 mm in 2 s (3.6 mm/s)**,
path length 34 mm, against a real fly's 10-15 mm/s. So the body is now
demonstrably capable of locomotion, which makes the connectome comparison fair
for the first time.

Re-running the connectome closed loop on the working body, driving DNp09:

| config | muscle act. | displacement | path length |
|---|---:|---:|---:|
| biophysical, budget 200 | 0.010 | 0.19 mm | 0.75 mm |
| instantaneous, budget 200 | 0.456 | 0.72 mm | 11.88 mm |
| instantaneous, budget 400 | 2.063 | 0.30 mm | 6.00 mm |
| instantaneous, budget 400, gain 3 | 2.275 | **2.64 mm** | 12.10 mm |
| *CPG, for comparison* | — | *7.27 mm* | *34.22 mm* |

This is a **materially different result** from the one reported above. The
connectome-driven fly is not inert: it generates 12 mm of leg movement and
2.6 mm of net displacement. It moves the legs and goes almost nowhere, which is
stumbling, not paralysis.

The earlier claim that displacement was pinned at 0.09-0.12 mm regardless of
drive was an artifact of the actuator bug, and is withdrawn.

**What survives unchanged:** the spectral analysis. Rhythm was measured on
*neural* output, not on body movement, so the actuator defect does not touch it.
0 of 42 joints rhythmic, flat spectra, and no rhythm from adding synaptic time
constants or adaptation. The connectome still does not produce a gait — it now
produces uncoordinated leg motion instead of none, and the gap to the CPG is
roughly 3x in displacement.

**Methodological lesson:** a negative result about a controller is worthless
until the plant is shown to be controllable. The CPG was the control condition
that exposed this, and it should have been run first.

## Learned odour valence steers the body

The mushroom body decides *whether* a smell is worth approaching; the antennal
lobe supplies *which way*. Both halves now feed the walking controller, and the
behaviour follows what the fly was taught rather than anything hard-coded.

Training (`scripts/teach_odor.py`) is two-sided differential conditioning on
real glomerular patterns: vinegar (DM1/VA2/DM4/DC2/VM2) paired with PAM reward,
geosmin (DA2) with PPL1 shock, 40 epochs.

| | vinegar | geosmin |
|---|---:|---:|
| before | +0.00342 | +0.00021 |
| after | +0.00686 | **-0.00186** |

Depression-only plasticity produces both signs, as it should: reward and
punishment DANs innervate different compartments, so depressing one set raises
the net valence and depressing the other lowers it. 5.5% of the 61,210 plastic
synapses end below gain 0.9, concentrated in MBON03, MBON24 and MBON02.

### Behaviour, both sources 40 mm out at +/-25 degrees

All distances below are from the **nearest body segment** to the source
centre. An earlier version of these numbers tracked a body index looked up in
the model's body list, which `get_body_positions` does not use -- it silently
returned a tarsus, about a millimetre ahead of the thorax during swing. The
"reached in 3.85 s" figures reported then were that error crossing an arrival
threshold. The corrected numbers:

| condition | closest to good | closest to bad |
|---|---:|---:|
| trained, vinegar at +25 | 2.8 mm | 28.2 mm |
| trained, arena mirrored | 2.5 mm (touched) | 26.6 mm |
| untrained brain | 3.7 mm | 23.3 mm |
| memory swapped, same arena | 2.9 mm to geosmin | 27.6 mm from vinegar |

The source marker has a 2.0 mm radius, so 2.5-2.9 mm is the fly's head
arriving within a millimetre of the sphere. Whether that counts as "found" is
a threshold choice; the closest-approach column is the measurement.

The swap is the control that matters. With geosmin rewarded and vinegar
punished, the physical arena unchanged, the fly walks the other way. Nothing in
the controller knows which source is which; the sign of a learned valence is
the whole difference.

### Where it fails

**Aversion can outweigh attraction.** In the swapped run the fly closes to
4.8 mm on the good source and then sails past (final 100 mm). The same odour
*alone* is reached in 3.02 s, so the memory is fine -- the problem is balance.
Geosmin is a single glomerulus, so the reward it can accumulate saturates at
+0.00225, while the punishment on 5-glomerulus vinegar reaches -0.00341. The
repulsion is larger than the attraction and pushes the fly off course.
Rescaling the tanh (`--valence-scale 0.0008`) does not fix it: both terms
saturate together, so the ratio survives.

**An aversive source on the direct path blocks the trip.** With geosmin at
25 mm and vinegar at 55 mm along the same bearing, the fly avoids the bad
source successfully (16.1 mm closest, from a 25 mm start) and then flees the
whole field rather than routing around it. Real flies detour; this controller
has no way to hold a goal while retreating, because there is no goal, only a
sum of gradients.


## Solving a maze by smell

`scripts/maze_forage.py` generates a perfect maze, puts a trained-attractive
odour at the far corner, and lets the fly walk. It is given no map and no
coordinates: two antennal concentrations, a learned valence, and antennal
contact.

**Odour that obeys walls.** A point source with an exponential falloff points
the gradient straight at the target through solid geometry, so the fly walks
into a wall and stays there. `odor_field.MazeOdorSource` computes a Dijkstra
distance over a raster of the open space instead, and the concentration falls
off with that geodesic. Following the gradient then leads *around corners*,
because around the corner is genuinely where the smell is stronger.

| maze (4x4, 20 mm corridors) | route | reached | walked |
|---|---:|---:|---:|
| seed 0 | 121 mm | yes | 124 mm |
| seed 1 | 111 mm | yes | 132 mm |
| seed 2 | 111 mm | yes | 126 mm |
| seed 3 | 111 mm | yes | 126 mm |
| seed 4 | 197 mm | yes | 287 mm |
| seed 5 | 215 mm | yes, 90 s | 853 mm |
| seed 7 | 205 mm | yes, 30 s | 359 mm |

Seven for seven. Path efficiency is 90%+ on the short mazes and drops to 25%
on seed 5, where the fly orbits the source several times before settling --
the turning circle at full speed is about as wide as a corridor, so the last
few millimetres are the hardest part of the trip.

**What the memory does, and does not, decide here.** An *untrained* fly also
solves the maze (seed 0, 124 mm walked), because vinegar starts out mildly
attractive: +0.00342 before any conditioning. Training doubles that, which
changes little when the odour is the only thing in the maze. The memory
becomes decisive when the sign has to flip. With **geosmin** at the exit:

| fly | outcome |
|---|---|
| taught geosmin = punished | closest 7.0 mm, ended 31.9 mm away, did not settle |
| taught geosmin = rewarded | solved the maze |

Same maze, same odour, same body. The only difference is which dopaminergic
population was active during 40 training trials.

## A brain with no body: supervised training on the wiring diagram

`src/flybrain/trainable.py` + `scripts/train_brain.py`. Nothing in this path
imports flygym or MuJoCo -- verified by loading it and checking `sys.modules`.
Inputs go in as arrays, outputs come out as arrays.

The connectome fixes what cannot be learned:

* **Topology.** No synapse is created or destroyed. If the EM volume shows no
  connection from A to B, gradient descent cannot invent one.
* **Sign.** Excitatory or inhibitory follows the presynaptic cell's predicted
  transmitter, because a neuron releases the same transmitter at all its
  terminals. Training cannot flip it.

What is learned is one non-negative gain per existing synapse, multiplying the
anatomical weight -- the same multiplicative representation `learning.py` uses
for dopamine-gated plasticity, so trained states stay portable across
re-extraction (joined on body-ID pairs, not row order).

### Results

Circuit: PN -> KC -> MBON, 4,437 neurons, 82,146 synapses, 276 inputs,
97 outputs. Leaky rate units, 8 timesteps, Adam.

| task | before | after | 30% input noise |
|---|---:|---:|---:|
| 6 real odours -> 6 MBONs | 0% (chance 17%) | **100%** | 100% |
| 10 arbitrary sparse patterns | — | **100%** | 100% |
| 4 user-supplied patterns | — | **100%** | 100% |

Output margins are large (+9 to +20), so these are not marginal wins, and
accuracy survives 30% input noise, so the circuit generalises rather than
memorising exact vectors. Roughly 72% of synapses end below anatomical
strength and 26% above.

Arbitrary circuits work too: `--types 'T4[a-d]' 'T5[a-d]' 'HS[NES]?' 'VS'
'DNa02'` builds a 13,606-neuron visual circuit with 215,057 synapses, 13,580
T4/T5 inputs and the 2 DNa02 steering outputs.

### What this is and is not

Gradient descent is not biological. Dopamine does not compute a gradient, and
`learning.py` implements what the fly actually does -- depression of KC->MBON
synapses on coincidence with reinforcement. This is an engineering tool for a
different question: not "how does the fly learn" but "what is this wiring
*capable* of". An unconstrained network of the same size would fit far more;
the constraint is the point.

## The GPU is worth 10x, including on the circuit I expected it not to help

`BrainNet.forward` is a gather, a multiply and a scatter-add over the synapse
list, repeated for each of `steps` timesteps:

    contrib = r[:, pre] * w              # gather, (batch, n_synapses)
    drive   = drive.index_add(1, post, contrib)

That is memory-bandwidth bound, not compute bound, so a GPU should win. What
it won by, measured at `steps=12`, `batch=32`, 200 trials, RTX 3050 Mobile
(4 GB, sm_86) against an i7-11800H:

| circuit | neurons | synapses | CPU s/epoch | GPU s/epoch | speedup | VRAM |
|---|---:|---:|---:|---:|---:|---:|
| mushroom body | 4,437 | 82,146 | 4.12 | 0.46 | 8.9x | 0.27 GiB |
| LC+LPLC -> DN | 6,016 | 383,079 | 25.82 | 2.10 | 12.3x | 1.22 GiB |

I predicted the mushroom body would see no gain, on the theory that kernel
launch overhead dominates at that size. It got 8.9x. The twelve timesteps
each dispatch only a handful of large kernels, so there is far less launch
overhead than the neuron count suggests -- the batch dimension keeps every
kernel wide.

Training agrees across devices but is not bit-identical: the same six odours
resolve to the same MBONs, with margins matching to about six significant
figures (+10.3848 on CPU against +10.3847 on CUDA). `index_add` on CUDA
accumulates with atomics in nondeterministic order, so float addition is not
associative between runs -- expect agreement, not reproducibility to the last
bit.

VRAM, not speed, is the ceiling. Autograd retains every timestep, so memory
goes as `steps x batch x synapses`. 1.22 GiB of the 3.61 available at the
sizes above leaves roughly 3x headroom; past that, halve `--batch` before
concluding the card is too small.

Only this path is accelerated. Brian2 (`network.py`, `simulate_vnc.py`)
generates C++ with no CUDA backend, and everything touching MuJoCo -- the
gait tuner, the maze, foraging -- runs on the CPU physics engine that flygym
uses. Moving those would mean porting to MJX, which is a rewrite rather than
a flag.

## Pixels into a measured retina

`vision.py` maps an image onto the neurons that look at it. The male-CNS
annotations give 1,771 optic-lobe neurons a hex column address
(`assignedOlHex1`, `assignedOlHex2`) -- 892 columns on the right eye, 875 on
the left -- so a receptor lattice can be built from the fly's own addressing
rather than assumed.

Transduction follows flyvis (Lappalainen et al., Nature 2024): a column's
value is the mean of a 13x13 pixel square centred on it. No optics, no
acceptance function, no temporal filter.

**Two things this does differently.** flyvis assumes spatial homogeneity and
tiles one local connectome across 721 synthetic receptors, because its source
data resolved only a few columns. Here the lattice is measured and the output
vector is indexed by real body IDs.

And the lattice geometry is isotropic. flyvis lays receptors out as
`y = d*(u + v/2), x = d*v`, which puts neighbours at d, 1.118d and 1.118d --
not a regular hexagon. That is harmless when stimuli are generated in the
same coordinates that sample them, but a photograph arrives skewed: a looming
disc rendered as a tilted rounded rectangle. Using `y = d*sqrt(3)/2*u,
x = d*(v + u/2)` puts all six neighbours at exactly d.

**The two eyes share hex addresses.** A left-eye L1 and a right-eye L1 can
both be at (5, 7). Keying the lookup on address alone silently maps one eye
onto the other and reports 87% coverage of a population that is 50% on this
side. `ColumnInput` filters on `somaSide`; `Binocular` composes the two and
asserts the index sets are disjoint.

### The photoreceptors are histaminergic, and that is a sign flip

R1-R6 are histaminergic: 3,377 of 3,377 at cell-type level here. Histamine at
the photoreceptor-to-lamina synapse is *inhibitory* -- it opens chloride
channels -- so real L1 and L2 depolarize to light **decrements**. A dark
object on a light field is what drives them.

Injecting raw luminance into L1 therefore gets the polarity of the entire
early visual system backwards, and the connectome cannot correct it here: **no
photoreceptor in the volume carries a column address.** R1-R6, R7p, R7y, R8p
and R8y all have zero, so there is no retinotopic way to inject into them and
let the measured R-to-L sign do the work. `ColumnInput` applies the inversion
by hand, `invert=True` by default.

With the correct polarity, L1 and L2 depolarize to a dark field and **Tm1 and
Tm2 follow them** -- which is the OFF pathway, reproduced from measured
wiring rather than fitted.

### What it learns

Circuit: L1/L2/L3/L5/Mi1/Tm1/Tm2/Tm9 -> T4/T5, 27,786 neurons and 433,598
synapses, 7,114 inputs of which 6,196 (87%) carry a column address.

| task | chance | train | held out | inputs |
|---|---:|---:|---:|---|
| object position (left / centre / right) | 33% | 100% | 100% | L1 L2 **L3** L5 |
| object position, balanced | 33% | 66.7% | **66.7%** | L1 L2 L5 |
| looming vs receding vs static | 33% | 66.7% | **66.7%** | L1 L2 L5 |

**The 100% is an artifact and the 66.7% is the real number.** L3 carries
column addresses on the right eye only -- 892 columns there, zero on the left,
though 880 L3 cells exist in the left lobe. Including it gives the model 36%
more input channels on one side, so "the object is on the right" has a
trivially unique signature that owes nothing to retinotopy. Dropping L3
balances the eyes to within 2.4% and the score falls to 66.7%, flat from
epoch 100 through 400. The confusion matrix shows exactly what is missing:

        pred:  left  centre  right
    left         8      0      0
    centre       0      8      0
    right        0      8      0     <- right is called centre, always

Left is separated cleanly; right and centre are not separable at all. A
lesson about the dataset rather than about the fly, and the reason
`Binocular.summary()` now warns when the eyes are unbalanced and
`unbalanced_types()` names the offenders.

Under the wrong (raw-luminance) polarity the unbalanced task reached 91.7%
train and 83.3% held out against 100% with the histamine sign flip -- but
since that 100% is itself an L3 artifact, the polarity comparison is only
suggestive, not a clean 17-point result.

The looming result is the more informative one, and it is a limit rather
than a score -- it lands on the ceiling exactly.

Looming and receding are the *same frames in opposite order*, so a single
frame cannot separate them. The best available strategy is to identify
"static" perfectly (1/3 of trials) and guess between the other two (half of
the remaining 2/3, so another 1/3): **2/3 = 66.7%**. Train and held-out both
come in at 66.7%. The circuit extracts all the information a single frame
contains and then stops.

An earlier run reported 43% at 20 epochs, which was undertraining, not the
limit. The limit is 66.7% and the network reaches it.

The cell types that exist to compute motion have nothing to compute it from,
because `BrainNet` takes one vector per trial. Sequence input -- injecting a
different `u` at each timestep instead of holding one constant -- is what
unlocks T4/T5 and the DNp01 looming test.

## What it can remember

Three different things get called memory. They behave nothing alike here.

`scripts/remember.py` runs all three.

### Synaptic: pattern to value

The one that works. Learned per-synapse gains on KC->MBON, saved by
`checkpoint.py`, and already load-bearing: it is what makes the fly walk
toward vinegar and solve the maze.

| patterns | clean cue | 10% noise | 30% noise |
|---:|---:|---:|---:|
| 6 | 100% | 100% | 100% |
| 50 | 100% | 100% | 100% |
| 97 | 100% | 100% | 94.8% |
| 150 | 100% | 100% | 94.7% |
| 250 | 100% | 100% | 77.6% |
| 400 | 100% | 100% | 71.7% |

Clean recall never breaks -- 400 patterns and no ceiling found. Recall from a
*degraded* cue is the real limit, and it holds to about 150 patterns before
the memories interfere. Since there are only 97 MBONs, beyond 97 patterns
several must share an output: 400 remembered items sorted into 97 answers,
not 400 distinct valences. A fly learns a handful of odours in its life, so
the substrate is not what constrains the animal.

### Synaptic memory of an image, and it persists

Same mechanism, visual input. Train on four scenes, save, load into a network
built fresh from the anatomy:

    after training      :  75.0%
    fresh net, no memory:   0.0%
    memory restored     :  75.0%

Zero to seventy-five on loading a file. The gains key to connectome body ids,
so a checkpoint survives re-extracting the circuit.

75% is a weak readout, not a weak memory: the outputs are individual Tm cells
in neighbouring columns, so the classes nearly overlap. The right anatomy for
visual memory is in the dataset and is not this -- **visual projection
neurons make 8,041 synapses onto KCgamma-d**, the visual Kenyon cell class,
with aMe12, aMe26, MeVP41 and LoVP42 the largest sources. Reading out at MBON
would give 97 separated outputs. Reaching it needs the whole
lamina->medulla->lobula->VPN->KC chain, which is past the dense-matrix limit
in `build()`.

### Activity: the heading ring

EPG, PEN, PEG, Delta7 and ER -- 430 neurons, 48,503 synapses -- are a ring
attractor, the fly's compass. Poke the EPG cells, remove the input, watch.

| global gain | softplus | saturating |
|---:|---|---|
| 2.0 | decays | decays |
| 4.0 | runaway | decays |
| 8.0 | runaway | **persists** |

**The nonlinearity was the whole problem, not the wiring.** `softplus` is
unbounded, so there is no high state to settle into and the ring can only
decay or explode -- the same silent-to-runaway transition this file already
reports for the spiking model, reappearing in a completely different circuit.
Bound the rate and the same anatomy holds its activity indefinitely with the
stimulus gone. `BrainNet(rate="saturating")` does that.

But it holds **one** state. Poking any subset of EPG cells settles to the
same global pattern: 23 different pokes, one attractor, pairwise similarity
1.00. That is a 1-bit flag, not a heading. A real ellipsoid body holds a
localised bump because local excitation is precisely balanced against
Delta7's global inhibition, and a uniform gain scale cannot produce that
balance. The topology supports a bump; the anatomical gains do not implement
one. Training would have to.

### What is still impossible

Nothing temporal. `forward` computes `inject` once and applies the same
vector at every timestep, so the network stares at a frozen frame. The state
`x` *does* carry across timesteps -- that leaky integrator is exactly the
short-term memory a motion detector needs -- but it is never given anything
changing to hold.

This is one wall, and it is the same wall behind every remaining limitation:
the 66.7% looming ceiling, T4/T5 having no motion to detect, predictive Pong,
and a heading that updates as the fly turns. Two changes lift it: index the
input per timestep, and make `tau` a learnable per-cell-type parameter, which
is the delay line a Hassenstein-Reichardt correlator is built from and what
flyvis learns as `tau_ti`.

### A memory state is a file you can move

`BrainNet.save`/`load` key everything to connectome body ids -- synapses on
`(pre, post)`, bias per neuron, input gain per input cell. Nothing is
positional, so a memory survives being re-extracted, reordered, moved to
another machine, or dropped into a circuit it was not trained on.

Trained on a GPU with non-default dynamics, loaded on a CPU into a network
built fresh from the anatomy:

    trained on cuda (steps=6 tau=3): 100.0%
    fresh on CPU, default dynamics :   0.0%
    after load on CPU              : 100.0%     exact round trip

The dynamics travel with the gains. `steps`, `tau`, `dt`, `rate` and `rmax`
are saved and restored, because the gains were fit assuming them -- restoring
gains alone gives a network that holds the memory and does not reproduce it.
A `saturating` memory loaded into a `softplus` network is a different network.

`load` reports whether the target can actually *use* what it loaded:

| target circuit | synapse coverage | inputs | outputs | usable |
|---|---:|---:|---:|---|
| same circuit, rebuilt | 100% | 100% | 100% | yes, 100% recall |
| `KC.*` `MBON.*` `APL` | 8% | 0% | 100% | memory, no interface |
| optic lobe | 0% | 0% | 0% | nothing transferred |

The middle row is the case worth understanding. The gains transferred --
61,210 synapses of the mushroom body landed in a 722,090-synapse circuit --
but that circuit takes Kenyon cells as input rather than projection neurons,
so the same vector no longer means the same thing. Portability of a memory is
not portability of the interface.

## The dopamine rule accumulates; gradient descent does not

Teaching a second thing with gradient descent erases the first, completely:

    learned odours              : 100.0%
    then learned 10 new patterns: 100.0%
    do the odours survive       :   0.0%

Training on both together recovers 100% on each, so this is interference
rather than capacity. It means a gradient-trained memory file holds only what
was trained in one sitting.

`learning.py` does not have the problem. Three odours taught one at a time,
never retraining the earlier ones:

| taught | retained at the end |
|---|---:|
| vinegar (1st, rewarded) | 86% |
| geosmin (2nd, punished) | 91% |
| banana (3rd, rewarded) | 100% |

Signs stay correct throughout -- vinegar +0.00595, geosmin -0.00170, banana
+0.00413 against a naive +0.00342 / +0.00021 / +0.00126. Only **8.4% of
synapses end up depressed**, which is the reason: each odour recruits a
different sparse ~5% subset of Kenyon cells, so the memories barely overlap.
Sparse coding is what buys incremental learning.

(A first attempt measured 18% retention. That was a broken test loop calling
`reset_state()` between trials, which discards the learned gains entirely
rather than just the eligibility trace. Clear `_trace`, as `tasks.py` does.)

### Memories from separate flies merge

Depression-only multiplicative gains compose. Two brains trained
independently, combined by taking the lower gain per synapse:

    fly A (vinegar+) : vinegar +0.00637  geosmin +0.00021
    fly B (geosmin-) : vinegar +0.00342  geosmin -0.00191
    merged (min)     : vinegar +0.00636  geosmin -0.00190

100% of each memory survives, and the depressed fractions add (3.3% + 2.9% ->
6.2%). Skills can therefore be trained separately and combined, rather than
having to share one training run.

### .flyckpt round-trips exactly

Three odours taught by the dopamine rule, saved, loaded into a fresh brain:

    EXACT gain array equality : True
    EXACT readout equality    : True
    valences identical        : True
    trials restored           : 120 -> 120
    params restored           : True
    matched 61210  (100.0% of target)

Not approximately -- `np.array_equal`. The format is a versioned zip holding
a readable manifest plus body-id-keyed arrays, so `unzip -p f.flyckpt
manifest.json` tells you what a checkpoint is without loading the connectome.

## Teaching a fly by reward and punishment

Gradient descent is gone from this path. `learning.py` is the trainer, and
what follows was measured with it.

### Variability belongs in the action, not the readout

`learning.py` had no randomness, which looked like a missing feature and was
not: the variability was already there, one layer down, in `navigate.py` and
`olfactory_brain.py`. That is the right place. A fly's behavioural
variability comes from premotor circuits, not from reading its memories
badly, so the synaptic readout should stay deterministic.

What was wrong was its shape. Fly spontaneous turning is heavy-tailed rather
than Gaussian -- closer to a Levy walk, and actively generated. `noise_tail`
is a Student-t degree of freedom, rescaled by `sqrt(df/(df-2))` so `noise`
still means one standard deviation:

| tail | sd | max excursion | P(abs(turn) > 0.10) |
|---:|---:|---:|---:|
| 3 | 0.0292 | 0.754 | 1.11% |
| 5 | 0.0298 | 0.266 | 0.80% |
| 30 (Gaussian) | 0.0301 | 0.142 | 0.095% |

Same size, eleven times as many large excursions. The rescaling matters: the
first version used df=2, whose variance is infinite, and would have silently
tripled the noise everywhere.

### Credit has to name the action, and the dopamine has to be gated

Two separate things, both required.

`learn` is classical -- it tags whatever the fly was *seeing*. That is right
for "this odour predicts sugar" and wrong once behaviour earned the outcome,
because exploration can make the action disagree with the valence that
produced it. `learn_operant` binds reinforcement to `sign(action * outcome)`,
so a noise-flipped action that succeeds reinforces the direction actually
taken rather than the one the valence preferred.

With several behavioural channels there is a second problem, and it is
fatal. Dopamine floods a compartment; every MBON reading it is affected. So
the synaptic update is identical whatever the fly chose, and no choice is
learnable. Gating dopamine to the compartments read by the chosen channel
fixes it -- which is what compartment specificity is for in the animal.

Two stimuli, two output channels, one requiring each:

    dopamine gated to the chosen channel : 100%
    dopamine flooding all compartments   :  50%     (chance)

`FlyBrain(mb, n_actions=k)` gives each action a disjoint slice of the MBON
population. Disjoint is what makes them independently trainable: the plastic
gains are shared, so overlapping readouts would move together and the fly
could never prefer one action over another. Which MBON drives which action is
not in the connectome; splitting in order is a placeholder for a behavioural
mapping that would have to come from experiment.

### Pong does not work, and the reason is instructive

`scripts/pong.py` is a playable tkinter window. The fly is taught only by
reward on intercept and punishment on miss. Over 500 balls:

| balls | hit rate | depressed |
|---:|---:|---:|
| 50 | 24% | 10.7% |
| 150 | 16% | 16.2% |
| 250 | 22% | 18.6% |
| 350 | 16% | 19.4% |
| 500 | 24% | 20.5% |

First five blocks 20.0%, last five 19.6%, chance 18.9%. No trend at all,
while the depressed fraction doubles and plateaus -- the whole plastic budget
spent on nothing.

This is not a bug in the rule or in the signs, both of which were checked.
The same two-channel machinery learns a single-decision task to 100%. The
difference is that a rally is about 137 decisions and yields one bit at the
end, attributed to the final frame. `trace_tau` is 2 *calls* against 137 per
rally, so the eligibility trace cannot reach the decisions that placed the
paddle.

Temporal credit assignment is therefore the wall, and it is the same wall as
the 66.7% looming ceiling and the unused motion detectors: this brain has no
way to relate something that happened now to something it did earlier.

Untested and plausible: dense per-frame reinforcement judged on whether a
move closed the gap, and feeding a longer history of positions and
velocities. Neither has been measured.
