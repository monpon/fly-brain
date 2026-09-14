# Known issues

## Running on Windows

The package installs and runs on Windows; the core path is verified end to end
(install, `flybrain doctor`, table download, Feather parse, cell-type query) on
Windows 10 with Python 3.12.10. Three things differ from Linux and are worth
knowing before you debug something that is not a bug.

**Brian2 needs MSVC for its fast path.** Without a C++ compiler, Brian2 falls
back to numpy code generation and the LIF numbers in FINDINGS.md become
unreachably slow. It does not announce this. Install it once:

```
winget install --id Microsoft.VisualStudio.2022.BuildTools `
  --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

`flybrain doctor` reports `compiler yes`/`no`. It detects MSVC through
`vswhere`, not by looking for `cl.exe` on PATH — `cl` is only on PATH inside a
Developer Command Prompt, so a PATH check reports "no compiler" on a machine
that has one.

**PyPI torch on Windows is CPU-only, silently.** `pip install flybrain[torch]`
gives you `2.14.0+cpu`, and `pick_device("auto")` then reports
`cpu (no CUDA available)` on a machine with a perfectly good GPU. This is the
Windows counterpart of the `+cpu` trap already documented in
`requirements.txt` for Linux. Get the CUDA build from PyTorch's own index:

```
pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.14.0+cu126
```

Check with `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.
Note the GPU works here under Windows with driver 610.88 even though the
Linux side of this machine never built its kernel module (see "No GPU" below).

Measured on Windows with `2.14.0+cu126`, same RTX 3050 Mobile and i7-11800H,
30 epochs at batch 16 — CPU and GPU gains agree to six decimals, as on Linux:

| circuit | synapses | CPU | CUDA | speedup |
|---|---:|---:|---:|---:|
| mushroom body | 82,146 | 6.03 s | 1.15 s | 5.2x |
| lamina | 212,468 | 22.70 s | 2.55 s | 8.9x |
| optomotor | 349,867 | 32.81 s | 3.86 s | 8.5x |

The advantage grows with circuit size and with batch, so a small circuit at a
small batch is not where to judge it: the same mushroom body at batch 32 with
only six patterns gives 2.8x, because the GPU is mostly idle.

**Long paths.** `LongPathsEnabled` is 0 by default and some packages
(notably numpy's bundled f2py test fixtures) exceed 260 characters when the
install prefix is already deep. Symptom is a mid-install `OSError` naming a
very long path. Either keep the venv somewhere shallow or enable long paths:

```
# admin
Set-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' LongPathsEnabled 1
```

**Multiprocessing is spawn, not fork.** `flybrain tune-gait` re-imports the
module tree in every worker and each one builds its own MuJoCo model, so
worker startup costs far more than on Linux. `--workers` now defaults to one
per core capped at 12 rather than a hardcoded 12.

## Ground contact sensors fail to compile (flygym 2.1.0 + mujoco 3.9.0)

`world.add_fly(...)` with the default `add_ground_contact_sensors=True` raises:

```
ValueError: Error: unrecognized name 'fly/lf_coxa' of sensorized object
Element name 'ground_contact_lf_leg', id 0
```

The body `fly/lf_coxa` *does* exist in the compiled model — verified by listing
`mjOBJ_BODY` names after attachment — and building an equivalent contact sensor
by hand compiles fine both directly and through `MjSpec.copy()`. So the sensor
construction in `_add_ground_contact_sensors` is not obviously wrong in
isolation; something about the state of the spec on the real `add_fly` path
breaks the name lookup. Not root-caused.

This is not a version mismatch: flygym 2.1.0 pins `mujoco>=3.9,<3.10`, and
3.9.0 is the only release in that range.

**Workaround** (`src/flybrain/body.py`): pass `add_ground_contact_sensors=False`.
Ground contact *geoms* are still created, so physics and walking are unaffected —
only the convenience sensors are missing. `Simulation.get_ground_contact_info()`
depends on them and will not work. Read foot contact from the MuJoCo contact
list instead if you need it.

Revisit when flygym releases a patch.

## No GPU

`nvidia-driver-595-open` is installed and Secure Boot is off, but no `nvidia.ko`
exists for kernel 7.0.0-30-generic and `dkms` is not installed, so the module was
never built. To fix:

```
sudo apt install dkms linux-headers-$(uname -r)
sudo apt install --reinstall nvidia-driver-595-open
# reboot
```

Everything in this scaffold runs CPU-only, so this is not blocking.

## Rendering backend

`MUJOCO_GL=glfw` works (needs a display). `egl` fails on the Intel iGPU and
`osmesa` is not installed. For headless rendering: `sudo apt install libosmesa6`
then set `MUJOCO_GL=osmesa`.

## Memory: the weights table is 151.8M rows

`pd.read_feather()` on `connectome-weights-*.feather` materializes a ~9 GB
DataFrame, which on a 14 GB machine leaves very little headroom.

`local_data.weights_between()` instead filters with `pyarrow.compute.is_in`
against a memory-mapped Arrow table and converts only matching rows to pandas.
Peak RSS for the optomotor circuit drops from 8.9 GB to 5.9 GB with identical
output (215,057 nonzero entries, 619,702 synapses either way).

Do not call `local_data.weights()` unless you genuinely need the whole table.

## Neurotransmitter table quirks

`body-neurotransmitters-*.feather` has 1.84M rows against 211,577 annotated
bodies — most rows are untyped segmentation fragments, and 91% of
`consensus_nt` values are `unclear`. This is not a problem in practice:
coverage on *typed* neurons is complete (T4a, T5a, HSE, DNa02 all resolve to
acetylcholine, matching the literature).

Two traps when reading it:
- the body-id column is `body`, not `bodyId` as in the annotations table
- do not pick the NT column by substring-matching `"nt"` — that hits
  `total_nt_predictions`, which is a count. Use `consensus_nt`.

## The fly collides with the ground and with nothing else

FlyGym gives every geom `contype=0, conaffinity=0` and then emits 55 explicit
`<pair>` elements, one per body segment against `ground_plane`. Dynamic
collision therefore never runs for the fly. Dropped into a maze it walks
straight through the walls: 125 mm of travel across a 90 mm maze without a
single contact.

Setting `contype`/`conaffinity` on the compiled model does not fix it. With
the thorax mesh overlapping a wall box by 0.045 mm -- `mj_geomDistance`
returns `-0.045`, so MuJoCo can compute the penetration -- no contact is
generated, at any mask combination including `1/1` on both geoms, and with
every `mjDSBL_*` flag tried one at a time. A primitive sphere added to the
same world *does* land on the same wall. So the mesh-versus-box path is what
is not running; not root-caused beyond that.

**Workaround** (`src/flybrain/body.py::add_wall_proxies`): give the fly a
handful of invisible, massless (`density=0`) sphere geoms on the thorax, head,
abdomen and both funiculi, with masks that match walls only. Primitive-box
collision works, so the fly now stops at walls, and the antennal spheres
double as the wall sensor.

## get_body_positions is not in model body order

`Simulation.get_body_positions(fly_name)` returns rows ordered by
`fly.get_bodysegs_order()`, which is *not* the order of `mjOBJ_BODY` names in
the compiled model. Looking an index up with
`[mj_id2name(...) for i in range(nbody)].index("fly/c_thorax")` returns a
plausible-looking but wrong row -- in practice a tarsus, roughly a millimetre
ahead of the thorax during swing. Any arrival threshold built on it is wrong
by about that much.

**Correct form:**

```python
thorax = [s.name for s in fly.get_bodysegs_order()].index("c_thorax")
```

## Ubuntu's prebuilt NVIDIA module is pinned to one kernel

`nvidia-smi` reported "couldn't communicate with the NVIDIA driver" with the
full `-595` driver stack installed and the card visible on the PCI bus. The
cause was not the driver: `linux-modules-nvidia-595-open-7.0.0-**14**-generic`
was installed while the running kernel was **7.0.0-31**. Ubuntu ships
precompiled kernel modules pinned to an exact kernel version, and `dkms` was
not installed, so nothing rebuilt the module when the kernel was upgraded.

```bash
sudo apt install linux-modules-nvidia-595-open-$(uname -r) dkms
sudo modprobe nvidia nvidia_uvm     # no reboot needed
```

Installing `dkms` is what stops it recurring on the next kernel bump.
`nvidia_uvm` is the module CUDA needs; loading `nvidia` alone gets you
`nvidia-smi` but not compute. A reboot is not required once the module exists
for the running kernel, but the desktop session keeps the *old* userspace
libraries mapped until you log out, which shows up as:

    NVRM: API mismatch: the client 'gnome-shell' has the version 595.58.03,
    NVRM: but this kernel module has the version 595.91.07

Harmless for CUDA in a fresh process.

## trainable.py crashed on CUDA, silently worked on CPU

`fit()` moves `X` and `Y` to the device once, then passed those same tensors
to `accuracy()`, which called `np.asarray()` on them. On CPU that is a no-op.
On CUDA it raises `TypeError: can't convert cuda:0 device type tensor to
numpy`. Fixed with a `BrainNet._tensor()` helper that passes through tensors
already on the device; the three conversion sites now share it.
