# Known issues

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
