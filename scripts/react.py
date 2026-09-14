#!/usr/bin/env python
"""Show the fly something and read what its neurons do about it.

No training. Gains sit at their anatomical value, so what comes back is the
wiring's own response -- which cell types care about this stimulus, and by how
much, relative to a blank field.

    .venv/bin/python scripts/react.py                      # dark vs bright flash
    .venv/bin/python scripts/react.py --stimulus looming
    .venv/bin/python scripts/react.py --image photo.png
    .venv/bin/python scripts/react.py --types 'L[1-5]' 'Tm.*' 'T4.*' --top 15
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from flybrain import trainable as T, vision as V

DEFAULT_TYPES = ["L1", "L2", "L3", "L5", "Mi1", "Mi4", "Mi9",
                 "Tm1", "Tm2", "Tm9", "Tm20", "C2", "C3", "T1"]


def stimuli(name, shape, frames):
    """(label, image) pairs to probe with."""
    if name == "flash":
        return [("dark field", V.flash(shape, 0.0)),
                ("bright field", V.flash(shape, 1.0))]
    if name == "looming":
        seq = V.looming(shape, frames=frames)
        return [(f"loom {i}/{frames - 1}", seq[i])
                for i in (0, frames // 2, frames - 1)]
    if name == "grating":
        return [("grating", V.grating(shape)),
                ("grating, shifted", V.grating(shape, phase=np.pi))]
    return [("dark disc", V.disc(shape, radius_px=min(shape) / 8))]


def main(a):
    circuit = T.build(a.types, a.inputs, a.outputs or a.inputs)
    print(circuit.summary())
    eyes = V.Binocular(circuit, invert=not a.no_invert)
    print(eyes.summary())
    shape = eyes.frame_size()

    net = T.BrainNet(circuit, steps=a.steps, seed=0,
                     device=T.pick_device(a.device))
    print(f"gains at anatomical strength -- this is the wiring, not a fit\n")

    if a.image:
        import PIL.Image
        img = PIL.Image.open(a.image).convert("L").resize((shape[1], shape[0]))
        probes = [(Path(a.image).name, np.asarray(img, np.float32) / 255.0)]
    else:
        probes = stimuli(a.stimulus, shape, a.frames)

    grey = eyes(V.flash(shape, a.baseline))
    for label, image in probes:
        rows = net.response_by_type(eyes(image), baseline=grey, limit=a.top)
        print(f"{label}  (change from a uniform {a.baseline:.1f} field)")
        for name, delta, n in rows:
            bar = "#" * min(40, int(abs(delta) * a.scale))
            sign = "+" if delta >= 0 else "-"
            print(f"  {name:6s} {delta:+9.4f}  {sign}{bar}")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimulus", default="flash",
                    choices=("flash", "looming", "grating", "disc"))
    ap.add_argument("--image", default=None)
    ap.add_argument("--types", nargs="*", default=DEFAULT_TYPES)
    ap.add_argument("--inputs", nargs="*", default=["L1", "L2", "L3", "L5"])
    ap.add_argument("--outputs", nargs="*", default=None)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--baseline", type=float, default=0.5)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--scale", type=float, default=60.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-invert", action="store_true",
                    help="inject raw luminance, skipping the histamine sign flip")
    main(ap.parse_args())
