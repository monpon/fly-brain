#!/usr/bin/env python
"""Show what the fly's retina makes of an image.

Renders a stimulus onto the measured hex lattice and draws both: the image as
given, and the 892 luminance values the right eye actually reports.

    flybrain see                       # looming disc
    flybrain see --stimulus grating
    flybrain see --image photo.png     # your own picture
    flybrain see --stimulus looming --frames 6
"""

import argparse
import sys
from pathlib import Path


import numpy as np

from flybrain import vision as V

OUT = Path("output/seen.png")


def make(name: str, shape, frames: int):
    if name == "flash":
        return V.flash(shape, 1.0)[None]
    if name == "grating":
        return V.grating(shape)[None]
    if name == "drifting":
        return V.drifting_grating(shape, frames=frames)
    if name == "disc":
        return V.disc(shape, radius_px=min(shape) / 6)[None]
    return V.looming(shape, frames=frames)


def load(path: Path, shape) -> np.ndarray:
    import PIL.Image

    img = PIL.Image.open(path).convert("L").resize((shape[1], shape[0]))
    return (np.asarray(img, dtype=np.float32) / 255.0)[None]


def main(a):
    ret = V.retina(a.side, a.type, a.kernel)
    eye = V.BoxEye(ret)
    print(ret.summary())
    shape = ret.frame_size()

    seq = load(Path(a.image), shape) if a.image else make(a.stimulus, shape,
                                                          a.frames)
    hexals = eye(seq)
    print(f"{len(seq)} frame(s) -> {hexals.shape[-1]} column values, "
          f"range {hexals.min():.2f}..{hexals.max():.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(seq)
    fig, axes = plt.subplots(2, n, figsize=(3.1 * n, 6.4), squeeze=False)
    # Axial coordinates to a plottable hexagonal layout. Same transform the
    # lattice was built with, flipped in y so the plot is not upside down.
    ys, xs = ret.centers[:, 0], ret.centers[:, 1]

    for i in range(n):
        axes[0][i].imshow(seq[i], cmap="gray", vmin=0, vmax=1)
        axes[0][i].set_title(f"frame {i}" if n > 1 else "stimulus", fontsize=9)
        axes[0][i].axis("off")

        axes[1][i].scatter(xs, -ys, c=hexals[i], s=a.dot, marker="h",
                           cmap="gray", vmin=0, vmax=1)
        axes[1][i].set_aspect("equal")
        axes[1][i].axis("off")
        if i == 0:
            axes[1][i].set_title(f"what eye {ret.side} reports "
                                 f"({ret.n} columns)", fontsize=9, loc="left")

    fig.tight_layout()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"saved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimulus", default="looming",
                    choices=("looming", "grating", "drifting", "flash", "disc"))
    ap.add_argument("--image", default=None, help="render your own picture")
    ap.add_argument("--side", default="R", choices=("R", "L"))
    ap.add_argument("--type", default="L1", help="cell type defining the lattice")
    ap.add_argument("--kernel", type=int, default=13)
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--dot", type=float, default=14.0)
    ap.add_argument("--out", default=str(OUT))
    main(ap.parse_args())
