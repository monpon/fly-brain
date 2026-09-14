"""Turn an image into activations of real, identified retinotopic neurons.

Everything else in this project reads the connectome as a graph. This reads it
as a *retina*: the male-CNS annotations assign 1,771 of its neurons a hex
column address (`assignedOlHex1`, `assignedOlHex2`), which is the fly's own
addressing of its ommatidia. Given those addresses, an image can be sampled at
the position each neuron actually looks at.

The transduction follows flyvis (Lappalainen et al., Nature 2024,
github.com/TuragaLab/flyvis, `flyvis/datasets/rendering/eye.py`): a receptor's
value is the mean of a `kernel_size` square of pixels centred on it, with
receptors laid out on a hexagonal lattice in axial coordinates. It is
deliberately crude -- no optics, no acceptance function, no temporal filter --
because that is what the published model uses and it works.

Where this departs from flyvis
------------------------------
flyvis assumes spatial homogeneity: one local connectome tiled across the
visual field as a hex convolution, with 721 receptors on a synthetic lattice.
It has to, because the data it was built on only resolved a few columns.

Male-CNS v1.0 resolves every column, so the lattice here is *measured*: 892
columns on the right eye, 879 on the left, each one a named cell with a body
ID. No tiling assumption, and the output vector is indexed by real neurons.

What this does not model
------------------------
All lamina cells in a column see the same photoreceptors. L1 and L2 differ in
temporal filtering and in which half of the ON/OFF split they feed, not in
where they look. Injecting the same luminance into both is therefore right
about space and silent about time: a single frame carries no motion, and the
cell types that exist to compute motion (T4, T5) have nothing to work with
until you feed a sequence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HEX1, HEX2 = "assignedOlHex1", "assignedOlHex2"


@dataclass
class Retina:
    """One eye's ommatidial lattice, as measured.

    `hex1`/`hex2` are the connectome's own axial coordinates. `centers` is
    where each column lands in pixel space once the lattice is laid out.
    """

    side: str
    body_ids: np.ndarray      # (n_cols,) one representative neuron per column
    hex1: np.ndarray          # (n_cols,) axial u
    hex2: np.ndarray          # (n_cols,) axial v
    centers: np.ndarray       # (n_cols, 2) pixel (y, x), origin at image centre
    kernel_size: int
    type_name: str

    @property
    def n(self) -> int:
        return len(self.body_ids)

    def frame_size(self) -> tuple[int, int]:
        """Smallest odd-sided image that contains every receptor footprint."""
        half = self.kernel_size / 2
        span = self.centers.max(axis=0) - self.centers.min(axis=0) + 2 * half
        return tuple(int(2 * np.ceil(s / 2) + 1) for s in span)

    def summary(self) -> str:
        h, w = self.frame_size()
        return (f"eye {self.side}: {self.n} columns of {self.type_name}, "
                f"{self.kernel_size}px receptors, needs a {h}x{w} frame")


def _axial_to_pixel(u: np.ndarray, v: np.ndarray, d: int) -> np.ndarray:
    """Axial hex coordinates to pixel centres, on a regular lattice.

        y = d * sqrt(3)/2 * u
        x = d * (v + u / 2)

    Every one of the six neighbours is then exactly `d` pixels away, which is
    why the box kernel is also `d` wide: the receptors tile the frame without
    gap or overlap.

    This differs from flyvis, which uses `y = d*(u + v/2), x = d*v` and leaves
    the sqrt(3) version commented out in `eye.py`. That shear is harmless when
    stimuli are generated in the same coordinates it samples, but it is not a
    regular hexagon -- neighbour distances come out d, 1.118d, 1.118d -- so
    photographs and round objects arrive on the retina visibly skewed. Since
    the point here is to map real images onto a measured eye, the lattice is
    made isotropic instead.
    """
    return np.stack([d * np.sqrt(3) / 2 * u.astype(float),
                     d * (v + u / 2.0)], axis=1)


def retina(side: str = "R", type_name: str = "L1", kernel_size: int = 13,
           annotations=None) -> Retina:
    """Build the lattice from one cell type's measured column addresses.

    `type_name` only selects *which* neurons define the lattice. Any of the
    types that tile one-per-column will do; L1 is the default because it is
    the largest complete tiling (1,767 cells over 1,767 columns) and the
    direct postsynaptic target of the photoreceptors.
    """
    if annotations is None:
        from . import local_data as L

        annotations = L.annotations()

    sel = annotations[(annotations.type == type_name)
                      & (annotations.somaSide == side)]
    sel = sel.dropna(subset=[HEX1, HEX2])
    if sel.empty:
        raise SystemExit(
            f"no {type_name} neurons with column addresses on side {side}. "
            "Only 15 types carry assignedOlHex1/2; L1, L2, L5, Mi1, Tm1 do."
        )
    # One neuron per column: a handful of types have a duplicate cell in a
    # column, and an image has only one value to give it.
    sel = sel.drop_duplicates(subset=[HEX1, HEX2])

    u = sel[HEX1].to_numpy(dtype=int)
    v = sel[HEX2].to_numpy(dtype=int)
    # Centre the lattice on the image: the raw addresses run 1..36 and 1..39,
    # so without this the eye looks at the bottom-right corner of the frame.
    centers = _axial_to_pixel(u - u.mean(), v - v.mean(), kernel_size)
    return Retina(side=side, body_ids=sel.bodyId.to_numpy(dtype=np.int64),
                  hex1=u, hex2=v, centers=centers, kernel_size=kernel_size,
                  type_name=type_name)


class BoxEye:
    """Cartesian pixels to hexals, by box filter. After flyvis's `BoxEye`.

    >>> eye = BoxEye(retina("R"))
    >>> eye(image).shape          # (n_columns,)
    """

    def __init__(self, ret: Retina):
        self.retina = ret
        k = ret.kernel_size
        off = np.arange(k) - (k - 1) // 2
        dy, dx = np.meshgrid(off, off, indexing="ij")
        # (n_cols, k*k) pixel indices, resolved once so sampling is a gather.
        self._offsets = np.stack([dy.ravel(), dx.ravel()], axis=1)

    def sample_points(self, shape: tuple[int, int]) -> np.ndarray:
        """Pixel indices each receptor averages over, clipped to the frame."""
        h, w = shape
        base = self.retina.centers + np.array([h / 2.0, w / 2.0])
        idx = np.rint(base[:, None, :] + self._offsets[None, :, :]).astype(int)
        idx[..., 0] = np.clip(idx[..., 0], 0, h - 1)
        idx[..., 1] = np.clip(idx[..., 1], 0, w - 1)
        return idx

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Mean luminance in each receptor's footprint.

        `image` is (H, W) greyscale, or (frames, H, W) for a sequence, in
        which case the result is (frames, n_columns).
        """
        a = np.asarray(image, dtype=np.float32)
        if a.ndim == 3:
            return np.stack([self(frame) for frame in a])
        if a.ndim != 2:
            raise ValueError(f"expected (H, W) or (frames, H, W), got {a.shape}")

        idx = self.sample_points(a.shape)
        return a[idx[..., 0], idx[..., 1]].mean(axis=1)


class ColumnInput:
    """Bridge: image in, `X` vector for a Circuit's input neurons out.

    The circuit's input neurons are matched to the lattice by column address,
    so the value a neuron receives is the luminance at the point on the retina
    that neuron actually looks at. Input neurons with no column address in the
    annotations -- T4 and T5 have none, for instance -- are left at zero and
    counted in `unmapped`.
    """

    def __init__(self, circuit, side: str = "R", kernel_size: int = 13,
                 lattice_type: str = "L1", annotations=None,
                 invert: bool = True, baseline: float = 0.5):
        from . import local_data as L

        if annotations is None:
            annotations = L.annotations()
        self.retina = retina(side, lattice_type, kernel_size, annotations)
        self.eye = BoxEye(self.retina)
        self.circuit = circuit

        # body id -> column. Keyed on side as well as address: the two eyes
        # reuse the same hex coordinates, so dropping the side silently maps
        # left-eye neurons onto the right eye's retina and reports ~87%
        # coverage of a population that is only ~50% on this side.
        placed = annotations.dropna(subset=[HEX1, HEX2])
        placed = placed[placed.somaSide == side]
        addr = dict(zip(placed.bodyId.to_numpy(dtype=np.int64),
                        zip(placed[HEX1].astype(int),
                            placed[HEX2].astype(int))))
        column_of = {(int(a), int(b)): i for i, (a, b)
                     in enumerate(zip(self.retina.hex1, self.retina.hex2))}

        # For each of the circuit's input slots, which hexal feeds it.
        src, dst = [], []
        for slot, node in enumerate(circuit.input_idx):
            key = addr.get(int(circuit.body_ids[node]))
            if key is None:
                continue
            hexal = column_of.get(key)
            if hexal is None:
                continue          # a column on the other eye
            src.append(hexal)
            dst.append(slot)

        self.invert = invert
        self.baseline = baseline
        self.src = np.asarray(src, dtype=np.int64)
        self.dst = np.asarray(dst, dtype=np.int64)
        self.n_inputs = len(circuit.input_idx)
        self.unmapped = self.n_inputs - len(self.dst)

    def frame_size(self) -> tuple[int, int]:
        return self.retina.frame_size()

    def _drive(self, hexals: np.ndarray) -> np.ndarray:
        """Luminance to the drive a lamina cell actually receives.

        R1-R6 are histaminergic -- 3,377 of 3,377 at cell-type level in this
        dataset -- and histamine at the photoreceptor-to-lamina synapse is
        inhibitory: it opens chloride channels. So L1 and L2 depolarize to
        light *decrements*, not increments, and a dark object on a light
        field is what drives them.

        The connectome cannot supply this inversion here, because no
        photoreceptor in the volume carries a column address: R1-R6, R7 and
        R8 all have zero. There is no retinotopic way to inject into them and
        let the R-to-L sign carry the polarity, so it is applied by hand.

        Injecting raw luminance instead (`invert=False`) gets the sign of the
        entire early visual system backwards.
        """
        if not self.invert:
            return hexals
        return self.baseline - hexals

    def __call__(self, image: np.ndarray, gain: float = 1.0) -> np.ndarray:
        """(H, W) -> (n_inputs,), or (frames, H, W) -> (frames, n_inputs)."""
        hexals = self._drive(self.eye(image))
        if hexals.ndim == 2:
            X = np.zeros((len(hexals), self.n_inputs), dtype=np.float32)
            X[:, self.dst] = hexals[:, self.src] * gain
            return X
        X = np.zeros(self.n_inputs, dtype=np.float32)
        X[self.dst] = hexals[self.src] * gain
        return X

    def summary(self) -> str:
        mapped = len(self.dst)
        pct = 100.0 * mapped / max(self.n_inputs, 1)
        return (f"{mapped}/{self.n_inputs} input neurons mapped to columns "
                f"({pct:.0f}%), {self.unmapped} left at zero; "
                f"{self.retina.n} columns on eye {self.retina.side}")


class Binocular:
    """Both eyes at once, summed into one `X` vector.

    The two eyes' neurons are disjoint, so their contributions never collide
    and a single vector carries both. Bilateral comparison is how the fly
    turns -- the same principle the odour navigator uses -- so a circuit that
    must choose a direction needs this rather than one eye.
    """

    def __init__(self, circuit, kernel_size: int = 13,
                 lattice_type: str = "L1", annotations=None,
                 invert: bool = True, baseline: float = 0.5):
        from . import local_data as L

        if annotations is None:
            annotations = L.annotations()
        self.left = ColumnInput(circuit, "L", kernel_size, lattice_type,
                                annotations, invert, baseline)
        self.right = ColumnInput(circuit, "R", kernel_size, lattice_type,
                                 annotations, invert, baseline)
        self.n_inputs = self.left.n_inputs
        overlap = np.intersect1d(self.left.dst, self.right.dst)
        if len(overlap):
            raise AssertionError(
                f"{len(overlap)} input neurons claimed by both eyes; the "
                "side filter in ColumnInput is not working"
            )
        self.mapped = len(self.left.dst) + len(self.right.dst)
        n_l, n_r = len(self.left.dst), len(self.right.dst)
        self.imbalance = (n_r - n_l) / max(n_l, 1)

    def unbalanced_types(self, annotations=None) -> list[str]:
        """Input cell types addressed on one eye only.

        The two optic lobes are not equally annotated. L3, C2 and Tm4 have
        zero column addresses on the left -- 880, 871 and 837 cells that
        exist in the volume and were never assigned to a column -- while the
        right is complete. The animal is symmetric at about 885 ommatidia per
        eye; the dataset is not.

        Including such a type gives the model more input channels on one side
        than the other, which is a confound for any task that turns on
        comparing the eyes. `L1`, `L2`, `L5`, `Mi1`, `Mi4`, `Mi9`, `Tm1`,
        `Tm2`, `Tm9`, `Tm20`, `C3` and `T1` are addressed on both.
        """
        from . import local_data as L

        if annotations is None:
            annotations = L.annotations()
        placed = annotations.dropna(subset=[HEX1, HEX2])
        types = self.left.circuit.types.astype(str)
        wanted = set(types[self.left.circuit.input_idx])
        odd = []
        for name in sorted(wanted):
            sides = set(placed[placed.type == name].somaSide.dropna())
            if {"L", "R"} - sides:
                odd.append(str(name))
        return odd

    def frame_size(self) -> tuple[int, int]:
        """A frame size that fits both eyes, which differ by a few columns."""
        a, b = self.left.frame_size(), self.right.frame_size()
        return (max(a[0], b[0]), max(a[1], b[1]))

    def __call__(self, left_image, right_image=None, gain: float = 1.0):
        """One image for both eyes, or a separate image per eye."""
        if right_image is None:
            right_image = left_image
        return self.left(left_image, gain) + self.right(right_image, gain)

    def summary(self) -> str:
        pct = 100.0 * self.mapped / max(self.n_inputs, 1)
        note = ""
        if abs(self.imbalance) > 0.05:
            note = (f" -- {abs(self.imbalance) * 100:.0f}% more on the "
                    f"{'right' if self.imbalance > 0 else 'left'}, see "
                    "unbalanced_types()")
        return (f"{self.mapped}/{self.n_inputs} input neurons mapped "
                f"({pct:.0f}%): {len(self.left.dst)} left eye, "
                f"{len(self.right.dst)} right eye{note}")


# -- stimuli ---------------------------------------------------------------
#
# Enough to drive the thing and check it responds. Each returns (H, W) or
# (frames, H, W) float arrays in [0, 1].

def flash(shape, value: float = 1.0) -> np.ndarray:
    return np.full(shape, float(value), dtype=np.float32)


def grating(shape, period_px: float = 26.0, phase: float = 0.0,
            angle_deg: float = 0.0) -> np.ndarray:
    """Square-wave grating. One period per 2 columns at the default period."""
    h, w = shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    t = np.radians(angle_deg)
    proj = x * np.cos(t) + y * np.sin(t)
    return (np.sin(2 * np.pi * proj / period_px + phase) > 0).astype(np.float32)


def drifting_grating(shape, frames: int = 16, period_px: float = 26.0,
                     angle_deg: float = 0.0, speed_px: float = 3.0):
    """A grating that moves -- the classic optomotor stimulus."""
    return np.stack([
        grating(shape, period_px, 2 * np.pi * f * speed_px / period_px, angle_deg)
        for f in range(frames)
    ])


def disc(shape, radius_px: float, centre=None, value: float = 0.0,
         background: float = 1.0) -> np.ndarray:
    h, w = shape
    cy, cx = (h / 2, w / 2) if centre is None else centre
    y, x = np.mgrid[0:h, 0:w]
    inside = (y - cy) ** 2 + (x - cx) ** 2 <= radius_px ** 2
    return np.where(inside, value, background).astype(np.float32)


def looming(shape, frames: int = 16, start_px: float = 4.0,
            end_px: float | None = None) -> np.ndarray:
    """A dark disc expanding on a light field: the escape stimulus.

    The thing DNp01, the giant fibre, exists to respond to. Growth is
    exponential, as it is for an object approaching at constant speed.

    `end_px` defaults to 70% of the frame's half-width, so the disc ends up
    filling most of the eye whatever the lattice size -- a fixed radius makes
    the stimulus weak or clipped depending on the frame.
    """
    if end_px is None:
        end_px = 0.7 * min(shape) / 2
    radii = start_px * (end_px / start_px) ** np.linspace(0, 1, frames)
    return np.stack([disc(shape, r) for r in radii])
