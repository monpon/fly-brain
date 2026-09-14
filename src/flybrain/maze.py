"""Generate mazes as matrices and turn them into MuJoCo geometry.

Two representations, and the conversion between them is the whole module:

  1. An **occupancy matrix** of shape (2R+1, 2C+1), dtype bool. True is wall.
     Odd indices are corridor cells, even indices are the walls between them,
     which is the standard trick for representing a maze as a single grid: a
     maze of R x C rooms needs R+1 x C+1 walls, so you interleave them.

  2. **Box geoms** in the MuJoCo world. Units are millimetres. The fly is about
     2.8 mm across, so a 10 mm corridor is roughly three fly-widths.

Walls are emitted as merged horizontal runs rather than one box per cell. A
21x21 maze has ~200 occupied cells but only ~60 runs, which means fewer geoms
for the collision detector to sweep.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


# Collision masks. FlyGym gives every geom contype=0/conaffinity=0 and then
# emits 55 explicit <pair> elements, one per body segment against the ground
# plane -- so by default the fly collides with the floor and with nothing
# else. Walls are (contype 1, affinity 2) and the fly's collision proxies
# (see body.add_wall_proxies) are (contype 2, affinity 1), which makes
# fly-wall pairs match while leaving fly-fly and wall-wall pairs unmatched.
WALL_CONTYPE, WALL_CONAFFINITY = 1, 2


@dataclass(frozen=True)
class MazeSpec:
    """Physical dimensions. MuJoCo units here are millimetres."""

    cell_mm: float = 10.0    # corridor width (fly is ~2.8 mm wide)
    wall_mm: float = 1.0     # wall thickness
    height_mm: float = 5.0   # wall height
    rgba: tuple = (0.45, 0.45, 0.5, 1.0)


def generate(rows: int, cols: int, *, seed: int | None = None) -> np.ndarray:
    """Carve a perfect maze with a randomised depth-first search.

    "Perfect" means exactly one path between any two cells: no loops, no
    isolated regions. Returns the (2R+1, 2C+1) occupancy matrix.
    """
    rng = np.random.default_rng(seed)
    grid = np.ones((2 * rows + 1, 2 * cols + 1), dtype=bool)

    def cell(r: int, c: int) -> tuple[int, int]:
        return 2 * r + 1, 2 * c + 1

    start = (0, 0)
    grid[cell(*start)] = False
    stack = [start]

    while stack:
        r, c = stack[-1]
        neighbours = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[cell(nr, nc)]:
                neighbours.append((nr, nc, dr, dc))

        if not neighbours:
            stack.pop()
            continue

        nr, nc, dr, dc = neighbours[rng.integers(len(neighbours))]
        wr, wc = cell(r, c)
        grid[wr + dr, wc + dc] = False   # knock out the wall between them
        grid[cell(nr, nc)] = False
        stack.append((nr, nc))

    return grid


def open_ends(grid: np.ndarray) -> np.ndarray:
    """Open the top-left and bottom-right border walls as entrance and exit."""
    grid = grid.copy()
    grid[1, 0] = False
    grid[-2, -1] = False
    return grid


def _axis_offsets(n: int, spec: MazeSpec) -> np.ndarray:
    """World-space (start, size) for each grid index along one axis.

    Even indices are walls (thin), odd indices are corridors (wide), so the
    grid is *not* uniformly spaced -- that is what keeps walls from being as
    thick as the corridors they separate.
    """
    sizes = np.where(np.arange(n) % 2 == 0, spec.wall_mm, spec.cell_mm)
    starts = np.concatenate([[0.0], np.cumsum(sizes)[:-1]])
    return np.stack([starts, sizes], axis=1)


def extent(grid: np.ndarray, spec: MazeSpec) -> tuple[float, float]:
    """Overall (width, depth) of the maze in mm."""
    rows = _axis_offsets(grid.shape[0], spec)
    cols = _axis_offsets(grid.shape[1], spec)
    return float(cols[:, 1].sum()), float(rows[:, 1].sum())


def wall_runs(grid: np.ndarray) -> list[tuple[int, int, int]]:
    """Merge each row's occupied cells into (row, col_start, col_end) runs."""
    runs = []
    for r in range(grid.shape[0]):
        c = 0
        while c < grid.shape[1]:
            if not grid[r, c]:
                c += 1
                continue
            start = c
            while c < grid.shape[1] and grid[r, c]:
                c += 1
            runs.append((r, start, c - 1))
    return runs


def add_to_world(world, grid: np.ndarray, spec: MazeSpec | None = None) -> dict:
    """Attach maze walls to a FlyGym world as box geoms.

    MuJoCo boxes are defined by *half*-extents and a centre position, which is
    the usual place to get the arithmetic wrong.
    """
    spec = spec or MazeSpec()
    rows = _axis_offsets(grid.shape[0], spec)
    cols = _axis_offsets(grid.shape[1], spec)

    runs = wall_runs(grid)
    for i, (r, c0, c1) in enumerate(runs):
        y0, dy = rows[r]
        x0 = cols[c0, 0]
        x1 = cols[c1, 0] + cols[c1, 1]

        world.mjcf_root.worldbody.add_geom(
            name=f"maze_wall_{i}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[float(x1 - x0) / 2, float(dy) / 2, spec.height_mm / 2],
            pos=[float(x0 + x1) / 2, float(y0 + dy / 2), spec.height_mm / 2],
            rgba=list(spec.rgba),
            contype=WALL_CONTYPE,
            conaffinity=WALL_CONAFFINITY,
        )

    width, depth = extent(grid, spec)
    return {"n_geoms": len(runs), "n_cells": int(grid.sum()),
            "width_mm": width, "depth_mm": depth}


def corridor_centre(grid: np.ndarray, r: int, c: int, spec: MazeSpec) -> tuple:
    """World (x, y) at the centre of corridor cell (r, c), zero-indexed."""
    rows = _axis_offsets(grid.shape[0], spec)
    cols = _axis_offsets(grid.shape[1], spec)
    gr, gc = 2 * r + 1, 2 * c + 1
    return (float(cols[gc, 0] + cols[gc, 1] / 2),
            float(rows[gr, 0] + rows[gr, 1] / 2))


def as_text(grid: np.ndarray) -> str:
    """Render the matrix as ASCII, for looking at it without a renderer."""
    return "\n".join(
        "".join("##" if v else "  " for v in row) for row in grid
    )


def plot_run(grid: np.ndarray, spec: MazeSpec, trail, goal, out,
             field=None, res_mm: float = 1.0) -> str:
    """Draw the maze, the odour field and the path the fly actually took."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    width, depth = extent(grid, spec)
    fig, ax = plt.subplots(figsize=(6, 6))

    if field is not None:
        ax.imshow(field, origin="lower", extent=(0, width, 0, depth),
                  cmap="YlGn", vmin=0, vmax=float(np.nanmax(field)))

    rows = _axis_offsets(grid.shape[0], spec)
    cols = _axis_offsets(grid.shape[1], spec)
    for r, c0, c1 in wall_runs(grid):
        y0, dy = rows[r]
        x0 = cols[c0, 0]
        x1 = cols[c1, 0] + cols[c1, 1]
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, dy,
                                   facecolor="0.35", edgecolor="none"))

    trail = np.asarray(trail)
    if len(trail) > 1:
        ax.plot(trail[:, 0], trail[:, 1], color="#d1495b", lw=1.4, zorder=3)
        ax.plot(*trail[0], "o", color="#1d3557", ms=6, zorder=4, label="start")
    ax.plot(goal[0], goal[1], "*", color="#2a9d8f", ms=18, zorder=4,
            label="odour source")

    ax.set_xlim(0, width)
    ax.set_ylim(0, depth)
    ax.set_aspect("equal")
    ax.set_xlabel("mm")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return str(out)
