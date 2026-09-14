"""Odour that has to travel down corridors, like the fly does.

A point source with an exponential falloff (`olfaction.OdorSource`) is fine in
an open arena and badly wrong in a maze: it leaks through walls, so the
gradient points *at* the source through solid geometry and the fly walks into
a wall and stays there. In still air inside a labyrinth, odour spreads along
the open space. The concentration at a point falls off with the length of the
shortest **open path** back to the source, not the straight-line distance.

So the field is a geodesic:

    c(p) = strength * exp(-d_geo(p, source) / decay_mm)

where `d_geo` is a Dijkstra distance over a raster of the free space. Follow
the gradient and you are led around corners, because around the corner is
genuinely where the smell is stronger. The fly still only ever reads two
numbers -- the concentration at each antenna -- and still has no idea where
the source is or what the maze looks like.

The free space is eroded by `clearance_mm` before the distance is computed, so
the field is undefined in the band a body-width from each wall. That is a
convenience, not biology: it keeps the ridge of the gradient down the middle
of the corridor instead of hugging the inside of every corner.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from . import maze as M


def occupancy_raster(grid: np.ndarray, spec: M.MazeSpec,
                     res_mm: float) -> tuple[np.ndarray, tuple[float, float]]:
    """Rasterise the maze to a uniform (ny, nx) bool array. True is wall."""
    rows = M._axis_offsets(grid.shape[0], spec)
    cols = M._axis_offsets(grid.shape[1], spec)
    width, depth = M.extent(grid, spec)

    nx = int(np.ceil(width / res_mm))
    ny = int(np.ceil(depth / res_mm))
    xs = (np.arange(nx) + 0.5) * res_mm
    ys = (np.arange(ny) + 0.5) * res_mm

    # Which maze grid index each raster centre falls in.
    col_edges = np.append(cols[:, 0], cols[-1, 0] + cols[-1, 1])
    row_edges = np.append(rows[:, 0], rows[-1, 0] + rows[-1, 1])
    ci = np.clip(np.searchsorted(col_edges, xs, "right") - 1, 0, grid.shape[1] - 1)
    ri = np.clip(np.searchsorted(row_edges, ys, "right") - 1, 0, grid.shape[0] - 1)
    return grid[np.ix_(ri, ci)], (res_mm, res_mm)


def _erode(wall: np.ndarray, cells: int) -> np.ndarray:
    """Thicken walls by `cells`, so the free space is what a body fits in."""
    if cells <= 0:
        return wall
    out = wall.copy()
    for _ in range(cells):
        padded = np.pad(out, 1, constant_values=True)
        out = (padded[:-2, 1:-1] | padded[2:, 1:-1] |
               padded[1:-1, :-2] | padded[1:-1, 2:] | out)
    return out


# 8-connected, with true diagonal cost. 4-connectivity would make the field
# manhattan and put spurious corners in the gradient.
_NEIGHBOURS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
               (-1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)),
               (1, -1, np.sqrt(2)), (1, 1, np.sqrt(2))]


def geodesic(wall: np.ndarray, start_ij: tuple[int, int],
             res_mm: float) -> np.ndarray:
    """Shortest open-path distance in mm from `start_ij` to every free cell."""
    ny, nx = wall.shape
    dist = np.full(wall.shape, np.inf)
    i0, j0 = start_ij
    if wall[i0, j0]:
        # The source sits inside the eroded band: fall back to the nearest
        # free cell rather than returning an empty field.
        free = np.argwhere(~wall)
        if len(free) == 0:
            raise SystemExit("no free space in the maze raster")
        i0, j0 = free[np.argmin(np.hypot(free[:, 0] - i0, free[:, 1] - j0))]

    dist[i0, j0] = 0.0
    queue = [(0.0, i0, j0)]
    while queue:
        d, i, j = heapq.heappop(queue)
        if d > dist[i, j]:
            continue
        for di, dj, step in _NEIGHBOURS:
            ni, nj = i + di, j + dj
            if not (0 <= ni < ny and 0 <= nj < nx) or wall[ni, nj]:
                continue
            nd = d + step * res_mm
            if nd < dist[ni, nj]:
                dist[ni, nj] = nd
                heapq.heappush(queue, (nd, ni, nj))
    return dist


def _fill_clearance(dist: np.ndarray, wall: np.ndarray,
                    res_mm: float) -> np.ndarray:
    """Extend the field into the band eroded away next to each wall.

    Without this the fly reads "no odour at all" whenever it brushes a wall,
    and a controller whose only response to that is to start casting will
    cast every time it takes a corner.
    """
    out = dist.copy()
    todo = ~np.isfinite(out) & ~wall
    for _ in range(int(np.ceil(6.0 / res_mm))):
        if not todo.any():
            break
        padded = np.pad(out, 1, constant_values=np.inf)
        best = np.minimum.reduce([
            padded[:-2, 1:-1], padded[2:, 1:-1],
            padded[1:-1, :-2], padded[1:-1, 2:],
        ]) + res_mm
        out = np.where(todo & np.isfinite(best), best, out)
        todo = ~np.isfinite(out) & ~wall
    return out


@dataclass
class MazeOdorSource:
    """Duck-compatible with `olfaction.OdorSource`, but obeys the walls."""

    position: np.ndarray
    distance: np.ndarray          # geodesic distance field, mm, inf in walls
    res_mm: float
    strength: float = 1.0
    decay_mm: float = 25.0
    odor: str = "vinegar"

    @classmethod
    def build(cls, grid: np.ndarray, spec: M.MazeSpec, position,
              *, res_mm: float = 1.0, clearance_mm: float = 2.0,
              **kwargs) -> "MazeOdorSource":
        wall, _ = occupancy_raster(grid, spec, res_mm)
        wall = _erode(wall, int(round(clearance_mm / res_mm)))
        j = int(np.clip(position[0] / res_mm, 0, wall.shape[1] - 1))
        i = int(np.clip(position[1] / res_mm, 0, wall.shape[0] - 1))
        raw, _ = occupancy_raster(grid, spec, res_mm)
        dist = _fill_clearance(geodesic(wall, (i, j), res_mm), raw, res_mm)
        return cls(position=np.asarray(position, dtype=float),
                   distance=dist, res_mm=res_mm, **kwargs)

    def reachable(self, point) -> bool:
        i = int(np.clip(point[1] / self.res_mm, 0, self.distance.shape[0] - 1))
        j = int(np.clip(point[0] / self.res_mm, 0, self.distance.shape[1] - 1))
        return bool(np.isfinite(self.distance[i, j]))

    def path_length(self, point) -> float:
        i = int(np.clip(point[1] / self.res_mm, 0, self.distance.shape[0] - 1))
        j = int(np.clip(point[0] / self.res_mm, 0, self.distance.shape[1] - 1))
        return float(self.distance[i, j])

    def concentration(self, points: np.ndarray) -> np.ndarray:
        """Concentration at world positions, bilinear in the distance field.

        The interpolation is on the *distance*, not on the concentration,
        because the exponential is steep enough that averaging it across a
        1 mm cell smears the gradient the antennae are trying to read.
        """
        points = np.atleast_2d(np.asarray(points, dtype=float))
        ny, nx = self.distance.shape
        gx = np.clip(points[:, 0] / self.res_mm - 0.5, 0, nx - 1)
        gy = np.clip(points[:, 1] / self.res_mm - 0.5, 0, ny - 1)
        x0, y0 = np.floor(gx).astype(int), np.floor(gy).astype(int)
        x1, y1 = np.minimum(x0 + 1, nx - 1), np.minimum(y0 + 1, ny - 1)
        fx, fy = gx - x0, gy - y0

        d = (self.distance[y0, x0] * (1 - fx) * (1 - fy)
             + self.distance[y0, x1] * fx * (1 - fy)
             + self.distance[y1, x0] * (1 - fx) * fy
             + self.distance[y1, x1] * fx * fy)
        # A point inside the eroded band has infinite distance in some of its
        # corners. Fall back to the nearest finite one rather than reporting
        # no smell at all, which would look to the fly like leaving the maze.
        bad = ~np.isfinite(d)
        if bad.any():
            corners = np.stack([self.distance[y0, x0], self.distance[y0, x1],
                                self.distance[y1, x0], self.distance[y1, x1]])
            d = np.where(bad, np.nanmin(np.where(np.isfinite(corners),
                                                 corners, np.nan), axis=0), d)
            d = np.where(np.isfinite(d), d, np.inf)
        return self.strength * np.exp(-d / self.decay_mm)
