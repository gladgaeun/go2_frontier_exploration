# Occupancy-grid mapper for the simulator, using the simulator's exact pose.
#
# Why not slam_toolbox in Habitat: odometry is exact, so SLAM has nothing to correct, and
# slam_toolbox's map->odom TF fought with the bridge's static map->odom. When slam_toolbox
# stopped publishing it (robot standing still), Nav2 could no longer look up map->base_link
# ("extrapolation into the future ... latest data is at time X") and the robot froze or spun.
# This mapper only publishes /map; map->odom stays the static identity from the bridge.
#
# Pure numpy, no ROS imports, so it can be unit-tested anywhere.

import numpy as np


class GridMapper:
    UNKNOWN, FREE, OCC = -1, 0, 100

    def __init__(self, resolution=0.05, size_m=60.0, l_hit=0.85, l_miss=-0.4,
                 l_min=-2.0, l_max=3.5, occ_thresh=0.7, free_thresh=-0.35,
                 no_return_range=3.0, sticky_occupied=True):
        self.res = float(resolution)
        self.n = int(round(size_m / self.res))
        self.origin = -0.5 * self.n * self.res          # grid centred on the start pose
        self.logodds = np.zeros((self.n, self.n), dtype=np.float32)   # [row=y, col=x]
        self.l_hit, self.l_miss = l_hit, l_miss
        self.l_min, self.l_max = l_min, l_max
        self.occ_thresh, self.free_thresh = occ_thresh, free_thresh
        # Beams without a return (inf: beyond max range, or a hole in the HM3D mesh) clear
        # free space only up to this distance, so mesh holes do not open fake corridors.
        self.no_return_range = no_return_range
        # The scene is static and the pose exact, so a cell once seen as occupied stays
        # occupied. Without this, beams grazing a wall (robot driving along it) pass through
        # wall cells that the discretised ray clips, mark them free and open 1-2 cell gaps in
        # the wall outline. Free cells then touch the unknown space behind the wall, and those
        # gaps become frontiers that can never be closed: the robot drives there, the frontier
        # stays, and exploration goes back and forth forever.
        self.sticky_occupied = sticky_occupied
        self.touched = False

    def _cells(self, x, y):
        ix = np.floor((x - self.origin) / self.res).astype(np.int64)
        iy = np.floor((y - self.origin) / self.res).astype(np.int64)
        ok = (ix >= 0) & (ix < self.n) & (iy >= 0) & (iy < self.n)
        return iy[ok] * self.n + ix[ok]

    def integrate(self, px, py, yaw, ranges, angle_min, angle_inc, range_max):
        r = np.asarray(ranges, dtype=np.float64)
        ang = yaw + angle_min + np.arange(r.size) * angle_inc
        hit = np.isfinite(r) & (r < range_max)
        free_len = np.where(hit, r, self.no_return_range)
        free_len = np.minimum(free_len, range_max)

        step = 0.5 * self.res
        t = np.arange(0.0, free_len.max() + step, step)                  # (S,)
        mask = t[None, :] < (free_len[:, None] - 0.5 * self.res)          # up to the hit cell (it is removed below)
        c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
        fx = (px + c * t[None, :])[mask]
        fy = (py + s * t[None, :])[mask]
        free = np.unique(self._cells(fx, fy))
        occ = np.unique(self._cells(px + r[hit] * np.cos(ang[hit]),
                                    py + r[hit] * np.sin(ang[hit])))
        free = np.setdiff1d(free, occ, assume_unique=True)

        flat = self.logodds.reshape(-1)
        if self.sticky_occupied:
            free = free[flat[free] < self.occ_thresh]
        flat[free] += self.l_miss
        flat[occ] += self.l_hit
        np.clip(flat, self.l_min, self.l_max, out=flat)
        self.touched = True

    def grid(self, pad_cells=20):
        """Return (data int8 row-major, width, height, origin_x, origin_y) cropped to the known area."""
        known = self.logodds != 0.0
        if not known.any():
            return None
        rows = np.where(known.any(axis=1))[0]
        cols = np.where(known.any(axis=0))[0]
        r0 = max(rows[0] - pad_cells, 0)
        r1 = min(rows[-1] + pad_cells + 1, self.n)
        c0 = max(cols[0] - pad_cells, 0)
        c1 = min(cols[-1] + pad_cells + 1, self.n)
        lo = self.logodds[r0:r1, c0:c1]
        out = np.full(lo.shape, self.UNKNOWN, dtype=np.int8)
        out[lo <= self.free_thresh] = self.FREE
        out[lo >= self.occ_thresh] = self.OCC
        return (out.reshape(-1), c1 - c0, r1 - r0,
                self.origin + c0 * self.res, self.origin + r0 * self.res)
