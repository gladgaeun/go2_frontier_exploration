# 3D voxel map with occupancy and traversability from RGB-D frames (TravExplorer, Sec. III-A/B).
#
#   RGB-D frame
#    +- depth points -> occupancy log-odds O ------> inflate (O \ T) -> O_inf --+
#    +- depth points -> 2D ray casting -----> M_g --+                          |
#    +- RGB mask + depth --------------------> M_s --+-> T (by mode) -- \ O_inf -> M_trav
#
# Layers: the floor under the robot is at z = z_r. The support layer k_r is the voxel layer
# that contains the floor surface; traversable voxels (M_g, M_s) are support-surface voxels the
# robot can stand on, and the robot body occupies the layers above them. The grid is aligned so
# that the start floor (z = z_ref) is at the centre of a layer.
#
# Geometric traversability M_g (Sec. III-A): hit points with h_min < p_z - z_r < h_max are
# projected onto the support plane and a 2-D ray is cast from each to the robot support state
# x_r. Every support voxel on a ray goes into M_g, except occupied ones. Support voxels contain
# the floor surface and are occupied by the floor points themselves, so "occupied" is read as
# an occupied voxel standing on the support voxel at collision height (an obstacle on the floor).
# M_g only grows; obstacles reaching into the robot body are removed by O_inf (eq. 1).
#
# Collision height: a voxel dz layers above a support voxel is an obstacle for a robot standing
# there if its centre height dz * res lies above h_min (the paper's h_min < p_z - z_r), i.e.
# dz >= dz_lo; up to n_h = ceil(h_max / res). Lower voxels (door thresholds, rug edges, low
# steps) are passable: they neither block M_g nor inflate.
#
# Inflation kernel (anisotropic, downward only): an occupied, non-traversable voxel at layer j
# blocks the support voxels within robot_radius in xy at layers j-n_h .. j-dz_lo, i.e. every
# position where it would be inside the robot body. Floor voxels therefore never block their
# own layer, while walls and table tops block the floor beneath and around them.
#
# Inflation is incremental: only the bounding box of the voxels whose occupancy or
# traversability changed in this frame (C_t, eq. 2) is re-inflated, from the sources around it
# (eq. 3). The result is identical to inflating the whole map.
#
# Semantic traversability M_s (Sec. III-A): voxels of the depth points whose pixels a segmentation
# labels robot-traversable (floor, stair, rug) go into M_s; the stair voxels are also kept apart
# (`stairs`) for the mode switch (semantic.ModeSwitch) and later the stair landmarks. M_s only
# grows, like M_g. In mode 's', T = M_s (eq. 1): occupied voxels in M_s (stair steps) are support
# surfaces and are not inflated, so stairs, which 2-D ray casting treats as obstacles, can become
# traversable.
#
# Stuck handling (eq. 17): when a forward motion does not move the robot as commanded, it is
# blocked by something the map does not show (e.g. a hole in an HM3D mesh wall that the depth
# camera looks through). The voxels in front of the robot are then marked as blocked: occupied
# for good, whatever later rays see, so the traversable map closes the gap and the planner
# replans around it.
#
# Pure numpy, no ROS imports, so it can be unit-tested anywhere.

import math
from collections import deque

import numpy as np


def traverse_2d(start, end):
    """All grid cells crossed by the segments start[i] -> end[i] (grid units, cell = floor).

    Exact traversal: the segment parameters where it crosses a vertical or horizontal grid line
    split it into pieces, and each piece lies in one cell (the cell of its midpoint). Sampling
    the segment at a fixed step instead skips cells where it cuts a cell corner.
    """
    start = np.asarray(start, dtype=np.float64).reshape(-1, 2)
    end = np.asarray(end, dtype=np.float64).reshape(-1, 2)
    d = end - start
    n = int(np.ceil(np.abs(d).max())) + 1 if len(d) else 0
    ts = [np.zeros((len(d), 1)), np.ones((len(d), 1))]
    for ax in (0, 1):
        step = np.sign(d[:, ax])
        first = np.where(step > 0, np.floor(start[:, ax]) + 1.0, np.ceil(start[:, ax]) - 1.0)
        first = np.where(np.isclose(start[:, ax], np.round(start[:, ax])) & (step < 0),
                         np.round(start[:, ax]), first)
        lines = first[:, None] + step[:, None] * np.arange(n)[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (lines - start[:, ax, None]) / d[:, ax, None]
        ts.append(np.where((step[:, None] != 0) & (t > 0) & (t < 1), t, np.inf))
    t = np.sort(np.concatenate(ts, axis=1), axis=1)
    with np.errstate(invalid='ignore'):                    # inf - inf past the last crossing
        mid = 0.5 * (t[:, :-1] + t[:, 1:])
        ok = np.isfinite(mid) & (t[:, 1:] - t[:, :-1] > 1e-12)
    pts = start[:, None, :] + d[:, None, :] * np.where(ok, mid, 0.0)[:, :, None]
    cells = np.floor(pts[ok]).astype(np.int64)
    return np.concatenate([np.floor(start).astype(np.int64), np.floor(end).astype(np.int64), cells])


class VoxelMapper:
    UNKNOWN, FREE, OCC = -1, 0, 100

    def __init__(self, resolution=0.05, size_xy=30.0, z_below=0.25, z_above=2.0,
                 l_hit=0.85, l_miss=-0.4, l_min=-2.0, l_max=3.5, l_occ=0.7,
                 robot_radius=0.25, h_min=0.1, h_max=0.6, mode='g', z_ref=0.0):
        self.res = float(resolution)
        self.nx = self.ny = int(round(size_xy / self.res))
        self.origin_xy = -0.5 * self.nx * self.res            # grid centred on the start pose
        k0 = int(round(z_below / self.res))                    # layer of the start floor (z = z_ref)
        self.nz = k0 + int(round(z_above / self.res)) + 1
        self.origin_z = z_ref - (k0 + 0.5) * self.res
        shape = (self.nx, self.ny, self.nz)
        self.logodds = np.zeros(shape, dtype=np.float32)
        self.trav_g = np.zeros(shape, dtype=bool)              # M_g: 2D ray casting
        self.trav_s = np.zeros(shape, dtype=bool)              # M_s: semantic segmentation
        self.stairs = np.zeros(shape, dtype=bool)              # stair-class voxels (part of M_s)
        self.stair_seen = np.zeros(shape, dtype=bool)          # every stair-labelled voxel, trusted or not
        self.stair_xy = np.zeros((0, 2))                       # their centres in xy, for distances
        self.stair_z = np.zeros(0)                             # and their heights
        self.inflated = np.zeros(shape, dtype=bool)            # O_inf
        self.blocked = np.zeros(shape, dtype=bool)             # v_front of stuck events, always occupied
        self._mark = np.zeros(self.logodds.size, dtype=bool)  # scratch for _unique, always all False
        # 2-D cell sets per support layer (several floors / stair steps share xy cells):
        # layer -> (nx, ny) bool. Read with a tolerance of layer_tol layers (z_r noise).
        self._visited = {}          # xy cells the robot centre passed
        self._tight = {}            # too narrow for the real body (mark_tight)
        self.layer_tol = 2
        self.grid_layers = 1          # trav_grid: layers within +-1 of z_r (one floor, +-5 cm)
        self.k_ref = k0             # start floor layer (default for the calls without z_r)
        self.last_visit = None
        self.l_hit, self.l_miss = l_hit, l_miss
        self.l_min, self.l_max, self.l_occ = l_min, l_max, l_occ
        # Points with h_min < z - z_r < h_max can collide with the robot (2D ray casting).
        self.h_min, self.h_max = h_min, h_max
        self.n_h = max(1, int(np.ceil(h_max / self.res)))
        self.dz_lo = min(int(np.floor(h_min / self.res + 1e-9)) + 1, self.n_h)   # 0.1 / 0.1 -> 2
        self.robot_radius = robot_radius
        r = robot_radius / self.res
        ri = int(np.floor(r))
        self.r_cells = ri
        self.disk = [(dx, dy) for dx in range(-ri, ri + 1) for dy in range(-ri, ri + 1)
                     if dx * dx + dy * dy <= r * r]
        if mode not in ('g', 's'):
            raise ValueError(f"mode must be 'g' or 's', got {mode!r}")
        self.mode = mode
        # B_m: AABB (voxel index, [lo, hi)) of the region where M_trav may have changed since
        # the last pop_update_box(); frontiers are updated only around it (Sec. III-C)
        self.update_box = None

    # -- indexing ---------------------------------------------------------------
    def _ijk(self, p):
        """World points (N,3) -> integer voxel indices (N,3) and an in-grid mask."""
        p = np.asarray(p, dtype=np.float64).reshape(-1, 3)
        ijk = np.empty(p.shape, dtype=np.int64)
        ijk[:, :2] = np.floor((p[:, :2] - self.origin_xy) / self.res)
        ijk[:, 2] = np.floor((p[:, 2] - self.origin_z) / self.res)
        ok = ((ijk[:, 0] >= 0) & (ijk[:, 0] < self.nx) & (ijk[:, 1] >= 0) & (ijk[:, 1] < self.ny)
              & (ijk[:, 2] >= 0) & (ijk[:, 2] < self.nz))
        return ijk, ok

    def _flat(self, p):
        ijk, ok = self._ijk(p)
        ijk = ijk[ok]
        return self._unique((ijk[:, 0] * self.ny + ijk[:, 1]) * self.nz + ijk[:, 2])

    def _unique(self, idx):
        """np.unique for voxel indices without sorting (millions of ray samples per frame)."""
        if idx.size < 50000:
            return np.unique(idx)
        self._mark[idx] = True
        out = np.flatnonzero(self._mark)
        self._mark[out] = False
        return out

    def layer(self, z):
        return int(np.floor((z - self.origin_z) / self.res))

    def centers(self, flat_idx):
        ix, iy, iz = np.unravel_index(flat_idx, self.logodds.shape)
        return np.stack([self.origin_xy + (ix + 0.5) * self.res,
                         self.origin_xy + (iy + 0.5) * self.res,
                         self.origin_z + (iz + 0.5) * self.res], axis=1)

    @property
    def occupied(self):
        return (self.logodds >= self.l_occ) | self.blocked

    def _occupied_at(self, flat_idx):
        return (self.logodds.reshape(-1)[flat_idx] >= self.l_occ) | self.blocked.reshape(-1)[flat_idx]

    def _source(self, box=None):
        """Traversability source T of the mapping mode (eq. 1), optionally only inside `box`.

        Mode 'g': M_g. Mode 's': M_g U M_s, not M_s alone (the paper selects one source): M_s
        holds only floor that was segmented within semantic range, so switching to it at the stairs
        opened holes in the finished floor, each one a frontier the robot went back for (and lost
        again on the way, back in mode g). M_s adds the stairs. M_g is not updated on / while
        climbing the stairs (there the support plane is in mid-air, and 2-D ray casting filled it);
        on a floor it is, also in mode s (integrate(update_g=...)), and going back to g the floor
        of M_s joins it (set_mode): the modes differ only in the stairs. M_g leaves out support
        voxels seen free (_void), e.g. the stairwell next to the upper floor."""
        sl = Ellipsis if box is None else box
        if self.mode == 'g':
            return self.trav_g[sl]
        return self.trav_g[sl] | self.trav_s[sl]

    # -- update -----------------------------------------------------------------
    def integrate(self, origin, points, hit, support, update_g=None):
        """Fuse one depth frame.

        origin  : (3,) camera position in the world frame
        points  : (N,3) ray end points in the world frame
        hit     : (N,) bool, False for rays truncated at max range (they only clear space)
        support : (3,) robot support point x_r (bottom centre of the robot)
        update_g: ray-cast M_g in this frame (default: in mode 'g' only). The node also does it in
                  mode 's' while the robot is on a floor (not on / climbing the stairs).
        Returns the number of changed voxels |C_t|.
        """
        o = np.asarray(origin, dtype=np.float64).reshape(3)
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        hit = np.asarray(hit, dtype=bool).reshape(-1)
        if pts.shape[0] == 0:
            return 0
        flat = self.logodds.reshape(-1)

        # 1. occupancy, standard log-odds: misses along each ray up to the end voxel, a hit there
        #    (a voxel hit by any ray of this frame is not also counted as a miss)
        d = pts - o
        length = np.linalg.norm(d, axis=1)
        step = 0.5 * self.res
        stop = np.where(hit, length - 0.5 * self.res, length)
        u = d / np.maximum(length, 1e-9)[:, None]
        # samples t = 0, step, 2 step, ... < stop of every ray, built per ray (no padding)
        count = np.maximum(np.ceil(stop / step), 0).astype(np.int64)
        ray = np.repeat(np.arange(len(count)), count)
        first = np.cumsum(count) - count
        t = (np.arange(ray.size) - first[ray]) * step
        samples = o[None, :] + u[ray] * t[:, None]
        occ = self._flat(pts[hit])
        free = np.setdiff1d(self._flat(samples), occ, assume_unique=True)
        touched = np.concatenate([free, occ])
        before = self._occupied_at(touched)
        flat[free] += self.l_miss
        flat[occ] += self.l_hit
        flat[touched] = np.clip(flat[touched], self.l_min, self.l_max)
        changed_occ = touched[before != self._occupied_at(touched)]

        # 2. geometric traversability M_g (not while on the stairs, see _source)
        if update_g is None:
            update_g = self.mode == 'g'
        changed_g = self._raycast_2d(pts[hit], support) if update_g else np.zeros(0, dtype=np.int64)
        # M_g voxels the camera has since seen through (no floor there) are dropped
        g = self.trav_g.reshape(-1)
        cand = free[g[free]]
        void = cand[self._void(cand)]
        g[void] = False

        changed = np.concatenate([changed_occ, changed_g, void])
        self._update_inflation(changed)
        return int(changed.size)

    def _raycast_2d(self, pts, support):
        """2-D ray casting on the support plane; returns the voxels newly added to M_g."""
        x_r, y_r, z_r = (float(v) for v in support)
        k = self.layer(z_r)
        if not 0 <= k < self.nz - self.dz_lo:
            return np.zeros(0, dtype=np.int64)
        robot = np.array([x_r, y_r])
        rc = np.floor((robot - self.origin_xy) / self.res).astype(np.int64)
        if not (0 <= rc[0] < self.nx and 0 <= rc[1] < self.ny):
            return np.zeros(0, dtype=np.int64)

        # points in the robot collision-height range, projected onto z = z_r (one per cell)
        rel = pts[:, 2] - z_r
        band = pts[(rel > self.h_min) & (rel < self.h_max), :2]
        cells = np.unique(np.floor((band - self.origin_xy) / self.res).astype(np.int64), axis=0)
        cells = cells[(cells[:, 0] >= 0) & (cells[:, 0] < self.nx)
                      & (cells[:, 1] >= 0) & (cells[:, 1] < self.ny)]
        ray_cells = rc[None, :]
        if cells.shape[0]:
            # a 2-D ray from each projected point (cell centre) to x_r: every cell it passes
            start = cells + 0.5                                     # grid units
            end = (robot - self.origin_xy) / self.res
            ray_cells = np.concatenate([ray_cells, traverse_2d(start, np.broadcast_to(end, start.shape))])
        ray_flat = np.unique((ray_cells[:, 0] * self.ny + ray_cells[:, 1]) * self.nz + k)
        # occupied voxels are skipped: an obstacle on the support voxel at collision height
        ray_flat = ray_flat[~self._occupied_at(ray_flat + self.dz_lo)]
        # and cells seen to be empty at floor height: the 2-D rays assume floor between the robot
        # and the obstacle points, over a stairwell they filled the void at the height of the
        # upper floor (00800: a floor strip in mid-air above the stairs, frontiers at its edges
        # the robot tried again and again, driving down the stairs toward them)
        ray_flat = ray_flat[~self._void(ray_flat)]

        g = self.trav_g.reshape(-1)
        added = ray_flat[~g[ray_flat]]
        g[added] = True
        return added

    def _void(self, flat_idx):
        """Support voxels with no floor: they and the two voxels below (z_r is +-1 layer) were
        seen free (log-odds at most 2 l_miss, two net misses). Unknown voxels are not void: 2-D
        ray casting is there to fill the floor the camera did not see."""
        flat_idx = np.asarray(flat_idx, dtype=np.int64)
        lo = self.logodds.reshape(-1)
        th = 2.0 * self.l_miss
        out = flat_idx % self.nz >= 2
        for dz in range(3):
            out &= lo[np.maximum(flat_idx - dz, 0)] <= th
        return out

    def integrate_semantic(self, points, stair_points=None):
        """Insert the voxels of traversable-class pixels (floor, stair, rug, ...) into M_s.

        stair_points (part of points) are also recorded as stair voxels. Returns the number of
        voxels newly added to M_s. In mode 's' the inflation is updated around them (they stop
        being obstacles); in mode 'g' M_s is only collected.
        """
        idx = self._flat(points)
        s = self.trav_s.reshape(-1)
        added = idx[~s[idx]]
        s[added] = True
        if stair_points is not None and len(stair_points):
            sidx = self._flat(stair_points)
            st = self.stairs.reshape(-1)
            new = sidx[~st[sidx]]
            if new.size:
                st[new] = True
                c = self.centers(new)
                self.stair_xy = np.concatenate([self.stair_xy, c[:, :2]])
                self.stair_z = np.concatenate([self.stair_z, c[:, 2]])
        if self.mode == 's':
            self._update_inflation(added)
        return int(added.size)

    def trusted_stairs(self, stair_points, other_points, z_r, window=0.6, column_gap=0.3, below=0.3):
        """Stair points trusted as support surfaces before the robot climbs.

        An open-vocabulary segmentation labels the whole staircase as stairs, with the railing and
        the underside of an upper flight, which would become "traversable" in mode 's'. Kept are
        stair points (a) at most `window` above the current floor z_r (the first steps; the window
        rises with z_r while climbing) and (b) at most `column_gap` above the lowest semantic
        surface in their xy column (this frame's floor / rug / stair points and M_s): a railing
        stands ~0.9 m above the tread below it, a riser only one step (~0.17 m). Dropped points stay
        what the occupancy makes them (obstacles), so the railing still blocks the steps beside it.
        Only surfaces from `below` under z_r up count as the lowest: where a staircase turns, the
        upper flight lies above the lower one, and measured from the lower flight ~1 m beneath,
        every tread of the upper flight was dropped and blocked the climb.
        """
        sp = np.asarray(stair_points, dtype=np.float64).reshape(-1, 3)
        sp = sp[sp[:, 2] - z_r <= window]
        if column_gap is None or not len(sp):
            return sp
        op = np.asarray(other_points, dtype=np.float64).reshape(-1, 3)
        op = op[op[:, 2] >= z_r - below]
        sp = sp[sp[:, 2] >= z_r - below]
        if not len(sp):
            return sp
        allp = np.concatenate([sp, op])
        ij = np.floor((allp[:, :2] - self.origin_xy) / self.res).astype(np.int64)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < self.nx) & (ij[:, 1] >= 0) & (ij[:, 1] < self.ny)
        key = np.where(ok, ij[:, 0] * self.ny + ij[:, 1], -1)
        cols, inv = np.unique(key, return_inverse=True)
        lowest = np.full(len(cols), np.inf)
        np.minimum.at(lowest, inv, allp[:, 2])
        valid = cols >= 0
        ci, cj = cols[valid] // self.ny, cols[valid] % self.ny
        k_lo = max(self.layer(z_r - below), 0)
        col_s = self.trav_s[ci, cj, k_lo:]                   # existing M_s in these columns, near z_r
        has = col_s.any(axis=1)
        z_ms = self.origin_z + (k_lo + np.argmax(col_s, axis=1) + 0.5) * self.res
        lv = lowest[valid]
        lv[has] = np.minimum(lv[has], z_ms[has])
        lowest[valid] = lv
        keep = (sp[:, 2] - lowest[inv[:len(sp)]] <= column_gap) & ok[:len(sp)]
        return sp[keep]

    def support_height(self, x, y, z_base, below=0.35, above=0.05, radius=0.15):
        """Height of the surface the robot stands on, from the map: in each xy column within
        `radius` of (x, y) the highest occupied voxel between z_base - below and z_base + above
        (z_base: the robot base, which in Habitat floats 0.03-0.2 m above the mesh floor), then
        the median over the columns. On a stair the lower and upper steps reach into the patch at
        its edges; the median is the step under the robot. None if nothing is mapped there.
        The surface under the robot was seen before it got there, so the map has it although the
        camera no longer sees it."""
        r = int(np.ceil(radius / self.res))
        ci, cj = int(np.floor((x - self.origin_xy) / self.res)), int(np.floor((y - self.origin_xy) / self.res))
        k0, k1 = max(self.layer(z_base - below), 0), min(self.layer(z_base + above) + 1, self.nz)
        i0, i1, j0, j1 = max(ci - r, 0), min(ci + r + 1, self.nx), max(cj - r, 0), min(cj + r + 1, self.ny)
        if k0 >= k1 or i0 >= i1 or j0 >= j1:
            return None
        di, dj = np.mgrid[i0 - ci:i1 - ci, j0 - cj:j1 - cj]
        disk = di * di + dj * dj <= r * r
        occ = (self.logodds[i0:i1, j0:j1, k0:k1] >= self.l_occ)[disk]        # (columns, layers)
        has = occ.any(axis=1)
        if not has.any():
            return None
        top = k1 - 1 - np.argmax(occ[has, ::-1], axis=1)                    # highest occupied layer
        return float(self.origin_z + (np.median(top) + 0.5) * self.res)

    def remember_stairs(self, stair_points):
        """Record stair-labelled voxels that are not trusted yet (promote_stairs may add them later)."""
        idx = self._flat(stair_points)
        self.stair_seen.reshape(-1)[idx] = True

    def promote_stairs(self, z_r, window=0.6, column_gap=0.3, below=0.3):
        """Trust remembered stair voxels that the stair window covers now that z_r rose.

        The window rises with z_r while climbing, but the camera cannot see the ~0.7 m in front of
        the robot: the last steps and the edge of the upper floor were seen from below (outside
        the window then) and never again, so they stayed obstacles and blocked the top of the
        climb. Same tests as trusted_stairs, on the remembered voxels. Returns the number added.
        """
        k_lo = max(self.layer(z_r - below), 0)
        k_hi = min(self.layer(z_r + window), self.nz - 1)
        if k_lo > k_hi:
            return 0
        slab = self.stair_seen[:, :, k_lo:k_hi + 1] & ~self.trav_s[:, :, k_lo:k_hi + 1]
        if not slab.any():
            return 0
        ii, jj, kk = np.nonzero(slab)
        surf = (self.trav_s[:, :, k_lo:k_hi + 1] | self.stair_seen[:, :, k_lo:k_hi + 1])[ii, jj, :]
        lowest = np.argmax(surf, axis=1)                         # lowest surface layer near z_r
        keep = (kk - lowest) * self.res <= column_gap + 1e-9
        if not keep.any():
            return 0
        flat = np.ravel_multi_index((ii[keep], jj[keep], kk[keep] + k_lo), self.logodds.shape)
        self.integrate_semantic(self.centers(flat), self.centers(flat))
        return int(keep.sum())

    def stair_landmarks(self, min_voxels=150, merge=0.15, part=0.25):
        """Stair landmarks (O_stair, Sec. V-A, simplified): connected groups of stair-labelled
        voxels (trusted or not), merged across gaps up to `merge` m, with at least min_voxels.
        The paper accumulates detections into instances with a confidence; here a staircase is
        reliable once enough of it has been seen. Returns a list of
        (entrance x, y, z, top x, y, z, voxels): entrance = mean xy of the lowest `part` m of the
        group (the foot of the stairs, z its lowest voxel), top = the same for its highest part."""
        from scipy import ndimage
        seen = self.stair_seen | self.stairs
        if not seen.any():
            return []
        nz_ = np.argwhere(seen)
        lo, hi = nz_.min(axis=0), nz_.max(axis=0) + 1
        sub = seen[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        r = max(1, int(round(merge / self.res)))
        grown = ndimage.binary_dilation(sub, structure=np.ones((2 * r + 1,) * 3, dtype=bool))
        labels, n = ndimage.label(grown)
        out = []
        for lab in range(1, n + 1):
            vox = np.argwhere(sub & (labels == lab))
            if len(vox) < min_voxels:
                continue
            c = (vox + lo + 0.5) * self.res + [self.origin_xy, self.origin_xy, self.origin_z]
            low = c[c[:, 2] <= c[:, 2].min() + part]
            high = c[c[:, 2] >= c[:, 2].max() - part]
            out.append((*low[:, :2].mean(axis=0), c[:, 2].min(), *high[:, :2].mean(axis=0), c[:, 2].max(), len(vox)))
        return out

    def stair_distance(self, x, y, z_r=None, dz=1.0):
        """xy distance from (x, y) to the nearest stair voxel (inf if none is mapped); with z_r,
        only stair voxels within dz of z_r count: on the upper floor the staircase below the robot
        is close in xy, and the mode stayed s on the whole upper floor."""
        xy = self.stair_xy
        if z_r is not None and len(self.stair_z) == len(xy):
            xy = xy[np.abs(self.stair_z - z_r) <= dz]
        if not len(xy):
            return float('inf')
        return float(np.sqrt(((xy - (x, y)) ** 2).sum(axis=1).min()))

    def semantic_points(self):
        """Centres of M_s voxels that are not stairs, and of the stair voxels."""
        return (self.centers(np.flatnonzero(self.trav_s & ~self.stairs)),
                self.centers(np.flatnonzero(self.stairs)))

    def _layer_of(self, z_r):
        return self.k_ref if z_r is None else self.layer(z_r)

    def _cells(self, store, k):
        """The 2-D cell set of layer k (created on first use)."""
        if k not in store:
            store[k] = np.zeros((self.nx, self.ny), dtype=bool)
        return store[k]

    def _cells_near(self, store, k):
        """Union of the cell sets of the layers within layer_tol of k."""
        out = np.zeros((self.nx, self.ny), dtype=bool)
        for kk, a in store.items():
            if abs(kk - k) <= self.layer_tol:
                out |= a
        return out

    def visited_at(self, z_r=None):
        return self._cells_near(self._visited, self._layer_of(z_r))

    def tight_at(self, z_r=None):
        return self._cells_near(self._tight, self._layer_of(z_r))

    @property
    def visited(self):
        return self.visited_at()

    @property
    def tight(self):
        return self.tight_at()

    def visit(self, x, y, z_r=None):
        """Record the robot centre at (x, y) on the support plane z_r (default: start floor);
        the segment from the previous visit is filled in."""
        p = (np.array([x, y]) - self.origin_xy) / self.res
        start = p if self.last_visit is None else self.last_visit
        c = traverse_2d(start, p)
        c = c[(c[:, 0] >= 0) & (c[:, 0] < self.nx) & (c[:, 1] >= 0) & (c[:, 1] < self.ny)]
        self._cells(self._visited, self._layer_of(z_r))[c[:, 0], c[:, 1]] = True
        self.last_visit = p

    def near_visited(self, xy, margin_cells=2, z_r=None):
        """True for points (N,2) whose blocked voxel would be inflated to within margin_cells of a
        visited cell. The margin keeps the robot's path open on the Nav2 grid, whose own
        inflation (point footprint, inscribed radius one cell) closes a gap of 1-2 cells."""
        ij = np.floor((np.asarray(xy, dtype=np.float64).reshape(-1, 2) - self.origin_xy) / self.res).astype(int)
        d = self.robot_radius / self.res + margin_cells
        r = int(np.ceil(d))
        di, dj = np.mgrid[-r:r + 1, -r:r + 1]
        win_mask = di * di + dj * dj <= d * d
        out = np.zeros(len(ij), dtype=bool)
        visited = self.visited_at(z_r)
        for n, (i, j) in enumerate(ij):
            i0, j0 = max(i - r, 0), max(j - r, 0)
            win = visited[i0:i + r + 1, j0:j + r + 1]
            m = win_mask[i0 - (i - r):i0 - (i - r) + win.shape[0], j0 - (j - r):j0 - (j - r) + win.shape[1]]
            out[n] = (win & m).any()
        return out

    def mark_blocked(self, xy, z_r):
        """Mark v_front as occupied: the lowest collision layer above the support plane at xy (N,2).

        Returns the number of newly blocked voxels. One voxel is enough to block the support
        voxels within robot_radius of it (inflation goes down to the support layer).
        Points whose block would be inflated over (or next to, see near_visited) cells the robot
        centre has passed through are skipped: the robot came that way, so it is passable, and closing it can trap the robot
        (no way back) when a failure is attributed to the wrong direction.
        """
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        xy = xy[~self.near_visited(xy, z_r=z_r)]
        k = self.layer(z_r) + self.dz_lo
        if not 0 <= k < self.nz:
            return 0
        z = self.origin_z + (k + 0.5) * self.res
        idx = self._flat(np.column_stack([xy, np.full(len(xy), z)]))
        b = self.blocked.reshape(-1)
        new = idx[~self._occupied_at(idx)]
        b[idx] = True
        self._update_inflation(new)
        return int(new.size)

    @staticmethod
    def _grow(cells):
        """Cells grown by one (8-neighbours)."""
        been = cells.copy()
        been[1:, :] |= cells[:-1, :]
        been[:-1, :] |= cells[1:, :]
        been[:, 1:] |= been[:, :-1].copy()
        been[:, :-1] |= been[:, 1:].copy()
        return been

    def visited_layers(self):
        """{layer: visited cells grown by one} of every support layer (SupportGraph.snapshot)."""
        return {k: self._grow(c) for k, c in self._visited.items() if c.any()}

    def tight_layers(self):
        """{layer: cells too narrow for the body} of every support layer."""
        return {k: c.copy() for k, c in self._tight.items() if c.any()}

    def _visited_grown(self, z_r=None):
        """Visited cells of the support plane z_r grown by one cell (Nav2's point footprint)."""
        visited = self.visited_at(z_r)
        been = visited.copy()
        been[1:, :] |= visited[:-1, :]
        been[:-1, :] |= visited[1:, :]
        been[:, 1:] |= been[:, :-1].copy()
        been[:, :-1] |= been[:, 1:].copy()
        return been

    def mark_tight(self, x, y, yaw, z_r, radius, ahead=0.7, half_width=0.5, keep=0.15):
        """Close the cells ahead of the robot where its real body (radius) touches a known obstacle.

        O_inf uses robot_radius, kept below the real radius so doorways stay open; when the
        collision guard stops the robot short of a known obstacle, the cells in front of it
        (0..ahead m, +-half_width) closer than `radius` to an obstacle voxel at collision height
        are taken out of the Nav2 grid (trav_grid), without inflating them again: a gap the
        robot fits through stays open, a narrower one closes. Cells the robot centre has passed
        through stay free, and so do the cells within `keep` m of the robot: after a contact (a
        stuck event) the cells right next to it are close to the obstacle too, and closing them
        made the robot's own cell INSCRIBED on the Nav2 grid, so no plan, spin or backup started
        from it any more. Returns the number of newly closed cells.
        """
        from scipy import ndimage
        k = self.layer(z_r)
        if not 0 <= k < self.nz:
            return 0
        m = int(np.ceil((max(ahead, half_width) + radius) / self.res)) + 1
        ci, cj = int(np.floor((x - self.origin_xy) / self.res)), int(np.floor((y - self.origin_xy) / self.res))
        i0, i1 = max(ci - m, 0), min(ci + m + 1, self.nx)
        j0, j1 = max(cj - m, 0), min(cj + m + 1, self.ny)
        kl, kh = k + self.dz_lo, min(k + self.n_h + 1, self.nz)
        if i0 >= i1 or j0 >= j1 or kl >= kh:
            return 0
        box = (slice(i0, i1), slice(j0, j1), slice(kl, kh))
        obs = (((self.logodds[box] >= self.l_occ) | self.blocked[box]) & ~self._source(box)).any(axis=2)
        if not obs.any():
            return 0
        dist = ndimage.distance_transform_edt(~obs) * self.res
        gx = self.origin_xy + (np.arange(i0, i1) + 0.5) * self.res - x
        gy = self.origin_xy + (np.arange(j0, j1) + 0.5) * self.res - y
        fx = np.cos(yaw) * gx[:, None] + np.sin(yaw) * gy[None, :]
        fy = -np.sin(yaw) * gx[:, None] + np.cos(yaw) * gy[None, :]
        # distances are between voxel centres; the obstacle surface may lie up to half a voxel closer
        new = (dist < radius + 0.5 * self.res) & (fx > 0.0) & (fx < ahead) & (np.abs(fy) < half_width)
        new &= np.hypot(fx, fy) >= keep
        new &= ~self._visited_grown(z_r)[i0:i1, j0:j1] & ~self.tight_at(z_r)[i0:i1, j0:j1]
        self._cells(self._tight, k)[i0:i1, j0:j1] |= new
        return int(new.sum())

    def set_mode(self, mode):
        if mode not in ('g', 's'):
            raise ValueError(f"mode must be 'g' or 's', got {mode!r}")
        if mode != self.mode:
            if mode == 'g':
                # the floor confirmed in mode s stays: M_s (without the stairs) goes into M_g.
                # M_g only fills the floor between the robot and obstacle points, M_s the floor the
                # camera saw; without this, going back to g dropped what mode s had explored
                # (frontier voxels 24 -> 139 at the switch in 00810), holes the robot went back for
                self.trav_g |= self.trav_s & ~self.stairs
            self.mode = mode
            self.reinflate_all()

    # -- inflation --------------------------------------------------------------
    def _dilate(self, src):
        """Dilate a boolean box by the inflation kernel (xy disk, z downward dz_lo..n_h)."""
        nx, ny, nz = src.shape
        xy = np.zeros_like(src)
        for dx, dy in self.disk:
            xs, xd = (slice(0, nx - dx), slice(dx, nx)) if dx >= 0 else (slice(-dx, nx), slice(0, nx + dx))
            ys, yd = (slice(0, ny - dy), slice(dy, ny)) if dy >= 0 else (slice(-dy, ny), slice(0, ny + dy))
            xy[xd, yd, :] |= src[xs, ys, :]
        out = np.zeros_like(src)
        for dz in range(self.dz_lo, min(self.n_h, nz - 1) + 1):
            out[:, :, :nz - dz] |= xy[:, :, dz:]          # source at layer j -> target j - dz
        return out

    def _update_inflation(self, changed):
        """Re-inflate around the changed voxels C_t (eqs. 2-3)."""
        if changed.size == 0:
            return
        ix, iy, iz = np.unravel_index(changed, self.logodds.shape)
        r, h = self.r_cells, self.n_h
        # O_inf can change inside A = bbox(C_t) + kernel; it is recomputed there from the
        # sources in B = A + kernel^-1 (xy +-r, z up to +n_h).
        a0 = (max(ix.min() - r, 0), max(iy.min() - r, 0), max(iz.min() - h, 0))
        a1 = (min(ix.max() + r + 1, self.nx), min(iy.max() + r + 1, self.ny), min(iz.max() + 1, self.nz))
        b0 = (max(a0[0] - r, 0), max(a0[1] - r, 0), a0[2])
        b1 = (min(a1[0] + r, self.nx), min(a1[1] + r, self.ny), min(a1[2] + h, self.nz))
        box = tuple(slice(lo, hi) for lo, hi in zip(b0, b1))
        src = ((self.logodds[box] >= self.l_occ) | self.blocked[box]) & ~self._source(box)
        dil = self._dilate(src)
        inner = tuple(slice(lo - blo, hi - blo) for lo, hi, blo in zip(a0, a1, b0))
        self.inflated[tuple(slice(lo, hi) for lo, hi in zip(a0, a1))] = dil[inner]
        # M_trav = T \ O_inf changes only where T changed (C_t) or O_inf was recomputed (A)
        self._touch(a0, a1)

    def _touch(self, lo, hi):
        if self.update_box is None:
            self.update_box = (tuple(lo), tuple(hi))
        else:
            (l0, h0) = self.update_box
            self.update_box = (tuple(map(min, l0, lo)), tuple(map(max, h0, hi)))

    def pop_update_box(self):
        """Return B_m accumulated since the last call (or None) and reset it."""
        box, self.update_box = self.update_box, None
        return box

    def reinflate_all(self):
        self.inflated = self._dilate(self.occupied & ~self._source())
        self._touch((0, 0, 0), self.logodds.shape)

    # -- outputs ----------------------------------------------------------------
    def traversable(self):
        """M_trav = T \\ O_inf as a boolean voxel array."""
        return self._source() & ~self.inflated

    def blocked_points(self):
        return self.centers(np.flatnonzero(self.blocked))

    def inflated_points(self, z_r):
        """O_inf on the support layer only (the full 3D set extends down below every obstacle)."""
        k = self.layer(z_r)
        if not 0 <= k < self.nz:
            return np.zeros((0, 3))
        ix, iy = np.nonzero(self.inflated[:, :, k])
        return self.centers((ix * self.ny + iy) * self.nz + k)

    def trav_grid(self, z_r, pad_cells=10):
        """2D grid of the support layer: FREE = M_trav, OCC = O_inf, UNKNOWN otherwise.

        Same return value as GridMapper.grid(): (data int8 row-major, width, height, ox, oy).
        Frontiers are FREE cells next to UNKNOWN ones.
        Cells the robot centre has passed through (grown by one cell, Nav2's point footprint)
        are FREE even inside O_inf: the robot was there, so they are passable, and an obstacle
        seen later from close by (or a radius rounded to voxels) must not leave the robot inside
        O_inf, where Nav2 can neither plan (start in an obstacle) nor recover (spin/backup
        refuse) and every frontier goal fails.
        """
        k = self.layer(z_r)
        if not 0 <= k < self.nz:
            return None
        been = self._visited_grown(z_r)
        tight = self.tight_at(z_r)
        # the layers within grid_layers of z_r: a floor rarely lies on one voxel layer (HM3D
        # 00810: 18 % of the ground floor one layer below z_r, the upper floor split 60/40 over
        # two layers); from one layer the Nav2 grid lost those parts and broke into pieces
        sl = slice(max(k - self.grid_layers, 0), min(k + self.grid_layers + 1, self.nz))
        inf = self.inflated[:, :, sl].any(axis=2)
        trav = self.traversable()[:, :, sl].any(axis=2) & ~inf      # O_inf of any of them wins
        free = ((trav & ~tight) | been).T   # [row=y, col=x]
        occ = (inf | tight).T
        known = free | occ
        if not known.any():
            return None
        rows = np.where(known.any(axis=1))[0]
        cols = np.where(known.any(axis=0))[0]
        r0, r1 = max(rows[0] - pad_cells, 0), min(rows[-1] + pad_cells + 1, self.ny)
        c0, c1 = max(cols[0] - pad_cells, 0), min(cols[-1] + pad_cells + 1, self.nx)
        out = np.full((r1 - r0, c1 - c0), self.UNKNOWN, dtype=np.int8)
        out[occ[r0:r1, c0:c1]] = self.OCC
        out[free[r0:r1, c0:c1]] = self.FREE
        return (out.reshape(-1), c1 - c0, r1 - r0,
                self.origin_xy + c0 * self.res, self.origin_xy + r0 * self.res)


class StuckDetector:
    """Eq. (17): forward motion that leaves the odometry behind the commanded motion.

    Over a sliding window, the pose x_hat is predicted from the pose at the start of the window
    and the velocity commands (unicycle model). The robot is stuck when it was commanded at
    least min_dist forward and ||x_odom - x_hat|| > delta (delta_stuck).
    delta is capped at ratio x the commanded distance: near obstacles Nav2 drives at ~0.07 m/s,
    0.07 m per 1 s window, so a fixed 0.15 m would never fire there, where the robot gets stuck.
    The window is short because Nav2 backs off after a brief contact (0.2-0.4 s in Habitat).
    """

    def __init__(self, window=1.0, delta=0.15, min_dist=0.05, ratio=0.5):
        self.window, self.delta, self.min_dist, self.ratio = window, delta, min_dist, ratio
        self.buf = deque()

    def reset(self):
        self.buf.clear()

    def update(self, t, x, y, yaw, vx, wz):
        """Add an odometry sample with the command in effect; returns True on a stuck event."""
        self.buf.append((t, x, y, yaw, vx, wz))
        while len(self.buf) > 2 and t - self.buf[1][0] >= self.window:
            self.buf.popleft()
        if t - self.buf[0][0] < 0.9 * self.window:
            return False
        _, px, py, pyaw, _, _ = self.buf[0]
        dist = 0.0
        prev_t = self.buf[0][0]
        for ti, _, _, _, v, w in list(self.buf)[1:]:
            dt = ti - prev_t
            prev_t = ti
            mid = pyaw + 0.5 * w * dt
            px += v * dt * math.cos(mid)
            py += v * dt * math.sin(mid)
            pyaw += w * dt
            dist += v * dt
        if dist < self.min_dist or math.hypot(x - px, y - py) <= min(self.delta, self.ratio * dist):
            return False
        self.reset()
        return True
