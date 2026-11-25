"""Room-type histogram of the current floor (9 numbers per decision time), plan sec. 5.2.

Per frame (one per decision time, 1 s) BLIP-2 ITC gives the raw similarity of the image to each
room prompt (cosine, max over the 32 query embeddings). The similarities, not probabilities, are
kept, so the softmax temperature and the discard rule are chosen by cross-validation without
rendering or running BLIP-2 again.

RoomMap is the value map (Sec. V-B) with 9 channels: the frame's similarities go to the grid
cells of its depth points on the robot's floor, with confidence cos^2 of the angle off the optical
axis, fused by VLFM's confidence-weighted average. Grid: the map frame of the voxel map (start at
the origin, z up), cells of the voxel resolution (0.05 m). Floors: floor_features.FloorTracker on
z_r, the same floor IDs as the other features. A frame whose best and second-best room
similarities differ by less than `margin` is ambiguous and is not fused (the discard rule).
histogram(T) is the mean over the current floor's cells of softmax(similarity / T); the sums are
kept per temperature and updated with the cells a frame changes.

Shared by the offline extractor (scripts/make_room_features.py) and, later, the online node.
"""
import math

import numpy as np

from floor_switch.features import FloorTracker

ROOMS = ('bedroom', 'bathroom', 'kitchen', 'living room', 'dining room', 'hallway', 'office',
         'laundry room', 'staircase')


def prompt(room):
    article = 'an' if room[0] in 'aeiou' else 'a'
    return f'a photo of {article} {room}.'


PROMPTS = tuple(prompt(r) for r in ROOMS)

BELOW, ABOVE = 0.3, 2.0  # m: points of a floor: floor - BELOW <= z < floor + ABOVE
MAX_RANGE = 5.0          # m (the paper's depth range)


def softmax(s, T):
    z = s / T
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def margin(sims):
    s = np.sort(np.asarray(sims))
    return float(s[-1] - s[-2])


def depth_points(depth, hfov_deg, cam_pos, yaw, pitch):
    """Points (N, 3, z up) and confidences of a z-depth image.

    cam_pos: camera position; yaw: heading about +z (0 = +x); pitch: + = up."""
    h, w = depth.shape
    f = (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    d = depth.astype(np.float64)
    ok = (d > 0.05) & (d <= MAX_RANGE)
    x_r = (u - w / 2.0) / f                      # right
    y_d = (v - h / 2.0) / f                      # down
    fwd, left, up = d, -x_r * d, -y_d * d
    cp, sp = math.cos(pitch), math.sin(pitch)
    fwd2, up2 = fwd * cp - up * sp, fwd * sp + up * cp
    cy, sy = math.cos(yaw), math.sin(yaw)
    X = cam_pos[0] + fwd2 * cy - left * sy
    Y = cam_pos[1] + fwd2 * sy + left * cy
    Z = cam_pos[2] + up2
    ang = np.arctan(np.hypot(x_r, y_d))
    conf = np.cos(np.clip(ang / math.radians(hfov_deg / 2.0), 0, 1) * math.pi / 2.0) ** 2
    return np.stack([X[ok], Y[ok], Z[ok]], 1), conf[ok]


class _FloorGrid:
    """Fused similarities of one floor on a dense grid that grows with the map, and the sums of
    softmax(similarity / T) over its observed cells per temperature."""

    GROW = 64                                    # cells added on a side when the map grows

    def __init__(self, n, temps):
        self.n, self.temps = n, temps
        self.i0 = self.j0 = None                 # cell index of grid row / column 0
        self.conf = np.zeros((0, 0))
        self.sims = np.zeros((0, 0, n))
        self.count = 0
        self.sums = {T: np.zeros(n) for T in temps}

    def _fit(self, ij):
        lo, hi = ij.min(axis=0), ij.max(axis=0)
        if self.i0 is None:
            self.i0, self.j0 = lo[0] - self.GROW, lo[1] - self.GROW
            H, W = hi[0] - lo[0] + 1 + 2 * self.GROW, hi[1] - lo[1] + 1 + 2 * self.GROW
            self.conf = np.zeros((H, W))
            self.sims = np.zeros((H, W, self.n))
            return
        H, W = self.conf.shape
        a = max(0, self.i0 - lo[0])
        b = max(0, hi[0] - (self.i0 + H - 1))
        c = max(0, self.j0 - lo[1])
        d = max(0, hi[1] - (self.j0 + W - 1))
        if a or b or c or d:
            a, b, c, d = (x + self.GROW if x else 0 for x in (a, b, c, d))
            self.conf = np.pad(self.conf, ((a, b), (c, d)))
            self.sims = np.pad(self.sims, ((a, b), (c, d), (0, 0)))
            self.i0 -= a
            self.j0 -= c

    def fuse(self, ij, c, s):
        """ij (M, 2) distinct cell indices, c (M,) confidences, s (n,) the frame's similarities."""
        self._fit(ij)
        r, q = ij[:, 0] - self.i0, ij[:, 1] - self.j0
        co, so = self.conf[r, q], self.sims[r, q]
        old = co > 0
        for T in self.temps:                      # the old cells leave the sums
            if old.any():
                self.sums[T] -= softmax(so[old], T).sum(axis=0)
        w = (co / np.maximum(co + c, 1e-12))[:, None]
        sn = w * so + (1.0 - w) * s[None, :]
        cn = (co * co + c * c) / np.maximum(co + c, 1e-12)
        self.sims[r, q], self.conf[r, q] = sn, cn
        self.count += int((~old).sum())
        for T in self.temps:
            self.sums[T] += softmax(sn, T).sum(axis=0)

    def histogram(self, T):
        return self.sums[T] / self.count if self.count else np.zeros(self.n)


class RoomMap:
    """Per-floor 9-channel room maps of one episode."""

    def __init__(self, temps, res=0.05, margin_min=0.0, n=len(ROOMS)):
        self.temps, self.res, self.margin_min, self.n = list(temps), res, margin_min, n
        self.ft = FloorTracker()
        self.grids = []

    def update(self, sims, pts, conf, z_r, climbing):
        """One decision time: track the floor and fuse the frame (sims (n,), pts (N, 3) in the
        map frame, conf (N,)). Returns the current floor index or None."""
        k = self.ft.update(z_r, climbing)
        while len(self.grids) < len(self.ft.floors):
            self.grids.append(_FloorGrid(self.n, self.temps))
        if k is None or climbing or len(pts) == 0 or margin(sims) < self.margin_min:
            return k
        f = self.ft.floors[k]
        m = (pts[:, 2] >= f - BELOW) & (pts[:, 2] < f + ABOVE)
        if not m.any():
            return k
        ij = np.floor(pts[m, :2] / self.res).astype(np.int64)
        keys, inv = np.unique(ij, axis=0, return_inverse=True)
        cmax = np.zeros(len(keys))
        np.maximum.at(cmax, inv.reshape(-1), conf[m])      # one confidence per cell and frame
        keep = cmax > 0
        if keep.any():
            self.grids[k].fuse(keys[keep], cmax[keep], np.asarray(sims, dtype=np.float64))
        return k

    def histogram(self, z_r, T):
        """Mean softmax(similarity / T) over the cells of the floor at z_r; zeros if none."""
        i = self.ft.find(z_r)
        if i is None or i >= len(self.grids):
            return np.zeros(self.n)
        return self.grids[i].histogram(T)
