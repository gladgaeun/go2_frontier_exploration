"""Floor-state features of the floor-switch model (15 numbers per decision time).

One implementation for both paths: the offline extractor (floor_switch/scripts/make_features.py)
feeds it the JSONL records of a collected episode, the online node will feed it the same fields
from its own state. Inputs per step (the recorded field names):
  fg   frontier_goal_node state: z_r, floor_band, clusters [(x, y, z, size, ...)],
       explored_floors [z], landmarks [(x, y, z, top_x, top_y, top_z, size, reliable, ...)], target
  inst instance_map_node instances: [{label, gamma, n_pos, n_neg, reliable, z10, ...}]
  vm   voxel_mapper_node state: origin_z, res, climbing, trav_cells_per_layer,
       value_max_per_layer, value_mean_per_layer, value_voxels_per_layer
  actions  benchmark actions since the start (distance / 0.25 m + rotation / 30 deg)

The room-type histogram (9) is computed separately (BLIP-2 ITC) and appended later.
"""
import math

import numpy as np

# the instance map's label of each ObjectNav category (prepare_episodes.PROMPTS)
PROMPTS = {'chair': 'chair', 'bed': 'bed', 'plant': 'potted plant', 'toilet': 'toilet',
           'tv_monitor': 'tv', 'sofa': 'couch'}
CATEGORIES = ('chair', 'bed', 'plant', 'toilet', 'tv_monitor', 'sofa')

NAMES = ('area', 'n_frontier', 'frontier_size', 'dwell', 'cand_gamma', 'cand_pos', 'cand_neg',
         'tentative', 'value_max', 'value_mean',
         'o_floors', 'o_area', 'o_exhausted', 'o_gamma', 'o_stair_floors')

FLOOR_MATCH = 0.5      # m: two floor heights closer than this are the same floor
STAIR_FLOOR = 1.0      # m: a stair end this far from every visited floor is another floor
OBJ_BELOW = 0.3        # m: an object belongs to floor f when f - OBJ_BELOW <= z10 < f + OBJ_ABOVE
OBJ_ABOVE = 1.8
NEW_FLOOR_HOLD = 5     # steps (1 s) of the same z_r off the stairs before a floor is visited


def _arr(v):
    if v is None:
        return np.zeros(0)
    if isinstance(v, str):                 # the recorder may have written a list as its repr
        import ast
        v = ast.literal_eval(v)
    return np.asarray(v, dtype=np.float64)


def _list(v):
    if isinstance(v, str):
        import ast
        v = ast.literal_eval(v)
    return v or []


class FloorTracker:
    """Floor IDs of one episode from the robot's floor estimate z_r (map z), one call per
    decision time. The start floor counts at once; a new floor needs NEW_FLOOR_HOLD steps of the
    same z_r off the stairs (not climbing). Shared by FloorFeatures and rooms.RoomMap."""

    def __init__(self):
        self.floors = []                   # visited floor heights (map z)
        self.pending = None                # (z, steps) of a new z_r not yet a floor
        self.cur = None                    # index of the current floor in self.floors

    def find(self, z):
        for i, f in enumerate(self.floors):
            if abs(z - f) < FLOOR_MATCH:
                return i
        return None

    def update(self, z_r, climbing):
        """Current floor index, or None while z_r is not a floor (yet)."""
        i = self.find(z_r)
        if i is not None:
            self.cur, self.pending = i, None
            return i
        if not self.floors:
            self.floors.append(z_r)
            self.cur = 0
            return 0
        if climbing:
            self.pending = None
            return None
        if self.pending is not None and abs(self.pending[0] - z_r) < FLOOR_MATCH:
            self.pending = (self.pending[0], self.pending[1] + 1)
        else:
            self.pending = (z_r, 1)
        if self.pending[1] >= NEW_FLOOR_HOLD:
            self.floors.append(self.pending[0])
            self.cur, self.pending = len(self.floors) - 1, None
            return self.cur
        return None


class FloorFeatures:
    """Keeps the per-floor history of one episode; step() returns the 15 features."""

    def __init__(self, target):
        self.target = target
        self.label = PROMPTS.get(target, target)
        self.ft = FloorTracker()
        self.dwell = {}                    # floor index -> actions spent on it
        self.last_actions = None

    # -- floors -------------------------------------------------------------------------------
    @property
    def floors(self):
        return self.ft.floors

    @property
    def cur(self):
        return self.ft.cur

    def _find(self, z):
        return self.ft.find(z)

    def _update_floor(self, z_r, climbing):
        self.ft.update(z_r, climbing)

    # -- per-floor sums of the voxel layers ----------------------------------------------------
    @staticmethod
    def _layers(vm, z, band):
        n = len(_arr(vm.get('trav_cells_per_layer')))
        if n == 0:
            return np.zeros(0, dtype=bool)
        zc = vm['origin_z'] + (np.arange(n) + 0.5) * vm['res']
        return np.abs(zc - z) <= band

    @classmethod
    def _area(cls, vm, z, band):
        m = cls._layers(vm, z, band)
        cells = _arr(vm.get('trav_cells_per_layer'))
        return float(cells[m].sum()) * vm['res'] ** 2 if m.any() else 0.0

    @classmethod
    def _value(cls, vm, z, band):
        m = cls._layers(vm, z, band)
        vmax, vmean, nv = (_arr(vm.get(k)) for k in
                           ('value_max_per_layer', 'value_mean_per_layer', 'value_voxels_per_layer'))
        if not m.any() or len(vmax) != len(m) or nv[m].sum() <= 0:
            return 0.0, 0.0
        return float(vmax[m].max()), float((vmean[m] * nv[m]).sum() / nv[m].sum())

    # -- target candidates ----------------------------------------------------------------------
    def _on_floor(self, inst, z):
        z10 = inst.get('z10')
        return z10 is not None and z - OBJ_BELOW <= z10 < z + OBJ_ABOVE

    def _cands(self, inst, z):
        return [i for i in inst if i.get('label') == self.label and self._on_floor(i, z)]

    # -- step -----------------------------------------------------------------------------------
    def step(self, fg, inst, vm, actions):
        """The 15 features (numpy array, NAMES order) for one decision time, or None before the
        first floor estimate."""
        z_r = fg.get('z_r')
        if z_r is None or not vm:
            return None
        band = float(fg.get('floor_band') or 0.3)
        climbing = bool(vm.get('climbing'))
        self._update_floor(z_r, climbing)
        da = 0.0 if self.last_actions is None else max(0.0, actions - self.last_actions)
        self.last_actions = actions
        if not climbing and self._find(z_r) is not None:
            self.dwell[self.cur] = self.dwell.get(self.cur, 0.0) + da
        inst = _list(inst)

        # current floor
        area = self._area(vm, z_r, band)
        cl = [c for c in _list(fg.get('clusters')) if abs(c[2] - z_r) <= band]
        cands = self._cands(inst, z_r)
        best = max(cands, key=lambda i: i.get('gamma', 0.0)) if cands else None
        vmax, vmean = self._value(vm, z_r, band)
        cur = [
            math.log1p(area),
            float(len(cl)),
            math.log1p(sum(c[3] for c in cl)),
            self.dwell.get(self.cur, 0.0) / 500.0 if self._find(z_r) is not None else 0.0,
            float(best['gamma']) if best else 0.0,
            float(best.get('n_pos', 0)) if best else 0.0,
            float(best.get('n_neg', 0)) if best else 0.0,
            float(any(not i.get('reliable') for i in cands)),
            vmax, vmean,
        ]

        # other floors (visited, not the current one)
        others = [f for i, f in enumerate(self.floors) if abs(f - z_r) >= FLOOR_MATCH]
        explored = _list(fg.get('explored_floors'))
        o_area = sum(self._area(vm, f, band) for f in others)
        o_exh = sum(any(abs(f - e) < FLOOR_MATCH for e in explored) for f in others)
        o_gamma = max([i.get('gamma', 0.0) for f in others for i in self._cands(inst, f)], default=0.0)
        ends = []
        for lm in _list(fg.get('landmarks')):
            if len(lm) > 7 and lm[7] < 0.5:            # tentative stairs: not known yet
                continue
            ends += [lm[2], lm[5]]
        new = []
        for z in ends:
            if all(abs(z - f) >= STAIR_FLOOR for f in self.floors) and \
                    all(abs(z - n) >= STAIR_FLOOR for n in new):
                new.append(z)
        oth = [float(len(others)), math.log1p(o_area), float(o_exh), float(o_gamma), float(len(new))]
        return np.array(cur + oth, dtype=np.float64)


def one_hot(target):
    v = np.zeros(len(CATEGORIES))
    if target in CATEGORIES:
        v[CATEGORIES.index(target)] = 1.0
    return v
