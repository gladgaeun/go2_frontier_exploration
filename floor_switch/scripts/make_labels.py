#!/usr/bin/env python3
"""Floor-switch labels: y = 1 when the target can be reached on the robot's current floor.

For every decision time of an episode (the 1 s `state` records of frontier_goal_node):
  - true pose: the Habitat trajectory (trajectory.txt, 0.2 s) at that sim time;
  - true floor k: the episode level nearest to the pose height (as the episode's A / B split:
    goal_floors are the levels nearest to the view points); floor k spans the heights between
    the midpoints to the levels below and above (a floor may have parts at other heights: a
    raised living room, a half level);
  - in_band = 0 (no label used there) when the robot is between floors: more than `band` from
    level k (stairs, a landing) or climbing (voxel_mapper state);
  - y = 1 when a view point of a goal object on floor k is reachable from the pose on the
    navmesh recomputed for the agent (radius 0.18, height 0.88, max climb 0.2) by a path that
    stays on floor k; d_floor is that path's length (m);
  - estimated floor: the level nearest to the robot's floor estimate z_r (map z, 0 = start
    height), to count the decision times where the features (estimated floor) and the label
    (true floor) disagree.

Output: <out>/<episode id>.csv with one row per decision time and <out>/summary.json.
Run in the Habitat env (habitat_sim), e.g. `conda activate hsim-test`.
"""
import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'trav_exploration', 'scripts', 'objectnav'))
from prepare_episodes import recomputed_pathfinder  # noqa: E402

FIELDS = ['t', 'x', 'y', 'z', 'floor', 'in_band', 'label', 'label_any', 'd_floor', 'd_any',
          'est_floor', 'floor_mismatch', 'z_r', 'tracking_target', 'goal_kind',
          'climbing', 'on_stairs']


def read_jsonl(path, kind):
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get('type') == kind:
                out.append(r)
    return out


def read_trajectory(path):
    a = np.loadtxt(path, ndmin=2)
    return a[:, 0], a[:, 1:4]              # sim time, habitat (x, y, z)


def m2h(e, p):
    """map frame (start at the origin, x forward, z up) -> habitat (y up)."""
    start, yaw0 = np.array(e['start']), e['yaw']
    c, s = math.cos(yaw0), math.sin(yaw0)
    d = np.array([c * p[0] - s * p[1], s * p[0] + c * p[1]])
    ros = np.array([-start[2], -start[0]]) + d
    return np.array([-ros[1], start[1] + (p[2] if len(p) > 2 else 0.0), -ros[0]])


def nearest_level(lv, y):
    return int(np.argmin([abs(y - l) for l in lv]))


class FloorPaths:
    """Shortest paths to the goal view points of one episode that stay on one floor."""

    def __init__(self, habitat_sim, pf, e):
        self.hs, self.pf, self.lv = habitat_sim, pf, e['levels']
        lv = self.lv
        self.slab = [((lv[k - 1] + lv[k]) / 2 if k > 0 else -1e9,
                      (lv[k] + lv[k + 1]) / 2 if k + 1 < len(lv) else 1e9) for k in range(len(lv))]
        self.by_floor = defaultdict(list)       # floor -> [view points of one goal object]
        self.all_vp = []
        for g in e['goals']:
            vps = [np.array(v, dtype=np.float32) for v in g['view_points']]
            vps = [self.pf.snap_point(v) for v in vps]
            vps = [np.array(v) for v in vps if not np.any(np.isnan(v))]
            if not vps:
                continue
            self.all_vp += vps
            per = defaultdict(list)
            for v in vps:
                per[nearest_level(self.lv, v[1])].append(v)
            for k, vs in per.items():
                self.by_floor[k].append(vs)

    def _path(self, p, ends):
        sp = self.hs.MultiGoalShortestPath()
        sp.requested_start = p
        sp.requested_ends = ends
        if not self.pf.find_path(sp) or not np.isfinite(sp.geodesic_distance):
            return None, None
        return sp.geodesic_distance, np.array(sp.points)

    def any_distance(self, p):
        d, _ = self._path(p, self.all_vp)
        return d

    def floor_distance(self, p, k):
        """Length of the shortest path to a goal view point on floor k that stays on floor k."""
        objs = self.by_floor.get(k, [])
        if not objs:
            return None
        lo, hi = self.slab[k]
        ok = lambda pts: pts is not None and np.all((pts[:, 1] >= lo) & (pts[:, 1] <= hi))
        d, pts = self._path(p, [v for vs in objs for v in vs])
        if d is not None and ok(pts):
            return d
        if d is None:
            return None
        best = None                     # the nearest one leaves the floor: each object on its own
        for vs in objs:
            d, pts = self._path(p, vs)
            if d is not None and ok(pts) and (best is None or d < best):
                best = d
        return best


def label_episode(habitat_sim, pf, e, run_dir, band):
    states = read_jsonl(os.path.join(run_dir, 'record', 'frontier_goal.jsonl'), 'state')
    vm = read_jsonl(os.path.join(run_dir, 'record', 'voxel_mapper.jsonl'), 'state')
    tt, hp = read_trajectory(os.path.join(run_dir, 'trajectory.txt'))
    if not states or len(tt) == 0:
        return None, {}
    vm_t = np.array([r['t'] for r in vm]) if vm else np.zeros(0)
    fp = FloorPaths(habitat_sim, pf, e)
    lv = e['levels']
    rows, xy_err = [], []
    for s in states:
        t = s['t']
        i = int(np.argmin(np.abs(tt - t)))
        p = hp[i]
        # clock check: the recorded robot position vs the trajectory at the same time
        if s.get('robot') is not None:
            q = m2h(e, list(s['robot']) + [0.0])
            xy_err.append(float(np.hypot(q[0] - p[0], q[2] - p[2])))
        k = nearest_level(lv, p[1])
        v = vm[int(np.argmin(np.abs(vm_t - t)))] if len(vm_t) else {}
        in_band = abs(p[1] - lv[k]) <= band and not v.get('climbing')
        ps = np.array(pf.snap_point(p.astype(np.float32)))
        if np.any(np.isnan(ps)):
            ps = p.astype(np.float32)
        d_floor = fp.floor_distance(ps, k)
        d_any = fp.any_distance(ps)
        z_r = s.get('z_r')
        est = nearest_level(lv, e['start'][1] + z_r) if z_r is not None else None
        rows.append({
            't': round(t, 2), 'x': round(float(p[0]), 3), 'y': round(float(p[1]), 3),
            'z': round(float(p[2]), 3), 'floor': k, 'in_band': int(in_band),
            'label': (int(d_floor is not None) if in_band else ''), 'label_any': int(d_floor is not None),
            'd_floor': ('' if d_floor is None else round(float(d_floor), 3)),
            'd_any': ('' if d_any is None else round(float(d_any), 3)),
            'est_floor': ('' if est is None else est),
            'floor_mismatch': ('' if est is None or not in_band else int(est != k)),
            'z_r': ('' if z_r is None else round(z_r, 3)),
            'tracking_target': int(bool(s.get('tracking_target'))),
            'goal_kind': s.get('goal_kind') or '',
            'climbing': int(bool(v.get('climbing'))), 'on_stairs': int(bool(v.get('on_stairs'))),
        })
    info = {'clock_xy_err_median': float(np.median(xy_err)) if xy_err else None,
            'clock_xy_err_p95': float(np.percentile(xy_err, 95)) if xy_err else None}
    return rows, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', default=os.path.expanduser('~/objectnav_runs/collect_v01'),
                    help='collection dir (episodes.json + one dir per episode)')
    ap.add_argument('--out', default=os.path.expanduser('~/objectnav_runs/floor_switch/labels'))
    ap.add_argument('--band', type=float, default=1.0,
                    help='m: farther than this from the nearest level = between floors (no label)')
    ap.add_argument('--ids', default='', help='comma-separated episode ids (default: all finished)')
    ap.add_argument('--agent-radius', type=float, default=0.18)
    ap.add_argument('--agent-height', type=float, default=0.88)
    ap.add_argument('--max-climb', type=float, default=0.2)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    import habitat_sim
    eps = json.load(open(os.path.join(args.runs, 'episodes.json')))
    if args.ids:
        want = set(args.ids.split(','))
        eps = [e for e in eps if e['id'] in want]
    eps = [e for e in eps if os.path.exists(os.path.join(args.runs, e['id'], 'result.json'))]
    os.makedirs(args.out, exist_ok=True)
    by_scene = defaultdict(list)
    for e in eps:
        by_scene[e['glb']].append(e)

    summary = {}
    sp = os.path.join(args.out, 'summary.json')
    if os.path.exists(sp) and not args.overwrite:
        summary = json.load(open(sp))
    n = 0
    for glb, es in sorted(by_scene.items()):
        todo = [e for e in es if args.overwrite or not os.path.exists(os.path.join(args.out, e['id'] + '.csv'))]
        if not todo:
            continue
        sim = recomputed_pathfinder(habitat_sim, glb, args.agent_radius, args.agent_height, args.max_climb)
        try:
            for e in todo:
                rows, info = label_episode(habitat_sim, sim.pathfinder, e, os.path.join(args.runs, e['id']), args.band)
                n += 1
                if rows is None:
                    print(f'[{n}/{len(eps)}] {e["id"]}: no records')
                    continue
                with open(os.path.join(args.out, e['id'] + '.csv'), 'w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=FIELDS)
                    w.writeheader()
                    w.writerows(rows)
                lab = [r for r in rows if r['in_band']]
                s = {'ab': 'A' if e['goal_on_start_floor'] else 'B', 'category': e['category'],
                     'n': len(rows), 'n_in_band': len(lab),
                     'n_pos': sum(r['label'] == 1 for r in lab),
                     'n_mismatch': sum(r['floor_mismatch'] == 1 for r in lab),
                     'start_label': rows[0]['label'], 'floors_visited': sorted({r['floor'] for r in lab}),
                     **info}
                summary[e['id']] = s
                print(f'[{n}/{len(eps)}] {e["id"]} {s["ab"]} {e["category"]}: {len(rows)} t, '
                      f'in band {len(lab)}, y=1 {s["n_pos"]}, start y {s["start_label"]}, '
                      f'mismatch {s["n_mismatch"]}, clock xy err {info["clock_xy_err_median"]:.3f}', flush=True)
        finally:
            sim.close()
        json.dump(summary, open(sp, 'w'), indent=1)
    json.dump(summary, open(sp, 'w'), indent=1)


if __name__ == '__main__':
    main()
