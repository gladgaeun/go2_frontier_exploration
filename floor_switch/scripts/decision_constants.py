#!/usr/bin/env python3
"""Constants of the v0 cost guard (floor-switch plan, sec. 9.3), measured on the collected episodes
of the train scenes:

  switch only when  remaining actions >= actions to the stairs + climb actions + margin

  1. actions per metre of path (benchmark actions = distance / 0.25 m + rotation / 30 deg, as
     run_episodes counts them), on the floor and while climbing (voxel_mapper `climbing`): turns
     make it more than 4 per metre. actions to the stairs = path length x the floor ratio.
  2. climb actions: actions of each climb (a run of `climbing` records); the climbs that ended on
     another true floor (labels) and the others; the 75th percentile is the guard's value.
  3. margin: actions from the arrival on a goal floor (first decision time in band on a floor with
     y = 1 after leaving the start floor) to the end, in the B episodes. Successes end there;
     failures are censored at the actions left (they needed at least that many). Kaplan-Meier
     median and 75th percentile, next to the successes-only percentiles.
Writes <out>/decision_constants.json.
"""
import argparse
import bisect
import csv
import json
import math
import os

import numpy as np


def read_jsonl(path, kind):
    out = []
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get('type') == kind:
                out.append(r)
    return out


def trajectory(path, step=0.25, turn=math.radians(30.0)):
    a = np.loadtxt(path, ndmin=2)
    d = np.hypot(np.diff(a[:, 1]), np.diff(a[:, 3]))
    r = np.abs((np.diff(a[:, 4]) + math.pi) % (2.0 * math.pi) - math.pi)
    return a[:, 0], np.concatenate([[0.0], np.cumsum(d)]), np.concatenate([[0.0], np.cumsum(d / step + r / turn)])


def at(tt, cum, t):
    k = bisect.bisect_right(tt, t + 1e-6) - 1
    return float(cum[max(k, 0)])


def km(times, events, qs):
    """Kaplan-Meier quantiles of times with event flags (False = censored); None past the curve."""
    o = np.argsort(times)
    t, ev = np.asarray(times)[o], np.asarray(events)[o]
    S, n, curve = 1.0, len(t), []
    for i in range(len(t)):
        if ev[i]:
            S *= 1.0 - 1.0 / (n - i)
        curve.append((t[i], S))
    out = {}
    for q in qs:
        hit = [ti for ti, s in curve if s <= 1.0 - q]
        out[q] = float(hit[0]) if hit else None
    return out, curve[-1][1]


def main():
    H = os.path.expanduser
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', default=H('~/objectnav_runs/collect_v01'))
    ap.add_argument('--labels', default=H('~/objectnav_runs/floor_switch/labels'))
    ap.add_argument('--split', default=H('~/objectnav_runs/floor_switch_split.json'))
    ap.add_argument('--out', default=H('~/objectnav_runs/floor_switch/p_model/catw'))
    ap.add_argument('--max-actions', type=float, default=500.0)
    args = ap.parse_args()

    train = {s[:5] for s in json.load(open(args.split))['train']}
    eps = [e for e in json.load(open(os.path.join(args.runs, 'episodes.json'))) if e['id'][:5] in train]
    res = {r['id']: r for r in csv.DictReader(open(os.path.join(args.runs, 'results.csv')))}

    flat_d = flat_a = climb_d = climb_a = 0.0
    climbs = []                      # (actions, metres, changed floor)
    arrivals = []                    # (actions from arrival to end, success)
    n_b = n_b_arrived = 0
    for e in eps:
        rd = os.path.join(args.runs, e['id'])
        lp = os.path.join(args.labels, e['id'] + '.csv')
        if not (os.path.exists(os.path.join(rd, 'trajectory.txt')) and os.path.exists(lp) and e['id'] in res):
            continue
        tt, dist, acts = trajectory(os.path.join(rd, 'trajectory.txt'))
        vm = read_jsonl(os.path.join(rd, 'record', 'voxel_mapper.jsonl'), 'state')
        lab = list(csv.DictReader(open(lp)))
        lab_t = [float(r['t']) for r in lab]

        # 1. actions per metre on the floor / while climbing (per trajectory step, by the latest record)
        vm_t = [r['t'] for r in vm]
        climbing = np.array([bool(vm[j]['climbing']) if j >= 0 else False
                             for j in (bisect.bisect_right(vm_t, t + 1e-6) - 1 for t in tt[1:])])
        dd, da = np.diff(dist), np.diff(acts)
        flat_d += dd[~climbing].sum(); flat_a += da[~climbing].sum()
        climb_d += dd[climbing].sum(); climb_a += da[climbing].sum()

        # 2. climbs: runs of climbing records
        run = None
        for r in vm + [{'t': 1e9, 'climbing': False}]:
            if r.get('climbing') and run is None:
                run = r['t']
            elif not r.get('climbing') and run is not None:
                t0, t1 = run, min(r['t'], tt[-1])
                f0 = lab[max(0, bisect.bisect_right(lab_t, t0) - 1)]['floor']
                f1 = lab[min(len(lab) - 1, bisect.bisect_left(lab_t, t1))]['floor']
                climbs.append((at(tt, acts, t1) - at(tt, acts, t0), at(tt, dist, t1) - at(tt, dist, t0), f0 != f1))
                run = None

        # 3. margin: B episodes, arrival on a goal floor
        if not e['goal_on_start_floor']:
            n_b += 1
            start_floor = next((r['floor'] for r in lab if r['in_band'] == '1'), None)
            arr = next((float(r['t']) for r in lab if r['in_band'] == '1' and r['floor'] != start_floor
                        and r['label'] == '1'), None)
            if arr is not None:
                n_b_arrived += 1
                total = float(json.load(open(os.path.join(rd, 'result.json')))['actions'])
                ok = res[e['id']]['success'] == 'True'
                left = total - at(tt, acts, arr) if ok else args.max_actions - at(tt, acts, arr)
                arrivals.append((max(0.0, left), ok))

    r_flat, r_climb = flat_a / flat_d, climb_a / max(climb_d, 1e-9)
    ca = np.array([c[0] for c in climbs]); cm = np.array([c[2] for c in climbs])
    print(f'{len(eps)} train episodes')
    print(f'1. actions per metre: floor {r_flat:.2f} ({flat_d:.0f} m), climbing {r_climb:.2f} ({climb_d:.0f} m) '
          f'(straight line: 4.00)')
    print(f'2. climbs: {len(ca)}, {int(cm.sum())} changed floor')
    for name, m in (('changed floor', cm), ('did not', ~cm), ('all', np.ones(len(ca), bool))):
        if m.any():
            print(f'   {name:14s} n {int(m.sum()):3d}  actions median {np.median(ca[m]):.0f}  '
                  f'p75 {np.percentile(ca[m], 75):.0f}  p90 {np.percentile(ca[m], 90):.0f}')
    at_ = np.array([a for a, _ in arrivals]); ok = np.array([s for _, s in arrivals])
    q, s_end = km(at_, ok, (0.5, 0.75))
    print(f'3. margin: {n_b} B episodes, {n_b_arrived} reached a goal floor, {int(ok.sum())} of them succeeded')
    print(f'   successes only: median {np.median(at_[ok]):.0f}  p75 {np.percentile(at_[ok], 75):.0f} actions')
    print(f'   Kaplan-Meier (failures censored at the actions left): median {q[0.5]}  p75 {q[0.75]}  '
          f'(share still not found at the last time: {s_end:.2f})')
    out = dict(actions_per_metre_floor=r_flat, actions_per_metre_climbing=r_climb,
               climb_actions=dict(changed_floor=dict(n=int(cm.sum()), median=float(np.median(ca[cm])),
                                                     p75=float(np.percentile(ca[cm], 75))),
                                  all=dict(n=len(ca), median=float(np.median(ca)), p75=float(np.percentile(ca, 75)))),
               margin=dict(n_arrived=n_b_arrived, n_success=int(ok.sum()),
                           success_median=float(np.median(at_[ok])), success_p75=float(np.percentile(at_[ok], 75)),
                           km_median=q[0.5], km_p75=q[0.75]))
    os.makedirs(args.out, exist_ok=True)
    json.dump(out, open(os.path.join(args.out, 'decision_constants.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
