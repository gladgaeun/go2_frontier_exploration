#!/usr/bin/env python3
"""Online vs offline check of the floor-switch node (plan step 10) on episodes run with
floor_switch:=shadow (or on) and --record.

For each episode with <run>/record/floor_switch.jsonl, per decision time (frontier_goal state t):
  features : the node's 15 features vs make_features.features_episode() on the same records
             (actions from trajectory.txt, as offline);
  actions  : the node's TF count vs the trajectory count;
  rooms    : the node's raw room similarities and histogram vs the offline frames of this run
             (render_frames.py -> embed_frames.py -> make_room_features.py with --runs <runs>,
             outputs under --offline; optional: skipped if not there);
  p        : the node's p vs PModel on the offline features + offline rooms (and + the node's
             rooms, to separate the room part);
  ready    : rule.SwitchRule replayed on the offline p vs the node's ready.
Prints per episode and pooled summaries (max / 95th percentile of |online - offline|).
"""
import argparse
import bisect
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from make_features import cumulative_actions, features_episode  # noqa: E402
from floor_switch.features import NAMES  # noqa: E402
from floor_switch.model import PModel  # noqa: E402
from floor_switch.rooms import ROOMS  # noqa: E402
from floor_switch.rule import SwitchRule  # noqa: E402


def q(a):
    a = np.abs(np.asarray(a, dtype=np.float64))
    a = a[np.isfinite(a)]
    return (f'max {a.max():.4f}  p95 {np.percentile(a, 95):.4f}  mean {a.mean():.4f}' if len(a) else 'n/a')


def main():
    H = os.path.expanduser
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', default=H('~/objectnav_runs/fs_shadow_v0'))
    ap.add_argument('--offline', default='', help='dir with frames/, itc/, rooms/ of these runs (default <runs>/offline)')
    ap.add_argument('--model', default=H('~/objectnav_runs/floor_switch/p_model/catw/online_model.json'))
    args = ap.parse_args()
    off = args.offline or os.path.join(args.runs, 'offline')
    pm = PModel(args.model)
    rm = pm.meta['rooms']
    rdir = os.path.join(off, 'rooms', f'm{rm["margin"]:g}_T{rm["T"]:g}')
    r = pm.meta['rule']
    eps = json.load(open(os.path.join(args.runs, 'episodes.json')))
    pool = {k: [] for k in ('feat', 'act', 'act_rel', 'sims', 'rooms', 'p', 'p_onrooms')}
    per_feat = {n: [] for n in NAMES}
    ready_on = ready_off = ready_both = 0
    for e in eps:
        rd = os.path.join(args.runs, e['id'])
        fp = os.path.join(rd, 'record', 'floor_switch.jsonl')
        if not os.path.exists(fp):
            continue
        on = [json.loads(l) for l in open(fp)]
        on = {round(float(s['fg_t']), 2): s for s in on if s.get('type') == 'step'}
        offx = {row[0]: np.array(row[1:]) for row in features_episode(rd, e['category'])}
        tt, acts = cumulative_actions(os.path.join(rd, 'trajectory.txt'))
        rooms_off, sims_off = {}, {}
        if os.path.exists(os.path.join(rdir, e['id'] + '.csv')):
            cols = [c.replace(' ', '_') for c in ROOMS]
            rooms_off = {round(float(w['t']), 2): np.array([float(w[c]) for c in cols])
                         for w in csv.DictReader(open(os.path.join(rdir, e['id'] + '.csv')))}
            it = np.load(os.path.join(off, 'itc', e['id'] + '.npz'))
            sims_off = {round(float(t), 2): s for t, s in zip(it['t'], it['sims'])}
        rule = SwitchRule(r['X'], r['tau'], r['actions_per_metre'], r['climb_p75'] + r['margin'])
        n = 0
        ep = {k: [] for k in pool}
        for t in sorted(on):
            s = on[t]
            k = bisect.bisect_right(tt, t + 1e-6) - 1
            a_off = float(acts[k]) if k >= 0 else 0.0
            ep['act'].append(s['actions'] - a_off)
            if a_off > 20:
                ep['act_rel'].append((s['actions'] - a_off) / a_off)
            if s.get('x') is None or t not in offx:
                continue
            n += 1
            d = np.array(s['x']) - offx[t]
            ep['feat'].append(np.abs(d).max())
            for j, nm in enumerate(NAMES):
                per_feat[nm].append(d[j])
            ro = rooms_off.get(t)
            if s.get('sims') is not None and t in sims_off:
                ep['sims'].append(np.abs(np.array(s['sims']) - sims_off[t]).max())
            if ro is not None:
                ep['rooms'].append(np.abs(np.array(s['rooms']) - ro).max())
                p_off = pm.p(np.concatenate([offx[t], ro]), e['category'])[0]
                ep['p'].append(s['p'] - p_off)
            else:
                p_off = None
            if s.get('rooms') is not None:
                ep['p_onrooms'].append(s['p'] - pm.p(np.concatenate([offx[t], s['rooms']]), e['category'])[0])
            if p_off is not None:
                fl = None if s.get('floor_z') is None else round(s['floor_z'], 3)
                rdy, _ = rule.step(p_off, fl, bool(s.get('tracking')), s.get('stairs'), a_off)
                ready_on += s['ready']; ready_off += rdy; ready_both += s['ready'] and rdy
        print(f'{e["id"]} ({e["category"]}): {len(on)} online steps, {n} compared; '
              f'actions at the end online {on[max(on)]["actions"]:.0f} vs trajectory {float(acts[-1]):.0f}')
        for k_ in ('feat', 'act', 'sims', 'rooms', 'p', 'p_onrooms'):
            print(f'  {k_:9s} {q(ep[k_])}')
            pool[k_] += ep[k_]
        pool['act_rel'] += ep['act_rel']
    print('\npooled |online - offline|')
    for k_ in ('feat', 'act', 'act_rel', 'sims', 'rooms', 'p', 'p_onrooms'):
        print(f'  {k_:9s} {q(pool[k_])}')
    print('per feature (max |diff|): ' + ', '.join(f'{nm} {np.abs(v).max():.3g}' for nm, v in per_feat.items() if v))
    print(f'ready: online {ready_on}, offline replay {ready_off}, both {ready_both}')


if __name__ == '__main__':
    main()
