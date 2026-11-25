#!/usr/bin/env python3
"""Floor-state features (15) of every decision time of the collected episodes.

Replays the JSONL records of each episode through floor_switch.features (the library the
online node uses too): at every frontier_goal state record (1 s) it takes the latest instance map
and voxel map records up to that time and the benchmark actions so far (distance / 0.25 m +
rotation / 30 deg of trajectory.txt, as run_episodes counts them).

Output: <out>/<episode id>.csv (t + floor_features.NAMES). Plain python3, no Habitat needed.
"""
import argparse
import bisect
import csv
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from floor_switch.features import NAMES, FloorFeatures  # noqa: E402


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


def cumulative_actions(path, step=0.25, turn=math.radians(30.0)):
    a = np.loadtxt(path, ndmin=2)
    d = np.hypot(np.diff(a[:, 1]), np.diff(a[:, 3]))
    r = np.abs((np.diff(a[:, 4]) + math.pi) % (2.0 * math.pi) - math.pi)
    return a[:, 0], np.concatenate([[0.0], np.cumsum(d / step + r / turn)])


def latest(times, recs, t):
    i = bisect.bisect_right(times, t + 1e-6) - 1
    return recs[i] if i >= 0 else None


def features_episode(run_dir, target):
    rd = os.path.join(run_dir, 'record')
    fg = read_jsonl(os.path.join(rd, 'frontier_goal.jsonl'), 'state')
    im = read_jsonl(os.path.join(rd, 'instance_map.jsonl'), 'instances')
    vm = read_jsonl(os.path.join(rd, 'voxel_mapper.jsonl'), 'state')
    if not fg:
        return None
    im_t, vm_t = [r['t'] for r in im], [r['t'] for r in vm]
    tt, acts = cumulative_actions(os.path.join(run_dir, 'trajectory.txt'))
    ff = FloorFeatures(target)
    rows = []
    for s in fg:
        t = s['t']
        i = latest(im_t, im, t)
        v = latest(vm_t, vm, t)
        k = bisect.bisect_right(tt, t + 1e-6) - 1
        a = float(acts[k]) if k >= 0 else 0.0
        x = ff.step(s, i['instances'] if i else [], v or {}, a)
        if x is None:
            continue
        rows.append([round(t, 2)] + [round(float(u), 5) for u in x])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', default=os.path.expanduser('~/objectnav_runs/collect_v01'))
    ap.add_argument('--out', default=os.path.expanduser('~/objectnav_runs/floor_switch/features'))
    ap.add_argument('--ids', default='')
    args = ap.parse_args()
    eps = json.load(open(os.path.join(args.runs, 'episodes.json')))
    if args.ids:
        want = set(args.ids.split(','))
        eps = [e for e in eps if e['id'] in want]
    eps = [e for e in eps if os.path.exists(os.path.join(args.runs, e['id'], 'result.json'))]
    os.makedirs(args.out, exist_ok=True)
    for n, e in enumerate(eps, 1):
        rows = features_episode(os.path.join(args.runs, e['id']), e['category'])
        if rows is None:
            print(f'[{n}/{len(eps)}] {e["id"]}: no records')
            continue
        with open(os.path.join(args.out, e['id'] + '.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(('t',) + NAMES)
            w.writerows(rows)
        print(f'[{n}/{len(eps)}] {e["id"]}: {len(rows)} t', flush=True)


if __name__ == '__main__':
    main()
