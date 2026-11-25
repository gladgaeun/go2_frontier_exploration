#!/usr/bin/env python3
"""Room-type histograms (9) of every decision time, for each discard margin and temperature.

Replays each episode's frames (render_frames.py geometry in the map frame + embed_frames.py
similarities) through floor_switch.rooms.RoomMap (0.05 m cells, FloorTracker floors): frame i is
fused into the map of the robot's floor unless its top-2 room similarity margin is below `margin`,
and the histogram at time t is the mean softmax(similarity / T) over that floor's cells.

Output: <out>/m<margin>_T<T>/<episode id>.csv (t + the 9 rooms). No Habitat or GPU needed.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from floor_switch.rooms import ROOMS, RoomMap, depth_points  # noqa: E402


def main():
    H = os.path.expanduser
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', default=H('~/objectnav_runs/collect_v01'))
    ap.add_argument('--frames', default=H('~/objectnav_runs/floor_switch/frames'))
    ap.add_argument('--itc', default=H('~/objectnav_runs/floor_switch/itc'))
    ap.add_argument('--out', default=H('~/objectnav_runs/floor_switch/rooms'))
    ap.add_argument('--temps', default='0.01,0.02,0.05,0.1')
    ap.add_argument('--margins', default='0,0.02,0.045,0.075',
                    help='discard frames whose best and second-best room similarities differ by less')
    ap.add_argument('--res', type=float, default=0.05, help='cell size (m), the voxel map resolution')
    ap.add_argument('--ids', default='')
    args = ap.parse_args()
    temps = [float(x) for x in args.temps.split(',')]
    margins = [float(x) for x in args.margins.split(',')]
    eps = json.load(open(os.path.join(args.runs, 'episodes.json')))
    if args.ids:
        want = set(args.ids.split(','))
        eps = [e for e in eps if e['id'] in want]
    cols = [r.replace(' ', '_') for r in ROOMS]
    name = lambda m, T: f'm{m:g}_T{T:g}'
    for m in margins:
        for T in temps:
            os.makedirs(os.path.join(args.out, name(m, T)), exist_ok=True)
    n = 0
    for e in eps:
        gp, ip = os.path.join(args.frames, 'geom', e['id'] + '.npz'), os.path.join(args.itc, e['id'] + '.npz')
        if not (os.path.exists(gp) and os.path.exists(ip)):
            continue
        g, s = np.load(gp), np.load(ip)
        sims, hfov = s['sims'].astype(np.float64), float(g['hfov'])
        pts = [depth_points(g['depth'][i].astype(np.float64), hfov, g['cam'][i], float(g['heading'][i]),
                            float(g['pitch'][i])) for i in range(len(g['t']))]
        for m in margins:
            rm = RoomMap(temps, res=args.res, margin_min=m)
            rows = {T: [] for T in temps}
            for i, t in enumerate(g['t']):
                z = float(g['z_r'][i])
                if not np.isnan(z):
                    rm.update(sims[i], pts[i][0], pts[i][1], z, bool(g['climbing'][i]))
                for T in temps:
                    h = np.zeros(len(ROOMS)) if np.isnan(z) else rm.histogram(z, T)
                    rows[T].append([round(float(t), 2)] + [round(float(v), 5) for v in h])
            for T in temps:
                with open(os.path.join(args.out, name(m, T), e['id'] + '.csv'), 'w', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(['t'] + cols)
                    w.writerows(rows[T])
        n += 1
        print(f'[{n}] {e["id"]}: {len(g["t"])} t', flush=True)


if __name__ == '__main__':
    main()
