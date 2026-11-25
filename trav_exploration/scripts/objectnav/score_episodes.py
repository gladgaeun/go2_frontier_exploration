#!/usr/bin/env python3
# Score ObjectNav runs (run_episodes.py) as Habitat ObjectNav does: an episode succeeds when the
# agent stopped (here: frontier_goal_node reported the target reached) within success_distance
# (geodesic) of a view point of an object of the goal category; SPL = S * l* / max(l*, l) with l*
# the episode's geodesic distance and l the agent's path length (3-D, from the trajectory).
#
# Run in the conda env with habitat-sim (hsim-test):
#   python score_episodes.py --episodes ~/objectnav_runs/val20/episodes.json
# Writes results.csv next to the episodes and prints SR / SPL / DTG overall, same floor vs
# cross-floor, and per category.

import argparse
import csv
import json
import os

import numpy as np


def path_length(traj):
    try:
        p = np.loadtxt(traj, ndmin=2)
    except (OSError, ValueError):
        return 0.0
    if len(p) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(p[:, 1:4], axis=0), axis=1).sum())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes', required=True)
    p.add_argument('--out', default='', help='results directory (default: next to episodes.json)')
    p.add_argument('--dataset-navmesh', action='store_true',
                   help="score on the scene's shipped .basis.navmesh (runs made with run_episodes --dataset-navmesh)")
    p.add_argument('--agent-radius', type=float, default=0.18)
    p.add_argument('--agent-height', type=float, default=0.88)
    p.add_argument('--max-climb', type=float, default=0.2)
    p.add_argument('--success-distance', type=float, default=0.2,
                   help='m to the nearest view point (the paper: 0.2; Habitat Challenge 2023 default 0.1)')
    args = p.parse_args()

    import habitat_sim
    eps = json.load(open(os.path.expanduser(args.episodes)))
    out = os.path.expanduser(args.out) or os.path.dirname(os.path.abspath(os.path.expanduser(args.episodes)))
    pfs, rows = {}, []
    for ep in eps:
        rp = os.path.join(out, ep['id'], 'result.json')
        if not os.path.exists(rp):
            continue
        res = json.load(open(rp))
        nav = ep['glb'].replace('.glb', '.navmesh')
        if nav not in pfs:
            # the navmesh the agent moved on (run_episodes): recomputed for the agent by default,
            # the scene's shipped file with --dataset-navmesh. Scoring on the shipped file while the
            # agent moved on the recomputed one gave no distance to the goal for some stops.
            for other in list(pfs):
                if pfs[other][1] is not None:
                    pfs[other][1].close()
                pfs.pop(other)
            if args.dataset_navmesh:
                pf = habitat_sim.PathFinder()
                pf.load_nav_mesh(nav)
                pfs[nav] = (pf, None)
            else:
                from prepare_episodes import recomputed_pathfinder
                sim = recomputed_pathfinder(habitat_sim, ep['glb'], args.agent_radius, args.agent_height,
                                            args.max_climb)
                pfs[nav] = (sim.pathfinder, sim)
        pf = pfs[nav][0]
        dtg = float('inf')
        if res.get('final_pose'):
            pos = np.array(res['final_pose'][:3], dtype=np.float32)
            snapped = np.array(pf.snap_point(pos))
            ends = [np.array(v, dtype=np.float32) for g in ep['goals'] for v in g['view_points']]
            if ends and np.all(np.isfinite(snapped)):
                mp = habitat_sim.MultiGoalShortestPath()
                mp.requested_start = snapped
                mp.requested_ends = ends
                if pf.find_path(mp):
                    dtg = float(mp.geodesic_distance)
            # Euclidean distance to the nearest goal object centre, for reference
        l_opt = float(ep['geodesic_distance'])
        length = path_length(os.path.join(out, ep['id'], 'trajectory.txt'))
        success = bool(res.get('stop_called')) and dtg <= args.success_distance
        spl = float(success) * l_opt / max(l_opt, length, 1e-6)
        rows.append({'id': ep['id'], 'scene': ep['scene_dir'], 'category': ep['category'],
                     'cross_floor': ep['cross_floor'], 'status': res.get('status'),
                     'stop_called': bool(res.get('stop_called')), 'success': success, 'spl': round(spl, 4),
                     'dtg': round(dtg, 3) if np.isfinite(dtg) else '', 'geodesic': round(l_opt, 2),
                     'path_length': round(length, 2), 'sim_time': res.get('sim_time'),
                     'wall_time': res.get('wall_time')})
    if not rows:
        print('no finished episodes')
        return
    with open(os.path.join(out, 'results.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def summary(name, rs):
        if not rs:
            return
        sr = 100.0 * np.mean([r['success'] for r in rs])
        spl = 100.0 * np.mean([r['spl'] for r in rs])
        stop = 100.0 * np.mean([r['stop_called'] for r in rs])
        d = [r['dtg'] for r in rs if r['dtg'] != '']
        print(f'{name:<16} n={len(rs):3d}  SR {sr:5.1f}  SPL {spl:5.1f}  stop {stop:5.1f}  '
              f'DTG median {np.median(d) if d else float("nan"):.2f} m')

    print(f'success: stop within {args.success_distance} m (geodesic) of a view point')
    summary('all', rows)
    summary('same floor', [r for r in rows if not r['cross_floor']])
    summary('cross-floor', [r for r in rows if r['cross_floor']])
    for c in sorted({r['category'] for r in rows}):
        summary(c, [r for r in rows if r['category'] == c])
    st = {}
    for r in rows:
        st[r['status']] = st.get(r['status'], 0) + 1
    print('end of episode:', ', '.join(f'{k} {v}' for k, v in sorted(st.items())))
    print(f"-> {os.path.join(out, 'results.csv')}")


if __name__ == '__main__':
    main()
