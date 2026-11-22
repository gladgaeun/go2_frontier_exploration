#!/usr/bin/env python3
# Pick ObjectNav episodes from the official HM3D ObjectNav v2 dataset (the paper's benchmark,
# Habitat Challenge 2023) and write the list the runner works through.
#
# Run in the conda env with habitat-sim (hsim-test):
#   python prepare_episodes.py --out ~/objectnav_runs/val20/episodes.json --per-scene 2
#
# Each episode gets what the runner and the scorer need: the scene .glb, the start pose
# (habitat x, y, z and the yaw of start_rotation), the category and the SAM3 / BLIP-2 prompt,
# the optimal geodesic distance (episode info, for SPL), the view points of every goal object
# of the category (success: geodesic distance to the nearest one), and the voxel map extent
# below / above the start floor so every floor of the building is in the map.

import argparse
import glob
import gzip
import json
import math
import os
import random

import numpy as np

# category -> text prompt for SAM3 / BLIP-2: the Habitat Challenge 2023 ObjectNav goal names
# (chair, couch, potted plant, bed, toilet, tv) for the dataset keys
PROMPTS = {'chair': 'chair', 'bed': 'bed', 'plant': 'potted plant', 'toilet': 'toilet',
           'tv_monitor': 'tv', 'sofa': 'couch'}


def levels(pf, n=3000, seed=0):
    """Floor heights (habitat y) of the navmesh: clusters of navigable points >= 5 % of them,
    more than 1.5 m apart."""
    pf.seed(seed)
    y = np.sort(np.array([pf.get_random_navigable_point()[1] for _ in range(n)]))
    hist, edges = np.histogram(y, bins=np.arange(y.min() - 0.1, y.max() + 0.2, 0.2))
    out = []
    for h, e in zip(hist, edges):
        if h < 0.05 * len(y):
            continue
        c = e + 0.1
        if out and c - out[-1][0] <= 1.5:
            out[-1][1] += h
        else:
            out.append([c, h])
    return [round(float(c), 2) for c, _ in out]


def cross_floor_episodes(pf, lv, goals, cats, n, rng, scene_dir, glb, habitat_sim):
    """Episodes that need the stairs: the start is on a floor with no object of the category
    (first view point of each goal object), the goals are on another floor and reachable on the
    navmesh. geodesic_distance = geodesic to the nearest view point (as the dataset's)."""
    if len(lv) < 2:
        print(f'{scene_dir}: 1 floor, no cross-floor episode')
        return []
    floor_of = lambda y: int(np.argmin([abs(y - l) for l in lv]))
    pf.seed(rng.randrange(1 << 30))
    pts = [np.array(pf.get_random_navigable_point()) for _ in range(4000)]
    pts = [q for q in pts if abs(q[1] - lv[floor_of(q[1])]) < 0.3 and pf.distance_to_closest_obstacle(q, 1.0) > 0.3]
    cand = []
    for cat, objs in goals.items():
        if cats and cat not in cats or not objs:
            continue
        ends = [np.array(v, dtype=np.float32) for g in objs for v in g['view_points']]
        floors = {floor_of((g['view_points'][0] if g['view_points'] else g['position'])[1]) for g in objs}
        for f in range(len(lv)):
            if f in floors:
                continue
            starts = [q for q in pts if floor_of(q[1]) == f]
            rng.shuffle(starts)
            for q in starts[:30]:
                mp = habitat_sim.MultiGoalShortestPath()
                mp.requested_start = q.astype(np.float32)
                mp.requested_ends = ends
                if pf.find_path(mp) and np.isfinite(mp.geodesic_distance) and mp.geodesic_distance < 40.0:
                    cand.append((cat, f, q, float(mp.geodesic_distance), floors))
                    break
    rng.shuffle(cand)
    by_cat, picked = {}, []
    for c in cand:
        by_cat.setdefault(c[0], []).append(c)
    order = sorted(by_cat)
    off = rng.randrange(len(order)) if order else 0
    order = order[off:] + order[:off]
    while len(picked) < n and any(by_cat.values()):
        for c in order:
            if by_cat[c] and len(picked) < n:
                picked.append(by_cat[c].pop(0))
    out = []
    for k, (cat, f, q, geo, floors) in enumerate(picked):
        y0 = float(q[1])
        out.append({
            'id': f'{scene_dir}_x{k:02d}', 'scene_dir': scene_dir, 'glb': glb,
            'start': [round(float(v), 4) for v in q], 'yaw': round(rng.uniform(-math.pi, math.pi), 4),
            'category': cat, 'prompt': PROMPTS.get(cat, cat), 'geodesic_distance': round(geo, 4),
            'levels': lv, 'start_floor': f, 'goal_floors': sorted(floors), 'cross_floor': True,
            'goal_on_start_floor': False,
            'map_z_below': max(0.25, round(y0 - min(lv) + 0.5, 2)),
            'map_z_above': max(1.5, round(max(lv) - y0 + 1.5, 2)),
            'goals': goals[cat], 'synthetic': True,
        })
    print(f'{scene_dir}: {len(lv)} floors {lv}, {len(cand)} cross-floor candidates, {len(out)} picked')
    return out


def main():
    home = os.path.expanduser('~')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['v1', 'v2'], default='v1',
                   help='HM3D ObjectNav v1 val (2000 episodes, 20 scenes: the paper\'s benchmark) or v2 val '
                        '(1000 episodes, 36 scenes)')
    p.add_argument('--episodes-dir', default='', help='default: ~/data/datasets/objectnav/hm3d/unz/objectnav_hm3d_<dataset>/val/content')
    p.add_argument('--scenes-dir', default=f'{home}/data/versioned_data/hm3d-0.2/hm3d/val')
    p.add_argument('--out', required=True)
    p.add_argument('--scenes', default='', help='comma-separated scene ids (e.g. 00800,TEEsavR23oF); all if empty')
    p.add_argument('--categories', default='', help='comma-separated categories; all six if empty')
    p.add_argument('--per-scene', type=int, default=2, help='episodes per scene (0: all)')
    p.add_argument('--floors', choices=['any', 'same', 'cross'], default='any',
                   help='same / cross: the closest goal object on the start floor / on another floor')
    p.add_argument('--source', choices=['official', 'cross'], default='official',
                   help='official: episodes of the dataset (their closest goal is almost always on the '
                        'start floor: 2 of 1000 in val); cross: new episodes that start on a floor without '
                        'any object of the category, scored with the dataset goals and view points')
    p.add_argument('--limit', type=int, default=0, help='keep the first N episodes after interleaving the scenes')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    if not args.episodes_dir:
        args.episodes_dir = f'{home}/data/datasets/objectnav/hm3d/unz/objectnav_hm3d_{args.dataset}/val/content'
    import habitat_sim
    want = [s.strip() for s in args.scenes.split(',') if s.strip()]
    cats = [c.strip() for c in args.categories.split(',') if c.strip()]
    rng = random.Random(args.seed)
    out = []
    for f in sorted(glob.glob(os.path.join(args.episodes_dir, '*.json.gz'))):
        d = json.load(gzip.open(f))
        if not d['episodes']:
            continue
        rel = d['episodes'][0]['scene_id']                     # hm3d_v0.2/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb
        scene_dir = rel.split('/')[-2]
        if want and not any(w in scene_dir for w in want):
            continue
        glb = os.path.join(args.scenes_dir, scene_dir, os.path.basename(rel))
        nav = glb.replace('.glb', '.navmesh')
        if not (os.path.exists(glb) and os.path.exists(nav)):
            print(f'skip {scene_dir}: scene not found under {args.scenes_dir}')
            continue
        pf = habitat_sim.PathFinder()
        pf.load_nav_mesh(nav)
        lv = levels(pf)
        goals = {}
        for key, objs in d['goals_by_category'].items():
            cat = key[len(os.path.basename(rel)) + 1:]
            goals[cat] = [{'object_id': o['object_id'], 'position': o['position'],
                           'view_points': [v['agent_state']['position'] for v in o['view_points']]}
                          for o in objs]
        if args.source == 'cross':
            out += cross_floor_episodes(pf, lv, goals, cats, args.per_scene or 4, rng, scene_dir, glb,
                                        habitat_sim)
            continue
        for i, e in enumerate(d['episodes']):
            e['_index'] = i                                 # episode_id repeats within a scene
        eps = [e for e in d['episodes'] if (not cats or e['object_category'] in cats)]
        sel = []
        for e in eps:
            cat = e['object_category']
            if cat not in goals:
                continue
            y0 = e['start_position'][1]
            floor_of = lambda y: int(np.argmin([abs(y - l) for l in lv])) if lv else 0
            f0 = floor_of(y0)
            gfloors = sorted({floor_of(v[1]) for g in goals[cat] for v in g['view_points'][:1] or [g['position']]})
            # cross-floor: the closest goal object (episode info) is on another floor
            cg = next((g for g in goals[cat] if str(g['object_id']) == str(e['info'].get('closest_goal_object_id'))), None)
            cy = (cg['view_points'][0][1] if cg['view_points'] else cg['position'][1]) if cg else y0
            same = floor_of(cy) == f0
            e['_cross'] = not same
            if args.floors == 'same' and not same or args.floors == 'cross' and same:
                continue
            sel.append((e, f0, gfloors))
        rng.shuffle(sel)
        if args.per_scene > 0:
            # spread over the categories first
            by_cat, picked = {}, []
            for item in sel:
                by_cat.setdefault(item[0]['object_category'], []).append(item)
            order = sorted(by_cat)
            off = rng.randrange(len(order)) if order else 0
            order = order[off:] + order[:off]               # a different first category per scene
            while len(picked) < args.per_scene and any(by_cat.values()):
                for c in order:
                    if by_cat[c] and len(picked) < args.per_scene:
                        picked.append(by_cat[c].pop(0))
            sel = picked
        for e, f0, gfloors in sel:
            cat = e['object_category']
            q = e['start_rotation']                            # [x, y, z, w], rotation about +y
            yaw = 2.0 * math.atan2(q[1], q[3])
            y0 = e['start_position'][1]
            below = max(0.25, round(y0 - min(lv) + 0.5, 2)) if lv else 0.25
            above = max(1.5, round(max(lv) - y0 + 1.5, 2)) if lv else 2.0
            out.append({
                'id': f"{scene_dir}_{e['_index']:04d}",
                'scene_dir': scene_dir, 'glb': glb,
                'start': [round(v, 4) for v in e['start_position']], 'yaw': round(yaw, 4),
                'category': cat, 'prompt': PROMPTS.get(cat, cat),
                'geodesic_distance': e['info']['geodesic_distance'],
                'levels': lv, 'start_floor': f0, 'goal_floors': gfloors,
                'cross_floor': e['_cross'],
                'goal_on_start_floor': f0 in gfloors,
                'map_z_below': below, 'map_z_above': above,
                'goals': goals[cat],
            })
        print(f'{scene_dir}: {len(lv)} floor(s) {lv}, {len(eps)} episodes, {len(sel)} picked')
    # interleave the scenes (1st episode of every scene, then the 2nd, ...): a run stopped or
    # scored half-way is a sample spread over all scenes and categories
    rank, seen = {}, {}
    for e in out:
        rank[e['id']] = seen.get(e['scene_dir'], 0)
        seen[e['scene_dir']] = rank[e['id']] + 1
    out.sort(key=lambda e: (rank[e['id']], e['scene_dir']))
    if args.limit > 0:
        out = out[:args.limit]
    os.makedirs(os.path.dirname(os.path.abspath(os.path.expanduser(args.out))), exist_ok=True)
    with open(os.path.expanduser(args.out), 'w') as fh:
        json.dump(out, fh)
    n_cross = sum(e['cross_floor'] for e in out)
    print(f'{len(out)} episodes ({n_cross} cross-floor) -> {args.out}')


if __name__ == '__main__':
    main()
