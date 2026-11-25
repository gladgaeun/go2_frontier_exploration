#!/usr/bin/env python3
"""Render the robot's camera views again at every decision time of the collected episodes.

At each frontier_goal state record (1 s) the pose (trajectory.txt: position, yaw, camera pitch) is
set in Habitat (HM3D v0.1 scene) and an RGB-D frame is rendered with the episode camera
(640 x 480, hfov 79) at the height habitat_server used in that episode (server.log: "RGB-D camera at
X m above the base = 0.88 m above the floor"; X differs per episode with the navmesh offset).
  <out>/rgb/<id>.npy    RGB resized to 224 x 224 (bicubic, BLIP-2's input; temporary, embed_frames
                        removes it once the embeddings are written)
  <out>/geom/<id>.npz   t, depth (120 x 160, z-depth m, every 4th pixel), camera position and
                        heading in the map frame of the voxel map (start at the origin, x forward,
                        z up), pitch, z_r (map z), climbing flag, camera height above the base
Run in the Habitat env (e.g. `conda activate hsim-test`).
"""
import argparse
import bisect
import re
import json
import math
import os

import numpy as np
from PIL import Image


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


def camera_height(server_log, default=0.88):
    """Camera height above the agent base that habitat_server used; its --camera-height default
    (0.88) when it did not see the floor at the start ("kept above the agent base")."""
    text = open(server_log).read()
    m = re.search(r'camera at ([0-9.]+) m above the base', text)
    if m:
        return float(m.group(1))
    return default if 'kept above the agent base' in text else None


def hab_to_map(e, p):
    """Habitat point (y up) -> map frame (start at the origin, x forward, z up): the inverse of
    the transform habitat_server's odometry starts from (see issue_counts.m2h)."""
    start, yaw0 = e['start'], e['yaw']
    rx, ry = -p[2] + start[2], -p[0] + start[0]
    c, s = math.cos(yaw0), math.sin(yaw0)
    return [c * rx + s * ry, -s * rx + c * ry, p[1] - start[1]]


def make_sim(habitat_sim, glb, w, h, hfov, cam_h):
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = glb
    cfg.load_semantic_mesh = False
    specs = []
    for uuid, typ in (('rgb', habitat_sim.SensorType.COLOR), ('depth', habitat_sim.SensorType.DEPTH)):
        s = habitat_sim.CameraSensorSpec()
        s.uuid, s.sensor_type, s.resolution, s.hfov = uuid, typ, [h, w], hfov
        s.position = [0.0, cam_h, 0.0]
        specs.append(s)
    ag = habitat_sim.agent.AgentConfiguration()
    ag.sensor_specifications = specs
    return habitat_sim.Simulator(habitat_sim.Configuration(cfg, [ag]))


def main():
    H = os.path.expanduser
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', default=H('~/objectnav_runs/collect_v01'))
    ap.add_argument('--out', default=H('~/objectnav_runs/floor_switch/frames'))
    ap.add_argument('--ids', default='')
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--hfov', type=float, default=79.0)
    ap.add_argument('--depth-stride', type=int, default=4)
    args = ap.parse_args()

    import habitat_sim
    import magnum as mn
    eps = json.load(open(os.path.join(args.runs, 'episodes.json')))
    if args.ids:
        want = set(args.ids.split(','))
        eps = [e for e in eps if e['id'] in want]
    eps = [e for e in eps if os.path.exists(os.path.join(args.runs, e['id'], 'result.json'))]
    for d in ('rgb', 'geom'):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)
    by_scene = {}
    for e in eps:
        by_scene.setdefault(e['glb'], []).append(e)
    n = 0
    st = args.depth_stride
    for glb, es in sorted(by_scene.items()):
        todo = [e for e in es if not os.path.exists(os.path.join(args.out, 'geom', e['id'] + '.npz'))]
        if not todo:
            n += len(es)
            continue
        sim = make_sim(habitat_sim, glb, args.width, args.height, args.hfov, 0.88)
        agent = sim.get_agent(0)
        try:
            for e in todo:
                n += 1
                rd = os.path.join(args.runs, e['id'])
                cam_h = camera_height(os.path.join(rd, 'server.log'))
                if cam_h is None:
                    print(f'[{n}/{len(eps)}] {e["id"]}: no camera height in server.log, skipped', flush=True)
                    continue
                fg = read_jsonl(os.path.join(rd, 'record', 'frontier_goal.jsonl'), 'state')
                vm = read_jsonl(os.path.join(rd, 'record', 'voxel_mapper.jsonl'), 'state')
                vm_t = [r['t'] for r in vm]
                tr = np.loadtxt(os.path.join(rd, 'trajectory.txt'), ndmin=2)
                tt = list(tr[:, 0])
                T, rgbs, deps, cams, heads, pitches, floors, climbs = [], [], [], [], [], [], [], []
                for s in fg:
                    t = s['t']
                    k = max(0, bisect.bisect_right(tt, t + 1e-6) - 1)
                    x, y, z, yaw, pitch = tr[k, 1:6]
                    state = habitat_sim.AgentState()
                    state.position = np.array([x, y, z], dtype=np.float32)
                    agent.set_state(state)
                    agent.scene_node.rotation = mn.Quaternion.rotation(mn.Rad(float(yaw)), mn.Vector3.y_axis())
                    q = mn.Quaternion.rotation(mn.Rad(float(pitch)), mn.Vector3.x_axis())
                    for uuid in ('rgb', 'depth'):        # set_state() puts the sensors back to the spec
                        agent._sensors[uuid].node.rotation = q
                        agent._sensors[uuid].node.translation = mn.Vector3(0.0, cam_h, 0.0)
                    obs = sim.get_sensor_observations()
                    rgb = Image.fromarray(obs['rgb'][:, :, :3]).resize((224, 224), Image.BICUBIC)
                    rgbs.append(np.asarray(rgb))
                    deps.append(obs['depth'][st // 2::st, st // 2::st].astype(np.float16))
                    cams.append(hab_to_map(e, [x, y + cam_h, z]))
                    heads.append(yaw - e['yaw'])
                    pitches.append(pitch)
                    z_r = s.get('z_r')
                    floors.append(np.nan if z_r is None else z_r)
                    j = bisect.bisect_right(vm_t, t + 1e-6) - 1
                    climbs.append(bool(vm[j].get('climbing')) if j >= 0 else False)
                    T.append(t)
                np.save(os.path.join(args.out, 'rgb', e['id'] + '.tmp.npy'), np.stack(rgbs))
                os.replace(os.path.join(args.out, 'rgb', e['id'] + '.tmp.npy'),
                           os.path.join(args.out, 'rgb', e['id'] + '.npy'))
                np.savez_compressed(os.path.join(args.out, 'geom', e['id'] + '.npz'), t=np.array(T),
                                    depth=np.stack(deps), cam=np.array(cams), heading=np.array(heads),
                                    pitch=np.array(pitches), z_r=np.array(floors),
                                    climbing=np.array(climbs), hfov=args.hfov, cam_h=cam_h)
                print(f'[{n}/{len(eps)}] {e["id"]}: {len(T)} frames', flush=True)
        finally:
            sim.close()
    open(os.path.join(args.out, 'render_done'), 'w').close()


if __name__ == '__main__':
    main()
