#!/usr/bin/env python3
# Habitat-sim side of the Habitat <-> ROS 2 bridge for go2_frontier_exploration.
# Runs inside the habitat conda environment (Python 3.9). It does NOT import ROS.
#
#   conda activate hsim-test
#   python habitat_server.py --scene /path/to/xxxxx.basis.glb
#
# Protocol (ZMQ REQ/REP, multipart messages):
#   request : [json]   {"op": "info"} | {"op": "reset"} | {"op": "step", "vx": m/s, "wz": rad/s, "dt": s}
#   reply   : [json header, scan float32 bytes, rgb uint8 bytes (may be empty)]
#
# Coordinate frames
#   Habitat world : x right, y up, z backward (agent looks along -z)
#   ROS (REP-103) : x forward, y left, z up
#   ros = (-hab_z, -hab_x, hab_y). The odom frame is placed at the start pose.

import argparse
import json
import math
import signal
import sys
import time

import numpy as np

# ----------------------------------------------------------------------------
# Pure helpers (no habitat import) -- unit-testable without a GPU
# ----------------------------------------------------------------------------


def wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def hab_to_ros_vec(p):
    """Habitat world vector -> ROS world vector."""
    return np.array([-p[2], -p[0], p[1]], dtype=np.float64)


def ros_yaw_from_hab_forward(fwd):
    """ROS yaw of the agent, given its forward vector expressed in Habitat world."""
    f = hab_to_ros_vec(fwd)
    return math.atan2(f[1], f[0])


def build_scan_lut(num_beams, depth_width, num_cams=4):
    """Map every LaserScan beam to (camera index, pixel column, cos of ray angle).

    Camera k looks at ROS yaw k*90 deg (0 front, 1 left, 2 back, 3 right), hfov 90 deg.
    Beams start at -pi and go counter-clockwise, matching sensor_msgs/LaserScan.
    """
    cam_span = 2.0 * math.pi / num_cams
    f = (depth_width / 2.0) / math.tan(cam_span / 2.0)          # focal length in pixels
    angles = -math.pi + np.arange(num_beams) * (2.0 * math.pi / num_beams)
    cam = np.round(angles / cam_span).astype(int) % num_cams
    local = (angles - cam * cam_span + math.pi) % (2.0 * math.pi) - math.pi
    # A positive (left) angle is to the left of the image centre, i.e. a smaller column.
    col = np.round(depth_width / 2.0 - f * np.tan(local) - 0.5).astype(int)
    col = np.clip(col, 0, depth_width - 1)
    off = (col + 0.5 - depth_width / 2.0) / f
    cosang = 1.0 / np.sqrt(1.0 + off * off)
    return cam, col, cosang


def depth_to_scan(depth_rows, lut, max_range, min_range=0.1):
    """depth_rows: array (num_cams, W) of z-depth on the horizontal row -> ranges (float32)."""
    cam, col, cosang = lut
    z = depth_rows[cam, col]
    r = z / cosang
    # depth 0 = no geometry hit (holes in HM3D meshes, open sky); beyond max = no return.
    # Readings closer than min_range are dropped too: they come from mesh pieces touching
    # the camera, and clamping them to min_range put a phantom obstacle on the robot itself
    # (Nav2: "No valid trajectories", Spin/BackUp: "Collision Ahead" in open space).
    r = np.where((z <= 0.0) | (r > max_range) | (r < min_range), np.inf, r)
    return r.astype(np.float32)


def obstacle_mask_from_navigable(nav, radius_px):
    """Approximate the real obstacles of one floor from its navmesh top-down view.

    The navmesh is the set of positions the robot CENTRE may occupy, i.e. free space eroded
    by the agent radius. Dilating it back by that radius gives the space the robot body can
    touch; everything else (walls, furniture, stairs, drop-offs) is an obstacle.
    Returns (obstacle_mask, pad) where the mask was padded by `pad` cells on every side.
    """
    pad = int(radius_px) + 2
    nav = np.pad(np.asarray(nav, dtype=bool), pad, constant_values=False)
    reach = nav.copy()
    r = int(radius_px)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy <= r * r and (dx or dy):
                reach |= np.roll(np.roll(nav, dy, axis=0), dx, axis=1)
    return ~reach, pad


def navmesh_ray_ranges(obst, x0, z0, mpp, hab_x, hab_z, yaw_ros, n_beams, max_range):
    """Ray-cast the beams of the virtual LiDAR in a top-down obstacle grid.

    obst[row, col]: row = (habitat z - z0)/mpp, col = (habitat x - x0)/mpp.
    Beam i points at ROS angle yaw_ros + (-pi + i*2pi/n); ROS (cos, sin) maps to habitat
    (dx, dz) = (-sin, -cos). Returns ranges (inf where nothing within max_range).
    """
    ang = yaw_ros - math.pi + np.arange(n_beams) * (2.0 * math.pi / n_beams)
    t = np.arange(0.5 * mpp, max_range, 0.5 * mpp)
    xs = hab_x - np.sin(ang)[:, None] * t[None, :]
    zs = hab_z - np.cos(ang)[:, None] * t[None, :]
    cols = np.floor((xs - x0) / mpp).astype(np.int64)
    rows = np.floor((zs - z0) / mpp).astype(np.int64)
    inside = (rows >= 0) & (rows < obst.shape[0]) & (cols >= 0) & (cols < obst.shape[1])
    blocked = np.ones_like(inside)            # outside the grid counts as obstacle
    blocked[inside] = obst[rows[inside], cols[inside]]
    first = np.argmax(blocked, axis=1)
    hit = blocked[np.arange(n_beams), first]
    return np.where(hit, t[first], np.inf).astype(np.float32)


class OdomTracker:
    """Turns Habitat world poses into odometry relative to the start pose (ROS frame)."""

    def __init__(self):
        self.p0 = None
        self.yaw0 = 0.0
        self.prev = None

    def reset(self, pos_ros, yaw_ros):
        self.p0 = np.asarray(pos_ros[:2], dtype=np.float64)
        self.yaw0 = yaw_ros
        self.prev = (0.0, 0.0, 0.0)

    def update(self, pos_ros, yaw_ros, dt):
        d = np.asarray(pos_ros[:2]) - self.p0
        c, s = math.cos(self.yaw0), math.sin(self.yaw0)
        x = c * d[0] + s * d[1]
        y = -s * d[0] + c * d[1]
        yaw = wrap_angle(yaw_ros - self.yaw0)
        px, py, pyaw = self.prev
        vx = wz = 0.0
        if dt > 0.0:
            vx = ((x - px) * math.cos(pyaw) + (y - py) * math.sin(pyaw)) / dt
            wz = wrap_angle(yaw - pyaw) / dt
        self.prev = (x, y, yaw)
        return x, y, yaw, vx, wz


# ----------------------------------------------------------------------------
# Habitat simulation
# ----------------------------------------------------------------------------


class HabitatRobot:
    def __init__(self, args):
        import habitat_sim
        import magnum as mn
        self.hs = habitat_sim
        self.mn = mn
        self.args = args

        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = args.scene
        if args.dataset_config:
            sim_cfg.scene_dataset_config_file = args.dataset_config
        sim_cfg.enable_physics = False
        sim_cfg.gpu_device_id = args.gpu

        specs = []
        for k in range(4):
            spec = habitat_sim.CameraSensorSpec()
            spec.uuid = f'scan_depth_{k}'
            spec.sensor_type = habitat_sim.SensorType.DEPTH
            spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            spec.resolution = [3, args.depth_width]          # [height, width]; middle row is horizontal
            spec.hfov = mn.Deg(90.0)
            spec.position = [0.0, args.laser_height, 0.0]
            spec.orientation = [0.0, k * math.pi / 2.0, 0.0]  # +yaw about Habitat +y == CCW in ROS
            if hasattr(spec, 'far'):
                spec.far = args.max_range + 1.0
            specs.append(spec)

        if args.rgb_width > 0:
            spec = habitat_sim.CameraSensorSpec()
            spec.uuid = 'rgb'
            spec.sensor_type = habitat_sim.SensorType.COLOR
            spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            spec.resolution = [args.rgb_height, args.rgb_width]
            spec.hfov = mn.Deg(90.0)
            spec.position = [0.0, args.camera_height, 0.0]
            specs.append(spec)

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.height = args.agent_height
        agent_cfg.radius = args.agent_radius
        agent_cfg.sensor_specifications = specs

        self.sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
        self.agent = self.sim.get_agent(0)
        self.node = self.agent.scene_node

        self._setup_navmesh()
        self.start_pos, self.start_yaw_hab = self._pick_start()
        self._build_floor_obstacles()

        self.vc = habitat_sim.physics.VelocityControl()
        self.vc.controlling_lin_vel = True
        self.vc.controlling_ang_vel = True
        self.vc.lin_vel_is_local = True
        self.vc.ang_vel_is_local = True

        self.lut = build_scan_lut(args.num_beams, args.depth_width)
        self.odom = OdomTracker()
        self.reset()

    # -- setup ---------------------------------------------------------------
    def _setup_navmesh(self):
        pf = self.sim.pathfinder
        if self.args.recompute_navmesh or not pf.is_loaded:
            settings = self.hs.NavMeshSettings()
            settings.set_defaults()
            settings.agent_radius = self.args.agent_radius
            settings.agent_height = self.args.agent_height
            # Single-floor exploration: steps higher than this are not walkable, so stairs are
            # not part of the navmesh (default Habitat value 0.2 m lets the robot climb them).
            settings.agent_max_climb = self.args.max_climb
            if hasattr(settings, 'include_static_objects'):
                settings.include_static_objects = True
            ok = self.sim.recompute_navmesh(pf, settings)
            print(f'[habitat_server] navmesh recomputed (radius={self.args.agent_radius} m, '
                  f'max climb={self.args.max_climb} m): {ok}')
        if not pf.is_loaded:
            sys.exit('[habitat_server] ERROR: no navmesh available for this scene')
        print(f'[habitat_server] navigable area: {pf.navigable_area:.1f} m^2')

    def _build_floor_obstacles(self):
        """Top-down obstacle grid of the start floor, used to add 'virtual' scan returns."""
        self.obst = None
        if not self.args.navmesh_obstacles:
            return
        pf = self.sim.pathfinder
        mpp = 0.05
        nav = np.asarray(pf.get_topdown_view(mpp, float(self.start_pos[1])), dtype=bool)
        obst, pad = obstacle_mask_from_navigable(nav, round(self.args.agent_radius / mpp))
        lower = pf.get_bounds()[0]
        self.obst = obst
        self.obst_x0 = float(lower[0]) - pad * mpp
        self.obst_z0 = float(lower[2]) - pad * mpp
        self.obst_mpp = mpp
        print(f'[habitat_server] floor obstacle grid {obst.shape[1]}x{obst.shape[0]} '
              f'({nav.sum() * mpp * mpp:.1f} m^2 navigable on this floor)')

    def _pick_start(self):
        pf = self.sim.pathfinder
        if self.args.start:
            x, y, z, yaw = [float(v) for v in self.args.start.split(',')]
            p = np.array(pf.snap_point(np.array([x, y, z], dtype=np.float32)))
            return p, yaw
        # Sample with our own seeded RNG (pathfinder.seed() did not change the result), keep
        # points with enough clearance, then pick among those on the LARGEST island so the
        # robot does not start in a pocket it cannot leave (e.g. boxed in by sofas).
        rng = np.random.default_rng(self.args.seed)
        lower, upper = [np.array(v, dtype=np.float64) for v in pf.get_bounds()]
        cands = []
        for _ in range(5000):
            q = rng.uniform(lower, upper).astype(np.float32)
            p = np.array(pf.snap_point(q), dtype=np.float64)
            if not np.all(np.isfinite(p)) or not pf.is_navigable(p):
                continue
            try:
                c = pf.distance_to_closest_obstacle(p, 2.0)
            except Exception:
                c = self.args.start_clearance
            if c < self.args.start_clearance:
                continue
            try:
                r = pf.island_radius(p)
            except Exception:
                r = 0.0
            cands.append((r, c, p))
            if len(cands) >= 300:
                break
        if not cands:
            print('[habitat_server] WARNING: no start pose with the requested clearance, '
                  'using any navigable point')
            p = np.array(pf.get_random_navigable_point(), dtype=np.float64)
            cands = [(0.0, 0.0, p)]
        r_max = max(c[0] for c in cands)
        pool = [c for c in cands if c[0] >= 0.9 * r_max]
        r, c, best = pool[int(rng.integers(len(pool)))]
        print(f'[habitat_server] start: island radius {r:.1f} m (largest {r_max:.1f} m), '
              f'clearance {min(c, 2.0):.2f} m, {len(cands)} candidates')
        yaw = float(rng.uniform(-math.pi, math.pi))
        return best, yaw

    # -- state -----------------------------------------------------------------
    def _set_pose(self, pos, yaw_hab):
        mn = self.mn
        self.node.translation = mn.Vector3(*[float(v) for v in pos])
        self.node.rotation = mn.Quaternion.rotation(mn.Rad(yaw_hab), mn.Vector3.y_axis())

    def _pose_ros(self):
        mn = self.mn
        pos = np.array(list(self.node.translation), dtype=np.float64)
        fwd = self.node.rotation.transform_vector(mn.Vector3(0.0, 0.0, -1.0))
        return pos, hab_to_ros_vec(pos), ros_yaw_from_hab_forward(list(fwd))

    def reset(self):
        self._set_pose(self.start_pos, self.start_yaw_hab)
        self.sim_time = 0.0
        self.steps = 0
        self.collisions = 0
        self.path_length = 0.0
        _, pos_ros, yaw_ros = self._pose_ros()
        self.odom.reset(pos_ros, yaw_ros)
        print('[habitat_server] start pose (habitat x,y,z,yaw): '
              f'{self.start_pos[0]:.3f},{self.start_pos[1]:.3f},{self.start_pos[2]:.3f},'
              f'{self.start_yaw_hab:.3f}   (pass with --start to reproduce)')

    def step(self, vx, wz, dt):
        hs, mn = self.hs, self.mn
        vx = float(np.clip(vx, -self.args.max_lin_vel, self.args.max_lin_vel))
        wz = float(np.clip(wz, -self.args.max_ang_vel, self.args.max_ang_vel))
        collided = False
        if dt > 0.0 and (abs(vx) > 1e-6 or abs(wz) > 1e-6):
            self.vc.linear_velocity = mn.Vector3(0.0, 0.0, -vx)   # local forward is -z
            self.vc.angular_velocity = mn.Vector3(0.0, wz, 0.0)
            start = hs.RigidState(self.node.rotation, self.node.translation)
            target = self.vc.integrate_transform(dt, start)
            end = self.sim.pathfinder.try_step(start.translation, target.translation)
            end = mn.Vector3(*[float(v) for v in end])
            wanted = target.translation - start.translation
            got = end - start.translation
            # navmesh blocked the motion (sliding along a wall also counts)
            if (mn.Vector2(wanted.x, wanted.z) - mn.Vector2(got.x, got.z)).length() > 1e-3:
                collided = True
                self.collisions += 1
            self.path_length += mn.Vector2(got.x, got.z).length()
            self.node.translation = end
            self.node.rotation = target.rotation.normalized()
        self.sim_time += dt
        self.steps += 1
        return collided

    def observe(self):
        obs = self.sim.get_sensor_observations()
        rows = np.stack([obs[f'scan_depth_{k}'][1, :] for k in range(4)]).astype(np.float32)
        ranges = depth_to_scan(rows, self.lut, self.args.max_range)
        if self.obst is not None:
            # A 2D LiDAR plane at laser_height misses low obstacles (first stair steps, low
            # tables, drop-offs). Add them from the navmesh so Nav2 and the map see them,
            # like the Go2's 3D LiDAR would.
            pos, _, yaw_ros = self._pose_ros()
            virt = navmesh_ray_ranges(self.obst, self.obst_x0, self.obst_z0, self.obst_mpp,
                                      pos[0], pos[2], yaw_ros, len(ranges), self.args.max_range)
            ranges = np.minimum(ranges, virt)
        rgb = b''
        if 'rgb' in obs:
            rgb = np.ascontiguousarray(obs['rgb'][:, :, :3]).tobytes()
        return ranges, rgb

    def header(self, dt, collided=False):
        pos, pos_ros, yaw_ros = self._pose_ros()
        x, y, yaw, vx, wz = self.odom.update(pos_ros, yaw_ros, dt)
        fwd = self.node.rotation.transform_vector(self.mn.Vector3(0.0, 0.0, -1.0))
        return {
            'sim_time': self.sim_time,
            'step': self.steps,
            'odom': {'x': x, 'y': y, 'yaw': yaw, 'vx': vx, 'wz': wz},
            'gt_habitat': {'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2]),
                           'yaw': math.atan2(-fwd.x, -fwd.z)},
            'collided': collided,
            'collisions': self.collisions,
            'path_length': self.path_length,
        }

    def info(self):
        a = self.args
        return {
            'scene': a.scene,
            'num_beams': a.num_beams,
            'max_range': a.max_range,
            'laser_height': a.laser_height,
            'camera_height': a.camera_height,
            'rgb_width': a.rgb_width,
            'rgb_height': a.rgb_height,
            'agent_radius': a.agent_radius,
        }


# ----------------------------------------------------------------------------
# Server loop
# ----------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Habitat-sim server for the ROS 2 bridge')
    p.add_argument('--scene', required=True, help='HM3D scene .basis.glb (navmesh next to it is used)')
    p.add_argument('--dataset-config', default='',
                   help='optional hm3d_*.scene_dataset_config.json')
    p.add_argument('--bind', default='tcp://127.0.0.1:5555')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--start', default='', help='habitat "x,y,z,yaw"; random navigable point if empty')
    p.add_argument('--min-island-radius', type=float, default=3.0)
    p.add_argument('--agent-radius', type=float, default=0.25,
                   help='navmesh radius; keep slightly ABOVE Nav2 robot_radius (0.22)')
    p.add_argument('--start-clearance', type=float, default=0.6,
                   help='min distance from obstacles for a random start pose [m]')
    p.add_argument('--agent-height', type=float, default=0.6)
    p.add_argument('--max-climb', type=float, default=0.1,
                   help='max step height on the navmesh [m]; 0.1 keeps stairs out (single floor)')
    p.add_argument('--no-navmesh-obstacles', dest='navmesh_obstacles', action='store_false',
                   help='do not add navmesh-derived low obstacles/stairs to the scan')
    p.add_argument('--no-recompute-navmesh', dest='recompute_navmesh', action='store_false')
    p.add_argument('--laser-height', type=float, default=0.4, help='scan plane height above the floor [m]')
    p.add_argument('--camera-height', type=float, default=0.5)
    p.add_argument('--num-beams', type=int, default=720)
    p.add_argument('--depth-width', type=int, default=256)
    p.add_argument('--max-range', type=float, default=10.0)
    p.add_argument('--rgb-width', type=int, default=320, help='0 disables the RGB camera')
    p.add_argument('--rgb-height', type=int, default=240)
    p.add_argument('--max-lin-vel', type=float, default=1.0)
    p.add_argument('--max-ang-vel', type=float, default=2.0)
    return p.parse_args(argv)


def main():
    args = parse_args()
    import zmq

    robot = HabitatRobot(args)
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(args.bind)
    print(f'[habitat_server] listening on {args.bind}  (Ctrl+C to quit)')

    running = True

    def _stop(*_):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    t_report = time.time()
    while running:
        if not poller.poll(200):
            continue
        req = json.loads(sock.recv().decode())
        op = req.get('op', 'step')
        try:
            if op == 'info':
                sock.send_multipart([json.dumps(robot.info()).encode(), b'', b''])
                continue
            if op == 'reset':
                robot.reset()
                collided, dt = False, 0.0
            else:
                dt = float(req.get('dt', 0.05))
                collided = robot.step(float(req.get('vx', 0.0)), float(req.get('wz', 0.0)), dt)
            ranges, rgb = robot.observe()
            finite = ranges[np.isfinite(ranges)]
            min_scan = float(finite.min()) if finite.size else float('inf')
            hdr = robot.header(dt, collided)
            sock.send_multipart([json.dumps(hdr).encode(), ranges.tobytes(), rgb])
        except Exception as e:  # always answer, or the REQ socket on the ROS side locks up
            sock.send_multipart([json.dumps({'error': repr(e)}).encode(), b'', b''])
            print(f'[habitat_server] error: {e!r}')
            continue

        if time.time() - t_report > 10.0:
            t_report = time.time()
            print(f'[habitat_server] t_sim={robot.sim_time:7.1f}s  path={robot.path_length:6.1f}m  '
                  f'collisions={robot.collisions}  closest_scan={min_scan:.2f}m')

    print(f'[habitat_server] done. sim_time={robot.sim_time:.1f}s path_length={robot.path_length:.1f}m '
          f'collisions={robot.collisions}')
    sock.close(0)
    ctx.term()
    robot.sim.close()


if __name__ == '__main__':
    main()
