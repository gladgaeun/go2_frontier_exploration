#!/usr/bin/env python3
# Collision guard between Nav2 and the robot: cmd_vel -> [guard] -> cmd_vel_safe (collision_guard.py).
#
# Inputs : cmd_vel (Nav2), depth image + CameraInfo + TF (current frame), voxel_map/obstacles
#          (obstacles seen before, out of the camera view when the robot is close),
#          voxel_map/trav_grid (support height z_r), /plan and navigate_to_pose action status
# Outputs: cmd_vel_safe (forward velocity cut while an obstacle is in the danger zone),
#          collision_guard/block (PointCloud2, map frame): voxels to mark as blocked in the voxel
#          map, sent when navigation goals fail twice at the same spot (after Nav2's own
#          recoveries); collision_guard/tight (PoseStamped): robot pose at a guard stop, the voxel
#          map closes the cells ahead too narrow for the real body (VoxelMapper.mark_tight).
# Escape: on those repeated failures the robot also drives ~1 m back along its own trail, with
# Nav2's commands ignored meanwhile (collision_guard.Trail / follow).

import math

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus, GoalStatusArray
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Header
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker

from trav_exploration.collision_guard import (AbortCounter, Gate, Trail, danger_points, follow, front_row,
                                             next_target)
from trav_exploration.voxel_mapper_node import quat_to_rot


class CollisionGuardNode(Node):

    def __init__(self):
        super().__init__('collision_guard')
        p = self.declare_parameter
        self.map_frame = p('map_frame', 'map').value
        self.base_frame = p('base_frame', 'base_link').value
        self.radius = float(p('robot_radius', 0.25).value)       # real body radius
        # 'full': stop before contact, close cells too narrow for the body, block and back out on
        # repeated Nav2 failures, and follow the 3-D paths of a stair climb. 'path_only': the 3-D
        # paths and the paper's execute-review only (Sec. IV-B conservative map update); Nav2's
        # commands pass through. A blocked robot: voxel_mapper_node stuck event (eq. 17, v_front
        # occupied); a robot whose controller will not drive the way the plan goes (a gap open
        # on M_trav but too narrow for Nav2 to follow: RPP 'collision ahead', recoveries, abort)
        # never commands forward motion, so eq. 17 does not see it: there two aborts at one spot
        # mark the way ahead occupied the same way (tried, failed -> not traversable). Nav2
        # cannot drive the stairs, so the 3-D path follower stays.
        self.mode = str(p('guard_mode', 'full').value)
        if self.mode not in ('full', 'path_only'):
            raise ValueError(f"guard_mode must be 'full' or 'path_only', got {self.mode!r}")
        # look-ahead beyond the body: radius + 0.1 = 0.35 m from the centre, below the 0.4 m goal
        # clearance, so the guard does not stop the robot short of goals placed near walls
        self.distance = float(p('stop_distance', 0.1).value)
        # v_front distance from the centre: map inflation radius (0.22) + 0.15, so the inflated
        # block leaves the robot's own cell free and it can turn away, and the row lies beyond
        # the voxel map's path protection (inflation + 2 cells, VoxelMapper.near_visited)
        self.block_distance = float(p('block_distance', 0.37).value)
        self.h_min = float(p('h_min', 0.1).value)
        # on a 3-D path (stair climb) an obstacle point this low above the support is the next
        # step (its nosing, not labelled stairs by SAM3), which the robot steps onto: the top of
        # the spiral stairs of 00820 held the robot 0.12 m below one for 25 retries
        self.step_clear = float(p('step_clear', 0.25).value)
        self.map_rel = np.zeros(0)
        self.h_max = float(p('h_max', 0.6).value)
        self.stride = int(p('pixel_stride', 8).value)
        self.near = float(p('near_range', 2.0).value)            # points farther away are ignored
        self.res = float(p('block_step', 0.05).value)
        self.min_forward = float(p('min_forward', 0.05).value)   # m/s that counts as driving forward
        self.gate = Gate(hold=float(p('hold_time', 1.0).value))
        # radius: Nav2's backup recovery moves the robot ~0.3 m between attempts; with 0.3 m the
        # aborts of a robot shuttling between two spots 0.32 m apart never counted as one place
        # (10 failed goals, ~5 min, before the escape)
        self.aborts = AbortCounter(n=int(p('abort_count', 2).value),
                                   window=float(p('abort_window', 120.0).value),
                                   radius=float(p('abort_radius', 0.6).value))

        # escape back along the trail: the trail cells were occupied by the robot centre, so only
        # a narrower danger zone is checked (map voxels are up to half a voxel off the surface)
        self.trail = Trail(spacing=0.1)
        self.escape_distance = float(p('escape_distance', 1.0).value)
        self.escape_timeout = float(p('escape_timeout', 20.0).value)
        self.escape_speed = float(p('escape_speed', 0.15).value)
        self.escape_radius = float(p('escape_radius', 0.2).value)
        self.escape_stop = float(p('escape_stop_distance', 0.05).value)
        self.escape = None           # (waypoints, next index, start time, last time not blocked)
        self.blocked_keys = set()    # cells of blocked voxels (virtual, not seen obstacles)
        self.n_escapes = 0
        self.create_timer(0.1, self._escape_step)
        self.create_timer(0.2, self._publish_body)

        self.cmd = (0.0, 0.0)
        self.support_z = None
        self.map_pts = np.zeros((0, 2))      # obstacle points at collision height, map frame (xy)
        self.rays = None
        self.plan = None
        self.seen_aborts = set()
        self.n_stops = self.n_abort_blocks = 0

        self.tf_buffer = Buffer()
        # TF on a node of its own: with spin_thread=True the listener's thread spins the node it is
        # given, so on this node it also ran the (slow) depth callbacks and TF fell seconds behind
        self.tf_node = rclpy.create_node(f'{self.get_name()}_tf', parameter_overrides=[
            Parameter('use_sim_time', value=bool(self.get_parameter('use_sim_time').value))])
        self.tf_listener = TransformListener(self.tf_buffer, self.tf_node, spin_thread=True)
        self.cmd_pub = self.create_publisher(Twist, p('cmd_out', 'cmd_vel_safe').value, 10)
        self.block_pub = self.create_publisher(PointCloud2, 'collision_guard/block', 10)
        self.tight_pub = self.create_publisher(PoseStamped, 'collision_guard/tight', 10)
        # the voxel mapper does not mark stuck events or close cells while the robot escapes
        self.escaping_pub = self.create_publisher(Bool, 'collision_guard/escaping', 10)
        self.body_pub = self.create_publisher(Marker, 'collision_guard/body', 1)
        # 3-D paths of the voxel map (stairs; trav3d.py) are followed like an escape: Nav2's
        # commands are ignored meanwhile and the forward check uses mapped obstacles only (the
        # steps ahead are traversable in mode s, but in the depth frame they look like obstacles)
        self.create_subscription(Path, 'voxel_map/path3d', self._on_path3d, 1)
        self.following_pub = self.create_publisher(Bool, 'collision_guard/following', 10)
        self.escape_kind = 'escape'
        self.progress = None          # (furthest waypoint reached, its time) of the current override
        self.path_speed = self.escape_speed
        self.last_vx = 0.0
        self.create_subscription(Twist, p('cmd_in', 'cmd_vel').value, self._on_cmd, 10)
        self.create_subscription(CameraInfo, 'camera/camera_info', self._on_info, 2)
        self.create_subscription(Image, 'camera/depth/image_raw', self._on_depth, 2)
        # obstacles = occupied voxels outside T: a stair step that is only inflated (railing next
        # to it) is not one
        self.create_subscription(PointCloud2, p('obstacle_topic', 'voxel_map/obstacles').value,
                                 self._on_occupied, 1)
        self.create_subscription(PointCloud2, 'voxel_map/blocked', self._on_blocked, 1)
        grid_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, 'voxel_map/trav_grid', self._on_grid, grid_qos)
        self.create_subscription(Path, 'plan', self._on_plan, 1)
        # Nav2 plans in 2-D (z = 0): for RViz, its paths are shown on the current floor (z_r)
        self.plan_display_pub = self.create_publisher(Path, 'collision_guard/plan_display', 1)
        self.local_plan_display_pub = self.create_publisher(Path, 'collision_guard/local_plan_display', 1)
        self.create_subscription(Path, 'local_plan',
                                 lambda m: self._raise_path(m, self.local_plan_display_pub), 1)
        self.create_subscription(GoalStatusArray, 'navigate_to_pose/_action/status', self._on_status, 10)
        self.get_logger().info(f'collision guard: radius {self.radius} m, stops {self.distance} m ahead')

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # -- velocity gate ------------------------------------------------------------
    def _on_cmd(self, msg):
        self.cmd = (msg.linear.x, msg.angular.z)
        if self.escape is not None:
            return                   # _escape_step drives the robot
        vx, wz = msg.linear.x, msg.angular.z
        if self.mode == 'full':
            vx, wz = self.gate.filter(self._now(), vx, wz)
        out = Twist()
        out.linear.x, out.angular.z = float(vx), float(wz)
        self.cmd_pub.publish(out)

    # -- inputs -------------------------------------------------------------------
    def _on_grid(self, msg):
        self.support_z = msg.info.origin.position.z

    def _on_plan(self, msg):
        self.plan = np.array([[s.pose.position.x, s.pose.position.y] for s in msg.poses]).reshape(-1, 2)
        self._raise_path(msg, self.plan_display_pub)

    def _raise_path(self, msg, pub):
        if self.support_z is None:
            return
        for ps in msg.poses:
            ps.pose.position.z = self.support_z + 0.03
        pub.publish(msg)

    def _on_occupied(self, msg):
        if self.support_z is None:
            return
        pts = np.array([[q[0], q[1], q[2]] for q in
                        point_cloud2.read_points(msg, field_names=('x', 'y', 'z'))]).reshape(-1, 3)
        rel = pts[:, 2] - self.support_z
        keep = (rel > self.h_min) & (rel < self.h_max)
        self.map_pts = pts[keep, :2]
        self.map_rel = rel[keep]

    def _key(self, xy):
        return [tuple(k) for k in np.floor(np.asarray(xy).reshape(-1, 2) / self.res).astype(int)]

    def _on_blocked(self, msg):
        xy = np.array([[q[0], q[1]] for q in point_cloud2.read_points(msg, field_names=('x', 'y'))])
        self.blocked_keys = set(self._key(xy)) if len(xy) else set()

    def _on_info(self, msg):
        if self.rays is not None:
            return
        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        v, u = np.mgrid[0:msg.height:self.stride, 0:msg.width:self.stride]
        self.pix = (v.reshape(-1), u.reshape(-1))
        self.rays = np.stack([(u.reshape(-1) - K[0, 2]) / K[0, 0],
                              (v.reshape(-1) - K[1, 2]) / K[1, 1], np.ones(u.size)], axis=1)

    def _lookup(self, frame, stamp):
        t = self.tf_buffer.lookup_transform(self.map_frame, frame, stamp, timeout=Duration(seconds=0.1))
        tr = t.transform.translation
        return np.array([tr.x, tr.y, tr.z]), quat_to_rot(t.transform.rotation)

    # -- danger check on every depth frame ------------------------------------------
    def _on_depth(self, msg):
        if self.rays is None or self.support_z is None or msg.encoding != '32FC1':
            return
        stamp = Time.from_msg(msg.header.stamp)
        try:
            cam_p, cam_R = self._lookup(msg.header.frame_id, stamp)
            base_p, base_R = self._lookup(self.base_frame, stamp)
        except TransformException:
            return
        self.trail.add(base_p[0], base_p[1])
        if self.mode != 'full':
            return
        depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, -1)[:, :msg.width]
        z = depth[self.pix].astype(np.float64)
        ok = np.isfinite(z) & (z > 0.0) & (z < self.near)
        pts = (self.rays[ok] * z[ok, None]) @ cam_R.T + cam_p
        rel = pts[:, 2] - self.support_z
        frame_pts = pts[(rel > self.h_min) & (rel < self.h_max), :2]

        # map frame -> base frame (xy)
        yaw = math.atan2(base_R[1, 0], base_R[0, 0])
        c, s = math.cos(yaw), math.sin(yaw)

        def to_base(xy):
            d = xy - base_p[:2]
            return np.column_stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]])

        near_map = self.map_pts[np.hypot(*(self.map_pts - base_p[:2]).T) < self.near] \
            if len(self.map_pts) else self.map_pts
        d_frame = danger_points(to_base(frame_pts), self.radius, self.distance)
        d_map = danger_points(to_base(near_map), self.radius, self.distance)
        t = self._now()
        new_stop = self.gate.update(t, len(d_frame) + len(d_map) > 0)
        if new_stop and self.cmd[0] > self.min_forward and self.escape is None:
            self.n_stops += 1
            # The stop means "straight on, the body touches a known obstacle", not "no way through":
            # a whole row across the robot width (inflated again) closed doorways the robot only
            # approached at an angle. The voxel map closes the cells ahead that are too close to
            # obstacles for the real body instead (VoxelMapper.mark_tight). Obstacle points of
            # this frame that may not be in the map yet are marked blocked.
            frame_world = d_frame @ np.array([[c, s], [-s, c]]) + base_p[:2] if len(d_frame) else d_frame
            if len(frame_world):
                self._publish_block(frame_world.reshape(-1, 2), msg.header.stamp)
            pose = PoseStamped(header=Header(stamp=msg.header.stamp, frame_id=self.map_frame))
            pose.pose.position.x, pose.pose.position.y = float(base_p[0]), float(base_p[1])
            pose.pose.orientation.z, pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
            self.tight_pub.publish(pose)
            self.get_logger().warn(
                f'[guard] #{self.n_stops} stop at ({base_p[0]:.2f}, {base_p[1]:.2f}): {len(d_frame)} frame + '
                f'{len(d_map)} map obstacle points within {self.radius + self.distance:.2f} m ahead; '
                f'cells ahead too narrow for the body closed')

    # -- repeated navigation failures -----------------------------------------------
    def _on_status(self, msg):
        for st in msg.status_list:
            gid = bytes(st.goal_info.goal_id.uuid)
            if st.status != GoalStatus.STATUS_ABORTED or gid in self.seen_aborts:
                continue
            self.seen_aborts.add(gid)
            try:
                base_p, base_R = self._lookup(self.base_frame, Time())
            except TransformException:
                return
            if self.escape is not None or not self.aborts.abort(self._now(), base_p[0], base_p[1]):
                continue
            # block the way the plan goes, just in front of the robot body
            yaw = math.atan2(base_R[1, 0], base_R[0, 0])
            if self.plan is not None and len(self.plan):
                d = np.hypot(*(self.plan - base_p[:2]).T)
                ahead = self.plan[d > self.radius + 0.1]
                if len(ahead):
                    yaw = math.atan2(ahead[0, 1] - base_p[1], ahead[0, 0] - base_p[0])
            block = front_row(base_p[0], base_p[1], yaw, self.radius, self.res, self.block_distance)
            self.n_abort_blocks += 1
            self._publish_block(block, self.get_clock().now().to_msg())
            self.get_logger().warn(
                f'[guard] navigation failed {self.aborts.n} times at ({base_p[0]:.2f}, {base_p[1]:.2f}): '
                f'way ahead (heading {math.degrees(yaw):.0f} deg) marked blocked'
                + (' (execute-review)' if self.mode == 'path_only' else ''))
            if self.mode == 'full':                # backing out along the trail: not in the paper
                self._start_escape(base_p)

    # -- escape back along the trail ------------------------------------------------
    def _start_escape(self, base_p):
        wps = self.trail.back(base_p[0], base_p[1], self.escape_distance)
        if not wps:
            return
        t = self._now()
        length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip([tuple(base_p[:2])] + wps, wps))
        # time for the route at escape speed (turns included), at least escape_timeout
        self.escape_limit = max(self.escape_timeout, 2.0 * length / self.escape_speed)
        self.escape = (wps, 0, t, t)
        self.escaping_pub.publish(Bool(data=True))
        self.n_escapes += 1
        self.get_logger().warn(
            f'[escape] #{self.n_escapes} from ({base_p[0]:.2f}, {base_p[1]:.2f}): driving back along the '
            f'trail to ({wps[-1][0]:.2f}, {wps[-1][1]:.2f}), Nav2 commands ignored meanwhile')

    def _on_path3d(self, msg):
        wps = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        if self.escape is not None and self.escape_kind == 'path':
            self._end_override('replaced' if wps else 'cancelled', None)
        if not wps:
            return
        try:
            base_p, _ = self._lookup(self.base_frame, Time())
        except TransformException:
            return
        wps = [w for w in wps if math.hypot(w[0] - base_p[0], w[1] - base_p[1]) > 0.05] or wps[-1:]
        t = self._now()
        length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip([tuple(base_p[:2])] + wps, wps))
        z0, z1 = msg.poses[0].pose.position.z, msg.poses[-1].pose.position.z
        # on the floor (to the foot of the stairs) twice as fast as on the stairs
        self.path_speed = self.escape_speed if abs(z1 - z0) > 0.1 else 2.0 * self.escape_speed
        self.escape_limit = max(self.escape_timeout, 2.0 * length / self.path_speed + 10.0)
        self.escape = (wps, 0, t, t)
        self.escape_kind = 'path'
        self.progress = None
        self.escaping_pub.publish(Bool(data=True))
        self.following_pub.publish(Bool(data=True))
        self.get_logger().info(f'[path3d] following {len(wps)} waypoints, {length:.1f} m, rise {z1 - z0:+.2f} m, '
                               f'Nav2 commands ignored meanwhile')

    def _end_override(self, end, base_p):
        kind, self.escape = self.escape_kind, None
        self.progress = None
        self.escape_kind = 'escape'
        self.escaping_pub.publish(Bool(data=False))
        where = '' if base_p is None else f' at ({base_p[0]:.2f}, {base_p[1]:.2f}, z {base_p[2]:.2f})'
        if kind == 'path':
            self.following_pub.publish(Bool(data=False))
            self.get_logger().info(f'[path3d] {end}{where}; Nav2 drives again')
        else:
            self.get_logger().warn(f'[escape] {end}{where}; Nav2 drives again')

    def _escape_step(self):
        if self.escape is None:
            return
        wps, i, t0, t_free = self.escape
        t = self._now()
        try:
            base_p, base_R = self._lookup(self.base_frame, Time())
        except TransformException:
            return
        yaw = math.atan2(base_R[1, 0], base_R[0, 0])
        path = self.escape_kind == 'path'
        # 3-D paths: a shorter lookahead cuts fewer corners (the robot got caught on the newel
        # post at the foot of the stairs and pushed against it for 30 s)
        i = next_target(wps, i, base_p[0], base_p[1], lookahead=0.25 if path else 0.35)
        vx = wz = 0.0
        end = None
        d_goal = math.hypot(wps[-1][0] - base_p[0], wps[-1][1] - base_p[1])
        # progress = getting further along the path (the end may first get farther away on a
        # winding path); turning in place is not "no progress" (a path starting behind the robot
        # takes ~5 s to face, and was cut after 3 s)
        best, t_best = self.progress if self.progress is not None else (i, t)
        if i > best or self.last_vx == 0.0:
            best, t_best = max(i, best), t
        self.progress = (best, t_best)
        if d_goal < 0.1:
            end = 'done'
        elif t - t0 > self.escape_limit:
            end = 'timed out'
        elif path and t - t_best > 3.0:
            end = 'no progress for 3 s'        # caught on something the map does not show: replan
        else:
            vx, wz = follow(base_p[0], base_p[1], yaw, wps[i],
                            speed=self.path_speed if path else self.escape_speed)
            c, s = math.cos(yaw), math.sin(yaw)
            # path_only (the paper): a 3-D path planned on M_trav is executed as it is; a robot
            # that really gets blocked is the execute-review's (voxel_mapper_node stuck event,
            # eq. 17). This extra check stopped the robot at the start of every stretch when it
            # stood next to a wall (00823: a landing, 0.03 m from the wall, 5 stretches 'blocked'
            # at its own position, no forward command, so no stuck event either).
            check = not (path and self.mode == 'path_only')
            # observed obstacles only: blocked voxels are guesses of where the robot got stuck,
            # and one marked while escaping lay on the trail and stopped every later escape
            d = self.map_pts - base_p[:2] if check else np.zeros((0, 2))
            close = np.hypot(d[:, 0], d[:, 1]) < 1.0 if len(d) else np.zeros(0, dtype=bool)
            if path and len(close) == len(self.map_rel) and len(close):
                close &= self.map_rel >= self.step_clear
            if close.any() and self.blocked_keys:
                idx = np.flatnonzero(close)
                virtual = np.array([k in self.blocked_keys for k in self._key(self.map_pts[idx])])
                close[idx[virtual]] = False
            near = d[close] if len(d) else d
            rel = np.column_stack([c * near[:, 0] + s * near[:, 1], -s * near[:, 0] + c * near[:, 1]])
            if vx > 0.0 and len(danger_points(rel, self.escape_radius, self.escape_stop)):
                vx = 0.0
                if t - t_free > 3.0:
                    end = 'blocked by an obstacle'
            else:
                t_free = t
        out = Twist()
        self.last_vx = float(vx)
        if end is None:
            out.linear.x, out.angular.z = float(vx), float(wz)
            self.escape = (wps, i, t0, t_free)
        else:
            self._end_override(end, base_p)
        self.cmd_pub.publish(out)

    def _publish_body(self):
        """The robot body (radius x collision height) as a cylinder on the floor, in base_link:
        red while the guard holds forward motion back."""
        if self.support_z is None:
            return
        try:
            base_p, _ = self._lookup(self.base_frame, Time())
        except TransformException:
            return
        m = Marker()
        m.header.frame_id = self.base_frame              # stamp 0: drawn at the latest TF
        m.ns, m.id, m.type, m.action = 'body', 0, Marker.CYLINDER, Marker.ADD
        m.pose.position.z = float(self.support_z - base_p[2] + self.h_max / 2)
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = 2.0 * self.radius
        m.scale.z = self.h_max
        danger = self.gate.active(self._now())
        m.color.r, m.color.g, m.color.b, m.color.a = (1.0, 0.2, 0.2, 0.55) if danger else (0.2, 0.5, 1.0, 0.4)
        self.body_pub.publish(m)

    def _publish_block(self, xy, stamp):
        h = Header(stamp=stamp, frame_id=self.map_frame)
        pts = np.column_stack([xy, np.full(len(xy), self.support_z)]).astype(np.float32)
        self.block_pub.publish(point_cloud2.create_cloud_xyz32(h, pts))


def main(args=None):
    rclpy.init(args=args)
    node = CollisionGuardNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
