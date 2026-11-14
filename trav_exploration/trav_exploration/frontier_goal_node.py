#!/usr/bin/env python3
# Frontier goal service on the 3D traversable frontiers: the drop-in counterpart of
# frontier_exploration's classical_frontier_detector for the voxel map.
#
# Inputs : voxel_map/frontier_clusters (centroid p_i + size per cluster, voxel_mapper_node),
#          voxel_map/trav_grid (support layer: free = M_trav, occupied = O_inf), TF map->base_link
# Service: frontier_pose (frontier_interfaces/FrontierGoal), same contract as the 2D detector:
#          the goal_rank-th best frontier goal, or an empty frame_id when there is none.
# So frontier_exploration_node.py (the exploration loop: reachability check, failed/reached goal
# blacklist, termination) runs unchanged on top of it. Goal placement is the 2D algorithm (see
# goal_selection.py). Clusters the robot cannot reach on the Nav2 grid are left out
# (goal_selection.reachable_mask) and shown on frontier_goal/unreachable (grey spheres).
#
# Order (goal_order): 'atsp' (default) is TravExplorer's open ATSP over {x_r} U F_g (tour.py): rank
# 0 is the first node of the tour, rank k the first node of the tour solved again without the
# first k (the explorer asks for the next rank when a goal is blacklisted or has no Nav2 path:
# the paper moves such a frontier to the dormant set and replans). The tour is drawn on
# frontier_goal/tour. 'score' is the 2D ranking (0.5 / distance + 0.5 * size).
# Dormant set (atsp): a frontier still there when the robot stands at its goal cannot be cleared
# by visiting it (floor under a table is hidden by the table top from the camera, so the edge of
# M_g stays). It is left out from then on, also after the explorer restarts with an empty
# blacklist (which is for retrying failed goals); shown as blue spheres on frontier_goal/unreachable.
# Blind-spot frontiers (Sec. IV-D, FOV-aware active perception): a frontier cluster closer to the
# robot than d_blind may lie in the blind spot of the forward-looking camera. Before such a
# frontier can become dormant, the robot pitches the camera down (camera/pitch_cmd, look_down_pitch
# = the 30 deg look_down action of the paper's simulation) for look_time s and back; if the hidden
# voxels were seen the frontier is gone, otherwise it is irreducible and goes to the dormant set.
# Each place is checked once. d_blind is not given in the paper: 1.5 m, about where the level
# camera (0.88 m high, 63 deg vertical FOV) starts to see the floor.
# Look-around (not in the paper): at the start and on arriving on a new floor the robot turns in
# place once (cmd_vel, look_around_speed) before the first plan, so the space behind it is seen.
# Without it a frontier at the edge of the first view (behind the robot) was reached with the
# robot facing away, stayed, and went dormant unseen (HM3D 00810: exploration ended after 2 goals).
#
# Floors (chain of exploration, Sec. VI-A, eq. 9 without target objects): only frontiers on the
# current floor (centroid within floor_band of z_r, the height of trav_grid) are candidates
# (F = frontiers of the current floor). After the frontier ranks, the next rank is the entrance of
# a stair landmark (voxel_map/stair_landmarks) that leads up from this floor and has not been
# climbed: A = O_stair once F is exhausted (or all of F is rejected by the explorer, i.e. dormant).
# This node then asks the voxel map to climb it (voxel_map/climb_at: a 3-D path to its foot, then
# the climb) and answers only when that is done, with no goal, so the exploration loop ends and
# the launch restarts it (empty blacklist) on the new floor.

import math
import threading
import time

import numpy as np
import rclpy
from frontier_interfaces.srv import FrontierGoal
from geometry_msgs.msg import Point, PointStamped, PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import ColorRGBA, Empty, Float64, String
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker

from trav_exploration.goal_selection import (clear_cells, is_reachable, pull_back, reachable_mask,
                                             select_frontier, snap_to_clear_cell)
from trav_exploration.tour import cost_matrix, geo_costs, nearest_node, solve_open_atsp, step_graph
from trav_interfaces.srv import GeoCosts


class FrontierGoalNode(Node):

    def __init__(self):
        super().__init__('frontier_goal')
        p = self.declare_parameter
        self.map_frame = p('map_frame', 'map').value
        self.base_frame = p('base_frame', 'base_link').value
        # 2D detector: goals >= goal_clearance from occupied /map cells (0 = 0.3 m pullback).
        # trav_grid obstacles are O_inf, already inflated by the robot radius (0.22 m), so 0.18
        # keeps the same 0.4 m from real obstacles as the HM3D 2D launch.
        self.goal_clearance = float(p('goal_clearance', 0.18).value)
        self.alpha = float(p('alpha', 0.5).value)
        self.min_dist = float(p('min_distance', 0.4).value)
        self.filter_unreachable = bool(p('filter_unreachable', True).value)
        self.goal_order = p('goal_order', 'atsp').value
        # eq. 14-16 (values not given in the paper). d_obs is measured to the real obstacle:
        # trav_grid obstacles are O_inf, grown by the inflation radius (voxel_mapper robot_radius)
        self.lambda_g = float(p('lambda_g', 0.0).value)
        self.lambda_esdf = float(p('lambda_esdf', 0.3).value)   # = voxel_mapper_node (2-D fallback only)
        self.d_safe = float(p('d_safe', 0.6).value)
        self.inflation_radius = float(p('inflation_radius', 0.22).value)
        self.exact_max = int(p('tour_exact_max', 12).value)
        self.plan = None              # ATSP state for the current rank sequence
        self.reach_radius = float(p('dormant_reach_radius', 0.4).value)
        self.dormant = []             # (centroid xyz, goal xy) of frontiers reached but not cleared
        self.n_plans = 0
        self.d_blind = float(p('d_blind', 1.5).value)
        self.look_pitch = float(p('look_down_pitch', math.radians(30.0)).value)   # 0 = off
        self.look_time = float(p('look_time', 2.0).value)                        # s (wall clock)
        self.looked = []              # centroids of the blind-spot frontiers already looked at
        self.pitch_pub = self.create_publisher(Float64, 'camera/pitch_cmd', 1)
        self.look_around = bool(p('look_around', True).value)
        self.look_around_speed = float(p('look_around_speed', 0.6).value)    # rad/s
        self.need_look_around = self.look_around
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 1)
        # floors and stairs
        self.floor_band = float(p('floor_band', 0.3).value)          # frontier centroid vs z_r
        self.stair_rise = float(p('stair_rise', 1.0).value)          # a landmark this high above z_r leads up
        self.climb_timeout = float(p('climb_timeout', 600.0).value)  # s (wall clock)
        self.z_r = None
        self.landmarks = []           # (entrance x, y, z, top x, y, z, size)
        self.climbed = []             # entrance xy of the stairs climbed (or given up)
        self.climb_done = threading.Event()
        self.climb_z = None

        self.clusters = []
        self.grid = None
        self.tf_buffer = Buffer()
        # TF on a node of its own: with spin_thread=True the listener's thread spins the node it is
        # given, so on this node it also ran the (slow) depth callbacks and TF fell seconds behind
        self.tf_node = rclpy.create_node(f'{self.get_name()}_tf', parameter_overrides=[
            Parameter('use_sim_time', value=bool(self.get_parameter('use_sim_time').value))])
        self.tf_listener = TransformListener(self.tf_buffer, self.tf_node, spin_thread=True)
        # the service may wait for a stair climb: inputs keep coming in on other threads
        inputs = ReentrantCallbackGroup()
        self.create_subscription(PointCloud2, 'voxel_map/frontier_clusters', self._on_clusters, 1,
                                 callback_group=inputs)
        grid_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, 'voxel_map/trav_grid', self._on_grid, grid_qos,
                                 callback_group=inputs)
        self.create_subscription(PointCloud2, 'voxel_map/stair_landmarks', self._on_landmarks, 1,
                                 callback_group=inputs)
        self.create_subscription(String, 'voxel_map/climb_state', self._on_climb_state, 10,
                                 callback_group=inputs)
        self.climb_pub = self.create_publisher(PointStamped, 'voxel_map/climb_at', 1)
        self.create_service(FrontierGoal, 'frontier_pose', self._on_request,
                            callback_group=MutuallyExclusiveCallbackGroup())
        self.unreachable_pub = self.create_publisher(Marker, 'frontier_goal/unreachable', 1)
        self.tour_pub = self.create_publisher(Marker, 'frontier_goal/tour', 1)
        self.reach = None             # (time, robot xy, reachable, unreachable clusters) cache
        # J_geo and reachability on the 3-D support graph of the map (voxel_mapper_node,
        # trav3d.SupportGraph, Sec. VI-B), as the paths Nav2 follows; the 2-D trav_grid graph only
        # when that service does not answer (logged)
        self.geo_cli = self.create_client(GeoCosts, 'voxel_map/geo_costs',
                                          callback_group=MutuallyExclusiveCallbackGroup())
        self.geo_warned = False
        self.create_timer(1.0, self._publish_unreachable, callback_group=inputs)
        self.get_logger().info(f'frontier goals from voxel_map/frontier_clusters, order {self.goal_order}, '
                               f'goal_clearance {self.goal_clearance} m')

    def _on_clusters(self, msg):
        self.clusters = [tuple(map(float, c)) for c in
                         point_cloud2.read_points(msg, field_names=('x', 'y', 'z', 'size'))]

    def _on_grid(self, msg):
        g = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        self.grid = (g, (msg.info.origin.position.x, msg.info.origin.position.y), msg.info.resolution)
        self.z_r = msg.info.origin.position.z              # the support plane of this slice

    def _on_landmarks(self, msg):
        self.landmarks = [tuple(map(float, lm)) for lm in point_cloud2.read_points(
            msg, field_names=('x', 'y', 'z', 'top_x', 'top_y', 'top_z', 'size'))]

    def _on_climb_state(self, msg):
        if msg.data.startswith('done'):
            self.climb_z = float(msg.data.split()[1])
            self.climb_done.set()

    def _floor_clusters(self):
        """Frontier clusters of the current floor (centroid within floor_band of z_r)."""
        if self.z_r is None:
            return list(self.clusters)
        return [c for c in self.clusters if abs(c[2] - self.z_r) <= self.floor_band]

    def _stairs_up(self):
        """Stair landmarks whose foot is on this floor and that lead up, not climbed yet."""
        if self.z_r is None:
            return []
        return [lm for lm in self.landmarks
                if abs(lm[2] - self.z_r) <= 0.4 and lm[5] > self.z_r + self.stair_rise
                and not any(math.hypot(lm[0] - c[0], lm[1] - c[1]) < 1.0 for c in self.climbed)]
    def _robot_xy(self):
        try:
            t = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, Time(),
                                                timeout=Duration(seconds=1.0))
            return t.transform.translation.x, t.transform.translation.y
        except TransformException as e:
            self.get_logger().warn(f'TF lookup failed in get_frontiers: {e}')
            return 0.0, 0.0

    def _geo_costs(self, sources, targets):
        """J_geo [len(sources), len(targets)] on the 3-D floor graph (voxel_map/geo_costs), or None."""
        if not sources or not targets:
            return np.zeros((len(sources), len(targets)))
        if not self.geo_cli.service_is_ready():
            return None
        req = GeoCosts.Request()
        req.sources = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in sources]
        req.targets = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in targets]
        fut = self.geo_cli.call_async(req)
        done = threading.Event()
        fut.add_done_callback(lambda _f: done.set())
        if not done.wait(10.0) or fut.result() is None or not fut.result().ok:
            return None
        return np.array(fut.result().costs, dtype=np.float64).reshape(len(sources), len(targets))

    def _warn_2d(self):
        if not self.geo_warned:
            self.geo_warned = True
            self.get_logger().warn('voxel_map/geo_costs (3-D floor graph) not answering: J_geo and '
                                   'reachability from the 2-D trav_grid instead')

    def _split(self, rx, ry):
        """(reachable, unreachable) clusters; cached for 1 s (the explorer asks rank after rank).
        Reachable: a path on the 3-D floor graph from the robot to the cluster's goal."""
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.reach is not None and now - self.reach[0] < 1.0 and \
                abs(self.reach[1][0] - rx) + abs(self.reach[1][1] - ry) < 0.1:
            return self.reach[2], self.reach[3]
        clusters = self._floor_clusters()
        if not self.filter_unreachable or self.grid is None:
            return clusters, []
        z = self.z_r if self.z_r is not None else 0.0
        clear = clear_cells(self.grid[0], self.grid[2], self.goal_clearance) if self.goal_clearance > 0.0 else None
        goals = [self._place(c, rx, ry, clear) for c in clusters]
        # goals lie on trav_grid, the floor at z_r (floor_z): the graph holds every floor, a goal
        # at the centroid's height (up to floor_band off) missed its floor node
        J = self._geo_costs([(rx, ry, z)], [(gl[0], gl[1], z) for gl in goals])
        if J is not None:
            ok = [c for c, j in zip(clusters, J[0]) if np.isfinite(j)]
            bad = [c for c, j in zip(clusters, J[0]) if not np.isfinite(j)]
            self.reach = (now, (rx, ry), ok, bad)
            return ok, bad
        self._warn_2d()
        g, origin, res = self.grid
        mask = reachable_mask(g, rx, ry, origin, res)
        ok, bad = [], []
        for c in clusters:
            (ok if is_reachable(mask, c[0], c[1], origin, res) else bad).append(c)
        self.reach = (now, (rx, ry), ok, bad)
        return ok, bad

    def _is_dormant(self, c, goal):
        # same floor only: the upper floor has frontiers at the xy of dormant ones below
        return any(abs(c[2] - d[0][2]) < 0.5 and
                   (math.hypot(c[0] - d[0][0], c[1] - d[0][1]) < 0.5 or
                    math.hypot(goal[0] - d[1][0], goal[1] - d[1][1]) < 0.3) for d in self.dormant)

    def _robot_yaw(self):
        try:
            q = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, Time(),
                                                timeout=Duration(seconds=1.0)).transform.rotation
            return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        except TransformException:
            return None

    def _turn_around(self):
        """Turn in place once (the explorer waits for this service, Nav2 is idle)."""
        self.need_look_around = False
        yaw = self._robot_yaw()
        if yaw is None:
            return
        self.get_logger().info(f'[look-around] turning in place once ({self.look_around_speed:.1f} rad/s)')
        turned, t0 = 0.0, time.time()
        timeout = 2.0 * 2.0 * math.pi / max(self.look_around_speed, 0.1) + 5.0
        cmd = Twist()
        cmd.angular.z = self.look_around_speed
        while turned < 2.0 * math.pi and time.time() - t0 < timeout:
            self.cmd_pub.publish(cmd)
            time.sleep(0.1)
            y = self._robot_yaw()
            if y is not None:
                turned += abs(math.atan2(math.sin(y - yaw), math.cos(y - yaw)))
                yaw = y
        self.cmd_pub.publish(Twist())
        time.sleep(1.0)                  # the last frames reach the map and the frontier clusters
        self.reach = None
        self.get_logger().info(f'[look-around] done: turned {math.degrees(turned):.0f} deg '
                               f'in {time.time() - t0:.1f} s')

    def _look_down(self, rx, ry, reachable):
        """Blind-spot check (Sec. IV-D) before the dormant check of a new plan: look down once at
        the frontiers within d_blind; returns the reachable clusters after the look."""
        if self.look_pitch <= 0.0 or self.n_plans == 0:
            return reachable
        near = [c for c in reachable
                if math.hypot(c[0] - rx, c[1] - ry) < self.d_blind
                and not any(abs(c[2] - l[2]) < 0.5 and math.hypot(c[0] - l[0], c[1] - l[1]) < 0.5
                            for l in self.looked)
                and not self._is_dormant(c, (c[0], c[1]))]
        if not near:
            return reachable
        self.get_logger().info(
            f'[blind] {len(near)} frontier(s) within {self.d_blind:.1f} m: looking down '
            f'{math.degrees(self.look_pitch):.0f} deg for {self.look_time:.1f} s')
        self.pitch_pub.publish(Float64(data=self.look_pitch))
        time.sleep(self.look_time)
        self.pitch_pub.publish(Float64(data=0.0))
        time.sleep(0.5)
        self.looked.extend((c[0], c[1], c[2]) for c in near)
        self.reach = None
        after, _ = self._split(rx, ry)
        left = [c for c in near if any(abs(c[2] - a[2]) < 0.5 and math.hypot(c[0] - a[0], c[1] - a[1]) < 0.5
                                       for a in after)]
        self.get_logger().info(f'[blind] after looking down: {len(near) - len(left)} of {len(near)} cleared')
        return after

    def _publish_dormant(self):
        m = Marker()
        m.header.frame_id = self.map_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id, m.type = 'dormant', 1, Marker.SPHERE_LIST
        m.action = Marker.ADD if self.dormant else Marker.DELETE
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.25
        m.color = ColorRGBA(r=0.2, g=0.4, b=1.0, a=0.9)
        for (cx, cy, cz), _ in self.dormant:
            q = Point()
            q.x, q.y, q.z = cx, cy, cz
            m.points.append(q)
        self.unreachable_pub.publish(m)

    def _publish_unreachable(self):
        if self.dormant:
            self._publish_dormant()
        if not self.clusters:
            return
        _, bad = self._split(*self._robot_xy())
        m = Marker()
        m.header.frame_id = self.map_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id, m.type = 'unreachable', 0, Marker.SPHERE_LIST
        m.action = Marker.ADD if bad else Marker.DELETE
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.25
        m.color = ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.9)
        for c in bad:
            q = Point()
            q.x, q.y, q.z = c[0], c[1], c[2]
            m.points.append(q)
        self.unreachable_pub.publish(m)

    def _place(self, c, rx, ry, clear=None):
        """Goal position for cluster c (2D placement: snapped to goal_clearance, or pulled back)."""
        gx, gy = c[0], c[1]
        if self.goal_clearance > 0.0:
            snapped = None
            if self.grid is not None:
                snapped = snap_to_clear_cell(*self.grid, gx, gy, self.goal_clearance, clear=clear)
            if snapped is None:
                return gx, gy, False
            return snapped[0], snapped[1], True
        gx, gy = pull_back(gx, gy, rx, ry)
        return gx, gy, True

    def _new_plan(self, rx, ry, reachable):
        """Goals and the ATSP cost matrix over {x_r} U F_g (tour.py)."""
        plan = {'goals': [], 'C': None, 'left': [], 'heads': []}
        if not reachable or self.grid is None:
            return plan
        g, origin, res = self.grid
        clear = clear_cells(g, res, self.goal_clearance) if self.goal_clearance > 0.0 else None
        placed = [(c, self._place(c, rx, ry, clear)) for c in reachable]
        # dormant set: the robot stands at the goal of a frontier that is still there (the
        # explorer asks for rank 0 after each navigation, 1 s after it ended)
        if self.n_plans > 0:
            for c, gl in placed:
                if math.hypot(gl[0] - rx, gl[1] - ry) < self.reach_radius and not self._is_dormant(c, gl):
                    self.dormant.append(((c[0], c[1], c[2]), (gl[0], gl[1])))
                    self.get_logger().info(
                        f'[dormant] frontier at ({c[0]:.2f}, {c[1]:.2f}) still there with the robot at its '
                        f'goal: left out from now on ({len(self.dormant)} dormant)')
        self.n_plans += 1
        n_all = len(placed)
        placed = [(c, gl) for c, gl in placed if not self._is_dormant(c, gl)]
        if len(placed) < n_all:
            self.get_logger().info(f'{n_all - len(placed)} reachable frontier clusters dormant; left out')
        # the 2D rule: clusters within min_distance are skipped unless none is farther away
        far = [(c, gl) for c, gl in placed if math.hypot(c[0] - rx, c[1] - ry) >= self.min_dist]
        placed = far or placed
        if not placed:
            return plan
        cands = [c for c, _ in placed]
        goals = [gl for _, gl in placed]
        z = self.z_r if self.z_r is not None else 0.0
        pts = [(gl[0], gl[1], z) for gl in goals]          # on trav_grid, at floor_z (see _split)
        J = self._geo_costs([(rx, ry, z)] + pts, pts)
        if J is not None:
            plan.update(goals=goals, C=cost_matrix(J, self.lambda_g), left=list(range(1, len(goals) + 1)))
            return plan
        self._warn_2d()
        graph, index, _ = step_graph(g, res, self.lambda_esdf, self.d_safe, self.inflation_radius)
        robot = nearest_node(index, rx, ry, origin, res, 0.5)
        nodes = []
        for (gx, gy, _), c in zip(goals, cands):
            v = nearest_node(index, gx, gy, origin, res, 0.4)
            nodes.append(v if v >= 0 else nearest_node(index, c[0], c[1], origin, res, 0.4))
        J = geo_costs(graph, [robot] + nodes, nodes)
        plan.update(goals=goals, C=cost_matrix(J, self.lambda_g), left=list(range(1, len(goals) + 1)))
        return plan

    def _tour_goal(self, rank, rx, ry, reachable):
        """Goal of the given rank: first node of the open tour without the first `rank` heads."""
        if rank == 0 or self.plan is None:
            self.plan = self._new_plan(rx, ry, reachable)
            self._solve_next()
            if self.plan['heads']:
                C, order = self.plan['C'], self.plan['order0']
                steps = [C[a, b] for a, b in zip([0] + order[:-1], order)]
                no_path = int(np.isinf(C[0, order]).sum())
                self.get_logger().info(
                    f'tour over {len(order)} frontiers: first J = {C[0, order[0]]:.2f}, '
                    f'tour J = {sum(c for c in steps if np.isfinite(c)):.2f}'
                    + (f', {no_path} without a path from the robot (M_inf, last)' if no_path else ''))
                self._publish_tour(order)
        while len(self.plan['heads']) <= rank and self.plan['left']:
            self._solve_next()
        if rank >= len(self.plan['heads']):
            return None
        return self.plan['goals'][self.plan['heads'][rank] - 1]

    def _solve_next(self):
        plan = self.plan
        if not plan['left']:
            return
        idx = [0] + plan['left']
        order = solve_open_atsp(plan['C'][np.ix_(idx, idx)], self.exact_max)
        order = [idx[k] for k in order]
        if not plan['heads']:
            plan['order0'] = order
        plan['heads'].append(order[0])
        plan['left'].remove(order[0])

    def _publish_tour(self, order):
        rx, ry = self._robot_xy()
        m = Marker()
        m.header.frame_id = self.map_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = 'tour', 0, Marker.LINE_STRIP, Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.04
        m.color = ColorRGBA(r=1.0, g=0.6, b=0.0, a=0.9)
        z = self.z_r if self.z_r is not None else 0.0      # on the current floor
        for x, y in [(rx, ry)] + [self.plan['goals'][j - 1][:2] for j in order]:
            q = Point()
            q.x, q.y, q.z = float(x), float(y), float(z)
            m.points.append(q)
        self.tour_pub.publish(m)

    def _stair_goal(self, rx, ry):
        """F exhausted: climb the nearest stair landmark leading up (A = O_stair). The voxel map
        drives the robot to its foot on a 3-D path and climbs (voxel_map/climb_at); this waits for
        it and returns None (no goal: the exploration loop ends and restarts on the new floor).
        Sending the foot as a Nav2 goal instead failed: the explorer blacklists reached goals and
        meanwhile new frontiers took the robot away again."""
        ups = self._stairs_up()
        if not ups:
            return None
        lm = min(ups, key=lambda l: math.hypot(l[0] - rx, l[1] - ry))
        z0 = self.z_r
        self.climb_done.clear()
        self.get_logger().info(f'[stairs] no frontier left on this floor (z_r {z0:.2f}): climbing the stairs at '
                               f'({lm[0]:.2f}, {lm[1]:.2f}), top {lm[5]:.2f} m')
        msg = PointStamped()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x, msg.point.y, msg.point.z = lm[0], lm[1], lm[2]
        self.climb_pub.publish(msg)
        done = self.climb_done.wait(self.climb_timeout)
        self.climbed.append((lm[0], lm[1]))
        z1 = self.climb_z if done else None
        if done and z1 is not None and z1 > z0 + 0.5 * self.stair_rise:
            self.get_logger().info(f'[stairs] climbed: z_r {z0:.2f} -> {z1:.2f}; exploring the new floor')
            self.need_look_around = self.look_around
        else:
            self.get_logger().warn(f'[stairs] climb did not reach a new floor ({"timed out" if not done else z1}); '
                                   f'these stairs are left out')
        self.reach = None
        self.plan = None
        return None

    def _on_request(self, request, response):
        self.get_logger().info(f'Received request, {request.goal_rank}')
        rx, ry = self._robot_xy()
        reachable, unreachable = self._split(rx, ry)
        if request.goal_rank == 0 and unreachable:
            self.get_logger().info(f'{len(unreachable)} of {len(self._floor_clusters())} frontier clusters '
                                   f'of this floor not reachable on the floor graph; left out')

        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.orientation.w = 1.0
        if self.goal_order == 'atsp':
            if request.goal_rank == 0 and self.need_look_around:
                self._turn_around()
                rx, ry = self._robot_xy()
                reachable, unreachable = self._split(rx, ry)
            if request.goal_rank == 0:
                reachable = self._look_down(rx, ry, reachable)
            goal = self._tour_goal(request.goal_rank, rx, ry, reachable)
            if goal is None and request.goal_rank == len(self.plan['heads']):
                goal = self._stair_goal(rx, ry)          # F exhausted: A = O_stair (eq. 9)
            if goal is not None and not goal[2]:
                self.get_logger().warn(
                    f'No free cell with {self.goal_clearance:.2f} m clearance near frontier '
                    f'({goal[0]:.2f}, {goal[1]:.2f}); sending it as is.')
            if goal is None:
                pose.header.frame_id = ''
            else:
                pose.header.frame_id = self.map_frame
                pose.pose.position.x, pose.pose.position.y = float(goal[0]), float(goal[1])
            response.goal_pose = pose
            self.get_logger().info(f'Sending goal x: {pose.pose.position.x:f} y: {pose.pose.position.y:f}.')
            return response

        goal = select_frontier(reachable, request.goal_rank, rx, ry, self.alpha, self.min_dist)
        if goal is None:
            pose.header.frame_id = ''                  # no frontier for this rank
        else:
            gx, gy = goal[0], goal[1]
            if self.goal_clearance > 0.0:
                snapped = None
                if self.grid is not None:
                    snapped = snap_to_clear_cell(*self.grid, gx, gy, self.goal_clearance)
                if snapped is None:
                    self.get_logger().warn(
                        f'No free cell with {self.goal_clearance:.2f} m clearance near frontier '
                        f'({gx:.2f}, {gy:.2f}); sending it as is.')
                else:
                    gx, gy = snapped
            else:
                gx, gy = pull_back(gx, gy, rx, ry)
            pose.header.frame_id = self.map_frame
            pose.pose.position.x, pose.pose.position.y = float(gx), float(gy)
        response.goal_pose = pose
        self.get_logger().info(f'Sending goal x: {pose.pose.position.x:f} y: {pose.pose.position.y:f}.')
        return response


def main(args=None):
    rclpy.init(args=args)
    node = FrontierGoalNode()
    executor = MultiThreadedExecutor()   # the service waits for a climb while inputs keep coming
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
