#!/usr/bin/env python3
# 3D voxel map (occupancy + traversability) from RGB-D frames, see voxel_mapper.py.
#
# Subscribes to a depth image + CameraInfo and looks up the camera and robot poses in TF, so it
# is not tied to Habitat: any registered depth camera works (32FC1 in m, or 16UC1 in mm).
# Publishes (map frame):
#   voxel_map/occupied     PointCloud2  occupied voxels (l >= l_occ) that are not in M_trav, up to
#                                       display_max_height above the floor (no ceiling)
#   voxel_map/obstacles    PointCloud2  occupied voxels outside T (the inflation sources), up to
#                                       display_max_height: what the robot body can hit (collision
#                                       guard). Unlike voxel_map/occupied it leaves out support
#                                       surfaces that are only inflated, e.g. a stair step next to
#                                       a railing, which stopped every stair climb
#   voxel_map/traversable  PointCloud2  M_trav = T \ O_inf, without the frontier voxels
#   voxel_map/frontiers    PointCloud2  traversable frontier voxels (frontier3d.py, Sec. III-C)
#   voxel_map/frontier_centroids  Marker  cluster centroids p_i
#   voxel_map/frontier_clusters   PointCloud2 one point per cluster: centroid p_i + field size
#                                       (input of frontier_goal_node)
#   voxel_map/inflated     PointCloud2  O_inf on the robot support layer
#   voxel_map/trav_grid    OccupancyGrid support layer: free = M_trav, occupied = O_inf
#   voxel_map/blocked      PointCloud2  voxels marked by stuck events (Nav2 obstacle layer input)
#   voxel_map/semantic     PointCloud2  M_s without stairs (semantic traversability, Sec. III-A)
#   voxel_map/stairs       PointCloud2  stair-class voxels of M_s
#   voxel_map/mode         String       traversability mode 'g' / 's' (latched)
#   voxel_map/mode_marker  Marker       the mode as text above the robot
#
# Semantic mode: a class image registered with the depth camera (camera/semantic/image_raw, mono8,
# codes in semantic.py; habitat_server --semantic) is paired with the depth frame by stamp. Pixels
# of traversable classes go into M_s, stair pixels also into the stair voxels, and ModeSwitch
# switches between g and s (eq. 1). Without class images the node stays in its start mode.
# The three display clouds do not overlap: floor voxels are occupied and traversable, frontier
# voxels are traversable, and each voxel is drawn once (occupied grey, traversable, frontier red).
# trav_grid has the same format as /map, so classical_frontier_detector can take its frontiers
# from it (launch argument frontier_map:=voxel).
#
# collision_guard/block (collision_guard_node: guard stops, repeated controller aborts) is
# marked the same way as a stuck event.
#
# Stuck detection (eq. 17): /odom is compared with the motion predicted from /cmd_vel. When a
# forward motion is blocked, the voxels in front of the robot (one row across its width) are
# marked as blocked, which removes them from M_trav and puts them on voxel_map/blocked so Nav2
# replans around them. Frontiers behind the blockage end up in the explorer's failed-goal list
# when Nav2 gives up on them (the dormant set).
#
# Support plane z_r: base_link is not reliably on the floor (in Habitat the agent sits on the
# navmesh, which floats up to ~0.2 m above the mesh floor because of the navmesh cell height),
# so z_r is estimated from the floor points of the depth image and the grid is created at the
# first estimate, with the floor at the centre of a voxel layer.

import math
import threading
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration as DurationMsg
from geometry_msgs.msg import Point, PointStamped, PoseStamped, Twist, TwistStamped
from nav2_msgs.action import ComputePathToPose
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.action import ActionServer
from rclpy.duration import Duration
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, ColorRGBA, Empty, Header, String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker

from trav_exploration.collision_guard import front_row
from trav_exploration.frontier3d import FrontierManager
from trav_exploration import trav3d
from trav_exploration.semantic import ModeSwitch, split_classes
from trav_exploration.voxel_mapper import StuckDetector, VoxelMapper
from trav_interfaces.srv import GeoCosts


def quat_to_rot(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class VoxelMapperNode(Node):

    def __init__(self):
        super().__init__('voxel_mapper')
        p = self.declare_parameter
        self.map_frame = p('map_frame', 'map').value
        self.base_frame = p('base_frame', 'base_link').value
        self.integrate_period = float(p('integrate_period', 0.5).value)   # sim s between frames
        self.publish_period = float(p('publish_period', 1.0).value)
        self.stride = int(p('pixel_stride', 4).value)
        self.min_range = float(p('min_range', 0.1).value)
        self.max_range = float(p('max_range', 5.0).value)
        self.mapper_args = dict(
            # 0.05 m: a 0.1 m grid widens every wall by up to a voxel and closed doorways
            # the robot fits through. 30 m x 30 m around the start, floor -0.25 m .. +2 m
            # (~17 M voxels, the same memory as 60 m x 60 m at 0.1 m).
            resolution=float(p('resolution', 0.05).value),
            size_xy=float(p('size_xy', 30.0).value),
            z_below=float(p('z_below', 0.25).value),
            z_above=float(p('z_above', 2.0).value),
            robot_radius=float(p('robot_radius', 0.25).value),
            h_min=float(p('h_min', 0.1).value),
            h_max=float(p('h_max', 0.6).value),
            mode=str(p('mode', 'g').value))
        self.mapper = None          # created at the first floor estimate
        # floor points: within floor_radius of the robot and below base_link + floor_max_above
        self.floor_radius = float(p('floor_radius', 3.0).value)
        self.floor_max_above = float(p('floor_max_above', 0.1).value)
        self.floor_max_jump = float(p('floor_max_jump', 0.15).value)
        self.floor_min_points = int(p('floor_min_points', 50).value)
        # Single floor, geometric mode: z_r is fixed at the first estimate. Tracking it lets it
        # drift by a voxel layer near walls (few floor points), and the support layer with it.
        self.track_floor = bool(p('track_floor', False).value)
        # follow_support: z_r is the mapped surface under the robot (VoxelMapper.support_height:
        # stair steps, other floors), quantized to voxel layers so it does not drift on a floor.
        # The navmesh is not a fixed height above the mesh floor (0.165 m on the ground floor of
        # 00800, 0.055 m upstairs, 0.03-0.2 m on the stairs), so base_link alone cannot give it.
        self.follow_support = bool(p('follow_support', True).value)
        self.last_base_z = None
        # traversable frontiers (Sec. III-C); K_h = 0 -> robot height in layers
        # min cluster size 3 voxels (0.15 m): frontiers lie on M_trav, which is O_inf-shrunk by
        # the robot radius on both sides of a doorway, so a 0.85 m door leaves a ~0.3 m frontier
        # (the 2D stack's 0.4 m threshold was for the uninflated map and dropped it)
        self.frontier_args = dict(min_size=int(p('frontier_min_size', 3).value),
                                  headroom_layers=int(p('frontier_headroom_layers', 0).value),
                                  boundary=str(p('frontier_boundary', 'source').value))
        self.frontiers = None
        self.t_frontier = 0.0
        # display only: occupied voxels above this height over the floor (ceiling) are not
        # published, or they hide everything in RViz's top-down view
        self.display_max_height = float(p('display_max_height', 1.5).value)
        # stuck detection (eq. 17)
        # Nav2 RPP slows down to 0.07-0.15 m/s near obstacles (cost regulation), which is exactly
        # where the robot runs into gaps narrower than the map shows (thin chair legs): the 1 s
        # window covers >= 0.05 m of commanded motion there, and delta shrinks with it (ratio).
        # Normal driving in Habitat leaves up to ~0.08 m between command and odometry over 2 s.
        # With the collision guard, cmd_vel must be the gated command (launch remaps it).
        self.stuck = StuckDetector(window=float(p('stuck_window', 1.0).value),
                                   delta=float(p('stuck_delta', 0.15).value),
                                   min_dist=float(p('stuck_min_dist', 0.05).value),
                                   ratio=float(p('stuck_ratio', 0.5).value)) \
            if bool(p('stuck_detection', True).value) else None
        # real body radius (habitat_server --agent-radius): mark_tight closes cells this close to
        # obstacles in front of the robot at a guard stop or stuck event
        self.body_radius = float(p('body_radius', 0.25).value)
        self.cmd_timeout = float(p('cmd_timeout', 0.5).value)   # = habitat_bridge cmd_timeout
        # semantic traversability and the g/s mode switch (semantic.py)
        self.switch = ModeSwitch(enter_time=float(p('mode_enter_time', 0.5).value),
                                 exit_time=float(p('mode_exit_time', 1.0).value),
                                 min_stair_points=int(p('mode_min_stair_points', 50).value),
                                 enter_range=float(p('mode_enter_range', 4.0).value),
                                 exit_distance=float(p('mode_exit_distance', 4.5).value),
                                 mode=self.mapper_args['mode']) \
            if bool(p('mode_switch', True).value) else None
        # class image and depth frame of one stamp are paired whichever comes first: the ground
        # truth (habitat_server --semantic) arrives before the depth frame, SAM3 ~0.2 s after it
        self.sem_cache = {}         # stamp (ns) -> class image, waiting for its depth frame
        self.depth_cache = {}       # stamp (ns) -> (depth image, frame id), waiting for its class image
        self.last_sem_integrate = None
        # stairs trusted as support before climbing (VoxelMapper.trusted_stairs): up to
        # stair_window above the floor, up to stair_column_gap above the lowest surface below
        self.stair_window = float(p('stair_window', 0.6).value)
        self.stair_column_gap = float(p('stair_column_gap', 0.3).value)
        # M_s takes a voxel per depth point, so the points must be denser than the voxels: with
        # every 4th pixel (the occupancy stride) the floor 2 m away is sampled every 5.6 cm along
        # the view and M_s was striped with holes, each a frontier. Every pixel up to 3.5 m keeps
        # the spacing <= 3.9 cm (camera 0.88 m above the floor, hfov 79 deg, 640 px).
        self.sem_stride = int(p('semantic_stride', 1).value)
        self.sem_max_range = float(p('semantic_max_range', 3.5).value)
        self.sem_rays = None
        self.n_sem_frames = 0
        self.n_stair_raw = self.n_stair_kept = 0      # stair points seen / trusted (log)
        self.cmd = (0.0, 0.0)
        self.cmd_time = -1e9
        self.n_stuck = 0
        self.escaping = False

        self.K = None
        self.rays = None            # unit-depth rays of the sub-sampled pixels, camera frame
        self.last_integrate = None
        self.last_publish = None
        self.support_z = None
        # height of the floor the robot explores: follows z_r on that floor (within floor_hold of
        # the height on arrival) and while climbing, not when z_r jumps without a climb. At the foot of the stairs z_r read the
        # first step (0.25 m on the ground floor of 00810): the floor's frontiers were no longer "on
        # this floor" and the climb started with the floor half explored. The floor graph,
        # trav_grid and the planner use it.
        self.floor_z = None
        self.floor_anchor = None    # floor_z when the robot arrived on this floor
        self.floor_hold = float(p('floor_hold', 0.15).value)
        self.frames = 0
        self.t_total = 0.0

        self.tf_buffer = Buffer()
        # TF on a node of its own: with spin_thread=True the listener's thread spins the node it is
        # given, so on this node it also ran the (slow) depth callbacks and TF fell seconds behind
        self.tf_node = rclpy.create_node(f'{self.get_name()}_tf', parameter_overrides=[
            Parameter('use_sim_time', value=bool(self.get_parameter('use_sim_time').value))])
        self.tf_listener = TransformListener(self.tf_buffer, self.tf_node, spin_thread=True)
        self.create_subscription(CameraInfo, p('camera_info_topic', 'camera/camera_info').value,
                                 self._on_info, 2)
        self.create_subscription(Image, p('semantic_topic', 'camera/semantic/image_raw').value,
                                 self._on_semantic, 10)
        self.create_subscription(Image, p('depth_topic', 'camera/depth/image_raw').value,
                                 self._on_depth, 2)
        self.occ_pub = self.create_publisher(PointCloud2, 'voxel_map/occupied', 1)
        self.obst_pub = self.create_publisher(PointCloud2, 'voxel_map/obstacles', 1)
        self.trav_pub = self.create_publisher(PointCloud2, 'voxel_map/traversable', 1)
        self.inf_pub = self.create_publisher(PointCloud2, 'voxel_map/inflated', 1)
        grid_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.grid_pub = self.create_publisher(OccupancyGrid, 'voxel_map/trav_grid', grid_qos)
        self.sem_pub = self.create_publisher(PointCloud2, 'voxel_map/semantic', 1)
        self.stairs_pub = self.create_publisher(PointCloud2, 'voxel_map/stairs', 1)
        self.mode_pub = self.create_publisher(String, 'voxel_map/mode', grid_qos)
        self.mode_marker_pub = self.create_publisher(Marker, 'voxel_map/mode_marker', 1)
        self.mode_pub.publish(String(data=self.mapper_args['mode']))
        self.blocked_pub = self.create_publisher(PointCloud2, 'voxel_map/blocked', 1)
        self.frontier_pub = self.create_publisher(PointCloud2, 'voxel_map/frontiers', 1)
        self.centroid_pub = self.create_publisher(Marker, 'voxel_map/frontier_centroids', 1)
        self.cluster_pub = self.create_publisher(PointCloud2, 'voxel_map/frontier_clusters', 1)
        if self.stuck is not None:
            # Command and odometry on a thread of their own: while the map thread is busy (a
            # mode switch re-inflates the whole map, a 3-D plan, a semantic frame) they queued
            # up and were taken out of order, a new command against older odometry, and the robot
            # was "stuck" right after every mode switch (false blocked voxels closed a hallway).
            # Marking the map stays on the map thread (_handle_stuck).
            fast = MutuallyExclusiveCallbackGroup()
            self.create_subscription(Twist, 'cmd_vel', self._on_cmd, 10, callback_group=fast)
            # Habitat bridge: the command each step executed, stamped like its odometry. Paired by
            # stamp, so the check no longer depends on when cmd_vel reached this node: after every
            # mode switch (map re-inflated, Nav2 replanning and stopping the robot) the stop came in
            # late and the stopped robot was "stuck" (false blocked voxels in the living room).
            self.cmd_exec = {}
            self.odom_waiting = {}       # odometry whose executed command has not arrived yet
            self.last_odom_t = -1.0
            self.create_subscription(TwistStamped, 'cmd_vel_executed', self._on_cmd_exec, 50, callback_group=fast)
            self.create_subscription(Odometry, 'odom', self._on_odom, 10, callback_group=fast)
            self.pending_stuck = None
            self.create_timer(0.1, self._handle_stuck)
        self.create_subscription(PointCloud2, 'collision_guard/block', self._on_block, 10)
        self.create_subscription(PoseStamped, 'collision_guard/tight', self._on_tight, 10)
        self.create_subscription(Bool, 'collision_guard/escaping', self._on_escaping, 10)
        # 3-D paths on the support voxels (trav3d.py, Sec. VI-B), followed by collision_guard_node:
        #   voxel_map/climb (Empty)   climb the stairs in view: mode s, then path after path to the
        #                             highest reachable stair voxel, until none is higher
        #   clicked_point (RViz "Publish Point")   one path to the clicked support voxel
        self.path_pub = self.create_publisher(Path, 'voxel_map/path3d', 1)
        # climb state for frontier_goal_node (which waits for the climb): 'climbing', 'done <z_r>'
        self.climb_state_pub = self.create_publisher(String, 'voxel_map/climb_state', 10)
        # stair landmarks O_stair (VoxelMapper.stair_landmarks): entrance x, y, z + top_z + size
        self.landmark_pub = self.create_publisher(PointCloud2, 'voxel_map/stair_landmarks', 1)
        self.landmark_marker_pub = self.create_publisher(Marker, 'voxel_map/stair_landmark_marker', 1)
        self.landmark_min_voxels = int(p('stair_landmark_min_voxels', 150).value)
        self.create_subscription(Empty, 'voxel_map/climb', self._on_climb, 1)
        # climb the stairs whose foot is at this point (frontier_goal_node, stair landmark):
        # first a 3-D path to the foot, then the climb
        self.create_subscription(PointStamped, 'voxel_map/climb_at', self._on_climb_at, 1)
        self.create_subscription(PointStamped, 'clicked_point', self._on_clicked, 1)
        self.create_subscription(Bool, 'collision_guard/following', self._on_following, 10)
        self.climbing = False
        self.following = False
        self.n_climb = 0
        # Path planning on the 3-D support graph (Sec. VI-B) for the whole exploration, as in the
        # paper (no 2-D planner): the Nav2 global planner is replaced by this node's
        # compute_path_to_pose action (Nav2's planner_server is not started; bt_navigator and the
        # explorer's path check call it, Nav2's controller follows the path), and J_geo of the
        # frontier tour comes from voxel_map/geo_costs (frontier_goal_node), and the stair climb
        # (voxel_map/climb_at) is a path on it too. All of them search one support graph of the
        # whole map (trav3d.SupportGraph): every floor and, in mode s, the stairs between them, as
        # in the paper. Built on the planning thread from a snapshot the map thread takes every
        # graph_snapshot_period s. Paths and J_geo to frontier goals start and end at floor_z.
        self.snap = None
        self.snap_wall = -1e9
        self.snapshot_period = float(p('graph_snapshot_period', 1.0).value)
        self.graph = None
        self.graph_of = None
        self.graph_lock = threading.Lock()
        # eq. 16 weights (not given in the paper): strong enough that paths keep to the middle of
        # a passage (as Nav2's inflation cost did for its 2-D planner); along the edge of O_inf
        # Nav2's controller saw a collision ahead in every corner and gave up
        self.graph_args = {'lambda_esdf': float(p('lambda_esdf', 0.3).value),
                           'd_safe': float(p('d_safe', 0.6).value)}
        self.plan_goal_radius = float(p('plan_goal_radius', 0.5).value)
        self.nav_plan_pub = self.create_publisher(Path, 'plan', 1)
        planning = MutuallyExclusiveCallbackGroup()
        self.path_server = ActionServer(self, ComputePathToPose, 'compute_path_to_pose', self._compute_path,
                                        callback_group=planning)
        self.create_service(GeoCosts, 'voxel_map/geo_costs', self._on_geo_costs, callback_group=planning)

    def _on_cmd(self, msg):
        self.cmd = (msg.linear.x, msg.angular.z)
        self.cmd_time = self.get_clock().now().nanoseconds * 1e-9

    # -- 3-D planning on the floor graph -------------------------------------------------------
    def _take_snapshot(self):
        """Map thread: copy what the support graph needs (SupportGraph.snapshot)."""
        m = self.mapper
        t0 = time.time()
        self.snap = trav3d.SupportGraph.snapshot(m, visited=m.visited_layers(), tight=m.tight_layers())
        self.snap_wall = time.time()
        self.t_snap = getattr(self, 't_snap', 0.0) + (self.snap_wall - t0)
        self.n_snap = getattr(self, 'n_snap', 0) + 1

    def _floor_graph(self):
        """The support graph of the whole map from the latest snapshot (built once per snapshot)."""
        with self.graph_lock:
            snap = self.snap
            if snap is None:
                return None
            if self.graph_of is not snap:
                t0 = time.time()
                self.graph = trav3d.SupportGraph(snap, **self.graph_args)
                self.graph_of = snap
                dt = time.time() - t0
                self.get_logger().info(
                    f'[plan] support graph: {self.graph.n} nodes, {self.graph.graph.nnz} edges in '
                    f'{1e3 * dt:.0f} ms (mode {snap["mode"]}; snapshot {1e3 * self.t_snap / self.n_snap:.0f} ms '
                    f'on average)', throttle_duration_sec=30.0)
            return self.graph

    def _path_msg(self, path, stamp):
        msg = Path(header=Header(stamp=stamp, frame_id=self.map_frame))
        for i, p in enumerate(path):
            q = path[min(i + 1, len(path) - 1)] - path[max(i - 1, 0)]
            yaw = math.atan2(q[1], q[0]) if np.hypot(q[0], q[1]) > 1e-6 else 0.0
            ps = PoseStamped(header=msg.header)
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, p)
            ps.pose.orientation.z, ps.pose.orientation.w = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
            msg.poses.append(ps)
        return msg

    def _compute_path(self, goal_handle):
        """nav2_msgs/ComputePathToPose on the support graph (replaces Nav2's planner_server)."""
        req = goal_handle.request
        result = ComputePathToPose.Result()
        t0 = time.time()
        g = self._floor_graph()
        if g is None or self.floor_z is None:
            goal_handle.abort()
            return result
        if req.use_start:
            s = req.start.pose.position
            start = np.array([s.x, s.y, self.floor_z])
        else:
            try:
                start = self._robot_support()
                start[2] = self.floor_z
            except TransformException:
                goal_handle.abort()
                return result
        q = req.goal.pose.position
        goal = np.array([q.x, q.y, self.floor_z])
        path, cost = g.path(start, goal, goal_radius=self.plan_goal_radius)
        if path is None or len(path) < 2:
            if path is None:
                self.get_logger().warn(f'[plan] no path on M_trav to ({q.x:.2f}, {q.y:.2f})',
                                       throttle_duration_sec=2.0)
                goal_handle.abort()
                return result
            path = np.vstack([path, path])           # already there: a path of one point
        result.path = self._path_msg(path, self.get_clock().now().to_msg())
        dt = time.time() - t0
        result.planning_time = DurationMsg(sec=int(dt), nanosec=int((dt % 1.0) * 1e9))
        self.nav_plan_pub.publish(result.path)
        goal_handle.succeed()
        return result

    def _on_geo_costs(self, request, response):
        """J_geo between points on the floor graph (frontier_goal_node's ATSP, eq. 11-16)."""
        g = self._floor_graph()
        response.ok = g is not None
        if g is None:
            response.costs = [float('inf')] * (len(request.sources) * len(request.targets))
            return response
        pts = lambda ps: [(p.x, p.y, p.z) for p in ps]
        src, tgt = pts(request.sources), pts(request.targets)
        C = g.costs(src, tgt)
        response.costs = C.reshape(-1).tolist()
        if C.size and not np.isfinite(C).any():
            s0 = g.node(src[0], 0.6)
            n_t = sum(g.node(t, 0.5) >= 0 for t in tgt)
            self.get_logger().warn(
                f'[plan] J_geo all inf: source ({src[0][0]:.2f}, {src[0][1]:.2f}, {src[0][2]:.2f}) '
                + (f'on node at z {g.positions()[s0, 2]:.2f}' if s0 >= 0 else 'not on the graph')
                + f'; {n_t} of {len(tgt)} targets on the graph (mode {self.graph_of["mode"]})',
                throttle_duration_sec=5.0)
        return response

    # -- 3-D paths --------------------------------------------------------------------------
    def _robot_support(self):
        base_p, _ = self._lookup(self.base_frame, Time())
        return np.array([base_p[0], base_p[1], self.support_z])

    def _send_path(self, path, why):
        msg = Path(header=Header(stamp=self.get_clock().now().to_msg(), frame_id=self.map_frame))
        for p in ([] if path is None else path):
            ps = PoseStamped(header=msg.header)
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, p)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.path_pub.publish(msg)
        if path is not None:
            self.following = True        # until the guard reports the end
            self.get_logger().info(f'[path3d] {why}: {len(path)} voxels to ({path[-1][0]:.2f}, {path[-1][1]:.2f}, '
                                   f'{path[-1][2]:.2f}), rise {path[-1][2] - path[0][2]:+.2f} m')

    def _start_climb(self):
        self.climbing = True
        self.n_climb = 0
        self.climb_last_end = None
        self.climb_failed = []                     # stretch ends the robot did not reach
        if self.switch is not None and self.switch.request('s') or self.mapper.mode != 's':
            # navigation to a stair entrance switches to mode s (Sec. IV-A): stairs are
            # traversable only in mode s, and with them the floors are connected in the graph
            self.mapper.set_mode('s')
            self.mode_pub.publish(String(data='s'))
            self.get_logger().warn('[mode] -> s: stair climb requested')
        self._take_snapshot()                      # the graph of mode s, with the stairs
        self.get_logger().info('[climb] started')
        self.climb_state_pub.publish(String(data='climbing'))

    def _on_climb(self, _msg):
        if self.mapper is None:
            return
        self._start_climb()
        self._climb_step()

    def _on_climb_at(self, msg):
        """Climb the stairs whose foot is at msg.point (frontier_goal_node, stair landmark O_stair):
        mode s, then one path on the support graph from the robot over the floor and up the
        stairs (the highest reachable stair node near the foot), then path after path from where
        the robot is, as more of the staircase comes into view."""
        if self.mapper is None:
            self.climb_state_pub.publish(String(data=f'done {self.support_z or 0.0:.3f}'))
            return
        foot = np.array([msg.point.x, msg.point.y, msg.point.z])
        self._start_climb()
        self._climb_step(center=foot)

    def _climb_step(self, center=None):
        try:
            start = self._robot_support()
        except TransformException:
            return
        graph = self._floor_graph()
        t0 = time.time()
        path = None
        if graph is not None:
            path, _ = graph.climb(start, center=center, exclude=self.climb_failed)
        if path is None:
            self.climbing = False
            self.get_logger().info(f'[climb] done after {self.n_climb} stretches: no higher stair voxel reachable '
                                   f'(z_r {self.support_z:.2f})')
            self.climb_state_pub.publish(String(data=f'done {self.support_z:.3f}'))
            return
        last = getattr(self, 'climb_last_end', None)
        if last is not None and np.linalg.norm(path[-1] - last) < 0.1:
            self.climbing = False                      # the same stretch again: nothing gained
            self.get_logger().info(f'[climb] done after {self.n_climb} stretches: the next stretch ends where '
                                   f'the last one did (z_r {self.support_z:.2f})')
            self.climb_state_pub.publish(String(data=f'done {self.support_z:.3f}'))
            return
        self.climb_last_end = path[-1]
        self.n_climb += 1
        self._send_path(path, f'climb stretch {self.n_climb} on the support graph '
                              f'({1e3 * (time.time() - t0):.0f} ms)')

    def _on_clicked(self, msg):
        if self.mapper is None:
            return
        try:
            start = self._robot_support()
        except TransformException:
            return
        goal = np.array([msg.point.x, msg.point.y, msg.point.z])
        t0 = time.time()
        graph = self._floor_graph()
        path, cost = (None, None) if graph is None else graph.path(start, goal, goal_radius=0.3)
        if path is None:
            self.get_logger().warn(f'[path3d] no path on M_trav to ({goal[0]:.2f}, {goal[1]:.2f}, {goal[2]:.2f}) '
                                   f'(mode {self.mapper.mode}; stairs are traversable in mode s)')
            return
        self._send_path(path, f'to the clicked point, J_geo {cost:.2f}, {1e3 * (time.time() - t0):.0f} ms')

    def _on_following(self, msg):
        was, self.following = self.following, msg.data
        if was and not msg.data and self.climbing and self.climb_last_end is not None:
            # stretch failed (no progress / timed out / blocked): do not aim there again
            try:
                p = self._robot_support()
                if np.hypot(*(p[:2] - self.climb_last_end[:2])) > 0.2:
                    self.climb_failed.append(self.climb_last_end)
                    self.climb_last_end = None
                    self.get_logger().info(f'[climb] stretch {self.n_climb} not reached; that target is left out '
                                           f'({len(self.climb_failed)} left out)')
            except TransformException:
                pass
        if was and not msg.data and self.climbing:
            # stretch done: plan the next one once the steps above are in view and mapped (the
            # stair window rises with z_r), on a snapshot taken after that
            self.climb_timer = self.create_timer(1.0, self._climb_timer)

    def _climb_timer(self):
        self.climb_timer.cancel()
        if self.climbing and not self.following:
            self._take_snapshot()
            self._climb_step()

    def _on_escaping(self, msg):
        # While the guard drives the robot back along its trail, a stuck event points the wrong
        # way: "ahead" is the way out. Marking it put a blocked voxel on the trail and closed the
        # cells around it, and the robot was shut in next to the bed.
        self.escaping = msg.data
        if self.stuck is not None:
            self.stuck.reset()

    def _on_cmd_exec(self, msg):
        key = msg.header.stamp.sec * 10 ** 9 + msg.header.stamp.nanosec
        self.cmd_exec[key] = (msg.twist.linear.x, msg.twist.angular.z)
        while len(self.cmd_exec) > 100:
            del self.cmd_exec[min(self.cmd_exec)]
        odom = self.odom_waiting.pop(key, None)
        if odom is not None:                           # its odometry came first
            self._on_odom(odom)

    def _on_odom(self, msg):
        t = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        key = msg.header.stamp.sec * 10 ** 9 + msg.header.stamp.nanosec
        if self.cmd_exec:
            if key not in self.cmd_exec:
                self.odom_waiting[key] = msg           # its command comes next (other topic)
                while len(self.odom_waiting) > 20:
                    del self.odom_waiting[min(self.odom_waiting)]
                return
            vx, wz = self.cmd_exec.pop(key)
        else:
            # no executed commands (not the Habitat bridge): cmd_vel as it arrives; the bridge
            # drops a command older than cmd_timeout, so does the prediction
            vx, wz = self.cmd if t - self.cmd_time <= self.cmd_timeout else (0.0, 0.0)
        if t <= self.last_odom_t:
            return                                     # older than one already used (held back)
        self.last_odom_t = t
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        pos = msg.pose.pose.position
        if not self.stuck.update(t, pos.x, pos.y, yaw, vx, wz) or self.mapper is None or self.escaping:
            return
        self.pending_stuck = msg.header.stamp

    def _handle_stuck(self):
        """Mark a stuck event (detected on the command/odometry thread) in the map."""
        stamp, self.pending_stuck = self.pending_stuck, None
        if stamp is None or self.mapper is None or self.escaping:
            return
        try:
            base_p, base_R = self._lookup(self.base_frame, Time.from_msg(stamp))
        except TransformException:
            try:      # a blocked robot has not moved: the latest pose is as good
                base_p, base_R = self._lookup(self.base_frame, Time())
            except TransformException as e:
                self.get_logger().warn(f'stuck, but no TF for the robot: {e}')
                return
        # v_front: the one voxel straight ahead, inflation radius + 0.15 m from the centre, so the
        # inflated block leaves the robot's own cell traversable (closer, the robot is trapped in
        # O_inf) and lies beyond the path protection of mark_blocked (inflation + 2 cells).
        # Inflated, one voxel closes any gap narrower than the robot; a whole row across the robot
        # width would also close a hallway the robot only grazed a wall of.
        m = self.mapper
        r = self.mapper_args['robot_radius']
        yaw = math.atan2(base_R[1, 0], base_R[0, 0])
        xy = front_row(base_p[0], base_p[1], yaw, 0.0, m.res, r + 0.15)
        n_new = m.mark_blocked(xy, self.support_z)
        self._close_tight(base_p[0], base_p[1], yaw, 'stuck')
        self._update_frontiers()
        self.n_stuck += 1
        self.get_logger().warn(
            f'[stuck] #{self.n_stuck} at ({base_p[0]:.2f}, {base_p[1]:.2f}): commanded forward motion '
            f'blocked, {n_new} voxel in front marked occupied')
        self._publish(stamp)

    def _close_tight(self, x, y, yaw, why):
        n = self.mapper.mark_tight(x, y, yaw, self.support_z, self.body_radius)
        self.get_logger().info(f'[tight] {why} at ({x:.2f}, {y:.2f}): {n} cells ahead closer than '
                               f'{self.body_radius:.2f} m to obstacles closed on trav_grid')
        return n

    def _on_tight(self, msg):
        if self.mapper is None or self.support_z is None or self.escaping:
            return
        q = msg.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._close_tight(msg.pose.position.x, msg.pose.position.y, yaw, 'collision guard stop')
        self._publish(msg.header.stamp)

    def _on_block(self, msg):
        if self.mapper is None:
            return
        xy = np.array([[q[0], q[1]] for q in point_cloud2.read_points(msg, field_names=('x', 'y'))]).reshape(-1, 2)
        n_new = self.mapper.mark_blocked(xy, self.support_z)
        self._update_frontiers()
        self.get_logger().info(f'[block] {n_new} voxels marked blocked by collision_guard')
        self._publish(msg.header.stamp)

    def _on_semantic(self, msg):
        if msg.encoding != 'mono8':
            self.get_logger().error(f'semantic image must be mono8, got {msg.encoding}',
                                    throttle_duration_sec=10.0)
            return
        key = msg.header.stamp.sec * 10 ** 9 + msg.header.stamp.nanosec
        self.sem_cache[key] = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, -1)[:, :msg.width]
        while len(self.sem_cache) > 20:
            del self.sem_cache[min(self.sem_cache)]
        if key in self.depth_cache:
            self._on_pair(key, msg.header.stamp)

    def _on_info(self, msg):
        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        if self.K is not None and np.allclose(K, self.K):
            return
        self.K = K
        v, u = np.mgrid[0:msg.height:self.stride, 0:msg.width:self.stride]
        self.pix = (v.reshape(-1), u.reshape(-1))
        self.rays = np.stack([(u.reshape(-1) - K[0, 2]) / K[0, 0],
                              (v.reshape(-1) - K[1, 2]) / K[1, 1],
                              np.ones(u.size)], axis=1)
        v, u = np.mgrid[0:msg.height:self.sem_stride, 0:msg.width:self.sem_stride]
        self.sem_pix = (v.reshape(-1), u.reshape(-1))
        self.sem_rays = np.stack([(u.reshape(-1) - K[0, 2]) / K[0, 0],
                                  (v.reshape(-1) - K[1, 2]) / K[1, 1],
                                  np.ones(u.size)], axis=1)

    def _lookup(self, frame, stamp):
        t = self.tf_buffer.lookup_transform(self.map_frame, frame, stamp, timeout=Duration(seconds=0.2))
        tr = t.transform.translation
        return np.array([tr.x, tr.y, tr.z]), quat_to_rot(t.transform.rotation)

    def _estimate_floor(self, pts, base_p):
        """Height of the support plane: mode of the heights of nearby low points (2 cm bins)."""
        near = np.hypot(pts[:, 0] - base_p[0], pts[:, 1] - base_p[1]) < self.floor_radius
        z = pts[near & (pts[:, 2] < base_p[2] + self.floor_max_above), 2]
        if self.support_z is not None:
            z = z[np.abs(z - self.support_z) < self.floor_max_jump]
        if z.size < self.floor_min_points:
            return self.support_z
        hist, edges = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.04, 0.02))
        i = int(np.argmax(hist))
        if hist[max(i - 1, 0):i + 2].sum() < self.floor_min_points:
            return self.support_z
        peak = 0.5 * (edges[i] + edges[i + 1])
        est = float(np.mean(z[np.abs(z - peak) < 0.03]))
        return est if self.support_z is None else 0.7 * self.support_z + 0.3 * est

    def _on_depth(self, msg):
        if self.rays is None:
            return
        now = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        key = msg.header.stamp.sec * 10 ** 9 + msg.header.stamp.nanosec
        skip = self.last_integrate is not None and 0.0 <= now - self.last_integrate < self.integrate_period
        if skip and self.switch is None and not self.sem_cache:
            return
        if msg.encoding == '32FC1':
            depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, -1)[:, :msg.width]
        elif msg.encoding in ('16UC1', 'mono16'):
            depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, -1)[:, :msg.width] * 1e-3
        else:
            self.get_logger().error(f'unsupported depth encoding {msg.encoding}', throttle_duration_sec=10.0)
            return
        self.depth_cache[key] = (depth, msg.header.frame_id)
        while len(self.depth_cache) > 30:                # 1.5 s at 20 Hz
            del self.depth_cache[min(self.depth_cache)]
        if key in self.sem_cache:
            self._on_pair(key, msg.header.stamp)
        if skip:
            return
        stamp = Time.from_msg(msg.header.stamp)
        try:
            cam_p, cam_R = self._lookup(msg.header.frame_id, stamp)
            base_p, _ = self._lookup(self.base_frame, stamp)
        except TransformException as e:
            self.get_logger().warn(f'no TF for the depth frame: {e}', throttle_duration_sec=5.0)
            return
        self.last_integrate = now

        t0 = time.time()
        z = depth[self.pix].astype(np.float64)
        valid = np.isfinite(z) & (z > 0.0)               # 0 = no geometry (mesh hole)
        sel = np.flatnonzero(valid)                      # index into self.pix of each point
        z, rays = z[valid], self.rays[valid]
        dist = z * np.linalg.norm(rays, axis=1)
        valid = dist >= self.min_range
        z, rays, dist, sel = z[valid], rays[valid], dist[valid], sel[valid]
        hit = dist <= self.max_range
        z = np.where(hit, z, z * self.max_range / dist)  # truncated rays only clear free space
        pts = (rays * z[:, None]) @ cam_R.T + cam_p
        if self.support_z is None or self.track_floor:
            self.support_z = self._estimate_floor(pts[hit], base_p)
        elif self.follow_support and self.mapper is not None:
            z_s = self.mapper.support_height(base_p[0], base_p[1], base_p[2])
            if z_s is not None:
                self.support_z = z_s
            elif self.last_base_z is not None:             # nothing mapped there: follow the base
                self.support_z += base_p[2] - self.last_base_z
        self.last_base_z = base_p[2]
        if self.support_z is None:
            self.get_logger().warn('no floor in view yet', throttle_duration_sec=5.0)
            return
        # floor_z stays within floor_hold of the anchor taken on arrival on the floor (start or
        # end of a climb); it is not re-anchored when z_r creeps away in small steps (00810, upper
        # floor: z_r 2.88 -> 2.75 -> 2.63 with base_link z = 3.000; floor_z followed each 0.125 m
        # step, the floor graph was built 25 cm below the floor and every frontier looked
        # unreachable, so exploration ended), nor when the controller takes a few stair steps.
        if self.floor_z is None or self.climbing:
            self.floor_z = self.floor_anchor = self.support_z
        elif abs(self.support_z - self.floor_anchor) <= self.floor_hold:
            self.floor_z = self.support_z
        if self.mapper is None:
            self.mapper = VoxelMapper(z_ref=self.support_z, **self.mapper_args)
            m = self.mapper
            self.get_logger().info(
                f'voxel map {m.nx}x{m.ny}x{m.nz} @ {m.res:.2f} m, mode={m.mode}, floor at '
                f'z={self.support_z:.3f} (base_link z={base_p[2]:.3f}), robot radius '
                f'{self.mapper_args["robot_radius"]} m, collision band {m.h_min}-{m.h_max} m')
            self.frontiers = FrontierManager(m, **self.frontier_args)
        k = self.mapper.layer(self.support_z)
        if k != getattr(self, 'support_layer', k):
            # the stair window rose (or fell): stairs seen before, outside it, may be trusted now
            n_prom = self.mapper.promote_stairs(self.support_z, self.stair_window, self.stair_column_gap)
            self.get_logger().info(f'[floor] support layer {self.support_layer} -> {k} '
                                   f'(z_r={self.support_z:.3f}, base_link z={base_p[2]:.3f})'
                                   + (f', {n_prom} remembered stair voxels now trusted' if n_prom else ''))
        self.support_layer = k
        # the robot's own path: blocked marks never close it (VoxelMapper.mark_blocked)
        self.mapper.visit(base_p[0], base_p[1], self.support_z)
        # robot support point x_r: bottom centre of the robot, on the estimated floor
        support = np.array([base_p[0], base_p[1], self.support_z])
        # M_g also in mode s while the robot is on a floor: frozen for the whole of mode s, the
        # floor explored meanwhile (157 s near the stairs of 00810) was in M_s only and fell out
        # of the map as holes and frontiers when the mode went back to g. Not on / climbing the
        # stairs, where the support plane is in mid-air. A climb starts with the way over the
        # floor to the stairs (one path on the support graph): there M_g is updated.
        d_st = self.mapper.stair_distance(base_p[0], base_p[1], self.support_z, dz=0.3)
        on_stairs = d_st < 0.5 or self.climbing and d_st < 1.5
        n_changed = self.mapper.integrate(cam_p, pts, hit, support,
                                          update_g=self.mapper.mode == 'g' or not on_stairs)
        if time.time() - self.snap_wall >= self.snapshot_period:
            self._take_snapshot()
        self.t_total += time.time() - t0
        self._update_frontiers()
        self.frames += 1
        if self.frames % 20 == 0:
            f = self.frontiers
            self.get_logger().info(
                f'[voxel] {self.frames} frames, map {1e3 * self.t_total / self.frames:.1f} ms/frame, '
                f'frontiers {1e3 * self.t_frontier / self.frames:.1f} ms/frame, last |C_t| = {n_changed}, '
                f'{len(f.clusters)} frontier clusters ({f.voxels().size} voxels)'
                + (f', stair points trusted {self.n_stair_kept}/{self.n_stair_raw}' if self.n_stair_raw else ''))

        if self.last_publish is None or not 0.0 <= now - self.last_publish < self.publish_period:
            self.last_publish = now
            self._publish(msg.header.stamp)

    def _on_pair(self, key, stamp):
        """Class image and depth frame of one stamp: mode switch on every pair, M_s every
        integrate_period."""
        sem = self.sem_cache.pop(key)
        depth, frame_id = self.depth_cache.pop(key)
        if self.mapper is None or self.rays is None:
            return
        if sem.shape != depth.shape:
            self.get_logger().error(f'semantic image {sem.shape} is not registered with the depth '
                                    f'image {depth.shape}', throttle_duration_sec=10.0)
            return
        t = stamp.sec + 1e-9 * stamp.nanosec
        if self.switch is not None:
            self._update_mode(sem, depth, stamp, t)
        if self.last_sem_integrate is not None and 0.0 <= t - self.last_sem_integrate < self.integrate_period:
            return
        try:
            cam_p, cam_R = self._lookup(frame_id, Time.from_msg(stamp))
        except TransformException as e:
            self.get_logger().warn(f'no TF for the semantic frame: {e}', throttle_duration_sec=5.0)
            return
        self.last_sem_integrate = t
        t0 = time.time()
        self._integrate_semantic(sem, depth, cam_p, cam_R)
        self.t_total += time.time() - t0
        self._update_frontiers()

    def _integrate_semantic(self, sem, depth, cam_p, cam_R):
        """M_s and the stair voxels from one class image (stairs only where trusted)."""
        trav, _ = split_classes(sem[self.sem_pix])
        z = depth[self.sem_pix].astype(np.float64)
        ok = trav & np.isfinite(z) & (z > 0.0)
        rays, z, cls = self.sem_rays[ok], z[ok], sem[self.sem_pix][ok]
        dist = z * np.linalg.norm(rays, axis=1)
        near = (dist >= self.min_range) & (dist <= self.sem_max_range)
        pts = (rays[near] * z[near, None]) @ cam_R.T + cam_p
        _, stairs = split_classes(cls[near])
        others = pts[~stairs]
        stair_pts = self.mapper.trusted_stairs(pts[stairs], others, self.support_z,
                                               self.stair_window, self.stair_column_gap)
        self.n_stair_raw += int(stairs.sum())
        self.n_stair_kept += len(stair_pts)
        self.mapper.integrate_semantic(np.concatenate([others, stair_pts]), stair_pts)
        self.mapper.remember_stairs(pts[stairs])
        self.n_sem_frames += 1

    def _update_mode(self, sem, depth, stamp, t):
        """g/s switch on every camera frame: stair pixels near the camera, robot to the stairs."""
        _, stairs = split_classes(sem[self.pix])
        z = depth[self.pix].astype(np.float64)
        rng = z * np.linalg.norm(self.rays, axis=1)
        n_near = int((stairs & np.isfinite(z) & (z > 0.0) & (rng <= self.switch.enter_range)).sum())
        d_map = float('inf')
        if self.switch.mode == 's':
            try:
                base_p, _ = self._lookup(self.base_frame, Time.from_msg(stamp))
            except TransformException:
                return
            d_map = self.mapper.stair_distance(base_p[0], base_p[1], self.support_z)
        before = self.switch.mode
        if self.climbing:
            return                                     # stay in mode s until the climb is over
        if not self.switch.update(t, n_near, d_map):
            return
        t0 = time.time()
        self.mapper.set_mode(self.switch.mode)           # re-inflates with the other source (eq. 1)
        try:
            base_p, _ = self._lookup(self.base_frame, Time())
        except TransformException:
            base_p = (float('nan'),) * 3
        why = (f'{n_near} stair pixels within {self.switch.enter_range:.1f} m for {self.switch.enter_time:.1f} s'
               if self.switch.mode == 's' else
               f'{d_map:.1f} m from the stairs for {self.switch.exit_time:.1f} s')
        self.get_logger().warn(
            f'[mode] {before} -> {self.switch.mode} at ({base_p[0]:.2f}, {base_p[1]:.2f}): {why} '
            f'(re-inflation {1e3 * (time.time() - t0):.0f} ms, {int(self.mapper.stairs.sum())} stair voxels)')
        self.mode_pub.publish(String(data=self.switch.mode))
        self._publish_mode_marker(Header(stamp=stamp, frame_id=self.map_frame))

    def _update_frontiers(self):
        t0 = time.time()
        self.frontiers.update(self.mapper.pop_update_box())
        self.t_frontier += time.time() - t0

    def _publish(self, stamp):
        m = self.mapper
        h = Header(stamp=stamp, frame_id=self.map_frame)
        trav = m.traversable()
        fvox = self.frontiers.voxels()
        trav_vox = np.setdiff1d(np.flatnonzero(trav), fvox, assume_unique=True)
        cloud = point_cloud2.create_cloud_xyz32
        top = m.layer(self.support_z + self.display_max_height) + 1
        occ = m.occupied & ~trav
        occ[:, :, top:] = False
        self.occ_pub.publish(cloud(h, m.centers(np.flatnonzero(occ)).astype(np.float32)))
        obst = occ & ~m._source()
        self.obst_pub.publish(cloud(h, m.centers(np.flatnonzero(obst)).astype(np.float32)))
        self.trav_pub.publish(cloud(h, m.centers(trav_vox).astype(np.float32)))
        self.frontier_pub.publish(cloud(h, m.centers(fvox).astype(np.float32)))
        mk = Marker(header=h, ns='frontier_centroids', id=0, type=Marker.SPHERE_LIST, action=Marker.ADD)
        mk.pose.orientation.w = 1.0
        mk.scale.x = mk.scale.y = mk.scale.z = 0.2
        mk.color = ColorRGBA(r=1.0, g=0.85, b=0.0, a=1.0)
        for c in self.frontiers.centroids():
            mk.points.append(Point(x=float(c[0]), y=float(c[1]), z=float(c[2]) + 0.1))
        self.centroid_pub.publish(mk)
        fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                  for i, n in enumerate(('x', 'y', 'z', 'size'))]
        clusters = [(*map(float, c.centroid), float(c.size)) for c in self.frontiers.clusters.values()]
        self.cluster_pub.publish(point_cloud2.create_cloud(h, fields, clusters))
        self.inf_pub.publish(point_cloud2.create_cloud_xyz32(
            h, m.inflated_points(self.support_z).astype(np.float32)))
        self.blocked_pub.publish(point_cloud2.create_cloud_xyz32(h, m.blocked_points().astype(np.float32)))
        if self.n_sem_frames:
            floor_s, stairs_s = m.semantic_points()
            self.sem_pub.publish(cloud(h, floor_s.astype(np.float32)))
            self.stairs_pub.publish(cloud(h, stairs_s.astype(np.float32)))
        self._publish_mode_marker(h)
        self._publish_landmarks(h)
        g = m.trav_grid(self.floor_z)
        if g is None:
            return
        data, w, hgt, ox, oy = g
        msg = OccupancyGrid()
        msg.header = h
        msg.info.map_load_time = stamp
        msg.info.resolution = m.res
        msg.info.width, msg.info.height = int(w), int(hgt)
        msg.info.origin.position.x = float(ox)
        msg.info.origin.position.y = float(oy)
        msg.info.origin.position.z = float(self.floor_z)
        msg.info.origin.orientation.w = 1.0
        msg.data = data.tolist()
        self.grid_pub.publish(msg)


    def _publish_landmarks(self, h):
        lms = self.mapper.stair_landmarks(self.landmark_min_voxels) if self.n_sem_frames else []
        fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                  for i, n in enumerate(('x', 'y', 'z', 'top_x', 'top_y', 'top_z', 'size'))]
        self.landmark_pub.publish(point_cloud2.create_cloud(h, fields, [tuple(map(float, lm)) for lm in lms]))
        mk = Marker(header=h, ns='stair_landmarks', id=0, type=Marker.SPHERE_LIST,
                    action=Marker.ADD if lms else Marker.DELETE)
        mk.pose.orientation.w = 1.0
        mk.scale.x = mk.scale.y = mk.scale.z = 0.35
        mk.color = ColorRGBA(r=0.0, g=0.9, b=0.9, a=0.9)
        for lm in lms:
            mk.points.append(Point(x=float(lm[0]), y=float(lm[1]), z=float(lm[2]) + 0.2))   # entrance
            mk.points.append(Point(x=float(lm[3]), y=float(lm[4]), z=float(lm[5])))         # top
        self.landmark_marker_pub.publish(mk)

    def _publish_mode_marker(self, h):
        try:
            base_p, _ = self._lookup(self.base_frame, Time())
        except TransformException:
            return
        mk = Marker(header=h, ns='mode', id=0, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD)
        mk.pose.position.x, mk.pose.position.y = float(base_p[0]), float(base_p[1])
        mk.pose.position.z = float(self.support_z) + 1.2
        mk.pose.orientation.w = 1.0
        mk.scale.z = 0.3
        semantic = self.mapper.mode == 's'
        mk.color = ColorRGBA(r=1.0, g=0.4, b=1.0, a=1.0) if semantic else ColorRGBA(r=0.2, g=1.0, b=0.4, a=1.0)
        mk.text = 'mode s (semantic)' if semantic else 'mode g (geometric)'
        self.mode_marker_pub.publish(mk)


def main(args=None):
    rclpy.init(args=args)
    node = VoxelMapperNode()
    executor = MultiThreadedExecutor(num_threads=3)   # map, command/odometry and planning threads
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
