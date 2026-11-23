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
#   voxel_map/frontier_clusters   PointCloud2 one point per cluster: centroid p_i + fields size,
#                                       semantic (share of its voxels traversable only through M_s:
#                                       >= 0.5 is a semantic frontier F_s; input of frontier_goal_node),
#                                       id, dormant (visited flag, frontier3d), fails
#   (in) voxel_map/frontier_dormant  Int64  cluster id frontier_goal_node found irreducible
#   (in) voxel_map/frontier_failed   Int64  cluster id of a goal that was not reached
#   (in) voxel_map/frontier_release  Empty  clear all dormant flags (only dormant frontiers left)
#   voxel_map/stuck        Empty        a stuck event, after v_front was marked (frontier_goal_node)
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
# of traversable classes go into M_s, stair pixels also into the stair voxels (in both modes), and
# the mode is s only for a stair transition: from the climb / descent request (_start_climb) until
# no stairs are in view for N_exit class frames after the climb (eq. 1, ModeSwitch). Without class
# images the node stays in g.
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
# replans around them. Frontiers behind the blockage get failed attempts when Nav2 gives up on
# them and become dormant (frontier_goal_node).
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
from std_msgs.msg import Bool, ColorRGBA, Empty, Float64, Header, Int64, String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker

from trav_exploration.collision_guard import front_row
from trav_exploration.frontier3d import FrontierManager
from trav_exploration import trav3d
from trav_exploration.semantic import ModeSwitch, split_classes
from trav_exploration.voxel_mapper import StuckDetector, VirtualGround, VoxelMapper, start_support
from trav_exploration.tour import cost_matrix, solve_open_atsp
from trav_exploration.recorder import Recorder
from trav_exploration.value_map import ValueMap, view_weight
from trav_interfaces.msg import ImageScore
from trav_interfaces.srv import FrontierTargets, GeoCosts, TargetGoal


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
        self.floor_wait = float(p('floor_wait', 15.0).value)               # s without a floor in view
        # start on a stair landing (no level floor in view): look down while waiting, then look
        # around with the camera start_look_min down (see the floor-wait branch); 0 = off
        self.start_look_min = float(p('start_look_min', math.radians(30.0)).value)
        self.start_look_max = float(p('start_look_max', math.radians(60.0)).value)
        self.start_look_step = float(p('start_look_step', math.radians(15.0)).value)
        self.start_look_period = float(p('start_look_period', 3.0).value)    # s (sim) per step
        self.start_look_move = float(p('start_look_move', 1.0).value)        # m
        self.start_look_hold = float(p('start_look_hold', 30.0).value)       # s (sim)
        self.camera_height = float(p('camera_height', 0.88).value)         # camera above the floor (habitat_server)
        # first z_r: the lowest floor cluster within this band of camera_height below the camera
        self.start_floor_band = float(p('start_floor_band', 0.10).value)
        # a floor estimate more than this under base_link at the start is not the robot's support
        # (voxel_mapper.start_support: a start on a stair landing)
        self.start_floor_max_below = float(p('start_floor_max_below', 0.25).value)
        # the stuck handler (eq. 17) also closes the cells ahead where the real body (body_radius)
        # touches an obstacle on trav_grid (mark_tight, the collision guard's narrow-gap closing).
        # Off with guard_mode path_only (launch): the paper marks v_front only, inflated in 3-D once
        self.stuck_close_tight = bool(p('stuck_close_tight', True).value)
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
        # min cluster size 7 voxels: smaller clusters are noise (paper: small clusters are
        # discarded). M_trav is O_inf-shrunk by the robot radius on both sides of a doorway: a
        # 0.85 m door leaves a ~0.3 m frontier (6 voxels in a row), which 7 can drop; it was 3
        self.frontier_args = dict(min_size=int(p('frontier_min_size', 7).value),
                                  headroom_layers=int(p('frontier_headroom_layers', 0).value),
                                  boundary=str(p('frontier_boundary', 'source').value),
                                  keep_share=float(p('dormant_keep_share', 0.6).value),
                                  need_unknown=bool(p('frontier_need_unknown', True).value))
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
                                 min_stair_points=int(p('mode_min_stair_points', 50).value),
                                 enter_range=float(p('mode_enter_range', 4.0).value),
                                 exit_distance=float(p('mode_exit_distance', 1.0).value),
                                 mode=self.mapper_args['mode'],
                                 auto_enter=bool(p('mode_auto_enter', False).value),
                                 exit_rule=str(p('mode_exit_rule', 'distance').value),
                                 exit_frames=int(p('mode_exit_frames', 5).value)) \
            if bool(p('mode_switch', True).value) else None
        # class image and depth frame of one stamp are paired whichever comes first: the ground
        # truth (habitat_server --semantic) arrives before the depth frame, SAM3 ~0.2 s after it
        self.sem_cache = {}         # stamp (ns) -> class image, waiting for its depth frame
        self.depth_cache = {}       # stamp (ns) -> (depth image, frame id), waiting for its class image
        self.last_sem_integrate = None
        # stairs trusted as support before climbing (VoxelMapper.trusted_stairs): up to
        # stair_window above the floor, up to stair_column_gap above the lowest surface below
        self.stair_window = float(p('stair_window', 0.6).value)
        # Step check of the stair landmarks (not in the paper; the verification of a tentative
        # hypothesis, Sec. V-A, by geometry): the landmarks on stair_check_input are republished on
        # voxel_map/stair_landmarks_checked with risers (VoxelMapper.stair_steps) and a verdict:
        # 1 a flight (>= stair_check_risers steps between foot and top), -1 no flight although
        # the raised part is observed (>= stair_check_min_top top-surface cells above the foot: a threshold, a
        # rug, a sofa that SAM3 called stairs; 00495: the climb went back and forth between two
        # 0.15 m spots for 600 s), 0 not observed enough yet (kept: real stairs seen from afar)
        self.stair_check_risers = int(p('stair_check_risers', 2).value)
        self.stair_check_min_top = int(p('stair_check_min_top', 60).value)     # 0.15 m^2
        self.stair_lm_msg = None
        self.stair_verdicts = {}      # rounded entrance -> verdict (logged on change)
        self.checked_lm_pub = self.create_publisher(PointCloud2, 'voxel_map/stair_landmarks_checked', 1)
        lm_in = str(p('stair_check_input', 'instance_map/stair_landmarks').value)
        if lm_in:
            self.create_subscription(PointCloud2, lm_in, lambda m: setattr(self, 'stair_lm_msg', m), 1)
            self.create_timer(1.0, lambda: self._defer(self._check_landmarks))
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
        # re-anchoring on the same floor: z_r read at another layer (beyond floor_hold) for
        # floor_reanchor_time s with base_link at the height it had at the anchor (no climb, no
        # step) is a corrected reading when the map holds more floor at the new height around the
        # robot (00813: z_r 0.084 at the start, the floor at -0.066; trav_grid and the goals kept
        # the slice at 0.084 and lost the floor). 00810's upper floor (z_r creeping 25 cm under
        # the floor) is kept out by the floor count.
        self.floor_reanchor_time = float(p('floor_reanchor_time', 3.0).value)
        self.floor_reanchor_base_dz = float(p('floor_reanchor_base_dz', 0.05).value)
        self.floor_reanchor_radius = float(p('floor_reanchor_radius', 3.0).value)
        self.anchor_base_z = None
        self.reanchor_cand = None   # (layer, since)
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
        # Map-thread task queue (_defer / _run_map_tasks): the climb / descent requests, the end of
        # a path stretch (collision_guard/following) and the delayed climb steps are received on
        # their own callback group and run on the map thread at its next depth frame. In the map
        # thread's group they starved behind the depth and class-image callbacks, which run back to
        # back (~0.35 s each): a request was never taken (2 of 119 climb requests in v1 val timed
        # out) and a climb stopped after its first stretch (the 1 s timer for the next one never
        # fired, 00803 on the dev set). They change the map (mode s) and the climb state, so they
        # run on the map thread.
        self.map_tasks = []                            # (due wall time, callable)
        self.map_tasks_lock = threading.Lock()
        self.requests_group = requests = MutuallyExclusiveCallbackGroup()
        self.create_subscription(Empty, 'voxel_map/climb', lambda m: self._defer(lambda: self._on_climb(m)), 1,
                                 callback_group=requests)
        # climb the stairs whose foot is at this point (frontier_goal_node, stair landmark):
        # first a 3-D path to the foot, then the climb
        self.create_subscription(PointStamped, 'voxel_map/climb_at',
                                 lambda m: self._defer(lambda: self._on_climb_at(m)), 1, callback_group=requests)
        # the climb called off (frontier_goal_node: reachable frontiers again on the floor it left)
        self.create_subscription(Empty, 'voxel_map/climb_cancel',
                                 lambda _m: self._defer(self._cancel_climb), 1, callback_group=requests)
        # go down the stairs whose head (top, on this floor) is at this point (frontier_goal_node)
        self.create_subscription(PointStamped, 'voxel_map/descend_at',
                                 lambda m: self._defer(lambda: self._on_descend_at(m)), 1, callback_group=requests)
        # one look_down step [rad] (the 30 deg action of the Habitat ObjectNav action space); how
        # many steps the camera goes down is decided by the Pi_vg feedback loop (VirtualGround)
        self.descend_pitch = float(p('descend_pitch', math.radians(30.0)).value)
        self.pitch_pub = self.create_publisher(Float64, 'camera/pitch_cmd', 1)
        # one owner of the camera pitch: this node sends max over its parts (Pi_vg, the start
        # look-down, the climb look-down, the fixed descent pitch) and frontier_goal_node's request
        # (camera/pitch_request, the blind-spot look-down). Two nodes commanding camera/pitch_cmd
        # each in turn nodded the camera every 0.5 s (00035: 31 changes in 6 min, 60-15-60-30-45...)
        self.pitch_parts = {}
        self.cmd_pitch = 0.0
        self.create_subscription(Float64, 'camera/pitch_request',
                                 lambda m: self._command_pitch('request', float(m.data)), 10,
                                 callback_group=self.requests_group)
        self.stuck_pub = self.create_publisher(Empty, 'voxel_map/stuck', 10)   # after the map update
        self.climb_dir = 1
        # virtual ground plane Pi_vg (Sec. IV-D): more than vg_n_down depth points below it near
        # the robot (descending stairs, the storey below) -> the downward-looking behaviour, closed
        # loop on the number of points observed below Pi_vg (also while going down stairs)
        self.vg = VirtualGround(depth=p('vg_depth', 0.3).value, radius=p('vg_radius', 2.5).value,
                                n_down=p('vg_n_down', 150).value, hold=p('vg_hold', 3.0).value,
                                step=self.descend_pitch,
                                max_pitch=p('vg_max_pitch', math.radians(60.0)).value,
                                settle=p('vg_settle', 3).value, gain=p('vg_gain', 0.1).value,
                                rescan_move=p('vg_rescan_move', 1.0).value) \
            if p('virtual_ground', True).value and self.descend_pitch > 0.0 else None
        self.vg_pitch = 0.0                        # last pitch the behaviour commanded
        self.create_subscription(PointStamped, 'clicked_point', self._on_clicked, 1)
        # spatial value map (Sec. V-B, eq. 7): the image-text relevance of an image (BLIP-2,
        # itm_node) is fused into the traversable voxels visible in its depth frame
        self.value = ValueMap()
        self.hfov = None
        self.n_itm = 0
        self.last_value_pub = -1e9
        self.value_pub = self.create_publisher(PointCloud2, 'voxel_map/value', 1)
        self.create_subscription(ImageScore, p('itm_topic', 'camera/itm_score').value, self._on_itm, 10)
        # dormant frontiers (frontier_goal_node): the visited flag of that cluster (frontier3d)
        self.frontier_max_attempts = int(p('frontier_max_attempts', 2).value)
        self.geo_target_radius = float(p('geo_target_radius', 0.2).value)    # voxel_map/geo_costs
        # goals placed on the support graph (frontier v_target, target goals) keep goal_clearance
        # from O_inf, as frontier_goal_node's goals on trav_grid did (clear_cells: more than
        # ceil(clearance / res) cells): d_obs (eq. 16, O_inf distance + robot radius) above it
        self.goal_clearance = float(p('goal_clearance', 0.18).value)
        self.create_subscription(Int64, 'voxel_map/frontier_failed',
                                 lambda m: self._defer(lambda: self._on_failed(m)), 10,
                                 callback_group=self.requests_group)
        self.create_subscription(Empty, 'voxel_map/frontier_release',
                                 lambda m: self._defer(lambda: self._on_release(m)), 10,
                                 callback_group=self.requests_group)
        self.create_subscription(Int64, 'voxel_map/frontier_dormant',
                                 lambda m: self._defer(lambda: self._on_dormant(m)), 10,
                                 callback_group=self.requests_group)
        self.create_subscription(Bool, 'collision_guard/following',
                                 lambda m: self._defer(lambda: self._on_following(m)), 10,
                                 callback_group=self.requests_group)
        self.climbing = False
        self.following = False
        # geometric fallback for treads the segmentation missed (turns of a staircase)
        self.geo_treads = bool(p('geometric_treads', True).value)
        # arrival, altitude side: the far end height of the stair landmark (frontier_goal_node)
        self.climb_end_z = self.climb_end_z_next = None
        self.create_subscription(Float64, 'voxel_map/climb_end_z',
                                 lambda m: setattr(self, 'climb_end_z_next', float(m.data)), 1,
                                 callback_group=self.requests_group)
        self.n_geo_treads = 0
        self.n_climb = 0
        # turning in place during a climb: on cmd_vel, remapped by the launch to the command the
        # robot executes (cmd_vel_safe with the guard; Nav2 is idle while frontier_goal waits)
        self.climb_turn_speed = float(p('climb_turn_speed', 0.6).value)     # rad/s
        # going up, nothing higher in view: the next flight can start at the robot's feet, under
        # the level camera (00873: the first step 0.5 m ahead of the landing, never seen; the
        # level camera sees the floor from 1.5 m on). Closed loop on what is to be seen (Sec.
        # IV-D): while frontier voxels within climb_look_radius of the robot (around its level)
        # are unobserved, the camera goes down climb_look_step more (up to climb_look_max),
        # planning again after each step; it stops when they are observed (registered in the
        # map, no longer frontier) or a stretch is found. Without such voxels it does not look
        # down at all. Then, if still nothing, the turn around (camera level: the next flight
        # rises, 00035 found it twice on the turn).
        self.climb_look_step = float(p('climb_look_step', math.radians(15.0)).value)   # 0 = off
        self.climb_look_max = float(p('climb_look_max', math.radians(60.0)).value)
        self.climb_look_radius = float(p('climb_look_radius', 1.5).value)
        self.climb_look_time = float(p('climb_look_time', 1.5).value)        # s (wall clock) per step
        self.climb_look_k = 0
        self.climb_pitch = 0.0
        self.climb_blocks = []
        self.climb_block_radius = float(p('climb_block_radius', 0.1).value)    # m around each point
        self.descend_look_dist = float(p('descend_look_dist', 1.5).value)   # m: Pi_vg forced from here
        self.vg_force_pending = False
        self.turn_cmd_pub = self.create_publisher(Twist, 'cmd_vel', 1)
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
        # diagnostics: when a climb ends with nothing reachable, save the map around the robot
        # (occupancy, M_g, M_s, stairs, O_inf, M_trav) to <dir>/climb_stop_<n>.npz ('' = off)
        self.debug_dump_dir = str(p('debug_dump_dir', '').value)
        # record_dir: periodic state for offline datasets (recorder.py; '' = off)
        self.recorder = Recorder(self, 'voxel_mapper', str(p('record_dir', '').value))
        self.record_period = float(p('record_period', 1.0).value)    # s of sim time
        self.last_record = -1e9
        self.n_dump = 0
        self.graph = None
        self.graph_of = None
        self.graph_lock = threading.Lock()
        # eq. 16 weights (not given in the paper): strong enough that paths keep to the middle of
        # a passage (as Nav2's inflation cost did for its 2-D planner); along the edge of O_inf
        # Nav2's controller saw a collision ahead in every corner and gave up
        self.graph_args = {'lambda_esdf': float(p('lambda_esdf', 0.3).value),
                           'd_safe': float(p('d_safe', 0.6).value)}
        self.plan_goal_radius = float(p('plan_goal_radius', 0.5).value)
        self.loose_stair_reach = int(p('loose_stair_reach', 6).value)     # cells, descent fallback
        # a climb ends on the first level it reaches by the exit rule of mode s (Sec. IV-A): not
        # the floor it started from (>= floor_min_change from it) and sufficiently far from the
        # stair region (> mode_exit_distance) for N_exit consecutive frames (mode_exit_frames). A
        # landing between two flights is next to the stairs and does not end it. (Earlier: >= 3 m^2
        # of flat floor connected to the robot, not in the paper.)
        self.floor_min_change = float(p('floor_min_change', 0.5).value)
        self.snap_loose = None
        self.nav_plan_pub = self.create_publisher(Path, 'plan', 1)
        planning = MutuallyExclusiveCallbackGroup()
        self.path_server = ActionServer(self, ComputePathToPose, 'compute_path_to_pose', self._compute_path,
                                        callback_group=planning)
        self.create_service(GeoCosts, 'voxel_map/geo_costs', self._on_geo_costs, callback_group=planning)
        self.create_service(FrontierTargets, 'voxel_map/frontier_targets', self._on_frontier_targets,
                            callback_group=planning)
        self.create_service(TargetGoal, 'voxel_map/target_goal', self._on_target_goal,
                            callback_group=planning)
        self.snap_clusters = []

    def _on_cmd(self, msg):
        self.cmd = (msg.linear.x, msg.angular.z)
        self.cmd_time = self.get_clock().now().nanoseconds * 1e-9

    # -- 3-D planning on the floor graph -------------------------------------------------------
    def _take_snapshot(self):
        """Map thread: copy what the support graph needs (SupportGraph.snapshot)."""
        m = self.mapper
        t0 = time.time()
        snap = trav3d.SupportGraph.snapshot(m, visited=m.visited_layers(), tight=m.tight_layers())
        # going down: also the stairs without the margin to O_inf, for a stalled descent
        # (_loose_graph)
        loose = trav3d.SupportGraph.snapshot(
            m, visited=m.visited_layers(), tight=m.tight_layers(), stair_margin=False) \
            if self.climbing and self.climb_dir < 0 else None
        if self.climbing and self.climb_blocks:
            # before the snapshot is published: the planning threads build graphs from self.snap
            # at any time, and a change after that broke one (node count and index differed)
            for sn in (snap, loose):
                if sn is not None:
                    self._apply_climb_blocks(sn)
        self.snap, self.snap_loose = snap, loose
        # the frontier clusters of the same moment (voxel_map/frontier_targets)
        self.snap_clusters = [] if self.frontiers is None else [
            (c.id, c.voxels.copy(), c.centroid.copy()) for c in list(self.frontiers.clusters.values())]
        self.snap_wall = time.time()
        self.t_snap = getattr(self, 't_snap', 0.0) + (self.snap_wall - t0)
        self.n_snap = getattr(self, 'n_snap', 0) + 1

    def _apply_climb_blocks(self, snap):
        """Remove the support nodes of this climb's failure points from a snapshot (in place)."""
        trav, res, org, lo = snap['trav'], snap['res'], snap['origin'], snap['lo']
        r = self.climb_block_radius
        for x, y, z in self.climb_blocks:
            i0, i1 = (int(np.floor((v - org[0]) / res)) - lo[0] for v in (x - r, x + r))
            j0, j1 = (int(np.floor((v - org[1]) / res)) - lo[1] for v in (y - r, y + r))
            k0, k1 = (int(np.floor((v - org[2]) / res)) - lo[2] for v in (z - 0.15, z + 0.7))
            i0, j0, k0 = max(i0, 0), max(j0, 0), max(k0, 0)
            i1, j1, k1 = min(i1 + 1, trav.shape[0]), min(j1 + 1, trav.shape[1]), min(k1 + 1, trav.shape[2])
            if i0 < i1 and j0 < j1 and k0 < k1:
                trav[i0:i1, j0:j1, k0:k1] = False

    def _block_climb_front(self):
        """A climb stretch not reached (Sec. VI-C, eq. 17 on the stairs): v_front cannot be marked
        occupied on a flight, the voxel ahead is the next tread (a support surface in mode s), so
        the support nodes 0.3-0.5 m ahead of the robot (and up to 0.7 m above it) are taken out of
        the graph for the rest of this climb: the replan cannot pass the same spot again (00034:
        five stretches failed at one point, each with another end behind it)."""
        if self.mapper is None or self.support_z is None:
            return
        try:
            base_p, base_R = self._lookup(self.base_frame, Time())
        except TransformException:
            return
        yaw = math.atan2(base_R[1, 0], base_R[0, 0])
        c, s_ = math.cos(yaw), math.sin(yaw)
        new = [(base_p[0] + d * c - w * s_, base_p[1] + d * s_ + w * c, self.support_z)
               for d in (0.3, 0.4, 0.5) for w in (-0.1, 0.0, 0.1)]
        self.climb_blocks.extend(new)
        self.get_logger().warn(f'[stuck] climb stretch not reached at ({base_p[0]:.2f}, {base_p[1]:.2f}, z_r '
                               f'{self.support_z:.2f}): support nodes 0.3-0.5 m ahead taken out for this climb '
                               f'({len(self.climb_blocks) // 9} spots)')
        self._take_snapshot()

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

    def _loose_graph(self):
        """Descent fallback: the support graph with the stair treads that the one-cell margin to
        O_inf removed and stair nodes linked within loose_stair_reach cells. In Habitat the steps
        are continuous; the map only lost them to the inflation of nosings SAM3 missed."""
        snap = getattr(self, 'snap_loose', None)
        if snap is None:
            return None
        if getattr(self, 'loose_of', None) is not snap:
            self.loose = trav3d.SupportGraph(snap, stair_reach=self.loose_stair_reach, **self.graph_args)
            self.loose_of = snap
        return self.loose

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
        # targets are goals snapped onto free trav_grid cells (M_trav at the floor): their own
        # node is within a cell. With 0.5 m, a goal behind a wall took the node on this side of
        # it and the frontier there counted as reachable (tried again and again, never reached)
        C = g.costs(src, tgt, target_radius=self.geo_target_radius)
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

    def _on_frontier_targets(self, request, response):
        """J_geo(x_r, F_i) = min over C_i of the path cost, and its voxel v_target (the paper's ATSP
        nodes). The goal of a frontier is no longer its centroid mapped to the nearest graph node:
        a centroid behind a wall took the node on this side of it, and the frontier counted as
        reachable."""
        g = self._floor_graph()
        clusters = self.snap_clusters
        response.ok = g is not None and self.mapper is not None
        if not response.ok:
            return response
        src = (request.source.x, request.source.y, request.source.z)
        res = g.cluster_costs(src, [v for _, v, _ in clusters], self.mapper.logodds.shape)
        pos = g.positions() if g.n else None
        for (cid, _, cen), (J, tgt) in zip(clusters, res):
            if tgt is not None and self.goal_clearance > 0.0:
                # the goal Nav2 gets: v_target, or the nearest node with the goal clearance within
                # 0.5 m of it (on the graph, at its own height; was snapped on trav_grid at z_r)
                k = g.clear_node(g.node(tgt, 0.1), self._min_d_obs(), source=src)
                if k >= 0:
                    tgt = tuple(pos[k])
            tgt = cen if tgt is None else tgt
            response.ids.append(int(cid))
            response.costs.append(float(J))
            response.targets.append(Point(x=float(tgt[0]), y=float(tgt[1]), z=float(tgt[2])))
        return response

    def _floor_count(self, x, y, z):
        """M_trav source voxels (T) within one layer of z and floor_reanchor_radius (xy) of (x, y)."""
        m = self.mapper
        k = m.layer(z)
        r = int(math.ceil(self.floor_reanchor_radius / m.res))
        i = int(math.floor((x - m.origin_xy) / m.res))
        j = int(math.floor((y - m.origin_xy) / m.res))
        box = (slice(max(i - r, 0), min(i + r + 1, m.nx)), slice(max(j - r, 0), min(j + r + 1, m.ny)),
               slice(max(k - m.grid_layers, 0), min(k + m.grid_layers + 1, m.nz)))
        if box[2].start >= box[2].stop:
            return 0
        return int(m._source(box).any(axis=2).sum())

    def _maybe_reanchor(self, base_p, now):
        """floor_z re-anchored at z_r when z_r stayed at another layer (beyond floor_hold) for
        floor_reanchor_time s on the same floor (base_link within floor_reanchor_base_dz of its
        height at the anchor) and the map holds more floor there than at floor_z around the robot."""
        if self.anchor_base_z is None or abs(base_p[2] - self.anchor_base_z) > self.floor_reanchor_base_dz:
            self.reanchor_cand = None
            return
        k = self.mapper.layer(self.support_z)
        if self.reanchor_cand is None or self.reanchor_cand[0] != k:
            self.reanchor_cand = (k, now)
            return
        if now - self.reanchor_cand[1] < self.floor_reanchor_time:
            return
        n_new = self._floor_count(base_p[0], base_p[1], self.support_z)
        n_old = self._floor_count(base_p[0], base_p[1], self.floor_z)
        self.reanchor_cand = (k, now)              # checked; again after floor_reanchor_time
        if n_new <= n_old:
            self.get_logger().info(
                f'[floor] z_r {self.support_z:.3f} for {self.floor_reanchor_time:.0f} s, floor_z kept at '
                f'{self.floor_z:.3f}: {n_new} floor cells there vs {n_old}', throttle_duration_sec=10.0)
            return
        self.get_logger().warn(
            f'[floor] floor_z re-anchored {self.floor_z:.3f} -> {self.support_z:.3f} (z_r there for '
            f'{self.floor_reanchor_time:.0f} s, base_link z {base_p[2]:.3f} unchanged; {n_new} floor cells '
            f'vs {n_old} within {self.floor_reanchor_radius:.0f} m)')
        self.floor_z = self.floor_anchor = self.support_z
        self.reanchor_cand = None

    def _record_state(self, base_p, on_stairs):
        """Floor state for offline datasets: the support, the floor the map works on, and the
        traversable source T per voxel layer (cells; floors and their areas are read from it)."""
        m = self.mapper
        per_layer = m._source().sum(axis=(0, 1))
        stairs_layer = m.stairs.sum(axis=(0, 1))
        # value map (Sec. V-B) per voxel layer: max and mean of the fused score s over the voxels
        # with a value (keys are flat indices of the voxel map: layer = key % nz)
        v_max = np.zeros(m.nz)
        v_mean = np.zeros(m.nz)
        v_n = np.zeros(m.nz, dtype=int)
        keys, sv = self.value.keys, self.value.s
        if len(keys) and len(keys) == len(sv):
            lay = keys % m.nz
            v_n = np.bincount(lay, minlength=m.nz)
            v_mean = np.bincount(lay, weights=sv, minlength=m.nz) / np.maximum(v_n, 1)
            np.maximum.at(v_max, lay, sv)
        self.recorder.record(
            'state', base=[float(v) for v in base_p], support_z=self.support_z, floor_z=self.floor_z,
            floor_anchor=self.floor_anchor, climbing=bool(self.climbing),
            climb_dir=int(getattr(self, 'climb_dir', 0)) if self.climbing else 0, mode=m.mode,
            on_stairs=bool(on_stairs), origin_z=float(m.origin_z), res=float(m.res),
            trav_cells_per_layer=per_layer.astype(int), stair_cells_per_layer=stairs_layer.astype(int),
            value_max_per_layer=v_max.round(4), value_mean_per_layer=v_mean.round(4),
            value_voxels_per_layer=v_n.astype(int))

    def _min_d_obs(self):
        res = self.mapper.res if self.mapper is not None else 0.05
        return self.mapper_args['robot_radius'] + math.ceil(self.goal_clearance / res - 1e-9) * res + 1e-6

    def _on_target_goal(self, request, response):
        """Goal for a target object on the support graph (trav3d.SupportGraph.target_goal)."""
        g = self._floor_graph()
        response.ok = g is not None and self.mapper is not None
        if not response.ok or not request.points:
            return response
        src = (request.source.x, request.source.y, request.source.z)
        pts = np.array([(q.x, q.y, q.z) for q in request.points])
        r = g.target_goal(src, pts, r_min=request.r_min, r_goal=request.r_goal, k_min=max(1, request.k_min),
                          min_d_obs=self._min_d_obs() if self.goal_clearance > 0.0 else 0.0)
        response.n_near, response.n_ring = r['n_near'], r['n_ring']
        response.n_clear, response.n_reach = r['n_clear'], r['n_reach']
        response.found = r['node'] >= 0
        if response.found:
            q = g.positions()[r['node']]
            response.goal = Point(x=float(q[0]), y=float(q[1]), z=float(q[2]))
            response.cost = r['J']
        else:
            response.cost = float('inf')
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

    def _start_climb(self, direction=1):
        self.climb_dir = direction                 # +1 up, -1 down
        self.climbing = True
        self.n_climb = 0
        self.climb_last_end = None
        self.climb_looked = False
        self.climb_failed = []                     # stretch ends the robot did not reach
        self.n_geo_treads = 0
        self.climb_looked = False                  # turned around since the last stretch
        self.climb_look_k = 0                      # look-down steps since the last stretch
        self.climb_z0 = self.support_z             # the floor the climb left
        self.climb_far_n = 0                       # consecutive frames far from the stairs on a new level
        self.climb_blocks = []                     # failure points of this climb (_block_climb_front)
        self.climb_far_d = float('inf')
        if self.switch is not None and self.switch.request('s') or self.mapper.mode != 's':
            # navigation to a stair entrance switches to mode s (Sec. IV-A): stairs are
            # traversable only in mode s, and with them the floors are connected in the graph
            self.mapper.set_mode('s')
            self.mode_pub.publish(String(data='s'))
            self.get_logger().warn('[mode] -> s: stair climb requested')
        if self.vg is not None and direction < 0:
            # descending stairs lie below the forward-looking camera (Sec. IV-D): the downward-
            # looking behaviour without its trigger, how far down by its feedback loop - from the
            # stair cluster on (within descend_look_dist of the stair head, or on stair voxels).
            # On the way there over the floor the camera stays level (Pi_vg with its trigger):
            # forced from the start it looked 30-60 deg down for metres of floor (00890).
            self.vg_force_pending = True
        elif self.vg is not None and self.vg.active:   # going up: camera level for the climb
            self.vg.reset()
            self._publish_vg_pitch('[Pi_vg] climb up')
        elif direction < 0 and self.descend_pitch > 0.0:
            self._command_pitch('descend', self.descend_pitch)           # no Pi_vg: one fixed step
        self._take_snapshot()                      # the graph of mode s, with the stairs
        self.get_logger().info('[climb] started' + (' (down)' if direction < 0 else ''))
        self.climb_state_pub.publish(String(data='climbing'))

    def _defer(self, fn, delay=0.0):
        """Run fn on the map thread at its first depth frame `delay` s (wall clock) from now."""
        with self.map_tasks_lock:
            self.map_tasks.append((time.time() + delay, fn))

    def _run_map_tasks(self):
        now = time.time()
        with self.map_tasks_lock:
            due = [t for t in self.map_tasks if t[0] <= now]
            self.map_tasks = [t for t in self.map_tasks if t[0] > now]
        for _, fn in sorted(due, key=lambda t: t[0]):
            fn()

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
        self.climb_center = foot
        self.climb_end_z, self.climb_end_z_next = self.climb_end_z_next, None
        self._start_climb()
        self._climb_step(center=foot)

    def _on_descend_at(self, msg):
        """Go down the stairs whose head is at msg.point (frontier_goal_node, a stair landmark whose
        top is on this floor): as the climb, mirrored (camera pitched down, see _start_climb)."""
        if self.mapper is None:
            self.climb_state_pub.publish(String(data=f'done {self.support_z or 0.0:.3f}'))
            return
        head = np.array([msg.point.x, msg.point.y, msg.point.z])
        self.climb_center = head
        self.climb_end_z, self.climb_end_z_next = self.climb_end_z_next, None
        self._start_climb(direction=-1)
        self._climb_step(center=head)

    def _check_landmarks(self):
        """Map thread: the step check of every stair landmark, republished (see stair_check_risers)."""
        msg = self.stair_lm_msg
        if msg is None or self.mapper is None:
            return
        names = [f.name for f in msg.fields]
        base = ('x', 'y', 'z', 'top_x', 'top_y', 'top_z', 'size')
        rows = []
        if 'reliable' in names:
            lms = [tuple(map(float, r)) for r in point_cloud2.read_points(
                msg, field_names=base + ('reliable', 'belief', 'gamma'))]
        else:
            lms = [tuple(map(float, r)) + (1.0, float(r[6]), 1.0) for r in point_cloud2.read_points(msg, field_names=base)]
        for lm in lms:
            risers, n_top, levels = self.mapper.stair_steps(lm[0:3], lm[3:6])
            verdict = 1 if risers >= self.stair_check_risers else -1 if n_top >= self.stair_check_min_top else 0
            key = (round(lm[0], 1), round(lm[1], 1))
            if verdict < 0 and self.stair_verdicts.get(key) != verdict:
                self.get_logger().info(
                    f'[stairs] landmark at ({lm[0]:.2f}, {lm[1]:.2f}, {lm[2]:.2f}) -> top z {lm[5]:.2f} is no flight: '
                    f'{risers} riser(s) between its levels {[round(v, 2) for v in levels]} ({n_top} raised top cells seen)')
            self.stair_verdicts[key] = verdict
            rows.append(lm + (float(risers), float(verdict)))
        fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate(
            base + ('reliable', 'belief', 'gamma', 'risers', 'verdict'))]
        self.checked_lm_pub.publish(point_cloud2.create_cloud(msg.header, fields, rows))

    def _cancel_climb(self):
        """Map thread: stop the climb where it is (path following, turn in place, look-down) and
        report it as cancelled; not a failure of the stairs."""
        if not self.climbing:
            return
        timer = getattr(self, 'turn_timer', None)
        if timer is not None and not timer.is_canceled():
            timer.cancel()
            self.turn_cmd_pub.publish(Twist())
        if getattr(self, 'climb_turning', False):
            self.climb_turning = False
        self.climb_cancelled = True
        self.get_logger().info(f'[climb] cancelled after {self.n_climb} stretches at z_r {self.support_z:.2f}')
        self._end_climb()
        if self.following:
            self.following = False
            self._send_path(None, 'climb cancelled')     # an empty path ends the guard's path following

    def _end_climb(self):
        if self.debug_dump_dir and self.mapper is not None and self.support_z is not None:
            try:                                       # the stairs as the climb leaves them (debug)
                p0 = self._robot_support()
                self._dump_map(p0, half=4.0, below=3.5, above=3.5, tag='end')
            except Exception as e:                     # noqa: BLE001 (debug only)
                self.get_logger().warn(f'[climb] dump at the end failed: {e}')
        self.climbing = False
        self.vg_force_pending = False
        self.climb_blocks = []
        self._climb_level()
        if getattr(self, 'climb_dir', 1) < 0 and self.descend_pitch > 0.0:
            if self.vg is not None:                # the trigger decides again from here on
                self.vg.reset()
            self.vg_pitch = 0.0
            self._command_pitch('vg', 0.0)
            self._command_pitch('descend', 0.0)
        # still on the stairs (a landing half-way): not a new floor (00800 with SAM3: the descent
        # stopped at -1.61 m, which was then explored as a floor)
        # (standing on a stair voxel: a robot at the stair head of the new floor is near the top
        # step, 0.2 m below it, not on it)
        on = self.mapper.stair_distance(*self._robot_support()[:2], self.support_z, dz=0.1) < 0.3 \
            if self.mapper is not None else False
        # 'turned': the climb's last act was the turn in place (nothing further in view), so the
        # robot has looked all around where it ends; frontier_goal_node then skips its look-around
        turned = bool(getattr(self, 'climb_looked', False))
        cancelled, self.climb_cancelled = getattr(self, 'climb_cancelled', False), False
        self.climb_state_pub.publish(String(data=f'done {self.support_z:.3f}' + (' stairs' if on else '')
                                            + (' turned' if turned else '') + (' cancelled' if cancelled else '')))

    def _command_pitch(self, part, pitch):
        """Set one part of the camera pitch and send the largest of them (rad, + = down)."""
        self.pitch_parts[part] = max(0.0, float(pitch))
        p = max(self.pitch_parts.values())
        if abs(p - self.cmd_pitch) > 1e-6:
            others = {k: round(math.degrees(v)) for k, v in self.pitch_parts.items() if v > 0.0}
            self.get_logger().info(f'[pitch] camera {math.degrees(self.cmd_pitch):.0f} -> {math.degrees(p):.0f} '
                                   f'deg down (parts {others or "none"})')
        self.cmd_pitch = p
        self.pitch_pub.publish(Float64(data=p))

    def _publish_vg_pitch(self, what):
        """Command the pitch of the downward-looking behaviour; log when it changes."""
        # never above the look-down of a start on a landing (start_pitch): Pi_vg levelled the
        # camera during that first look-around when nothing was below for a moment (00800 0028)
        pitch = max(self.vg.pitch, getattr(self, 'start_pitch', 0.0))
        if abs(pitch - self.vg_pitch) > 1e-6:
            self.get_logger().info(f'{what}: camera {math.degrees(self.vg_pitch):.0f} -> '
                                   f'{math.degrees(pitch):.0f} deg down'
                                   + (f' ({self.vg.decision})' if self.vg.decision else ''))
        self.vg_pitch = pitch
        self._command_pitch('vg', pitch)

    def _vg_force_at_stairs(self, base_p, t):
        """Going down: force Pi_vg once the robot is at the stair cluster (see _start_climb)."""
        c = getattr(self, 'climb_center', None)
        d = math.hypot(base_p[0] - c[0], base_p[1] - c[1]) if c is not None else 0.0
        on = self.mapper is not None and self.mapper.stair_distance(base_p[0], base_p[1], self.support_z, dz=0.1) < 0.3
        if d <= self.descend_look_dist or on:
            self.vg_force_pending = False
            self.vg.force(t, (base_p[0], base_p[1]))
            self._publish_vg_pitch(f'[Pi_vg] at the stairs ({d:.1f} m from the stair head), going down')

    def _virtual_ground(self, pts, base_p, t):
        """Pi_vg (Sec. IV-D): the downward-looking behaviour, closed loop on the number of points
        observed below the virtual ground plane under the robot (VirtualGround). Going up stairs
        the camera stays level; going down the behaviour runs without its trigger (force)."""
        if self.vg is None or self.climbing and getattr(self, 'climb_dir', 1) > 0 \
                or getattr(self, 'climb_turning', False):
            return
        if self.climbing and getattr(self, 'vg_force_pending', False):
            self._vg_force_at_stairs(base_p, t)
        n = self.vg_n = self.vg.count(pts, base_p[0], base_p[1], self.support_z)
        was, p0 = self.vg.decision, self.vg_pitch
        active, changed = self.vg.update(n, t, (base_p[0], base_p[1]))
        if changed or active or self.vg_pitch != 0.0:
            # republished every frame while active: the blind-spot look of frontier_goal_node
            # levels the camera when it is done
            self._publish_vg_pitch(f'[Pi_vg] {n} points more than {self.vg.depth:.2f} m below '
                                   f'z_r={self.support_z:.2f} within {self.vg.radius:.1f} m')
        if active and self.vg.settled and self.vg.decision != was and abs(self.vg.pitch - p0) < 1e-6:
            self.get_logger().info(f'[Pi_vg] {math.degrees(self.vg.pitch):.0f} deg down: {self.vg.decision}')

    def _semantic_frontier_goal(self, graph, start, center, radius=4.0, min_rise=0.1):
        """Stair transition stage (Sec. VI-A): the semantic frontiers F_s on the staircase above
        the robot (within radius of center) are the candidates; the first node of the open ATSP
        tour over them (J_geo on the support graph) is the next goal. Returns (path, cluster
        centroid, number of candidates) or (None, None, 0)."""
        m = self.mapper
        sg = getattr(self, 'climb_dir', 1)                 # going down: everything mirrored in z
        c0 = start if center is None else center
        cands = []
        for c in self.frontiers.clusters.values():
            if sg * (c.centroid[2] - start[2]) <= min_rise or np.hypot(*(c.centroid[:2] - c0[:2])) > radius:
                continue
            if m.semantic_only_at(c.voxels).mean() < 0.5:
                continue
            node, J = graph.reach_near(start, m.centers(c.voxels), radius=1.0, dz=0.3,
                                       highest=sg > 0, lowest=sg < 0, min_xy=0.35)
            if node < 0 or sg * (graph.positions()[node][2] - start[2]) <= min_rise:
                continue                               # no stand beyond here: no progress
            p = graph.positions()[node]
            if any(np.linalg.norm(p - e) < 0.3 for e in self.climb_failed):
                continue
            cands.append((node, J, c))
        if not cands:
            return None, None, 0
        k = 0
        if len(cands) > 1:
            pts = [graph.positions()[n] for n, _, _ in cands]
            J = graph.costs([start] + pts, pts)
            order = solve_open_atsp(cost_matrix(J))
            k = order[0] - 1
        node, _, c = cands[k]
        return graph.path_to(start, node), c.centroid, len(cands)

    def _climb_step(self, center=None):
        """Next stretch of the stair climb: to a semantic frontier F_s above the robot, or, when
        there is none (the top of the stairs: the upper floor is a geometric frontier there), to
        the highest reachable node near the stair voxels and onto the landing (SupportGraph.climb)."""
        try:
            start = self._robot_support()
        except TransformException:
            return
        graph = self._floor_graph()
        t0 = time.time()
        path, why = None, ''
        last = getattr(self, 'climb_last_end', None)
        if graph is not None and graph.node(start, 0.6, dz=0.15) < 0:
            # z_r misread: at the top edge the robot stands half over the steps below and z_r read
            # 2.06 m on the 2.98 m landing (00800); no node there ended the climb. The robot is at
            # the end of the last stretch (or on the nearest node at another height).
            # Next to the stairs in mode s the robot's own cells can lie in O_inf (00810: the climb
            # ended at once, 0 stretches): plan from the nearest node within 1.5 m, the path
            # follower drives there first.
            if last is not None and np.hypot(*(start[:2] - last[:2])) < 0.4:
                start = np.array(last, dtype=np.float64)
            else:
                n = graph.node(start, 1.5, dz=0.3)
                if n < 0:
                    n = graph.node(start, 0.6, dz=1.0)
                if n >= 0:
                    start = graph.positions()[n].copy()
            self.get_logger().info(f'[climb] z_r {self.support_z:.2f} has no node under the robot: planning from '
                                   f'({start[0]:.2f}, {start[1]:.2f}, {start[2]:.2f})')
        if self.switch is not None and self.climb_far_n >= self.switch.exit_frames:
            # the transition is over (Sec. IV-A exit rule): a new level, sufficiently far from the
            # stair region for N_exit consecutive frames. (Before: >= 3 m^2 of flat floor connected
            # to the robot, a criterion the paper does not have.)
            self.get_logger().info(f'[climb] done after {self.n_climb} stretches: on z_r {self.support_z:.2f} '
                                   f'({self.support_z - self.climb_z0:+.2f} m from the start), '
                                   f'{self.climb_far_d:.1f} m from the stairs for {self.climb_far_n} frames '
                                   f'(> {self.switch.exit_distance:.1f} m)')
            self._end_climb()
            return
        if graph is not None:
            path, cen, n = self._semantic_frontier_goal(graph, start, center)
            if path is not None and last is not None and np.linalg.norm(path[-1] - last) < 0.1:
                self.climb_failed.append(path[-1])             # reached, nothing new seen: skip it
                path, cen, n = self._semantic_frontier_goal(graph, start, center)
            side = 'above' if self.climb_dir > 0 else 'below'
            if path is not None:
                why = (f'semantic frontier F_s at ({cen[0]:.2f}, {cen[1]:.2f}, {cen[2]:.2f}), '
                       f'{n} F_s {side}')
            else:
                path, _ = graph.climb(start, center=center, exclude=self.climb_failed,
                                      step_off=self.n_climb > 0, down=self.climb_dir < 0)
                why = (f'no F_s {side}: ' + ('highest stair node / landing' if self.climb_dir > 0
                                              else 'lowest stair node / floor below'))
        if path is None and self.climb_dir < 0 and self.climb_looked:
            # turned around and still nothing lower: the treads below may be there but cut off
            # by the margin to O_inf (00800 with SAM3: stopped at -2.36 m, lower floor -3.16 m)
            loose = self._loose_graph()
            if loose is not None:
                path, cen, n = self._semantic_frontier_goal(loose, start, center)
                if path is not None:
                    why = (f'semantic frontier F_s at ({cen[0]:.2f}, {cen[1]:.2f}, {cen[2]:.2f}), {n} F_s below, '
                           f'stairs without the margin to O_inf')
                else:
                    path, _ = loose.climb(start, center=center, exclude=self.climb_failed,
                                          step_off=self.n_climb > 0, down=True)
                    why = 'no F_s below: lowest stair node / floor below, stairs without the margin to O_inf'
        if path is None and self.climb_looked and graph is not None and self.mapper.stair_distance(
                *self._robot_support()[:2], self.support_z, dz=0.1) < 0.3:     # as _end_climb
            # still on a step with nothing further along the stairs: step off onto the floor at
            # their end (00800 going down: stopped on the bottom step, 0.2 m above the floor)
            for g, tag in ((graph, ''), (self._loose_graph() if self.climb_dir < 0 else None,
                                         ', stairs without the margin to O_inf')):
                if g is not None:
                    path, _ = g.step_off(start, down=self.climb_dir < 0, exclude=self.climb_failed)
                    if path is not None:
                        why = 'stepping off the last step onto the floor' + tag
                        break
        if path is None and self.n_climb == 0 and graph is not None and center is not None:
            # nothing to climb toward yet: first go to the entrance of the stairs on this floor.
            # Going down, the steps below become trusted only when seen from the stair head
            # (00800 from upstairs: the descent ended at once, 7 m away from the stairs).
            path, _ = graph.path(start, center, goal_radius=1.0)
            if path is not None and len(path) > 1:
                why = 'to the entrance of the stairs first'
            else:
                path = None
        if path is None and self.n_climb > 0 and not self.climb_looked and self.climb_dir > 0 \
                and self.climb_look_step > 0.0:
            # nothing higher in view: look down while the frontier at the robot's feet is unseen
            # (see climb_look_step)
            n = self._near_frontier_voxels(start)
            nxt = self.climb_pitch + self.climb_look_step
            if n > 0 and nxt <= self.climb_look_max + 1e-6:
                self.climb_look_k += 1
                self._climb_set_pitch(nxt, f'[climb] nothing higher in view at z_r {self.support_z:.2f}, {n} '
                                           f'frontier voxels within {self.climb_look_radius:.1f} m unobserved: '
                                           f'looking down')
                self._defer(self._climb_look_done, self.climb_look_time)
                return
            if self.climb_look_k:
                self.get_logger().info('[climb] look-down: ' + ('the frontier at the feet observed' if n == 0 else
                                       f'{n} frontier voxels still unobserved at '
                                       f'{math.degrees(self.climb_pitch):.0f} deg'))
            self._climb_level()
        if path is None and self.n_climb > 0 and not self.climb_looked:
            # nothing higher in view half-way up: the flight turns at a landing and the next one
            # is outside the camera's view (00810 with SAM3: the climb ended at 1.48 m). Turn in
            # place once, then plan again with what was seen.
            self.climb_looked = True
            if self.climb_dir < 0:
                # going down: the flight below the landing may have been seen from upstairs
                # (remembered stair voxels) but lie more than the window below z_r
                n = self.mapper.promote_stairs(self.support_z, self.stair_window, self.stair_column_gap, below=1.0)
                if n:
                    self.get_logger().info(f'[climb] {n} remembered stair voxels down to 1 m below trusted')
            self.get_logger().info(f'[climb] nothing {"higher" if self.climb_dir > 0 else "lower"} in view at '
                                   f'z_r {self.support_z:.2f}: turning around once to look for the rest of the stairs')
            self._climb_turn(start_time=time.time())
            return
        if path is not None and any(np.linalg.norm(path[-1] - e) < 0.3 for e in self.climb_failed):
            # every planner above skips the stretch ends not reached; one that still ends there
            # would be sent again and again (00803: the same landing node 7 times)
            self.get_logger().info(f'[climb] the next stretch ends where one was not reached before '
                                   f'({path[-1][0]:.2f}, {path[-1][1]:.2f}, {path[-1][2]:.2f}); not again')
            path = None
            if self.n_climb > 0 and not self.climb_looked:
                self.climb_looked = True
                self.get_logger().info(f'[climb] turning around once to look for the rest of the stairs')
                self._climb_turn(start_time=time.time())
                return
        if path is None:
            diag = ''
            if graph is not None:
                s = graph.node(start, 0.6, dz=0.3)
                if s < 0:
                    diag = '; no support-graph node at the robot'
                else:
                    dist, _ = graph._search(s)
                    near = graph.near_stairs()
                    diag = (f'; from ({start[0]:.2f}, {start[1]:.2f}, {start[2]:.2f}): {int(np.isfinite(dist).sum())} '
                            f'reachable nodes, {int((np.isfinite(dist) & near).sum())} of them near stair voxels '
                            f'({int(near.sum())} near stairs in all)')
            self.get_logger().info(f'[climb] done after {self.n_climb} stretches: no semantic frontier or '
                                   f'{"higher" if self.climb_dir > 0 else "lower"} stair voxel reachable '
                                   f'(z_r {self.support_z:.2f}){diag}')
            self._dump_map(start)
            self._end_climb()
            return
        if last is not None and np.linalg.norm(path[-1] - last) < 0.1:
            self.get_logger().info(f'[climb] done after {self.n_climb} stretches: the next stretch ends where '
                                   f'the last one did (z_r {self.support_z:.2f})')
            self._end_climb()                          # the same stretch again: nothing gained
            return
        self.climb_last_end = path[-1]
        self.climb_looked = False
        self.climb_look_k = 0
        if self.climb_dir > 0:
            self._climb_level()                        # going up: the camera level for the climb
        self.n_climb += 1
        self._send_path(path, f'climb stretch {self.n_climb} ({why}; {1e3 * (time.time() - t0):.0f} ms)')

    def _dump_map(self, start, half=3.0, below=1.0, above=2.5, tag='stop'):
        """debug_dump_dir: the voxel layers around the robot where a climb stopped (or ended)."""
        if not self.debug_dump_dir:
            return
        import os
        m = self.mapper
        x, y, z = (float(v) for v in start)
        i0, j0 = (max(int((c - half - m.origin_xy) / m.res), 0) for c in (x, y))
        i1, j1 = min(int((x + half - m.origin_xy) / m.res), m.nx), min(int((y + half - m.origin_xy) / m.res), m.ny)
        k0, k1 = max(m.layer(z - below), 0), min(m.layer(z + above) + 1, m.nz)
        sl = (slice(i0, i1), slice(j0, j1), slice(k0, k1))
        os.makedirs(self.debug_dump_dir, exist_ok=True)
        path = os.path.join(self.debug_dump_dir, f'climb_{tag}_' + time.strftime('%Y%m%d_%H%M%S') + f'_{self.n_dump}.npz')
        self.n_dump += 1
        np.savez_compressed(
            path, occupied=m.occupied[sl], trav_g=m.trav_g[sl], trav_s=m.trav_s[sl], stairs=m.stairs[sl],
            stair_seen=m.stair_seen[sl], inflated=m.inflated[sl], source=m._source(sl), traversable=m.traversable()[sl],
            origin=np.array([m.origin_xy + i0 * m.res, m.origin_xy + j0 * m.res, m.origin_z + k0 * m.res]),
            res=m.res, robot=np.array([x, y, z]), support_z=self.support_z, climb_dir=getattr(self, 'climb_dir', 1),
            mode=m.mode, dz_lo=m.dz_lo, n_h=m.n_h, r_cells=m.r_cells, sem_src=m.sem_src[sl],
            logodds=m.logodds[sl])
        self.get_logger().info(f'[climb] map around the {tag} saved to {path}')

    def _climb_turn(self, start_time):
        """Turn in place (one revolution at climb_turn_speed), then the next climb step."""
        dur = 2.0 * math.pi / self.climb_turn_speed
        self.turn_start = start_time
        if getattr(self, 'climb_dir', 1) < 0 and self.vg is not None:
            # going down, Pi_vg held the camera 30-60 deg down (the stair well, the steps below);
            # the turn looks for the rest of the stairs or the floor around: camera level, Pi_vg
            # paused until the turn is over (_climb_turn_done)
            self.climb_turning = True
            self.vg.reset()
            self._publish_vg_pitch('[Pi_vg] climb turn')
        # the turn commands on the requests group (they do not touch the map); the climb step
        # after it on the map thread
        self.turn_timer = self.create_timer(0.1, lambda: self._climb_turn_tick(dur),
                                            callback_group=self.requests_group)

    def _climb_turn_tick(self, dur):
        elapsed = time.time() - self.turn_start
        msg = Twist()
        if elapsed < dur:
            msg.angular.z = self.climb_turn_speed
            self.turn_cmd_pub.publish(msg)
            return
        self.turn_cmd_pub.publish(msg)                 # stop
        self.turn_timer.cancel()
        self._defer(self._climb_turn_done, max(0.0, dur + 1.0 - elapsed))   # let the last frames in

    def _climb_turn_done(self):
        was = getattr(self, 'climb_turning', False)
        self.climb_turning = False
        if self.climbing and not self.following:
            self._take_snapshot()
            self._climb_step()
        if was and self.climbing and getattr(self, 'climb_dir', 1) < 0 and self.vg is not None \
                and not getattr(self, 'vg_force_pending', False):
            try:                                       # still going down: Pi_vg again (as _start_climb)
                self.vg.force(None, tuple(self._robot_support()[:2]))
            except TransformException:
                return
            self._publish_vg_pitch('[Pi_vg] going down stairs')

    def _near_frontier_voxels(self, start):
        """Frontier voxels of the clusters within climb_look_radius of the robot (x-y), from 0.3 m
        below to 0.8 m above its support (the landing and the first steps of a next flight)."""
        if self.frontiers is None:
            return 0
        x, y, z = (float(v) for v in start)
        return sum(len(c.voxels) for c in list(self.frontiers.clusters.values())
                   if math.hypot(c.centroid[0] - x, c.centroid[1] - y) <= self.climb_look_radius
                   and -0.3 <= c.centroid[2] - z <= 0.8)

    def _climb_look_done(self):
        if self.climbing and not self.following:
            self._take_snapshot()                      # with what the lowered camera saw
            self._climb_step()

    def _climb_set_pitch(self, pitch, what):
        if abs(pitch - self.climb_pitch) > 1e-6:
            self.get_logger().info(f'{what}: camera {math.degrees(self.climb_pitch):.0f} -> '
                                   f'{math.degrees(pitch):.0f} deg down')
        self.climb_pitch = pitch
        self._command_pitch('climb', pitch)

    def _climb_level(self):
        """Camera level again after the look-down of an upward climb (going down, Pi_vg owns it)."""
        if self.climb_pitch != 0.0:
            self._climb_set_pitch(0.0, '[climb] camera level again')

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
                    # execute-review (Sec. VI-C, eq. 17): the failure goes into the map at once,
                    # v_front occupied, so the replan does not take the same way (00366: two
                    # stretches failed at the stair top and the next path went back down the
                    # stairs and around to the same place)
                    self._defer(self._block_climb_front)
            except TransformException:
                pass
        if was and not msg.data and self.climbing:
            # stretch done: plan the next one once the steps above are in view and mapped (the
            # stair window rises with z_r), on a snapshot taken after that
            self._defer(self._climb_timer, 1.0)

    def _climb_timer(self):
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
        if self.stuck_close_tight:
            self._close_tight(base_p[0], base_p[1], yaw, 'stuck')
        self._update_frontiers()
        self.n_stuck += 1
        self.get_logger().warn(
            f'[stuck] #{self.n_stuck} at ({base_p[0]:.2f}, {base_p[1]:.2f}): commanded forward motion '
            f'blocked, {n_new} voxel in front marked occupied')
        self._publish(stamp)
        self._take_snapshot()                          # the graph sees the block at once
        self.stuck_pub.publish(Empty())                # frontier_goal_node: review the goal

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
        self.hfov = 2.0 * math.atan(0.5 * msg.width / K[0, 0])
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

    def _on_itm(self, msg):
        """Value map update (eq. 7) from one image's relevance: onto the traversable voxels its
        depth frame sees (within max_range), weighted by cos^2(pi theta / theta_FOV) of their
        horizontal angle to the camera heading."""
        if self.mapper is None or self.rays is None or self.hfov is None:
            return
        key = msg.header.stamp.sec * 10 ** 9 + msg.header.stamp.nanosec
        entry = self.depth_cache.get(key)
        if entry is None:
            self.get_logger().warn('[value] no depth frame for an image-text score (kept 1.5 s)',
                                   throttle_duration_sec=10.0)
            return
        depth, frame = entry
        try:
            cam_p, cam_R = self._lookup(frame, Time.from_msg(msg.header.stamp))
        except TransformException:
            return
        m = self.mapper
        z = depth[self.pix].astype(np.float64)
        ok = np.isfinite(z) & (z > 0.0)
        pc = self.rays[ok] * z[ok, None]                      # camera (optical) frame
        ok = np.linalg.norm(pc, axis=1) <= self.max_range
        pc = pc[ok]
        w = view_weight(np.arctan2(pc[:, 0], pc[:, 2]), self.hfov)
        ijk, inside = m._ijk(pc @ cam_R.T + cam_p)
        ijk, w = ijk[inside], w[inside]
        flat = (ijk[:, 0] * m.ny + ijk[:, 1]) * m.nz + ijk[:, 2]
        # the hit voxel or the one below it (a floor point on a layer boundary)
        below = np.where(ijk[:, 2] > 0, flat - 1, flat)
        keys = np.where(m.traversable_at(flat), flat, np.where(m.traversable_at(below), below, -1))
        sel = keys >= 0
        n = self.value.update(keys[sel], w[sel], msg.score)
        self.n_itm += 1
        if self.n_itm % 100 == 0:
            self.get_logger().info(f'[value] {self.n_itm} images, {len(self.value)} voxels with a value, '
                                   f'last score {msg.score:.3f} on {n} voxels, max value '
                                   f'{self.value.s.max() if len(self.value) else 0.0:.3f}')
        now = time.time()
        if now - self.last_value_pub >= 1.0:
            self.last_value_pub = now
            self._publish_value(Header(stamp=msg.header.stamp, frame_id=self.map_frame))

    def _publish_value(self, h):
        if not len(self.value):
            return
        m = self.mapper
        k = self.value.keys
        ijk = np.stack([k // (m.ny * m.nz), (k // m.nz) % m.ny, k % m.nz], axis=1)
        xyz = (ijk + 0.5) * m.res + [m.origin_xy, m.origin_xy, m.origin_z]
        data = np.column_stack([xyz, self.value.s, self.value.r]).astype(np.float32)
        fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                  for i, n in enumerate(('x', 'y', 'z', 'value', 'reliability'))]
        self.value_pub.publish(point_cloud2.create_cloud(h, fields, data))

    def _lookup(self, frame, stamp):
        t = self.tf_buffer.lookup_transform(self.map_frame, frame, stamp, timeout=Duration(seconds=0.2))
        tr = t.transform.translation
        return np.array([tr.x, tr.y, tr.z]), quat_to_rot(t.transform.rotation)

    def _estimate_floor(self, pts, base_p, cam_z=None):
        """Height of the support plane: mode of the heights of nearby low points (2 cm bins).

        The first estimate (no z_r yet, cam_z given) is the lowest cluster of at least
        floor_min_points points within start_floor_band of camera_height below the camera, not
        the mode of the points up to floor_max_above over base_link: base_link floats 0.07-0.18 m
        over the floor in HM3D, so that window reached 0.17-0.28 m over the floor, and with the
        level camera seeing little floor near the robot a low furniture top outnumbered it
        (00813 0040, 00829 0028, 00880 0093: z_r 0.16-0.25 m over the floor habitat_server
        measured; the floor was never in M_g and the robot was left on an island). Lowest: the
        floor is the lowest surface of the room; the band around the camera height keeps a lower
        storey seen down a stairwell out."""
        near = np.hypot(pts[:, 0] - base_p[0], pts[:, 1] - base_p[1]) < self.floor_radius
        if self.support_z is None and cam_z is not None:
            ref = cam_z - self.camera_height
            z = pts[near & (np.abs(pts[:, 2] - ref) <= self.start_floor_band), 2]
            if z.size < self.floor_min_points:
                return None
            hist, edges = np.histogram(z, bins=np.arange(ref - self.start_floor_band,
                                                         ref + self.start_floor_band + 0.02, 0.02))
            for i in range(len(hist)):
                lo, hi = max(i - 1, 0), min(i + 2, len(hist))
                if hist[lo:hi].sum() >= self.floor_min_points:
                    j = lo + int(np.argmax(hist[lo:hi]))             # the peak of the lowest cluster
                    peak = 0.5 * (edges[j] + edges[j + 1])
                    return float(np.mean(z[np.abs(z - peak) < 0.03]))
            return None
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
        if self.map_tasks:
            self._run_map_tasks()                      # climb requests and steps (map thread)
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
        if getattr(self, 'start_pitch', 0.0) > 0.0 and self.support_z is not None:
            # started on a landing: the first look-around (frontier_goal_node) with the camera
            # start_look_min down, until the robot has moved start_look_move m or start_look_hold s
            # passed; then level again
            if getattr(self, 'start_look', None) is None:
                self.start_look = (base_p[0], base_p[1], now)
                if abs(self.start_pitch - self.start_look_min) > 1e-6:
                    self.start_pitch = self.start_look_min
                    self._command_pitch('start', self.start_pitch)
                self.get_logger().info(f'[start] on a landing: looking around with the camera '
                                       f'{math.degrees(self.start_pitch):.0f} deg down')
            sx, sy, st = self.start_look
            if math.hypot(base_p[0] - sx, base_p[1] - sy) >= self.start_look_move or now - st >= self.start_look_hold:
                self.start_pitch = 0.0
                self._command_pitch('start', 0.0)
                self.get_logger().info('[start] camera level again')
        if self.support_z is None or self.track_floor:
            first = self.support_z is None
            self.support_z = self._estimate_floor(pts[hit], base_p, cam_z=cam_p[2] if first else None)
            if first and self.support_z is not None:
                est = self.support_z
                self.support_z, rejected = start_support(est, base_p[2], cam_p[2], self.camera_height,
                                                         self.start_floor_max_below)
                if rejected:
                    self.get_logger().warn(
                        f'level floor in view at z={est:.3f}, {base_p[2] - est:.2f} m under base_link: not '
                        f'the support of the robot (a stair landing?); support taken {self.camera_height:.2f} m '
                        f'below the camera, z_r={self.support_z:.3f} (base_link z={base_p[2]:.3f})')
            if self.support_z is None:
                # no level floor in view: the robot starts on the stairs (HM3D ObjectNav v1 has
                # such starts, 00843 / 00873): the map never began and the episode ended at once.
                # After floor_wait s (the look-around at the start shows the floor if there is
                # one in reach) the support is camera_height below the camera (habitat_server
                # puts it 0.88 m above the floor under the start; base_link is 0.2 m below to
                # 0.08 m above the floor in HM3D, so base_link - 0.05 m was 0.15 m off in 00835
                # and every frontier was unreachable).
                # (not t0: that is the start of this frame's timing, the map ms/frame of the log)
                w0 = self.floor_wait_start = getattr(self, 'floor_wait_start', None) or now
                # a start on a stair landing (00800 0028, 00835 0072): with the camera level the
                # flights up and down and the floors they lead to stay below its view, and the map
                # stayed a patch around the landing. Look down while waiting, one step (start_
                # look_step, 15 deg) every start_look_period s from start_look_min to start_look_max
                if self.start_look_max > 0.0:
                    k = int((now - w0) / self.start_look_period)
                    pitch = min(self.start_look_min + k * self.start_look_step, self.start_look_max)
                    if abs(pitch - getattr(self, 'start_pitch', 0.0)) > 1e-6:
                        self.get_logger().info(f'[start] no level floor in view: camera '
                                               f'{math.degrees(getattr(self, "start_pitch", 0.0)):.0f} -> '
                                               f'{math.degrees(pitch):.0f} deg down')
                        self.start_pitch = pitch
                        self._command_pitch('start', pitch)
                if now - w0 >= self.floor_wait:
                    self.support_z = cam_p[2] - self.camera_height
                    self.get_logger().warn(f'no level floor in view for {self.floor_wait:.0f} s (on the stairs?): '
                                           f'support taken {self.camera_height:.2f} m below the camera, '
                                           f'z_r={self.support_z:.3f} (base_link z={base_p[2]:.3f})')
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
            self.anchor_base_z = base_p[2]
            self.reanchor_cand = None
        elif abs(self.support_z - self.floor_anchor) <= self.floor_hold:
            self.floor_z = self.support_z
            self.reanchor_cand = None
        elif self.mapper is not None:
            self._maybe_reanchor(base_p, now)
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
            below = self.stair_window if self.climbing and getattr(self, 'climb_dir', 1) < 0 else 0.3
            n_prom = self.mapper.promote_stairs(self.support_z, self.stair_window, self.stair_column_gap,
                                                below=below)
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
        self._virtual_ground(pts[hit], base_p, now)
        n_changed = self.mapper.integrate(cam_p, pts, hit, support,
                                          update_g=self.mapper.mode == 'g' or not on_stairs)
        if self.recorder and now - self.last_record >= self.record_period:
            self.last_record = now
            try:                                       # a record never takes the map down
                self._record_state(base_p, on_stairs)
            except Exception as e:                     # noqa: BLE001
                self.get_logger().warn(f'[record] state not written: {e!r}', throttle_duration_sec=30.0)
        if self.geo_treads and self.climbing and self.mapper.mode == 's' and self.frames % 3 == 0:
            # treads of the turn that SAM3 left unlabelled: by step geometry (VoxelMapper.geometric_treads)
            going_up = getattr(self, 'climb_dir', 1) > 0
            cc = getattr(self, 'climb_center', None)
            add, steps = self.mapper.geometric_treads(base_p[0], base_p[1], self.support_z,
                                               up=1.2 if going_up else 0.3, down=0.3 if going_up else 1.2,
                                               hint=None if cc is None else (cc[0], cc[1], 1.5))
            if add.size:
                # the steps also as stair voxels: the climb planner goes over nodes near stairs
                self.mapper.integrate_semantic(self.mapper.centers(add),
                                               self.mapper.centers(steps) if steps.size else None,
                                               src=VoxelMapper.SRC_GEO, stair_src=VoxelMapper.SRC_GEO_STEP)
                self.n_geo_treads += int(add.size)
                self.get_logger().info(f'[climb] {add.size} voxels of unlabelled treads by step geometry '
                                       f'({self.n_geo_treads} this climb)', throttle_duration_sec=2.0)
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
                + (f', stair points trusted {self.n_stair_kept}/{self.n_stair_raw}' if self.n_stair_raw else '')
                + (f', Pi_vg {getattr(self, "vg_n", 0)} below' + (' (looking down)' if self.vg.active else '')
                   if self.vg is not None else ''))

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
        # going down, the steps below are trusted as far as the window reaches above when climbing
        below = self.stair_window if self.climbing and getattr(self, 'climb_dir', 1) < 0 else 0.3
        stair_pts = self.mapper.trusted_stairs(pts[stairs], others, self.support_z,
                                               self.stair_window, self.stair_column_gap, below=below)
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
        if self.switch.mode == 's' and (self.switch.exit_rule == 'distance' or self.climbing):
            try:
                base_p, _ = self._lookup(self.base_frame, Time.from_msg(stamp))
            except TransformException:
                return
            d_map = self.mapper.stair_distance(base_p[0], base_p[1], self.support_z)
        before = self.switch.mode
        if self.climbing:
            # stay in mode s until the climb is over; the climb ends by the exit rule of the mode
            # (Sec. IV-A): on another level than the floor it left, farther than exit_distance from
            # the stairs for exit_frames consecutive frames (counted here, acted on in _climb_step)
            new_level = self.support_z is not None and abs(self.support_z - self.climb_z0) >= self.floor_min_change
            self.climb_far_n = self.climb_far_n + 1 if new_level and d_map > self.switch.exit_distance else 0
            self.climb_far_d = d_map
            if self.climb_far_n == self.switch.exit_frames and not self.following:
                self._climb_step()                     # waiting (look, turn): end it now
            return
        if self.switch.mode == 's' and self.support_z is not None:
            # a climb that ended on a landing: the robot stands on stair voxels, which are not
            # traversable in mode g, so no switch back while it is there (stairs out of view or not)
            try:
                rx, ry = self._robot_support()[:2]
            except TransformException:
                return
            if self.mapper.stair_distance(rx, ry, self.support_z, dz=0.1) < 0.3:
                self.switch.n_run = 0
                return
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
               f'no stairs within {self.switch.enter_range:.1f} m in view for {self.switch.exit_frames} frames'
               if self.switch.exit_rule == 'sight' else
               f'{d_map:.1f} m from the stairs for {self.switch.exit_frames} frames')
        self.get_logger().warn(
            f'[mode] {before} -> {self.switch.mode} at ({base_p[0]:.2f}, {base_p[1]:.2f}): {why} '
            f'(re-inflation {1e3 * (time.time() - t0):.0f} ms, {int(self.mapper.stairs.sum())} stair voxels)')
        self.mode_pub.publish(String(data=self.switch.mode))
        self._publish_mode_marker(Header(stamp=stamp, frame_id=self.map_frame))

    def _on_dormant(self, msg):
        if self.frontiers is None:
            return
        c = self.frontiers.mark_visited(int(msg.data))
        if c is None:
            self.get_logger().info(f'[dormant] frontier cluster {msg.data} is gone already')
        else:
            self.get_logger().info(
                f'[dormant] frontier cluster {msg.data} (now {c.id}, {c.size} voxels at '
                f'({c.centroid[0]:.2f}, {c.centroid[1]:.2f})): visited until its voxels change')

    def _on_release(self, _msg):
        if self.frontiers is None:
            return
        n = self.frontiers.release_dormant()
        self.get_logger().info(f'[dormant] released: {n} dormant frontier cluster(s) are candidates again')

    def _on_failed(self, msg):
        if self.frontiers is None:
            return
        c = self.frontiers.mark_failed(int(msg.data), self.frontier_max_attempts)
        if c is not None:
            self.get_logger().info(
                f'[dormant] frontier cluster {msg.data} (now {c.id}) not reached: attempt {c.fails} of '
                f'{self.frontier_max_attempts}' + ('; dormant until its voxels change' if c.visited else ''))

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
        cl = list(self.frontiers.clusters.values())
        sem = [float(m.semantic_only_at(c.voxels).mean()) if c.size else 0.0 for c in cl]
        mk = Marker(header=h, ns='frontier_centroids', id=0, type=Marker.SPHERE_LIST, action=Marker.ADD)
        mk.pose.orientation.w = 1.0
        mk.scale.x = mk.scale.y = mk.scale.z = 0.2
        mk.color = ColorRGBA(r=1.0, g=0.85, b=0.0, a=1.0)
        for c, f in zip(cl, sem):                       # F_g yellow, F_s magenta
            mk.points.append(Point(x=float(c.centroid[0]), y=float(c.centroid[1]), z=float(c.centroid[2]) + 0.1))
            mk.colors.append(ColorRGBA(r=1.0, g=0.2, b=1.0, a=1.0) if f >= 0.5 else ColorRGBA(r=1.0, g=0.85, b=0.0, a=1.0))
        self.centroid_pub.publish(mk)
        fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                  for i, n in enumerate(('x', 'y', 'z', 'size', 'semantic', 'id', 'dormant', 'fails'))]
        clusters = [(*map(float, c.centroid), float(c.size), f, float(c.id), float(c.visited), float(c.fails))
                    for c, f in zip(cl, sem)]
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
