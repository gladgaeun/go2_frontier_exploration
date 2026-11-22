#!/usr/bin/env python3
# Frontier goal service on the 3D traversable frontiers: the drop-in counterpart of
# frontier_exploration's classical_frontier_detector for the voxel map.
#
# Inputs : voxel_map/frontier_clusters (centroid p_i + size per cluster, voxel_mapper_node),
#          voxel_map/trav_grid (support layer: free = M_trav, occupied = O_inf), TF map->base_link
# Service: frontier_pose (frontier_interfaces/FrontierGoal), same contract as the 2D detector:
#          the goal_rank-th best frontier goal, or an empty frame_id when there is none.
# explore_executor_node (the exploration loop: goal at the robot / without a path -> next rank,
# Nav2, termination) runs on top of it; it keeps no (x, y) blacklist (the 2-D Go2 loop's turned
# down frontiers of another floor, 00891): failures are the dormant clusters here and the blocked
# voxels of execute-review (eq. 17). Goal placement is the 2D algorithm (see
# goal_selection.py). Clusters the robot cannot reach on the Nav2 grid are left out
# (goal_selection.reachable_mask) and shown on frontier_goal/unreachable (grey spheres).
#
# Reachability and goals (the paper's ATSP nodes): voxel_map/frontier_targets gives J_geo(x_r, F_i)
# = min over the cluster's own voxels of the path cost, and that voxel v_target; a cluster is
# reachable when J is finite, and its goal is snapped to a clear cell from v_target (not from the
# centroid, which may lie behind a wall).
# Order (goal_order): 'atsp' (default) is TravExplorer's open ATSP over {x_r} U F_g (tour.py): rank
# 0 is the first node of the tour, rank k the first node of the tour solved again without the
# first k (the explorer asks for the next rank when a goal is at the robot or has no path:
# the paper moves such a frontier to the dormant set and replans). The tour is drawn on
# frontier_goal/tour. 'score' is the 2D ranking (0.5 / distance + 0.5 * size).
# Dormant set (atsp): a frontier still there when the robot stands at its goal cannot be cleared
# by visiting it (floor under a table is hidden by the table top from the camera, so the edge of
# M_g stays). Only the cluster of the goal last sent is checked (its neighbours were not visited),
# and the flag is the visited flag of that cluster in the voxel map (voxel_map/frontier_dormant ->
# frontier3d): it stays while the cluster's voxels stay, and is reset when an update changes them
# (the paper resets visited flags with the clusters an update removes), so new space seen there
# is explored again. Shown as blue spheres on frontier_goal/unreachable.
# Stuck on the way (voxel_map/stuck, after the map marked v_front): one failed attempt for the
# frontier goal and the Nav2 goal is canceled at once, the tour solved again on the updated map
# (left to Nav2, its recoveries backed up, spun and tried the same goal again where the navmesh had
# a gap); stuck again there, the frontier is dormant. Not within stuck_grace (3 s) of a new goal.
# Goal validation while driving (execute-review, Sec. IV-D): when the cluster of the frontier goal
# being driven to is gone from frontier_vanish_msgs (2) consecutive cluster messages (~1-2 s, the
# hysteresis), the Nav2 goal is canceled (navigate_to_pose cancel, all goals) and the explorer
# asks again: the tour is solved anew. A canceled goal is not a failure of the cluster.
# Repeated failures (execute-review, Sec. IV-D): when the goal last sent was not reached (Nav2
# gave up, the guard held the robot) or the explorer turned it down (at the robot, no path), that
# cluster gets one failed attempt (voxel_map/frontier_failed ->
# frontier3d, the count goes with the cluster while its voxels stay); at frontier_max_attempts (2)
# it is dormant. The map side of the review is there already: the guard stops within 0.1 m and
# marks the cells ahead too narrow (mark_tight), and two Nav2 failures at one spot mark v_front
# occupied (collision_guard/block).
# Blind-spot frontiers (Sec. IV-D, FOV-aware active perception): a frontier cluster closer to the
# robot than d_blind (d_i = |p_i - x_r|, its centroid) and inside the camera's horizontal FOV lies
# below the lower edge of the forward-looking camera's view (a cluster beside or behind the robot
# is out of view, not in the blind spot: pitching does not show it). Before such a
# frontier can become dormant, the robot looks down (camera/pitch_cmd), closed loop as Pi_vg in
# voxel_mapper_node: one look_down step (look_down_pitch, 15 deg) at a time, look_time s each, a
# further step while frontier voxels of the near clusters are still unobserved (up to
# look_max_pitch, 60 deg), then level again; if the hidden
# voxels were seen the frontier is gone, otherwise it is irreducible and goes to the dormant set
# at once (look_demote). Their dormant flag is the cluster's visited flag
# (below), so it goes when the space there changes. Each place is checked once. d_blind is not given in the paper: 1.5 m, about where the level
# camera (0.88 m high, 63 deg vertical FOV) starts to see the floor.
# Look-around (not in the paper): at the start and on arriving on a new floor the robot turns in
# place once (cmd_vel, look_around_speed) before a plan, so the space behind it is seen - only
# when the frontiers are few: fewer than look_around_min_frontiers reachable, not dormant clusters
# on this floor. With enough frontiers in view the exploration starts at once; the turn stays
# pending until the next plan with too few (or the next floor). Without it a frontier at the edge
# of the first view (behind the robot) was reached with the robot facing away, stayed, and went
# dormant unseen (HM3D 00810: exploration ended after 2 goals).
#
# Target objects (Sec. V-A, eq. 9 "target objects dominate"): with `target` set (an object query of
# the instance map, e.g. "couch"), a reliable cluster of it (instance_map/targets) that stands on
# this floor comes before every frontier: rank 0 is a free cell next to it (within
# success_distance - 0.35 m, the cheapest on the support graph) and the frontier ranks move up by
# one. Once the robot is within success_distance (xy) of the object, there is no goal any more:
# the exploration ends with the target found.
# Semantic frontiers (Sec. VI-A, eq. 14): a cluster with at least semantic_share of its voxels
# traversable only through M_s (voxel_map/frontier_clusters field semantic; the stairs with the
# classes floor / rug / stairs) is in F_s, the others in F_g. Entering a candidate costs
# J_front = lambda_g (F_g) or lambda_s (F_s) more in the ATSP, so F_g comes softly first.
#
# Spatial value map (Sec. V-B): with a target, before it is found, each frontier goal gets the
# semantic utility u_j = the maximum value of voxel_map/value within value_radius of it (on this
# floor), and entering it costs lambda_v (1 - u_j) more in the ATSP (value_weight; the paper
# only says that low-value frontiers are down-weighted and high-value ones preferred). With
# value_normalize, u_j is first rescaled over the candidates of this plan, (u - min) / (max - min):
# BLIP-2 image-text scores stay within ~0.0-0.4, so lambda_v (1 - u) differed by about 1 m between
# the best and the worst frontier and hardly moved the tour; rescaled, the lowest-value frontier
# costs lambda_v more than the highest one. Below value_min_spread (all about equal) no term.
#
# Floors (chain of exploration, Sec. VI-A, eq. 9 after the target objects): only frontiers on the
# current floor (centroid within floor_band of z_r, the height of trav_grid) are candidates
# (F = frontiers of the current floor). After the frontier ranks, the next rank is the entrance of
# a stair landmark (stair_landmark_topic: instance_map/stair_landmarks, the landmark branch of the
# instance map, or voxel_map/stair_landmarks) that leads up from this floor and has not been
# climbed: A = O_stair once F is exhausted (or all of F is rejected by the explorer, i.e. dormant).
# This node then asks the voxel map to climb it (voxel_map/climb_at: a 3-D path to its foot, then
# the climb) and answers only when that is done, with no goal; the explorer asks rank 0 again
# (its empty-goal retries) on the new floor.

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
from std_msgs.msg import ColorRGBA, Empty, Float64, Int64, String
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker

from trav_exploration.goal_selection import (clear_cells, is_reachable, pull_back, reachable_mask,
                                             select_frontier, snap_to_clear_cell)
from trav_exploration.tour import cost_matrix, geo_costs, nearest_node, solve_open_atsp, step_graph
from trav_exploration.value_map import utilities
from action_msgs.srv import CancelGoal
from trav_interfaces.srv import FrontierTargets, GeoCosts, TargetGoal


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
        # J_front (eq. 14): lambda_g for geometric frontiers F_g, lambda_s for semantic ones F_s
        # (0 <= lambda_g < lambda_s; values not given in the paper): F_g softly first, F_s (on
        # the stairs: traversable only through M_s) as complementary candidates
        self.lambda_g = float(p('lambda_g', 0.0).value)
        self.lambda_s = float(p('lambda_s', 1.0).value)
        self.semantic_share = float(p('semantic_share', 0.5).value)   # cluster share of M_s-only voxels
        self.lambda_esdf = float(p('lambda_esdf', 0.3).value)   # = voxel_mapper_node (2-D fallback only)
        self.d_safe = float(p('d_safe', 0.6).value)
        self.inflation_radius = float(p('inflation_radius', 0.22).value)
        self.exact_max = int(p('tour_exact_max', 12).value)
        self.plan = None              # ATSP state for the current rank sequence
        self.reach_radius = float(p('dormant_reach_radius', 0.4).value)
        self.pursued = None           # (cluster, goal) of the frontier goal last sent
        # goal validation while driving (execute-review): the frontier goal's cluster gone from
        # frontier_vanish_msgs consecutive cluster messages (1 s apart: the hysteresis against a
        # cluster flickering for a frame) -> the Nav2 goal is canceled and the explorer asks again
        self.vanish_msgs = int(p('frontier_vanish_msgs', 2).value)
        self.n_vanished = 0
        self.cancel_cli = self.create_client(CancelGoal, 'navigate_to_pose/_action/cancel_goal',
                                             callback_group=MutuallyExclusiveCallbackGroup())
        # stuck on the way (voxel_mapper_node, eq. 17, v_front already marked): one failed attempt
        # for the frontier goal; unreachable now (J_geo inf on the updated graph) or dormant ->
        # cancel the Nav2 goal at once, so the tour is solved again (not after Nav2's recoveries)
        self.stuck_counted = False
        self.stuck_grace = float(p('stuck_grace', 3.0).value)   # wall s after a goal is sent
        self.pursued_t = 0.0
        self.create_subscription(Empty, 'voxel_map/stuck', self._on_stuck, 10,
                                 callback_group=MutuallyExclusiveCallbackGroup())
        self.dormant_pending = set()  # cluster ids sent to the voxel map, until its flag comes back
        self.dormant_pub = self.create_publisher(Int64, 'voxel_map/frontier_dormant', 10)
        self.dormant_pub_failed = self.create_publisher(Int64, 'voxel_map/frontier_failed', 10)
        self.max_attempts = int(p('frontier_max_attempts', 2).value)   # = voxel_mapper_node
        # last resort before exploration ends with only dormant frontiers left (no stairs either):
        # all dormant flags are cleared (voxel_map/frontier_release), the robot looks around once
        # more and the tour is solved again. At most dormant_release_max times per episode; after
        # the first, only when the robot moved dormant_release_move m or the frontier voxels changed
        # by dormant_release_change since the last release (no new observation: ending is right).
        # A dormant set that the robot never leaves (a cramped start, frontiers that stay after
        # looking down) otherwise ended the episode within a minute.
        self.release_max = int(p('dormant_release_max', 3).value)
        self.release_move = float(p('dormant_release_move', 1.0).value)
        self.release_change = float(p('dormant_release_change', 0.2).value)
        self.n_release = 0
        self.release_at = None        # (robot xy, frontier voxels) at the last release
        self.release_pub = self.create_publisher(Empty, 'voxel_map/frontier_release', 10)
        self.camera_hfov = math.radians(float(p('camera_hfov', 79.0).value))
        self.look_demote = bool(p('look_demote', True).value)
        self.n_plans = 0
        self.d_blind = float(p('d_blind', 1.5).value)
        self.look_pitch = float(p('look_down_pitch', math.radians(15.0)).value)   # 0 = off
        self.look_time = float(p('look_time', 1.5).value)          # s (wall clock) per look_down step
        # closed-loop look (Sec. IV-D): look_down_pitch at a time while frontier voxels of the near
        # clusters are unobserved, up to look_max_pitch; it stops when they are observed
        self.look_max_pitch = float(p('look_max_pitch', math.radians(60.0)).value)
        self.looked = []              # centroids of the blind-spot frontiers already looked at
        self.pitch_pub = self.create_publisher(Float64, 'camera/pitch_cmd', 1)
        self.look_around = bool(p('look_around', True).value)
        self.look_around_speed = float(p('look_around_speed', 0.6).value)    # rad/s
        self.need_look_around = self.look_around
        self.look_around_min = int(p('look_around_min_frontiers', 3).value)   # 0: always turn
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 1)
        # floors and stairs
        self.floor_band = float(p('floor_band', 0.3).value)          # frontier centroid vs z_r
        self.stair_rise = float(p('stair_rise', 1.0).value)          # a climb this high reached a new floor (x 0.5)
        # a stair landmark whose top is this high above z_r leads up: the entrance is what matters
        # (the climb follows the support graph as far as it goes). An instance-map stair cluster
        # often holds only the lower steps (the flight seen from below splits into clusters); with
        # 1 m the stairs of 00800 were never climbed.
        self.stair_lead_rise = float(p('stair_lead_rise', 0.4).value)
        self.climb_timeout = float(p('climb_timeout', 600.0).value)  # s (wall clock)
        self.z_r = None
        self.landmarks = []           # (entrance x, y, z, top x, y, z, size)
        # entrance xy of the stairs given up (failed twice: the repeated-failure rule, Sec. VI-C).
        # Stairs climbed are not left out by force: on the new floor the candidate pool is rebuilt
        # (Sec. VI-A), stairs come in only once its frontiers are exhausted, and a floor explored
        # already is not a destination (_floor_explored)
        self.climbed = []
        self.climb_fails = {}         # entrance xy -> failed climbs
        self.explored_floors = []     # z_r of the floors whose frontiers were exhausted
        self.climb_done = threading.Event()
        self.climb_z = None
        self.climb_on_stairs = False
        self.stair_resume = None      # (landmark, direction, entrance) of a climb that stopped on the stairs
        # target object (O_tar of the instance map)
        self.target = str(p('target', '').value).split('=', 1)[0].strip()   # "name" or "name=prompt"
        self.success_distance = float(p('success_distance', 1.0).value)
        self.targets = {}             # cluster id -> (gamma, points (N, 3))
        self.target_found = False
        self.target_logged = None
        # a target becoming reliable and reachable while the robot drives to a frontier: the Nav2
        # goal is canceled at once, so the next request goes to it (A = O_tar first, eq. 9);
        # before, the robot drove on to the frontier first (00873: 5 s more after bed #90).
        # Checked every target_preempt_period s (0 = off), not while a request is being answered.
        self.target_preempt_period = float(p('target_preempt_period', 1.0).value)
        self.serving = False          # inside _on_request (climbs, turns, look-downs wait in there)
        # one rule for reaching a target and for placing its goal: target_min_points points of the
        # cluster within success_distance of the robot. The goal is a free cell with that many
        # within success_distance - target_goal_tolerance (Nav2's xy_goal_tolerance), so standing
        # anywhere Nav2 calls the goal reached is a reach. Before, the goal was 0.35-0.65 m from
        # the nearest single point and the reach needed 3 within 1 m: at a sparse end of bed #76
        # (00873) Nav2 reached the goal, the target was not reached, and the robot left for #77.
        self.target_min_points = int(p('target_min_points', 3).value)
        self.target_goal_tolerance = float(p('target_goal_tolerance', 0.25).value)
        self.target_skip = {}         # cluster id -> why it is not a goal now (logged on change)
        self.targets_sparse = {}      # cluster id -> points left after the stray filter (< 5)
        # spatial value map (with a target)
        self.value_weight = float(p('value_weight', 3.0).value)    # lambda_v [m of J_geo]
        self.value_radius = float(p('value_radius', 1.0).value)
        self.value_normalize = bool(p('value_normalize', True).value)
        self.value_min_spread = float(p('value_min_spread', 0.05).value)
        self.value_pts = np.zeros((0, 3))
        self.value_vals = np.zeros(0)

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
        self.create_subscription(PointCloud2, p('stair_landmark_topic', 'voxel_map/stair_landmarks').value,
                                 self._on_landmarks, 1, callback_group=inputs)
        if self.target:
            self.create_subscription(PointCloud2, p('target_topic', 'instance_map/targets').value,
                                     self._on_targets, 1, callback_group=inputs)
        self.target_goal_pub = self.create_publisher(Marker, 'frontier_goal/target_goal', 1)
        if self.target and self.value_weight > 0.0:
            self.create_subscription(PointCloud2, p('value_topic', 'voxel_map/value').value,
                                     self._on_value, 1, callback_group=inputs)
        self.create_subscription(String, 'voxel_map/climb_state', self._on_climb_state, 10,
                                 callback_group=inputs)
        self.climb_pub = self.create_publisher(PointStamped, 'voxel_map/climb_at', 1)
        self.descend_pub = self.create_publisher(PointStamped, 'voxel_map/descend_at', 1)
        # the landmark's far end height (top going up, foot going down), sent just before
        self.climb_end_pub = self.create_publisher(Float64, 'voxel_map/climb_end_z', 1)
        # A = O_stair only while F is exhausted (eq. 9, decided at every replan in the paper): on
        # the way to a stair landmark taken because F ran out, reachable frontiers (not dormant, not
        # semantic) appearing on the floor the robot is still on call the climb off and exploration
        # goes on there (00495: a tentative landmark held the robot for minutes while new frontiers
        # appeared). Not when the stairs were taken for a target on another floor, nor once the
        # robot is off its floor (climb_abort_dz).
        self.climb_cancel_pub = self.create_publisher(Empty, 'voxel_map/climb_cancel', 1)
        self.climb_abort = bool(p('climb_abort_on_frontiers', True).value)
        self.climb_abort_dz = float(p('climb_abort_dz', 0.3).value)
        self.climb_abort_delay = float(p('climb_abort_delay', 3.0).value)   # s after the request
        # only frontiers at the height of the floor left (centroid within climb_abort_floor_dz) and
        # away from the stairs taken (>= climb_abort_stair_dist in x-y from entrance -> other end)
        self.climb_abort_floor_dz = float(p('climb_abort_floor_dz', 0.15).value)
        self.climb_abort_stair_dist = float(p('climb_abort_stair_dist', 1.5).value)
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
        # J_geo(x_r, F_i) = min over the cluster voxels, and v_target (voxel_map/frontier_targets)
        self.ft_cli = self.create_client(FrontierTargets, 'voxel_map/frontier_targets',
                                         callback_group=MutuallyExclusiveCallbackGroup())
        self.ftargets = {}            # cluster id -> (J, (x, y, z) of v_target)
        # goals for target objects on the 3-D support graph (voxel_map/target_goal)
        self.tg_cli = self.create_client(TargetGoal, 'voxel_map/target_goal',
                                         callback_group=MutuallyExclusiveCallbackGroup())
        self.create_timer(1.0, self._publish_unreachable, callback_group=inputs)
        if self.target and self.target_preempt_period > 0.0:
            self.create_timer(self.target_preempt_period, self._preempt_for_target,
                              callback_group=MutuallyExclusiveCallbackGroup())
        self.get_logger().info(f'frontier goals from voxel_map/frontier_clusters, order {self.goal_order}, '
                               f'goal_clearance {self.goal_clearance} m')

    def _on_clusters(self, msg):
        # (x, y, z, size, semantic, id, dormant, fails); defaults when the map does not send them
        names = [f.name for f in msg.fields]
        order = ('x', 'y', 'z', 'size', 'semantic', 'id', 'dormant', 'fails')
        default = {'semantic': 0.0, 'id': -1.0, 'dormant': 0.0, 'fails': 0.0}
        fields = [n for n in order if n in names]
        clusters = []
        for c in point_cloud2.read_points(msg, field_names=fields):
            v = dict(zip(fields, map(float, c)))
            clusters.append(tuple(v.get(n, default.get(n, 0.0)) for n in order))
        ids = {int(c[5]) for c in clusters}
        self.dormant_pending &= {int(c[5]) for c in clusters if c[6] < 0.5}   # flag not back yet
        self.dormant_pending &= ids
        self.clusters = clusters
        self._validate_goal()

    def _on_grid(self, msg):
        g = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        self.grid = (g, (msg.info.origin.position.x, msg.info.origin.position.y), msg.info.resolution)
        self.z_r = msg.info.origin.position.z              # the support plane of this slice

    def _on_landmarks(self, msg):
        base = ('x', 'y', 'z', 'top_x', 'top_y', 'top_z', 'size')
        if 'verdict' in [f.name for f in msg.fields]:      # checked by voxel_mapper (step structure)
            self.landmarks = [tuple(map(float, lm)) for lm in point_cloud2.read_points(
                msg, field_names=base + ('reliable', 'belief', 'gamma', 'verdict'))]
        elif 'reliable' in [f.name for f in msg.fields]:   # instance map: tentative ones too
            self.landmarks = [tuple(map(float, lm)) for lm in point_cloud2.read_points(
                msg, field_names=base + ('reliable', 'belief', 'gamma'))]
        else:                                               # voxel landmarks: all reliable
            self.landmarks = [tuple(map(float, lm)) + (1.0, float(lm[6]), 1.0)
                              for lm in point_cloud2.read_points(msg, field_names=base)]

    def _on_value(self, msg):
        a = point_cloud2.read_points_numpy(msg, field_names=('x', 'y', 'z', 'value'))
        self.value_pts, self.value_vals = a[:, :3].astype(np.float64), a[:, 3].astype(np.float64)

    def _is_semantic(self, c):
        return len(c) > 4 and c[4] >= self.semantic_share

    def _node_cost(self, cands, goals):
        """Cost of entering each candidate: J_front (eq. 14) + the value-map term (Sec. V-B)."""
        cost = np.array([self.lambda_s if self._is_semantic(c) else self.lambda_g for c in cands])
        self.n_semantic = int(sum(self._is_semantic(c) for c in cands))
        v = self._value_cost(goals)
        return cost if v is None else cost + v

    def _value_cost(self, goals):
        """lambda_v (1 - u_j) of each goal, or None without a target / value map."""
        self.last_u = None
        if not self.target or self.value_weight <= 0.0 or not len(self.value_vals):
            return None
        z = self.z_r if self.z_r is not None else 0.0
        u = utilities(self.value_pts, self.value_vals, [(g[0], g[1], z) for g in goals], self.value_radius)
        self.last_u = u
        if self.value_normalize:
            spread = float(u.max() - u.min()) if len(u) else 0.0
            if spread < self.value_min_spread:
                return np.zeros(len(u))              # no frontier stands out
            u = (u - u.min()) / spread
        return self.value_weight * (1.0 - u)

    def _on_targets(self, msg):
        from scipy.spatial import cKDTree
        pts = np.array([tuple(p) for p in point_cloud2.read_points(msg, field_names=('x', 'y', 'z', 'id', 'gamma'))],
                       dtype=np.float64).reshape(-1, 5)
        targets, sparse = {}, {}
        for i in np.unique(pts[:, 3]):
            p = pts[pts[:, 3] == i]
            # stray points (a mask edge bleeding onto the background behind the object) are left
            # out: a point needs 4 neighbours within 0.25 m. With them a 35-point bed cluster had
            # points at the top of the stairs, metres from the bed, and the robot "reached" it there
            n = np.array([len(x) for x in cKDTree(p[:, :3]).query_ball_point(p[:, :3], 0.25)]) - 1
            p = p[n >= 4]
            if len(p) >= 5:
                targets[int(i)] = (float(p[0, 4]), p[:, :3])
            else:
                sparse[int(i)] = len(p)
        self.targets = targets
        self.targets_sparse = sparse

    def _target_skipped(self, i, why):
        """A reliable target that is no goal now: log why, once per change (the check runs every
        second while driving)."""
        if self.target_skip.get(i) != why:
            self.target_skip[i] = why
            if why:
                self.get_logger().info(f'[target] {self.target} #{i} reliable but not a goal now: {why}')

    def _target_goal(self, rx, ry):
        """O_tar on this floor: ('reached', id, d) when target_min_points points of a reliable
        target cluster are within success_distance of the robot, (goal (x, y, True), id, J) for
        the free cell nearest to the robot on the support graph from which the same holds (with
        Nav2's goal tolerance), or None."""
        for i, n in self.targets_sparse.items():
            self._target_skipped(i, f'{n} points left after the stray-point filter (< 5)')
        if not self.targets or self.z_r is None:
            return None
        z = self.z_r
        sd, k_min = self.success_distance, max(1, self.target_min_points)
        objs = []
        for i, (_, pts) in self.targets.items():
            if z - 0.3 <= pts[:, 2].min() <= z + 2.0:              # stands on (or hangs above) this floor
                objs.append((i, pts))
            else:
                self._target_skipped(i, f'not on this floor (lowest point z {pts[:, 2].min():.2f}, z_r {z:.2f})')
        if not objs:
            return None
        for i, pts in objs:
            dd = np.hypot(pts[:, 0] - rx, pts[:, 1] - ry)
            if (dd <= sd).sum() >= k_min:                           # a few points, not one stray
                k = int(np.argmin(dd))
                self.get_logger().info(f'[target] #{i}: {(dd <= sd).sum()} of {len(pts)} points '
                                       f'within {sd:.1f} m, nearest ({pts[k, 0]:.2f}, '
                                       f'{pts[k, 1]:.2f}, {pts[k, 2]:.2f})')
                return 'reached', i, float(dd[k])
        r_goal = max(sd - self.target_goal_tolerance, 0.35)
        cands = []                                  # (goal, id, J) per reachable target
        goal_z = {}                                 # id -> height of its goal node
        for i, pts in objs:
            # the goal is a node of the 3-D support graph (voxel_mapper_node): the floor around
            # the object at its own height. On trav_grid (the slice at z_r) the floor 15 cm under
            # z_r was missing, and no chair of 00813 had a free cell near it.
            r = self._target_goal_3d(rx, ry, z, pts, r_goal, k_min)
            if r is None:
                r = self._target_goal_2d(rx, ry, z, i, pts, r_goal, k_min)
                if r is not None:
                    cands.append(r)
                continue
            if r.found:
                self._target_skipped(i, '')
                cands.append(((float(r.goal.x), float(r.goal.y), True), i, float(r.cost)))
                goal_z[i] = float(r.goal.z)
            elif r.n_near == 0:
                self._target_skipped(i, f'no graph node within 0.35-{r_goal:.2f} m of its points')
            elif r.n_ring == 0:
                self._target_skipped(i, f'no graph node with {k_min} of its points within {r_goal:.2f} m '
                                        f'({r.n_near} nodes with fewer)')
            elif r.n_clear == 0:
                self._target_skipped(i, f'none of its {r.n_ring} goal nodes {self.goal_clearance:.2f} m '
                                        f'clear of O_inf')
            else:
                self._target_skipped(i, f'none of its {r.n_clear} goal nodes reachable on the graph')
        if len(cands) > 1:
            # A = O_tar (eq. 9): the first node of the open ATSP over the targets' goals
            pts = [(g[0], g[1], goal_z.get(i, z)) for g, i, _ in cands]
            J = self._geo_costs([(rx, ry, z)] + pts, pts)
            if J is not None:
                return cands[solve_open_atsp(cost_matrix(J))[0] - 1]
        return min(cands, key=lambda c: c[2]) if cands else None

    def _target_goal_3d(self, rx, ry, z, pts, r_goal, k_min):
        """voxel_map/target_goal for one target cluster; None when the service does not answer."""
        if not self.tg_cli.service_is_ready():
            return None
        req = TargetGoal.Request()
        req.source = Point(x=float(rx), y=float(ry), z=float(z))
        req.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in pts]
        req.r_min, req.r_goal, req.k_min = 0.35, float(r_goal), int(k_min)
        fut = self.tg_cli.call_async(req)
        done = threading.Event()
        fut.add_done_callback(lambda _f: done.set())
        if not done.wait(10.0) or fut.result() is None or not fut.result().ok:
            return None
        return fut.result()

    def _target_goal_2d(self, rx, ry, z, i, pts, r_goal, k_min):
        """Fallback without the 3-D graph service: the free trav_grid cell (goal clearance) with
        k_min points within r_goal, cheapest from the robot. (goal, id, J) or None."""
        if self.grid is None:
            return None
        from scipy.spatial import cKDTree
        g, (ox, oy), res = self.grid
        free = g == 0
        if self.goal_clearance > 0.0:
            free &= clear_cells(g, res, self.goal_clearance)
        rows, cols = np.nonzero(free)
        if not len(rows):
            return None
        cells = np.column_stack([ox + (cols + 0.5) * res, oy + (rows + 0.5) * res])
        tree = cKDTree(pts[:, :2])
        d, _ = tree.query(cells, k=1)
        near = cells[(d >= 0.35) & (d <= r_goal)]
        if not len(near):
            self._target_skipped(i, f'no free cell within {r_goal:.2f} m of its points (2-D)')
            return None
        n_in = np.array([len(x) for x in tree.query_ball_point(near, r_goal)])
        ring = near[n_in >= k_min]
        if not len(ring):
            self._target_skipped(i, f'no free cell with {k_min} of its points within {r_goal:.2f} m (2-D)')
            return None
        ring = ring[::max(1, len(ring) // 150)]
        J = self._geo_costs([(rx, ry, z)], [(x, y, z) for x, y in ring])
        J = J[0] if J is not None else np.hypot(ring[:, 0] - rx, ring[:, 1] - ry)
        k = int(np.argmin(J))
        if not np.isfinite(J[k]):
            self._target_skipped(i, f'none of its {len(ring)} goal cells reachable (2-D)')
            return None
        self._target_skipped(i, '')
        return (float(ring[k, 0]), float(ring[k, 1]), True), i, float(J[k])

    def _on_climb_state(self, msg):
        if msg.data.startswith('done'):
            parts = msg.data.split()
            self.climb_z = float(parts[1])
            self.climb_on_stairs = 'stairs' in parts[2:]
            self.climb_turned = 'turned' in parts[2:]
            self.climb_cancelled = 'cancelled' in parts[2:]
            self.climb_done.set()

    def _floor_clusters(self):
        """Frontier clusters of the current floor (centroid within floor_band of z_r)."""
        if self.z_r is None:
            return list(self.clusters)
        return [c for c in self.clusters if abs(c[2] - self.z_r) <= self.floor_band]

    def _stair_candidates(self, reliable=True, direction=None):
        """Stair landmarks with one end on this floor, not used yet: [(landmark, direction,
        entrance on this floor)]. Up (+1): the foot is on this floor and the top at least
        stair_lead_rise higher; down (-1): it reaches up to within 1 m of this floor and its foot is
        at least stair_lead_rise lower.
        The reliable ones, or (reliable=False) the tentative ones; direction None: both."""
        if self.z_r is None:
            return []
        out = []
        for lm in self.landmarks:
            if (lm[7] >= 0.5) != reliable:
                continue
            if len(lm) > 10 and lm[10] < 0:
                continue                   # no step structure there (voxel_mapper stair check)
            if abs(lm[2] - self.z_r) <= 0.4 and lm[5] > self.z_r + self.stair_lead_rise:
                d, e = 1, (lm[0], lm[1], lm[2])
            elif lm[5] >= self.z_r - 1.0 and lm[2] < self.z_r - self.stair_lead_rise:
                # the top may be the railing (a stair cluster holds every stair-labelled pixel),
                # or the flight split into clusters where steps were not seen (00800 from
                # upstairs: -0.06..-0.26 m and -0.61..-1.61 m): reaching up to within 1 m of this
                # floor is enough; the descent first drives to the stair head on this floor
                d, e = -1, (lm[3], lm[4], self.z_r)
            else:
                continue
            if direction is not None and d != direction:
                continue
            if any(math.hypot(e[0] - c[0], e[1] - c[1]) < 1.0 for c in self.climbed):
                continue
            out.append((lm, d, e))
        return out

    def _floor_explored(self, direction):
        """The next floor that way was explored (its frontiers exhausted) already."""
        return any(1.0 < direction * (z - self.z_r) < 4.5 for z in self.explored_floors)

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

    def _frontier_targets(self, rx, ry, z):
        """{cluster id: (J_geo, v_target)} from the robot, or None (service not there)."""
        if not self.ft_cli.service_is_ready():
            return None
        req = FrontierTargets.Request()
        req.source = Point(x=float(rx), y=float(ry), z=float(z))
        fut = self.ft_cli.call_async(req)
        done = threading.Event()
        fut.add_done_callback(lambda _f: done.set())
        if not done.wait(10.0) or fut.result() is None or not fut.result().ok:
            return None
        r = fut.result()
        return {int(i): (float(j), (t.x, t.y, t.z)) for i, j, t in zip(r.ids, r.costs, r.targets)}

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
        # one snapshot of trav_grid for clear and the placement: the subscription replaces
        # self.grid on another thread, and a grown grid with the old clear array went out of
        # bounds (IndexError in snap_to_clear_cell, frontier_goal_node died)
        grid = self.grid
        clear = clear_cells(grid[0], grid[2], self.goal_clearance) if self.goal_clearance > 0.0 else None
        # the paper's J_geo(x_r, F_i): min over the cluster's own voxels (graph nodes); its
        # v_target is where the goal is placed. Clusters newer than the graph snapshot: the old
        # test on the placed goal below.
        ft = self._frontier_targets(rx, ry, z)
        self.ftargets = ft or {}
        goals = [self._place(c, rx, ry, clear, grid=grid) for c in clusters]
        # goals lie on trav_grid, the floor at z_r (floor_z): the graph holds every floor, a goal
        # at the centroid's height (up to floor_band off) missed its floor node
        rest = [i for i, c in enumerate(clusters) if int(c[5]) not in self.ftargets]
        J = self._geo_costs([(rx, ry, z)], [(goals[i][0], goals[i][1], z) for i in rest]) if rest else \
            np.zeros((1, 0))
        if J is not None or not rest:
            jj = dict(zip(rest, J[0])) if rest else {}
            fin = [self.ftargets[int(c[5])][0] if i not in jj else jj[i] for i, c in enumerate(clusters)]
            ok = [c for c, j in zip(clusters, fin) if np.isfinite(j)]
            bad = [c for c, j in zip(clusters, fin) if not np.isfinite(j)]
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

    def _is_dormant(self, c, goal=None):
        return c[6] >= 0.5 or int(c[5]) in self.dormant_pending

    def _on_stuck(self, _msg):
        """Stuck on the way to the frontier goal (v_front already marked): one failed attempt, and
        the Nav2 goal is canceled at once, the tour solved again on the updated map. Left to Nav2,
        its recoveries (back up, spin: actions spent) tried the same goal again where the navmesh
        had a gap our map did not show; a second stuck there makes the frontier dormant. A stuck
        event within stuck_grace s of sending the goal is the robot still caught where the last
        goal left it, not a failure of this one."""
        pv = self.pursued
        if pv is None or time.time() - self.pursued_t < self.stuck_grace:
            return
        pc, pg = pv
        c = self._match(pc, self._floor_clusters())
        if c is None:
            return
        rx, ry = self._robot_xy()
        if not self.stuck_counted:
            self.stuck_counted = True
            self._fail(c, f'stuck on the way at ({rx:.2f}, {ry:.2f})')
        self.pursued = None                           # counted already; not again at the review
        self.get_logger().info(
            f'[replan] stuck on the way to the frontier at ({c[0]:.2f}, {c[1]:.2f}): navigation canceled, '
            f'the tour is solved again on the updated map')
        if self.cancel_cli.service_is_ready():
            self.cancel_cli.call_async(CancelGoal.Request())

    def _validate_goal(self):
        """The frontier the robot drives to is gone (seen from on the way, or no longer
        traversable): cancel the Nav2 goal, so the explorer asks for a goal again and the tour is
        solved anew over F_act. Before, the robot drove on to where nothing was left to see."""
        pv = self.pursued
        if pv is None or self.vanish_msgs <= 0:
            self.n_vanished = 0
            return
        pc, pg = pv
        if self._match(pc, self.clusters) is not None:
            self.n_vanished = 0
            return
        self.n_vanished += 1
        if self.n_vanished < self.vanish_msgs or not self.cancel_cli.service_is_ready():
            return
        self.n_vanished = 0
        self.pursued = None                         # cleared, not failed
        rx, ry = self._robot_xy()
        self.get_logger().info(
            f'[replan] frontier at ({pc[0]:.2f}, {pc[1]:.2f}) gone on the way ({math.hypot(pg[0] - rx, pg[1] - ry):.1f} m '
            f'before its goal): navigation canceled, the tour is solved again')
        self.cancel_cli.call_async(CancelGoal.Request())   # zero goal id and stamp: cancel all

    def _match(self, c, pool):
        """Cluster c in pool: the same id, or the nearest centroid within 0.5 m on its floor (the
        map re-segments clusters and gives them new ids)."""
        same = [a for a in pool if c[5] >= 0 and int(a[5]) == int(c[5])]
        if same:
            return same[0]
        near = [(math.hypot(a[0] - c[0], a[1] - c[1]), i) for i, a in enumerate(pool)
                if abs(a[2] - c[2]) < 0.5]
        near = [x for x in near if x[0] < 0.5]
        return pool[min(near)[1]] if near else None

    def _review_pursued(self, rx, ry, why_not):
        """Execute-review of the frontier goal last sent (Sec. IV-D), at the next request: the
        robot stands at its goal and the frontier is still there -> dormant (irreducible); the
        goal was not reached (Nav2 gave up, the guard held the robot, the explorer found no Nav2
        path or a failed goal there) -> one failed attempt, dormant at frontier_max_attempts
        (repeated failures, F_act -> F_dor). Other clusters near the robot are left alone.
        A cluster gone meanwhile was cleared."""
        pc, pg = self.pursued
        self.pursued = None
        c = self._match(pc, self._floor_clusters())
        if c is None or self._is_dormant(c):
            return
        r = self.reach_radius if pg[2] else self.reach_radius + 0.5
        if math.hypot(pg[0] - rx, pg[1] - ry) < r:
            self._demote(c, 'still there with the robot at its goal')
        elif not self.stuck_counted:                  # a stuck event counted this attempt already
            self._fail(c, why_not)

    def _fail(self, c, why):
        if c[5] < 0:
            return
        self.dormant_pub_failed.publish(Int64(data=int(c[5])))
        n = int(c[7]) + 1
        if n >= self.max_attempts:
            self.dormant_pending.add(int(c[5]))
        self.get_logger().info(
            f'[dormant] frontier at ({c[0]:.2f}, {c[1]:.2f}) {why}: attempt {n} of {self.max_attempts}'
            + ('; left out until its voxels change' if n >= self.max_attempts else ''))

    def _demote(self, c, why):
        """Cluster c goes to the dormant set (its visited flag in the voxel map)."""
        if self._is_dormant(c) or c[5] < 0:
            return False
        self.dormant_pending.add(int(c[5]))
        self.dormant_pub.publish(Int64(data=int(c[5])))
        n = sum(1 for k in self.clusters if k[6] >= 0.5) + len(self.dormant_pending)
        self.get_logger().info(
            f'[dormant] frontier at ({c[0]:.2f}, {c[1]:.2f}) {why}: left out until its voxels '
            f'change ({n} dormant)')
        return True

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
        # blind spot = below the lower edge of the forward-looking camera's view: only clusters
        # inside its horizontal FOV are in it; one beside or behind the robot is out of view, and
        # pitching does not bring it in (00034 / 00217: 15 of 18 looks were at such clusters and
        # went down to the lowest pitch every time). It is judged again when the robot faces it.
        yaw = self._robot_yaw()
        half = 0.5 * self.camera_hfov - math.radians(5.0)   # margin: the edges see little
        n_all = len(near)
        near = [c for c in near if yaw is not None and
                abs(math.remainder(math.atan2(c[1] - ry, c[0] - rx) - yaw, math.tau)) < half]
        if not near:
            self.get_logger().info(f'[blind] {n_all} frontier(s) within {self.d_blind:.1f} m, none in the '
                                   f'camera view: no look down', throttle_duration_sec=10.0)
            return reachable
        in_view = near
        def match(c, after):                # the same cluster after the look (re-segmented: nearest)
            same = [a for a in after if c[5] >= 0 and int(a[5]) == int(c[5])]
            if same:
                return same[0]
            close = [(math.hypot(c[0] - a[0], c[1] - a[1]), i) for i, a in enumerate(after)
                     if abs(c[2] - a[2]) < 0.5]
            close = [x for x in close if x[0] < 0.5]
            return after[min(close)[1]] if close else None

        def unresolved(after):              # frontier voxels of the near clusters still there
            return sum(int(a[3]) for a in (match(c, after) for c in near) if a is not None)

        # closed loop on what is to be seen (Sec. IV-D): one look_down step at a time while frontier
        # voxels of the near clusters are left unobserved, up to look_max_pitch; not a set angle.
        # (Before, it stopped when a step resolved too little: 30 deg that saw nothing new ended
        # the look although 60 deg might have seen the floor.)
        n_ref = sum(int(c[3]) for c in near)
        self.get_logger().info(
            f'[blind] {len(near)} frontier(s) within {self.d_blind:.1f} m in the camera view ({n_all - len(near)} '
            f'out of view left), {n_ref} voxels: looking down in {math.degrees(self.look_pitch):.0f} deg steps')
        k_max = max(1, int(self.look_max_pitch / self.look_pitch + 1e-6))
        k = 0
        while k < k_max:
            k += 1
            self.pitch_pub.publish(Float64(data=k * self.look_pitch))
            time.sleep(self.look_time)
            self.reach = None
            after, _ = self._split(rx, ry)
            n = unresolved(after)
            more = n > 0 and k < k_max
            self.get_logger().info(
                f'[blind] {math.degrees(k * self.look_pitch):.0f} deg down: {n} of the {n_ref} voxels still '
                f'unobserved: ' + ('observed' if n == 0 else 'one step further' if more else 'at the lowest pitch'))
            if not more:
                break
        self.pitch_pub.publish(Float64(data=0.0))
        time.sleep(0.5)
        self.looked.extend((c[0], c[1], c[2]) for c in near)
        self.reach = None
        after, _ = self._split(rx, ry)
        left = [c for c in near if match(c, after) is not None]
        n_dem = 0
        if self.look_demote:
            # in view, looked at from close by and still there: the space behind it cannot be seen
            # from here (a gap under furniture, a narrow slot): irreducible (Sec. IV-D)
            for c in in_view:
                a = match(c, after)
                if a is not None and self._demote(a, 'still there after looking down at it'):
                    n_dem += 1
        self.get_logger().info(f'[blind] after looking down: {len(near) - len(left)} of {len(near)} cleared'
                               + (f', {n_dem} irreducible (dormant)' if n_dem else ''))
        if n_dem:
            after = [a for a in after if not self._is_dormant(a)]
        return after

    def _release_dormant(self, rx, ry):
        """Clear all dormant flags when only dormant frontiers are left; True when released."""
        if self.n_release >= self.release_max:
            return False
        dormant = [c for c in self._floor_clusters() if self._is_dormant(c)]
        if not dormant:
            return False
        vox = sum(c[3] for c in self._floor_clusters())
        if self.release_at is not None:
            (x0, y0), v0 = self.release_at
            moved = math.hypot(rx - x0, ry - y0)
            change = abs(vox - v0) / max(v0, 1.0)
            if moved < self.release_move and change < self.release_change:
                self.get_logger().info(
                    f'[dormant] only dormant frontiers left; not released again: moved {moved:.2f} m, '
                    f'frontier voxels changed {100 * change:.0f} % since the last release')
                return False
        self.n_release += 1
        self.release_at = ((rx, ry), vox)
        self.get_logger().warn(
            f'[dormant] only dormant frontiers left ({len(dormant)}), no stairs: releasing them '
            f'({self.n_release} of {self.release_max}) and looking around once more')
        self.dormant_pending.clear()
        self.release_pub.publish(Empty())
        t0 = time.time()
        while time.time() - t0 < 3.0 and any(c[6] >= 0.5 for c in self.clusters):
            time.sleep(0.2)
        self._turn_around()
        self.reach = None
        return True

    def _publish_dormant(self):
        m = Marker()
        m.header.frame_id = self.map_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id, m.type = 'dormant', 1, Marker.SPHERE_LIST
        dormant = [c for c in self.clusters if self._is_dormant(c)]
        m.action = Marker.ADD if dormant else Marker.DELETE
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.25
        m.color = ColorRGBA(r=0.2, g=0.4, b=1.0, a=0.9)
        for c in dormant:
            q = Point()
            q.x, q.y, q.z = c[0], c[1], c[2]
            m.points.append(q)
        self.unreachable_pub.publish(m)

    def _publish_unreachable(self):
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

    def _place(self, c, rx, ry, clear=None, grid=None):
        """Goal position for cluster c (2D placement: snapped to goal_clearance, or pulled back).
        grid: the trav_grid snapshot `clear` was computed from (default: the current one)."""
        gx, gy = c[0], c[1]
        t = self.ftargets.get(int(c[5])) if len(c) > 5 else None
        if t is not None and np.isfinite(t[0]):
            # v_target: the cheapest voxel of the cluster, already moved by voxel_mapper_node to a
            # node with the goal clearance (on the 3-D graph at its own height; snapped on
            # trav_grid, the slice at z_r, it missed a floor 15 cm under z_r)
            return t[1][0], t[1][1], True
        if self.goal_clearance > 0.0:
            snapped = None
            if grid is None:
                grid = self.grid
            if grid is not None and (clear is None or clear.shape == grid[0].shape):
                snapped = snap_to_clear_cell(*grid, gx, gy, self.goal_clearance, clear=clear)
            if snapped is None:
                return gx, gy, False
            return snapped[0], snapped[1], True
        gx, gy = pull_back(gx, gy, rx, ry)
        return gx, gy, True

    def _goal_z(self, c, z):
        """Height of a frontier goal for J_geo: its v_target node on the 3-D graph, else z_r (a goal
        placed on trav_grid, at floor_z)."""
        t = self.ftargets.get(int(c[5])) if len(c) > 5 else None
        return t[1][2] if t is not None and np.isfinite(t[0]) else z

    def _new_plan(self, rx, ry, reachable):
        """Goals and the ATSP cost matrix over {x_r} U F_g (tour.py)."""
        plan = {'goals': [], 'C': None, 'left': [], 'heads': []}
        if not reachable or self.grid is None:
            return plan
        grid = self.grid
        g, origin, res = grid
        clear = clear_cells(g, res, self.goal_clearance) if self.goal_clearance > 0.0 else None
        placed = [(c, self._place(c, rx, ry, clear, grid=grid)) for c in reachable]
        # dormant set: the robot stands at the goal of a frontier that is still there (the
        # explorer asks for rank 0 after each navigation, 1 s after it ended)
        # a goal left where no clear cell was found (e.g. a semantic frontier at the edge of the
        # stairs) is not a node: the path ends on the nearest one, up to plan_goal_radius (0.5 m)
        # away, and the robot never came within reach_radius (00800: the same F_s at the top of
        # the stairs was the tour head 8 times): reach_radius + 0.5 then
        if self.n_plans > 0 and self.pursued is not None:
            self._review_pursued(rx, ry, 'not reached (navigation failed or stopped)')
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
        pts = [(gl[0], gl[1], self._goal_z(c, z)) for c, gl in zip(cands, goals)]
        J = self._geo_costs([(rx, ry, z)] + pts, pts)
        if J is not None:
            plan.update(goals=goals, C=cost_matrix(J, 0.0, self._node_cost(cands, goals)),
                        left=list(range(1, len(goals) + 1)), cands=cands,
                        sem={i + 1 for i, c in enumerate(cands) if self._is_semantic(c)})
            return plan
        self._warn_2d()
        graph, index, _ = step_graph(g, res, self.lambda_esdf, self.d_safe, self.inflation_radius)
        robot = nearest_node(index, rx, ry, origin, res, 0.5)
        nodes = []
        for (gx, gy, _), c in zip(goals, cands):
            v = nearest_node(index, gx, gy, origin, res, 0.4)
            nodes.append(v if v >= 0 else nearest_node(index, c[0], c[1], origin, res, 0.4))
        J = geo_costs(graph, [robot] + nodes, nodes)
        plan.update(goals=goals, C=cost_matrix(J, 0.0, self._node_cost(cands, goals)),
                    left=list(range(1, len(goals) + 1)), cands=cands,
                    sem={i + 1 for i, c in enumerate(cands) if self._is_semantic(c)})
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
                u = getattr(self, 'last_u', None) if self.target and self.value_weight > 0.0 else None
                self.get_logger().info(
                    f'tour over {len(order)} frontiers: first J = {C[0, order[0]]:.2f}, '
                    f'tour J = {sum(c for c in steps if np.isfinite(c)):.2f}'
                    + (f', {no_path} without a path from the robot (M_inf, last)' if no_path else '')
                    + (f', {self.n_semantic} semantic (F_s, first is '
                       f'{"F_s" if order[0] in self.plan.get("sem", ()) else "F_g"})'
                       if getattr(self, 'n_semantic', 0) else '')
                    + (f', value u: first {u[order[0] - 1]:.2f}, max {u.max():.2f}'
                       if u is not None and len(u) == len(self.plan['goals']) else ''))
                self._publish_tour(order)
        while len(self.plan['heads']) <= rank and self.plan['left']:
            self._solve_next()
        if rank >= len(self.plan['heads']):
            return None
        i = self.plan['heads'][rank] - 1
        self.plan['sent'] = (self.plan['cands'][i], self.plan['goals'][i])
        return self.plan['goals'][i]

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

    def _publish_target_goal(self, x, y):
        m = Marker()
        m.header.frame_id = self.map_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = 'target_goal', 0, Marker.CYLINDER, Marker.ADD
        m.pose.position.x, m.pose.position.y = float(x), float(y)
        m.pose.position.z = float(self.z_r if self.z_r is not None else 0.0) + 0.05
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = 0.3
        m.scale.z = 0.1
        m.color = ColorRGBA(r=1.0, g=0.2, b=0.6, a=0.9)
        self.target_goal_pub.publish(m)

    def _stair_goal(self, rx, ry, want_dir=None):
        """A = O_stair (eq. 9): F is exhausted (or a target was seen on another floor, want_dir):
        go up or down the stairs to a floor not explored yet (or toward the target). Among the
        reliable stair landmarks the first node of the open ATSP over their entrances on this
        floor (J_geo on the support graph) is taken; without one, the tentative landmark with the
        largest belief gamma n is verified (Sec. V-A: "a tentative hypothesis for later
        verification"; the climb is the verification). The voxel map drives the robot to the
        entrance and up (voxel_map/climb_at) or down (voxel_map/descend_at); this waits for it and
        returns None (no goal: the explorer asks again on the new floor). Sending the entrance as
        a Nav2 goal instead failed: the 2-D explorer then blacklisted reached goals, and
        meanwhile new frontiers took the robot away again."""
        def allowed(c):
            return c[1] == want_dir if want_dir is not None else not self._floor_explored(c[1])
        cands = [c for c in self._stair_candidates() if allowed(c)]
        resume = getattr(self, 'stair_resume', None)
        if resume is not None and want_dir is None:
            # the last climb stopped on these stairs: go on the same way (from the bottom step
            # the stairs no longer lead "down" by the candidate rule, and exploration ended)
            # from where the robot stands: sent to the stair head again, it drove back up first
            cands = [(resume[0], resume[1], (rx, ry, self.z_r))]
            self.get_logger().info(f'[stairs] going on {"up" if resume[1] > 0 else "down"} the stairs '
                                   f'where the last climb stopped')
        if cands:
            k = 0
            if len(cands) > 1:
                z = self.z_r
                pts = [(e[0], e[1], z) for _, _, e in cands]
                J = self._geo_costs([(rx, ry, z)] + pts, pts)
                if J is not None:
                    k = solve_open_atsp(cost_matrix(J))[0] - 1
                else:
                    k = int(np.argmin([math.hypot(e[0] - rx, e[1] - ry) for _, _, e in cands]))
            lm, d, e = cands[k]
        else:
            # no reliable stair cluster: verify the tentative one with the largest belief gamma n
            tent = [c for c in self._stair_candidates(reliable=False) if allowed(c)]
            if not tent:
                return None
            lm, d, e = max(tent, key=lambda c: c[0][8])
            self.get_logger().info(f'[stairs] no reliable stair landmark: verifying the tentative one at '
                                   f'({e[0]:.2f}, {e[1]:.2f}) (confidence {lm[9]:.2f}, belief {lm[8]:.0f})')
        z0 = self.z_r
        self.climb_done.clear()
        way = 'up' if d > 0 else 'down'
        reason = ('a target on the floor ' + ('above' if d > 0 else 'below') if want_dir is not None
                  else f'no frontier left on this floor (z_r {z0:.2f})')
        self.get_logger().info(f'[stairs] {reason}: going {way} the stairs at ({e[0]:.2f}, {e[1]:.2f}), '
                               f'{"top" if d > 0 else "foot"} {lm[5] if d > 0 else lm[2]:.2f} m'
                               + (f' ({len(cands)} candidates, ATSP)' if len(cands) > 1 else ''))
        msg = PointStamped()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x, msg.point.y, msg.point.z = e
        self.climb_end_pub.publish(Float64(data=float(lm[5] if d > 0 else lm[2])))
        # the frontiers of this floor when it was found exhausted (unreachable or dormant): only
        # frontiers that appear after that call the climb off (mode s, taken for the climb, may
        # reach some of these through the stair voxels)
        self.climb_old_fronts = [(c[0], c[1], c[2]) for c in self._floor_clusters()]
        # the stairs taken (entrance -> other end, x-y): frontiers next to them are the edge of the
        # stair well seen on the way (00495: looking down from the stair head made three, and the
        # descent was called off for them)
        self.climb_seg = ((e[0], e[1]), (lm[3], lm[4]) if d > 0 else (lm[0], lm[1]))
        (self.climb_pub if d > 0 else self.descend_pub).publish(msg)
        done, asked, t_req = False, False, time.time()
        while time.time() - t_req < self.climb_timeout:
            if self.climb_done.wait(1.0):
                done = True
                break
            if asked or time.time() - t_req < self.climb_abort_delay:
                continue
            # a reliable target on the level the robot is on now (reached, or with a goal cell it
            # can reach): targets first (eq. 9), the climb is called off (00802: the toilet was
            # confirmed 0.7 m away during a climb and the robot went on climbing and turning)
            tgt = self._target_goal(*self._robot_xy()) if self.target else None
            if tgt is not None:
                self.get_logger().info(f'[stairs] {self.target} #{tgt[1]} ' + ('reached' if tgt[0] == 'reached' else
                                       'reachable') + ' during the climb: calling the climb off (targets first, eq. 9)')
                self.climb_cancel_pub.publish(Empty())
                asked = True
                continue
            if not self.climb_abort or want_dir is not None:
                continue
            n = self._frontiers_back(z0)
            if n:
                self.get_logger().info(f'[stairs] {n} reachable frontier cluster(s) on this floor (z_r {z0:.2f}) on '
                                       f'the way to the stairs: calling the climb off (F before O_stair, eq. 9)')
                self.climb_cancel_pub.publish(Empty())
                asked = True
        if done and getattr(self, 'climb_cancelled', False):
            # called off: not a failure of the stairs, and this floor is not exhausted after all
            self.explored_floors = [z for z in self.explored_floors if abs(z - z0) >= 0.5]
            self.stair_resume = None
            self.reach = None
            self.plan = None
            return None
        z1 = self.climb_z if done else None
        if done and self.climb_on_stairs:
            self.get_logger().warn(f'[stairs] stopped on the stairs at z_r {z1:.2f}: not a new floor')
        self.stair_resume = None
        if done and z1 is not None and not self.climb_on_stairs and d * (z1 - z0) > 0.5 * self.stair_rise:
            self.get_logger().info(f'[stairs] went {way}: z_r {z0:.2f} -> {z1:.2f}; exploring the new floor')
            # the climb ended with its own turn in place where the robot now stands: the space
            # around was seen once, a second turn (the look-around) only repeats it
            self.need_look_around = self.look_around and not getattr(self, 'climb_turned', False)
            if getattr(self, 'climb_turned', False):
                self.get_logger().info('[look-around] skipped on the new floor: the climb ended with a turn in place')
        else:
            # left out after the second failed try: the real stairs of 00810 failed once (the
            # robot stood in O_inf next to them) and the exploration went on to a wall taken for stairs
            key = next((k for k in self.climb_fails if math.hypot(k[0] - e[0], k[1] - e[1]) < 1.0), (e[0], e[1]))
            self.climb_fails[key] = self.climb_fails.get(key, 0) + 1
            out = self.climb_fails[key] >= 2
            if out:
                self.climbed.append((e[0], e[1]))
            elif done and self.climb_on_stairs:
                self.stair_resume = (lm, d, e)
            self.get_logger().warn(f'[stairs] going {way} did not reach a new floor '
                                   f'({"timed out" if not done else z1}); '
                                   + ('these stairs are left out' if out else 'trying them once more later'))
        self.reach = None
        self.plan = None
        return None

    def _frontiers_back(self, z0):
        """During a climb: the number of reachable, not dormant, geometric frontier clusters on the
        floor the climb left (centroid within climb_abort_floor_dz of its height), away from the
        stairs taken (climb_abort_stair_dist), and new since the climb was decided (no cluster of
        then within 0.5 m), while the robot is still on that floor (z_r within climb_abort_dz)."""
        if self.z_r is None or abs(self.z_r - z0) > self.climb_abort_dz:
            return 0
        rx, ry = self._robot_xy()
        self.reach = None
        reachable, _ = self._split(rx, ry)
        old = getattr(self, 'climb_old_fronts', [])
        seg = getattr(self, 'climb_seg', None)

        def off_stairs(c):                 # x-y distance to the stairs taken (entrance -> other end)
            if seg is None:
                return True
            (ax, ay), (bx, by) = seg
            dx, dy = bx - ax, by - ay
            L2 = dx * dx + dy * dy
            t = min(1.0, max(0.0, ((c[0] - ax) * dx + (c[1] - ay) * dy) / L2)) if L2 > 1e-9 else 0.0
            return math.hypot(c[0] - (ax + t * dx), c[1] - (ay + t * dy)) >= self.climb_abort_stair_dist

        return sum(1 for c in reachable if not self._is_semantic(c) and not self._is_dormant(c, (c[0], c[1]))
                   and abs(c[2] - z0) <= self.climb_abort_floor_dz and off_stairs(c)
                   and not any(abs(c[2] - o[2]) < 0.5 and math.hypot(c[0] - o[0], c[1] - o[1]) < 0.5 for o in old))

    def _targets_elsewhere(self):
        """Reliable target clusters on another floor: [(id, direction)], above (+1) when their
        lowest point is more than 2 m above z_r, below (-1) when their highest is under the floor."""
        if not self.targets or self.z_r is None:
            return []
        out = []
        for i, (_, pts) in self.targets.items():
            if pts[:, 2].min() > self.z_r + 2.0:
                out.append((i, 1))
            elif pts[:, 2].max() < self.z_r - 0.3:
                out.append((i, -1))
        return out

    def _preempt_for_target(self):
        """Driving to a frontier while a reliable target on this floor is reachable (or reached):
        cancel the Nav2 goal, so the explorer asks again and goes to the target."""
        if self.serving or self.target_found or self.pursued is None or not self.targets:
            return
        if not self.cancel_cli.service_is_ready():
            return
        rx, ry = self._robot_xy()
        tgt = self._target_goal(rx, ry)
        if tgt is None or self.serving or self.pursued is None:
            return
        self.pursued = None                           # cleared, not failed
        what = f'#{tgt[1]} reached' if tgt[0] == 'reached' else f'#{tgt[1]} reachable (J_geo {tgt[2]:.2f})'
        self.get_logger().info(f'[target] {self.target} {what} while driving to a frontier: navigation '
                               f'canceled, going to the target')
        self.cancel_cli.call_async(CancelGoal.Request())   # zero goal id and stamp: cancel all

    def _on_request(self, request, response):
        self.serving = True
        try:
            return self._answer(request, response)
        finally:
            self.serving = False

    def _answer(self, request, response):
        self.get_logger().info(f'Received request, {request.goal_rank}')
        rx, ry = self._robot_xy()
        reachable, unreachable = self._split(rx, ry)
        if request.goal_rank == 0 and unreachable:
            self.get_logger().info(f'{len(unreachable)} of {len(self._floor_clusters())} frontier clusters '
                                   f'of this floor not reachable on the floor graph; left out')

        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.orientation.w = 1.0
        if self.target_found:
            pose.header.frame_id = ''                  # target reached: exploration over
            response.goal_pose = pose
            return response
        tgt = self._target_goal(rx, ry) if self.target else None
        if tgt is not None and tgt[0] == 'reached':
            self.target_found = True
            self.get_logger().info(f'[target] reached {self.target} #{tgt[1]}: {tgt[2]:.2f} m away '
                                   f'(success distance {self.success_distance:.1f} m); exploration ends')
            pose.header.frame_id = ''
            response.goal_pose = pose
            return response
        if tgt is None and self.target and request.goal_rank == 0:
            other = self._targets_elsewhere()
            if other:
                tid, d = other[0]
                if self._stair_candidates(direction=d) or self._stair_candidates(reliable=False, direction=d):
                    self.get_logger().info(f'[target] {self.target} #{tid} reliable on the floor '
                                           f'{"above" if d > 0 else "below"}: taking the stairs first (eq. 9)')
                    self._stair_goal(rx, ry, want_dir=d)
                    pose.header.frame_id = ''
                    response.goal_pose = pose
                    return response
        offset = 0
        if tgt is not None and self.goal_order == 'atsp':
            offset = 1                                 # A = O_tar first (eq. 9), the frontiers after
            if request.goal_rank == 0:
                (gx, gy, _), tid, J = tgt
                if self.target_logged != tid:
                    self.target_logged = tid
                    self.get_logger().info(f'[target] {self.target} #{tid} reliable on this floor: going to '
                                           f'({gx:.2f}, {gy:.2f}), J_geo {J:.2f}')
                self._tour_goal(0, rx, ry, reachable)  # the frontier ranks behind it
                self.plan.pop('sent', None)
                self.pursued = None                    # the target goal is not a frontier
                self._publish_target_goal(gx, gy)
                pose.header.frame_id = self.map_frame
                pose.pose.position.x, pose.pose.position.y = gx, gy
                response.goal_pose = pose
                self.get_logger().info(f'Sending goal x: {gx:f} y: {gy:f} (target).')
                return response
        if self.goal_order == 'atsp':
            if request.goal_rank == 0 and self.need_look_around:
                live = [c for c in reachable if not self._is_dormant(c, (c[0], c[1]))]
                if self.look_around_min <= 0 or len(live) < self.look_around_min:
                    self.get_logger().info(f'[look-around] {len(live)} reachable frontier cluster(s) on this '
                                           f'floor (fewer than {self.look_around_min}): looking around first')
                    self._turn_around()
                    rx, ry = self._robot_xy()
                    reachable, unreachable = self._split(rx, ry)
                else:
                    self.get_logger().info(f'[look-around] skipped: {len(live)} reachable frontier clusters '
                                           f'on this floor (>= {self.look_around_min})', throttle_duration_sec=30.0)
            if request.goal_rank == 0:
                reachable = self._look_down(rx, ry, reachable)
            rank = request.goal_rank - offset
            if request.goal_rank > 0 and self.pursued is not None:
                # the explorer turned the last goal down: within 0.2 m of the robot (at it) or no
                # path to it
                self._review_pursued(rx, ry, 'turned down by the explorer (at the robot / no path)')
            goal = self._tour_goal(rank, rx, ry, reachable)
            self.pursued = self.plan.pop('sent', None) if goal is not None else None
            self.stuck_counted = False
            self.pursued_t = time.time()
            if goal is None and rank == len(self.plan['heads']) and self.z_r is not None:
                # (z_r unknown: no voxel-map support yet, nothing to say about this floor; a None
                # in explored_floors crashed the node in 00832 at 31 s)
                if not any(abs(z - self.z_r) < 0.5 for z in self.explored_floors):
                    self.explored_floors.append(self.z_r)    # its frontiers are exhausted
                goal = self._stair_goal(rx, ry)          # F exhausted: A = O_stair (eq. 9)
            # (rank 0 only: an empty answer to a later rank ends the explorer's loop)
            if goal is None and request.goal_rank == 0 and self.target \
                    and self._target_goal(*self._robot_xy()) is not None:
                # a target confirmed during the climb or found reachable now: no dormant release and
                # look-around first; the explorer asks again at once and the target branch takes it
                self.get_logger().info(f'[target] {self.target} confirmed or reachable now: asking again before '
                                       f'releasing dormant frontiers')
                pose.header.frame_id = ''
                response.goal_pose = pose
                return response
            if goal is None and self._release_dormant(rx, ry):
                rx, ry = self._robot_xy()
                reachable, unreachable = self._split(rx, ry)
                goal = self._tour_goal(0, rx, ry, reachable)
                self.pursued = self.plan.pop('sent', None) if goal is not None else None
                self.pursued_t = time.time()
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
