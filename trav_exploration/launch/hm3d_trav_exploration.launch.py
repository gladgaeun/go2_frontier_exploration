# Frontier exploration on the 3D voxel traversability map in Habitat-sim (HM3D, single floor):
#   habitat_bridge -> voxel_mapper_node (M_trav, 3D frontiers) -> frontier_goal_node
#   -> explore_executor_node (goal after goal to Nav2; no 2-D (x, y) blacklist) -> Nav2 (controller on
#      voxel_map/trav_grid; global path from voxel_mapper_node's 3-D support graph planner)
#   -> collision_guard_node (cmd_vel -> cmd_vel_safe: 3-D stair paths; guard_mode:=full also stops
#      before contact and marks blocked voxels)
#
# Start habitat_server.py (conda env) first, then:
#   ros2 launch trav_exploration hm3d_trav_exploration.launch.py
# The 2D stack (habitat_bridge hm3d_single_floor_exploration.launch.py) is left as it is.
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, LogInfo, OpaqueFunction,
                            RegisterEventHandler, TimerAction)
from launch.event_handlers import OnProcessExit
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterFile
from launch_ros.parameter_descriptions import ParameterValue
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    pkg_dir = get_package_share_directory('trav_exploration')

    args = [
        DeclareLaunchArgument('zmq_address', default_value='tcp://127.0.0.1:5555'),
        DeclareLaunchArgument('nav2_params', default_value=os.path.join(pkg_dir, 'config', 'nav2_trav.yaml')),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('explore', default_value='true',
                              description='false = sim + voxel map + Nav2 only (drive with RViz goals)'),
        DeclareLaunchArgument('explore_delay', default_value='15.0',
                              description='seconds (wall) to wait for the map and Nav2 before exploring'),
        DeclareLaunchArgument('explore_restarts', default_value='2',
                              description='times the exploration loop is restarted after it ends'),
        DeclareLaunchArgument('frontier_boundary', default_value='source', choices=['source', 'trav'],
                              description="frontier boundary test, see frontier3d.py ('trav' = literal)"),
        DeclareLaunchArgument('goal_order', default_value='atsp', choices=['atsp', 'score'],
                              description="frontier order: 'atsp' = TravExplorer open tour, 'score' = 2D ranking"),
        # hard constraint (Sec. IV-B): O_inf below the body radius, so a gap the body can pass stays
        # open; how narrow it is goes into the soft ESDF cost (eq. 15-16, d_safe), and a gap the
        # controller then cannot follow is closed by the execute-review (collision_guard).
        # = the body (0.18 m, Habitat ObjectNav agent), as M_trav of the paper removes the obstacles
        # inflated by the robot (Sec. VI-B). Earlier 0.15, which floored to 2 cells (0.15 / 0.05 =
        # 2.9999...), i.e. 0.10 m in effect: paths ran 0.08 m into the walls on the stairs. 0.16 (3
        # cells) once closed half the treads of the narrow stairs of 00803 (71 % -> 50 %
        # traversable): to be checked there.
        DeclareLaunchArgument('inflation_radius', default_value='0.18',
                              description='O_inf radius (voxel_mapper robot_radius); body 0.18 m (Habitat ObjectNav agent)'),
        DeclareLaunchArgument('segmentation', default_value='gt', choices=['gt', 'sam3'],
                              description="semantic mode input: 'gt' = habitat_server --semantic (HM3D "
                                          "annotation), 'sam3' = semantic_seg_node + sam3_server.py"),
        DeclareLaunchArgument('sam3_address', default_value='tcp://127.0.0.1:5557'),
        DeclareLaunchArgument('target', default_value='',
                              description="target object query of the instance map (e.g. 'couch'; segmentation:=sam3): "
                                          "the robot goes to it once it is reliable and stops there; '' = explore only"),
        DeclareLaunchArgument('value_map', default_value='true',
                              description='spatial value map with a target: itm_node + blip2_server.py (blip2 conda env)'),
        DeclareLaunchArgument('blip2_address', default_value='tcp://127.0.0.1:5558'),
        # instance map reliability (Sec. V-A): a cluster is a target (O_tar) once gamma >= min_confidence,
        # n >= min_count evidence points and detected in >= min_detections frames
        DeclareLaunchArgument('instance_min_confidence', default_value='0.5'),
        DeclareLaunchArgument('instance_min_count', default_value='60',
                              description='evidence points (0.1 m voxels) before an object cluster is a target'),
        DeclareLaunchArgument('instance_min_detections', default_value='4',
                              description='frames with a detection before a cluster is a target'),
        DeclareLaunchArgument('debug_dump_dir', default_value='',
                              description='save the voxel map around every climb stop here (diagnostics)'),
        DeclareLaunchArgument('floor_switch', default_value='off', choices=['off', 'shadow', 'on'],
                              description='floor-switch model (floor_switch package, v0): off; shadow = '
                                          'floor_switch_node runs and records p, the exploration is unchanged; '
                                          'on = frontier_goal switches floors on it (blip2_server needed)'),
        DeclareLaunchArgument('floor_switch_model',
                              default_value=os.path.expanduser('~/objectnav_runs/floor_switch/p_model/catw/online_model.json')),
        DeclareLaunchArgument('record_dir', default_value='',
                              description='periodic state of voxel_mapper / frontier_goal / instance_map as '
                                          'JSON Lines here (offline datasets; empty = off)'),
        DeclareLaunchArgument('stair_landmarks', default_value='auto', choices=['auto', 'instance', 'voxel'],
                              description="O_stair: 'instance' = stair clusters of the instance map (sam3), 'voxel' = "
                                          "connected stair voxels (before the instance map); 'auto' = instance with sam3"),
        DeclareLaunchArgument('map_z_below', default_value='0.25',
                              description='voxel map depth below the start floor [m] (3.5 when starting upstairs)'),
        DeclareLaunchArgument('map_z_above', default_value='2.0',
                              description='voxel map height above the start floor [m] (4.0 for two floors)'),
        # path_only: the guard follows the 3-D stair paths only; Nav2's commands pass through and
        # a blocked robot is left to the execute-review of voxel_mapper / frontier_goal (the paper's
        # way). full: also stop before contact, close narrow cells, back out on repeated failures
        DeclareLaunchArgument('guard_mode', default_value='path_only'),
        # true: g -> s also when stairs are seen for mode_enter_time (the paper's N_enter), not only
        # when the planner takes the stairs; s -> g stays the N_exit sight rule
        DeclareLaunchArgument('mode_auto_enter', default_value='false'),
        DeclareLaunchArgument('guard', default_value='true',
                              description='false = no collision guard (Nav2 drives the robot directly)'),
    ]

    # Publishes /clock itself, so it must run on wall time. mapper 'gt' also provides the static
    # map->odom; its /map is not used here (shown in RViz as the "Map" display, off by default).
    # With the guard, the robot takes its velocity from cmd_vel_safe (Nav2's cmd_vel after the guard).
    # with segmentation:=sam3 the bridge does not publish the ground-truth class image (same topic)
    use_gt = PythonExpression(["'", LaunchConfiguration('segmentation'), "' == 'gt'"])
    bridge_params = [{'use_sim_time': False, 'zmq_address': LaunchConfiguration('zmq_address'), 'mapper': 'gt',
                      'publish_semantic': ParameterValue(use_gt, value_type=bool)}]
    bridge = Node(
        package='habitat_bridge', executable='bridge_node', name='habitat_bridge', output='screen',
        parameters=bridge_params, remappings=[('cmd_vel', 'cmd_vel_safe')],
        condition=IfCondition(LaunchConfiguration('guard')))
    bridge_unguarded = Node(
        package='habitat_bridge', executable='bridge_node', name='habitat_bridge', output='screen',
        parameters=bridge_params, condition=UnlessCondition(LaunchConfiguration('guard')))

    guard = Node(
        package='trav_exploration', executable='collision_guard_node', name='collision_guard', output='screen',
        condition=IfCondition(LaunchConfiguration('guard')),
        parameters=[{'use_sim_time': True,
                     'robot_radius': 0.18,     # body = habitat_server --agent-radius (Habitat ObjectNav)
                     'guard_mode': LaunchConfiguration('guard_mode'),
                     'stop_distance': 0.1,
                     'h_min': 0.15, 'h_max': 0.88}])    # = voxel_mapper

    # stuck detection compares odometry with the command the robot executes
    executed_cmd = PythonExpression(["'cmd_vel_safe' if '", LaunchConfiguration('guard'),
                                     "'.lower() == 'true' else 'cmd_vel'"])
    # O_stair: the instance-map landmarks (SAM3) or the voxel map's own; voxel_mapper checks their
    # step structure and republishes them on voxel_map/stair_landmarks_checked (frontier_goal)
    landmark_topic = PythonExpression([
        "'instance_map/stair_landmarks' if '", LaunchConfiguration('stair_landmarks'), "' == 'instance' or ('",
        LaunchConfiguration('stair_landmarks'), "' == 'auto' and '", LaunchConfiguration('segmentation'),
        "' == 'sam3') else 'voxel_map/stair_landmarks'"])
    voxel = Node(
        package='trav_exploration', executable='voxel_mapper_node', name='voxel_mapper', output='screen',
        remappings=[('cmd_vel', executed_cmd)],
        parameters=[{'use_sim_time': True,
                     'stair_check_input': landmark_topic,
                     'resolution': 0.05,
                     # inflation radius = habitat_server --agent-radius (0.18, Habitat ObjectNav);
                     # cells whose centre is within it (disk of 3.6 cells at 0.05 m). (A doorway the
                     # navmesh agent just fits through may keep no traversable cell.)
                     'robot_radius': ParameterValue(LaunchConfiguration('inflation_radius'), value_type=float),
                     'h_max': 0.88,            # = habitat_server --agent-height (Habitat ObjectNav)
                     # body = habitat_server --agent-radius: mark_tight closes cells closer than
                     # this to obstacles at a guard stop; left at 0.25 it closed passages the
                     # 0.18 m body passes and boxed the robot in (HM3D v1 00832)
                     'body_radius': 0.18,
                     # door thresholds up to ~0.15 m are not obstacles: the navmesh (cell height
                     # 0.2 m) lets the agent over them, and a Go2 steps over them; with 0.1 the
                     # walk-in closet's threshold (0.125-0.175 m) closed its doorway on the map
                     'h_min': 0.15,
                     'max_range': 5.0,         # TravExplorer sim: depth 0-5 m
                     # map extent above the start floor: 2 m for one floor, ~4 m to hold the
                     # upper floor of a two-storey house (stairs; habitat_server --max-climb 0.2)
                     'z_above': ParameterValue(LaunchConfiguration('map_z_above'), value_type=float),
                     'z_below': ParameterValue(LaunchConfiguration('map_z_below'), value_type=float),
                     'frontier_boundary': LaunchConfiguration('frontier_boundary'),
                     'mode_auto_enter': ParameterValue(LaunchConfiguration('mode_auto_enter'), value_type=bool),
                     # path_only = the paper's stuck handling: v_front only, no 2-D narrow-gap closing
                     'stuck_close_tight': ParameterValue(PythonExpression(
                         ["'", LaunchConfiguration('guard_mode'), "' != 'path_only'"]), value_type=bool),
                     'debug_dump_dir': LaunchConfiguration('debug_dump_dir'),
                     'record_publish': ParameterValue(PythonExpression(["'", LaunchConfiguration('floor_switch'), "' != 'off'"]), value_type=bool),
                     'record_dir': LaunchConfiguration('record_dir')}])

    # SAM3 segmentation (start sam3_server.py in the sam3 conda env first): the class image of the
    # semantic mode, and the detections of the instance map's queries (stairs, the target)
    use_sam3 = IfCondition(PythonExpression(["'", LaunchConfiguration('segmentation'), "' == 'sam3'"]))
    segmenter = Node(
        package='trav_exploration', executable='semantic_seg_node', name='semantic_seg', output='screen',
        condition=use_sam3,
        parameters=[{'use_sim_time': True, 'zmq_address': LaunchConfiguration('sam3_address'),
                     'detect_queries': ['stairs'],
                     'target': LaunchConfiguration('target')}])
    # probabilistic instance map (Sec. V-A): O_stair (landmark branch) and O_tar (object branch)
    instances = Node(
        package='trav_exploration', executable='instance_map_node', name='instance_map', output='screen',
        condition=use_sam3,
        parameters=[{'use_sim_time': True,
                     'landmark_queries': ['stairs'],
                     'object_queries': LaunchConfiguration('target'),
                     'max_range': 5.0,
                     'min_confidence': ParameterValue(LaunchConfiguration('instance_min_confidence'), value_type=float),
                     'min_count': ParameterValue(LaunchConfiguration('instance_min_count'), value_type=int),
                     'min_detections': ParameterValue(LaunchConfiguration('instance_min_detections'), value_type=int),
                     'record_publish': ParameterValue(PythonExpression(["'", LaunchConfiguration('floor_switch'), "' != 'off'"]), value_type=bool),
                     'record_dir': LaunchConfiguration('record_dir')}])
    # spatial value map (Sec. V-B): BLIP-2 image-text relevance of the target (start blip2_server.py
    # in the blip2 conda env first); only with a target
    itm = Node(
        package='trav_exploration', executable='itm_node', name='itm', output='screen',
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration('target'), "' != '' and '",
                                                LaunchConfiguration('value_map'), "'.lower() == 'true'"])),
        parameters=[{'use_sim_time': True, 'zmq_address': LaunchConfiguration('blip2_address'),
                     'target': LaunchConfiguration('target')}])

    goals = Node(
        package='trav_exploration', executable='frontier_goal_node', name='frontier_goal', output='screen',
        parameters=[{'use_sim_time': True,
                     # trav_grid obstacles are O_inf (+0.22 m): 0.4 m from real obstacles like the
                     # 2D HM3D launch (goal_clearance 0.4 on /map)
                     'goal_clearance': 0.18,
                     'goal_order': LaunchConfiguration('goal_order'),
                     'target': LaunchConfiguration('target'),
                     'value_weight': ParameterValue(PythonExpression(
                         ["3.0 if '", LaunchConfiguration('value_map'), "'.lower() == 'true' else 0.0"]),
                         value_type=float),
                     'stair_landmark_topic': 'voxel_map/stair_landmarks_checked',
                     'record_publish': ParameterValue(PythonExpression(["'", LaunchConfiguration('floor_switch'), "' != 'off'"]), value_type=bool),
                     'floor_switch_enable': ParameterValue(PythonExpression(
                         ["'", LaunchConfiguration('floor_switch'), "' == 'on'"]), value_type=bool),
                     'record_dir': LaunchConfiguration('record_dir'),
                     'inflation_radius': ParameterValue(LaunchConfiguration('inflation_radius'), value_type=float)}])     # = voxel_mapper robot_radius (d_obs in eq. 16)

    # floor-switch model (floor_switch package, v0): p of the target on this floor from the records
    # of voxel_mapper / instance_map / frontier_goal (record_publish) and the room types (BLIP-2 ITC
    # on blip2_server); frontier_goal switches floors on it with floor_switch:=on
    floor_switch = Node(
        package='floor_switch', executable='floor_switch_node', name='floor_switch', output='screen',
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration('floor_switch'), "' != 'off'"])),
        parameters=[{'use_sim_time': True, 'model': LaunchConfiguration('floor_switch_model'),
                     'blip2_address': LaunchConfiguration('blip2_address'),
                     'record_dir': LaunchConfiguration('record_dir')}])

    # Nav2 as in nav2_bringup navigation_launch.py, without the planner_server: the global path
    # is planned on the 3-D support graph of the voxel map (TravExplorer Sec. VI-B) by
    # voxel_mapper_node's compute_path_to_pose action; the behaviour trees (config/bt_*.xml) do
    # not use the planner_server's global costmap. Nav2 follows the path (controller, recoveries).
    # Started after the voxel node: bt_navigator needs the compute_path_to_pose server when it
    # loads its tree.
    nav2_params = ParameterFile(RewrittenYaml(source_file=LaunchConfiguration('nav2_params'), root_key='',
                                              param_rewrites={'use_sim_time': 'true', 'autostart': 'true'},
                                              convert_types=True), allow_substs=True)
    tf_remap = [('/tf', 'tf'), ('/tf_static', 'tf_static')]
    bt_xml = {'default_nav_to_pose_bt_xml': os.path.join(pkg_dir, 'config', 'bt_navigate_to_pose.xml'),
              'default_nav_through_poses_bt_xml': os.path.join(pkg_dir, 'config', 'bt_navigate_through_poses.xml')}
    nav2_nodes = [
        Node(package='nav2_controller', executable='controller_server', output='screen',
             parameters=[nav2_params], remappings=tf_remap + [('cmd_vel', 'cmd_vel_nav')]),
        Node(package='nav2_smoother', executable='smoother_server', name='smoother_server', output='screen',
             parameters=[nav2_params], remappings=tf_remap),
        Node(package='nav2_behaviors', executable='behavior_server', name='behavior_server', output='screen',
             parameters=[nav2_params], remappings=tf_remap),
        Node(package='nav2_bt_navigator', executable='bt_navigator', name='bt_navigator', output='screen',
             parameters=[nav2_params, bt_xml], remappings=tf_remap),
        Node(package='nav2_waypoint_follower', executable='waypoint_follower', name='waypoint_follower',
             output='screen', parameters=[nav2_params], remappings=tf_remap),
        Node(package='nav2_velocity_smoother', executable='velocity_smoother', name='velocity_smoother',
             output='screen', parameters=[nav2_params],
             remappings=tf_remap + [('cmd_vel', 'cmd_vel_nav'), ('cmd_vel_smoothed', 'cmd_vel')]),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager', name='lifecycle_manager_navigation',
             output='screen',
             parameters=[{'use_sim_time': True}, {'autostart': True},
                         {'node_names': ['controller_server', 'smoother_server', 'behavior_server',
                                         'bt_navigator', 'waypoint_follower', 'velocity_smoother']}]),
    ]
    nav2 = TimerAction(period=3.0, actions=nav2_nodes)

    # The exploration loop (explore_executor_node): asks frontier_pose for the rank-th best goal,
    # checks it with the planner, sends it to Nav2. It replaces the 2-D Go2 loop
    # (frontier_exploration_node.py, unchanged for the 2-D stack), whose (x, y) blacklist of failed
    # and canceled goals turned down frontiers of another floor (00891); a failure goes into the map
    # (execute-review, eq. 17) and frontier_goal_node's dormant clusters instead. No name= (two
    # nodes in it: the executor and its BasicNavigator).
    # When the loop ends (no goal left), it is started again up to explore_restarts times: the
    # dormant clusters are released by then or the map has changed.
    def explorer_node():
        return Node(package='trav_exploration', executable='explore_executor_node',
                    output='screen', parameters=[{'use_sim_time': True,
                                                  'record_dir': LaunchConfiguration('record_dir')}])

    def restart(left):
        def on_exit(event, context):
            if context.is_shutdown or event.returncode != 0 or left <= 0:
                return None
            node = explorer_node()
            return [LogInfo(msg=f'[explore] exploration loop ended; restarting it '
                                f'({left - 1} restarts left)'),
                    node,
                    RegisterEventHandler(OnProcessExit(target_action=node, on_exit=restart(left - 1)))]
        return on_exit

    def start_explorer(context):
        node = explorer_node()
        left = int(LaunchConfiguration('explore_restarts').perform(context))
        return [node, RegisterEventHandler(OnProcessExit(target_action=node, on_exit=restart(left)))]

    explorer = TimerAction(
        period=LaunchConfiguration('explore_delay'),
        condition=IfCondition(LaunchConfiguration('explore')),
        actions=[OpaqueFunction(function=start_explorer)])

    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2', output='screen',
        arguments=['-d', os.path.join(pkg_dir, 'config', 'hm3d_trav.rviz')],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(LaunchConfiguration('rviz')))

    return LaunchDescription(args + [bridge, bridge_unguarded, guard, voxel, segmenter, instances, itm, goals, floor_switch, nav2,
                                     explorer,
                                     rviz])
