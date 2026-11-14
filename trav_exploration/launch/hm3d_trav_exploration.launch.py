# Frontier exploration on the 3D voxel traversability map in Habitat-sim (HM3D, single floor):
#   habitat_bridge -> voxel_mapper_node (M_trav, 3D frontiers) -> frontier_goal_node
#   -> frontier_exploration_node (2D exploration loop, unchanged) -> Nav2 (controller on
#      voxel_map/trav_grid; global path from voxel_mapper_node's 3-D support graph planner)
#   -> collision_guard_node (cmd_vel -> cmd_vel_safe: stops before contact, marks blocked voxels)
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
        DeclareLaunchArgument('inflation_radius', default_value='0.22',
                              description='O_inf radius (voxel_mapper robot_radius); real body 0.25 m'),
        DeclareLaunchArgument('segmentation', default_value='gt', choices=['gt', 'sam3'],
                              description="semantic mode input: 'gt' = habitat_server --semantic (HM3D "
                                          "annotation), 'sam3' = semantic_seg_node + sam3_server.py"),
        DeclareLaunchArgument('sam3_address', default_value='tcp://127.0.0.1:5557'),
        DeclareLaunchArgument('map_z_above', default_value='2.0',
                              description='voxel map height above the start floor [m] (4.0 for two floors)'),
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
                     'robot_radius': 0.25,     # real body = habitat_server --agent-radius
                     'stop_distance': 0.1,
                     'h_min': 0.15, 'h_max': 0.6}])     # = voxel_mapper

    # stuck detection compares odometry with the command the robot executes
    executed_cmd = PythonExpression(["'cmd_vel_safe' if '", LaunchConfiguration('guard'),
                                     "'.lower() == 'true' else 'cmd_vel'"])
    voxel = Node(
        package='trav_exploration', executable='voxel_mapper_node', name='voxel_mapper', output='screen',
        remappings=[('cmd_vel', executed_cmd)],
        parameters=[{'use_sim_time': True,
                     'resolution': 0.05,
                     # inflation radius; like Nav2 robot_radius in the 2D stack, slightly below
                     # habitat_server --agent-radius (0.25): with 0.25 a doorway the navmesh
                     # agent just fits through has no traversable cell left
                     'robot_radius': ParameterValue(LaunchConfiguration('inflation_radius'), value_type=float),
                     'h_max': 0.6,             # = habitat_server --agent-height
                     # door thresholds up to ~0.15 m are not obstacles: the navmesh (cell height
                     # 0.2 m) lets the agent over them, and a Go2 steps over them; with 0.1 the
                     # walk-in closet's threshold (0.125-0.175 m) closed its doorway on the map
                     'h_min': 0.15,
                     'max_range': 5.0,         # TravExplorer sim: depth 0-5 m
                     # map extent above the start floor: 2 m for one floor, ~4 m to hold the
                     # upper floor of a two-storey house (stairs; habitat_server --max-climb 0.2)
                     'z_above': ParameterValue(LaunchConfiguration('map_z_above'), value_type=float),
                     'frontier_boundary': LaunchConfiguration('frontier_boundary')}])

    # SAM3 segmentation (start sam3_server.py in the sam3 conda env first)
    segmenter = Node(
        package='trav_exploration', executable='semantic_seg_node', name='semantic_seg', output='screen',
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration('segmentation'), "' == 'sam3'"])),
        parameters=[{'use_sim_time': True, 'zmq_address': LaunchConfiguration('sam3_address')}])

    goals = Node(
        package='trav_exploration', executable='frontier_goal_node', name='frontier_goal', output='screen',
        parameters=[{'use_sim_time': True,
                     # trav_grid obstacles are O_inf (+0.22 m): 0.4 m from real obstacles like the
                     # 2D HM3D launch (goal_clearance 0.4 on /map)
                     'goal_clearance': 0.18,
                     'goal_order': LaunchConfiguration('goal_order'),
                     'inflation_radius': ParameterValue(LaunchConfiguration('inflation_radius'), value_type=float)}])     # = voxel_mapper robot_radius (d_obs in eq. 16)

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

    # The 2D exploration loop, unchanged: asks frontier_pose for the rank-th best goal, checks it
    # with the Nav2 planner, blacklists failed and reached goals. No name= (two nodes in it).
    # A goal that failed once (Nav2 giving up in a narrow passage) stays blacklisted, and when
    # every other frontier is unreachable the loop ends although that one is reachable. So when it
    # ends, it is started again with an empty blacklist, up to explore_restarts times: unreachable
    # goals then fail at path planning at once, a reachable one gets another try.
    def explorer_node():
        return Node(package='frontier_exploration', executable='frontier_exploration_node.py',
                    output='screen', parameters=[{'use_sim_time': True, 'skip_reached_goals': True}])

    def restart(left):
        def on_exit(event, context):
            if context.is_shutdown or event.returncode != 0 or left <= 0:
                return None
            node = explorer_node()
            return [LogInfo(msg=f'[explore] exploration loop ended; restarting it with an empty '
                                f'goal blacklist ({left - 1} restarts left)'),
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

    return LaunchDescription(args + [bridge, bridge_unguarded, guard, voxel, segmenter, goals, nav2, explorer,
                                     rviz])
