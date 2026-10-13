# Full stack for frontier exploration in Habitat-sim:
#   habitat_bridge -> slam_toolbox -> Nav2 -> classical_frontier_detector -> frontier_exploration_node
#
# Start habitat_server.py (conda env) first, then:
#   ros2 launch habitat_bridge hm3d_single_floor_exploration.launch.py
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition, LaunchConfigurationEquals
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bridge_dir = get_package_share_directory('habitat_bridge')
    fe_dir = get_package_share_directory('frontier_exploration')
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')

    zmq_address = LaunchConfiguration('zmq_address')
    nav2_params = LaunchConfiguration('nav2_params')
    slam_params = LaunchConfiguration('slam_params')
    rviz = LaunchConfiguration('rviz')
    explore = LaunchConfiguration('explore')
    explore_delay = LaunchConfiguration('explore_delay')
    region_size_thresh = LaunchConfiguration('region_size_thresh')

    args = [
        DeclareLaunchArgument('zmq_address', default_value='tcp://127.0.0.1:5555'),
        DeclareLaunchArgument('nav2_params',
                              default_value=os.path.join(bridge_dir, 'config', 'nav2_hm3d_single_floor.yaml')),
        DeclareLaunchArgument('slam_params',
                              default_value=os.path.join(bridge_dir, 'config', 'slam_toolbox_hm3d_single_floor.yaml')),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('explore', default_value='true',
                              description='false = bring up sim+SLAM+Nav2 only (drive with RViz goals)'),
        DeclareLaunchArgument('explore_delay', default_value='15.0',
                              description='seconds (wall) to wait for SLAM/Nav2 before exploring'),
        DeclareLaunchArgument('region_size_thresh', default_value='8',
                              description='min frontier cells (0.05 m each) to count as a frontier'),
        DeclareLaunchArgument('mapper', default_value='gt', choices=['gt', 'slam_toolbox'],
                              description='gt: occupancy map from the exact simulator pose (default). '
                                          'slam_toolbox: run SLAM like on the real Go2.'),
    ]

    # Publishes /clock itself, so it must run on wall time.
    bridge = Node(
        package='habitat_bridge', executable='bridge_node', name='habitat_bridge', output='screen',
        parameters=[{'use_sim_time': False, 'zmq_address': zmq_address,
                     'mapper': LaunchConfiguration('mapper')}])

    # Only with mapper:=slam_toolbox. It then owns map->odom (the bridge does not publish it).
    slam = Node(
        package='slam_toolbox', executable='async_slam_toolbox_node', name='slam_toolbox',
        output='screen', condition=LaunchConfigurationEquals('mapper', 'slam_toolbox'),
        parameters=[slam_params, {'use_sim_time': True, 'transform_publish_period': 0.05}])

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2_bringup_dir, 'launch', 'navigation_launch.py')),
        launch_arguments={'use_sim_time': 'true',
                          'params_file': nav2_params,
                          'autostart': 'true',
                          'use_composition': 'False'}.items())

    detector = Node(
        package='frontier_exploration', executable='classical_frontier_detector',
        name='classical_frontier_detector', output='screen',
        parameters=[{'use_sim_time': True,
                     'region_size_thresh': region_size_thresh,
                     'robot_width': 0.5,
                     # move goals to a known-free cell >= 0.4 m from obstacles (Go2 launch: off)
                     'goal_clearance': 0.4,
                     'occupancy_map_msg': 'map'}])

    # No name= here: this process holds two nodes (frontier_explorer + basic_navigator) and
    # name= would rename both to the same name.
    explorer = TimerAction(
        period=explore_delay,
        condition=IfCondition(explore),
        actions=[Node(
            package='frontier_exploration', executable='frontier_exploration_node.py',
            output='screen', parameters=[{'use_sim_time': True,
                                          # don't go back to a frontier already reached
                                          'skip_reached_goals': True}])])

    rviz_node = Node(
        package='rviz2', executable='rviz2', name='rviz2', output='screen',
        # frontier.rviz + robot pose arrow, global/local plan, local costmap, footprint
        arguments=['-d', os.path.join(bridge_dir, 'config', 'hm3d_single_floor.rviz')],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(rviz))

    return LaunchDescription(args + [bridge, slam, nav2, detector, explorer, rviz_node])
