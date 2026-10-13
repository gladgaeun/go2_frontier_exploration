import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_frontier_exploration = get_package_share_directory('frontier_exploration')
    
    rviz_config_file = os.path.join(pkg_frontier_exploration, 'rviz', 'frontier.rviz')

    # 1. RViz2 Node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        output='screen'
    )

    # 2. C++ Frontier Detector Node
    detector_node = Node(
        package='frontier_exploration',
        executable='classical_frontier_detector',
        name='classical_frontier_detector',
        output='screen',
        parameters=[
            {"region_size_thresh": 4},
            {"robot_width": 0.4},
            {"occupancy_map_msg": "map"}
        ]
    )

    # 3. Python Frontier Exploration Node
    exploration_node = Node(
        package='frontier_exploration',
        executable='frontier_exploration_node.py',
        name='frontier_exploration_node',
        output='screen'
    )

    # Include all nodes in the launch description
    return LaunchDescription([
        rviz_node,
        detector_node,
        exploration_node
    ])