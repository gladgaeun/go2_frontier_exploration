from glob import glob

from setuptools import setup

package_name = 'trav_exploration'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml') + glob('config/*.rviz') + glob('config/*.xml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Gaeun Bang',
    maintainer_email='glad@ewha.ac.kr',
    description='Frontier exploration on a 3D voxel traversability map (TravExplorer)',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'voxel_mapper_node = trav_exploration.voxel_mapper_node:main',
            'frontier_goal_node = trav_exploration.frontier_goal_node:main',
            'collision_guard_node = trav_exploration.collision_guard_node:main',
            'semantic_seg_node = trav_exploration.semantic_seg_node:main',
            'instance_map_node = trav_exploration.instance_map_node:main',
            'itm_node = trav_exploration.itm_node:main',
            'explore_executor_node = trav_exploration.explore_executor_node:main',
        ],
    },
)
