from glob import glob

from setuptools import setup

package_name = 'habitat_bridge'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml') + glob('config/*.rviz')),
        ('share/' + package_name + '/scripts', glob('scripts/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Gaeun Bang',
    maintainer_email='glad@ewha.ac.kr',
    description='Habitat-sim <-> ROS 2 bridge for frontier exploration',
    license='BSD-3-Clause',
    entry_points={
        'console_scripts': [
            'bridge_node = habitat_bridge.bridge_node:main',
        ],
    },
)
