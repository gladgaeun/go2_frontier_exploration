# habitat_bridge

HM3D single-floor frontier exploration in Habitat-sim with the same ROS 2 stack as the Go2
(`frontier_exploration` + Nav2). The robot explores the floor it starts on (stairs are kept
out of the navmesh) and stops with `Exploration complete!` when no frontier is left.

Habitat-sim and ROS are connected as a bridge, so the exploration nodes are the ones that run
on the real robot. Habitat lives in a conda env (Python 3.9) and ROS 2 Humble in
the system Python (3.10), so the two run as separate processes connected by ZMQ:

```
[conda: habitat_server.py]  <-- ZMQ -->  [ROS 2: bridge_node]
  HM3D scene, 4 depth cams -> 360 scan      /clock, /odom, TF odom->base_link
  /cmd_vel integration + navmesh collision  /scan, /camera/image_raw, sub /cmd_vel
```

## Setup (once)

```bash
# habitat side
conda activate hsim-test
pip install pyzmq

# ROS side (new terminal, no conda)
sudo apt install -y python3-zmq ros-humble-slam-toolbox ros-humble-navigation2 ros-humble-nav2-bringup \
    ros-humble-rmw-cyclonedds-cpp
cd ~/ros2_ws && colcon build --symlink-install && source install/setup.bash
```

## Run

Terminal 1 (habitat):
```bash
conda activate hsim-test
python ~/ros2_ws/src/go2_frontier_exploration/habitat_bridge/scripts/habitat_server.py \
    --scene ~/data/versioned_data/hm3d-0.2/hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb
```
It prints the start pose; pass it back with `--start x,y,z,yaw` to repeat the same run.

Terminal 2 (ROS, no conda; tested with CycloneDDS):
```bash
source ~/ros2_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp   # sudo apt install ros-humble-rmw-cyclonedds-cpp
ros2 launch habitat_bridge hm3d_single_floor_exploration.launch.py
```

Useful launch arguments: `explore:=false` (sim + mapping + Nav2 only, send goals from RViz),
`rviz:=false`, `explore_delay:=20.0`, `region_size_thresh:=8`,
`mapper:=slam_toolbox` (run SLAM like on the Go2; default `gt` builds /map from the exact
simulator pose, which avoids two publishers of map->odom).

## Files

| file | role |
|---|---|
| `scripts/habitat_server.py` | conda side: HM3D scene, virtual 2D LiDAR from 4 depth cameras, RGB, /cmd_vel with navmesh collision |
| `habitat_bridge/bridge_node.py` | ROS side: /clock /odom /tf /scan /camera/image_raw /map, /cmd_vel to the server |
| `habitat_bridge/gt_mapper.py` | occupancy grid from the exact simulator pose |
| `launch/hm3d_single_floor_exploration.launch.py` | bridge + Nav2 + frontier detector + explorer + RViz |
| `config/nav2_hm3d_single_floor.yaml` | Nav2 parameters |
| `config/slam_toolbox_hm3d_single_floor.yaml` | only with `mapper:=slam_toolbox` |
| `config/hm3d_single_floor.rviz` | RViz layout |

Differences from the Go2 setup: costmaps are built from
/map only, the controller is Regulated Pure Pursuit, and frontier goals are moved to a free
cell at least 0.4 m from obstacles (`goal_clearance` of `classical_frontier_detector`; 0 = the
Go2 behaviour, a 0.3 m pullback toward the robot).
The explorer skips frontier goals it has already reached (`skip_reached_goals`), and the GT
mapper keeps a cell occupied once it has been seen occupied, so grazing beams cannot open
gaps in walls. Exploration ends with `Exploration complete!` when no new frontier is left.

## Server options worth knowing

| option | default | note |
|---|---|---|
| `--agent-radius` | 0.25 | navmesh radius, kept slightly above Nav2 `robot_radius` (0.22) |
| `--start-clearance` | 0.6 | random start pose keeps this distance from obstacles |
| `--laser-height` | 0.4 | height of the virtual 2D LiDAR plane |
| `--max-range` | 10.0 | keep equal to slam_toolbox `max_laser_range` |
| `--rgb-width 0` | 320 | disables the RGB camera |
| `--seed` | 0 | random start pose selection |
| `--max-climb` | 0.1 | navmesh step height; 0.1 keeps stairs out (single-floor exploration) |
| `--no-navmesh-obstacles` | (on) | by default low obstacles/stairs/drop-offs from the navmesh are added to /scan |

## Quick checks

```bash
ros2 topic hz /scan          # ~20 Hz
ros2 run tf2_ros tf2_echo map base_link
```
In RViz (fixed frame `map`): the scan should line up with the walls while the robot turns.
If it mirrors or rotates the wrong way, the frame conversion is off - report it.
