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
simulator pose, which avoids two publishers of map->odom),
`voxel_map:=false` (no 3D voxel map), `frontier_map:=voxel` (frontiers from the voxel map
instead of /map, see below).

## 3D voxel map

The voxel map (occupancy + traversability from the RGB-D camera, TravExplorer) lives in the
[`trav_exploration`](../trav_exploration/README.md) package; this launch file starts its
`voxel_mapper_node` (`voxel_map:=false` to disable).

## Files

| file | role |
|---|---|
| `scripts/habitat_server.py` | conda side: HM3D scene, virtual 2D LiDAR from 4 depth cameras, RGB-D camera, /cmd_vel with navmesh collision |
| `habitat_bridge/bridge_node.py` | ROS side: /clock /odom /tf /scan /camera/{image_raw,depth/image_raw,camera_info} /map, /cmd_vel to the server |
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
| `--rgb-width`, `--rgb-height` | 640, 480 | RGB-D camera; `--rgb-width 0` disables it (and the depth image) |
| `--camera-height`, `--camera-hfov` | 0.88, 79 | RGB-D camera as in TravExplorer (Habitat ObjectNav 2023) |
| `--camera-height-ref` | floor | `floor`: camera height above the floor measured at the start (the agent base sits on the navmesh, ~0.16 m above the mesh floor in 00800); `base`: above the agent base |
| `--no-depth` | (on) | no depth image next to the RGB camera (the voxel map needs it) |
| `--semantic` | off | ground-truth class image registered with the RGB camera on `camera/semantic/image_raw` (mono8: 0 other, 1 floor, 2 rug/mat, 3 stairs) from the HM3D semantic annotation; finds `hm3d_annotated_*scene_dataset_config.json` above the scene unless `--dataset-config` is given (scene coordinates are the same with it) |
| `--seed` | 0 | random start pose selection |
| `--max-climb` | 0.1 | navmesh step height; 0.1 keeps stairs out (single-floor exploration), 0.2 (Habitat default) connects the floors of 00800 (stair climb) |
| `--no-navmesh-obstacles` | (on) | by default low obstacles/stairs/drop-offs from the navmesh are added to /scan |

Camera pitch: `camera/pitch_cmd` (std_msgs/Float64, rad, + = down) tilts the RGB / depth / semantic
cameras (the scan cameras stay level), like the look_down action of TravExplorer's simulation;
`base_link -> camera_link` is published on /tf every step with the pitch the server applied,
stamped like the images (it used to be a static transform).

## Quick checks

```bash
ros2 topic hz /scan          # ~20 Hz
ros2 run tf2_ros tf2_echo map base_link
```
In RViz (fixed frame `map`): the scan should line up with the walls while the robot turns.
If it mirrors or rotates the wrong way, the frame conversion is off - report it.
