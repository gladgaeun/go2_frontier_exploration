# trav_exploration

Frontier exploration on a 3D voxel traversability map, following TravExplorer
([arXiv 2605.19958](https://arxiv.org/abs/2605.19958), Sec. III-A to C). The 2D exploration stack
(`frontier_exploration`) is left unchanged; this package is built up step by step.

Status: voxel map with geometric and semantic traversability (mode switch g/s; HM3D ground truth
or SAM3), traversable 3D frontiers, exploration on them in Habitat with the paper's frontier
order (open ATSP), and the chain of exploration over two floors: current-floor frontiers, then a
stair landmark, the climb (3-D path search on support voxels) and the next floor. Not yet:
going down, semantic frontiers F_s, target objects / value map.

## Run (HM3D in Habitat-sim)

```bash
# terminal 1 (conda env with habitat-sim)
conda activate hsim-test
python ~/ros2_ws/src/go2_frontier_exploration/habitat_bridge/scripts/habitat_server.py \
    --scene ~/data/versioned_data/hm3d-0.2/hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb
# terminal 2
source ~/ros2_ws/install/setup.bash && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 launch trav_exploration hm3d_trav_exploration.launch.py
```
SAM3 instead of the ground-truth segmentation (start the habitat server without `--semantic`):
```bash
# terminal 3 (SAM3 needs Python 3.12; hf auth login once for the gated checkpoint)
conda activate sam3
python ~/ros2_ws/src/go2_frontier_exploration/trav_exploration/scripts/sam3_server.py
# terminal 2
ros2 launch trav_exploration hm3d_trav_exploration.launch.py segmentation:=sam3
# with a target object (instance map, see "Instance map"): stops once it reaches it
ros2 launch trav_exploration hm3d_trav_exploration.launch.py segmentation:=sam3 target:=couch map_z_above:=4.0
```
Arguments: `explore:=false` (map + Nav2 only, send goals from RViz), `rviz:=false`,
`frontier_boundary:=trav` (literal frontier boundary test, see below), `goal_order:=score`
(the 2D frontier ranking instead of the ATSP tour).

```
habitat_bridge (RGB-D, odom, TF)
  -> voxel_mapper_node: voxel map, M_trav, 3D frontier clusters
  -> frontier_goal_node: frontier_pose service (rank-th best goal)
  -> explore_executor_node: goal loop, termination (no (x, y) blacklist)
  -> Nav2 without planner_server: global path from voxel_mapper_node's compute_path_to_pose
     (3-D support graph), controller / recoveries on voxel_map/trav_grid (config/nav2_trav.yaml)
```

| file | role |
|---|---|
| `voxel_mapper.py` | voxel map: occupancy, M_g (2-D ray casting), M_s + stair voxels, incremental inflation, M_trav, stuck detector |
| `semantic.py` | traversable classes of the segmentation image, g/s mode switch with hysteresis |
| `semantic_seg_node.py` | ROS node: camera/image_raw -> sam3_server -> camera/semantic/image_raw (+ camera/detections) |
| `instance_map.py` | probabilistic instance map (Sec. V-A, eq. 4-6): association, fusion, negative evidence |
| `instance_map_node.py` | ROS node: detections + depth -> instance map -> O_stair, O_tar, markers |
| `value_map.py` | spatial value map (Sec. V-B, eq. 7), view weight, frontier utility |
| `itm_node.py`, `scripts/blip2_server.py` | BLIP-2 image-text relevance of each camera image (blip2 conda env, ZMQ) |
| `trav3d.py` | 3-D path search on support voxels: one support graph of the map (planning, J_geo, stair climb) |
| `scripts/sam3_server.py` | SAM3 text-prompt segmentation server (sam3 conda env, ZMQ) |
| `frontier3d.py` | traversable frontiers: candidates, 26-connected clusters, incremental update |
| `tour.py` | frontier order: open ATSP over {x_r} U F_g with traversable path costs (Sec. VI-A/B) |
| `goal_selection.py` | goal placement (and the 2D ranking for `goal_order:=score`), ported from the 2D detector |
| `voxel_mapper_node.py` | ROS node: depth + TF in, map / frontier topics out |
| `frontier_goal_node.py` | ROS node: `frontier_pose` service (same contract as `classical_frontier_detector`) |
| `config/nav2_trav.yaml` | Nav2 with costmaps from `voxel_map/trav_grid` (controller, recoveries) |
| `config/bt_navigate_*.xml` | Nav2 behaviour trees without the planner_server's global costmap |
| `../trav_interfaces` | `GeoCosts.srv` (J_geo between points on the support graph), `Detections.msg` (instance masks + confidences), `ImageScore.msg` (image-text relevance) |
| `launch/hm3d_trav_exploration.launch.py` | everything above + RViz (`config/hm3d_trav.rviz`) |

## Voxel map

```
depth -> point cloud -> occupancy log-odds O -> inflate (O \ T) ----------------> O_inf --+
depth -> points with h_min < p_z - z_r < h_max -> 2-D ray casting -> M_g --+              |
RGB mask + depth (semantic mode, not yet) -----------------------> M_s --+-> T (mode) - O_inf -> M_trav
```

- **Occupancy**: standard log-odds per voxel (misses along each depth ray, a hit at its end);
  voxels with l >= l_occ are occupied.
- **Geometric traversability M_g**: hit points in the robot collision-height range
  h_min < p_z - z_r < h_max are projected onto the support plane z = z_r, and a 2-D ray is cast
  from each projected point to the robot support state x_r. Support voxels the ray passes
  through (exact grid traversal) are inserted into M_g; occupied voxels are skipped.
  - Support voxels are the voxels of the floor surface (the robot stands on them). They are
    occupied by the floor points themselves, so "occupied" means an obstacle standing on the
    support voxel at collision height.
  - M_g only grows. An obstacle that reaches into the robot body without touching the floor
    (table top, or the unobserved inside of a box) leaves its support voxels in M_g, and O_inf
    removes them from M_trav.
- **Collision height**: a voxel dz layers above a support voxel is an obstacle if its centre
  height dz * res is above h_min (and up to h_max). Lower voxels (door thresholds, rug edges)
  are passable and do not block a doorway.
- **O_inf**: occupied voxels at collision height and not in T are inflated by `robot_radius` in
  xy and downward only, so floor voxels never block their own layer. Incremental: the bounding
  box of the changed voxels C_t is re-inflated; the result equals global inflation.
- **M_trav** = T \ O_inf (eq. 1), T = M_g in mode `g`.
- **z_r**: estimated once from the floor points of the depth image (not from base_link: in
  Habitat the agent sits on the navmesh, ~0.16 m above the mesh floor in scene 00800).
  `track_floor:=true` keeps estimating it.
- **Stuck handling** (eq. 17): `/odom` vs. the motion predicted from the executed command over 1 s
  (the Habitat bridge publishes it as `cmd_vel_executed`, stamped like the odometry of the same
  step, and the two are paired by stamp; otherwise `cmd_vel_safe` as it arrives: when the node
  was busy re-inflating after a mode switch, Nav2's stop came in late and every switch produced a
  false "stuck" with blocked voxels), threshold min(0.15 m, 0.5 x commanded distance), so
  it also fires on the brief contacts where Nav2 creeps near obstacles. A blocked forward motion
  marks the voxel straight ahead (0.37 m from the centre) as permanently occupied and publishes
  it on `voxel_map/blocked` (Nav2 `blocked_layer`). Inflated, it closes a gap narrower than the
  robot but not a hallway whose wall the robot grazed.
- **The robot's own path** (not in the paper): cells the robot centre passed through are FREE on
  `trav_grid` even inside O_inf, and blocked marks (stuck, collision guard) are skipped where
  their inflation would come within 2 cells of that path. Nav2 then never finds the robot inside
  an obstacle or its way back closed, which made every goal fail and ended exploration early.
- **Escape** (not in the paper, `collision_guard_node`): when navigation fails twice at one spot
  (within 0.6 m: Nav2's backup recovery moves the robot ~0.3 m between attempts),
  the robot drives ~1 m back along its own trail without Nav2 (forward danger check on). A
  narrow doorway can be open for the robot but closed on the map (inflation, blocked marks,
  Nav2's margins); it got shut in a small room that way.
  While it escapes, the voxel mapper makes no stuck marks and closes no cells (`collision_guard/
  escaping`), and the escape only stops for observed obstacles, not for blocked voxels: a stuck
  event during an escape once put a blocked voxel on the trail, and every later escape stopped in
  front of it next to the bed.
- **Cells closed after a stop** (`mark_tight`) stay at least 0.15 m from the robot. After a
  contact the cells right next to it are close to the obstacle too, and closing them made the
  robot's own cell INSCRIBED for Nav2: every plan, spin and backup failed at once.
- **Unreachable frontiers** (not in the paper, `frontier_goal_node`): clusters not connected to
  the robot on the Nav2 grid (behind walls, seen through mesh holes) are not offered as goals and
  are shown as grey spheres on `frontier_goal/unreachable`.

## Semantic mode (`semantic.py`, Sec. III-A)

The input is a class image per RGB pixel (0 other, 1 floor, 2 rug/mat, 3 stairs) on
`camera/semantic/image_raw`, from one of:
- `segmentation:=gt` (default): habitat_server `--semantic` renders the HM3D ground-truth annotation.
- `segmentation:=sam3`: `semantic_seg_node` sends the RGB images to `scripts/sam3_server.py`
  (SAM3, open-vocabulary, text prompts "floor", "rug", "stairs"; stairs win where masks overlap),
  which runs in its own conda env (Python 3.12). ~0.2 s per image on an RTX 5070 Ti, so only the
  newest image is segmented.
Class image and depth frame of a stamp are paired whichever arrives first. Without class images
nothing changes: the map stays in mode `g`.
- **M_s**: the voxels of depth points whose pixels are floor, rug or stairs; stair voxels are also
  kept apart (`voxel_map/stairs`, orange). Every pixel within 3.5 m (`semantic_stride`,
  `semantic_max_range`): with every 4th pixel, floor 2 m away is sampled every 5.6 cm along the
  view, M_s was striped with holes and each hole edge was a frontier.
- **Trusted stairs** (not in the paper): SAM3 labels the whole staircase as stairs, with the
  railing and the underside of the upper flight; in M_s they would be support surfaces, so the
  railing would no longer block. Stair points go into M_s only if they lie at most 0.6 m above
  the current floor z_r (the first ~3 steps; `stair_window`) and at most 0.3 m above the lowest
  semantic surface in their xy column (a railing stands ~0.9 m above the tread, a riser one step;
  `stair_column_gap`). The rest of the staircase is added while climbing, when z_r rises (as the
  paper completes FOV-limited stair observations on the way, Fig. 6d). The mode switch still
  counts all stair pixels. Applied to the ground truth as well.
- **Eq. 1**: mode `g` plans on M_g, mode `s` on **M_g U M_s** (not M_s alone as in the paper):
  M_s holds only floor that was segmented within 3.5 m, so switching to it at the stairs opened
  holes in the finished floor; each hole was a frontier the robot went back for, losing it again
  on the way (back in mode g), over and over. M_g is updated only in mode g (on the stairs the
  support plane is in mid-air). In `s`, occupied voxels in M_s (stair steps) are support surfaces
  and not inflated, so stairs are traversable. A switch re-inflates the map (~0.2 s).
- **Mode switch**: mode `s` is the mode of the stair transition. The planner switches to it when it
  decides to take the stairs (climb / descent request, `[mode] -> s: stair climb requested`); it
  switches back to `g` as in the paper (Sec. IV-A) when the robot remains sufficiently far from the
  stair region for N_exit consecutive class frames (`mode_exit_rule` distance: farther than
  `mode_exit_distance` 1.0 m in x-y from every mapped stair voxel near its height for
  `mode_exit_frames` 5, ~1.8 s at the SAM3 rate of ~2.8 frames/s; the paper gives no values; the
  earlier rule `mode_exit_rule:=sight` was "no stairs in view for 5 frames"), never during a climb (`climbing`: from the climb / descent request to the end of the
  climb step loop) and never while the robot stands on stair voxels (a climb that ended on a
  landing: in `g` its own position would not be traversable). On a landing half-way the stairs stay in view and the mode stays
  `s`; on the new floor it ends once the robot is away from the stairs. Where the stairs are is
  recorded while exploring in mode `g` by the instance map (O_stair) and the voxel map's stair
  voxels; M_s is collected in both modes. Earlier, `g -> s` also happened whenever at least 50
  stair pixels within 4 m were seen for 0.5 s, and `s -> g` only 4.5 m away from every mapped
  stair voxel: the robot was in mode `s` for 21 % of the time while exploring floors near stairs
  (v1 val, 140 episodes; 59 % of the time in `s`), not climbing. `mode_auto_enter:=true
  mode_exit_distance:=4.5` brings that back (`mode_enter_time`, `mode_min_stair_points`,
  `mode_enter_range`, `mode_exit_distance`). `[mode]` in the log,
  `voxel_map/mode` (latched String) and a text above the robot in RViz.
- **Remembered stairs**: stair points outside the window are kept (`stair_seen`) and re-tested
  whenever z_r moves to another layer: the camera cannot see the ~0.7 m in front of the robot, so
  the last steps and the edge of the upper floor, seen from below, were never seen again from
  close by and stayed obstacles at the top of the climb.

## Stair climb (`trav3d.py`, Sec. VI-B)

```bash
# terminal 1: stairs on the navmesh (Habitat's default climb), ground-truth classes
python habitat_server.py --scene ... --semantic --max-climb 0.2
# terminal 2: map 4 m high (two floors)
ros2 launch trav_exploration hm3d_trav_exploration.launch.py map_z_above:=4.0
# when the stairs are in view (mode s), start the climb
ros2 topic pub --once /voxel_map/climb std_msgs/msg/Empty "{}"
```
- **3-D path search** on M_trav (support voxels, inflated obstacles removed): neighbours within
  2 cells in xy (0.1 m, the paper's foothold-scale neighbourhood that bridges small FOV gaps) and
  5 layers (0.25 m, one step) in z; cost eq. 15-16. The same support graph of the whole map as
  for the paths on a floor (below), so in mode s the floors are connected over the stairs.
- **Climb** (stair transition stage, Sec. VI-A): `voxel_map/climb` switches to mode `s` (the
  paper's switch on a request to navigate to a stair entrance) and takes a fresh snapshot. Each
  stretch goes to a **semantic frontier F_s** on the staircase above the robot (within 4 m of the
  foot of the stairs, then of the robot; the first node of an open ATSP over them on the support
  graph), to the highest reachable node within 1 m of its voxels (the frontier itself, at the
  edge of the trusted stairs, lies in O_inf); when the robot is there (and the steps above have
  been seen from the new height, 1 s), the next one. Without an F_s above (the robot still far
  from the stairs, or at the top: there the upper floor is geometric) the stretch goes to the
  highest reachable node near the stair voxels (`SupportGraph.climb`), at the top edge ~1 m onto
  the upper floor. Done when neither exists. 00800: 13 stretches, 11 of them to an F_s, each
  ~0.4 m up. RViz "Publish Point" sends one path to the clicked voxel instead.
- **Following** (collision_guard_node, like the escape): Nav2's commands are ignored, the robot
  drives along the path's xy (Habitat's navmesh gives the height), the forward check uses mapped
  obstacles only (`voxel_map/obstacles`: occupied voxels outside T; a stair step that is only
  inflated next to the railing is not one). A stretch ends at its end, on its time limit, or after
  3 s without getting closer (replanned from there).
- **z_r** is the mapped surface under the robot (`VoxelMapper.support_height`, median of the top
  occupied voxels below the base within 0.15 m), not base_link minus an offset: the navmesh lies
  0.165 m above the ground floor of 00800, 0.055 m above the upper floor and 0.03-0.2 m above the
  steps. `visited` / `tight` cells are kept per layer, so the ground floor's free cells do not
  open the upper floor at the same xy. `trav_grid` (Nav2) is the slice at the current z_r and the
  layers right above / below it (a floor rarely lies on one voxel layer: one layer only cut 18 % of
  the ground floor and 60 % of the upper floor of 00810 out of the Nav2 grid).
- **Not yet**: going down.

## Floors: chain of exploration (Sec. VI-A, eq. 9)

```bash
python habitat_server.py --scene ... --semantic --max-climb 0.2 --start=-9.457,0.163,-0.633,-1.566
ros2 launch trav_exploration hm3d_trav_exploration.launch.py map_z_above:=4.0
```
(`--start`: with stairs on the navmesh the random start pose moves to a spot from which the
exploration ended at once; this is the usual start of 00800.)
- **F = frontiers of the current floor**: frontier_goal_node takes only clusters whose centroid
  lies within 0.3 m of z_r (`floor_band`).
- **O_stair**: with `segmentation:=sam3` the reliable stair clusters of the instance map
  (`instance_map/stair_landmarks`, below); with the ground-truth classes (no confidences) or
  `stair_landmarks:=voxel`, `voxel_map/stair_landmarks` (cyan spheres on
  `voxel_map/stair_landmark_marker`): connected groups of stair-labelled voxels (merged over
  0.15 m gaps) with at least 150 voxels (`stair_landmark_min_voxels`). Both give the entrance
  (lowest part) and the top.
- **A = O_stair once F is exhausted**: after the frontier ranks (or when the explorer rejected
  them all), frontier_goal_node takes the nearest landmark whose foot is on this floor and whose
  top is >= 0.4 m higher (`stair_lead_rise`: an instance-map stair cluster often holds only the
  lower steps, the flight seen from below splits into clusters; with 1 m the stairs of 00800
  were never climbed; the climb itself goes up as far as the support graph reaches), sends its foot on `voxel_map/climb_at` and waits. The voxel map switches
  to mode s (held until the climb is over) and sends one path on the support graph from the robot
  over the floor and up the stairs, then the next stretches. Then frontier_goal_node answers "no goal" and the explorer asks again
  (rank 0, its empty-goal retries) on the new floor, where the candidate pool is rebuilt (Sec.
  VI-A): stairs come in only once its frontiers are exhausted, toward a floor not explored yet.
  Stairs climbed are not left out by force; stairs given up (two failed climbs) are.
  (Sending the foot as a Nav2 goal failed: the 2-D explorer blacklisted reached goals, and new
  frontiers took the robot away again before it climbed.)
- **Path following** (guard): progress means getting further along the path (turning in place
  is not "no progress"); paths on one floor are driven at 0.3 m/s, on stairs at 0.15 m/s.

## Going down, targets on other floors, ATSP over O_tar / O_stair (Sec. IV-D, VI-A)

```bash
# start upstairs: the map must reach the floor below
ros2 launch trav_exploration hm3d_trav_exploration.launch.py map_z_below:=3.5 map_z_above:=1.5
```
- **Stair candidates** (frontier_goal_node `_stair_candidates`): up when a landmark's foot is on
  this floor and its top >= 0.4 m higher; down when it reaches up to within 1 m of this floor
  (its top may be a railing, or the flight split into clusters) and its foot is >= 0.4 m lower.
  The entrance on this floor is the foot (up) or the top's xy at z_r (down).
- **Explored floors**: a floor whose frontiers are exhausted is remembered; stairs leading to an
  explored floor (1-4.5 m that way) are not taken. Start upstairs: upper floor, down, ground
  floor, done (00800: down in 14 stretches, -0.06 -> -3.16 m, no climb back).
- **Going down** (`voxel_map/descend_at`): the climb mirrored in z (`SupportGraph.climb(down=True)`,
  `reach_near(lowest=True)`, F_s below the robot), the camera pitched down 30 deg (Sec. IV-D:
  descending stairs lie outside the forward view), stair points trusted down to 0.6 m below z_r.
  When nothing is reachable at first, the robot drives to the stair head and plans again from
  there; a stand must be >= 0.35 m away (xy) or the path follower counts it as reached on the
  step above.
- **Stalled descent** (not in the paper): when nothing lower is reachable even after the
  turn-around, the stretch is searched once more on a graph whose stair voxels are cut only by
  the O_inf of their own layer, without the one-cell margin (`stair_margin=False` in `SupportGraph.snapshot`, built by the map
  thread while descending) and link within `loose_stair_reach` (6 cells, 0.3 m). From above,
  SAM3 misses the nosings, their inflation leaves treads 1-2 cells wide and the margin removed
  them (00800: stopped at -2.36 m, the lower floor at -3.16 m).
- **Stepping off the last step** (not in the paper, `SupportGraph.step_off`): a climb that would
  end with the robot still on a stair voxel goes on to the cheapest node off the stairs 0.1-0.5 m
  below (above, going up) and 0.4-2 m away. The path follower counts a goal 0.1 m ahead as
  reached, and the robot stayed on the bottom step (00800: z_r -2.96 m, floor -3.16 m).
- **Stair landmark check** (not in the paper; the verification of a tentative hypothesis, Sec.
  V-A, by geometry): voxel_mapper takes the landmarks (`stair_check_input`, the instance map's in
  SAM3 mode) and republishes them on `voxel_map/stair_landmarks_checked` (frontier_goal's
  `stair_landmark_topic`) with `risers` and a `verdict`. In a 0.6 m corridor around foot -> top,
  top surfaces (an occupied voxel with two free above) are grouped by height (>= 10 cells a
  level); `risers` is the longest chain of levels each 0.08-0.27 m above the last
  (`VoxelMapper.stair_steps`). Verdict 1 with >= 2 risers (`stair_check_risers`), -1 when the
  raised part is observed (>= 60 top cells above the foot, `stair_check_min_top`) without such a
  chain (a threshold, a rug, a sofa SAM3 called stairs), 0 otherwise (seen from afar: kept).
  frontier_goal never climbs a landmark with verdict -1 (00495: a 0.15 m spot pair was climbed
  back and forth for 600 s).
- **Tentative stair landmarks** are offered only with gamma >= `landmark_tentative_min_conf`
  (0.3, instance_map_node): the negative evidence of eq. 6 (stairs in view, not detected) takes a
  cluster out of O_stair, not only out of the reliable ones (00495: gamma 0.16 was climbed).
- **The climb ends by the exit rule of mode s** (Sec. IV-A): on another level than the floor it
  left (>= `floor_min_change` 0.5 m) and farther than `mode_exit_distance` (1.0 m) from the stair
  region for `mode_exit_frames` (5) consecutive class frames, counted on every frame during the
  climb (`[climb] done ... m from the stairs for 5 frames`). A landing between two flights is next
  to the stairs and does not end it. Earlier the climb ended on >= 3 m^2 of flat floor connected
  to the robot (`floor_min_area`, 1 m^2 past the landmark's far end with a floor frontier), a
  criterion the paper does not have; at a stair head with little floor in view the robot kept
  chasing stair frontiers (00366).
- **A climb stretch not reached** (the path follower made no progress) is an execution failure
  (Sec. VI-C, eq. 17). On a flight v_front cannot be marked occupied: the voxel ahead is the next
  tread, a support surface in mode s, and cells near the robot's own path are protected from
  blocked marks (0 voxels marked in 00034). So the support-graph nodes 0.3-0.5 m ahead of the
  robot (0.2 m wide, up to 0.7 m above its support; `climb_block_radius` 0.1 m) are taken out of
  the snapshot for the rest of this climb, and the graph is rebuilt: the replan cannot pass the
  same spot again (00034: five stretches failed at one point, each with another end behind it;
  00366: the next path went back down the stairs and around). The stretch end is left out too.
- **Stepping onto the next step** (collision_guard, `step_clear` 0.25 m): on a 3-D path,
  obstacle points less than 0.25 m above the support are the next step (a nosing SAM3 did not
  label stairs) and do not stop the robot; at the top of the 00820 stairs one 0.12 m above z_r
  held it back 25 times.
- **Resuming a stopped climb** (frontier_goal_node `stair_resume`): a climb that ended on the
  stairs is continued the same way from where the robot stands at the next stair decision.
  From the bottom step the stairs no longer counted as leading down (foot < 0.4 m below), no
  candidate was left and the exploration ended in front of the stairs.
- **Targets on another floor** (eq. 9, O_tar first): a reliable target cluster whose lowest point
  is > 2 m above z_r (or whose highest is below the floor) sends the robot up (down) the stairs
  before the frontiers of this floor.
- **ATSP over the active set**: several reliable stair landmarks (or several targets on this
  floor) are ordered by an open ATSP over their entrances (goals) on J_geo; the first is taken.
  Without a reliable stair landmark the tentative one with the largest belief is verified.

## ObjectNav evaluation (`scripts/objectnav/`)

Episodes of the paper's benchmark (Sec. VII-A: Habitat Challenge 2023 setup, HM3D with 2000
episodes across 20 scenes and 6 categories = HM3D ObjectNav **v1** val; success distance 0.2 m,
sensor range 0-5 m, move_forward 0.25 m, turn 30 deg), run one after another and scored as Habitat
does. `--dataset v2` takes ObjectNav v2 val (36 scenes, 1000 episodes) instead. The agent is the challenge's: radius 0.18 m, height 0.88 m, RGB-D 640x480 at 0.88 m, hfov 79 deg
(habitat_server defaults). It slides along walls on contact: the challenge's allow_sliding False
is for its discrete agent; with continuous commands the agent was pinned on the stairs (up to
15949 collisions in an episode, `--no-sliding` to have it anyway).

```bash
# once: the episodes (260 MB)
mkdir -p ~/data/datasets/objectnav/hm3d && cd ~/data/datasets/objectnav/hm3d
curl -O https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2.zip
curl -O https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip
mkdir -p unz && cd unz && unzip ../objectnav_hm3d_v2.zip && unzip ../objectnav_hm3d_v1.zip
# 1. pick episodes (hsim-test env); the paper's benchmark: all 2000, interleaved over the scenes
python prepare_episodes.py --per-scene 0 --out ~/objectnav_runs/paper_hm3d_v1/episodes.json
python prepare_episodes.py --out ~/objectnav_runs/val/episodes.json --per-scene 2
python prepare_episodes.py --source cross --out ~/objectnav_runs/cross/episodes.json --per-scene 2
# 2. run them (system Python, ROS sourced; SAM3 and BLIP-2 servers up); resumable
python3 run_episodes.py --episodes ~/objectnav_runs/val/episodes.json
# 3. score (hsim-test env): results.csv + SR / SPL per floor split and category
python score_episodes.py --episodes ~/objectnav_runs/val/episodes.json
```
- **prepare_episodes.py**: `--source official` takes the dataset's episodes (start pose,
  category, geodesic distance, the view points of every goal object), spread over the
  categories (`--per-scene`, `--scenes`, `--categories`). Their closest goal is on the start floor
  in 998 of the 1000 val episodes, so `--source cross` makes episodes that need the stairs: the
  start is on a floor without any object of the category, the goals are reachable on the navmesh
  (35 in val with 2 per scene). The voxel map spans every floor of the building
  (`map_z_below` / `map_z_above` from the navmesh levels).
- **run_episodes.py**: per episode habitat_server (`--trajectory`: the agent pose in habitat
  coordinates every 0.2 s of sim time) and the launch with `target:=<category>=<prompt>` (tv
  for tv_monitor, potted plant for plant). The episode ends when frontier_goal_node reports the
  target reached (the stop action), when the exploration loop ended for good, or at
  `--max-actions` 500: the continuous motion counted in the benchmark's actions (distance / 0.25 m
  + rotation / 30 deg, from the trajectory); `--max-sim-time` 1200 s is a safety limit. Logs, trajectory and result.json go to `<out>/<episode id>/`. A model
  server error (SAM3 / BLIP-2, e.g. "CUDA context lost") stops the run.
- **score_episodes.py**: success = stop within 0.2 m (`--success-distance`; the paper's value, Habitat Challenge 2023 has 0.1) geodesic of a
  view point of a goal object (final pose snapped to the scene's navmesh); SPL with the episode's
  geodesic distance and the 3-D path length of the trajectory; DTG = distance to goal at the end.
- Differences from the paper's setup that remain: continuous velocity commands (Nav2 RPP)
  instead of the discrete actions (the 500-action limit is applied to their equivalent), the
  stop decided by frontier_goal_node (within 1 m of the instance points), no MP3D (needs the
  Matterport terms of use), ATSP by Held-Karp / 2-opt instead of LKH. v1 val has 399 episodes
  whose closest goal is on another floor (v2 val: 2).

## Instance map (`instance_map.py`, `instance_map_node.py`, Sec. V-A)

```bash
conda activate sam3 && python .../trav_exploration/scripts/sam3_server.py
ros2 launch trav_exploration hm3d_trav_exploration.launch.py segmentation:=sam3 target:=couch map_z_above:=4.0
```
- **Detections** d_q = (P_q, s_q, q): `sam3_server.py` returns, besides the class image of the
  semantic mode, every instance mask of the queries Q = Q_land U Q_obj ("stairs" and `target`)
  with its SAM3 score (>= `--detect-confidence` 0.3; the class image keeps 0.5).
  `semantic_seg_node` publishes them on `camera/detections` (`trav_interfaces/Detections`).
  `instance_map_node` pairs them with the depth frame of the same stamp, erodes the mask by 2 px
  (edges bleed onto the background) and back-projects it within 5 m.
- **Clusters** (eq. 4-6): same label and centroid distance < 1 m (`assoc_distance`), or point
  sets overlapping by >= 20 % of the smaller one: positive evidence, P_y <- D(P_y U P_q)
  (0.1 m voxels), n_y <- n_y + n_q, weighted confidence (eq. 5). Otherwise a new cluster. A
  cluster of a query that is in view (inside the image, within 5 m, not occluded in the depth
  image: at least 5 of its voxels) but got no detection receives zero-confidence evidence (eq. 6).
- **Use**: a cluster is reliable with gamma >= 0.5 (`min_confidence`), n >= 60 (`min_count`, objects;
  `landmark_min_count` 100, stairs) and detections in >= 5 frames (`min_detections`); launch arguments
  `instance_min_confidence`, `instance_min_count`, `instance_min_detections` (earlier 30 points and 2
  frames: a bar stool, an armchair and a couch seen in 2 frames became a chair / sofa / bed target); otherwise it stays tentative (grey on `instance_map/markers`, label with gamma
  and n). Reliable stair clusters are O_stair (`instance_map/stair_landmarks`), reliable target
  clusters O_tar (`instance_map/targets`).
- **Point clouds** (RViz `InstancePoints`, `InstanceDetections`, colour = RGB8): 
  `instance_map/points` holds P_y of every cluster (voxel centres; fields rgb = label colour,
  grey when tentative, id, gamma: "Color Transformer: Intensity", channel gamma colours by
  confidence); `instance_map/detections` the back-projected mask pixels P_q of the last frame
  (not downsampled; fields rgb, score), i.e. what SAM3 marked.
- **O_tar first** (eq. 9): with `target:=...`, a reliable target cluster on this floor (lowest
  point within -0.3..+2 m of z_r) is rank 0 for the explorer: the free trav_grid cell 0.35-0.65 m
  from the object with the lowest J_geo (pink cylinder on `frontier_goal/target_goal`); the
  frontier ranks move up by one. Within `success_distance` (1 m, xy) of the object the node
  answers "no goal" and the exploration ends with the target found.
- **Choices not in the paper**: n_q counts the downsampled points (same scale as the visible
  voxels of eq. 6); association also by overlap (a staircase or a sofa seen from different places
  has centroids metres apart); min_count; min_detections (00800: one SAM3 false positive,
  "refrigerator" 0.55 with 35 points, on the upper floor as the robot went down the stairs was
  reliable at once, never looked at again, and sent the robot back up from the ground floor);
  the thresholds above.
- **Seen in 00800** (SAM3): a leather recliner (not annotated in HM3D) was detected as "couch"
  (0.67) right at the start and reached; later misses lowered it to 0.50 (tentative). The
  staircase split into a lower cluster (entrance at the foot, reliable) and an upper one; the
  lower one led the climb.

## Spatial value map (`value_map.py`, Sec. V-B)

```bash
conda activate blip2 && python -u .../trav_exploration/scripts/blip2_server.py      # BLIP-2 ITM, port 5558
ros2 launch trav_exploration hm3d_trav_exploration.launch.py segmentation:=sam3 target:=bed map_z_above:=4.0
```
(`blip2` conda env: Python 3.12, torch 2.11 cu128 as the sam3 env, transformers; the model
`Salesforce/blip2-itm-vit-g` is downloaded on first start. `value_map:=false` turns it off.)
- **Relevance s^** per RGB image: `itm_node` sends the image to `blip2_server.py`, which returns
  the image-text matching probability for "Seems like there is a {target} ahead." (VLFM's prompt;
  the paper gives neither prompt nor head; `--head itc` gives the cosine similarity instead), on
  `camera/itm_score` (`trav_interfaces/ImageScore`).
- **Projection** (`voxel_mapper_node._on_itm`): onto the traversable voxels hit by the depth frame
  of the same stamp (within 5 m; the hit voxel or the one below), each with
  w = cos^2(pi theta / theta_FOV) of its horizontal angle to the camera heading (theta_FOV = 79 deg).
- **Fusion** (eq. 7): s <- (r s + w s^) / (r + w), r <- (r^2 + w^2) / (r + w), per voxel, stored
  sparsely (only seen voxels). `voxel_map/value` (RViz `VoxelValue`, rainbow 0-1): x, y, z, value,
  reliability, at 1 Hz.
- **On the image** (`camera/itm_image`, RViz `ITM`): each scored image tinted with the colour of
  its score s^ (the same rainbow, 0 red ... 1 magenta), each column as opaque as its weight w;
  the white curve is w across the image, the bar on the right marks s^. BLIP-2 gives one score
  per image, not per pixel: the tint shows how much of it each column passes on to the voxels.
- **Use** (frontier_goal_node, with a target, while O_tar is empty): semantic utility u_j = the
  maximum value within 1 m (`value_radius`, xy, same floor) of the frontier goal; the ATSP adds
  lambda_v (1 - u_j) to every cost of entering goal j (`value_weight` lambda_v = 3 m of J_geo;
  the paper only says low-value frontiers are down-weighted, high-value ones preferred). u_j is
  first rescaled over the candidates of the plan, (u - min) / (max - min) (`value_normalize`):
  BLIP-2 scores stay within ~0.0-0.4, so the raw term differed by ~1 m between the best and the
  worst frontier; rescaled, the lowest-value one costs the full lambda_v more. When the spread is
  below 0.05 (`value_min_spread`) there is no value term. The tour log shows the raw u of the
  first goal and the maximum.

## Semantic frontiers (`frontier3d.py`, `frontier_goal_node.py`, Sec. VI-A)

- **F_s / F_g**: a cluster with at least half of its voxels traversable only through M_s
  (`VoxelMapper.semantic_only_at`: in M_s, not in M_g, and not floor that joins M_g at s -> g;
  with the classes floor / rug / stairs: stair voxels) is a semantic frontier F_s, the others
  geometric frontiers F_g (`voxel_map/frontier_clusters` field `semantic`; F_s magenta on
  `voxel_map/frontier_centroids`). The paper gives no rule.
- **Stair frontiers**: the steps above the trusted window are obstacles until z_r rises, and
  their inflation covered the last trusted steps, so the staircase had no frontier at all. Stair
  voxels are now candidates even in O_inf, with their own boundary: a neighbour column with no T
  and no wall within +-0.25 m (one step) but with an unknown voxel or a stair voxel not trusted
  yet. The sides of the staircase (walls, the void seen next to a railing) are not frontiers.
- **J_front** (eq. 14): entering a candidate costs lambda_g (F_g, 0) or lambda_s (F_s, 1.0 m;
  values not in the paper) more in the ATSP: F_g softly first, F_s complementary.
- **Stair transition**: the climb goes from F_s to F_s (see "Stair climb").

## Traversable frontiers (`frontier3d.py`)

A voxel of M_trav is a frontier candidate if no M_trav voxel lies within K_h layers above it
(headroom) and a horizontal 4-neighbour column is not traversable. Candidates form clusters by
26-connectivity (voxels, centroid, AABB); small clusters are dropped. Updates are local: clusters
whose AABB meets the changed region B_m are removed and regrown (same result as a full
re-extraction). The neighbour column test uses T (M_g), not M_trav: with M_trav every voxel next
to an inflated wall would be a frontier and a closed room would never be explored out
(`frontier_boundary:=trav` gives that literal reading).

## Frontier order (`tour.py`, Sec. VI-A/B)

Without semantics the candidate pool (eq. 8) is X = {x_r} U F_g, so the active set (eq. 9) is the
geometric frontier clusters and the ATSP nodes (eq. 10) are the robot plus every reachable cluster.
- **Cost matrix** (eq. 11-13): c_rA(j) = J_geo(x_r, v_j) + lambda_g, C_AA(i, j) = J_geo(v_i, v_j) +
  lambda_g, and 0 back to the robot (c_Ar = 0): an open tour, the robot does not return.
  lambda_g is the same for every geometric frontier, so it does not change the order (eq. 14; it
  matters only with semantic frontiers).
- **J_geo** (eq. 15-16): shortest path on the 3-D support graph of the floor (see Path planning),
  from `voxel_map/geo_costs`; the same graph the robot's paths come from. Clusters without a
  path from the robot are left out (grey spheres on `frontier_goal/unreachable`). The 2-D
  trav_grid graph is used only if that service does not answer (logged).
- **Node of a cluster**: its goal position (the 2D placement below), so the cost is the path the
  robot will drive.
- **Solver**: exact (Held-Karp) up to 12 clusters, nearest neighbour + 2-opt / or-opt above.
  The goal is the first node of the tour (orange line on `frontier_goal/tour`).
- **Dormant set**: when the robot stands at the goal last sent (within 0.4 m, checked at the
  request after each navigation) and that frontier cluster is still there, that one cluster is
  dormant; other clusters near the robot are not touched. The flag is the cluster's visited flag
  in the voxel map (`voxel_map/frontier_dormant` -> frontier3d) and is sent back with the
  clusters (`id`, `dormant` fields). It stays while the cluster's voxels stay: a cluster grown again
  by a map update keeps it when >= 80 % of its voxels were dormant (`dormant_keep_share`), and loses
  it otherwise, i.e. when the space there changed (the paper resets visited flags with the
  clusters an update removes). Frontiers along a table edge are like that: the floor under the
  table is hidden by the table top, never enters M_g, and the edge of M_trav stays a frontier
  however often the robot visits it. Blue spheres on `frontier_goal/unreachable`.
- **Repeated failures**: when the goal last sent was not reached (Nav2 gave up, the guard held the
  robot) or the explorer turned it down (at the robot, no path), its
  cluster gets a failed attempt (`voxel_map/frontier_failed`, field `fails`); at
  `frontier_max_attempts` (2) it is dormant, with the same release (the count goes when the
  cluster's voxels change). Goals behind a wall that the 3-D graph reaches through a gap the
  body does not fit were tried again and again before.
- **J_geo to a cluster and its goal** (the paper's ATSP nodes): `voxel_map/frontier_targets`
  (trav_interfaces/FrontierTargets) gives, per frontier cluster of the latest graph snapshot,
  J_geo(x_r, F_i) = the minimum path cost from the robot over the cluster's own voxels (frontier
  voxels lie on M_trav, so they are graph nodes) and that voxel, v_target. A cluster is reachable
  when J is finite, and its goal is snapped to a clear cell from v_target, not from the centroid
  (which may lie inside or behind a wall, or off a curved cluster). Clusters newer than the
  snapshot use the check below.
- **No goals behind walls**: a frontier goal counts as reachable only if the support-graph node
  it maps to lies within 0.2 m (`geo_target_radius`; goals are snapped onto free trav_grid
  cells, whose own node is within a cell), and the Nav2 planner has a tolerance of 0.15 m. With
  0.5 m in both, a goal behind a wall took a node / a plan end on this side of it, the frontier
  behind counted as reachable and the explorer accepted it.
- **Next ranks**: when the explorer rejects a goal (at the robot, no path) it asks for
  the next rank; rank k is the first node of the tour solved again without the first k nodes (the
  paper's dormant set: the frontier is left out and the planner replans).

Compared with the 2D score (0.5 / distance + 0.5 * size), the tour looks at all frontiers at
once: it finishes a room's frontiers before leaving it instead of jumping to a bigger frontier
elsewhere and coming back. `goal_order:=score` restores the 2D ranking.

## Path planning on the 3-D support graph (`trav3d.SupportGraph`, Sec. VI-B)

As in the paper, paths and their costs come from the traversable support voxels, not from a 2-D
planner: Nav2's planner_server is not started, `voxel_mapper_node` serves `compute_path_to_pose`
(called by bt_navigator at 1 Hz and by the explorer's path check), and Nav2's controller (RPP)
follows the path with its recoveries. `voxel_map/geo_costs` gives J_geo for the tour, and the
stair climb is a path on the same graph.
- **One graph of the whole map** ("connectivity across stairs and floors"): every M_trav voxel
  is a node, linked within 2 cells in xy and 5 layers in z; step cost r |dk| + lambda_esdf *
  Phi(v_j) (eq. 15-16); stair nodes reach 4 cells (0.2 m) in xy: from above, a SAM3 mask
  misses the nosings, they stay occupied, their inflation covers most of the tread below, and
  1-2 cells of each tread were left 0.15-0.2 m apart (00800 going down: the stairs were not
  connected to the floor with 2-cell neighbours). In mode s the stairs are in it and connect the floors; in mode g they are
  obstacles (eq. 1), as in the paper. Paths and J_geo to frontier goals start and end at
  `floor_z`, and a point snaps to a node within 0.15 m of its height (with every floor in the
  graph, a point could snap to a stair step or another surface above its floor; in the first run
  with one graph every frontier of the upper floor of 00800 came out unreachable, with the limit
  and the goals sent at floor_z (they lie on trav_grid) J_geo was never all inf after the climb). Frontier selection stays per floor (eq. 9). Built on a planning thread from a
  snapshot the map thread takes every second (`graph_snapshot_period`); the edges are built from
  the node list (00800, two floors: ~20 k nodes, ~250 k edges, ~50 ms).
  Before, the graph held the current floor only (0.1 m around z_r, no stair voxels) and the climb
  searched a second, local graph: with the first steps in a +-0.3 m floor band a path to a
  frontier at the foot of the stairs led up them and z_r followed; `floor_z` now keeps the floor.
- **Changes to M_trav for Nav2** (not in the paper): voxels 4-adjacent to O_inf (also in the layers
  right above / below) are not nodes, a one-cell margin; Nav2's controller treats a path cell next
  to O_inf as a collision. Cells the robot centre passed are nodes (as in trav_grid); cells marked
  too narrow are not.
- **d_obs** (eq. 16): per layer, the xy distance to O_inf of that layer and the ones right above /
  below (or a too-narrow cell) + the O_inf radius, the distance Nav2 sees. Measured to the occupied voxels within 0.3 m of the floor it
  missed walls and furniture at the robot's height, the paths ran along the edge of O_inf and RPP
  aborted in every corner ("collision ahead", 28 aborts per run); with this d_obs 0 on the
  ground floor of 00810.
- **Goal**: the reachable node nearest to the goal within 0.5 m (`plan_goal_radius`).

## Mode switches without holes (deviations, see `_source`)

- Mode s: T = M_g U M_s (eq. 1 takes one source). M_g keeps being ray-cast in mode s while the
  robot is on a floor; only on / while climbing the stairs it is not (support plane in mid-air).
- M_g leaves out support cells the camera has seen through at floor height (the voxel and the
  two below it free, log-odds <= 2 l_miss), and drops M_g voxels once they are seen through.
  2-D ray casting assumes floor between the robot and the obstacle points; next to the
  stairwell of the upper floor (00800) it filled the void above the stairs at the upper floor's
  height, with frontiers at its edges the robot went for again and again (and drove down the
  stairs toward them). Cells never observed are still filled, as in the paper.
- Back to mode g, the floor of M_s (without stairs) joins M_g. Before, the floor explored in mode
  s (the robot stays in s up to 4.5 m from the stairs) dropped out at the switch as holes and
  frontiers (frontier voxels 24 -> 139 in 00810). So the modes differ only in the stairs.
- Leaving mode s counts only stair voxels within 1 m of z_r: on the upper floor the staircase
  below is close in xy, and the mode stayed s on the whole upper floor.
- `floor_z` (voxel_mapper_node): the floor height the graph, trav_grid and the planner use. It
  follows z_r while climbing and otherwise only within 0.15 m of its height on arrival on the
  floor (start or end of a climb): at the foot of the stairs z_r read the first step and the
  floor's frontiers were no longer "on this floor"; on the upper floor of 00810 z_r crept down
  in two 0.125 m steps with the base level, a floor_z that followed each step built the graph
  25 cm below the floor and exploration ended with every frontier unreachable.

## Collision guard mode (`guard_mode`)

`guard_mode:=path_only` (default): collision_guard_node only follows the 3-D paths of a stair
climb (Nav2 cannot drive the stairs); Nav2's commands pass through unchanged. A robot blocked on
a floor is left to the paper's execute-review: the stuck event (eq. 17) marks v_front occupied,
the frontier goal gets a failed attempt and is canceled at once, the tour is solved again, and
after two failed attempts the frontier is dormant. `guard_mode:=full` (before): also stops the
robot 0.1 m before contact, closes the cells ahead too narrow for the body (`mark_tight`), and
on two Nav2 failures at one spot blocks the way ahead and backs the robot out along its trail.
The 3-D path follower stops on map obstacles (`voxel_map/obstacles`) in both modes.

## Geometric treads (`VoxelMapper.geometric_treads`)

Where a staircase turns, SAM3 often leaves the treads and risers of the turn unlabelled. They
stayed occupied and outside T, so they were inflation sources (S_inf = O minus T); their O_inf cut
the support graph between the trusted steps and `voxel_map/obstacles` held the robot in the
collision guard (13 of 15 dev climb stops had 90-310 such voxels within 0.6 m). While climbing
(mode s, every 3rd depth frame, `geometric_treads`), a geometric fallback after the paper's
step-height reasoning adds them to M_s: an occupied voxel that is a top surface with n_h layers of
headroom, reached from the robot's support over surfaces at most 0.25 m apart in height and one
cell apart in xy, within 2 m of the robot and within 0.5 m of stair-labelled voxels (a sofa or an
ottoman beside the floor is never one), or within 1.5 m of the stairs the planner asked to climb or
descend (`climb_at` / `descend_at`, an instance-map landmark: going down from a stair head the voxel
map held no stair label at all, and the descent ended at once); the occupied voxels under it down
to one step (its riser) go with it. Treads with another surface one step (0.1-0.25 m) higher or lower within
0.4 m are also stair voxels (the climb planner goes over nodes near stairs); the floor reached at
the end is not, and stays unlabelled for the arrival test (labelling tread patches smaller than a
floor labelled the floor below the stairs too, patch by patch as it came into view, and the robot
never arrived: 00847). In T they are no inflation sources and no guard obstacles, so the guard needs no
exception on the stairs. Replayed on the 15 dev climb-stop dumps (going up), 8 got a path to
higher stair nodes (+0.2 to +0.85 m).

## Blind spot and look-around (`frontier_goal_node`)

- **Blind-spot frontiers** (Sec. IV-D): a frontier within `d_blind` (1.5 m) of the robot is looked
  at once with the camera pitched down 30 deg (`camera/pitch_cmd`, habitat_bridge) for 2 s before
  it can become dormant. The ones inside the camera's horizontal FOV during the look (79 deg,
  `camera_hfov`, 5 deg margin) that are still there afterwards are irreducible and go to the
  dormant set at once (`look_demote`; a gap under furniture, a narrow slot the camera cannot see
  into from anywhere the robot can stand). Frontiers outside the view were not looked at and stay.
  Before, they stayed candidates until the robot stood at their goal: in 110 multi-floor episodes
  only 68 of 1245 looked-at frontiers were cleared by the look, the rest were driven to one by
  one. The flag goes when the voxels of that cluster change. d_blind is not given in the paper
  (1.5 m: where the level camera starts to see the floor).
- **Virtual ground plane Pi_vg** (Sec. IV-D, last paragraph; `VirtualGround`): Pi_vg lies
  `vg_depth` (0.3 m) below z_r; when more than `vg_n_down` (150) depth points of a frame lie
  below it within `vg_radius` (2.5 m, xy) of the robot (a staircase going down, the storey below
  seen past the stair head), voxel_mapper_node pitches the camera down (`descend_pitch`, 30 deg),
  and levels it again after `vg_hold` (3 s) without (`[Pi_vg]` in the log, `Pi_vg N below` in
  the `[voxel]` line). Not during a climb, which sets the camera itself (down while descending).
  The paper gives no values for N_down or the plane's height. `virtual_ground:=false` to disable.
- **Look-around** (not in the paper): at the start and on arriving on a new floor the robot turns
  in place once before the first plan (`look_around`, `look_around_speed` 0.6 rad/s). Without it a
  frontier at the edge of the first view, behind the robot, went dormant unseen and the
  exploration of 00810 ended after 2 goals.

## Goal placement and exploration loop (from the 2D stack)

Where to send the robot is ported unchanged from `frontier_exploration`: frontiers closer than
0.4 m skipped, the goal moved to the nearest free cell with `goal_clearance` (0.18 m to O_inf =
0.4 m to real obstacles, as in the 2D HM3D launch).

The exploration loop is `explore_executor_node` (this package), the loop of the 2D
`frontier_exploration_node` (left unchanged for the 2D stack) without its blacklist: it asks
`frontier_pose` rank after rank, turns down a goal within 0.2 m of the robot or without a path,
sends the goal to Nav2 (`navigate_to_pose`, 75 s per goal) and ends when no goal is left. The 2D
loop put every failed **or canceled** goal on a list by (x, y) within 0.5 m, without z: a goal
canceled by the replan on one floor turned down a different frontier cluster 0.4 m away in x-y on
the floor above (00891: the new floor then had "no frontier left" and the robot went back to the
stairs). Failures are kept in 3-D instead: the blocked voxels of execute-review (eq. 17) and the
failed attempts / dormant flag of the frontier cluster. The launch still restarts the loop when it
ends, `explore_restarts` times (default 2, `explore_restarts:=0` to disable), for frontiers whose
dormant flags were released by then.

## Parameters not given in the paper

| parameter | default |
|---|---|
| `resolution` | 0.05 m (0.1 m closed 0.5 m doorways) |
| log-odds hit / miss / l_occ / clamp | 0.85 / -0.4 / 0.7 / [-2, 3.5] |
| `h_min`, `h_max` | 0.15 m (launch), 0.88 m (Habitat ObjectNav agent height) |
| `robot_radius` (inflation, xy) | 0.15 m (launch `inflation_radius`; below the 0.18 m navmesh agent of Habitat ObjectNav, as 0.22 / 0.25 before: runs before 2026-11-16 used a 0.25 m x 0.6 m agent that slides along walls) |
| `max_range` | 5.0 m (paper's simulation depth range) |
| `pixel_stride` | 4 (depth pixels used per frame) |
| `size_xy`, `z_below`, `z_above` | 30 m, 0.25 m, 2.0 m (map extent around the start) |
| `frontier_headroom_layers` (K_h) | robot height in layers (6) |
| `lambda_g` (J_front) | 0 (does not change the order with geometric frontiers only) |
| `lambda_esdf`, `d_safe` (eq. 15-16) | 0.3, 0.6 m on the support graph (paths keep to the middle of a passage, as Nav2's inflation cost did; also on the stairs) |
| `d_blind`, look-down pitch / time | 1.5 m, 30 deg (the paper's simulated look_down), 2 s |
| `tour_exact_max` | 12 clusters (Held-Karp above that is too slow; heuristic) |
| `frontier_min_size` | 7 voxels: smaller clusters are noise. M_trav is shrunk by the robot radius at doorways, a 0.85 m door leaves a ~0.3 m frontier (6 voxels in a row) that 7 may drop (was 3) |

Topics (map frame): `voxel_map/occupied` (grey, without M_trav and ceiling), `voxel_map/traversable`
(M_trav without frontiers), `voxel_map/frontiers` (red), `voxel_map/frontier_centroids`,
`voxel_map/frontier_clusters` (centroid + size), `voxel_map/inflated`, `voxel_map/trav_grid`,
`voxel_map/blocked`, `voxel_map/semantic` (M_s without stairs, purple, off in RViz),
`voxel_map/stairs` (orange), `voxel_map/mode`, `voxel_map/mode_marker`.

Camera in Habitat (`habitat_server.py` defaults): 640x480 RGB-D at 0.88 m above the floor (the server
measures the navmesh-to-floor offset at the start and lowers the camera by it), hfov 79 deg, as in
the paper's simulation setup.

Tests (no ROS): `python3 -m pytest trav_exploration/test`
