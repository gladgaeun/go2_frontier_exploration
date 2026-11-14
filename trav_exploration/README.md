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
```
Arguments: `explore:=false` (map + Nav2 only, send goals from RViz), `rviz:=false`,
`frontier_boundary:=trav` (literal frontier boundary test, see below), `goal_order:=score`
(the 2D frontier ranking instead of the ATSP tour).

```
habitat_bridge (RGB-D, odom, TF)
  -> voxel_mapper_node: voxel map, M_trav, 3D frontier clusters
  -> frontier_goal_node: frontier_pose service (rank-th best goal)
  -> frontier_exploration_node (2D stack, unchanged): goal loop, blacklists, termination
  -> Nav2 without planner_server: global path from voxel_mapper_node's compute_path_to_pose
     (3-D support graph), controller / recoveries on voxel_map/trav_grid (config/nav2_trav.yaml)
```

| file | role |
|---|---|
| `voxel_mapper.py` | voxel map: occupancy, M_g (2-D ray casting), M_s + stair voxels, incremental inflation, M_trav, stuck detector |
| `semantic.py` | traversable classes of the segmentation image, g/s mode switch with hysteresis |
| `semantic_seg_node.py` | ROS node: camera/image_raw -> sam3_server -> camera/semantic/image_raw |
| `trav3d.py` | 3-D path search on support voxels: one support graph of the map (planning, J_geo, stair climb) |
| `scripts/sam3_server.py` | SAM3 text-prompt segmentation server (sam3 conda env, ZMQ) |
| `frontier3d.py` | traversable frontiers: candidates, 26-connected clusters, incremental update |
| `tour.py` | frontier order: open ATSP over {x_r} U F_g with traversable path costs (Sec. VI-A/B) |
| `goal_selection.py` | goal placement (and the 2D ranking for `goal_order:=score`), ported from the 2D detector |
| `voxel_mapper_node.py` | ROS node: depth + TF in, map / frontier topics out |
| `frontier_goal_node.py` | ROS node: `frontier_pose` service (same contract as `classical_frontier_detector`) |
| `config/nav2_trav.yaml` | Nav2 with costmaps from `voxel_map/trav_grid` (controller, recoveries) |
| `config/bt_navigate_*.xml` | Nav2 behaviour trees without the planner_server's global costmap |
| `../trav_interfaces` | `GeoCosts.srv` (J_geo between points on the support graph) |
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
- **Mode switch** (hysteresis, values not in the paper), checked on every camera frame (20 Hz;
  counting stair pixels is cheap, only every 0.5 s a frame goes into the map): `g -> s` when at
  least 50 stair pixels within 4 m of the camera are seen for 0.5 s without a frame missing them
  (`mode_enter_time`, `mode_min_stair_points`, `mode_enter_range`); `s -> g` when the robot stays
  farther than 4.5 m from every mapped stair voxel for 1 s (`mode_exit_time`, `mode_exit_distance`).
  The paper's N_enter / N_exit frames are kept as durations, since the node drops frames when busy.
  Entering by sight within 4 m (from the camera) and leaving by distance beyond 4.5 m (xy) keeps a
  robot that sees stairs from afar from switching every few seconds. (With the check only on map frames, 3 m and 10
  frames, the switch lagged the view by ~5 s.) `[mode]` in the log, `voxel_map/mode` (latched
  String) and a text above the robot in RViz.
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
- **Climb**: `voxel_map/climb` switches to mode `s` (the paper's switch on a request to navigate
  to a stair entrance), takes a fresh snapshot and sends a path to the highest reachable node near
  the stair voxels (`SupportGraph.climb`, within 4 m of the foot of the stairs, then of the robot);
  when the robot is there (and the steps above have been seen from the new height, 1 s), the next
  one, until no higher stair voxel is reachable. At the top edge the last stretch goes ~1 m onto
  the upper floor. RViz "Publish Point" sends one path to the clicked voxel instead.
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

## Floors: chain of exploration (Sec. VI-A, eq. 9 without target objects)

```bash
python habitat_server.py --scene ... --semantic --max-climb 0.2 --start=-9.457,0.163,-0.633,-1.566
ros2 launch trav_exploration hm3d_trav_exploration.launch.py map_z_above:=4.0
```
(`--start`: with stairs on the navmesh the random start pose moves to a spot from which the
exploration ended at once; this is the usual start of 00800.)
- **F = frontiers of the current floor**: frontier_goal_node takes only clusters whose centroid
  lies within 0.3 m of z_r (`floor_band`); dormant frontiers are matched per floor too.
- **O_stair**: `voxel_map/stair_landmarks` (cyan spheres on `voxel_map/stair_landmark_marker`):
  connected groups of stair-labelled voxels (merged over 0.15 m gaps) with at least 150 voxels
  (`stair_landmark_min_voxels`), with the entrance (lowest part) and the top. A simplified
  instance map: the paper fuses detections with confidences (eq. 4-6); here a group counts once
  enough of it has been seen.
- **A = O_stair once F is exhausted**: after the frontier ranks (or when the explorer rejected
  them all), frontier_goal_node takes the nearest landmark whose foot is on this floor and whose
  top is >= 1 m higher, sends its foot on `voxel_map/climb_at` and waits. The voxel map switches
  to mode s (held until the climb is over) and sends one path on the support graph from the robot
  over the floor and up the stairs, then the next stretches. Then frontier_goal_node answers "no goal": the exploration loop ends and the launch
  restarts it (empty blacklist) on the new floor. Stairs climbed or given up are not tried again.
  (Sending the foot as a Nav2 goal failed: the explorer blacklists reached goals, and new
  frontiers took the robot away again before it climbed.)
- **Path following** (guard): progress means getting further along the path (turning in place
  is not "no progress"); paths on one floor are driven at 0.3 m/s, on stairs at 0.15 m/s.

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
- **Dormant set**: a frontier that is still there when the robot stands at its goal (within
  0.4 m, checked at the request after each navigation) is left out from then on, also after the
  explorer restarts. Frontiers along a table edge are like that: the floor under the table is
  hidden by the table top, never enters M_g, and the edge of M_trav stays a frontier however often
  the robot visits it; the restarts sent it round them again and again. Blue spheres on
  `frontier_goal/unreachable`. (The paper puts frontiers with repeated failures there; failed
  goals are still retried by the explorer restarts.)
- **Next ranks**: when the explorer rejects a goal (blacklisted, reached, no Nav2 path) it asks for
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
  Phi(v_j) (eq. 15-16). In mode s the stairs are in it and connect the floors; in mode g they are
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

## Blind spot and look-around (`frontier_goal_node`)

- **Blind-spot frontiers** (Sec. IV-D): a frontier within `d_blind` (1.5 m) of the robot is looked
  at once with the camera pitched down 30 deg (`camera/pitch_cmd`, habitat_bridge) for 2 s before
  it can become dormant; still there afterwards, it is irreducible (dormant). d_blind is not
  given in the paper (1.5 m: where the level camera starts to see the floor).
- **Look-around** (not in the paper): at the start and on arriving on a new floor the robot turns
  in place once before the first plan (`look_around`, `look_around_speed` 0.6 rad/s). Without it a
  frontier at the edge of the first view, behind the robot, went dormant unseen and the
  exploration of 00810 ended after 2 goals.

## Goal placement and exploration loop (from the 2D stack)

Where to send the robot is ported unchanged from `frontier_exploration`: frontiers closer than
0.4 m skipped, the goal moved to the nearest free cell with `goal_clearance` (0.18 m to O_inf =
0.4 m to real obstacles, as in the 2D HM3D launch). The exploration loop is the 2D
`frontier_exploration_node`.
It blacklists a goal after one failure, so a reachable frontier that Nav2 once gave up on (in a
narrow passage) ends exploration early; the launch restarts the loop with an empty blacklist
when it ends, `explore_restarts` times (default 2, `explore_restarts:=0` to disable). After a
real completion this costs one or two rounds of retrying the frontiers that cannot be reached.

## Parameters not given in the paper

| parameter | default |
|---|---|
| `resolution` | 0.05 m (0.1 m closed 0.5 m doorways) |
| log-odds hit / miss / l_occ / clamp | 0.85 / -0.4 / 0.7 / [-2, 3.5] |
| `h_min`, `h_max` | 0.1 m, 0.6 m |
| `robot_radius` (inflation, xy) | 0.22 m (launch; Nav2 radius of the 2D stack, below the 0.25 m navmesh agent) |
| `max_range` | 5.0 m (paper's simulation depth range) |
| `pixel_stride` | 4 (depth pixels used per frame) |
| `size_xy`, `z_below`, `z_above` | 30 m, 0.25 m, 2.0 m (map extent around the start) |
| `frontier_headroom_layers` (K_h) | robot height in layers (6) |
| `lambda_g` (J_front) | 0 (does not change the order with geometric frontiers only) |
| `lambda_esdf`, `d_safe` (eq. 15-16) | 0.3, 0.6 m on the support graph (paths keep to the middle of a passage, as Nav2's inflation cost did; also on the stairs) |
| `d_blind`, look-down pitch / time | 1.5 m, 30 deg (the paper's simulated look_down), 2 s |
| `tour_exact_max` | 12 clusters (Held-Karp above that is too slow; heuristic) |
| `frontier_min_size` | 3 voxels (0.15 m): M_trav is shrunk by the robot radius at doorways, a 0.85 m door leaves a ~0.3 m frontier |

Topics (map frame): `voxel_map/occupied` (grey, without M_trav and ceiling), `voxel_map/traversable`
(M_trav without frontiers), `voxel_map/frontiers` (red), `voxel_map/frontier_centroids`,
`voxel_map/frontier_clusters` (centroid + size), `voxel_map/inflated`, `voxel_map/trav_grid`,
`voxel_map/blocked`, `voxel_map/semantic` (M_s without stairs, purple, off in RViz),
`voxel_map/stairs` (orange), `voxel_map/mode`, `voxel_map/mode_marker`.

Camera in Habitat (`habitat_server.py` defaults): 640x480 RGB-D at 0.88 m above the floor (the server
measures the navmesh-to-floor offset at the start and lowers the camera by it), hfov 79 deg, as in
the paper's simulation setup.

Tests (no ROS): `python3 -m pytest trav_exploration/test`
