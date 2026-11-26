# floor_switch

Floor-switch model on top of `trav_exploration` (the TravExplorer port): predicts p, the probability
that the target is on the robot's current floor, and decides Stay / Switch.

- `floor_switch/features.py`: floor-state features (15 numbers per decision time). Shared by the
  offline extractor and, later, the online node.
- `scripts/make_labels.py`: labels from the Habitat trajectory and the recomputed navmesh
  (Habitat env, e.g. `conda activate hsim-test`).
- `scripts/make_features.py`: features of the collected episodes from their JSONL records.
- `floor_switch/rooms.py`: room-type histogram (9) of the current floor: 9-channel room map on the
  voxel map grid (0.05 m), FloorTracker floors, frames with a top-2 similarity margin below
  `margin` discarded, softmax temperature T on the fused raw similarities.
- `scripts/render_frames.py`: renders the camera views again at every decision time (pose and
  pitch from trajectory.txt, camera height from server.log; Habitat env).
- `scripts/embed_frames.py`: BLIP-2 ITC image embeddings (32 x 256) and raw room similarities
  (BLIP-2 env).
- `scripts/make_room_features.py`: room histograms per margin and temperature.
- `scripts/train_baseline.py`: logistic / MLP baselines, scene-grouped 5-fold CV (needs torch,
  e.g. `conda activate blip2`); `--rooms <dir>` adds the room histogram (24 features).

- `scripts/calibrate.py`: per-category calibration (plan sec. 8.5) and its reliability check
  (ECE, Brier), with Platt scaling (slope + offset per category, fitted on the other folds) as
  the fallback; writes the constants and the out-of-fold calibrated p.
- `scripts/decision_constants.py`: cost guard constants from the records (actions per metre,
  climb actions, margin with Kaplan-Meier on censored failures).
- `scripts/select_rule.py`: replays the v0 switch rule on the out-of-fold p and chooses X and tau
  (saved actions, wrong-switch rate, net), with the B share 15-30 % sensitivity.

- `scripts/export_model.py`: the five CV fold models (MLP32, rebuilt with the CV's folds, seeds and
  settings, each checked against its out-of-fold logits in `oof.csv`) with the Platt and rule
  constants -> `online_model.json`; online logit = the mean of the five (the Platt constants were
  fitted on these models' out-of-fold logits). Numpy forward checked against torch; sim_dev check.
- `floor_switch/model.py`, `floor_switch/rule.py`: the p model in numpy and the switch rule
  (history, cost guard), shared by the node and `select_rule.py`.
- `floor_switch/floor_switch_node.py` (online, v0): reads the records of voxel_mapper /
  instance_map / frontier_goal on `record/*` (trav_exploration recorder, `record_publish`), counts
  actions from TF, sends a frame per decision time to blip2_server (`itc` request) for the room map,
  and publishes p and `ready` on `floor_switch/state`. frontier_goal_node switches at its decision
  times (rank-0 requests, every 10 s) with `floor_switch_enable`: the floor gets a defer flag,
  reliable stairs to a floor neither explored nor deferred, no climb abort for new frontiers.
  Launch: `hm3d_trav_exploration.launch.py floor_switch:=off|shadow|on` (shadow: p recorded in
  `<record_dir>/floor_switch.jsonl`, exploration unchanged).

v0 choices (train scenes, `~/objectnav_runs/floor_switch/p_model/catw/`):
- p model: MLP32 (x 24 + one-hot), trained with per-category episode weights (`--cat-weights`;
  `--mlp-epochs 200 --mlp-lr 3e-3 --mlp-wd 1e-3`); online: the mean logit of the five fold models.
- calibration: Platt, p = sigmoid(0.251 x logit + offset[category]) (`calibration.json`); the
  sec. 8.5 constant shift leaves ECE 0.131 (the logits are ~4x too confident), Platt 0.050.
- rule: switch when the last 45 p (1 s) are all below 0.80, not while tracking a target, and only
  if actions left >= 7.23 x stair distance (m) + 96 (climb p75) + 75 (margin)
  (`decision_constants.json`, `rule_selection.json`).

Room histogram setting in use: margin 0.075, T 0.1 (`rooms/m0.075_T0.1`), the best of
margin {0, 0.02, 0.045, 0.075} x T {0.01, 0.02, 0.05, 0.1} in CV (differences within the fold spread).

Data: `~/objectnav_runs/collect_v01` (episodes), outputs in `~/objectnav_runs/floor_switch`.
