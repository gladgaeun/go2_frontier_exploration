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

v0 choices (train scenes, `~/objectnav_runs/floor_switch/p_model/catw/`):
- p model: MLP32 (x 24 + one-hot), trained with per-category episode weights (`--cat-weights`).
- calibration: Platt, p = sigmoid(0.251 x logit + offset[category]) (`calibration.json`); the
  sec. 8.5 constant shift leaves ECE 0.131 (the logits are ~4x too confident), Platt 0.050.
- rule: switch when the last 45 p (1 s) are all below 0.80, not while tracking a target, and only
  if actions left >= 7.23 x stair distance (m) + 96 (climb p75) + 75 (margin)
  (`decision_constants.json`, `rule_selection.json`).

Room histogram setting in use: margin 0.075, T 0.1 (`rooms/m0.075_T0.1`), the best of
margin {0, 0.02, 0.045, 0.075} x T {0.01, 0.02, 0.05, 0.1} in CV (differences within the fold spread).

Data: `~/objectnav_runs/collect_v01` (episodes), outputs in `~/objectnav_runs/floor_switch`.
