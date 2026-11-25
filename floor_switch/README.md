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

Room histogram setting in use: margin 0.075, T 0.1 (`rooms/m0.075_T0.1`), the best of
margin {0, 0.02, 0.045, 0.075} x T {0.01, 0.02, 0.05, 0.1} in CV (differences within the fold spread).

Data: `~/objectnav_runs/collect_v01` (episodes), outputs in `~/objectnav_runs/floor_switch`.
