# floor_switch

Floor-switch model on top of `trav_exploration` (the TravExplorer port): predicts p, the probability
that the target is on the robot's current floor, and decides Stay / Switch.

- `floor_switch/features.py`: floor-state features (15 numbers per decision time). Shared by the
  offline extractor and, later, the online node.
- `scripts/make_labels.py`: labels from the Habitat trajectory and the recomputed navmesh
  (Habitat env, e.g. `conda activate hsim-test`).
- `scripts/make_features.py`: features of the collected episodes from their JSONL records.
- `scripts/train_baseline.py`: logistic / MLP baselines, scene-grouped 5-fold CV (needs torch,
  e.g. `conda activate blip2`).

Data: `~/objectnav_runs/collect_v01` (episodes), outputs in `~/objectnav_runs/floor_switch`.
