# Semantic mode of TravExplorer (Sec. III-A): which segmentation classes are traversable, and the
# switch between the geometric (g) and semantic (s) traversability modes.
#
# Class image (habitat_server --semantic now, SAM3 later): one code per RGB pixel.
#   0 other, 1 floor, 2 rug (rugs, mats, carpets), 3 stairs
# Traversable classes (M_s): floor, rug, stairs (the paper's "floor, stair, and rug").
#
# Mode switch. Mode s is the mode of the stair transition (Sec. IV-A / VI-A): the planner switches
# to it when it decides to take the stairs (request('s'), voxel_mapper_node._start_climb) and back
# to g when the robot is off the stairs on a floor (request('g'), _end_climb). Recording where
# stairs are while the robot explores a floor in mode g is the instance map's job (O_stair) and the
# voxel map's stair voxels, not the mode's.
#   g -> s : request('s') only. With auto_enter (the earlier behaviour, off by default) also stairs
#            seen for enter_time without a frame missing them (at least min_stair_points stair
#            pixels within enter_range of the camera): the robot was in mode s for 21 % of the
#            time while exploring floors near stairs (v1 val, 140 episodes), not climbing.
#   s -> g : exit_rule='distance' (default, the paper, Sec. IV-A: "switches back to geometric mode
#            only after the robot remains sufficiently far from the stair region for N_exit
#            consecutive frames"): the robot farther than exit_distance (xy) from every mapped
#            stair voxel near its height for exit_frames consecutive class frames; never during a
#            climb (voxel_mapper_node ignores the switch while climbing and ends the climb by the
#            same rule). The paper gives no values: exit_distance 1.0 m, exit_frames 5 (class
#            frames at the segmentation rate, SAM3 ~2.8 Hz: ~1.8 s).
#            exit_rule='sight' (the earlier rule): no stairs in view for exit_frames consecutive
#            class frames (fewer than min_stair_points stair pixels within enter_range).
# N_enter (auto_enter) is kept as a duration (enter_time): every camera frame is checked.

# Pure python, no ROS imports.

import numpy as np

OTHER, FLOOR, RUG, STAIRS = 0, 1, 2, 3
TRAVERSABLE = (FLOOR, RUG, STAIRS)


def split_classes(classes):
    """Masks (traversable, stairs) for an array of class codes."""
    c = np.asarray(classes)
    return np.isin(c, TRAVERSABLE), c == STAIRS


class ModeSwitch:

    def __init__(self, enter_time=0.5, min_stair_points=50, enter_range=4.0,
                 exit_distance=1.0, mode='g', auto_enter=False, exit_rule='distance', exit_frames=5):
        self.auto_enter = bool(auto_enter)
        if exit_rule not in ('sight', 'distance'):
            raise ValueError(f"exit_rule must be 'sight' or 'distance', got {exit_rule!r}")
        self.exit_rule, self.exit_frames = exit_rule, int(exit_frames)
        self.n_run = 0         # consecutive exit frames (mode s): far from the stairs / none in view
        self.enter_time = float(enter_time)
        self.min_stair_points = int(min_stair_points)
        self.enter_range, self.exit_distance = float(enter_range), float(exit_distance)
        self.mode = mode
        self.since = None      # start of the current run of stair frames (g) / far frames (s)

    def request(self, mode):
        """Planner request (e.g. navigation to a stair entrance): switch at once."""
        changed = mode != self.mode
        self.mode = mode
        self.since = None
        self.n_run = 0
        return changed

    def update(self, t, n_near_stair_points, stair_distance):
        """One frame at time t (s): stair pixels within enter_range, xy distance of the robot to
        the nearest mapped stair voxel (inf if none). Returns True if the mode changed."""
        if self.mode == 'g':
            if not self.auto_enter:            # mode s only when the planner takes the stairs
                self.since = None
                return False
            cond, hold, target = n_near_stair_points >= self.min_stair_points, self.enter_time, 's'
        else:
            # N_exit consecutive frames: far from the stairs (distance) / no stairs in view (sight)
            far = (stair_distance > self.exit_distance if self.exit_rule == 'distance'
                   else n_near_stair_points < self.min_stair_points)
            if not far:
                self.n_run = 0
                return False
            self.n_run += 1
            if self.n_run >= self.exit_frames:
                return self.request('g')
            return False
        if not cond:
            self.since = None
            return False
        if self.since is None:
            self.since = t
        if t - self.since >= hold:
            return self.request(target)
        return False
