# Semantic mode of TravExplorer (Sec. III-A): which segmentation classes are traversable, and the
# switch between the geometric (g) and semantic (s) traversability modes.
#
# Class image (habitat_server --semantic now, SAM3 later): one code per RGB pixel.
#   0 other, 1 floor, 2 rug (rugs, mats, carpets), 3 stairs
# Traversable classes (M_s): floor, rug, stairs (the paper's "floor, stair, and rug").
#
# Mode switch with hysteresis (values not given in the paper):
#   g -> s : stairs seen for enter_time without a frame missing them (at least min_stair_points
#            stair pixels within enter_range of the camera), or request('s') from the planner
#            (stair entrance landmark)
#   s -> g : the robot farther than exit_distance from every mapped stair voxel for exit_time
# The paper counts N_enter / N_exit consecutive frames. Here every camera frame is checked (stair
# pixels are cheap to count; only every 0.5 s a frame goes into the map), and the counts are
# kept as durations: the node drops frames when it is busy, so a frame count is not a fixed time.
# exit_distance > enter_range: with stairs entering by sight and leaving by distance at the same
# range, a robot that keeps seeing stairs from afar would switch every few seconds (and each
# switch re-inflates the whole map). enter_range is measured from the camera, exit_distance in xy:
# a robot more than exit_distance away cannot see stair pixels within enter_range.

# Pure python, no ROS imports.

import numpy as np

OTHER, FLOOR, RUG, STAIRS = 0, 1, 2, 3
TRAVERSABLE = (FLOOR, RUG, STAIRS)


def split_classes(classes):
    """Masks (traversable, stairs) for an array of class codes."""
    c = np.asarray(classes)
    return np.isin(c, TRAVERSABLE), c == STAIRS


class ModeSwitch:

    def __init__(self, enter_time=0.5, exit_time=1.0, min_stair_points=50, enter_range=4.0,
                 exit_distance=4.5, mode='g'):
        self.enter_time, self.exit_time = float(enter_time), float(exit_time)
        self.min_stair_points = int(min_stair_points)
        self.enter_range, self.exit_distance = float(enter_range), float(exit_distance)
        self.mode = mode
        self.since = None      # start of the current run of stair frames (g) / far frames (s)

    def request(self, mode):
        """Planner request (e.g. navigation to a stair entrance): switch at once."""
        changed = mode != self.mode
        self.mode = mode
        self.since = None
        return changed

    def update(self, t, n_near_stair_points, stair_distance):
        """One frame at time t (s): stair pixels within enter_range, xy distance of the robot to
        the nearest mapped stair voxel (inf if none). Returns True if the mode changed."""
        if self.mode == 'g':
            cond, hold, target = n_near_stair_points >= self.min_stair_points, self.enter_time, 's'
        else:
            cond, hold, target = stair_distance > self.exit_distance, self.exit_time, 'g'
        if not cond:
            self.since = None
            return False
        if self.since is None:
            self.since = t
        if t - self.since >= hold:
            return self.request(target)
        return False
