"""v0 switch rule (floor-switch plan, sec. 9): shared by scripts/select_rule.py (offline replay)
and the online node.

Per decision time (1 s) the calibrated p of the current floor is kept; the history is cleared when
the floor changes (a new FloorTracker floor, or off the floors: stairs), and when target tracking
starts or ends; nothing is kept while a target is tracked. ready: the last X p are all below tau,
and the cost guard passes:
    actions left >= actions/m x distance to the nearest known stair entrance + climb p75 + margin
(no known stairs on this floor: not ready). Whether the robot switches is decided by
frontier_goal_node at its decision times (frontier_pose requests and every 10 s).
"""
import math
from collections import deque


def stair_distance(state):
    """Distance (m, straight line) from the robot to the nearest known (reliable) stair entrance
    of its floor (foot on this floor going up, or reaching down from it); None if none.
    state: a frontier_goal state record (robot, z_r, landmarks)."""
    z, rob = state.get('z_r'), state.get('robot')
    if z is None or not rob or rob[0] is None:
        return None
    best = None
    for lm in state.get('landmarks') or []:
        if len(lm) > 7 and lm[7] < 0.5:
            continue
        if abs(lm[2] - z) <= 0.4 or (lm[2] < z - 0.5 and lm[5] >= z - 1.0):
            d = math.hypot(lm[0] - rob[0], lm[1] - rob[1])
            best = d if best is None or d < best else best
    return best


class SwitchRule:
    def __init__(self, X, tau, actions_per_metre, guard_fixed, max_actions=500.0):
        self.X, self.tau = int(X), float(tau)
        self.k, self.fixed, self.max_actions = float(actions_per_metre), float(guard_fixed), float(max_actions)
        self.hist = deque(maxlen=self.X)
        self.floor = None
        self.tracking = False

    def step(self, p, floor, tracking, stairs, actions):
        """One decision time. p: calibrated p (None: no estimate); floor: FloorTracker index or None.
        Returns (ready, need): need = the actions the guard asks for (None without stairs)."""
        if floor is None or floor != self.floor:
            self.hist.clear()
            self.floor = floor
        if tracking != self.tracking:
            self.hist.clear()
            self.tracking = tracking
        need = None if stairs is None else self.k * stairs + self.fixed
        if floor is None or tracking or p is None:
            return False, need
        self.hist.append(p)
        ready = (len(self.hist) >= self.X and max(self.hist) < self.tau and need is not None
                 and self.max_actions - actions >= need)
        return ready, need
