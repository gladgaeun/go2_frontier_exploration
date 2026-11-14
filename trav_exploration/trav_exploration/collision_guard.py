# Collision guard: stop forward motion before the robot body touches an obstacle, and decide
# where to mark the map as blocked (TravExplorer's failure handling, Sec. on eq. 17: the voxel
# in front of a robot that cannot go on is marked occupied and the planner replans).
#
# Planning runs on M_trav, inflated by a radius rounded to voxels (and kept slightly below the
# real robot radius so doorways stay open). The guard enforces the real radius on points:
#   - obstacle points at collision height (h_min < z - z_r < h_max), from the current depth
#     frame and from the voxel map (the camera cannot see low obstacles right in front of it)
#   - danger zone: ahead of the robot, |y| < radius, 0 < x < radius + distance (base frame)
#   - while in danger, forward velocity is cut to 0 (turning and reversing stay allowed)
# Navigation goals that fail repeatedly at one spot (Nav2 refusing, even after its recoveries,
# to drive into a gap it planned through) are counted as failures too.
# Escape: after such repeated failures the robot can be shut in by its own map (inflation,
# blocked marks and Nav2's margins closing a narrow doorway it came through) while the way back
# is physically free. It then drives back along its own trail (Trail, follow) for ~1 m, without
# Nav2, with the forward danger check still on.
#
# Pure numpy, no ROS imports.

import math

import numpy as np


def danger_points(xy_base, radius, distance):
    """Points (base frame, N x 2) in the zone the robot sweeps when it drives forward."""
    xy = np.asarray(xy_base, dtype=np.float64).reshape(-1, 2)
    m = (xy[:, 0] > 0.0) & (xy[:, 0] < radius + distance) & (np.abs(xy[:, 1]) < radius)
    return xy[m]


def front_row(x, y, yaw, radius, step, distance):
    """Positions of one row across the robot width, `distance` m in front of its centre (v_front).

    The map inflates blocked voxels like any obstacle, so distance must exceed the inflation
    radius: otherwise the robot's own cell ends up in O_inf and it can no longer turn away.
    """
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    left = np.array([-math.sin(yaw), math.cos(yaw)])
    lateral = np.arange(-radius, radius + 1e-6, step)
    return np.array([x, y]) + distance * fwd + lateral[:, None] * left


class Gate:
    """Forward-velocity gate with a hold time after the last danger."""

    def __init__(self, hold=1.0):
        self.hold = hold
        self.last_danger = -1e9

    def update(self, t, danger):
        """Feed one danger check; returns True on the rising edge (a new stop)."""
        was = self.active(t)
        if danger:
            self.last_danger = t
        return danger and not was

    def active(self, t):
        return t - self.last_danger < self.hold

    def filter(self, t, vx, wz):
        if self.active(t) and vx > 0.0:
            return 0.0, wz
        return vx, wz


class AbortCounter:
    """n navigation failures within `window` s while the robot stays within `radius` m."""

    def __init__(self, n=2, window=120.0, radius=0.3):
        self.n, self.window, self.radius = n, window, radius
        self.events = []

    def abort(self, t, x, y):
        self.events = [(te, xe, ye) for te, xe, ye in self.events
                       if t - te <= self.window and math.hypot(x - xe, y - ye) <= self.radius]
        self.events.append((t, x, y))
        if len(self.events) >= self.n:
            self.events = []
            return True
        return False


class Trail:
    """Breadcrumbs of the robot centre, one every `spacing` m: positions it physically occupied."""

    def __init__(self, spacing=0.1, max_len=5000):
        self.spacing, self.max_len = spacing, max_len
        self.pts = []

    def add(self, x, y):
        if not self.pts or math.hypot(x - self.pts[-1][0], y - self.pts[-1][1]) >= self.spacing:
            self.pts.append((x, y))
            del self.pts[:-self.max_len]

    def back(self, x, y, distance=1.0, max_length=3.0, loop_radius=0.15, recent=300):
        """Waypoints from (x, y) back along the trail until one is `distance` m away in a straight
        line, or the waypoints cover max_length m. Loops are cut: from each point the walk jumps to
        the earliest crumb within loop_radius of it among the last `recent` crumbs, i.e. to when
        the robot got there in this visit (a robot that jittered in place leaves a long trail that
        goes nowhere). Not over the whole trail: a long run passes the same corridor many times,
        and the walk jumped to a pass minutes earlier and went back the way the robot came then
        (escape routes of ~4 m that timed out)."""
        if not self.pts:
            return []
        pts = np.asarray(self.pts[-recent:])

        def earliest_near(p, upto):
            d = np.hypot(pts[:upto + 1, 0] - p[0], pts[:upto + 1, 1] - p[1])
            near = np.flatnonzero(d < loop_radius)
            return int(near[0]) if len(near) else upto

        out, length, prev = [], 0.0, (x, y)
        k = earliest_near((x, y), len(pts) - 1)
        while k >= 0:
            p = (float(pts[k, 0]), float(pts[k, 1]))
            out.append(p)
            length += math.hypot(p[0] - prev[0], p[1] - prev[1])
            prev = p
            if math.hypot(p[0] - x, p[1] - y) >= distance or length >= max_length or k == 0:
                break
            k = earliest_near(pts[k - 1], k - 1)
        return out


def next_target(waypoints, i, x, y, lookahead=0.35):
    """Index of the waypoint to head for: the first from i on that is lookahead m away (the
    last one at the end), so crumbs scattered around the start do not make the robot turn back
    and forth."""
    while i < len(waypoints) - 1 and math.hypot(waypoints[i][0] - x, waypoints[i][1] - y) < lookahead:
        i += 1
    return i


def follow(x, y, yaw, waypoint, speed=0.15, turn_speed=0.6, turn_angle=0.6, gain=1.5):
    """Unicycle command toward a waypoint: turn in place when it is more than turn_angle off,
    else drive forward (the guard's danger check covers forward motion only)."""
    err = math.atan2(waypoint[1] - y, waypoint[0] - x) - yaw
    err = math.atan2(math.sin(err), math.cos(err))
    if abs(err) > turn_angle:
        return 0.0, math.copysign(turn_speed, err)
    return speed, max(-turn_speed, min(turn_speed, gain * err))
