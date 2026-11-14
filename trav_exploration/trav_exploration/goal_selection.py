# Frontier goal selection for the 3D traversable frontiers.
#
# Where exactly to send the robot for a frontier is not given in the paper, and is ported
# unchanged from the 2D stack (frontier_exploration), with the 2D ranking (goal_order:=score;
# the default order is the paper's open ATSP, tour.py):
#   select_frontier    <- selectFrontier()  in frontier_utils.cpp
#   snap_to_clear_cell <- FrontierExplorer::snapToClearCell() in classical_frontier_detector.cpp
#   pull_back          <- the goal_clearance == 0 branch of FrontierExplorer::get_frontiers()
# Added (not in the 2D stack): reachable_mask / is_reachable drop clusters the robot cannot reach
# on the Nav2 grid (seen through a mesh hole, behind a wall or a gap blocked for the robot).
# The 2D loop would try each of them, and they kept the map showing frontiers after the end.
# Differences: a cluster's size is its number of frontier voxels (0.1 m) instead of 0.05 m grid
# cells, and the distance limit for crossing unknown cells in snap_to_clear_cell is in metres
# (0.2 m, 4 cells of the 2D map) so it does not depend on the grid resolution.
#
# Pure python/numpy (+ scipy.ndimage), no ROS imports.

import math
from collections import deque

import numpy as np


def select_frontier(clusters, rank, rx, ry, alpha=0.5, min_dist=0.4):
    """Return the rank-th best cluster (x, y, z, size), or None.

    score = alpha / max(d, 0.1) + (1 - alpha) * size, d = xy distance to the robot. Clusters
    closer than min_dist are skipped unless no cluster is farther away.
    """
    if not clusters:
        return None
    scored, valid = [], []
    for c in clusters:
        d = math.hypot(rx - c[0], ry - c[1])
        s = (1.0 / max(d, 0.1)) * alpha + float(c[3]) * (1.0 - alpha)
        scored.append((s, c))
        if d >= min_dist:
            valid.append((s, c))
    candidates = sorted(valid or scored, key=lambda sc: sc[0], reverse=True)
    if 0 <= rank < len(candidates):
        return candidates[rank][1]
    return None


def clear_cells(grid, res, clearance):
    """Cells with `clearance` m to every occupied cell and the grid border: the is_clear test of
    snap_to_clear_cell for all cells at once."""
    from scipy import ndimage
    r = int(math.ceil(clearance / res))
    occ = np.pad(grid > 50, 1, constant_values=True)
    return ndimage.distance_transform_edt(~occ)[1:-1, 1:-1] > r


def snap_to_clear_cell(grid, origin, res, x, y, clearance, max_search=2.0, unknown_reach=0.2, clear=None):
    """Move (x, y) to the nearest free cell with `clearance` m to every occupied cell.

    grid: int8 occupancy grid [row=y, col=x] (0 free, 100 occupied, -1 unknown).
    Breadth-first search from the goal through free cells (and unknown cells only within
    unknown_reach of the start), so the goal stays on the frontier's side of obstacles.
    clear: clear_cells(grid, res, clearance), to snap many goals on one grid faster.
    Returns (x, y) or None.
    """
    h, w = grid.shape
    ox, oy = origin
    sx, sy = int(math.floor((x - ox) / res)), int(math.floor((y - oy) / res))
    if not (0 <= sx < w and 0 <= sy < h):
        return None
    r = int(math.ceil(clearance / res))
    max_steps = int(math.ceil(max_search / res))
    unknown_steps = int(round(unknown_reach / res))
    disk = [(dx, dy) for dy in range(-r, r + 1) for dx in range(-r, r + 1) if dx * dx + dy * dy <= r * r]

    def is_clear(cx, cy):
        if clear is not None:
            return clear[cy, cx]
        for dx, dy in disk:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < w and 0 <= ny < h) or grid[ny, nx] > 50:
                return False
        return True

    dist = np.full((h, w), -1, dtype=np.int32)
    dist[sy, sx] = 0
    q = deque([(sx, sy)])
    nbr = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
    while q:
        cx, cy = q.popleft()
        if grid[cy, cx] == 0 and is_clear(cx, cy):
            return ox + (cx + 0.5) * res, oy + (cy + 0.5) * res
        d = dist[cy, cx]
        if d >= max_steps:
            continue
        for dx, dy in nbr:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < w and 0 <= ny < h) or dist[ny, nx] >= 0:
                continue
            v = grid[ny, nx]
            # never through obstacles; unknown cells only right next to the start
            if v > 50 or (v != 0 and d >= unknown_steps):
                continue
            dist[ny, nx] = d + 1
            q.append((nx, ny))
    return None


def pull_back(x, y, rx, ry, dist_min=0.8, pullback=0.3):
    """Move the goal pullback m toward the robot (into known space) if it is farther than dist_min."""
    d = math.hypot(x - rx, y - ry)
    if d > dist_min:
        x -= pullback / d * (x - rx)
        y -= pullback / d * (y - ry)
    return x, y


def reachable_mask(grid, x, y, origin, res, search=0.5):
    """Cells of the grid Nav2 can plan through from (x, y), as a boolean [row=y, col=x] mask.

    NavFn treats INSCRIBED cost (>= 253) as an obstacle, and the inflation layer gives it to the
    cells within the 0.05 m robot_radius of an occupied cell, i.e. its 4-neighbours: those are
    removed from the free cells before taking the 8-connected component of the robot. The robot
    may stand on such a cell (NavFn clears the start cell), so it starts from the nearest
    plannable cell within `search` m. Returns an all-False mask when there is none.
    """
    from scipy import ndimage
    occ = grid > 50
    blocked = ndimage.binary_dilation(occ, structure=ndimage.generate_binary_structure(2, 1))
    passable = (grid == 0) & ~blocked
    labels, _ = ndimage.label(passable, structure=np.ones((3, 3), dtype=bool))
    h, w = grid.shape
    cx, cy = int(math.floor((x - origin[0]) / res)), int(math.floor((y - origin[1]) / res))
    r = int(math.ceil(search / res))
    y0, y1, x0, x1 = max(cy - r, 0), min(cy + r + 1, h), max(cx - r, 0), min(cx + r + 1, w)
    win = labels[y0:y1, x0:x1]
    if not (win > 0).any():
        return np.zeros_like(passable)
    yy, xx = np.nonzero(win)
    k = np.argmin((yy + y0 - cy) ** 2 + (xx + x0 - cx) ** 2)
    return labels == win[yy[k], xx[k]]


def is_reachable(mask, x, y, origin, res, radius=0.4):
    """True if a reachable cell lies within radius m of (x, y) (a frontier centroid lies on the
    boundary of M_trav, next to the free cells the robot would stand on)."""
    h, w = mask.shape
    cx, cy = int(math.floor((x - origin[0]) / res)), int(math.floor((y - origin[1]) / res))
    r = int(math.ceil(radius / res))
    win = mask[max(cy - r, 0):min(cy + r + 1, h), max(cx - r, 0):min(cx + r + 1, w)]
    return bool(win.any())
