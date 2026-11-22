# Frontier visiting order of TravExplorer (Sec. VI-A/B), geometric frontiers only.
#
# Candidate pool X = {x_r} U O U F (eq. 8). Without semantics O = {} and F = F_g, so the active
# set (eq. 9) is A = F_g and the ATSP node set (eq. 10) is X_A = {x_r} U F_g.
# Cost matrix (eq. 11-13): c_rr = 0, c_Ar = 0 (open tour, no return to the robot),
#   c_rA(j)   = J_geo(x_r, v_j) + J_front(v_j)
#   C_AA(i,j) = J_geo(v_i, v_j) + J_front(v_j),  0 on the diagonal
# J_front = lambda_g for every node (eq. 14): it adds the same amount to every tour, so it does not
# change the order (kept as a parameter for when semantic frontiers come in).
# J_geo is the shortest path cost over traversable cells (Sec. VI-B): step cost (eq. 15-16)
#   J_step(v_i, v_j) = r |dk_ij| + lambda_esdf * max(0, (d_safe - d_obs(v_j)) / d_safe)
# with one multi-target Dijkstra per source candidate. Single floor: the traversable voxels of the
# support layer are the cells of trav_grid, 8-connected (the paper's foothold neighbourhoods for
# stairs are not needed). M_inf (no path) is np.inf.
# The goal is the first node of the open tour. Exact (Held-Karp) up to `exact_max` nodes, above
# that nearest neighbour + 2-opt / or-opt (the paper uses an ATSP solver [40], not specified).
#
# Pure numpy/scipy, no ROS.

import math

import numpy as np


def passable_cells(grid):
    """Cells NavFn plans through: free and not 4-adjacent to an occupied cell (INSCRIBED cost of
    the 0.05 m Nav2 robot_radius), as in goal_selection.reachable_mask."""
    from scipy import ndimage
    occ = grid > 50
    blocked = ndimage.binary_dilation(occ, structure=ndimage.generate_binary_structure(2, 1))
    return (grid == 0) & ~blocked


def step_graph(grid, res, lambda_esdf=0.1, d_safe=0.5, obstacle_offset=0.0):
    """Sparse directed graph of eq. 15 over the passable cells of grid [row=y, col=x].

    d_obs = distance to the nearest occupied cell + obstacle_offset (trav_grid obstacles are
    O_inf, inflated by the robot radius: offset = that radius gives the distance to the real
    obstacle). Returns (csr graph, cell index map [-1 = not a node], passable mask).
    """
    from scipy import ndimage, sparse
    passable = passable_cells(grid)
    h, w = grid.shape
    d_obs = ndimage.distance_transform_edt(~(grid > 50)) * res + obstacle_offset
    phi = np.maximum(0.0, (d_safe - d_obs) / d_safe) if d_safe > 0 else np.zeros_like(d_obs)
    index = np.full((h, w), -1, dtype=np.int64)
    index[passable] = np.arange(int(passable.sum()))
    rows, cols, vals = [], [], []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            # cell a = (y, x) -> cell b = (y + dy, x + dx), both passable
            ya, yb = slice(max(0, -dy), h - max(0, dy)), slice(max(0, dy), h - max(0, -dy))
            xa, xb = slice(max(0, -dx), w - max(0, dx)), slice(max(0, dx), w - max(0, -dx))
            both = passable[ya, xa] & passable[yb, xb]
            rows.append(index[ya, xa][both])
            cols.append(index[yb, xb][both])
            vals.append(res * math.hypot(dx, dy) + lambda_esdf * phi[yb, xb][both])
    n = int(passable.sum())
    g = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n, n))
    return g, index, passable


def nearest_node(index, x, y, origin, res, radius):
    """Graph node of the passable cell nearest to (x, y) within radius m, or -1."""
    h, w = index.shape
    cx, cy = int(math.floor((x - origin[0]) / res)), int(math.floor((y - origin[1]) / res))
    r = int(math.ceil(radius / res))
    y0, y1, x0, x1 = max(cy - r, 0), min(cy + r + 1, h), max(cx - r, 0), min(cx + r + 1, w)
    if y0 >= y1 or x0 >= x1:
        return -1
    win = index[y0:y1, x0:x1]
    yy, xx = np.nonzero(win >= 0)
    if len(yy) == 0:
        return -1
    d2 = (yy + y0 - cy) ** 2 + (xx + x0 - cx) ** 2
    k = np.argmin(d2)
    if d2[k] > r * r:
        return -1
    return int(win[yy[k], xx[k]])


def geo_costs(graph, sources, targets):
    """J_geo between graph nodes: [len(sources), len(targets)], np.inf where there is no path.
    One multi-target Dijkstra per source (Sec. VI-B)."""
    from scipy.sparse.csgraph import dijkstra
    sources = np.asarray(sources, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    out = np.full((len(sources), len(targets)), np.inf)
    ok_s = sources >= 0
    ok_t = targets >= 0
    if ok_s.any() and ok_t.any():
        d = dijkstra(graph, directed=True, indices=sources[ok_s])
        out[np.ix_(ok_s, ok_t)] = d[:, targets[ok_t]]
    return out


def cost_matrix(J_geo, lambda_g=0.0, node_cost=None):
    """Eq. 11-13 for X_A = {x_r} U F_g. J_geo is [1 + n, n]: row 0 from the robot, row 1 + i from
    candidate i. node_cost (n,): a further cost of entering candidate j, added like J_front(v_j)
    (the value-map term lambda_v (1 - u_j), Sec. V-B). Returns the (1 + n) x (1 + n) matrix,
    node 0 = robot."""
    n = J_geo.shape[1]
    extra = lambda_g + (np.zeros(n) if node_cost is None else np.asarray(node_cost, dtype=np.float64))
    C = np.zeros((n + 1, n + 1))
    C[0, 1:] = J_geo[0] + extra                    # c_rA (eq. 13)
    C[1:, 1:] = J_geo[1:] + extra[None, :]         # C_AA (eq. 12)
    np.fill_diagonal(C, 0.0)                       # c_rr = 0, C_AA(i, i) = 0
    C[1:, 0] = 0.0                                 # c_Ar = 0: open tour
    return C


def tour_cost(C, order):
    """Cost of the open tour robot -> order[0] -> order[1] ... (order of candidate nodes >= 1)."""
    prev, s = 0, 0.0
    for j in order:
        s += C[prev, j]
        prev = j
    return s


def _held_karp(C):
    n = C.shape[0] - 1
    full = 1 << n
    dp = np.full((full, n), np.inf)
    parent = np.full((full, n), -1, dtype=np.int64)
    for j in range(n):
        dp[1 << j, j] = C[0, j + 1]
    A = C[1:, 1:]
    for mask in range(1, full):
        row = dp[mask]
        if not np.isfinite(row).any():
            continue
        for k in range(n):
            if mask & (1 << k):
                continue
            # best predecessor j in mask for appending k
            cand = row + A[:, k]
            j = int(np.argmin(cand))
            m2 = mask | (1 << k)
            if cand[j] < dp[m2, k]:
                dp[m2, k] = cand[j]
                parent[m2, k] = j
    last = int(np.argmin(dp[full - 1]))
    order, mask = [], full - 1
    while last >= 0:
        order.append(last + 1)
        prev = int(parent[mask, last])
        mask &= ~(1 << last)
        last = prev
    return order[::-1]


def _local_search(C, order):
    order = list(order)
    best = tour_cost(C, order)
    improved = True
    while improved:
        improved = False
        n = len(order)
        for i in range(n - 1):                     # 2-opt: reverse order[i:j+1]
            for j in range(i + 1, n):
                cand = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                c = tour_cost(C, cand)
                if c < best - 1e-9:
                    order, best, improved = cand, c, True
        for i in range(n):                         # or-opt: move one node elsewhere
            rest = order[:i] + order[i + 1:]
            for j in range(n):
                if j == i:
                    continue
                cand = rest[:j] + [order[i]] + rest[j:]
                c = tour_cost(C, cand)
                if c < best - 1e-9:
                    order, best, improved = cand, c, True
                    break
    return order


def solve_open_atsp(C, exact_max=12):
    """Order of the candidate nodes (1..n) of the open tour from node 0 with the least cost."""
    n = C.shape[0] - 1
    if n <= 0:
        return []
    if n == 1:
        return [1]
    # inf (no path) would make every tour inf: a large finite M instead
    finite = C[np.isfinite(C)]
    big = (finite.max() if finite.size else 1.0) * (n + 1) + 1.0
    C = np.where(np.isfinite(C), C, big)
    if n <= exact_max:
        return _held_karp(C)
    order, left, prev = [], set(range(1, n + 1)), 0
    while left:                                    # nearest neighbour start
        j = min(left, key=lambda k: C[prev, k])
        order.append(j)
        left.remove(j)
        prev = j
    return _local_search(C, order)


def grid_path(grid, origin, res, start, goal, goal_radius=1.0, start_radius=0.5, **graph_args):
    """Shortest path (list of xy) on the passable cells of a 2-D grid from start to the passable
    cell nearest to goal within goal_radius that is reachable; None if there is none. Used to
    drive to the foot of the stairs on the current floor (the foot itself is an obstacle there)."""
    from scipy.sparse.csgraph import dijkstra
    graph, index, passable = step_graph(grid, res, **graph_args)
    s = nearest_node(index, start[0], start[1], origin, res, start_radius)
    if s < 0:
        return None
    dist, pred = dijkstra(graph, directed=True, indices=s, return_predecessors=True)
    rows, cols = np.nonzero(index >= 0)
    nodes = index[rows, cols]
    cx, cy = origin[0] + (cols + 0.5) * res, origin[1] + (rows + 0.5) * res
    d_goal = np.hypot(cx - goal[0], cy - goal[1])
    ok = np.isfinite(dist[nodes]) & (d_goal <= goal_radius)
    if not ok.any():
        return None
    k = np.flatnonzero(ok)[np.argmin(d_goal[ok])]
    t = int(nodes[k])
    path = [t]
    while path[-1] != s:
        path.append(int(pred[path[-1]]))
    order = np.empty(len(nodes), dtype=np.int64)
    order[nodes] = np.arange(len(nodes))
    return [(float(cx[order[v]]), float(cy[order[v]])) for v in path[::-1]]
