# Unit tests for tour (open ATSP over {x_r} U F_g, pure numpy/scipy, no ROS).
#   python3 -m pytest trav_exploration/test/test_tour.py

import itertools
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from trav_exploration.goal_selection import clear_cells, snap_to_clear_cell  # noqa: E402
from trav_exploration.tour import (_local_search, cost_matrix, geo_costs, nearest_node,  # noqa: E402
                                   solve_open_atsp, step_graph, tour_cost)


def brute(C):
    n = C.shape[0] - 1
    return min(itertools.permutations(range(1, n + 1)), key=lambda o: tour_cost(C, o))


def test_open_tour_goes_to_the_near_end_first():
    # robot at 0 between a frontier at -1 and three at +2, +3, +4: going to -1 first and then
    # sweeping right (1 + 5) beats sweeping right first and coming back (2 + 2 + 7)
    xs = [-1.0, 2.0, 3.0, 4.0]
    pts = [0.0] + xs
    J = np.array([[abs(b - a) for b in xs] for a in pts])
    C = cost_matrix(J)
    assert C[1:, 0].tolist() == [0, 0, 0, 0] and C[0, 0] == 0          # open tour (eq. 11)
    order = solve_open_atsp(C)
    assert [xs[j - 1] for j in order] == [-1.0, 2.0, 3.0, 4.0]


def test_held_karp_and_local_search_are_optimal_on_random_instances():
    rng = np.random.default_rng(0)
    for n in (3, 5, 7):
        for _ in range(5):
            C = cost_matrix(rng.uniform(0.5, 5.0, (n + 1, n)), lambda_g=0.3)
            best = tour_cost(C, brute(C))
            assert math.isclose(tour_cost(C, solve_open_atsp(C)), best)
            assert tour_cost(C, _local_search(C, list(range(1, n + 1)))) <= best * 1.3
    C = cost_matrix(rng.uniform(0.5, 5.0, (16, 15)))
    order = solve_open_atsp(C)                                          # heuristic above exact_max
    assert sorted(order) == list(range(1, 16))


def test_lambda_g_does_not_change_the_order():
    rng = np.random.default_rng(1)
    J = rng.uniform(0.5, 5.0, (7, 6))
    assert solve_open_atsp(cost_matrix(J, 0.0)) == solve_open_atsp(cost_matrix(J, 2.0))


def test_geo_cost_goes_around_a_wall_and_pays_near_obstacles():
    res = 0.1
    g = np.zeros((30, 30), dtype=np.int8)       # 3 x 3 m free
    g[0:25, 15] = 100                           # wall at x = 1.5 m with a gap at the top
    graph, index, _ = step_graph(g, res, lambda_esdf=0.0)
    a = nearest_node(index, 0.55, 0.55, (0.0, 0.0), res, 0.3)
    b = nearest_node(index, 2.45, 0.55, (0.0, 0.0), res, 0.3)
    d = geo_costs(graph, [a], [b])[0, 0]
    assert d > 4.0                              # around the wall, not 1.9 m straight
    g2 = g.copy()
    g2[:, 15] = 100                             # closed: no path (M_inf)
    graph2, index2, _ = step_graph(g2, res)
    assert np.isinf(geo_costs(graph2, [a], [nearest_node(index2, 2.45, 0.55, (0, 0), res, 0.3)])[0, 0])
    # the ESDF term (eq. 16) only adds cost
    graph3, _, _ = step_graph(g, res, lambda_esdf=0.5, d_safe=0.5)
    assert geo_costs(graph3, [a], [b])[0, 0] > d


def test_clear_mask_matches_the_per_cell_test():
    rng = np.random.default_rng(2)
    g = np.where(rng.random((40, 40)) < 0.03, 100, 0).astype(np.int8)
    g[rng.random((40, 40)) < 0.05] = -1
    clear = clear_cells(g, 0.05, 0.18)
    for _ in range(30):
        x, y = rng.uniform(0.0, 2.0, 2)
        assert snap_to_clear_cell(g, (0.0, 0.0), 0.05, x, y, 0.18) == \
            snap_to_clear_cell(g, (0.0, 0.0), 0.05, x, y, 0.18, clear=clear)


def test_grid_path_reaches_the_nearest_free_cell_to_an_obstacle_goal():
    from trav_exploration.tour import grid_path
    res = 0.1
    g = np.zeros((30, 30), dtype=np.int8)
    g[10:20, 20:30] = 100                          # the foot of the stairs is an obstacle on the grid
    path = grid_path(g, (0.0, 0.0), res, (0.5, 1.5), (2.5, 1.5), goal_radius=1.0, lambda_esdf=0.0)
    assert path is not None and path[0][0] < 0.7
    end = path[-1]
    assert np.hypot(end[0] - 2.5, end[1] - 1.5) <= 1.0                         # next to it
    assert g[int(end[1] / res), int(end[0] / res)] == 0                         # on a free cell
    assert grid_path(g, (0.0, 0.0), res, (0.5, 1.5), (2.5, 1.5), goal_radius=0.3) is None


def test_cost_matrix_node_cost_enters_each_candidate():
    # the value-map term lambda_v (1 - u_j) is added on the way into candidate j, not out of it
    J = np.array([[1.0, 1.0], [0.0, 2.0], [2.0, 0.0]])
    C = cost_matrix(J, 0.5, node_cost=[3.0, 0.0])
    assert np.allclose(C, [[0.0, 4.5, 1.5], [0.0, 0.0, 2.5], [0.0, 5.5, 0.0]])
    # a high-value candidate farther away now comes first in the tour
    order = solve_open_atsp(C)
    assert order[0] == 2
