import numpy as np
from planners.rrt3d import collision_free, steer, reconstruct_path

# =========================================================
# RRT* Node
# =========================================================

class Node:
    def __init__(self, p, parent=None, cost=0.0):
        self.p = np.asarray(p, float)
        self.parent = parent
        self.cost = cost


def nearest(nodes, p):
    dists = [np.linalg.norm(n.p - p) for n in nodes]
    return nodes[int(np.argmin(dists))]


def near(nodes, p, radius):
    return [n for n in nodes if np.linalg.norm(n.p - p) <= radius]


# =========================================================
# RRT* core
# =========================================================

def rrt_star_3d(
    start,
    goal,
    bounds,
    aabbs,
    step_len=0.15,
    goal_sample_rate=0.2,
    max_iter=8000,
    goal_tol=0.15,
    search_radius=0.4,
    seed=None,
):
    """
    RRT* in 3D with rewiring.
    Returns: optimal path (N,3) or None
    """
    rng = np.random.default_rng(seed)

    start = np.asarray(start, float)
    goal = np.asarray(goal, float)
    lo, hi = bounds

    nodes = [Node(start, parent=None, cost=0.0)]

    for _ in range(max_iter):
        # sample
        if rng.random() < goal_sample_rate:
            p_rand = goal
        else:
            p_rand = rng.uniform(lo, hi)

        n_near = nearest(nodes, p_rand)
        p_new = steer(n_near.p, p_rand, step_len)

        if not collision_free(n_near.p, p_new, aabbs):
            continue

        # choose best parent
        near_nodes = near(nodes, p_new, search_radius)
        parent = n_near
        min_cost = n_near.cost + np.linalg.norm(p_new - n_near.p)

        for n in near_nodes:
            if collision_free(n.p, p_new, aabbs):
                c = n.cost + np.linalg.norm(p_new - n.p)
                if c < min_cost:
                    parent = n
                    min_cost = c

        new_node = Node(p_new, parent=parent, cost=min_cost)
        nodes.append(new_node)

        # rewire
        for n in near_nodes:
            c_new = new_node.cost + np.linalg.norm(n.p - new_node.p)
            if c_new < n.cost and collision_free(n.p, new_node.p, aabbs):
                n.parent = new_node
                n.cost = c_new

        if np.linalg.norm(p_new - goal) < goal_tol:
            return reconstruct_path(new_node)

    return None


if __name__ == "__main__":
    print("[OK] rrt_star_3d.py loaded")
