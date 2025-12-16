import numpy as np

# =========================================================
# Utility: collision checking
# =========================================================

def segment_intersects_aabb(p1, p2, aabb, n_samples=20):
    """
    Conservative segment-AABB collision test by sampling.
    p1, p2: endpoints
    aabb: (lo, hi)
    """
    lo, hi = aabb
    for s in np.linspace(0.0, 1.0, n_samples):
        p = p1 + s * (p2 - p1)
        if np.all(p >= lo) and np.all(p <= hi):
            return True
    return False


def collision_free(p1, p2, aabbs):
    for aabb in aabbs:
        if segment_intersects_aabb(p1, p2, aabb):
            return False
    return True


# =========================================================
# RRT core
# =========================================================

class Node:
    def __init__(self, p, parent=None):
        self.p = np.asarray(p, float)
        self.parent = parent


def nearest(nodes, p):
    dists = [np.linalg.norm(n.p - p) for n in nodes]
    return nodes[int(np.argmin(dists))]


def steer(p_from, p_to, step_len):
    d = p_to - p_from
    L = np.linalg.norm(d)
    if L <= step_len:
        return p_to
    return p_from + step_len * d / L


def reconstruct_path(node):
    path = []
    while node is not None:
        path.append(node.p)
        node = node.parent
    return np.array(path[::-1])


def rrt_3d(
    start,
    goal,
    bounds,
    aabbs,
    step_len=0.15,
    goal_sample_rate=0.2,
    max_iter=8000,
    goal_tol=0.15,
    seed=None,
):
    """
    Basic RRT in 3D.
    Returns: (N,3) path or None
    """
    rng = np.random.default_rng(seed)

    start = np.asarray(start, float)
    goal = np.asarray(goal, float)
    lo, hi = bounds

    nodes = [Node(start)]

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

        new_node = Node(p_new, parent=n_near)
        nodes.append(new_node)

        if np.linalg.norm(p_new - goal) < goal_tol:
            return reconstruct_path(new_node)

    return None


# =========================================================
# Path smoothing
# =========================================================

def shortcut_smooth_rrt(
    path,
    aabbs,
    n_iter=200,
    seed=None,
):
    """
    Shortcut smoothing for RRT / RRT*
    """
    rng = np.random.default_rng(seed)
    path = path.copy()

    if path.shape[0] < 3:
        return path

    for _ in range(n_iter):
        i = rng.integers(0, path.shape[0] - 2)
        j = rng.integers(i + 2, path.shape[0])

        if collision_free(path[i], path[j], aabbs):
            path = np.vstack([path[:i+1], path[j:]])

    return path


if __name__ == "__main__":
    print("[OK] rrt3d.py loaded")
