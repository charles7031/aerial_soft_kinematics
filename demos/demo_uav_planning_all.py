"""
Demo: UAV 3D Planning + Tracking
================================

Pipeline:
    A* / RRT / RRT*  →  Waypoints  →  PID Tracking

Saved figures:
    outputs_uav_planning/
        scene_astar.png
        scene_rrt.png
        scene_rrtstar.png
        error_astar.png
        error_rrt.png
        error_rrtstar.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Planning algorithms
# ============================================================
from planners.astar_grid3d import (
    astar_3d,
    grid_path_to_world,
    shortcut_smooth
)

from planners.rrt3d import (
    rrt_3d,
    shortcut_smooth_rrt
)

from planners.rrt_star_3d import (
    rrt_star_3d
)

# ============================================================
# UAV tracking controller (idealized outer-loop PID)
# ============================================================
from uav.pid_tracker import WaypointPIDTracker


# ============================================================
# 1. Build 3D environment
# ============================================================
def build_scene_grid(origin, res, dims):
    """
    Build:
      - occupancy grid (for A*)
      - list of AABBs (for RRT / RRT*)
    """
    origin = np.asarray(origin, float)
    Nx, Ny, Nz = dims

    occ = np.zeros((Nx, Ny, Nz), dtype=bool)
    aabbs = []

    def add_box(lo, hi):
        lo = np.asarray(lo, float)
        hi = np.asarray(hi, float)
        aabbs.append((lo, hi))

        ilo = np.floor((lo - origin) / res).astype(int)
        ihi = np.floor((hi - origin) / res).astype(int)
        ilo = np.clip(ilo, 0, [Nx-1, Ny-1, Nz-1])
        ihi = np.clip(ihi, 0, [Nx-1, Ny-1, Nz-1])

        occ[
            ilo[0]:ihi[0]+1,
            ilo[1]:ihi[1]+1,
            ilo[2]:ihi[2]+1
        ] = True

    # ----- obstacles -----
    add_box([0.35, 0.20, 0.00], [0.55, 0.80, 0.90])
    add_box([0.75, 0.00, 0.00], [0.90, 0.55, 0.80])
    add_box([0.10, 0.70, 0.00], [0.30, 0.90, 0.60])

    return occ, aabbs


def world_to_grid(p, origin, res):
    return tuple(np.floor((p - origin) / res).astype(int))


# ============================================================
# 2. Visualization helpers
# ============================================================
def plot_aabbs_3d(ax, aabbs, alpha=0.2):
    for (lo, hi) in aabbs:
        xs = [lo[0], hi[0]]
        ys = [lo[1], hi[1]]
        zs = [lo[2], hi[2]]
        corners = np.array([[x,y,z] for x in xs for y in ys for z in zs])

        edges = [
            (0,1),(0,2),(0,4),
            (3,1),(3,2),(3,7),
            (5,1),(5,4),(5,7),
            (6,2),(6,4),(6,7),
        ]
        for i,j in edges:
            ax.plot(
                [corners[i,0], corners[j,0]],
                [corners[i,1], corners[j,1]],
                [corners[i,2], corners[j,2]],
                color="k", alpha=alpha
            )


# ============================================================
# 3. Main experiment
# ============================================================
def run(planner="astar"):
    """
    planner ∈ {"astar", "rrt", "rrtstar"}
    """

    # -------------------------
    # Output directory
    # -------------------------
    out_dir = "outputs_uav_planning"
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------
    # Mission definition
    # -------------------------
    start = np.array([0.05, 0.05, 0.05])
    goal  = np.array([1.00, 1.00, 1.00])

    # -------------------------
    # Grid parameters (A*)
    # -------------------------
    origin = np.array([0.0, 0.0, 0.0])
    res = 0.05
    dims = (26, 26, 26)

    occ, aabbs = build_scene_grid(origin, res, dims)

    # ========================================================
    # Planning
    # ========================================================
    if planner == "astar":
        s_idx = world_to_grid(start, origin, res)
        g_idx = world_to_grid(goal, origin, res)

        path_idx = astar_3d(occ, s_idx, g_idx, diag=True)
        if path_idx is None:
            raise RuntimeError("A* failed.")

        waypoints = grid_path_to_world(path_idx, origin, res)
        waypoints = shortcut_smooth(
            waypoints, occ, origin, res, n_iter=400, seed=0
        )

    elif planner == "rrt":
        bounds = (origin, origin + res*np.array(dims))

        path = rrt_3d(
            start, goal,
            bounds=bounds,
            aabbs=aabbs,
            step_len=0.15,
            goal_sample_rate=0.2,
            max_iter=8000,
            goal_tol=0.15,
            seed=0
        )
        if path is None:
            raise RuntimeError("RRT failed.")

        waypoints = shortcut_smooth_rrt(path, aabbs, n_iter=400, seed=0)

    elif planner == "rrtstar":
        bounds = (origin, origin + res*np.array(dims))

        path = rrt_star_3d(
            start, goal,
            bounds=bounds,
            aabbs=aabbs,
            step_len=0.15,
            goal_sample_rate=0.2,
            max_iter=8000,
            goal_tol=0.15,
            search_radius=0.30,
            seed=0
        )
        if path is None:
            raise RuntimeError("RRT* failed.")

        waypoints = shortcut_smooth_rrt(path, aabbs, n_iter=400, seed=0)

    else:
        raise ValueError("planner must be astar / rrt / rrtstar")

    # ========================================================
    # Waypoint tracking (idealized PID)
    # ========================================================
    dt = 0.01
    T = 10.0
    N = int(T / dt)

    tracker = WaypointPIDTracker(dt=dt)
    tracker.reset(start)

    tol_wp = 0.05
    wp_id = 0

    t_log = np.zeros(N)
    p_log = np.zeros((N,3))

    for i in range(N):
        t_log[i] = i * dt
        p_ref = waypoints[wp_id]

        p, _, _, _ = tracker.step_to_waypoint(p_ref)

        if np.linalg.norm(p_ref - p) < tol_wp and wp_id < len(waypoints)-1:
            wp_id += 1

        p_log[i] = p

    # ========================================================
    # Visualization & save
    # ========================================================
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection="3d")

    plot_aabbs_3d(ax, aabbs)
    ax.plot(waypoints[:,0], waypoints[:,1], waypoints[:,2], "--", label="planned path")
    ax.plot(p_log[:,0], p_log[:,1], p_log[:,2], label="tracked trajectory")
    ax.scatter(*start, s=60, label="start")
    ax.scatter(*goal, s=60, label="goal")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(f"{planner.upper()} → Waypoints → PID Tracking")
    ax.legend()

    fig.savefig(os.path.join(out_dir, f"scene_{planner}.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # error curve
    err = np.linalg.norm(goal[None,:] - p_log, axis=1)
    fig2 = plt.figure()
    plt.plot(t_log, err)
    plt.axhline(0.05, linestyle="--", label="5 cm")
    plt.xlabel("time (s)")
    plt.ylabel("||p - goal|| (m)")
    plt.title(f"Goal Tracking Error ({planner.upper()})")
    plt.grid(True)
    plt.legend()

    fig2.savefig(os.path.join(out_dir, f"error_{planner}.png"), dpi=300, bbox_inches="tight")
    plt.show()

    print(f"[OK] {planner.upper()} finished, waypoints = {len(waypoints)}")


# ============================================================
# Entry point
# ============================================================
def main():
    run("astar")
    run("rrt")
    run("rrtstar")


if __name__ == "__main__":
    main()
