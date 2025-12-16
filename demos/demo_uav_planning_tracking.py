import os
import numpy as np
import matplotlib.pyplot as plt

from planners.astar_grid3d import astar_3d, grid_path_to_world, shortcut_smooth
from planners.rrt3d import rrt_3d, shortcut_smooth_rrt
from uav.pid_tracker import WaypointPIDTracker

def build_scene_grid(origin, res, dims):
    """
    Build a simple 3D grid with a few box obstacles (AABB),
    returned as occupancy grid + list of AABBs (for plotting / RRT).
    """
    origin = np.asarray(origin, float)
    Nx, Ny, Nz = dims
    occ = np.zeros((Nx,Ny,Nz), dtype=bool)

    # AABBs in world coordinates: (lo, hi)
    aabbs = []

    def add_box(lo, hi):
        lo = np.asarray(lo, float); hi = np.asarray(hi, float)
        aabbs.append((lo, hi))
        # mark occupancy
        ilo = np.floor((lo - origin)/res).astype(int)
        ihi = np.floor((hi - origin)/res).astype(int)
        ilo = np.clip(ilo, 0, np.array([Nx-1,Ny-1,Nz-1]))
        ihi = np.clip(ihi, 0, np.array([Nx-1,Ny-1,Nz-1]))
        occ[ilo[0]:ihi[0]+1, ilo[1]:ihi[1]+1, ilo[2]:ihi[2]+1] = True

    # obstacles (you can tweak these)
    add_box([0.35, 0.20, 0.00], [0.55, 0.80, 0.90])
    add_box([0.75, 0.00, 0.00], [0.90, 0.55, 0.80])
    add_box([0.10, 0.70, 0.00], [0.30, 0.90, 0.60])

    return occ, aabbs

def world_to_grid(p, origin, res):
    ijk = np.floor((np.asarray(p)-origin)/res).astype(int)
    return tuple(ijk.tolist())

def plot_aabbs_3d(ax, aabbs, alpha=0.15):
    # simple wireframe boxes
    for (lo, hi) in aabbs:
        lo = np.asarray(lo); hi = np.asarray(hi)
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
        for (i,j) in edges:
            ax.plot([corners[i,0], corners[j,0]],
                    [corners[i,1], corners[j,1]],
                    [corners[i,2], corners[j,2]],
                    alpha=alpha)

def run(planner="astar"):
    # =========================
    # Scene / mission settings
    # =========================
    out_dir = "outputs_uav_planning"
    os.makedirs(out_dir, exist_ok=True)

    start = np.array([0.05, 0.05, 0.05])
    goal  = np.array([1.00, 1.00, 1.00])

    # A* grid parameters
    origin = np.array([0.0, 0.0, 0.0])
    res = 0.05
    dims = (26, 26, 26)     # covers ~1.3m cube

    occ, aabbs = build_scene_grid(origin, res, dims)

    # =========================
    # Planning
    # =========================
    if planner == "astar":
        s_idx = world_to_grid(start, origin, res)
        g_idx = world_to_grid(goal, origin, res)
        path_idx = astar_3d(occ, s_idx, g_idx, diag=True)
        if path_idx is None:
            raise RuntimeError("A* failed: no path.")
        waypoints = grid_path_to_world(path_idx, origin, res)
        waypoints = shortcut_smooth(waypoints, occ=occ, origin=origin, res=res,
                                    n_iter=400, seed=0)
    elif planner == "rrt":
        bounds = (origin, origin + res*np.array(dims))
        path = rrt_3d(start, goal, bounds=bounds, aabbs=aabbs,
                      step_len=0.15, goal_sample_rate=0.20,
                      max_iter=8000, goal_tol=0.15, seed=0)
        if path is None:
            raise RuntimeError("RRT failed: try increase max_iter or loosen step_len.")
        waypoints = shortcut_smooth_rrt(path, aabbs=aabbs, n_iter=400, seed=0)
    else:
        raise ValueError("planner must be 'astar' or 'rrt'")

    # =========================
    # Tracking (waypoints -> PID)
    # =========================
    dt = 0.01
    T = 10.0
    N = int(T/dt)

    tracker = WaypointPIDTracker(dt=dt)
    tracker.reset(start)

    tol_wp = 0.05  # waypoint reach tolerance (m)
    wp_id = 0

    t_log = np.zeros(N)
    p_log = np.zeros((N,3))
    e_log = np.zeros((N,3))
    wp_log = np.zeros(N, dtype=int)
    thrust_log = np.zeros((N,4))

    for i in range(N):
        t = i*dt
        t_log[i] = t

        # current target waypoint
        p_ref = waypoints[wp_id]
        # step
        p, v, thrusts, e = tracker.step_to_waypoint(p_ref)

        # advance waypoint if close
        if np.linalg.norm(p_ref - p) < tol_wp and wp_id < waypoints.shape[0]-1:
            wp_id += 1

        p_log[i] = p
        e_log[i] = (goal - p)  # goal error (for display)
        wp_log[i] = wp_id
        thrust_log[i] = thrusts

    # =========================
    # Plot & save
    # =========================
    # 3D scene plot
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection="3d")
    plot_aabbs_3d(ax, aabbs)

    ax.plot(waypoints[:,0], waypoints[:,1], waypoints[:,2], "--", label=f"{planner.upper()} path")
    ax.plot(p_log[:,0], p_log[:,1], p_log[:,2], label="tracked trajectory")
    ax.scatter(start[0], start[1], start[2], s=60, label="start")
    ax.scatter(goal[0], goal[1], goal[2], s=60, label="goal")

    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.set_title(f"Planning ({planner.upper()}) → Waypoints → PID Tracking")
    ax.legend()
    fig.savefig(os.path.join(out_dir, f"scene_{planner}.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # error curve to goal
    err = np.linalg.norm(goal[None,:] - p_log, axis=1)
    fig2 = plt.figure()
    plt.plot(t_log, err, label="||p - goal||")
    plt.axhline(0.05, linestyle="--", label="5cm")
    plt.xlabel("time (s)")
    plt.ylabel("error (m)")
    plt.title("Goal Tracking Error")
    plt.grid(True)
    plt.legend()
    fig2.savefig(os.path.join(out_dir, f"error_{planner}.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # motor thrusts (equal in this simplified model; acceptable with ideal inner-loop)
    fig3 = plt.figure()
    for k in range(4):
        plt.plot(t_log, thrust_log[:,k], label=f"motor {k+1}")
    plt.xlabel("time (s)")
    plt.ylabel("thrust (N)")
    plt.title("Motor Thrusts (Ideal Inner-loop Abstraction)")
    plt.grid(True)
    plt.legend()
    fig3.savefig(os.path.join(out_dir, f"thrust_{planner}.png"), dpi=300, bbox_inches="tight")
    plt.show()

    print(f"[OK] Saved to: {out_dir}/")
    print(f"Planner={planner}, waypoints={waypoints.shape[0]}, final_pos={p_log[-1]}")

def main():
    # change to "rrt" if you want RRT demo
    run(planner="astar")
    # run(planner="rrt")

if __name__ == "__main__":
    main()
