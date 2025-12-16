import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from planners.astar_grid3d import astar_3d, grid_path_to_world
from planners.rrt3d import rrt_3d
from planners.rrt_star_3d import rrt_star_3d
from uav.pid_tracker import WaypointPIDTracker
from dynamics.dynamic_obstacles import get_dynamic_aabbs


# ===============================
# AABB 可视化
# ===============================
def plot_aabbs_3d(ax, aabbs, color="r", alpha=0.25):
    for lo, hi in aabbs:
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
        for i,j in edges:
            ax.plot(
                [corners[i,0], corners[j,0]],
                [corners[i,1], corners[j,1]],
                [corners[i,2], corners[j,2]],
                color=color, alpha=alpha
            )


# ===============================
# 主函数
# ===============================
def run_dynamic(planner="rrtstar"):

    out_dir = "outputs_uav_dynamic"
    os.makedirs(out_dir, exist_ok=True)

    start = np.array([0.05, 0.05, 0.05])
    goal  = np.array([1.0, 1.0, 1.0])

    bounds = (np.zeros(3), np.array([1.3, 1.3, 1.3]))

    # A* grid
    origin = np.zeros(3)
    res = 0.05
    dims = (26,26,26)

    dt = 0.05
    T  = 12.0
    steps = int(T / dt)

    tracker = WaypointPIDTracker(dt=dt)
    tracker.reset(start)

    p = start.copy()
    traj = []
    obs_log = []

    t = 0.0

    for k in range(steps):

        dyn_aabbs = get_dynamic_aabbs(t)
        obs_log.append(dyn_aabbs)

        # ===== 每 0.5 秒重规划 =====
        if k % int(0.5/dt) == 0:

            if planner == "astar":
                occ = np.zeros(dims, dtype=bool)
                for lo, hi in dyn_aabbs:
                    ilo = np.floor((lo-origin)/res).astype(int)
                    ihi = np.floor((hi-origin)/res).astype(int)
                    occ[ilo[0]:ihi[0]+1, ilo[1]:ihi[1]+1, ilo[2]:ihi[2]+1] = True

                s = tuple(np.floor((p-origin)/res).astype(int))
                g = tuple(np.floor((goal-origin)/res).astype(int))
                idx_path = astar_3d(occ, s, g, diag=True)

                if idx_path is not None:
                    waypoints = grid_path_to_world(idx_path, origin, res)
                else:
                    waypoints = np.array([p, goal])

            elif planner == "rrt":
                waypoints = rrt_3d(
                    p, goal, bounds, dyn_aabbs,
                    max_iter=2000, step_len=0.15, goal_tol=0.15, seed=0
                )

            else:  # RRT*
                waypoints = rrt_star_3d(
                    p, goal, bounds, dyn_aabbs,
                    max_iter=2500, step_len=0.15,
                    rewire_radius=0.30, goal_tol=0.15, seed=0
                )

            wp = waypoints[1] if waypoints is not None and len(waypoints)>1 else goal

        p, _, _, _ = tracker.step_to_waypoint(wp)

        traj.append(p.copy())
        t += dt

        if np.linalg.norm(goal - p) < 0.05:
            break

    traj = np.array(traj)

    # ===============================
    # GIF
    # ===============================
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection="3d")

    def update(i):
        ax.cla()
        ax.plot(traj[:i,0], traj[:i,1], traj[:i,2], lw=2)
        ax.scatter(*traj[i], s=40)
        ax.scatter(*start, s=60)
        ax.scatter(*goal, s=60)

        plot_aabbs_3d(ax, obs_log[i])

        ax.set_xlim(0,1.3)
        ax.set_ylim(0,1.3)
        ax.set_zlim(0,1.3)
        ax.set_title(f"Dynamic Obstacle Avoidance ({planner.upper()})")

    ani = animation.FuncAnimation(fig, update, frames=len(traj), interval=80)

    gif_path = os.path.join(out_dir, f"dynamic_{planner}.gif")
    ani.save(gif_path, writer="pillow")
    print(f"[OK] saved {gif_path}")


def main():
    run_dynamic("astar")
    run_dynamic("rrt")
    run_dynamic("rrtstar")


if __name__ == "__main__":
    main()
