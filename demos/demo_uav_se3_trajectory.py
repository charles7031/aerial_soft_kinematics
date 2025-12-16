# demos/demo_uav_se3_trajectory.py
import os
import numpy as np
import matplotlib.pyplot as plt

from uav.se3_kinematics import fk_uav


def lerp(a, b, t):
    return (1 - t) * a + t * b


def smoothstep(t):
    # cubic smoothstep
    return t * t * (3 - 2 * t)


def demo_uav_trajectory(
    p0=np.array([0.0, 0.0, 0.3]),
    p1=np.array([2.0, 1.0, 1.2]),
    n=120,
    out_dir="outputs_uav",
    save_snapshots=True,
    snapshot_every=30,
):
    os.makedirs(out_dir, exist_ok=True)

    ts = np.linspace(0.0, 1.0, n)
    ps = np.zeros((n, 3))
    rpy = np.zeros((n, 3))

    for i, t in enumerate(ts):
        s = smoothstep(t)
        p = lerp(p0, p1, s)

        # yaw points toward the goal direction in XY plane
        d = p1 - p
        yaw = np.arctan2(d[1], d[0])
        roll = 0.0
        pitch = 0.0

        ps[i] = p
        rpy[i] = [roll, pitch, yaw]

    # --- Plot 3D path ---
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(ps[:, 0], ps[:, 1], ps[:, 2], linewidth=2, label="UAV path")
    ax.scatter([p0[0]], [p0[1]], [p0[2]], marker="o", s=60, label="start")
    ax.scatter([p1[0]], [p1[1]], [p1[2]], marker="x", s=80, label="goal")

    # draw yaw direction arrows at a few keyframes
    for i in range(0, n, max(1, n // 8)):
        p = ps[i]
        yaw = rpy[i, 2]
        # body x-axis in world projected (arrow length)
        L = 0.25
        ax.quiver(p[0], p[1], p[2],
                  L*np.cos(yaw), L*np.sin(yaw), 0.0,
                  length=1.0, normalize=False)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("UAV SE(3) Kinematics Demo: 3D Trajectory + Heading")
    ax.legend()
    ax.grid(True)

    # set equal-ish scale
    mins = ps.min(axis=0)
    maxs = ps.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = (maxs - mins).max()
    ax.set_xlim(center[0]-0.6*span, center[0]+0.6*span)
    ax.set_ylim(center[1]-0.6*span, center[1]+0.6*span)
    ax.set_zlim(center[2]-0.6*span, center[2]+0.6*span)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "uav_3d_trajectory.png"), dpi=300)

    # --- Optional snapshots (academic reporting) ---
    if save_snapshots:
        for i in range(0, n, snapshot_every):
            p = ps[i]
            T = fk_uav(p, rpy[i])  # not strictly needed, but shows FK usage

            # simple 2D projection snapshot (XY)
            f2, ax2 = plt.subplots()
            ax2.plot(ps[:i+1, 0], ps[:i+1, 1], linewidth=2)
            ax2.scatter([p0[0]], [p0[1]], marker="o", s=60, label="start")
            ax2.scatter([p1[0]], [p1[1]], marker="x", s=80, label="goal")
            ax2.scatter([p[0]], [p[1]], marker="o", s=40, label="current")

            yaw = rpy[i, 2]
            ax2.arrow(p[0], p[1], 0.25*np.cos(yaw), 0.25*np.sin(yaw),
                      head_width=0.05, length_includes_head=True)

            ax2.set_aspect("equal", adjustable="box")
            ax2.grid(True)
            ax2.set_title(f"t={i/(n-1):.2f}, p={p.round(2).tolist()}")
            ax2.legend()
            f2.tight_layout()
            f2.savefig(os.path.join(out_dir, f"snapshot_{i:03d}.png"), dpi=200)
            plt.close(f2)

    plt.show()
    print(f"Saved to: {out_dir}/")


def main():
    demo_uav_trajectory()


if __name__ == "__main__":
    main()
