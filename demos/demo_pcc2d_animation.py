import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

from soft_arm.pcc2d_fk import pcc2d_fk
from soft_arm.pcc2d_ik import pcc2d_ik_opt
from soft_arm.pcc2d_fk import pcc2d_fk_dense


def animate_grasp(
    n_seg,
    L,
    kappa_max,
    target_xy,
    tol=0.02,          # ===== 抓取容差（米）=====
    n_frames=70,
    interval=60,
):
    """
    Animate a planar PCC arm gradually reaching a target.
    Entering the tolerance circle means successful grasp.
    """

    # ===== Solve IK once =====
    kappa_opt, info = pcc2d_ik_opt(
        target_xy=target_xy,
        n_seg=n_seg,
        L=L,
        kappa_max=kappa_max,
    )

    print(
        f"[N={n_seg}] IK success={info['success']}, "
        f"final err={info['err']:.4f} m"
    )

    # ===== Interpolation from straight arm to IK shape =====
    alphas = np.linspace(0.0, 1.0, n_frames)

    # ===== Figure setup =====
    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    reach = n_seg * L
    ax.set_xlim(-0.05, reach + 0.05)
    ax.set_ylim(-reach, reach)

    # ----- Target -----
    ax.scatter(
        [target_xy[0]], [target_xy[1]],
        marker="x",
        s=80,
        color="green",
        label="target"
    )

    # ----- Tolerance circle -----
    tol_circle = Circle(
        (target_xy[0], target_xy[1]),
        tol,
        fill=False,
        linestyle="--",
        linewidth=2,
        color="green",
        label="grasp tolerance"
    )
    ax.add_patch(tol_circle)

    # ----- Arm & EE artists -----
    line_arm, = ax.plot([], [], lw=3, color="tab:blue", label="soft arm")
    ee_dot, = ax.plot([], [], "o", color="red", label="end-effector")

    ax.set_title(
        f"PCC 2D Grasp Animation | "
        f"N={n_seg}, L={L:.2f} m, κ∈[-{kappa_max},{kappa_max}] 1/m\n"
        f"tol = {tol*100:.1f} cm"
    )
    ax.legend(loc="upper left")

    # ===== Animation update =====
    def update(frame):
        alpha = alphas[frame]
        kappa = alpha * kappa_opt

        ee, _, pts = pcc2d_fk(kappa, L)

        pts_dense = pcc2d_fk_dense(kappa, L, n_per_seg=25)
        line_arm.set_data(pts_dense[:, 0], pts_dense[:, 1])
        ee_dot.set_data([ee[0]], [ee[1]])

        # ----- Grasp condition -----
        dist = np.linalg.norm(ee - target_xy)
        if dist <= tol:
            ee_dot.set_color("green")   # grasped
        else:
            ee_dot.set_color("red")     # not yet

        return line_arm, ee_dot

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=interval,
        blit=False,     # ← Windows / TkAgg 稳定
    )

    plt.show()


def main():
    # ===== Global parameters =====
    L = 0.10                         # 5 cm per segment
    target_xy = np.array([0.20, 0.10])
    kappa_max = 100.0                  # curvature bound (1/m)
    tol = 0.02                        # 2 cm grasp tolerance

    for n_seg in [3, 4, 5]:
        animate_grasp(
            n_seg=n_seg,
            L=L,
            kappa_max=kappa_max,
            target_xy=target_xy,
            tol=tol,
            n_frames=70,
            interval=60,
        )


if __name__ == "__main__":
    main()
