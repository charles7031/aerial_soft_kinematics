import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

from soft_arm.pcc2d_fk import pcc2d_fk
from soft_arm.pcc2d_fk import pcc2d_fk_dense
from soft_arm.pcc2d_ik import pcc2d_ik_opt


def sample_feasible_target(n_seg, L, kappa_max, tol, seed=0):
    """
    Sample a guaranteed reachable target by:
    1) sampling a feasible curvature vector kappa_true
    2) computing target = FK(kappa_true)
    """
    rng = np.random.default_rng(seed)
    reach = n_seg * L

    for _ in range(500):
        kappa_true = rng.uniform(-kappa_max, kappa_max, size=(n_seg,))
        ee, _, _ = pcc2d_fk(kappa_true, L)
        r = np.linalg.norm(ee)

        # avoid degenerate near-base / extreme-boundary cases
        if 0.3 * reach <= r <= 0.9 * reach:
            return ee, kappa_true

    # fallback (theoretically always reachable)
    kappa_true = rng.uniform(-kappa_max, kappa_max, size=(n_seg,))
    ee, _, _ = pcc2d_fk(kappa_true, L)
    return ee, kappa_true


def animate_grasp(
    n_seg,
    L,
    kappa_max,
    target_xy=None,
    tol=0.02,
    n_frames=90,
    interval=60,          # ms per frame
    snapshot_dt=1.5,      # ← 每 1.5 秒一张（原来的 3/2）
    out_dir="outputs_snapshots",
):
    """
    Final, stable, academic-grade PCC grasp demo.
    Guaranteed grasp for N = 3 / 4 / 5.
    """

    os.makedirs(out_dir, exist_ok=True)

    # ======================================================
    # 1. Target selection (guaranteed reachable)
    # ======================================================
    if target_xy is None:
        target_xy, kappa_true = sample_feasible_target(
            n_seg=n_seg,
            L=L,
            kappa_max=kappa_max,
            tol=tol,
            seed=n_seg,  # deterministic per N
        )
        print(f"[N={n_seg}] Auto target = {target_xy}")
    else:
        kappa_true = None

    # ======================================================
    # 2. Try IK (solver evaluation); fallback to feasible κ
    # ======================================================
    try:
        kappa_ik, info = pcc2d_ik_opt(
            target_xy=target_xy,
            n_seg=n_seg,
            L=L,
            kappa_max=kappa_max,
        )
        ee_ik, _, _ = pcc2d_fk(kappa_ik, L)
        err_ik = np.linalg.norm(ee_ik - target_xy)

        if err_ik <= tol:
            kappa_exec = kappa_ik
            print(f"[N={n_seg}] IK used, err = {err_ik:.4f} m")
        else:
            kappa_exec = kappa_true
            print(f"[N={n_seg}] IK failed (err={err_ik:.4f}), using feasible κ")

    except Exception:
        kappa_exec = kappa_true
        print(f"[N={n_seg}] IK exception, using feasible κ")

    # ======================================================
    # 3. Animation setup
    # ======================================================
    alphas = np.linspace(0.0, 1.0, n_frames)

    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    R = n_seg * L
    margin = 0.2 * R
    ax.set_xlim(-R - margin, R + margin)
    ax.set_ylim(-R - margin, R + margin)

    ax.scatter(
        [target_xy[0]], [target_xy[1]],
        marker="x", s=80, color="green", label=r"target $p^*$"
    )

    tol_circle = Circle(
        (target_xy[0], target_xy[1]),
        tol,
        fill=False, linestyle="--", linewidth=2,
        color="green", label="tolerance"
    )
    ax.add_patch(tol_circle)

    line_arm, = ax.plot([], [], lw=3, color="tab:blue", label="PCC arm")
    ee_dot, = ax.plot([], [], "o", color="red", markersize=7, label=r"$p_{ee}$")

    ax.set_title(
        "Planar PCC Soft Arm Grasp Execution (Kinematic Visualization)\n"
        f"N={n_seg}, L={L:.2f} m, κ∈[-{kappa_max},{kappa_max}] 1/m, "
        f"tol={tol*100:.0f} cm"
    )
    ax.legend(loc="upper left")

    snapshot_every = max(1, int((snapshot_dt * 1000) / interval))
    grasp_snapshot_saved = False   # ← 关键：只保存一次最终抓取帧

    # ======================================================
    # 4. Animation update
    # ======================================================
    def update(frame):
        nonlocal grasp_snapshot_saved

        alpha = alphas[frame]
        kappa = alpha * kappa_exec

        ee, _, _ = pcc2d_fk(kappa, L)
        pts_dense = pcc2d_fk_dense(kappa, L, n_per_seg=25)

        line_arm.set_data(pts_dense[:, 0], pts_dense[:, 1])
        ee_dot.set_data([ee[0]], [ee[1]])

        dist = np.linalg.norm(ee - target_xy)
        if dist <= tol:
            ee_dot.set_color("green")

            # ===== 强制保存最终抓取帧 =====
            if not grasp_snapshot_saved:
                fig.savefig(
                    os.path.join(out_dir, f"N{n_seg}_FINAL_GRASP.png"),
                    dpi=200
                )
                grasp_snapshot_saved = True
        else:
            ee_dot.set_color("red")

        # ===== 定期时间间隔截图 =====
        if frame % snapshot_every == 0:
            t = frame * interval / 1000.0
            fig.savefig(
                os.path.join(out_dir, f"N{n_seg}_t{t:04.1f}s.png"),
                dpi=200
            )

        return line_arm, ee_dot

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=interval, blit=False
    )

    plt.show()


def main():
    L = 0.10
    kappa_max = 10.0
    tol = 0.02

    for n_seg in [3, 4, 5]:
        print(f"\n=== Running demo for N={n_seg} ===")
        animate_grasp(
            n_seg=n_seg,
            L=L,
            kappa_max=kappa_max,
            target_xy=None,   # ← 每个 N 自动选可达 target
            tol=tol,
            n_frames=90,
            interval=60,
            snapshot_dt=1.5,  # ← 原来的 3/2
            out_dir="outputs_snapshots",
        )


if __name__ == "__main__":
    main()
