# demos/demo_montecarlo_pcc2d_scatter.py
import os
import numpy as np
import matplotlib.pyplot as plt

from soft_arm.pcc2d_fk import pcc2d_fk


def montecarlo_workspace(
    n_seg: int,
    L: float,
    kappa_max: float,
    n_samples: int,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)

    kappas = rng.uniform(
        -kappa_max, kappa_max,
        size=(n_samples, n_seg)
    )

    ee_pts = np.zeros((n_samples, 2), dtype=float)

    for i in range(n_samples):
        ee, _, _ = pcc2d_fk(kappas[i], L)
        ee_pts[i] = ee

    return ee_pts


def plot_workspace_on_ax(
    ax,
    ee_pts: np.ndarray,
    n_seg: int,
    L: float,
    kappa_max: float,
    target_xy: np.ndarray,
    tol: float,
):
    """
    Draw one workspace scatter on a given axis.
    """
    dists = np.linalg.norm(ee_pts - target_xy[None, :], axis=1)
    success = dists <= tol
    success_rate = success.mean()

    ax.scatter(
        ee_pts[~success, 0],
        ee_pts[~success, 1],
        s=3,
        alpha=0.25,
        color="tab:blue",
    )
    ax.scatter(
        ee_pts[success, 0],
        ee_pts[success, 1],
        s=8,
        alpha=0.9,
        color="tab:red",
    )

    ax.scatter(
        [target_xy[0]], [target_xy[1]],
        marker="x", s=40, color="black"
    )

    circle = plt.Circle(
        (target_xy[0], target_xy[1]),
        tol,
        fill=False,
        linestyle="--",
        linewidth=1.5,
        color="black",
    )
    ax.add_patch(circle)

    reach = n_seg * L
    margin = 0.2 * reach
    ax.set_xlim(-reach - margin, reach + margin)
    ax.set_ylim(-reach - margin, reach + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    ax.set_title(
        f"N={n_seg}, κ={kappa_max}\n"
        f"{success_rate*100:.1f}% success",
        fontsize=9,
    )

    return success_rate


def main():
    # ===== Parameters =====
    L = 0.10
    segment_list = [3, 4, 5]
    kappa_max_list = [5.0, 10.0, 15.0]
    n_samples = 30000
    target_xy = np.array([0.25, 0.10])
    tol = 0.01
    out_dir = "outputs_montecarlo_2d"
    # ======================

    os.makedirs(out_dir, exist_ok=True)

    print("=== PCC 2D Monte Carlo Workspace ===")
    print(f"Target = {target_xy.tolist()}, tol = {tol} m")

    # =========================================================
    # 1. Individual figures (unchanged behavior)
    # =========================================================
    results = {}  # cache for combined plot

    for n_seg in segment_list:
        for kmax in kappa_max_list:
            ee_pts = montecarlo_workspace(
                n_seg=n_seg,
                L=L,
                kappa_max=kmax,
                n_samples=n_samples,
                seed=n_seg + int(kmax * 10),
            )

            results[(n_seg, kmax)] = ee_pts

            # ---- single plot ----
            fig, ax = plt.subplots()
            rate = plot_workspace_on_ax(
                ax=ax,
                ee_pts=ee_pts,
                n_seg=n_seg,
                L=L,
                kappa_max=kmax,
                target_xy=target_xy,
                tol=tol,
            )

            fname = os.path.join(
                out_dir,
                f"workspace_scatter_N{n_seg}_k{int(kmax)}.png"
            )
            fig.savefig(fname, dpi=200)
            plt.close(fig)

            print(
                f"N={n_seg}, κ_max={kmax:>4.1f}  "
                f"success={rate*100:>6.2f}%"
            )

    # =========================================================
    # 2. Combined 3×3 figure
    # =========================================================
    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(12, 12),
        sharex=False,
        sharey=False,
    )

    for i, n_seg in enumerate(segment_list):
        for j, kmax in enumerate(kappa_max_list):
            ax = axes[i, j]
            ee_pts = results[(n_seg, kmax)]

            plot_workspace_on_ax(
                ax=ax,
                ee_pts=ee_pts,
                n_seg=n_seg,
                L=L,
                kappa_max=kmax,
                target_xy=target_xy,
                tol=tol,
            )

            if i == 0:
                ax.set_xlabel(f"κ_max = {kmax}", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"N = {n_seg}", fontsize=10)

    fig.suptitle(
        "PCC 2D Monte Carlo End-Effector Workspace\n"
        "Rows: number of segments, Columns: curvature bound",
        fontsize=14,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(
        os.path.join(out_dir, "workspace_scatter_3x3.png"),
        dpi=300,
    )
    plt.close(fig)

    print(f"Saved all figures to: {out_dir}/")


if __name__ == "__main__":
    main()
