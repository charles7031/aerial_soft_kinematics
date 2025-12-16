# demos/demo_workspace_parametric_sweep.py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

from soft_arm.pcc2d_fk import pcc2d_fk


def montecarlo_workspace(n_seg, L, kappa_max, n_samples, seed=0):
    rng = np.random.default_rng(seed)
    kappas = rng.uniform(-kappa_max, kappa_max, size=(n_samples, n_seg))

    ee_pts = np.zeros((n_samples, 2))
    for i in range(n_samples):
        ee, _, _ = pcc2d_fk(kappas[i], L)
        ee_pts[i] = ee
    return ee_pts


def workspace_area(ee_pts):
    if len(ee_pts) < 3:
        return 0.0
    hull = ConvexHull(ee_pts)
    return hull.area


def main():
    # ===== Parameters =====
    L = 0.10
    segment_list = [3, 4, 5]

    kappa_vals = np.linspace(0.0, 20.0, 21)   # κ = 0,1,2,...,20
    n_samples = 15000

    out_dir = "outputs_workspace_parametric"
    os.makedirs(out_dir, exist_ok=True)
    # ======================

    area_results = {n: [] for n in segment_list}

    print("=== Workspace parametric sweep: κ ∈ [0, 20] ===")

    for n_seg in segment_list:
        for kmax in kappa_vals:
            if kmax < 1e-6:
                # κ=0 ⇒ straight arm, area=0
                area_results[n_seg].append(0.0)
                continue

            ee_pts = montecarlo_workspace(
                n_seg=n_seg,
                L=L,
                kappa_max=kmax,
                n_samples=n_samples,
                seed=n_seg + int(kmax * 10),
            )

            area = workspace_area(ee_pts)
            area_results[n_seg].append(area)

        print(f"N={n_seg} done.")

    # =====================================================
    # Plot: Workspace area vs κ_max
    # =====================================================
    fig, ax = plt.subplots(figsize=(7, 5))

    for n_seg in segment_list:
        ax.plot(
            kappa_vals,
            area_results[n_seg],
            linewidth=2,
            marker="o",
            markersize=3,
            label=f"N = {n_seg}",
        )

    ax.set_xlabel(r"Curvature bound $\kappa_{\max}$ (1/m)")
    ax.set_ylabel("Workspace area (m²)")
    ax.set_title("Reachable Workspace vs Curvature Bound")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        os.path.join(out_dir, "workspace_area_vs_kappa_sweep.png"),
        dpi=300,
    )
    plt.show()

    print(f"Saved figure to: {out_dir}/workspace_area_vs_kappa_sweep.png")


if __name__ == "__main__":
    main()
