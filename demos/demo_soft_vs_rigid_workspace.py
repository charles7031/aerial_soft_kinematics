# demos/demo_soft_vs_rigid_workspace.py
import os
import numpy as np
import matplotlib.pyplot as plt

from soft_arm.pcc2d_fk import pcc2d_fk


# =========================================================
# Rigid serial arm forward kinematics (planar)
# =========================================================
def rigid_arm_fk(theta, L):
    """
    theta: (N,) joint angles
    L: segment length
    """
    x, y = 0.0, 0.0
    phi = 0.0

    for th in theta:
        phi += th
        x += L * np.cos(phi)
        y += L * np.sin(phi)

    return np.array([x, y])


# =========================================================
# Monte Carlo workspace sampling
# =========================================================
def montecarlo_soft_workspace(n_seg, L, kappa_max, n_samples, seed=0):
    rng = np.random.default_rng(seed)
    kappas = rng.uniform(-kappa_max, kappa_max, size=(n_samples, n_seg))

    ee_pts = np.zeros((n_samples, 2))
    for i in range(n_samples):
        ee, _, _ = pcc2d_fk(kappas[i], L)
        ee_pts[i] = ee
    return ee_pts


def montecarlo_rigid_workspace(n_seg, L, theta_max, n_samples, seed=0):
    rng = np.random.default_rng(seed)
    thetas = rng.uniform(-theta_max, theta_max, size=(n_samples, n_seg))

    ee_pts = np.zeros((n_samples, 2))
    for i in range(n_samples):
        ee_pts[i] = rigid_arm_fk(thetas[i], L)
    return ee_pts


# =========================================================
# Plot comparison
# =========================================================
def plot_soft_vs_rigid(
    ee_soft,
    ee_rigid,
    n_seg,
    L,
    kappa_max,
    theta_max,
    out_dir,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    reach = n_seg * L
    margin = 0.2 * reach

    # --- Soft arm ---
    axes[0].scatter(
        ee_soft[:, 0], ee_soft[:, 1],
        s=3, alpha=0.25, color="tab:blue"
    )
    axes[0].set_title("PCC Soft Arm")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")

    # --- Rigid arm ---
    axes[1].scatter(
        ee_rigid[:, 0], ee_rigid[:, 1],
        s=3, alpha=0.25, color="tab:orange"
    )
    axes[1].set_title("Rigid Serial Arm")
    axes[1].set_xlabel("x (m)")

    for ax in axes:
        ax.set_xlim(-reach - margin, reach + margin)
        ax.set_ylim(-reach - margin, reach + margin)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)

    fig.suptitle(
        f"Workspace Comparison (N={n_seg}, L={L:.2f} m)\n"
        f"Soft: κ∈[-{kappa_max},{kappa_max}]  |  "
        f"Rigid: θ∈[-{theta_max:.2f},{theta_max:.2f}] rad",
        fontsize=12,
    )

    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"soft_vs_rigid_N{n_seg}.png")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(fname, dpi=300)
    plt.close(fig)

    print(f"Saved: {fname}")


# =========================================================
# Main
# =========================================================
def main():
    # ===== Parameters =====
    L = 0.10
    segment_list = [3, 4, 5]
    kappa_max = 10.0
    theta_max = kappa_max * L
    n_samples = 30000
    out_dir = "outputs_soft_vs_rigid"
    # ======================

    for n_seg in segment_list:
        ee_soft = montecarlo_soft_workspace(
            n_seg=n_seg,
            L=L,
            kappa_max=kappa_max,
            n_samples=n_samples,
            seed=n_seg,
        )

        ee_rigid = montecarlo_rigid_workspace(
            n_seg=n_seg,
            L=L,
            theta_max=theta_max,
            n_samples=n_samples,
            seed=100 + n_seg,
        )

        plot_soft_vs_rigid(
            ee_soft=ee_soft,
            ee_rigid=ee_rigid,
            n_seg=n_seg,
            L=L,
            kappa_max=kappa_max,
            theta_max=theta_max,
            out_dir=out_dir,
        )


if __name__ == "__main__":
    main()
