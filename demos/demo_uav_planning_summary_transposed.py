import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def load_img(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return mpimg.imread(path)

def main():
    out_dir = "outputs_uav_planning"
    save_path = os.path.join(out_dir, "planning_comparison_all_transposed.png")

    planners = ["astar", "rrt", "rrtstar"]
    titles = {
        "astar":   "A*",
        "rrt":     "RRT",
        "rrtstar": "RRT*"
    }

    # --------------------------------------------------
    # 2 rows × 3 columns
    # row 0: 3D scenes
    # row 1: error curves
    # --------------------------------------------------
    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(18, 10)
    )

    # ======================
    # Row 0: 3D scenes
    # ======================
    for j, p in enumerate(planners):
        img = load_img(os.path.join(out_dir, f"scene_{p}.png"))
        axes[0, j].imshow(img)
        axes[0, j].axis("off")
        axes[0, j].set_title(
            f"{titles[p]}: Planning + Tracking",
            fontsize=14
        )

    # ======================
    # Row 1: error curves
    # ======================
    for j, p in enumerate(planners):
        img = load_img(os.path.join(out_dir, f"error_{p}.png"))
        axes[1, j].imshow(img)
        axes[1, j].axis("off")
        axes[1, j].set_title(
            f"{titles[p]}: Tracking Error",
            fontsize=14
        )

    # ======================
    # Global title
    # ======================
    plt.suptitle(
        "UAV Motion Planning and Tracking Comparison\n"
        "A* vs RRT vs RRT*",
        fontsize=18
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"[OK] Transposed summary saved to:\n  {save_path}")

if __name__ == "__main__":
    main()
