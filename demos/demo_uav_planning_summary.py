import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def load_img(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return mpimg.imread(path)

def main():
    out_dir = "outputs_uav_planning"
    save_path = os.path.join(out_dir, "planning_comparison_all.png")

    planners = ["astar", "rrt", "rrtstar"]
    titles = {
        "astar":   "A* (Grid-based)",
        "rrt":     "RRT",
        "rrtstar": "RRT*"
    }

    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(14, 18)
    )

    for i, p in enumerate(planners):
        # ----------------------
        # Left: 3D scene
        # ----------------------
        scene_img = load_img(os.path.join(out_dir, f"scene_{p}.png"))
        axes[i, 0].imshow(scene_img)
        axes[i, 0].axis("off")
        axes[i, 0].set_title(
            f"{titles[p]}: Planning + Tracking",
            fontsize=14
        )

        # ----------------------
        # Right: error curve
        # ----------------------
        err_img = load_img(os.path.join(out_dir, f"error_{p}.png"))
        axes[i, 1].imshow(err_img)
        axes[i, 1].axis("off")
        axes[i, 1].set_title(
            f"{titles[p]}: Tracking Error",
            fontsize=14
        )

    plt.suptitle(
        "UAV Motion Planning and Tracking Comparison\n"
        "A* vs RRT vs RRT*",
        fontsize=18
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"[OK] Summary figure saved to:\n  {save_path}")

if __name__ == "__main__":
    main()
