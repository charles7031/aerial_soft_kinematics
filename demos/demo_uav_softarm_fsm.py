"""
UAV + Soft Arm Grasp Demo (Course-aligned)
=========================================

课程主旨对齐点（你 PPT 可以直接用）：
1) 任务/行为规划：有限状态机 FSM（SEARCH -> PLAN_UAV -> TRACK_UAV -> PLAN_ARM -> GRASP -> DONE）
2) 感知与估计：目标观测 = 真值 + 噪声；一阶低通滤波（模拟传感器滤波/估计）
3) 运动规划：
   - UAV 在 3D 空间：RRT*（rrt_star_3d）
   - 软体机械臂末端在 2D 平面：RRT*（本文件内实现 rrt_star_2d）
4) 控制与跟踪：UAV 外环 Waypoint PID（WaypointPIDTracker）
5) 操作与逆解：软臂 PCC2D IK（pcc2d_ik_opt），把末端路径转成曲率序列

可视化要求（按你要的效果）：
- UAV 始终画成“+”形（黑色）
- UAV 下方挂 4 段串联 PCC 软臂（画在 x-z 平面，y 固定 = UAV y）
- 三维地图 + 障碍物 AABB（灰色线框）
- UAV 附近一个大球表示视野范围（灰色球壳）
- 目标进入视野后：滤波更新；规划 UAV 到“goal 斜上方 pregrasp 点”；到位后规划软臂末端 RRT* 并 IK 跟踪到抓取
- 存图 + 存 GIF

运行：
python -m demos.demo_uav_grasp_rrtstar
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# ========== 你的已有模块 ==========
from planners.rrt_star_3d import rrt_star_3d
from planners.rrt3d import shortcut_smooth_rrt
from uav.pid_tracker import WaypointPIDTracker
from soft_arm.pcc2d_fk import pcc2d_fk_dense
from soft_arm.pcc2d_ik import pcc2d_ik_opt


# ============================================================
# 0) 一些通用工具
# ============================================================
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

def aabb_wireframe(ax, lo, hi, alpha=0.25):
    """在 3D 里画 AABB 线框（灰色障碍物）"""
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    xs = [lo[0], hi[0]]
    ys = [lo[1], hi[1]]
    zs = [lo[2], hi[2]]
    corners = np.array([[x, y, z] for x in xs for y in ys for z in zs])
    edges = [
        (0,1),(0,2),(0,4),
        (3,1),(3,2),(3,7),
        (5,1),(5,4),(5,7),
        (6,2),(6,4),(6,7),
    ]
    for i, j in edges:
        ax.plot([corners[i,0], corners[j,0]],
                [corners[i,1], corners[j,1]],
                [corners[i,2], corners[j,2]],
                alpha=alpha)

def plot_uav_plus(ax, p, size=0.10, lw=2.5):
    """
    把 UAV 画成“+”形（黑色两条线段），在 xy 平面上画（z 作为高度点）
    """
    p = np.asarray(p, float)
    x, y, z = p.tolist()
    ax.plot([x-size, x+size], [y, y], [z, z], color="k", lw=lw)
    ax.plot([x, x], [y-size, y+size], [z, z], color="k", lw=lw)

def plot_vision_sphere(ax, center, R, alpha=0.08, n_u=18, n_v=12):
    """画视野球壳（线框/半透明）"""
    c = np.asarray(center, float)
    u = np.linspace(0, 2*np.pi, n_u)
    v = np.linspace(0, np.pi, n_v)
    xs = c[0] + R*np.outer(np.cos(u), np.sin(v))
    ys = c[1] + R*np.outer(np.sin(u), np.sin(v))
    zs = c[2] + R*np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, alpha=alpha)

def world_to_arm_plane(uav_p, goal_p):
    """
    我们把软臂运动平面固定为 UAV 的 x-z 平面（y 不变 = uav_y）：
    - PCC2D 平面坐标定义为 (x_pcc, y_pcc)：
        x_pcc: 水平（对应 world x）
        y_pcc: 向下为正（对应 world z 方向的负号）
    - 所以：
        x_pcc = goal_x - uav_x
        y_pcc = uav_z - goal_z   (因为 goal 在下方时 y_pcc>0)
    """
    u = np.asarray(uav_p, float)
    g = np.asarray(goal_p, float)
    x_pcc = g[0] - u[0]
    y_pcc = u[2] - g[2]
    return np.array([x_pcc, y_pcc], float)

def arm_plane_to_world(uav_p, pcc_xy):
    """
    把 PCC2D 平面点 (x_pcc, y_pcc) 映射回 3D 世界坐标：
      x_world = uav_x + x_pcc
      y_world = uav_y
      z_world = uav_z - y_pcc
    """
    u = np.asarray(uav_p, float)
    x_pcc, y_pcc = float(pcc_xy[0]), float(pcc_xy[1])
    return np.array([u[0] + x_pcc, u[1], u[2] - y_pcc], float)


# ============================================================
# 1) 2D RRT*（给软臂末端用，末端规划也用 RRT*）
# ============================================================
class Node2D:
    def __init__(self, p, parent=None, cost=0.0):
        self.p = np.asarray(p, float)
        self.parent = parent
        self.cost = float(cost)

def reconstruct_2d(n):
    path = []
    while n is not None:
        path.append(n.p)
        n = n.parent
    return np.array(path[::-1])

def rrt_star_2d(
    start_xy,
    goal_xy,
    bounds_xy,          # (lo_xy, hi_xy)
    circle_obstacles=None,  # [(c_xy, r), ...]  可空
    step_len=0.03,
    goal_sample_rate=0.25,
    max_iter=3000,
    goal_tol=0.03,
    search_radius=0.10,
    seed=0,
):
    """
    2D RRT*：用于软臂末端路径规划（在 PCC 平面里规划末端点轨迹）。
    为了课程匹配：这就是典型 sampling-based motion planning。

    circle_obstacles: 用圆表示“末端禁止区”（可选）。本 demo 默认不放末端障碍，
                      你后续可以加“末端不能碰到机身”等约束。
    """
    rng = np.random.default_rng(seed)
    start_xy = np.asarray(start_xy, float)
    goal_xy  = np.asarray(goal_xy,  float)
    lo, hi = bounds_xy

    if circle_obstacles is None:
        circle_obstacles = []

    def collision_free(p, q, n_samples=18):
        # 线段采样检查是否落入圆障碍
        for s in np.linspace(0.0, 1.0, n_samples):
            x = p + s*(q - p)
            for (c, r) in circle_obstacles:
                c = np.asarray(c, float)
                if np.linalg.norm(x - c) <= float(r):
                    return False
        return True

    nodes = [Node2D(start_xy, parent=None, cost=0.0)]

    for _ in range(max_iter):
        # 采样：以一定概率直接采 goal（加速收敛）
        if rng.random() < goal_sample_rate:
            p_rand = goal_xy
        else:
            p_rand = rng.uniform(lo, hi)

        # 最近点
        dists = [np.linalg.norm(n.p - p_rand) for n in nodes]
        n_near = nodes[int(np.argmin(dists))]

        # steer
        d = p_rand - n_near.p
        L = np.linalg.norm(d)
        if L < 1e-9:
            continue
        p_new = n_near.p + (step_len * d / L if L > step_len else d)

        if not collision_free(n_near.p, p_new):
            continue

        # near set（rewire）
        near_nodes = [n for n in nodes if np.linalg.norm(n.p - p_new) <= search_radius]

        # 选最优父节点
        parent = n_near
        min_cost = n_near.cost + np.linalg.norm(p_new - n_near.p)
        for n in near_nodes:
            if collision_free(n.p, p_new):
                c = n.cost + np.linalg.norm(p_new - n.p)
                if c < min_cost:
                    parent = n
                    min_cost = c

        new_node = Node2D(p_new, parent=parent, cost=min_cost)
        nodes.append(new_node)

        # rewire
        for n in near_nodes:
            c_new = new_node.cost + np.linalg.norm(n.p - new_node.p)
            if c_new < n.cost and collision_free(n.p, new_node.p):
                n.parent = new_node
                n.cost = c_new

        # 到达 goal
        if np.linalg.norm(p_new - goal_xy) < goal_tol:
            return reconstruct_2d(new_node)

    return None


# ============================================================
# 2) 环境（3D 障碍）与任务设定
# ============================================================
def build_static_aabbs():
    """
    3D 静态障碍：给 UAV RRT* 用（课程里的 map + obstacle）。
    你可以继续加更多盒子。
    """
    aabbs = []
    aabbs.append((np.array([0.25, 0.10, 0.00]), np.array([0.45, 0.55, 0.70])))
    aabbs.append((np.array([0.55, 0.45, 0.00]), np.array([0.85, 0.65, 0.80])))
    aabbs.append((np.array([0.15, 0.70, 0.00]), np.array([0.35, 0.90, 0.60])))
    return aabbs


# ============================================================
# 3) FSM 状态机（课程对齐：Task planning）
# ============================================================
class State:
    SEARCH   = "SEARCH"
    PLAN_UAV = "PLAN_UAV"
    TRACK_UAV= "TRACK_UAV"
    PLAN_ARM = "PLAN_ARM"
    GRASP    = "GRASP"
    DONE     = "DONE"


# ============================================================
# 4) 主 Demo（全程存 GIF）
# ============================================================
def run_demo(seed=0):
    out_dir = "outputs_uav_grasp_rrtstar"
    ensure_dir(out_dir)

    rng = np.random.default_rng(seed)

    # ---------- 任务目标 ----------
    goal_true = np.array([0.95, 0.10, 0.08])   # 目标在低处（便于“吊住”）
    start_uav = np.array([0.10, 0.10, 0.25])

    # “goal 斜上方”操作点（pregrasp hover point）
    # 斜上方：x/y 方向稍微偏一点，z 高一点
    pregrasp_offset = np.array([-0.18, +0.12, +0.30])
    pregrasp_true = goal_true + pregrasp_offset

    # ---------- 视野（感知） ----------
    vision_R = 0.55                  # 视野半径（球）
    meas_sigma = 0.04                # 观测噪声（m）
    alpha = 0.25                     # 一阶低通滤波系数（估计/滤波）
    goal_hat = None                  # 目标滤波估计

    # ---------- UAV 规划边界 ----------
    bounds_lo = np.array([0.0, 0.0, 0.0])
    bounds_hi = np.array([1.2, 1.2, 1.0])
    bounds = (bounds_lo, bounds_hi)
    aabbs = build_static_aabbs()

    # ---------- UAV 跟踪控制 ----------
    dt = 0.02
    tracker = WaypointPIDTracker(dt=dt)
    tracker.reset(start_uav)

    # ---------- 软臂（4 段 PCC） ----------
    n_seg = 4
    L_seg = 0.12
    kappa_max = 8.0
    # 初始软臂认为“竖直向下”：在 PCC 平面里就是 ee 起点在 (0, sum(L))
    ee_start_pcc = np.array([0.0, n_seg*L_seg], float)

    # ---------- FSM ----------
    state = State.SEARCH
    uav_waypoints = None
    uav_wp_id = 0

    arm_ee_path = None
    arm_path_id = 0
    kappa_cmd = np.zeros(n_seg, float)

    # ---------- 仿真/可视化时长 ----------
    T = 18.0
    steps = int(T / dt)

    # ---------- 记录 ----------
    uav_traj = []
    state_log = []

    # ---------- Matplotlib 动画 ----------
    fig = plt.figure(figsize=(8.6, 7.2))
    ax = fig.add_subplot(111, projection="3d")

    def setup_axes():
        ax.clear()
        ax.set_xlim(bounds_lo[0], bounds_hi[0])
        ax.set_ylim(bounds_lo[1], bounds_hi[1])
        ax.set_zlim(bounds_lo[2], bounds_hi[2])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(f"UAV + Soft Arm Grasp Demo | State = {state}")

    # 让 GIF 更直观：我们每帧都画“当前动态过程”
    frames = []

    # ========== 主循环 ==========
    for k in range(steps):
        t = k * dt

        p_uav = tracker.p.copy()
        uav_traj.append(p_uav.copy())
        state_log.append(state)

        # ---------- 感知：判断目标是否进入视野 ----------
        dist_to_goal = np.linalg.norm(goal_true - p_uav)
        in_view = dist_to_goal <= vision_R

        # 如果进入视野：产生带噪声观测 + 一阶滤波
        if in_view:
            z = goal_true + rng.normal(0.0, meas_sigma, size=3)  # noisy measurement
            if goal_hat is None:
                goal_hat = z.copy()
            else:
                goal_hat = (1 - alpha) * goal_hat + alpha * z

        # ====================================================
        # FSM：每个状态做什么（课程：任务规划/行为切换）
        # ====================================================
        if state == State.SEARCH:
            # 关键修复：SEARCH 一定要“动起来”
            # 用一个简单搜索策略：朝着 +x 方向缓慢推进（你也可以改成蛇形/螺旋）
            p_ref = p_uav + np.array([0.015, 0.000, 0.000])
            p_ref = clamp(p_ref, bounds_lo + 0.02, bounds_hi - 0.02)
            tracker.step_to_waypoint(p_ref)

            # 进入视野后切换到 UAV 规划
            if goal_hat is not None:
                state = State.PLAN_UAV

        elif state == State.PLAN_UAV:
            # 用滤波后的目标估计来计算 pregrasp（更“课程”：估计->规划）
            pregrasp_hat = goal_hat + pregrasp_offset

            # UAV 3D RRT* 规划：从当前 UAV 到 pregrasp
            path = rrt_star_3d(
                start=p_uav,
                goal=pregrasp_hat,
                bounds=bounds,
                aabbs=aabbs,
                step_len=0.12,
                goal_sample_rate=0.30,
                max_iter=5000,
                goal_tol=0.08,
                search_radius=0.35,   # 注意：你的 rrt_star_3d.py 用的是 search_radius，不是 rewire_radius
                seed=seed,
            )
            if path is None:
                # 规划失败：回到 SEARCH（或者你可以放宽参数）
                state = State.SEARCH
            else:
                # 可选：对 RRT* 路径做 shortcut 平滑（与课程一致：规划后处理）
                path = shortcut_smooth_rrt(path, aabbs=aabbs, n_iter=250, seed=seed)

                uav_waypoints = path
                uav_wp_id = 0
                state = State.TRACK_UAV

        elif state == State.TRACK_UAV:
            # 跟踪 UAV RRT* 路径点（Waypoint PID）
            if uav_waypoints is None or len(uav_waypoints) < 2:
                state = State.SEARCH
            else:
                p_ref = uav_waypoints[uav_wp_id]
                p_new, _, _, _ = tracker.step_to_waypoint(p_ref)

                # 到达当前 waypoint 则推进
                if np.linalg.norm(p_ref - p_new) < 0.05 and uav_wp_id < len(uav_waypoints) - 1:
                    uav_wp_id += 1

                # 到达末端：进入软臂规划
                if uav_wp_id >= len(uav_waypoints) - 1 and np.linalg.norm(uav_waypoints[-1] - p_new) < 0.06:
                    state = State.PLAN_ARM

        elif state == State.PLAN_ARM:
            # 软臂末端目标：在 UAV x-z 平面中，指向 goal（用滤波后的 goal_hat 更贴近课程）
            if goal_hat is None:
                state = State.SEARCH
            else:
                # 把目标投影到软臂平面坐标
                target_pcc = world_to_arm_plane(tracker.p.copy(), goal_hat)

                # 末端起点（PCC 平面）：(0, arm_len) 表示竖直向下
                start_pcc = ee_start_pcc.copy()

                # 给末端规划一个边界（相对 UAV 的局部平面范围）
                lo = np.array([-0.60, 0.02])
                hi = np.array([+0.60, 0.70])

                # 末端 RRT*（你要求：软臂末端规划也用 RRT*）
                ee_path = rrt_star_2d(
                    start_xy=start_pcc,
                    goal_xy=target_pcc,
                    bounds_xy=(lo, hi),
                    circle_obstacles=[],     # 这里可扩展成末端避障
                    step_len=0.03,
                    goal_sample_rate=0.30,
                    max_iter=3500,
                    goal_tol=0.03,
                    search_radius=0.10,
                    seed=seed + 7,
                )

                if ee_path is None:
                    # 规划失败：稍微放宽或回去 hover
                    state = State.SEARCH
                else:
                    arm_ee_path = ee_path
                    arm_path_id = 0
                    state = State.GRASP

                    # 额外存一张“末端规划图”（方便写报告）
                    fig_tmp = plt.figure()
                    plt.plot(ee_path[:,0], ee_path[:,1], "--", label="EE RRT* path (PCC plane)")
                    plt.scatter([start_pcc[0]],[start_pcc[1]], label="start")
                    plt.scatter([target_pcc[0]],[target_pcc[1]], label="goal")
                    plt.xlabel("x_pcc (m)")
                    plt.ylabel("y_pcc (down, m)")
                    plt.title("Soft Arm End-Effector Planning (RRT*)")
                    plt.grid(True)
                    plt.legend()
                    fig_tmp.savefig(os.path.join(out_dir, "arm_plan_rrtstar.png"), dpi=300, bbox_inches="tight")
                    plt.close(fig_tmp)

        elif state == State.GRASP:
            # 沿着末端路径逐点做 IK，并用“曲率指令 kappa_cmd”平滑逼近（像个简单控制器）
            if arm_ee_path is None:
                state = State.SEARCH
            else:
                target_xy = arm_ee_path[min(arm_path_id, len(arm_ee_path)-1)]

                # IK：求 4 段曲率使末端到 target_xy
                k_opt, info = pcc2d_ik_opt(
                    target_xy=target_xy,
                    n_seg=n_seg,
                    L=L_seg,
                    kappa_max=kappa_max,
                    base_xy=(0.0, 0.0),
                    base_theta=0.0,
                    reg=1e-3,
                )

                # 简单一阶跟踪（避免曲率瞬变）
                kappa_cmd = 0.85 * kappa_cmd + 0.15 * k_opt

                # 判断是否到达当前末端 waypoint
                ee_now = pcc2d_fk_dense(kappa_cmd, L_seg, n_per_seg=18)[-1]  # (x_pcc, y_pcc)
                if np.linalg.norm(ee_now - target_xy) < 0.02 and arm_path_id < len(arm_ee_path)-1:
                    arm_path_id += 1

                # 末端到最终目标：done
                if arm_path_id >= len(arm_ee_path)-1 and np.linalg.norm(ee_now - arm_ee_path[-1]) < 0.03:
                    state = State.DONE

        elif state == State.DONE:
            # DONE：保持悬停，不再变化
            pass

        # ====================================================
        # 画当前帧：让“动态过程”在 GIF 里真的看得见
        # ====================================================
        setup_axes()

        # 画障碍
        for (lo, hi) in aabbs:
            aabb_wireframe(ax, lo, hi, alpha=0.18)

        # 画目标（红点）
        ax.scatter(goal_true[0], goal_true[1], goal_true[2], s=90, color="r", label="goal")

        # 画 UAV（+）
        plot_uav_plus(ax, tracker.p.copy(), size=0.08, lw=3.0)

        # 画视野球
        plot_vision_sphere(ax, tracker.p.copy(), vision_R, alpha=0.07)

        # 画 UAV 轨迹（已走过）
        tr = np.array(uav_traj)
        ax.plot(tr[:,0], tr[:,1], tr[:,2], alpha=0.6, label="UAV trajectory")

        # 如果有 RRT* 规划的 UAV 路径，就画出来
        if uav_waypoints is not None and len(uav_waypoints) >= 2:
            ax.plot(uav_waypoints[:,0], uav_waypoints[:,1], uav_waypoints[:,2],
                    "--", alpha=0.9, label="UAV planned path (RRT*)")

        # 画软臂（挂在 UAV 下方，位于 y=uav_y 的 x-z 平面）
        # PCC2D 密集点：pts_pcc[:,0]=x_pcc, pts_pcc[:,1]=y_pcc(向下)
        pts_pcc = pcc2d_fk_dense(kappa_cmd, L_seg, n_per_seg=18)
        pts_world = np.array([arm_plane_to_world(tracker.p.copy(), xy) for xy in pts_pcc])
        ax.plot(pts_world[:,0], pts_world[:,1], pts_world[:,2], lw=3.0, label="soft arm (PCC x-z plane)")

        # 如果进入 GRASP 状态：画末端规划路径（映射到 world）
        if arm_ee_path is not None and len(arm_ee_path) >= 2:
            ee_world = np.array([arm_plane_to_world(tracker.p.copy(), xy) for xy in arm_ee_path])
            ax.plot(ee_world[:,0], ee_world[:,1], ee_world[:,2], "--", alpha=0.8, label="EE planned path (RRT*)")

        # 标注估计点（可选）
        if goal_hat is not None:
            ax.scatter(goal_hat[0], goal_hat[1], goal_hat[2], s=50, alpha=0.7, label="goal (filtered)")

        ax.legend(loc="upper left")
        fig.canvas.draw()

        # 把当前帧存入动画
        frames.append([ax])

        # 提前结束
        if state == State.DONE and k > int(2.0 / dt):
            break

    # ====================================================
    # 存 UAV 规划与跟踪对比图（静态图）
    # ====================================================
    fig_u = plt.figure()
    ax_u = fig_u.add_subplot(111, projection="3d")
    ax_u.set_title("UAV Planning + Tracking (RRT*)")
    for (lo, hi) in aabbs:
        aabb_wireframe(ax_u, lo, hi, alpha=0.18)
    tr = np.array(uav_traj)
    ax_u.plot(tr[:,0], tr[:,1], tr[:,2], label="tracked")
    if uav_waypoints is not None:
        ax_u.plot(uav_waypoints[:,0], uav_waypoints[:,1], uav_waypoints[:,2], "--", label="planned")
    ax_u.scatter(start_uav[0], start_uav[1], start_uav[2], s=60, label="start")
    ax_u.scatter(pregrasp_true[0], pregrasp_true[1], pregrasp_true[2], s=60, label="pregrasp")
    ax_u.scatter(goal_true[0], goal_true[1], goal_true[2], s=80, label="goal")
    ax_u.set_xlabel("x"); ax_u.set_ylabel("y"); ax_u.set_zlabel("z")
    ax_u.legend()
    fig_u.savefig(os.path.join(out_dir, "uav_path_rrtstar.png"), dpi=300, bbox_inches="tight")
    plt.close(fig_u)

    # ====================================================
    # 存最终大图（终帧）
    # ====================================================
    fig.savefig(os.path.join(out_dir, "uav_softarm_rrtstar.png"), dpi=300, bbox_inches="tight")

    # ====================================================
    # 存 GIF（动态过程）
    # ====================================================
    # 用 PillowWriter 输出 gif（Windows 下最稳）
    gif_path = os.path.join(out_dir, "uav_softarm_rrtstar.gif")
    ani = animation.ArtistAnimation(fig, frames, interval=int(dt*1000), blit=False)
    ani.save(gif_path, writer=animation.PillowWriter(fps=int(1/dt)))

    plt.close(fig)

    print(f"[OK] saved: {gif_path}")
    print(f"[OK] saved: {os.path.join(out_dir, 'uav_softarm_rrtstar.png')}")
    print(f"[OK] saved: {os.path.join(out_dir, 'uav_path_rrtstar.png')}")
    print(f"[OK] saved: {os.path.join(out_dir, 'arm_plan_rrtstar.png')}")


def main():
    run_demo(seed=0)

if __name__ == "__main__":
    main()
