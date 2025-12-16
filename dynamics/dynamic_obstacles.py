import numpy as np

class MovingAABB:
    """
    Axis-Aligned Bounding Box with time-varying position
    用于动态障碍（闸门 / 行人 / 其他 UAV）
    """

    def __init__(self, lo0, hi0, vel):
        """
        lo0, hi0 : 初始 AABB 边界 (3,)
        vel      : 速度向量 (3,) m/s
        """
        self.lo0 = np.asarray(lo0, float)
        self.hi0 = np.asarray(hi0, float)
        self.vel = np.asarray(vel, float)

    def at_time(self, t):
        """
        返回时间 t 时刻的 AABB
        """
        shift = self.vel * t
        return self.lo0 + shift, self.hi0 + shift


def get_dynamic_aabbs(t):
    """
    定义所有动态障碍（统一接口）
    返回：[(lo, hi), ...]
    """

    aabbs = []

    # ========== 动态障碍 1：移动闸门 ==========
    gate = MovingAABB(
        lo0=[0.45, 0.20, 0.00],
        hi0=[0.55, 0.30, 0.90],
        vel=[0.0, 0.15, 0.0],   # y 方向匀速移动
    )

    lo, hi = gate.at_time(t)
    # 往返运动（简单反射）
    if hi[1] > 0.80 or lo[1] < 0.20:
        gate.vel[1] *= -1

    aabbs.append((lo, hi))

    # ========== 动态障碍 2：横穿“行人” ==========
    walker = MovingAABB(
        lo0=[0.10, 0.60, 0.45],
        hi0=[0.20, 0.70, 0.55],
        vel=[0.20, 0.0, 0.0],   # x 方向穿越
    )

    lo, hi = walker.at_time(t)
    aabbs.append((lo, hi))

    return aabbs
