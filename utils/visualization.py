# utils/visualization.py
import numpy as np

def draw_uav_cross_3d(ax, p, diag=0.30, color="k"):
    """
    Draw UAV as a 3D cross (+ shape).
    Diagonal distance between tips = diag (meters)
    """
    p = np.asarray(p).reshape(3)
    a = diag / np.sqrt(2)

    ex = np.array([1.0, 0.0, 0.0])
    ey = np.array([0.0, 1.0, 0.0])

    px1, px2 = p + a*ex, p - a*ex
    py1, py2 = p + a*ey, p - a*ey

    ax.plot([px1[0],px2[0]], [px1[1],px2[1]], [px1[2],px2[2]],
            color=color, lw=2)
    ax.plot([py1[0],py2[0]], [py1[1],py2[1]], [py1[2],py2[2]],
            color=color, lw=2)


def draw_sensing_sphere(ax, p, R):
    """
    Draw sensing range as a wireframe sphere.
    """
    p = np.asarray(p).reshape(3)
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = R*np.cos(u)*np.sin(v) + p[0]
    y = R*np.sin(u)*np.sin(v) + p[1]
    z = R*np.cos(v) + p[2]
    ax.plot_wireframe(x, y, z, color="gray", alpha=0.25)


def draw_pcc_arm(ax, base, pts_2d):
    """
    Draw a planar PCC arm under UAV in X-Z plane.
    pts_2d: (N,2), x forward, y downward
    """
    base = np.asarray(base).reshape(3)
    x = base[0] + pts_2d[:,0]
    y = np.full_like(x, base[1])
    z = base[2] - pts_2d[:,1]

    ax.plot(x, y, z, color="tab:blue", lw=3)
