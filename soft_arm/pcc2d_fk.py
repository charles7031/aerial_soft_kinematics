# soft_arm/pcc2d_fk.py
import numpy as np


def _rot2(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]], dtype=float)


def pcc2d_fk(kappa: np.ndarray, L: float, base_xy=(0.0, 0.0), base_theta: float = 0.0):
    """
    Planar PCC forward kinematics for an N-segment constant-curvature arm.

    Args:
        kappa: shape (N,), curvature of each segment (1/m)
        L: segment length (m), fixed for all segments
        base_xy: (x, y) base position in world
        base_theta: base heading angle in world (rad)

    Returns:
        ee_xy: (2,) end-effector position
        ee_theta: end heading angle
        pts: (N+1, 2) polyline points along segment endpoints (for plotting)
    """
    kappa = np.asarray(kappa, dtype=float).reshape(-1)
    x, y = float(base_xy[0]), float(base_xy[1])
    theta = float(base_theta)

    pts = [(x, y)]

    eps = 1e-8
    for k in kappa:
        if abs(k) < eps:
            # Straight segment
            dp_local = np.array([L, 0.0], dtype=float)
            dtheta = 0.0
        else:
            # Arc in local frame (segment frame)
            dtheta = k * L
            dp_local = np.array([np.sin(dtheta) / k,
                                 (1.0 - np.cos(dtheta)) / k], dtype=float)

        dp_world = _rot2(theta) @ dp_local
        x += dp_world[0]
        y += dp_world[1]
        theta += dtheta

        pts.append((x, y))

    return np.array([x, y], dtype=float), float(theta), np.array(pts, dtype=float)

def pcc2d_fk_dense(kappa, L, n_per_seg=20):
    """
    Dense FK for visualization: sample each PCC segment as a circular arc.
    """
    kappa = np.asarray(kappa, dtype=float)
    x, y = 0.0, 0.0
    theta = 0.0

    pts = [(x, y)]

    for k in kappa:
        if abs(k) < 1e-8:
            # straight segment
            s_vals = np.linspace(0.0, L, n_per_seg)
            for s in s_vals[1:]:
                dx = s * np.cos(theta)
                dy = s * np.sin(theta)
                pts.append((x + dx, y + dy))
            x += L * np.cos(theta)
            y += L * np.sin(theta)
        else:
            s_vals = np.linspace(0.0, L, n_per_seg)
            for s in s_vals[1:]:
                dtheta = k * s
                dx = (np.sin(dtheta) / k) * np.cos(theta) \
                   - ((1 - np.cos(dtheta)) / k) * np.sin(theta)
                dy = (np.sin(dtheta) / k) * np.sin(theta) \
                   + ((1 - np.cos(dtheta)) / k) * np.cos(theta)
                pts.append((x + dx, y + dy))
            # advance frame
            dtheta = k * L
            x += (np.sin(dtheta) / k) * np.cos(theta) \
               - ((1 - np.cos(dtheta)) / k) * np.sin(theta)
            y += (np.sin(dtheta) / k) * np.sin(theta) \
               + ((1 - np.cos(dtheta)) / k) * np.cos(theta)
            theta += dtheta

    return np.array(pts)
