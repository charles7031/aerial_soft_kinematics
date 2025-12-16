# uav/se3_kinematics.py
import numpy as np


def rot_x(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]], dtype=float)


def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], dtype=float)


def rot_z(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=float)


def R_to_rpy(R):
    """
    Inverse of ZYX rpy_to_R.
    Returns roll, pitch, yaw.
    Handles standard (non-gimbal-lock) case robustly.
    """
    # pitch = asin(-R[2,0]) under ZYX
    sy = -R[2, 0]
    sy = np.clip(sy, -1.0, 1.0)
    pitch = np.arcsin(sy)

    # near gimbal lock
    eps = 1e-8
    if np.cos(pitch) < eps:
        # yaw and roll coupled; choose yaw = 0
        yaw = 0.0
        roll = np.arctan2(-R[0, 1], R[1, 1])
    else:
        yaw = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])

    return roll, pitch, yaw


def fk_uav(p, rpy):
    """
    p: (3,) position in world
    rpy: (3,) roll, pitch, yaw
    return: 4x4 homogeneous transform T_WB
    """
    p = np.asarray(p, dtype=float).reshape(3)
    rpy = np.asarray(rpy, dtype=float).reshape(3)
    R = rpy_to_R(rpy[0], rpy[1], rpy[2])

    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def ik_uav(T):
    """
    T: 4x4 transform
    return: p (3,), rpy (3,)
    """
    T = np.asarray(T, dtype=float).reshape(4, 4)
    R = T[:3, :3]
    p = T[:3, 3]
    roll, pitch, yaw = R_to_rpy(R)
    return p, np.array([roll, pitch, yaw], dtype=float)

def rpy_to_R(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]])
    Ry = np.array([[ cp, 0, sp],
                   [  0, 1,  0],
                   [-sp, 0, cp]])
    Rx = np.array([[1,  0,   0],
                   [0, cr, -sr],
                   [0, sr,  cr]])
    return Rz @ Ry @ Rx