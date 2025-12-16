# uav/quadrotor_model.py
import numpy as np

class QuadrotorPlus:
    """
    '+' configuration quadrotor:
      motor order: [front, left, rear, right]
      body axes: x forward, y left, z up
    """

    def __init__(self, diag=0.30, k_yaw=0.02):
        # diagonal tip distance = diag => arm length a = diag/sqrt(2)
        self.L = diag / np.sqrt(2.0)   # center -> motor distance
        self.k_yaw = float(k_yaw)      # toy yaw coefficient for mixing

        # motor positions in BODY frame (z=0 plane)
        self.motor_pos_body = np.array([
            [ self.L,  0.0, 0.0],   # front
            [ 0.0,  self.L, 0.0],   # left
            [-self.L,  0.0, 0.0],   # rear
            [ 0.0, -self.L, 0.0],   # right
        ], dtype=float)

    def mix(self, F, tau_x, tau_y, tau_z):
        """
        Mix total thrust + body torques into 4 motor thrusts.

        Equations:
          f1+f2+f3+f4 = F
          (f2 - f4)*L = tau_x
          (f3 - f1)*L = tau_y
          (f1 - f2 + f3 - f4)*k_yaw = tau_z

        Return:
          f: (4,) [front, left, rear, right]
        """
        L = self.L
        k = self.k_yaw

        f1 = 0.25 * F - 0.5 * (tau_y / L) + 0.25 * (tau_z / k)
        f2 = 0.25 * F + 0.5 * (tau_x / L) - 0.25 * (tau_z / k)
        f3 = 0.25 * F + 0.5 * (tau_y / L) + 0.25 * (tau_z / k)
        f4 = 0.25 * F - 0.5 * (tau_x / L) - 0.25 * (tau_z / k)

        return np.array([f1, f2, f3, f4], dtype=float)

    def motor_positions_world(self, p, R):
        """
        p: (3,) world position
        R: (3,3) rotation body->world
        """
        return (R @ self.motor_pos_body.T).T + p[None, :]
