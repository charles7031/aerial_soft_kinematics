import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =========================================================
# 1. Parameters (physical & control)
# =========================================================
m = 1.2                 # UAV mass (kg)
g = 9.81                # gravity (m/s^2)
dt = 0.01               # simulation timestep (s)
T = 6.0                 # total simulation time (s)

# PID gains (POSITION LOOP)
Kp = np.array([2.0, 2.0, 4.0])
Kd = np.array([2.0, 2.0, 3.0])
Ki = np.array([0.0, 0.0, 0.5])

tol = 0.02              # position tolerance (m)

# =========================================================
# 2. Reference trajectory (up first, then forward)
# =========================================================
def reference_trajectory(t):
    """
    Piecewise-smooth reference:
    - z: 0 -> 1 m in 3 s
    - x,y: 0 -> 1 m from 3 s to 6 s
    """
    p_ref = np.zeros(3)
    v_ref = np.zeros(3)

    if t <= 3.0:
        s = 0.5 * (1 - np.cos(np.pi * t / 3.0))
        ds = 0.5 * np.pi / 3.0 * np.sin(np.pi * t / 3.0)
        p_ref[2] = s
        v_ref[2] = ds
    else:
        p_ref[2] = 1.0
        s = 0.5 * (1 - np.cos(np.pi * (t - 3.0) / 3.0))
        ds = 0.5 * np.pi / 3.0 * np.sin(np.pi * (t - 3.0) / 3.0)
        p_ref[0] = s
        p_ref[1] = s
        v_ref[0] = ds
        v_ref[1] = ds

    return p_ref, v_ref

# =========================================================
# 3. State initialization
# =========================================================
N = int(T / dt)
t_log = np.zeros(N)

p = np.zeros(3)      # position
v = np.zeros(3)      # velocity
p_int = np.zeros(3)  # integral error

p_log = np.zeros((N, 3))
p_ref_log = np.zeros((N, 3))
thrust_log = np.zeros((N, 4))

# =========================================================
# 4. Simulation loop
# =========================================================
for i in range(N):
    t = i * dt
    t_log[i] = t

    p_ref, v_ref = reference_trajectory(t)
    p_ref_log[i] = p_ref

    # --- position control (PID) ---
    e = p_ref - p
    edot = v_ref - v
    p_int += e * dt

    a_cmd = (
        Kp * e +
        Kd * edot +
        Ki * p_int
    )

    # --- gravity compensation ---
    a_cmd[2] += g

    # --- translational dynamics ---
    a = a_cmd
    v += a * dt
    p += v * dt

    # --- thrust distribution (simplified) ---
    total_thrust = m * a_cmd[2]
    thrust_log[i, :] = total_thrust / 4.0

    p_log[i] = p

# =========================================================
# 5. Plot & save results
# =========================================================
out_dir = "outputs_uav_demo"
os.makedirs(out_dir, exist_ok=True)

# --- 3D trajectory ---
fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111, projection="3d")
ax.plot(p_log[:,0], p_log[:,1], p_log[:,2], label="UAV trajectory")
ax.scatter(0,0,0, c="b", s=60, label="start")
ax.scatter(1,1,1, c="r", s=60, label="goal")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_zlabel("z (m)")
ax.legend()
ax.set_title("Quadrotor 3D Trajectory")

fig.savefig(os.path.join(out_dir, "uav_3d_trajectory.png"), dpi=300)
plt.show()

# --- xyz vs time ---
fig2 = plt.figure()
plt.plot(t_log, p_log[:,0], label="x")
plt.plot(t_log, p_log[:,1], label="y")
plt.plot(t_log, p_log[:,2], label="z")
plt.xlabel("time (s)")
plt.ylabel("position (m)")
plt.legend()
plt.grid()
fig2.savefig(os.path.join(out_dir, "uav_xyz_vs_time.png"), dpi=300)
plt.show()

# --- position error ---
err = np.linalg.norm(p_log - p_ref_log, axis=1)
fig3 = plt.figure()
plt.plot(t_log, err, label="||p - p*||")
plt.axhline(tol, linestyle="--", color="r", label="tol")
plt.xlabel("time (s)")
plt.ylabel("error (m)")
plt.legend()
plt.grid()
fig3.savefig(os.path.join(out_dir, "uav_position_error.png"), dpi=300)
plt.show()

# --- motor thrusts ---
fig4 = plt.figure()
for k in range(4):
    plt.plot(t_log, thrust_log[:,k], label=f"motor {k+1}")
plt.xlabel("time (s)")
plt.ylabel("thrust (N)")
plt.legend()
plt.grid()
fig4.savefig(os.path.join(out_dir, "uav_motor_thrusts.png"), dpi=300)
plt.show()
