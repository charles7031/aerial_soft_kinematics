import numpy as np

class WaypointPIDTracker:
    """
    Outer-loop waypoint tracker.
    Dynamics: translational only (ideal inner-loop attitude).
    """
    def __init__(self, m=1.2, g=9.81, dt=0.01,
                 Kp=(2,2,4), Kd=(2,2,3), Ki=(0,0,0.5),
                 max_acc=(3,3,5)):
        self.m = float(m)
        self.g = float(g)
        self.dt = float(dt)
        self.Kp = np.array(Kp, float)
        self.Kd = np.array(Kd, float)
        self.Ki = np.array(Ki, float)
        self.max_acc = np.array(max_acc, float)

        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.e_int = np.zeros(3)

    def reset(self, p0):
        self.p = np.array(p0, float).reshape(3)
        self.v[:] = 0.0
        self.e_int[:] = 0.0

    def step_to_waypoint(self, p_ref, v_ref=np.zeros(3)):
        p_ref = np.array(p_ref, float).reshape(3)
        v_ref = np.array(v_ref, float).reshape(3)

        e = p_ref - self.p
        edot = v_ref - self.v
        self.e_int += e * self.dt

        a_cmd = self.Kp*e + self.Kd*edot + self.Ki*self.e_int

        # saturate commanded acceleration (for stability)
        a_cmd = np.clip(a_cmd, -self.max_acc, self.max_acc)

        # gravity compensation in z
        a_cmd[2] += self.g

        # translational dynamics integrate
        a = a_cmd - np.array([0,0,self.g])  # net accel
        self.v += a * self.dt
        self.p += self.v * self.dt

        # total thrust -> distribute equally (since inner-loop ideal)
        T = self.m * a_cmd[2]
        thrusts = np.full(4, T/4.0)

        return self.p.copy(), self.v.copy(), thrusts, e
