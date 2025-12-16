# soft_arm/pcc2d_ik.py
import numpy as np
from scipy.optimize import minimize
from soft_arm.pcc2d_fk import pcc2d_fk


def pcc2d_ik_opt(
    target_xy: np.ndarray,
    n_seg: int,
    L: float,
    kappa_max: float,
    base_xy=(0.0, 0.0),
    base_theta: float = 0.0,
    reg: float = 1e-3,
):
    """
    IK by bounded optimization:
        min_kappa ||ee(kappa)-target||^2 + reg*||kappa||^2

    Returns:
        kappa_opt: (n_seg,)
        info: dict with final error
    """
    target_xy = np.asarray(target_xy, dtype=float).reshape(2)
    bounds = [(-kappa_max, kappa_max)] * n_seg
    k0 = np.zeros(n_seg, dtype=float)

    def cost(kappa):
        ee, _, _ = pcc2d_fk(kappa, L, base_xy=base_xy, base_theta=base_theta)
        err2 = float(np.sum((ee - target_xy) ** 2))
        return err2 + reg * float(np.sum(np.asarray(kappa) ** 2))

    res = minimize(cost, k0, method="SLSQP", bounds=bounds)
    k_opt = np.asarray(res.x, dtype=float)

    ee, _, _ = pcc2d_fk(k_opt, L, base_xy=base_xy, base_theta=base_theta)
    err = float(np.linalg.norm(ee - target_xy))

    return k_opt, {"success": bool(res.success), "err": err, "message": res.message}