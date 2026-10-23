"""Offline constrained stance-force allocation; not connected to robot actuation.

This solves a small convex force least-squares problem with a friction-pyramid
inner approximation. Predicted/optimized contact forces are NOT sensed GRFs.
Importing this module never starts PhysX or issues motor commands.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import LinearConstraint, minimize


def _skew(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _tangent_frame(normal):
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    helper = np.array([0., 0., 1.]) if abs(n[2]) < 0.9 else np.array([1., 0., 0.])
    t1 = np.cross(n, helper)
    t1 = t1 / np.linalg.norm(t1)
    return n, t1, np.cross(n, t1)


def allocate_stance_forces(
        foot_positions_rel_com_m, contact_normals_world, active_contact,
        desired_contact_wrench, friction_coefficient, max_normal_force_n,
        contact_confidence=None, previous_forces_n=None,
        wrench_weights=None, force_penalty=1e-5, change_penalty=1e-4,
        wrench_tolerance=0.5, max_iterations=250):
    """Allocate desired world-frame contact wrench [Fxyz,Mxyz] to stance feet.

    Args:
        foot_positions_rel_com_m: (n,3) foot positions relative to estimated CoM.
        contact_normals_world: (n,3) unit-ish *estimated* contact normals.
        active_contact: (n,) bool confirmed+planned support.
        desired_contact_wrench: (6,) desired TOTAL stance GRF wrench, INCLUDING
            gravity support, not merely a corrective wrench.
        friction_coefficient: conservative scalar mu (not simulator truth).
        max_normal_force_n: scalar or (n,) positive upper bound on each normal.
        contact_confidence: (n,) confidence in [0,1], bounds shrink accordingly.
        previous_forces_n: (n,3) last desired GRFs for smoothing, optional.

    Returns:
        forces_n: (n,3) full array with EXACT zero on inactive contacts.
        optimizer_success: numerical optimization and constraint feasibility.
        meets_wrench: true only if the actual optimization's wrench mismatch is
            below the requested (N/Nm mixed-unit) tolerance. Caller must decide
            appropriately scaled per-axis acceptance criteria.
        achieved_contact_wrench: total summed contact force and moment.
        wrench_residual: desired minus optimized wrench; objective_residual_norm:
            weighted norm. This is NOT a measured delivered wrench.
    """
    r = np.asarray(foot_positions_rel_com_m, dtype=float)
    normals = np.asarray(contact_normals_world, dtype=float)
    active = np.asarray(active_contact, dtype=bool)
    wrench = np.asarray(desired_contact_wrench, dtype=float)
    if r.ndim != 2 or r.shape[1] != 3 or normals.shape != r.shape or active.shape != (len(r),):
        raise ValueError("expected n foot positions, normals, and active mask")
    nfeet = len(r)
    if wrench.shape != (6,) or not np.all(np.isfinite(wrench)):
        raise ValueError("desired wrench must be finite (6,)")
    max_n = np.broadcast_to(np.asarray(max_normal_force_n, dtype=float), (nfeet,)).copy()
    conf = (np.ones(nfeet, dtype=float) if contact_confidence is None
            else np.asarray(contact_confidence, dtype=float))
    previous = (np.zeros((nfeet, 3), dtype=float) if previous_forces_n is None
                else np.asarray(previous_forces_n, dtype=float))
    weights = (np.ones(6, dtype=float) if wrench_weights is None
               else np.asarray(wrench_weights, dtype=float))
    if (conf.shape != (nfeet,) or previous.shape != r.shape or weights.shape != (6,)
            or not all(np.all(np.isfinite(x)) for x in (r, normals, max_n, conf, previous, weights))
            or np.any(np.linalg.norm(normals, axis=1) < 1e-8)
            or np.any(max_n <= 0) or np.any(conf < 0) or np.any(conf > 1)
            or np.any(weights <= 0) or not np.isfinite(friction_coefficient)
            or friction_coefficient < 0 or force_penalty < 0 or change_penalty < 0
            or wrench_tolerance < 0):
        raise ValueError("invalid finite force-allocation parameters")
    use = np.flatnonzero(active & (conf > 0))
    forces = np.zeros((nfeet, 3), dtype=float)
    if not len(use):
        return dict(forces_n=forces, optimizer_success=False, meets_wrench=False,
                    achieved_contact_wrench=np.zeros(6), wrench_residual=wrench.copy(),
                    objective_residual_norm=float(np.linalg.norm(weights * wrench)),
                    message="no confirmed active contacts")
    A = np.zeros((6, len(use) * 3), dtype=float)
    constraints = []
    lows = []
    highs = []
    for local, index in enumerate(use):
        sl = slice(3*local, 3*local+3)
        A[:3, sl] = np.eye(3)
        A[3:, sl] = _skew(r[index])
        nn, t1, t2 = _tangent_frame(normals[index])
        fac = float(friction_coefficient) / np.sqrt(2.)
        for direction, low, high in ((nn, 0., max_n[index]*conf[index]),
                                     (t1-fac*nn, -np.inf, 0.),
                                     (-t1-fac*nn, -np.inf, 0.),
                                     (t2-fac*nn, -np.inf, 0.),
                                     (-t2-fac*nn, -np.inf, 0.)):
            row = np.zeros(A.shape[1])
            row[sl] = direction
            constraints.append(row)
            lows.append(low)
            highs.append(high)
    C = np.asarray(constraints)
    low = np.asarray(lows)
    high = np.asarray(highs)
    W2 = weights ** 2
    fprevious = previous[use].ravel()

    def objective(x):
        e = A @ x - wrench
        return (np.dot(W2*e, e) + force_penalty*np.dot(x, x) +
                change_penalty*np.dot(x-fprevious, x-fprevious))

    def gradient(x):
        e = A @ x - wrench
        return 2.*(A.T@(W2*e) + force_penalty*x + change_penalty*(x-fprevious))

    result = minimize(objective, fprevious, jac=gradient, method="SLSQP",
                      constraints=[LinearConstraint(C, low, high)],
                      options={"ftol": 1e-10, "maxiter": max_iterations})
    if not np.all(np.isfinite(result.x)):
        result_x = np.zeros(len(use)*3, dtype=float)
    else:
        result_x = result.x
    constraint_violation = max(
        float(np.max(np.maximum(low - C@result_x, 0.))),
        float(np.max(np.maximum(C@result_x - high, 0.))))
    valid = bool(result.success and constraint_violation <= 1e-6)
    if valid:
        forces[use] = result_x.reshape(len(use), 3)
    achieved = A@result_x if valid else np.zeros(6)
    residual = wrench - achieved
    return dict(forces_n=forces, optimizer_success=valid,
                meets_wrench=bool(valid and np.linalg.norm(residual) <= wrench_tolerance),
                achieved_contact_wrench=achieved, wrench_residual=residual,
                objective_residual_norm=float(np.linalg.norm(weights*residual)),
                max_constraint_violation=constraint_violation,
                message=str(result.message))
