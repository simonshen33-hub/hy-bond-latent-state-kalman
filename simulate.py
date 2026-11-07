"""simulate.py -- simulate the baseline three-state model (validation: simulation first).

Every input is supplied by the caller; nothing here has a default value:
    nat  : natural-space parameters (model_v1.all_names(spec) all present)
    spec : model_v1.Spec
    OF   : (n_series, n_t) order flow; OF[:, t] drives the transition t -> t+1
    mask : (n_series, n_t, p) bool, True where row r of y_t is observed (p = len(spec.rows))
    rng  : numpy Generator (the caller chooses the seed)
The initial state of each series is drawn from the stationary distribution used by the filter
(model_v1.initial), so simulation and filter share one initialisation.
"""
from __future__ import annotations

import numpy as np

import model_v1


def simulate(nat, spec, OF, mask, rng):
    """Returns states (n_series, n_t, 3) and y (n_series, n_t, p) with NaN where mask is False."""
    model_v1.check_nat(nat, spec)
    OF = np.asarray(OF, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    ns, nt = OF.shape
    p = len(spec.rows)
    if mask.shape != (ns, nt, p):
        raise ValueError(f'mask must be {(ns, nt, p)}, got {mask.shape}')
    if not np.all(np.isfinite(OF)):
        raise ValueError('OF has non-finite entries')
    mu = np.array([nat['mu_C'], nat['mu_L'], nat['mu_D']])
    phi = np.array([nat['phi_C'], nat['phi_L'], nat['phi_D']])
    Lq = np.linalg.cholesky(model_v1.q_matrix(nat))
    Lh = np.linalg.cholesky(model_v1.h_matrix(nat, spec.rows))
    Z = model_v1.z_matrix(nat, spec.rows)
    a_inf, P_inf = model_v1.initial(nat, spec)

    alpha = np.empty((ns, nt, 3))
    alpha[:, 0] = a_inf + rng.standard_normal((ns, 3)) @ np.linalg.cholesky(P_inf).T
    for t in range(nt - 1):
        nxt = mu + phi * (alpha[:, t] - mu) + rng.standard_normal((ns, 3)) @ Lq.T
        nxt[:, 2] -= nat['gamma'] * OF[:, t]
        alpha[:, t + 1] = nxt
    y = alpha @ Z.T + rng.standard_normal((ns, nt, p)) @ Lh.T
    y[~mask] = np.nan
    return alpha, y


def to_panel(y, OF):
    """(n_series, n_t, p) arrays -> flat rows + ptr for ssm (every day kept, unobserved = NaN)."""
    y = np.asarray(y, dtype=float)
    ns, nt, p = y.shape
    ptr = np.arange(ns + 1, dtype=np.int64) * nt
    return y.reshape(ns * nt, p), np.asarray(OF, dtype=float).reshape(ns * nt), ptr
