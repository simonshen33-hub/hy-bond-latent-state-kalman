"""estimate.py -- pooled maximum likelihood for the baseline three-state model:

    theta_hat = argmax sum_i sum_t log p(y_{i,t} | y_{i,1:t-1}; theta)

one parameter set shared by all series in the panel. The optimiser works on the unconstrained vector
of model_v1 (tanh for phi, log / log-Cholesky for Q and H). Standard errors: inverse numerical Hessian
of the log-likelihood in u-space, mapped to natural parameters by the delta method.

Numerical settings (optimiser, tolerances, finite-difference steps) are the library defaults of
scipy.optimize.minimize and statsmodels.tools.numdiff unless the caller passes them; the start
point is always supplied by the caller.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from statsmodels.tools.numdiff import approx_fprime, approx_hess3

import model_v1
import ssm


def panel_loglik(nat, spec, y, ptr, of_rows):
    """Total log-likelihood and observed-row count. -inf if the system is inadmissible."""
    try:
        sys = model_v1.system(nat, spec, of_rows)
        a1, P1 = model_v1.initial(nat, spec)
    except (ValueError, np.linalg.LinAlgError):
        return -np.inf, 0
    ll, nobs = ssm.loglik(y, ptr, sys, a1, P1[None])
    return float(ll.sum()), int(nobs.sum())


def fit(spec, y, ptr, of_rows, start_nat, method='L-BFGS-B', options=None, se=True):
    """Pooled MLE. Returns dict(nat, u, ll, nobs, success, message, nit, se, cov_u)."""
    model_v1.check_nat(start_nat, spec)
    u0 = model_v1.to_u(start_nat, spec)
    _, n_obs = panel_loglik(start_nat, spec, y, ptr, of_rows)
    if n_obs == 0:
        raise ValueError('no observed rows in the panel')

    def obj(u):
        ll, _ = panel_loglik(model_v1.from_u(u, spec), spec, y, ptr, of_rows)
        return -ll / n_obs if np.isfinite(ll) else np.inf

    res = minimize(obj, u0, method=method, options=options)
    nat = model_v1.from_u(res.x, spec)
    ll, nobs = panel_loglik(nat, spec, y, ptr, of_rows)
    out = dict(nat=nat, u=res.x, ll=ll, nobs=nobs, success=bool(res.success), message=str(res.message),
               nit=int(getattr(res, 'nit', -1)), se=None, cov_u=None)
    if se:
        out.update(standard_errors(res.x, spec, y, ptr, of_rows))
    return out


def standard_errors(u, spec, y, ptr, of_rows):
    """Delta-method standard errors of the free natural parameters at u."""
    names = model_v1.free_names(spec)

    def tot(uu):
        ll, _ = panel_loglik(model_v1.from_u(uu, spec), spec, y, ptr, of_rows)
        return ll

    Hs = approx_hess3(np.asarray(u, dtype=float), tot)
    try:
        cov_u = np.linalg.inv(-Hs)
    except np.linalg.LinAlgError:
        return dict(se=None, cov_u=None, hess_u=Hs)

    def nat_vec(uu):
        d = model_v1.from_u(uu, spec)
        return np.array([d[k] for k in names])

    J = approx_fprime(np.asarray(u, dtype=float), nat_vec, centered=True)
    cov_nat = J @ cov_u @ J.T
    var = np.diag(cov_nat)
    se = {k: (float(np.sqrt(v)) if v >= 0 else np.nan) for k, v in zip(names, var)}
    return dict(se=se, cov_u=cov_u, cov_nat=cov_nat, hess_u=Hs)
