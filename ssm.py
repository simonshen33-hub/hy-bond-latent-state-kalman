"""ssm.py -- textbook linear-Gaussian state-space model (Durbin & Koopman 2012, ch. 4).

    y_t         = Z_t alpha_t + d_t + eta_t,           eta_t     ~ N(0, H_t)
    alpha_{t+1} = T_t alpha_t + c_t + R_t eps_{t+1},   eps_{t+1} ~ N(0, Q_t)

Prediction   a_{t|t-1} = T_{t-1} a_{t-1|t-1} + c_{t-1},   P_{t|t-1} = T P T' + R Q R'
Update       v = y - Z a - d,   F = Z P Z' + H,   K = P Z' F^-1,
             a_{t|t} = a_{t|t-1} + K v,   P_{t|t} = P_{t|t-1} - K Z P_{t|t-1}
Loglik       l_t = -1/2 (p_t log 2pi + log|F_t| + v' F_t^-1 v),   p_t = number of observed rows
Missing      NaN rows of y_t are dropped from Z_t, d_t, H_t (Lecture 1: H_jj -> inf).  With no row
             observed: a_{t|t} = a_{t|t-1}, P_{t|t} = P_{t|t-1}, l_t = 0 (prediction step only).

Panel layout: independent series (bonds) share the code path, each with its own rows.
    rows of series b : ptr[b] .. ptr[b+1]-1, in time order
System arrays carry a leading row axis of length 1 (constant) or n_rows (aligned with y):
    Z (., p, m)   d (., p)   H (., p, p)   T (., m, m)   c (., m)   RQR (., m, m)
T, c, RQR on row g give the transition FROM row g TO row g+1 of the same series.
a1 (1 or n_series, m), P1 (1 or n_series, m, m): prior a_{1|0}, P_{1|0} of each series.

This module sets no model numbers: every value enters through the arguments.

Public API
----------
as_system(Z, d, H, T, c, RQR, n_rows) -> dict        shape-checked, contiguous float64 arrays
kfilter(y, ptr, sys, a1, P1, store=True) -> dict     loglik per series (+ moments if store)
loglik(y, ptr, sys, a1, P1) -> ndarray               loglik per series only (parallel over series)
smooth(res, ptr, sys) -> dict                        Rauch-Tung-Striebel smoother (historical only)
stationary(T, c, RQR) -> (a_inf, P_inf)              a_inf = (I-T)^-1 c,  P_inf = T P T' + RQR
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange
from scipy.linalg import solve_discrete_lyapunov

LOG2PI = math.log(2.0 * math.pi)


# ----------------------------------------------------------------------------------------------
# small dense linear algebra (numba, no exceptions: non-PD F is reported, not raised)
# ----------------------------------------------------------------------------------------------
@njit(cache=True)
def _chol(A, L, n):
    """Lower Cholesky factor of the leading n x n block of symmetric A into L.
    Returns False if that block is not positive definite."""
    for j in range(n):
        s = A[j, j]
        for k in range(j):
            s -= L[j, k] * L[j, k]
        if not (s > 0.0):
            return False
        L[j, j] = math.sqrt(s)
        for i in range(j + 1, n):
            s = A[i, j]
            for k in range(j):
                s -= L[i, k] * L[j, k]
            L[i, j] = s / L[j, j]
    return True


@njit(cache=True)
def _chol_solve(L, X, n, ncol):
    """In place: X[:n, :ncol] <- (L L')^-1 X[:n, :ncol] (leading n x n block of L)."""
    for col in range(ncol):
        for i in range(n):                      # forward: L z = b
            s = X[i, col]
            for j in range(i):
                s -= L[i, j] * X[j, col]
            X[i, col] = s / L[i, i]
        for i in range(n - 1, -1, -1):          # backward: L' x = z
            s = X[i, col]
            for j in range(i + 1, n):
                s -= L[j, i] * X[j, col]
            X[i, col] = s / L[i, i]


@njit(cache=True)
def _row(n_lead, g):
    return g if n_lead > 1 else 0


# ----------------------------------------------------------------------------------------------
# one series (work buffers allocated once per series; matrices are small, loops are explicit)
# ----------------------------------------------------------------------------------------------
@njit(cache=True)
def _series(y, g0, g1, Z, d, H, T, c, RQR, a1, P1, store,
            A_pred, P_pred, A_filt, P_filt, V, F_out):
    """Filter rows g0..g1-1. Returns (loglik, n_obs, ok). ok=False when some F_t is not PD."""
    m = a1.shape[0]
    p = y.shape[1]
    a = a1.copy()
    P = P1.copy()
    an = np.empty(m)
    TP = np.empty((m, m))
    idx = np.empty(p, np.int64)
    Zs = np.empty((p, m))
    v = np.empty((p, 1))
    PZt = np.empty((m, p))
    F = np.empty((p, p))
    L = np.zeros((p, p))
    Fv = np.empty((p, 1))
    X = np.empty((p, m))                        # F^-1 Z P
    ll = 0.0
    nobs = 0
    for g in range(g0, g1):
        if store:
            A_pred[g, :] = a
            P_pred[g, :, :] = P
        k = 0
        for j in range(p):
            if not np.isnan(y[g, j]):
                idx[k] = j
                k += 1
        if k > 0:
            zi = _row(Z.shape[0], g)
            di = _row(d.shape[0], g)
            hi = _row(H.shape[0], g)
            for r in range(k):
                j = idx[r]
                s = y[g, j] - d[di, j]
                for q in range(m):
                    Zs[r, q] = Z[zi, j, q]
                    s -= Z[zi, j, q] * a[q]
                v[r, 0] = s
                Fv[r, 0] = s
            for q in range(m):                  # PZt = P Zs'
                for r in range(k):
                    s = 0.0
                    for q2 in range(m):
                        s += P[q, q2] * Zs[r, q2]
                    PZt[q, r] = s
            for r in range(k):                  # F = Zs P Zs' + Hs (symmetric by construction)
                for r2 in range(r + 1):
                    s = 0.5 * (H[hi, idx[r], idx[r2]] + H[hi, idx[r2], idx[r]])
                    for q in range(m):
                        s += Zs[r, q] * PZt[q, r2]
                    F[r, r2] = s
                    F[r2, r] = s
            if not _chol(F, L, k):
                return -np.inf, nobs, False
            _chol_solve(L, Fv, k, 1)
            for r in range(k):
                for q in range(m):
                    X[r, q] = PZt[q, r]
            _chol_solve(L, X, k, m)
            logdet = 0.0
            quad = 0.0
            for r in range(k):
                logdet += 2.0 * math.log(L[r, r])
                quad += v[r, 0] * Fv[r, 0]
            ll += -0.5 * (k * LOG2PI + logdet + quad)
            nobs += k
            for q in range(m):                  # a += P Z' F^-1 v ;  P -= P Z' F^-1 Z P
                s = 0.0
                for r in range(k):
                    s += PZt[q, r] * Fv[r, 0]
                a[q] += s
            for q in range(m):
                for q2 in range(q + 1):
                    s = 0.0
                    for r in range(k):
                        s += PZt[q, r] * X[r, q2]
                    val = 0.5 * ((P[q, q2] - s) + (P[q2, q] - s))
                    P[q, q2] = val
                    P[q2, q] = val
            if store:
                for r in range(k):
                    V[g, idx[r]] = v[r, 0]
                    for r2 in range(k):
                        F_out[g, idx[r], idx[r2]] = F[r, r2]
        if store:
            A_filt[g, :] = a
            P_filt[g, :, :] = P
        if g < g1 - 1:
            ti = _row(T.shape[0], g)
            ci = _row(c.shape[0], g)
            qi = _row(RQR.shape[0], g)
            for q in range(m):                  # a <- T a + c
                s = c[ci, q]
                for q2 in range(m):
                    s += T[ti, q, q2] * a[q2]
                an[q] = s
            for q in range(m):
                a[q] = an[q]
            for q in range(m):                  # TP = T P
                for q2 in range(m):
                    s = 0.0
                    for q3 in range(m):
                        s += T[ti, q, q3] * P[q3, q2]
                    TP[q, q2] = s
            for q in range(m):                  # P <- T P T' + RQR
                for q2 in range(q + 1):
                    s = 0.0
                    for q3 in range(m):
                        s += TP[q, q3] * T[ti, q2, q3]
                    val = s + 0.5 * (RQR[qi, q, q2] + RQR[qi, q2, q])
                    P[q, q2] = val
                    P[q2, q] = val
    return ll, nobs, True


@njit(parallel=True, cache=True)
def _panel_ll(y, ptr, Z, d, H, T, c, RQR, a1, P1):
    nb = ptr.shape[0] - 1
    out = np.empty(nb)
    nob = np.empty(nb, np.int64)
    m = a1.shape[1]
    p = y.shape[1]
    for b in prange(nb):
        e2 = np.empty((0, m))
        e3 = np.empty((0, m, m))
        ev = np.empty((0, p))
        ef = np.empty((0, p, p))
        bi = np.int64(b)
        ai = _row(a1.shape[0], bi)
        pi = _row(P1.shape[0], bi)
        ll, n, ok = _series(y, ptr[bi], ptr[bi + 1], Z, d, H, T, c, RQR, a1[ai], P1[pi], False,
                            e2, e3, e2, e3, ev, ef)
        out[b] = ll if ok else -np.inf
        nob[b] = n
    return out, nob


@njit(cache=True)
def _panel_store(y, ptr, Z, d, H, T, c, RQR, a1, P1, A_pred, P_pred, A_filt, P_filt, V, F_out):
    nb = ptr.shape[0] - 1
    out = np.empty(nb)
    nob = np.empty(nb, np.int64)
    for b in range(nb):
        ai = _row(a1.shape[0], b)
        pi = _row(P1.shape[0], b)
        ll, n, ok = _series(y, ptr[b], ptr[b + 1], Z, d, H, T, c, RQR, a1[ai], P1[pi], True,
                            A_pred, P_pred, A_filt, P_filt, V, F_out)
        out[b] = ll if ok else -np.inf
        nob[b] = n
    return out, nob


@njit(cache=True)
def _rts(ptr, T, A_pred, P_pred, A_filt, P_filt, A_sm, P_sm):
    nb = ptr.shape[0] - 1
    for b in range(nb):
        g0, g1 = ptr[b], ptr[b + 1]
        if g1 <= g0:
            continue
        A_sm[g1 - 1] = A_filt[g1 - 1]
        P_sm[g1 - 1] = P_filt[g1 - 1]
        for g in range(g1 - 2, g0 - 1, -1):
            Tg = T[_row(T.shape[0], g)]
            # J = P_{t|t} T' P_{t+1|t}^-1
            J = np.linalg.solve(P_pred[g + 1], Tg @ P_filt[g]).T
            A_sm[g] = A_filt[g] + J @ (A_sm[g + 1] - A_pred[g + 1])
            Pg = P_filt[g] + J @ (P_sm[g + 1] - P_pred[g + 1]) @ J.T
            P_sm[g] = 0.5 * (Pg + Pg.T)


# ----------------------------------------------------------------------------------------------
# python API
# ----------------------------------------------------------------------------------------------
def _lead(x, ndim_tail, name, n_rows):
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    if x.ndim == ndim_tail:
        x = x[None]
    if x.ndim != ndim_tail + 1:
        raise ValueError(f'{name}: expected {ndim_tail} or {ndim_tail + 1} dims, got {x.ndim}')
    if x.shape[0] not in (1, n_rows):
        raise ValueError(f'{name}: leading axis must be 1 or n_rows={n_rows}, got {x.shape[0]}')
    if not np.all(np.isfinite(x)):
        raise ValueError(f'{name}: non-finite entries')
    return np.ascontiguousarray(x)


def as_system(Z, d, H, T, c, RQR, n_rows):
    """Validate and broadcast system arrays (see module docstring for shapes)."""
    s = dict(Z=_lead(Z, 2, 'Z', n_rows), d=_lead(d, 1, 'd', n_rows), H=_lead(H, 2, 'H', n_rows),
             T=_lead(T, 2, 'T', n_rows), c=_lead(c, 1, 'c', n_rows), RQR=_lead(RQR, 2, 'RQR', n_rows))
    p, m = s['Z'].shape[1:]
    if s['d'].shape[1] != p or s['H'].shape[1:] != (p, p):
        raise ValueError('d / H do not match Z rows')
    if s['T'].shape[1:] != (m, m) or s['c'].shape[1] != m or s['RQR'].shape[1:] != (m, m):
        raise ValueError('T / c / RQR do not match the state dimension')
    return s


def _check_panel(y, ptr, sys, a1, P1):
    y = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
    ptr = np.ascontiguousarray(np.asarray(ptr, dtype=np.int64))
    if y.ndim != 2 or y.shape[1] != sys['Z'].shape[1]:
        raise ValueError('y must be (n_rows, p) matching Z')
    if ptr[0] != 0 or ptr[-1] != y.shape[0] or np.any(np.diff(ptr) < 0):
        raise ValueError('ptr must run 0..n_rows, non-decreasing')
    if np.any(np.isinf(y)):
        raise ValueError('y: inf entries (use NaN for missing)')
    m = sys['Z'].shape[2]
    nb = ptr.shape[0] - 1
    a1 = np.ascontiguousarray(np.atleast_2d(np.asarray(a1, dtype=np.float64)))
    P1 = np.asarray(P1, dtype=np.float64)
    P1 = np.ascontiguousarray(P1[None] if P1.ndim == 2 else P1)
    if a1.shape[1] != m or a1.shape[0] not in (1, nb):
        raise ValueError('a1 must be (1 or n_series, m)')
    if P1.shape[1:] != (m, m) or P1.shape[0] not in (1, nb):
        raise ValueError('P1 must be (1 or n_series, m, m)')
    return y, ptr, a1, P1


def loglik(y, ptr, sys, a1, P1):
    """Log-likelihood of each series (-inf where some F_t is not PD) and its count of observed rows."""
    y, ptr, a1, P1 = _check_panel(y, ptr, sys, a1, P1)
    return _panel_ll(y, ptr, sys['Z'], sys['d'], sys['H'], sys['T'], sys['c'], sys['RQR'], a1, P1)


def kfilter(y, ptr, sys, a1, P1, store=True):
    """Run the filter. Returns dict: ll (n_series,), nobs (n_series,), and if store:
    a_pred/P_pred = a_{t|t-1}, P_{t|t-1};  a_filt/P_filt = a_{t|t}, P_{t|t};
    v (n_rows, p) and F (n_rows, p, p), NaN where the row was not observed."""
    if not store:
        ll, nob = loglik(y, ptr, sys, a1, P1)
        return dict(ll=ll, nobs=nob)
    y, ptr, a1, P1 = _check_panel(y, ptr, sys, a1, P1)
    n, p = y.shape
    m = sys['Z'].shape[2]
    A_pred = np.full((n, m), np.nan)
    P_pred = np.full((n, m, m), np.nan)
    A_filt = np.full((n, m), np.nan)
    P_filt = np.full((n, m, m), np.nan)
    V = np.full((n, p), np.nan)
    F = np.full((n, p, p), np.nan)
    ll, nob = _panel_store(y, ptr, sys['Z'], sys['d'], sys['H'], sys['T'], sys['c'], sys['RQR'],
                           a1, P1, A_pred, P_pred, A_filt, P_filt, V, F)
    return dict(ll=ll, nobs=nob, a_pred=A_pred, P_pred=P_pred, a_filt=A_filt, P_filt=P_filt, v=V, F=F)


def smooth(res, ptr, sys):
    """RTS smoother a_{t|n}, P_{t|n} from a stored kfilter result. Report: for historical decomposition
    only, never for out-of-sample evaluation."""
    ptr = np.asarray(ptr, dtype=np.int64)
    A_sm = np.full_like(res['a_filt'], np.nan)
    P_sm = np.full_like(res['P_filt'], np.nan)
    _rts(ptr, sys['T'], res['a_pred'], res['P_pred'], res['a_filt'], res['P_filt'], A_sm, P_sm)
    return dict(a_sm=A_sm, P_sm=P_sm)


def stationary(T, c, RQR):
    """Stationary mean and covariance of alpha_{t+1} = T alpha_t + c + R eps (constant system).
    Raises if T has an eigenvalue on or outside the unit circle (no stationary distribution)."""
    T = np.asarray(T, dtype=np.float64)
    if np.max(np.abs(np.linalg.eigvals(T))) >= 1.0:
        raise ValueError('T is not stable: no stationary distribution')
    m = T.shape[0]
    a_inf = np.linalg.solve(np.eye(m) - T, np.asarray(c, dtype=np.float64))
    P_inf = solve_discrete_lyapunov(T, np.asarray(RQR, dtype=np.float64))
    return a_inf, 0.5 * (P_inf + P_inf.T)
