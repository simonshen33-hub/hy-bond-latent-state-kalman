"""afns.py -- shadow-rate arbitrage-free Nelson-Siegel model, B-AFNS(3), independent-factor version, for the daily
Treasury zero curve (step 1 of the two-step estimation; see README, 'Rate factor').

Model (Christensen, Diebold and Rudebusch 2011; shadow rate as in Christensen and Rudebusch 2015, with the
Krippner 2012/2013 option-based forward rate; lower bound 0):
  state X = (L, S, C); shadow short rate r = L + S
  Q:  dX = K^Q (theta^Q - X) dt + Sigma dW^Q,  theta^Q = 0,  K^Q = [[0,0,0],[0,lam,-lam],[0,0,lam]]
  P:  dX = K^P (theta^P - X) dt + Sigma dW^P,  K^P = diag(k1,k2,k3),  Sigma = diag(s1,s2,s3)   (independent factors)
  shadow instantaneous forward at horizon s (b(s) = integrated-rate loadings):
     fbar(s) = L + S e^{-lam s} + C lam s e^{-lam s} - 1/2 b(s)' Sigma Sigma' b(s),
     b(s) = (s, (1-e^{-lam s})/lam, (1-e^{-lam s})/lam - s e^{-lam s})
  conditional Q-variance of the shadow short rate at horizon s:
     omega(s)^2 = s1^2 s + s2^2 (1-e^{-2 lam s})/(2 lam) + s3^2 [(1-e^{-2 lam s})/(4 lam) - s e^{-2 lam s}/2
                  - lam s^2 e^{-2 lam s}/2]
  forward rate that respects the lower bound 0 (Krippner):  f = fbar Phi(fbar/omega) + omega phi(fbar/omega)
  zero yield (continuous compounding):  y(tau) = (1/tau) int_0^tau f(s) ds   (Gauss-Legendre, NQ nodes)
  Jacobian for the EKF:  dy/dX_j = (1/tau) int_0^tau Phi(fbar/omega) l_j(s) ds,  l = (1, e^{-lam s}, lam s e^{-lam s})
Observation: the 10 zero yields y_obs = y(X) + e,  e ~ N(0, diag(e1^2..e10^2)).
Data (decided): curve.py zero rates (pchip bootstrap of FRED par yields) at 3M, 6M, 1, 2, 3, 5, 7, 10, 20, 30
years -- all are nodes of curve.py, so no interpolation -- converted from semiannual to continuous compounding
(y = 2 ln(1 + z/2)); every FRED business day; the time step between two days is calendar days / 365.
P transition: exact OU discretisation over each step (independent factors).
Filter: EKF, started at the unconditional P-distribution on the first data day; the log-likelihood counts the
days of the estimation window only (as the credit model).
Parameters (20): lam, k1..k3, th1..th3, s1..s3, e1..e10. Units: yields and theta in decimal per year, Sigma in
decimal per sqrt(year), time in years.
NQ (quadrature nodes) is a numerical-precision setting chosen by the convergence check in test_afns.py, not a
model setting.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

import curve

HERE = Path(__file__).parent
TAUS = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
NAMES = ['lam', 'k1', 'k2', 'k3', 'th1', 'th2', 'th3', 's1', 's2', 's3'] + [f'e{i + 1}' for i in range(len(TAUS))]
NP = len(NAMES)
IDX = {n: i for i, n in enumerate(NAMES)}
POS = [IDX[n] for n in NAMES if not n.startswith('th')]
NQ = 64
LOG2PI = math.log(2.0 * math.pi)
SQ2 = math.sqrt(2.0)
SQ2PI = math.sqrt(2.0 * math.pi)


def to_u(nat):
    u = np.array(nat, dtype=float).copy()
    u[POS] = np.log(u[POS])
    return u


def from_u(u):
    nat = np.array(u, dtype=float).copy()
    nat[POS] = np.exp(nat[POS])
    return nat


def load_yields():
    """(dates (D,), Y (D, 10) continuous-compounding zero yields in decimal, dt (D,) years since previous day)."""
    fred = curve.load_fred(HERE / 'data' / 'fred')
    zc = curve.build(fred, 'pchip')
    cols = [int(np.argmin(np.abs(zc.x - t))) for t in TAUS]
    if not np.allclose(zc.x[cols], TAUS):
        raise ValueError('AFNS tenors must be curve.py nodes')
    z = zc.z_nodes[:, cols]
    Y = 2.0 * np.log1p(z / 2.0)
    dates = pd.to_datetime(zc.dates)
    dt = np.r_[np.nan, np.diff(dates.values).astype('timedelta64[D]').astype(float) / 365.0]
    return dates, Y, dt


def gauss_nodes(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w            # on [0, 1]


@njit(cache=True)
def _ncdf(x):
    return 0.5 * math.erfc(-x / SQ2)


@njit(cache=True)
def _npdf(x):
    return math.exp(-0.5 * x * x) / SQ2PI


@njit(cache=True)
def _fbar_omega(s, X, lam, s1, s2, s3):
    e = math.exp(-lam * s)
    b2 = (1.0 - e) / lam
    b3 = b2 - s * e
    fb = X[0] + X[1] * e + X[2] * lam * s * e - 0.5 * (s1 * s1 * s * s + s2 * s2 * b2 * b2 + s3 * s3 * b3 * b3)
    e2 = e * e
    om2 = (s1 * s1 * s + s2 * s2 * (1.0 - e2) / (2.0 * lam)
           + s3 * s3 * ((1.0 - e2) / (4.0 * lam) - 0.5 * s * e2 - 0.5 * lam * s * s * e2))
    return fb, math.sqrt(max(om2, 0.0)), e


@njit(cache=True)
def yields_and_jac(X, lam, s1, s2, s3, taus, qx, qw, shadow, y_out, J_out):
    """Zero yields at taus (bound 0 unless shadow) and their Jacobian with respect to X."""
    for i in range(taus.shape[0]):
        tau = taus[i]
        y = 0.0
        j0 = 0.0
        j1 = 0.0
        j2 = 0.0
        for q in range(qx.shape[0]):
            s = tau * qx[q]
            fb, om, e = _fbar_omega(s, X, lam, s1, s2, s3)
            if shadow:
                f = fb
                ph = 1.0
            elif om > 0.0:
                z = fb / om
                ph = _ncdf(z)
                f = fb * ph + om * _npdf(z)
            else:
                f = fb if fb > 0.0 else 0.0
                ph = 1.0 if fb > 0.0 else 0.0
            w = qw[q]
            y += w * f
            j0 += w * ph
            j1 += w * ph * e
            j2 += w * ph * lam * s * e
        y_out[i] = y
        J_out[i, 0] = j0
        J_out[i, 1] = j1
        J_out[i, 2] = j2


@njit(cache=True)
def ekf(p_sets, p_idx, Y, dt, t0, t1, ll_t0, qx, qw, shadow, record, xf, Pf, yfit, xpred):
    """EKF log-likelihood of days ll_t0..t1 (filter from t0). Parameters of day t: p_sets[p_idx[t]].
    record: filtered states/variances, fitted yields at the filtered state, and the predicted state of each day
    (xpred; the state shock of day t is then xf[t] - xpred[t])."""
    for s_ in range(p_sets.shape[0]):
        if not (p_sets[s_, 0] > 0.0):
            return -np.inf
        for j in range(3):
            if not (p_sets[s_, 1 + j] > 0.0 and p_sets[s_, 7 + j] > 0.0):
                return -np.inf
    p = p_sets[p_idx[t0]]
    lam = p[0]
    k = p[1:4]
    th = p[4:7]
    sg = p[7:10]
    ev = p[10:]
    n = Y.shape[1]
    x = np.zeros(3)
    P = np.zeros((3, 3))
    for j in range(3):
        x[j] = th[j]
        P[j, j] = sg[j] * sg[j] / (2.0 * k[j])
    yv = np.zeros(n)
    J = np.zeros((n, 3))
    F = np.zeros((n, n))
    Fi = np.zeros((n, n))
    PJ = np.zeros((3, n))
    K = np.zeros((3, n))
    v = np.zeros(n)
    ll = 0.0
    for t in range(t0, t1 + 1):
        p = p_sets[p_idx[t]]
        lam = p[0]
        k = p[1:4]
        th = p[4:7]
        sg = p[7:10]
        ev = p[10:]
        if t > t0:
            d = dt[t]
            for j in range(3):
                a = math.exp(-k[j] * d)
                x[j] = th[j] + a * (x[j] - th[j])
                for c in range(3):
                    P[j, c] *= a * math.exp(-k[c] * d)
                P[j, j] += sg[j] * sg[j] * (1.0 - a * a) / (2.0 * k[j])
        if record:
            for c in range(3):
                xpred[t, c] = x[c]
        yields_and_jac(x, lam, sg[0], sg[1], sg[2], TAUS_NB, qx, qw, shadow, yv, J)
        for i in range(n):
            v[i] = Y[t, i] - yv[i]
            for c in range(3):
                s_ = 0.0
                for r in range(3):
                    s_ += P[c, r] * J[i, r]
                PJ[c, i] = s_
        for i in range(n):
            for jj in range(n):
                s_ = 0.0
                for c in range(3):
                    s_ += J[i, c] * PJ[c, jj]
                F[i, jj] = s_
            F[i, i] += ev[i] * ev[i]
        # Cholesky inverse and log-determinant
        L = np.zeros((n, n))
        ok = True
        for a_ in range(n):
            s_ = F[a_, a_]
            for b_ in range(a_):
                s_ -= L[a_, b_] * L[a_, b_]
            if not (s_ > 0.0):
                ok = False
                break
            L[a_, a_] = math.sqrt(s_)
            for i in range(a_ + 1, n):
                s2 = F[i, a_]
                for b_ in range(a_):
                    s2 -= L[i, b_] * L[a_, b_]
                L[i, a_] = s2 / L[a_, a_]
        if not ok:
            return -np.inf
        logdet = 0.0
        for a_ in range(n):
            logdet += 2.0 * math.log(L[a_, a_])
        Li = np.zeros((n, n))
        for i in range(n):
            Li[i, i] = 1.0 / L[i, i]
            for jj in range(i):
                s_ = 0.0
                for c in range(jj, i):
                    s_ -= L[i, c] * Li[c, jj]
                Li[i, jj] = s_ / L[i, i]
        for i in range(n):
            for jj in range(n):
                s_ = 0.0
                for c in range(max(i, jj), n):
                    s_ += Li[c, i] * Li[c, jj]
                Fi[i, jj] = s_
        if t >= ll_t0:
            q_ = 0.0
            for i in range(n):
                for jj in range(n):
                    q_ += v[i] * Fi[i, jj] * v[jj]
            ll += -0.5 * (n * LOG2PI + logdet + q_)
        for c in range(3):
            for i in range(n):
                s_ = 0.0
                for jj in range(n):
                    s_ += PJ[c, jj] * Fi[jj, i]
                K[c, i] = s_
        for c in range(3):
            s_ = 0.0
            for i in range(n):
                s_ += K[c, i] * v[i]
            x[c] += s_
        Pn = np.zeros((3, 3))
        for c in range(3):
            for r in range(3):
                s_ = P[c, r]
                for i in range(n):
                    s_ -= K[c, i] * PJ[r, i]
                Pn[c, r] = s_
        for c in range(3):
            for r in range(3):
                P[c, r] = 0.5 * (Pn[c, r] + Pn[r, c])
        if record:
            for c in range(3):
                xf[t, c] = x[c]
                for r in range(3):
                    Pf[t, c, r] = P[c, r]
            yields_and_jac(x, lam, sg[0], sg[1], sg[2], TAUS_NB, qx, qw, shadow, yv, J)
            for i in range(n):
                yfit[t, i] = yv[i]
    return ll


TAUS_NB = TAUS.copy()


def loglik(nat, Y, dt, t0, t1, ll_t0, shadow=False, record=False, nq=None, p_sets=None, p_idx=None):
    """EKF log-likelihood; nat = one parameter vector, or p_sets / p_idx = day-varying parameter sets."""
    if p_sets is None:
        p_sets = np.asarray(nat, float)[None, :]
        p_idx = np.zeros(Y.shape[0], np.int64)
    if not np.all(np.isfinite(p_sets)):
        return (-np.inf, None) if record else -np.inf
    qx, qw = gauss_nodes(nq or NQ)
    D = Y.shape[0]
    xf = np.full((D, 3), np.nan)
    Pf = np.full((D, 3, 3), np.nan)
    yfit = np.full(Y.shape, np.nan)
    xp = np.full((D, 3), np.nan)
    ll = ekf(np.ascontiguousarray(p_sets, dtype=float), np.asarray(p_idx, np.int64), Y, dt, t0, t1, ll_t0, qx, qw,
             shadow, record, xf, Pf, yfit, xp)
    if record:
        return ll, dict(x=xf, P=Pf, yfit=yfit, xpred=xp)
    return ll


def model_yields(X, nat, shadow=False, taus=None, nq=None):
    """Yields at taus (default TAUS) for one state X."""
    p = np.asarray(nat, float)
    tt = TAUS if taus is None else np.asarray(taus, float)
    qx, qw = gauss_nodes(nq or NQ)
    y = np.zeros(len(tt))
    J = np.zeros((len(tt), 3))
    yields_and_jac(np.asarray(X, float), p[0], p[7], p[8], p[9], tt, qx, qw, shadow, y, J)
    return y


# ---------------------------------------------------------------- start values (H-1, decided 2026-10-11)
def ns_loadings(lam, taus=TAUS):
    e = np.exp(-lam * taus)
    b2 = (1.0 - e) / (lam * taus)
    return np.column_stack([np.ones_like(taus), b2, b2 - e])


def static_ns(Y, lam):
    """Daily OLS of the yields on the Nelson-Siegel loadings: (factors (D,3), residuals (D,10))."""
    Xd = ns_loadings(lam)
    beta, *_ = np.linalg.lstsq(Xd, Y.T, rcond=None)
    return beta.T, Y - beta.T @ Xd.T


def start_values(Y, dt, a, b):
    """Data rule on days a..b: lam minimises the total static-NS squared error within the bounds implied by the
    tenors (curvature peak 1.7932/lam inside [0.25, 30] years); AR(1) of each static factor -> k, theta, sigma
    (exact OU mapping with the mean step); e_i = sd of the static residuals."""
    from scipy.optimize import minimize_scalar
    Yw = Y[a:b + 1]
    lo, hi = 1.7932 / TAUS[-1], 1.7932 / TAUS[0]
    res = minimize_scalar(lambda l: float((static_ns(Yw, l)[1] ** 2).sum()), bounds=(lo, hi), method='bounded')
    lam = float(res.x)
    f, r = static_ns(Yw, lam)
    d = float(np.nanmean(dt[a + 1:b + 1]))
    p = np.zeros(NP)
    p[0] = lam
    phi_max = float(np.nextafter(1.0, 0.0))          # numerical device: a unit-root AR(1) maps to k -> 0+
    for j in range(3):
        y1, y0 = f[1:, j], f[:-1, j]
        A = np.column_stack([np.ones_like(y0), y0])
        (c0, phi), *_ = np.linalg.lstsq(A, y1, rcond=None)
        resid = y1 - A @ np.array([c0, phi])
        phi = min(float(phi), phi_max)
        kj = -math.log(phi) / d
        p[1 + j] = kj
        p[4 + j] = c0 / (1.0 - phi) if phi < phi_max else float(f[:, j].mean())
        q = float(resid.var(ddof=2))
        p[7 + j] = math.sqrt(q * 2.0 * kj / (1.0 - phi * phi))
    p[10:] = r.std(axis=0, ddof=1)
    return p, dict(lam_bounds=[lo, hi], mean_step_years=d)
