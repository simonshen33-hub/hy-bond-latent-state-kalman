"""test_afns.py -- component checks of afns.py (results in out/afns/test_afns.json).

1. Quadrature: yields with NQ nodes vs 4 x NQ nodes (several states), max |difference| in bp.
2. Far from the bound the B-AFNS yields equal the Gaussian AFNS (shadow) yields.
3. Gaussian AFNS yields (closed-form convexity in fbar) vs Monte Carlo of E^Q[exp(-int r)] under the Q dynamics.
4. Krippner's B-AFNS yields vs Monte Carlo of the exact Black (1995) model E^Q[exp(-int max(r, 0))]
   (Christensen-Rudebusch 2015 report the option-based approximation error to be an order of magnitude below
   the fit error).
5. Analytic Jacobian vs central finite differences.
6. EKF log-likelihood on the full data at the data-rule start values (finite) and the time per evaluation.
The Monte Carlo settings (paths, step, seed 0) are test devices, not model settings.

    python test_afns.py
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
from numba import njit

import afns

HERE = Path(__file__).parent
OUT = HERE / 'out' / 'afns'


@njit(cache=True)
def _mc(X0, a, lamdt_a, cholSC, sdL, taus, dt, npath, seed, floor):
    """E^Q[exp(-int_0^tau r)]: exact Gaussian one-step transition of (L, S, C) under K^Q (L random walk;
    (S, C) with e^{-A dt} = a [[1, lam dt], [0, 1]] and innovation covariance chol cholSC), trapezoid for the
    integral of r, antithetic pairs."""
    nst = int(round(taus[-1] / dt))
    out = np.zeros(taus.shape[0])
    for p_ in range(npath):
        np.random.seed(seed + p_)
        Z = np.random.randn(nst, 3)
        for sgn in (1.0, -1.0):
            L, S, C = X0[0], X0[1], X0[2]
            integ = 0.0
            k = 0
            for st in range(nst):
                r0 = L + S
                z1 = sgn * Z[st, 0]
                z2 = sgn * Z[st, 1]
                z3 = sgn * Z[st, 2]
                eS = cholSC[0, 0] * z2
                eC = cholSC[1, 0] * z2 + cholSC[1, 1] * z3
                Sn = a * S + lamdt_a * C + eS
                Cn = a * C + eC
                Ln = L + sdL * z1
                r1 = Ln + Sn
                if floor:
                    integ += 0.5 * dt * (max(r0, 0.0) + max(r1, 0.0))
                else:
                    integ += 0.5 * dt * (r0 + r1)
                L, S, C = Ln, Sn, Cn
                if k < taus.shape[0] and abs((st + 1) * dt - taus[k]) < 0.5 * dt:
                    out[k] += math.exp(-integ)
                    k += 1
    for k in range(taus.shape[0]):
        out[k] /= 2.0 * npath
    return out


def mc_yields(X0, p, taus, floor, dt=1.0 / 365.0, npath=20000, seed=0):
    lam, s1, s2, s3 = p[0], p[7], p[8], p[9]
    a = math.exp(-lam * dt)
    u, w = np.polynomial.legendre.leggauss(200)
    u, w = 0.5 * dt * (u + 1.0), 0.5 * dt * w
    e2 = np.exp(-2 * lam * u)
    vSS = np.sum(w * e2 * (s2 ** 2 + s3 ** 2 * (lam * u) ** 2))
    vSC = np.sum(w * e2 * s3 ** 2 * lam * u)
    vCC = np.sum(w * e2 * s3 ** 2)
    chol = np.linalg.cholesky(np.array([[vSS, vSC], [vSC, vCC]]))
    P = _mc(np.asarray(X0, float), a, a * lam * dt, chol, s1 * math.sqrt(dt), np.asarray(taus, float), dt, npath,
            seed, floor)
    return -np.log(P) / np.asarray(taus)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dates, Y, dt = afns.load_yields()
    res = {'data': dict(first=str(dates[0].date()), last=str(dates[-1].date()), days=int(len(dates)))}
    a = int(np.searchsorted(dates, np.datetime64('2017-06-26')))
    b = int(np.searchsorted(dates, np.datetime64('2020-06-30'), side='right') - 1)
    p0, info = afns.start_values(Y, dt, a, b)
    res['start_first_window'] = dict(zip(afns.NAMES, map(float, p0)), **info)
    print('start', res['start_first_window'], flush=True)
    states = {'high': [0.05, -0.01, 0.0], 'mid': [0.03, -0.02, 0.01], 'low': [0.015, -0.014, -0.01],
              'zero': [0.012, -0.0115, -0.02]}
    # 1 quadrature
    q = {}
    for nm, X in states.items():
        y1 = afns.model_yields(X, p0, nq=afns.NQ)
        y4 = afns.model_yields(X, p0, nq=4 * afns.NQ)
        q[nm] = float(np.abs(y1 - y4).max() * 1e4)
    res['quadrature_maxdiff_bp'] = q
    print('quadrature', q, flush=True)
    # 2 far from the bound
    far = {}
    for lvl in (0.05, 0.10, 0.20, 0.40):
        X = [lvl, 0.0, 0.0]
        far[str(lvl)] = ((afns.model_yields(X, p0) - afns.model_yields(X, p0, shadow=True)) * 1e4).round(4).tolist()
    res['bafns_minus_gaussian_bp_by_level'] = dict(taus=afns.TAUS.tolist(), **far)
    res['far_from_bound_bp'] = float(np.abs(np.array(far['0.4'])).max())
    print('far from bound', res['far_from_bound_bp'], flush=True)
    # 3, 4 Monte Carlo
    taus = np.array([1.0, 3.0, 5.0, 10.0])
    mc = {}
    for nm in ('mid', 'zero'):
        X = states[nm]
        t = time.time()
        g_mc = mc_yields(X, p0, taus, floor=False)
        g_an = afns.model_yields(X, p0, shadow=True, taus=taus)
        b_mc = mc_yields(X, p0, taus, floor=True)
        b_an = afns.model_yields(X, p0, shadow=False, taus=taus)
        mc[nm] = dict(taus=taus.tolist(), gaussian_bp_analytic_minus_mc=((g_an - g_mc) * 1e4).round(3).tolist(),
                      bafns_bp_krippner_minus_exact_mc=((b_an - b_mc) * 1e4).round(3).tolist(),
                      bafns_minus_gaussian_bp=((b_an - g_an) * 1e4).round(2).tolist(), seconds=time.time() - t)
        print('mc', nm, mc[nm], flush=True)
    res['monte_carlo'] = mc
    # 5 Jacobian
    X = np.array(states['zero'])
    qx, qw = afns.gauss_nodes(afns.NQ)
    yv = np.zeros(len(afns.TAUS))
    J = np.zeros((len(afns.TAUS), 3))
    afns.yields_and_jac(X, p0[0], p0[7], p0[8], p0[9], afns.TAUS, qx, qw, False, yv, J)
    Jn = np.zeros_like(J)
    for j in range(3):
        h = 1e-6
        e = np.zeros(3)
        e[j] = h
        Jn[:, j] = (afns.model_yields(X + e, p0) - afns.model_yields(X - e, p0)) / (2 * h)
    res['jacobian_max_abs_diff'] = float(np.abs(J - Jn).max())
    print('jacobian', res['jacobian_max_abs_diff'], flush=True)
    # 6 EKF
    t = time.time()
    ll = afns.loglik(p0, Y, dt, 0, b, a)
    res['ekf_start_ll_first_window'] = float(ll)
    res['ekf_seconds_per_ll'] = time.time() - t
    t = time.time()
    afns.loglik(p0, Y, dt, 0, b, a)
    res['ekf_seconds_per_ll_compiled'] = time.time() - t
    print('ekf', ll, res['ekf_seconds_per_ll_compiled'], flush=True)
    (OUT / 'test_afns.json').write_text(json.dumps(res, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
