"""fourstate2.py -- model v2: per-bond starts, a prior for the permanent level, and the issue spread as one
extra observation.

Same states, observations and prediction step as fourstate (v1). Changes:
  * Start of a bond = its own first trading day: I = 0 and D = 0 (variance 0), L at its stationary
    distribution, u ~ N(prior mean, s2_u). Prior mean of u (peer_mode):
      0  zero (the same-rating level)
      1  a per-bond input u_prior[b] minus iss_d (u_prior predicts issue Z - G, and issue Z = s + iss_d + e)
      2  mean filtered u of the bonds in the same cluster (clustering in force that day) that are active and
         not matured, as of the day before (their predicted means); zero if there are none.
    (A bond that is already trading when the filter starts -- not used in v2 estimation, which always runs from
    the first data day -- falls back to the v1 start: stationary I, flow-accumulated D.)
  * Issue Z-spread, once, on the first trading day, as an observation of the fair spread:
        iss = s + iss_d + e,  e ~ N(0, iss_v)       (use_iss; NaN = no usable issue Z)
  * The filter runs from t0; the log-likelihood counts only days t >= ll_t0 (estimation: t0 = first data day,
    ll_t0 = window start, so the bond states are carried into the window and the parameters are fit on it).
Parameters: fourstate.NAMES + iss_d, iss_v (35).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

import fourstate as fs
from fourstate import LOG2PI, _inv3, _q3, _d_start

NAMES = fs.NAMES + ['iss_d', 'iss_v']
NP = len(NAMES)
IDX = {n: i for i, n in enumerate(NAMES)}
PHI = [IDX[n] for n in ('phi_a', 'phi_b', 'phi_c', 'phi_I', 'phi_L', 'phi_D')]
POS = [IDX[n] for n in ('q_a', 'q_b', 'q_c', 'q_I', 's2_u', 'q_L', 'h1', 'h2', 'h3', 'q_D', 'iss_v')]
NMAX = 5          # rows of one bond-day update: 3 measures + daily spread + issue Z


def to_u(nat):
    u = np.array(nat, dtype=float).copy()
    u[PHI] = np.arctanh(u[PHI])
    with np.errstate(divide='ignore'):
        u[POS] = np.log(u[POS])
    return u


def from_u(u):
    nat = np.array(u, dtype=float).copy()
    nat[PHI] = np.tanh(nat[PHI])
    nat[POS] = np.exp(nat[POS])
    return nat


@njit(cache=True)
def _solve_spd(W, p, Winv):
    """Inverse and log-determinant of the leading p x p block of SPD W (Cholesky). Returns (ok, logdet)."""
    n = W.shape[0]
    L = np.zeros((n, n))
    for j in range(p):
        s = W[j, j]
        for k in range(j):
            s -= L[j, k] * L[j, k]
        if not (s > 0.0):
            return False, 0.0
        L[j, j] = math.sqrt(s)
        for i in range(j + 1, p):
            s = W[i, j]
            for k in range(j):
                s -= L[i, k] * L[j, k]
            L[i, j] = s / L[j, j]
    logdet = 0.0
    for j in range(p):
        logdet += 2.0 * math.log(L[j, j])
    Li = np.zeros((n, n))
    for i in range(p):
        Li[i, i] = 1.0 / L[i, i]
        for j in range(i):
            s = 0.0
            for k in range(j, i):
                s -= L[i, k] * Li[k, j]
            Li[i, j] = s / L[i, i]
    for i in range(p):
        for j in range(p):
            s = 0.0
            for k in range(max(i, j), p):
                s += Li[k, i] * Li[k, j]
            Winv[i, j] = s
    return True, logdet


@njit(cache=True)
def peer_mean(b, t, m, act, mat_day, clus, ci):
    """Mean predicted u of the active, not-matured bonds in b's cluster (clustering ci); (mean, count)."""
    if ci < 0:
        return 0.0, 0
    lab = clus[ci, b]
    if lab < 0:
        return 0.0, 0
    s = 0.0
    n = 0
    for a in range(m.shape[0]):
        if act[a] and mat_day[a] >= t and clus[ci, a] == lab:
            s += m[a, 1]
            n += 1
    if n == 0:
        return 0.0, 0
    return s / n, n


@njit(cache=True)
def ci_filter2(par_sets, par_idx, notch, s_obs, V, x, OF, first_day, mat_day, iss, u_prior, clus, clus_of_day,
               peer_mode, t0, t1, ll_t0, n0, record, fm_out, fv_out, g_out, um_out, use_G, use_L, use_iss):
    """Conditional-independence filter of fourstate.ci_filter with the v2 start and the issue-Z observation.
    record: fm_out / fv_out (B, D) fair-spread mean / variance, g_out (D, 3) common states,
    um_out (B, D, 2) filtered means of I and u."""
    B = notch.shape[0]
    g = np.zeros(3)
    Pg = np.zeros((3, 3))
    m = np.zeros((B, 4))
    Bm = np.zeros((B, 4, 3))
    S = np.zeros((B, 4, 4))
    act = np.zeros(B, np.bool_)
    newb = np.zeros(B, np.bool_)
    pm = np.zeros(B)
    ll = 0.0
    Pinv = np.zeros((3, 3))
    Lam = np.zeros((3, 3))
    Lami = np.zeros((3, 3))
    eta = np.zeros(3)
    gnew = np.zeros(3)
    Hx = np.zeros((NMAX, 4))
    Hg = np.zeros((NMAX, 3))
    yv = np.zeros(NMAX)
    dv = np.zeros(NMAX)
    Rv = np.zeros(NMAX)
    W = np.zeros((NMAX, NMAX))
    Wi = np.zeros((NMAX, NMAX))
    M = np.zeros((NMAX, 3))
    HxS = np.zeros((NMAX, 4))
    K = np.zeros((4, NMAX))
    e = np.zeros(NMAX)
    Vx = np.zeros((4, 4))
    Cxg = np.zeros((4, 3))
    Cn = np.zeros((4, 3))
    Vn = np.zeros((4, 4))
    Dm = np.zeros((4, 3))
    zx = np.zeros(4)
    zg = np.zeros(3)
    h = np.zeros(3)
    for t in range(t0, t1 + 1):
        p = par_sets[par_idx[t]]
        mu = p[0:3]
        ph = p[3:6]
        qg = p[6:9]
        phI, qI, s2u = p[9], p[10], p[11]
        phL, qL = p[12], p[13]
        al = p[14:17]
        be = p[17:20]
        hh = p[20:23]
        phD, qD, lam = p[23], p[24], p[25]
        iss_d, iss_v = p[33], p[34]
        if not (abs(ph[0]) < 1.0 and abs(ph[1]) < 1.0 and abs(ph[2]) < 1.0 and abs(phI) < 1.0
                and abs(phL) < 1.0 and abs(phD) < 1.0):
            return -np.inf
        if not (qg[0] > 0.0 and qg[1] > 0.0 and qg[2] > 0.0 and qI > 0.0 and s2u >= 0.0 and qL > 0.0
                and qD >= 0.0 and hh[0] > 0.0 and hh[1] > 0.0 and hh[2] > 0.0):
            return -np.inf
        if use_iss and not (iss_v > 0.0):
            return -np.inf
        tv = (phI, 1.0, phL, phD)
        qv = (qI, 0.0, qL, qD)
        zx[0] = 1.0
        zx[1] = 1.0
        zx[2] = 1.0 if use_L else 0.0
        zx[3] = lam
        # ---- common predict
        if not use_G:
            for r in range(3):
                g[r] = mu[r]
                for c in range(3):
                    Pg[r, c] = 0.0
            for r in range(3):
                for c in range(3):
                    Pinv[r, c] = 0.0
        elif t == t0:
            for r in range(3):
                g[r] = mu[r]
                for c in range(3):
                    Pg[r, c] = 0.0
                Pg[r, r] = qg[r] / (1.0 - ph[r] * ph[r])
        if t > t0:
            Pold = Pg.copy()
            if use_G:
                for r in range(3):
                    g[r] = mu[r] + ph[r] * (g[r] - mu[r])
                for r in range(3):
                    for c in range(3):
                        Pg[r, c] = ph[r] * ph[c] * Pold[r, c]
                    Pg[r, r] += qg[r]
                if _inv3(Pg, Pinv) <= 0.0:
                    return -np.inf
            # ---- bonds predict
            for b in range(B):
                if not act[b]:
                    continue
                for r in range(4):
                    for c in range(3):
                        s_ = 0.0
                        for k in range(3):
                            s_ += Bm[b, r, k] * Pold[k, c]
                        Cxg[r, c] = s_
                for r in range(4):
                    for c in range(4):
                        s_ = S[b, r, c]
                        for k in range(3):
                            s_ += Cxg[r, k] * Bm[b, c, k]
                        Vx[r, c] = s_
                for r in range(4):
                    for c in range(3):
                        Dm[r, c] = 0.0
                n_new = notch[b, t]
                n_old = notch[b, t - 1]
                if np.isfinite(n_new) and np.isfinite(n_old) and n_new != n_old:
                    Dm[0, 1] = n_new - n_old
                    Dm[0, 2] = (n_new - n0) ** 2 - (n_old - n0) ** 2
                for r in range(4):
                    for c in range(3):
                        s_ = tv[r] * Cxg[r, c] * ph[c]
                        for k in range(3):
                            s_ -= Dm[r, k] * Pg[k, c]
                        Cn[r, c] = s_
                for r in range(4):
                    for c in range(4):
                        s_ = tv[r] * Vx[r, c] * tv[c]
                        for k in range(3):
                            for l in range(3):
                                s_ += Dm[r, k] * Pg[k, l] * Dm[c, l]
                        for k in range(3):
                            s_ -= tv[r] * Cxg[r, k] * ph[k] * Dm[c, k]
                            s_ -= Dm[r, k] * ph[k] * Cxg[c, k] * tv[c]
                        Vn[r, c] = s_
                    Vn[r, r] += qv[r]
                of = OF[b, t - 1] if np.isfinite(OF[b, t - 1]) else 0.0
                m0 = tv[0] * m[b, 0] - (Dm[0, 1] * g[1] + Dm[0, 2] * g[2])
                m[b, 2] = tv[2] * m[b, 2]
                m[b, 3] = tv[3] * m[b, 3] - of
                m[b, 0] = m0
                for r in range(4):
                    for c in range(3):
                        s_ = 0.0
                        for k in range(3):
                            s_ += Cn[r, k] * Pinv[k, c]
                        Bm[b, r, c] = s_
                for r in range(4):
                    for c in range(4):
                        s_ = Vn[r, c]
                        for k in range(3):
                            s_ -= Bm[b, r, k] * Cn[c, k]
                        S[b, r, c] = s_
                for r in range(4):
                    for c in range(r):
                        q_ = 0.5 * (S[b, r, c] + S[b, c, r])
                        S[b, r, c] = q_
                        S[b, c, r] = q_
        # ---- activate new bonds (peer means use the bonds active before today)
        ci = clus_of_day[t]
        for b in range(B):
            newb[b] = False
            if t >= first_day[b] and not act[b]:
                newb[b] = True
                if peer_mode == 1:
                    pm[b] = u_prior[b] - iss_d
                elif peer_mode == 2:
                    pv, _n = peer_mean(b, t, m, act, mat_day, clus, ci)
                    pm[b] = pv
                else:
                    pm[b] = 0.0
        for b in range(B):
            if not newb[b]:
                continue
            act[b] = True
            for r in range(4):
                m[b, r] = 0.0
                for c in range(3):
                    Bm[b, r, c] = 0.0
                for c in range(4):
                    S[b, r, c] = 0.0
            m[b, 1] = pm[b]
            S[b, 1, 1] = s2u
            S[b, 2, 2] = qL / (1.0 - phL * phL)
            if t > first_day[b]:
                # already trading when the filter starts (v1 start for I and D)
                S[b, 0, 0] = qI / (1.0 - phI * phI)
                dmean, dvar = _d_start(OF, b, first_day[b], t, phD, qD)
                m[b, 3] = dmean
                S[b, 3, 3] = dvar
        # ---- measurement update (exact for the star structure)
        detP = 1.0
        if use_G:
            detP = _inv3(Pg, Pinv)
            if detP <= 0.0:
                return -np.inf
        for r in range(3):
            eta[r] = 0.0
            for c in range(3):
                Lam[r, c] = Pinv[r, c]
        nobs = 0
        slogw = 0.0
        squad = 0.0
        for b in range(B):
            if not act[b]:
                continue
            pdim = 0
            for k in range(3):
                y = x[k, b, t]
                if use_L and np.isfinite(y):
                    for c in range(4):
                        Hx[pdim, c] = 0.0
                    for c in range(3):
                        Hg[pdim, c] = 0.0
                    Hx[pdim, 2] = be[k]
                    yv[pdim] = y
                    dv[pdim] = al[k]
                    Rv[pdim] = hh[k]
                    pdim += 1
            nn = notch[b, t]
            if np.isfinite(nn):
                x1 = nn - n0
                y = s_obs[b, t]
                if np.isfinite(y):
                    for c in range(4):
                        Hx[pdim, c] = zx[c]
                    Hg[pdim, 0] = 1.0
                    Hg[pdim, 1] = x1
                    Hg[pdim, 2] = x1 * x1
                    yv[pdim] = y
                    dv[pdim] = 0.0
                    Rv[pdim] = V[b, t]
                    pdim += 1
                if use_iss and t == first_day[b] and np.isfinite(iss[b]):
                    for c in range(4):
                        Hx[pdim, c] = zx[c]
                    Hg[pdim, 0] = 1.0
                    Hg[pdim, 1] = x1
                    Hg[pdim, 2] = x1 * x1
                    yv[pdim] = iss[b]
                    dv[pdim] = iss_d
                    Rv[pdim] = iss_v
                    pdim += 1
            if pdim == 0:
                continue
            for r in range(pdim):
                for c in range(4):
                    s_ = 0.0
                    for k in range(4):
                        s_ += Hx[r, k] * S[b, k, c]
                    HxS[r, c] = s_
            for r in range(pdim):
                for c in range(pdim):
                    s_ = 0.0
                    for k in range(4):
                        s_ += HxS[r, k] * Hx[c, k]
                    W[r, c] = s_
                W[r, r] += Rv[r]
            ok, logw = _solve_spd(W, pdim, Wi)
            if not ok:
                return -np.inf
            for r in range(pdim):
                pr = dv[r]
                for k in range(4):
                    pr += Hx[r, k] * m[b, k]
                for k in range(3):
                    pr += Hg[r, k] * g[k]
                e[r] = yv[r] - pr
                for c in range(3):
                    s_ = Hg[r, c]
                    for k in range(4):
                        s_ += Hx[r, k] * Bm[b, k, c]
                    M[r, c] = s_
            for r in range(3):
                for c in range(3):
                    s_ = 0.0
                    for i in range(pdim):
                        for j in range(pdim):
                            s_ += M[i, r] * Wi[i, j] * M[j, c]
                    Lam[r, c] += s_
                s_ = 0.0
                for i in range(pdim):
                    for j in range(pdim):
                        s_ += M[i, r] * Wi[i, j] * e[j]
                eta[r] += s_
            q_ = 0.0
            for i in range(pdim):
                for j in range(pdim):
                    q_ += e[i] * Wi[i, j] * e[j]
            squad += q_
            slogw += logw
            nobs += pdim
            for r in range(4):
                for c in range(pdim):
                    s_ = 0.0
                    for k in range(pdim):
                        s_ += HxS[k, r] * Wi[k, c]
                    K[r, c] = s_
            for r in range(4):
                s_ = 0.0
                for c in range(pdim):
                    s_ += K[r, c] * e[c]
                m[b, r] += s_
                for c in range(3):
                    s2 = 0.0
                    for k in range(pdim):
                        s2 += K[r, k] * M[k, c]
                    Bm[b, r, c] -= s2
            for r in range(4):
                for c in range(4):
                    s_ = 0.0
                    for k in range(pdim):
                        s_ += K[r, k] * HxS[k, c]
                    S[b, r, c] -= s_
            for r in range(4):
                for c in range(r):
                    q2 = 0.5 * (S[b, r, c] + S[b, c, r])
                    S[b, r, c] = q2
                    S[b, c, r] = q2
        if nobs > 0 and not use_G:
            if t >= ll_t0:
                ll += -0.5 * (nobs * LOG2PI + slogw + squad)
        elif nobs > 0:
            detL = _inv3(Lam, Lami)
            if detL <= 0.0:
                return -np.inf
            quad = 0.0
            for r in range(3):
                gnew[r] = g[r]
                for c in range(3):
                    quad += eta[r] * Lami[r, c] * eta[c]
                    gnew[r] += Lami[r, c] * eta[c]
            if t >= ll_t0:
                ll += -0.5 * (nobs * LOG2PI + slogw + math.log(detP) + math.log(detL) + squad - quad)
            for b in range(B):
                if not act[b]:
                    continue
                for r in range(4):
                    s_ = 0.0
                    for c in range(3):
                        s_ += Bm[b, r, c] * (gnew[c] - g[c])
                    m[b, r] += s_
            for r in range(3):
                g[r] = gnew[r]
                for c in range(3):
                    Pg[r, c] = Lami[r, c]
        if record:
            for r in range(3):
                g_out[t, r] = g[r]
            for b in range(B):
                if not act[b]:
                    continue
                um_out[b, t, 0] = m[b, 0]
                um_out[b, t, 1] = m[b, 1]
                nn = notch[b, t]
                if not np.isfinite(nn):
                    continue
                x1 = nn - n0
                zg[0] = 1.0
                zg[1] = x1
                zg[2] = x1 * x1
                mv = zg[0] * g[0] + zg[1] * g[1] + zg[2] * g[2]
                for r in range(4):
                    mv += zx[r] * m[b, r]
                for c in range(3):
                    s_ = zg[c]
                    for r in range(4):
                        s_ += Bm[b, r, c] * zx[r]
                    h[c] = s_
                vv = _q3(h, Pg)
                for r in range(4):
                    for c in range(4):
                        vv += zx[r] * S[b, r, c] * zx[c]
                fm_out[b, t] = mv
                fv_out[b, t] = vv
    return ll


@njit(cache=True)
def exact_filter2(p, notch, s_obs, V, x, OF, first_day, mat_day, iss, u_prior, clus, clus_of_day, peer_mode,
                  t0, t1, ll_t0, n0, use_iss, record, fm_out, fv_out):
    """Full joint covariance over 3 + 4B states (validation of ci_filter2): same v2 start, issue-Z observation and
    likelihood window; peer means from the joint filtered u."""
    B = notch.shape[0]
    m = 3 + 4 * B
    a = np.zeros(m)
    P = np.zeros((m, m))
    act = np.zeros(B, np.bool_)
    newb = np.zeros(B, np.bool_)
    pm = np.zeros(B)
    ub = np.zeros((B, 4))
    mu = p[0:3]
    ph = p[3:6]
    qg = p[6:9]
    phI, qI, s2u = p[9], p[10], p[11]
    phL, qL = p[12], p[13]
    al = p[14:17]
    be = p[17:20]
    hh = p[20:23]
    phD, qD, lam = p[23], p[24], p[25]
    iss_d, iss_v = p[33], p[34]
    tv = np.ones(m)
    for r in range(3):
        tv[r] = ph[r]
    for b in range(B):
        tv[3 + 4 * b] = phI
        tv[3 + 4 * b + 2] = phL
        tv[3 + 4 * b + 3] = phD
    ll = 0.0
    Pz = np.zeros(m)
    idx = np.zeros(7, np.int64)
    zv = np.zeros(7)
    for t in range(t0, t1 + 1):
        if t == t0:
            for r in range(3):
                a[r] = mu[r]
                P[r, r] = qg[r] / (1.0 - ph[r] * ph[r])
        else:
            for r in range(3):
                a[r] = mu[r] + ph[r] * (a[r] - mu[r])
            gb, gc = a[1], a[2]
            for b in range(B):
                if not act[b]:
                    continue
                o = 3 + 4 * b
                a[o] = phI * a[o]
                a[o + 2] = phL * a[o + 2]
                a[o + 3] = phD * a[o + 3] - (OF[b, t - 1] if np.isfinite(OF[b, t - 1]) else 0.0)
            for r in range(m):
                for c in range(m):
                    P[r, c] *= tv[r] * tv[c]
            for r in range(3):
                P[r, r] += qg[r]
            for b in range(B):
                if not act[b]:
                    continue
                n_new = notch[b, t]
                n_old = notch[b, t - 1]
                if np.isfinite(n_new) and np.isfinite(n_old) and n_new != n_old:
                    d1 = n_new - n_old
                    d2 = (n_new - n0) ** 2 - (n_old - n0) ** 2
                    o = 3 + 4 * b
                    a[o] -= d1 * gb + d2 * gc
                    for c in range(m):
                        P[o, c] -= d1 * P[1, c] + d2 * P[2, c]
                    for r in range(m):
                        P[r, o] -= d1 * P[r, 1] + d2 * P[r, 2]
            for b in range(B):
                if not act[b]:
                    continue
                o = 3 + 4 * b
                P[o, o] += qI
                P[o + 2, o + 2] += qL
                P[o + 3, o + 3] += qD
        for b in range(B):
            ub[b, 1] = a[3 + 4 * b + 1]
        ci = clus_of_day[t]
        for b in range(B):
            newb[b] = False
            if t >= first_day[b] and not act[b]:
                newb[b] = True
                if peer_mode == 1:
                    pm[b] = u_prior[b] - iss_d
                elif peer_mode == 2:
                    pv, _n = peer_mean(b, t, ub, act, mat_day, clus, ci)
                    pm[b] = pv
                else:
                    pm[b] = 0.0
        for b in range(B):
            if not newb[b]:
                continue
            act[b] = True
            o = 3 + 4 * b
            for r in range(4):
                a[o + r] = 0.0
            a[o + 1] = pm[b]
            P[o + 1, o + 1] = s2u
            P[o + 2, o + 2] = qL / (1.0 - phL * phL)
            if t > first_day[b]:
                P[o, o] = qI / (1.0 - phI * phI)
                dmean, dvar = _d_start(OF, b, first_day[b], t, phD, qD)
                a[o + 3] = dmean
                P[o + 3, o + 3] = dvar
        for b in range(B):
            if not act[b]:
                continue
            o = 3 + 4 * b
            for k in range(5):
                if k < 3:
                    y = x[k, b, t]
                    if not np.isfinite(y):
                        continue
                    nz = 1
                    idx[0] = o + 2
                    zv[0] = be[k]
                    d0 = al[k]
                    hv = hh[k]
                else:
                    nn = notch[b, t]
                    if not np.isfinite(nn):
                        continue
                    if k == 3:
                        y = s_obs[b, t]
                        if not np.isfinite(y):
                            continue
                        d0 = 0.0
                        hv = V[b, t]
                    else:
                        if not (use_iss and t == first_day[b] and np.isfinite(iss[b])):
                            continue
                        y = iss[b]
                        d0 = iss_d
                        hv = iss_v
                    x1 = nn - n0
                    nz = 7
                    idx[0], idx[1], idx[2] = 0, 1, 2
                    zv[0], zv[1], zv[2] = 1.0, x1, x1 * x1
                    idx[3], idx[4], idx[5], idx[6] = o, o + 1, o + 2, o + 3
                    zv[3], zv[4], zv[5], zv[6] = 1.0, 1.0, 1.0, lam
                for r in range(m):
                    s = 0.0
                    for q in range(nz):
                        s += P[r, idx[q]] * zv[q]
                    Pz[r] = s
                f = hv
                pred = d0
                for q in range(nz):
                    f += zv[q] * Pz[idx[q]]
                    pred += zv[q] * a[idx[q]]
                if not (f > 0.0):
                    return -np.inf
                v = y - pred
                if t >= ll_t0:
                    ll += -0.5 * (LOG2PI + math.log(f) + v * v / f)
                for r in range(m):
                    a[r] += Pz[r] / f * v
                for r in range(m):
                    pr = Pz[r] / f
                    for c in range(m):
                        P[r, c] -= pr * Pz[c]
        if record:
            for b in range(B):
                if not act[b]:
                    continue
                nn = notch[b, t]
                if not np.isfinite(nn):
                    continue
                o = 3 + 4 * b
                x1 = nn - n0
                idx[0], idx[1], idx[2] = 0, 1, 2
                zv[0], zv[1], zv[2] = 1.0, x1, x1 * x1
                idx[3], idx[4], idx[5], idx[6] = o, o + 1, o + 2, o + 3
                zv[3], zv[4], zv[5], zv[6] = 1.0, 1.0, 1.0, lam
                mv = 0.0
                vv = 0.0
                for q in range(7):
                    mv += zv[q] * a[idx[q]]
                    for q2 in range(7):
                        vv += zv[q] * P[idx[q], idx[q2]] * zv[q2]
                fm_out[b, t] = mv
                fv_out[b, t] = vv
    return ll
