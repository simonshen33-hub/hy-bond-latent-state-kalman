"""fourstate4e.py -- model v4 on the OAS observations with the iterated EKF update.

Economic content: the call's value depends on the bond's latent credit spread, not on the day's median print. The OAS
version (oas_obs.oas_transform) values the call at the OAS implied by the day's median Z (q*, the data anchor); here the
call is re-valued at the filter's updated spread -- the day's trades combined with yesterday's state and the issuer's
other bonds -- by relinearizing the tree map Z = h(s) there and re-running the day's update (iterated EKF, Bell and
Cathey 1993: Gauss-Newton on the day's posterior). The first pass is exactly the OAS transform.

Liquidity 'discount' (the reading of oas_obs.py in which the issuer's call decision sees credit only): where cld_a
(the anchor's liquidity c_ld, bp) and dld_a (dZ/dc_ld) are finite, the credit-only spread s_cr (G + beta C + B + u)
moves the call decision and the price, the liquidity part c_ld (L + lam_D D) only the discounting: Z = Hc(s_cr) + d_ld (c_ld - cld)
around the anchor (credit curve at the anchor's liquidity, liquidity linear: it has no call threshold). Linearized at
(lp, lpl): pseudo-observation lp + r lpl + (Z - h) / h'  of  s_cr + r c_ld,  r = d_ld / h'(lp), variance V / h'^2 -- the
measurement row carries r on L and D. Both points are relinearized. With max_it = 0 and straight-line curves this is
the OAS transform with separate credit and liquidity slopes (the 'discount' OAS version without iteration).

ci_filter4e = fourstate4.ci_filter4 (unchanged copy, generated) with the spread observation replaced by the
pseudo-observation of the current linearization point and the day's measurement update wrapped in the relinearization
loop; the log-likelihood is of Z (the pseudo-observation density times 1 / h'). Inputs beyond ci_filter4: s_z, V_z
(daily Z observation and variance, fourstate.daily_obs), qc (anchor q*, NaN = no usable observation, as in oas_transform), Hc
(B, D, G: the tree's Z at qc + offs), offs (G, increasing, bp), max_it, tol (bp), it_out (D: relinearizations used per
day). h between grid nodes: monotone piecewise cubic Hermite (Fritsch-Carlson node slopes, end slopes = end secants);
beyond the grid: straight line with the end secant. Slope floored at C_MIN (a flat h carries no information about s;
the floor keeps the update finite and the Z density right: V / C_MIN^2 with log C_MIN in the Jacobian). After
DAMP_AFTER relinearizations a day moves its linearization points half way to the new posterior (an oscillating
Gauss-Newton step settles; the fixed point is unchanged).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from fourstate import LOG2PI, _inv3, _q3
from fourstate2 import _solve_spd
from fourstate4 import IDX, NAMES, NP, NROW   # noqa: F401  (same parameter set as v4)

C_MIN = 1e-4
DAMP_AFTER = 5     # numerical device: after 5 relinearizations a day moves its points half way (same fixed point)


@njit(cache=True)
def _node_slope(xk, yk, k):
    n = xk.shape[0]
    if k == 0:
        return (yk[1] - yk[0]) / (xk[1] - xk[0])
    if k == n - 1:
        return (yk[n - 1] - yk[n - 2]) / (xk[n - 1] - xk[n - 2])
    h0 = xk[k] - xk[k - 1]
    h1 = xk[k + 1] - xk[k]
    d0 = (yk[k] - yk[k - 1]) / h0
    d1 = (yk[k + 1] - yk[k]) / h1
    if d0 * d1 <= 0.0:
        return 0.0
    w1 = 2.0 * h1 + h0
    w2 = h1 + 2.0 * h0
    return (w1 + w2) / (w1 / d0 + w2 / d1)


@njit(cache=True)
def curve(hrow, offs, center, s):
    """(h(s), h'(s)) from the grid values hrow at center + offs."""
    n = offs.shape[0]
    xx = s - center
    if xx <= offs[0]:
        d = (hrow[1] - hrow[0]) / (offs[1] - offs[0])
        return hrow[0] + d * (xx - offs[0]), d
    if xx >= offs[n - 1]:
        d = (hrow[n - 1] - hrow[n - 2]) / (offs[n - 1] - offs[n - 2])
        return hrow[n - 1] + d * (xx - offs[n - 1]), d
    i = 0
    while offs[i + 1] < xx:
        i += 1
    h = offs[i + 1] - offs[i]
    u = (xx - offs[i]) / h
    dk = _node_slope(offs, hrow, i)
    dk1 = _node_slope(offs, hrow, i + 1)
    y = ((2 * u ** 3 - 3 * u ** 2 + 1) * hrow[i] + (u ** 3 - 2 * u ** 2 + u) * h * dk
         + (-2 * u ** 3 + 3 * u ** 2) * hrow[i + 1] + (u ** 3 - u ** 2) * h * dk1)
    dy = ((6 * u ** 2 - 6 * u) / h * hrow[i] + (3 * u ** 2 - 4 * u + 1) * dk
          + (-6 * u ** 2 + 6 * u) / h * hrow[i + 1] + (3 * u ** 2 - 2 * u) * dk1)
    return y, dy


@njit(cache=True)
def ci_filter4e(par_sets, par_idx, notch, s_obs, V, x, OF, first_day, mat_day, iss, u_prior, grp, slot, gsize,
                smax, beta, wq, use_peer, t0, t1, ll_t0, n0, record, fm_out, fv_out, g_out, um_out, c_out, use_G,
                use_L, use_iss, s_z, V_z, qc, Hc, offs, max_it, tol, it_out, cld_a, dld_a):
    B = notch.shape[0]
    NG = gsize.shape[0]
    g = np.zeros(3)
    Pg = np.zeros((3, 3))
    m = np.zeros((NG, smax))
    Bm = np.zeros((NG, smax, 3))
    S = np.zeros((NG, smax, smax))
    act = np.zeros(B, np.bool_)
    gact = np.zeros(NG, np.bool_)
    newb = np.zeros(B, np.bool_)
    pm = np.zeros(B)
    members = np.full((NG, max(1, (smax - 1) // 4)), -1, np.int64)
    for b in range(B):
        members[grp[b], slot[b]] = b
    ll = 0.0
    Pinv = np.zeros((3, 3))
    Lam = np.zeros((3, 3))
    Lami = np.zeros((3, 3))
    eta = np.zeros(3)
    gnew = np.zeros(3)
    hq = np.zeros((NROW, 5))
    sl = np.zeros(5, np.int64)
    Hg = np.zeros((NROW, 3))
    yv = np.zeros(NROW)
    dv = np.zeros(NROW)
    Rv = np.zeros(NROW)
    W = np.zeros((NROW, NROW))
    Wi = np.zeros((NROW, NROW))
    M = np.zeros((NROW, 3))
    e = np.zeros(NROW)
    HxS = np.zeros((NROW, smax))
    K = np.zeros((smax, NROW))
    tv = np.zeros(smax)
    qv = np.zeros(smax)
    Cxg = np.zeros((smax, 3))
    Vx = np.zeros((smax, smax))
    Cn = np.zeros((smax, 3))
    Vn = np.zeros((smax, smax))
    Dm = np.zeros((smax, 3))
    zg = np.zeros(3)
    h = np.zeros(3)
    cz = np.zeros(5)
    cw = np.zeros(5)
    g_s = np.zeros(3)                     # predicted state of the day: every relinearization restarts from it
    Pg_s = np.zeros((3, 3))
    m_s = np.zeros((NG, smax))
    Bm_s = np.zeros((NG, smax, 3))
    S_s = np.zeros((NG, smax, smax))
    lp = np.full(B, np.nan)               # linearization point (OAS, bp) of each bond's spread observation today
    lpn = np.full(B, np.nan)
    yps = np.full(B, np.nan)              # pseudo-observation in OAS units and its variance
    vps = np.full(B, np.nan)
    lpl = np.full(B, np.nan)              # liquidity linearization point (liquidity 'discount'), its update, and the
    lpln = np.full(B, np.nan)             # weight r of L and D in the measurement row (1 otherwise)
    rps = np.ones(B)
    for t in range(t0, t1 + 1):
        p = par_sets[par_idx[t]]
        mu = p[0:3]
        ph = p[3:6]
        qg = p[6:9]
        phB, qB, s2u = p[9], p[10], p[11]
        phL, qL = p[12], p[13]
        al = p[14:17]
        be = p[17:20]
        hh = p[20:23]
        phD, qD, lam = p[23], p[24], p[25]
        iss_d, iss_v = p[33], p[34]
        phC, qC = p[35], p[36]
        if not (abs(ph[0]) < 1.0 and abs(ph[1]) < 1.0 and abs(ph[2]) < 1.0 and abs(phB) < 1.0
                and abs(phL) < 1.0 and abs(phD) < 1.0 and abs(phC) < 1.0):
            return -np.inf
        if not (qg[0] > 0.0 and qg[1] > 0.0 and qg[2] > 0.0 and qB > 0.0 and s2u >= 0.0 and qL > 0.0
                and qD >= 0.0 and hh[0] > 0.0 and hh[1] > 0.0 and hh[2] > 0.0 and qC > 0.0):
            return -np.inf
        if use_iss and not (iss_v > 0.0):
            return -np.inf
        zL = 1.0 if use_L else 0.0
        cz[0] = 1.0
        cz[1] = 1.0
        cz[2] = 1.0
        cz[3] = zL
        cz[4] = lam
        # ---- common predict
        if not use_G:
            for r in range(3):
                g[r] = mu[r]
                for c in range(3):
                    Pg[r, c] = 0.0
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
            # ---- group predict
            for k in range(NG):
                if not gact[k]:
                    continue
                s = 1 + 4 * gsize[k]
                for r in range(s):
                    tv[r] = 0.0
                    qv[r] = 0.0
                    for c in range(3):
                        Dm[r, c] = 0.0
                tv[0] = phC
                qv[0] = qC
                for j in range(gsize[k]):
                    b = members[k, j]
                    if not act[b]:
                        continue
                    o = 1 + 4 * j
                    tv[o] = phB
                    qv[o] = qB * wq[b, t] * wq[b, t]
                    tv[o + 1] = 1.0
                    tv[o + 2] = phL
                    tv[o + 3] = phD
                    qv[o + 2] = qL
                    qv[o + 3] = qD
                    n_new = notch[b, t]
                    n_old = notch[b, t - 1]
                    if np.isfinite(n_new) and np.isfinite(n_old) and n_new != n_old:
                        Dm[o, 1] = n_new - n_old
                        Dm[o, 2] = (n_new - n0) ** 2 - (n_old - n0) ** 2
                for r in range(s):
                    for c in range(3):
                        s_ = 0.0
                        for q in range(3):
                            s_ += Bm[k, r, q] * Pold[q, c]
                        Cxg[r, c] = s_
                for r in range(s):
                    for c in range(r, s):
                        s_ = S[k, r, c]
                        for q in range(3):
                            s_ += Cxg[r, q] * Bm[k, c, q]
                        Vx[r, c] = s_
                        Vx[c, r] = s_
                for r in range(s):
                    for c in range(3):
                        s_ = tv[r] * Cxg[r, c] * ph[c]
                        for q in range(3):
                            s_ -= Dm[r, q] * Pg[q, c]
                        Cn[r, c] = s_
                for r in range(s):
                    for c in range(r, s):
                        s_ = tv[r] * Vx[r, c] * tv[c]
                        for q in range(3):
                            for l in range(3):
                                s_ += Dm[r, q] * Pg[q, l] * Dm[c, l]
                        for q in range(3):
                            s_ -= tv[r] * Cxg[r, q] * ph[q] * Dm[c, q]
                            s_ -= Dm[r, q] * ph[q] * Cxg[c, q] * tv[c]
                        Vn[r, c] = s_
                        Vn[c, r] = s_
                    Vn[r, r] += qv[r]
                m[k, 0] = tv[0] * m[k, 0]
                for j in range(gsize[k]):
                    b = members[k, j]
                    if not act[b]:
                        continue
                    o = 1 + 4 * j
                    of = OF[b, t - 1] if np.isfinite(OF[b, t - 1]) else 0.0
                    m[k, o] = tv[o] * m[k, o] - (Dm[o, 1] * g[1] + Dm[o, 2] * g[2])
                    m[k, o + 2] = tv[o + 2] * m[k, o + 2]
                    m[k, o + 3] = tv[o + 3] * m[k, o + 3] - of
                for r in range(s):
                    for c in range(3):
                        s_ = 0.0
                        for q in range(3):
                            s_ += Cn[r, q] * Pinv[q, c]
                        Bm[k, r, c] = s_
                for r in range(s):
                    for c in range(r, s):
                        s_ = Vn[r, c]
                        for q in range(3):
                            s_ -= Bm[k, r, q] * Cn[c, q]
                        S[k, r, c] = s_
                        S[k, c, r] = s_
        # ---- activate new bonds (prior mean of u as in v2: regression prior, or 0 with use_peer=False)
        for b in range(B):
            newb[b] = False
            if t >= first_day[b] and not act[b]:
                newb[b] = True
                if use_peer:
                    pm[b] = u_prior[b] - iss_d
                else:
                    pm[b] = 0.0
        for b in range(B):
            if not newb[b]:
                continue
            k = grp[b]
            if not gact[k]:
                gact[k] = True
                m[k, 0] = 0.0
                for r in range(smax):
                    S[k, 0, r] = 0.0
                    S[k, r, 0] = 0.0
                for c in range(3):
                    Bm[k, 0, c] = 0.0
            act[b] = True
            o = 1 + 4 * slot[b]
            for r in range(4):
                m[k, o + r] = 0.0
                for c in range(3):
                    Bm[k, o + r, c] = 0.0
                for c in range(smax):
                    S[k, o + r, c] = 0.0
                    S[k, c, o + r] = 0.0
            m[k, o + 1] = pm[b]
            S[k, o + 1, o + 1] = s2u
            S[k, o + 2, o + 2] = qL / (1.0 - phL * phL)
        # ---- iterated EKF (2026-11-05): the spread observation Z = h(s) + noise, h = the tree's Z of the flat path at
        # the bond's spread s (curve Hc on the grid offs around the day's anchor qc). Linearized at lp: pseudo-
        # observation lp + (Z - h(lp)) / h'(lp), variance V / h'(lp)^2. First pass lp = the anchor (= the OAS
        # transform of oas_obs.oas_transform); then lp = the bond's updated spread, the whole day re-updated
        # from the predicted state, until no lp moves by more than tol (bp) or max_it relinearizations.
        for b in range(B):
            lp[b] = np.nan
            lpl[b] = np.nan
            rps[b] = 1.0
            yps[b] = np.nan
            vps[b] = np.nan
            if (act[b] and np.isfinite(notch[b, t]) and np.isfinite(s_z[b, t]) and np.isfinite(V_z[b, t])
                    and np.isfinite(qc[b, t])):
                lp[b] = qc[b, t]
                if np.isfinite(cld_a[b, t]) and np.isfinite(dld_a[b, t]):
                    lpl[b] = cld_a[b, t]
        if max_it > 0:
            for r in range(3):
                g_s[r] = g[r]
                for c in range(3):
                    Pg_s[r, c] = Pg[r, c]
            for k in range(NG):
                if not gact[k]:
                    continue
                s = 1 + 4 * gsize[k]
                for r in range(s):
                    m_s[k, r] = m[k, r]
                    for c in range(3):
                        Bm_s[k, r, c] = Bm[k, r, c]
                    for c in range(s):
                        S_s[k, r, c] = S[k, r, c]
        dll = 0.0
        jac = 0.0
        it = 0
        for it in range(max_it + 1):
            if it > 0:
                for r in range(3):
                    g[r] = g_s[r]
                    for c in range(3):
                        Pg[r, c] = Pg_s[r, c]
                for k in range(NG):
                    if not gact[k]:
                        continue
                    s = 1 + 4 * gsize[k]
                    for r in range(s):
                        m[k, r] = m_s[k, r]
                        for c in range(3):
                            Bm[k, r, c] = Bm_s[k, r, c]
                        for c in range(s):
                            S[k, r, c] = S_s[k, r, c]
            dll = 0.0
            jac = 0.0
            for b in range(B):
                if not np.isfinite(lp[b]):
                    continue
                hv, cv = curve(Hc[b, t], offs, qc[b, t], lp[b])
                if cv < C_MIN:
                    cv = C_MIN
                if np.isfinite(lpl[b]):        # liquidity 'discount': credit curve + linear liquidity term
                    rps[b] = dld_a[b, t] / cv
                    hv = hv + dld_a[b, t] * (lpl[b] - cld_a[b, t])
                    yps[b] = lp[b] + rps[b] * lpl[b] + (s_z[b, t] - hv) / cv
                else:
                    rps[b] = 1.0
                    yps[b] = lp[b] + (s_z[b, t] - hv) / cv
                vps[b] = V_z[b, t] / (cv * cv)
                if t >= ll_t0:
                    jac += math.log(cv)
            # ---- measurement update: groups independent given g; bonds of a group in sequence
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
            for k in range(NG):
                if not gact[k]:
                    continue
                s = 1 + 4 * gsize[k]
                for j in range(gsize[k]):
                    b = members[k, j]
                    if not act[b]:
                        continue
                    o = 1 + 4 * j
                    sl[0] = 0
                    sl[1] = o
                    sl[2] = o + 1
                    sl[3] = o + 2
                    sl[4] = o + 3
                    pdim = 0
                    for kk in range(3):
                        y = x[kk, b, t]
                        if use_L and np.isfinite(y):
                            for q in range(5):
                                hq[pdim, q] = 0.0
                            for c in range(3):
                                Hg[pdim, c] = 0.0
                            hq[pdim, 3] = be[kk]
                            yv[pdim] = y
                            dv[pdim] = al[kk]
                            Rv[pdim] = hh[kk]
                            pdim += 1
                    nn = notch[b, t]
                    if np.isfinite(nn):
                        x1 = nn - n0
                        y = yps[b]
                        if np.isfinite(y):
                            for q in range(5):
                                hq[pdim, q] = cz[q]
                            hq[pdim, 0] = beta[b, t]
                            hq[pdim, 3] = cz[3] * rps[b]
                            hq[pdim, 4] = cz[4] * rps[b]
                            Hg[pdim, 0] = 1.0
                            Hg[pdim, 1] = x1
                            Hg[pdim, 2] = x1 * x1
                            yv[pdim] = y
                            dv[pdim] = 0.0
                            Rv[pdim] = vps[b]
                            pdim += 1
                        if use_iss and t == first_day[b] and np.isfinite(iss[b]):
                            for q in range(5):
                                hq[pdim, q] = cz[q]
                            hq[pdim, 0] = beta[b, t]
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
                        for c in range(s):
                            s_ = 0.0
                            for q in range(5):
                                if hq[r, q] != 0.0:
                                    s_ += hq[r, q] * S[k, sl[q], c]
                            HxS[r, c] = s_
                    for r in range(pdim):
                        for c in range(pdim):
                            s_ = 0.0
                            for q in range(5):
                                s_ += HxS[r, sl[q]] * hq[c, q]
                            W[r, c] = s_
                        W[r, r] += Rv[r]
                    ok, logw = _solve_spd(W, pdim, Wi)
                    if not ok:
                        return -np.inf
                    for r in range(pdim):
                        pr = dv[r]
                        for q in range(5):
                            pr += hq[r, q] * m[k, sl[q]]
                        for c in range(3):
                            pr += Hg[r, c] * g[c]
                        e[r] = yv[r] - pr
                        for c in range(3):
                            s_ = Hg[r, c]
                            for q in range(5):
                                s_ += hq[r, q] * Bm[k, sl[q], c]
                            M[r, c] = s_
                    for r in range(3):
                        for c in range(3):
                            s_ = 0.0
                            for i in range(pdim):
                                for jj in range(pdim):
                                    s_ += M[i, r] * Wi[i, jj] * M[jj, c]
                            Lam[r, c] += s_
                        s_ = 0.0
                        for i in range(pdim):
                            for jj in range(pdim):
                                s_ += M[i, r] * Wi[i, jj] * e[jj]
                        eta[r] += s_
                    q_ = 0.0
                    for i in range(pdim):
                        for jj in range(pdim):
                            q_ += e[i] * Wi[i, jj] * e[jj]
                    squad += q_
                    slogw += logw
                    nobs += pdim
                    for r in range(s):
                        for c in range(pdim):
                            s_ = 0.0
                            for q in range(pdim):
                                s_ += HxS[q, r] * Wi[q, c]
                            K[r, c] = s_
                    for r in range(s):
                        s_ = 0.0
                        for c in range(pdim):
                            s_ += K[r, c] * e[c]
                        m[k, r] += s_
                        for c in range(3):
                            s2 = 0.0
                            for q in range(pdim):
                                s2 += K[r, q] * M[q, c]
                            Bm[k, r, c] -= s2
                    for r in range(s):
                        for c in range(r, s):
                            s_ = 0.0
                            for q in range(pdim):
                                s_ += K[r, q] * HxS[q, c]
                            v_ = S[k, r, c] - s_
                            S[k, r, c] = v_
                            S[k, c, r] = v_
            if nobs > 0 and not use_G:
                if t >= ll_t0:
                    dll = -0.5 * (nobs * LOG2PI + slogw + squad)
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
                    dll = -0.5 * (nobs * LOG2PI + slogw + math.log(detP) + math.log(detL) + squad - quad)
                for k in range(NG):
                    if not gact[k]:
                        continue
                    s = 1 + 4 * gsize[k]
                    for r in range(s):
                        s_ = 0.0
                        for c in range(3):
                            s_ += Bm[k, r, c] * (gnew[c] - g[c])
                        m[k, r] += s_
                for r in range(3):
                    g[r] = gnew[r]
                    for c in range(3):
                        Pg[r, c] = Lami[r, c]
            if nobs == 0:
                break
            chg = 0.0
            for b in range(B):
                if not np.isfinite(lp[b]):
                    continue
                k = grp[b]
                o = 1 + 4 * slot[b]
                x1 = notch[b, t] - n0
                mcr = g[0] + x1 * g[1] + x1 * x1 * g[2] + beta[b, t] * m[k, 0] + cz[1] * m[k, o] + cz[2] * m[k, o + 1]
                mld = cz[3] * m[k, o + 2] + cz[4] * m[k, o + 3]
                if np.isfinite(lpl[b]):
                    d_ = max(abs(mcr - lp[b]), abs(mld - lpl[b]))
                    lpn[b] = mcr
                    lpln[b] = mld
                else:
                    d_ = abs(mcr + mld - lp[b])
                    lpn[b] = mcr + mld
                if d_ > chg:
                    chg = d_
            if chg <= tol or it == max_it:
                break
            w_ = 1.0 if it < DAMP_AFTER else 0.5
            for b in range(B):
                if np.isfinite(lp[b]):
                    lp[b] = lp[b] + w_ * (lpn[b] - lp[b])
                if np.isfinite(lpl[b]):
                    lpl[b] = lpl[b] + w_ * (lpln[b] - lpl[b])
        it_out[t] = it
        if t >= ll_t0:
            ll += dll - jac         # density of Z: the pseudo-observation's density times |d pseudo / d Z| = 1 / h'
        if record:
            for r in range(3):
                g_out[t, r] = g[r]
            for k in range(NG):
                if gact[k]:
                    c_out[k, t] = m[k, 0]
            for b in range(B):
                if not act[b]:
                    continue
                k = grp[b]
                o = 1 + 4 * slot[b]
                sl[0] = 0
                sl[1] = o
                sl[2] = o + 1
                sl[3] = o + 2
                sl[4] = o + 3
                bt = beta[b, t]
                um_out[b, t, 0] = bt * m[k, 0] + m[k, o]
                um_out[b, t, 1] = m[k, o + 1]
                nn = notch[b, t]
                if not np.isfinite(nn):
                    continue
                x1 = nn - n0
                zg[0] = 1.0
                zg[1] = x1
                zg[2] = x1 * x1
                cw[0] = bt
                for q in range(1, 5):
                    cw[q] = cz[q]
                mv = zg[0] * g[0] + zg[1] * g[1] + zg[2] * g[2]
                for q in range(5):
                    mv += cw[q] * m[k, sl[q]]
                for c in range(3):
                    s_ = zg[c]
                    for q in range(5):
                        s_ += Bm[k, sl[q], c] * cw[q]
                    h[c] = s_
                vv = _q3(h, Pg)
                for q in range(5):
                    for q2 in range(5):
                        vv += cw[q] * S[k, sl[q], sl[q2]] * cw[q2]
                fm_out[b, t] = mv
                fv_out[b, t] = vv
    return ll
