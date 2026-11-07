"""fourstate4.py -- model v4: v2 plus an issuer factor.

I of bond b in issuer k is split into  I_b = beta_{b,t} C_k + B_b :
  C_k  issuer factor, AR(1) with phi_C, q_C, shared by all bonds of the issuer; 0 (variance 0) on the day the
       issuer's first bond starts trading
  B_b  the bond's own part, AR(1) with phi_B, q_B (positions of v2's phi_I, q_I); 0 (variance 0) on the bond's
       first trading day; a rating reclassification of the bond is offset here: B_b gets -(d1 b_t + d2 c_t)
  beta loading, an input: a duration-times-spread rule refreshed monthly with no look-ahead
  wq   weight of B's shock: variance q_B wq[b, t]^2 on day t (wq = 1: v4; variant dts_B, 2026-10-11: the same
       rule with all bonds as one group, passed in as wq)
u, L, D as in v2 (u unchanged, decided 2026-10-11): prior mean of u for a new bond = the v2 regression prior
u_prior - iss_d (0 with use_peer=False), variance s2_u. Spread observation of bond b:
G_t(n_b) + beta_{b,t} C_k + B_b + u_b + L_b + lam_D D_b (+ iss_d for the issue Z).
A new bond of an issuer that is already trading inherits the issuer's current C at once.
With q_C -> 0 the model is v2 (checked in test_filters_synth.py).
Parameters: fourstate2.NAMES (phi_I, q_I read as phi_B, q_B) + phi_C, q_C (37).

Filter: issuers are conditionally independent given the common states g (the v2 star structure with an issuer
block in place of a bond); inside an issuer the update is exact, bond by bond. Slots: 0 = C_k; bond j of the
issuer at 1 + 4 j: B, u, L, D.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from fourstate import LOG2PI, _inv3, _q3
import fourstate2 as fs2
from fourstate2 import _solve_spd
from fourstate3 import group_layout   # noqa: F401  (re-exported)

NAMES = fs2.NAMES + ['phi_C', 'q_C']
NP = len(NAMES)
IDX = {n: i for i, n in enumerate(NAMES)}
PHI = fs2.PHI + [IDX['phi_C']]
POS = fs2.POS + [IDX['q_C']]


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

NROW = 5


@njit(cache=True)
def ci_filter4(par_sets, par_idx, notch, s_obs, V, x, OF, first_day, mat_day, iss, u_prior, grp, slot, gsize,
               smax, beta, wq, use_peer, t0, t1, ll_t0, n0, record, fm_out, fv_out, g_out, um_out, c_out, use_G,
               use_L, use_iss):
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
                    y = s_obs[b, t]
                    if np.isfinite(y):
                        for q in range(5):
                            hq[pdim, q] = cz[q]
                        hq[pdim, 0] = beta[b, t]
                        Hg[pdim, 0] = 1.0
                        Hg[pdim, 1] = x1
                        Hg[pdim, 2] = x1 * x1
                        yv[pdim] = y
                        dv[pdim] = 0.0
                        Rv[pdim] = V[b, t]
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


@njit(cache=True)
def exact_filter4(p, notch, s_obs, V, x, OF, first_day, mat_day, iss, u_prior, grp, NG, beta, wq, use_peer, t0,
                  t1, ll_t0, n0, use_iss, record, fm_out, fv_out):
    """Full joint covariance (validation of ci_filter4): states g (3), C_k (NG), and B, u, L, D per bond."""
    B = notch.shape[0]
    m = 3 + NG + 4 * B
    a = np.zeros(m)
    P = np.zeros((m, m))
    act = np.zeros(B, np.bool_)
    gact = np.zeros(NG, np.bool_)
    newb = np.zeros(B, np.bool_)
    pm = np.zeros(B)
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
    tv = np.ones(m)
    for r in range(3):
        tv[r] = ph[r]
    for k in range(NG):
        tv[3 + k] = phC
    for b in range(B):
        o = 3 + NG + 4 * b
        tv[o] = phB
        tv[o + 2] = phL
        tv[o + 3] = phD
    ll = 0.0
    Pz = np.zeros(m)
    idx = np.zeros(8, np.int64)
    zv = np.zeros(8)
    for t in range(t0, t1 + 1):
        if t == t0:
            for r in range(3):
                a[r] = mu[r]
                P[r, r] = qg[r] / (1.0 - ph[r] * ph[r])
        else:
            for r in range(3):
                a[r] = mu[r] + ph[r] * (a[r] - mu[r])
            gb, gc = a[1], a[2]
            for k in range(NG):
                if gact[k]:
                    a[3 + k] = phC * a[3 + k]
            for b in range(B):
                if not act[b]:
                    continue
                o = 3 + NG + 4 * b
                a[o] = phB * a[o]
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
                    o = 3 + NG + 4 * b
                    a[o] -= d1 * gb + d2 * gc
                    for c in range(m):
                        P[o, c] -= d1 * P[1, c] + d2 * P[2, c]
                    for r in range(m):
                        P[r, o] -= d1 * P[r, 1] + d2 * P[r, 2]
            for k in range(NG):
                if gact[k]:
                    P[3 + k, 3 + k] += qC
            for b in range(B):
                if not act[b]:
                    continue
                o = 3 + NG + 4 * b
                P[o, o] += qB * wq[b, t] * wq[b, t]
                P[o + 2, o + 2] += qL
                P[o + 3, o + 3] += qD
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
                a[3 + k] = 0.0
            act[b] = True
            o = 3 + NG + 4 * b
            for r in range(4):
                a[o + r] = 0.0
            a[o + 1] = pm[b]
            P[o + 1, o + 1] = s2u
            P[o + 2, o + 2] = qL / (1.0 - phL * phL)
        for b in range(B):
            if not act[b]:
                continue
            o = 3 + NG + 4 * b
            kI = 3 + grp[b]
            for kk in range(5):
                if kk < 3:
                    y = x[kk, b, t]
                    if not np.isfinite(y):
                        continue
                    nz = 1
                    idx[0] = o + 2
                    zv[0] = be[kk]
                    d0 = al[kk]
                    hv = hh[kk]
                else:
                    nn = notch[b, t]
                    if not np.isfinite(nn):
                        continue
                    if kk == 3:
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
                    nz = 8
                    idx[0], idx[1], idx[2] = 0, 1, 2
                    zv[0], zv[1], zv[2] = 1.0, x1, x1 * x1
                    idx[3], idx[4], idx[5], idx[6], idx[7] = kI, o, o + 1, o + 2, o + 3
                    zv[3], zv[4], zv[5], zv[6], zv[7] = beta[b, t], 1.0, 1.0, 1.0, lam
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
                o = 3 + NG + 4 * b
                x1 = nn - n0
                idx[0], idx[1], idx[2] = 0, 1, 2
                zv[0], zv[1], zv[2] = 1.0, x1, x1 * x1
                idx[3], idx[4], idx[5], idx[6], idx[7] = 3 + grp[b], o, o + 1, o + 2, o + 3
                zv[3], zv[4], zv[5], zv[6], zv[7] = beta[b, t], 1.0, 1.0, 1.0, lam
                mv = 0.0
                vv = 0.0
                for q in range(8):
                    mv += zv[q] * a[idx[q]]
                    for q2 in range(8):
                        vv += zv[q] * P[idx[q], idx[q2]] * zv[q2]
                fm_out[b, t] = mv
                fv_out[b, t] = vv
    return ll
