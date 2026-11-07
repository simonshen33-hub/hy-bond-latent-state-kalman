"""fourstate.py -- the four-state panel Kalman model, version 1 (daily, spread space).

States
  common  g_t = (a_t, b_t, c_t)      G_t(n) = a + b (n - n0) + c (n - n0)^2, AR(1) each, diagonal Q
  bond i  x_it = (I, u, L, D)        I AR(1) mean 0; u permanent (Q = 0, prior N(0, s2_u));
                                     L AR(1) mean 0; D AR(1) mean 0 (mu_D = 0 normalisation),
                                     D_{t+1} = phi_D D_t - OF_t + eps (gamma_OF = 1)
Observations on bond-day (i, t)
  s_obs = G_t(n_it) + I + u + L + lam_D D + e,  e ~ N(0, V_it)   (daily combination of the trades, below)
  x_k   = alpha_k + beta_k L + e_k,  e_k ~ N(0, h_k),  k = 1..3   (liquidity measures)
Trades j on (i, t):  Z_j = s + b_j + e_j,  e_j ~ N(0, R_j)
  b_j = g1 + g2 lv_j (customer buy), g3 + g4 lv_j (customer sell), 0 otherwise
  log R_j = d0 + d1 lv_j + d2 customer_j            (lv = standardised log par volume)
  s_obs = sum (Z_j - b_j)/R_j / sum 1/R_j,  V = 1 / sum 1/R_j; the within-day remainder of the trade likelihood
  is added exactly (wll), so the total equals the trade-level likelihood with the state fixed within the day.
Rating change (notch n_old -> n_new effective on day t): I_t gets -(E[b_t] d1 + E[c_t] d2),
  d1 = n_new - n_old, d2 = (n_new - n0)^2 - (n_old - n0)^2  (the G jump is offset inside I: a reclassification is not news
  about the bond).
Initial: common states and I, u, L of each bond (when it becomes active) at the stationary distribution; D = 0 on
  the bond's first trading day, or the flow-accumulated value if the bond started before the window (_d_start).

Filters
  approx_filter : partitioned filter (common 3x3 + per-bond 4x4, cross-covariances dropped after each day);
                  s_obs of a day enter jointly through the common block (exact under that independence),
                  each bond's own update uses the leave-one-out common posterior.
  exact_filter  : full joint covariance (3 + 4B states), sequential univariate updates -- for validation.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

LOG2PI = math.log(2.0 * math.pi)
NAMES = ['mu_a', 'mu_b', 'mu_c', 'phi_a', 'phi_b', 'phi_c', 'q_a', 'q_b', 'q_c',
         'phi_I', 'q_I', 's2_u',
         'phi_L', 'q_L', 'al1', 'al2', 'al3', 'be1', 'be2', 'be3', 'h1', 'h2', 'h3',
         'phi_D', 'q_D', 'lam_D',
         'g1', 'g2', 'g3', 'g4', 'd0', 'd1', 'd2']
NP = len(NAMES)
IDX = {n: i for i, n in enumerate(NAMES)}
PHI = [IDX[n] for n in ('phi_a', 'phi_b', 'phi_c', 'phi_I', 'phi_L', 'phi_D')]
POS = [IDX[n] for n in ('q_a', 'q_b', 'q_c', 'q_I', 's2_u', 'q_L', 'h1', 'h2', 'h3', 'q_D')]


def to_u(nat):
    u = np.array(nat, dtype=float).copy()
    u[PHI] = np.arctanh(u[PHI])
    u[POS] = np.log(u[POS])
    return u


def from_u(u):
    nat = np.array(u, dtype=float).copy()
    nat[PHI] = np.tanh(nat[PHI])
    nat[POS] = np.exp(nat[POS])
    return nat


# ----------------------------------------------------------------------------------------------
# panel operations
# ----------------------------------------------------------------------------------------------
@njit(cache=True)
def activity(ptr, t_day, t_vol_log, t_cat, hide, B, D):
    """Per bond-day: trade count, par volume, customer net volume (buy - sell), skipping hidden days."""
    n = np.zeros((B, D), np.int32)
    vol = np.zeros((B, D))
    net = np.zeros((B, D))
    for b in range(B):
        for j in range(ptr[b], ptr[b + 1]):
            t = t_day[j]
            if hide[b, t]:
                continue
            v = math.exp(t_vol_log[j])
            n[b, t] += 1
            vol[b, t] += v
            if t_cat[j] == 1:
                net[b, t] += v
            elif t_cat[j] == 2:
                net[b, t] -= v
    return n, vol, net


@njit(cache=True)
def measures(n, vol, amt, first_day):
    """x1 = log(1 + trades), x2 = volume / amount issued, x3 = log(business days since the previous
    trading day); NaN before the bond is active (x3 also NaN until a first visible trade)."""
    B, D = n.shape
    x = np.full((3, B, D), np.nan)
    for b in range(B):
        last = -1
        for t in range(first_day[b], D):
            x[0, b, t] = math.log(1.0 + n[b, t])
            if amt[b] > 0:
                x[1, b, t] = vol[b, t] / amt[b]
            if last >= 0:
                x[2, b, t] = math.log(t - last)
            if n[b, t] > 0:
                last = t
    return x


@njit(parallel=True, cache=True)
def daily_obs(ptr, t_day, t_z, t_lvc, t_cat, hide, g, d, B, D, t0, t1):
    """s_obs, V and within-day log-likelihood remainder per bond-day on days t0..t1 (NaN / 0 elsewhere or
    where no visible trade)."""
    s = np.full((B, D), np.nan)
    V = np.full((B, D), np.nan)
    wll = np.zeros((B, D))
    for b in prange(B):
        lo = ptr[b] + np.searchsorted(t_day[ptr[b]:ptr[b + 1]], t0)
        hi = ptr[b] + np.searchsorted(t_day[ptr[b]:ptr[b + 1]], t1 + 1)
        j = lo
        while j < hi:
            t = t_day[j]
            k = j
            while k < hi and t_day[k] == t:
                k += 1
            if not hide[b, t]:
                sw = 0.0
                swy = 0.0
                swyy = 0.0
                slr = 0.0
                for q in range(j, k):
                    c = t_cat[q]
                    lv = t_lvc[q]
                    bias = 0.0
                    cust = 0.0
                    if c == 1:
                        bias = g[0] + g[1] * lv
                        cust = 1.0
                    elif c == 2:
                        bias = g[2] + g[3] * lv
                        cust = 1.0
                    lr = d[0] + d[1] * lv + d[2] * cust
                    w = math.exp(-lr)
                    y = t_z[q] - bias
                    sw += w
                    swy += w * y
                    swyy += w * y * y
                    slr += lr
                yb = swy / sw
                s[b, t] = yb
                V[b, t] = 1.0 / sw
                m = k - j
                ss = max(swyy - sw * yb * yb, 0.0)
                wll[b, t] = -0.5 * ((m - 1) * LOG2PI + slr + math.log(sw) + ss)
            j = k
    return s, V, wll


# ----------------------------------------------------------------------------------------------
# small linear algebra
# ----------------------------------------------------------------------------------------------
@njit(cache=True)
def _inv3(A, out):
    a, b, c = A[0, 0], A[0, 1], A[0, 2]
    d, e, f = A[1, 0], A[1, 1], A[1, 2]
    g, h, i = A[2, 0], A[2, 1], A[2, 2]
    co0 = e * i - f * h
    co1 = -(d * i - f * g)
    co2 = d * h - e * g
    det = a * co0 + b * co1 + c * co2
    if not (det > 0.0):
        return -1.0
    out[0, 0] = co0 / det
    out[0, 1] = -(b * i - c * h) / det
    out[0, 2] = (b * f - c * e) / det
    out[1, 0] = co1 / det
    out[1, 1] = (a * i - c * g) / det
    out[1, 2] = -(a * f - c * d) / det
    out[2, 0] = co2 / det
    out[2, 1] = -(a * h - b * g) / det
    out[2, 2] = (a * e - b * d) / det
    return det


@njit(cache=True)
def _q3(z, M):
    s = 0.0
    for r in range(3):
        for c in range(3):
            s += z[r] * M[r, c] * z[c]
    return s


# ----------------------------------------------------------------------------------------------
# approximate (partitioned) filter
# ----------------------------------------------------------------------------------------------
@njit(cache=True)
def approx_filter(par_sets, par_idx, notch, s_obs, V, x, OF, first_day, t0, t1, n0, record,
                  fm_out, fv_out):
    """Run days t0..t1 (inclusive). par_sets (K, NP) natural parameters; par_idx[t] picks the set used for the
    transition into day t and the update on day t. Returns total log-likelihood (-inf on failure).
    If record: fm_out / fv_out (B, D) receive the filtered fair-spread mean / variance on active bond-days."""
    B = notch.shape[0]
    g = np.zeros(3)
    Pg = np.zeros((3, 3))
    xs = np.zeros((B, 4))
    Px = np.zeros((B, 4, 4))
    act = np.zeros(B, np.bool_)
    ll = 0.0
    Pinv = np.zeros((3, 3))
    A = np.zeros((3, 3))
    Ai = np.zeros((3, 3))
    Am1 = np.zeros((3, 3))
    eta = np.zeros(3)
    zg = np.zeros(3)
    zx = np.zeros(4)
    gp = np.zeros(3)
    Pgp = np.zeros((3, 3))
    nb_obs = np.zeros(B, np.int64)
    rr = np.zeros(B)
    ee = np.zeros(B)
    zz = np.zeros((B, 3))
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
        # ---- common: init or predict
        if t == t0:
            for r in range(3):
                g[r] = mu[r]
                for c in range(3):
                    Pg[r, c] = 0.0
                Pg[r, r] = qg[r] / (1.0 - ph[r] * ph[r])
        else:
            for r in range(3):
                g[r] = mu[r] + ph[r] * (g[r] - mu[r])
            for r in range(3):
                for c in range(3):
                    Pg[r, c] = ph[r] * ph[c] * Pg[r, c]
                Pg[r, r] += qg[r]
        # ---- bonds: init or predict
        for b in range(B):
            if t < first_day[b]:
                continue
            if not act[b]:
                act[b] = True
                xs[b, 0] = 0.0
                xs[b, 1] = 0.0
                xs[b, 2] = 0.0
                xs[b, 3] = 0.0
                for r in range(4):
                    for c in range(4):
                        Px[b, r, c] = 0.0
                Px[b, 0, 0] = qI / (1.0 - phI * phI)
                Px[b, 1, 1] = s2u
                Px[b, 2, 2] = qL / (1.0 - phL * phL)
                Px[b, 3, 3] = qD / (1.0 - phD * phD)
                continue
            tr0 = phI
            xs[b, 0] = phI * xs[b, 0]
            xs[b, 2] = phL * xs[b, 2]
            xs[b, 3] = phD * xs[b, 3] - (OF[b, t - 1] if np.isfinite(OF[b, t - 1]) else 0.0)
            tv = (phI, 1.0, phL, phD)
            for r in range(4):
                for c in range(4):
                    Px[b, r, c] = tv[r] * tv[c] * Px[b, r, c]
            Px[b, 0, 0] += qI
            Px[b, 2, 2] += qL
            Px[b, 3, 3] += qD
            n_new = notch[b, t]
            n_old = notch[b, t - 1]
            if np.isfinite(n_new) and np.isfinite(n_old) and n_new != n_old:
                d1 = n_new - n_old
                d2 = (n_new - n0) ** 2 - (n_old - n0) ** 2
                xs[b, 0] -= g[1] * d1 + g[2] * d2
                Px[b, 0, 0] += d1 * d1 * Pg[1, 1] + 2.0 * d1 * d2 * Pg[1, 2] + d2 * d2 * Pg[2, 2]
        # ---- liquidity measures (own states only)
        for b in range(B):
            if not act[b]:
                continue
            for k in range(3):
                y = x[k, b, t]
                if not np.isfinite(y):
                    continue
                pz0 = Px[b, 0, 2] * be[k]
                pz1 = Px[b, 1, 2] * be[k]
                pz2 = Px[b, 2, 2] * be[k]
                pz3 = Px[b, 3, 2] * be[k]
                f = be[k] * pz2 + hh[k]
                if not (f > 0.0):
                    return -np.inf
                v = y - al[k] - be[k] * xs[b, 2]
                ll += -0.5 * (LOG2PI + math.log(f) + v * v / f)
                kk = (pz0 / f, pz1 / f, pz2 / f, pz3 / f)
                pzv = (pz0, pz1, pz2, pz3)
                for r in range(4):
                    xs[b, r] += kk[r] * v
                for r in range(4):
                    for c in range(4):
                        Px[b, r, c] -= kk[r] * pzv[c]
        # ---- spread observations: joint through the common block
        detP = _inv3(Pg, Pinv)
        if detP <= 0.0:
            return -np.inf
        for r in range(3):
            eta[r] = 0.0
            for c in range(3):
                A[r, c] = Pinv[r, c]
        n_obs = 0
        sum_logr = 0.0
        sum_e2r = 0.0
        zx[0] = 1.0
        zx[1] = 1.0
        zx[2] = 1.0
        zx[3] = lam
        for b in range(B):
            if not act[b]:
                continue
            y = s_obs[b, t]
            nn = notch[b, t]
            if not (np.isfinite(y) and np.isfinite(nn)):
                continue
            x1 = nn - n0
            zz[n_obs, 0] = 1.0
            zz[n_obs, 1] = x1
            zz[n_obs, 2] = x1 * x1
            m = xs[b, 0] + xs[b, 1] + xs[b, 2] + lam * xs[b, 3]
            o = 0.0
            for r in range(4):
                for c in range(4):
                    o += zx[r] * Px[b, r, c] * zx[c]
            r_ = V[b, t] + o
            if not (r_ > 0.0):
                return -np.inf
            e = y - m - (zz[n_obs, 0] * g[0] + zz[n_obs, 1] * g[1] + zz[n_obs, 2] * g[2])
            nb_obs[n_obs] = b
            rr[n_obs] = r_
            ee[n_obs] = e
            for r in range(3):
                eta[r] += zz[n_obs, r] * e / r_
                for c in range(3):
                    A[r, c] += zz[n_obs, r] * zz[n_obs, c] / r_
            sum_logr += math.log(r_)
            sum_e2r += e * e / r_
            n_obs += 1
        if n_obs > 0:
            detA = _inv3(A, Am1)
            if detA <= 0.0:
                return -np.inf
            quad = 0.0
            for r in range(3):
                for c in range(3):
                    quad += eta[r] * Am1[r, c] * eta[c]
            ll += -0.5 * (n_obs * LOG2PI + sum_logr + math.log(detP) + math.log(detA) + sum_e2r - quad)
            # posterior of g (all bonds)
            for r in range(3):
                gp[r] = g[r]
                for c in range(3):
                    gp[r] += Am1[r, c] * eta[c]
                    Pgp[r, c] = Am1[r, c]
            # own updates with leave-one-out common posterior
            for q in range(n_obs):
                b = nb_obs[q]
                for r in range(3):
                    zg[r] = zz[q, r]
                for r in range(3):
                    for c in range(3):
                        Ai[r, c] = A[r, c] - zg[r] * zg[c] / rr[q]
                da = _inv3(Ai, Am1)
                if da <= 0.0:
                    return -np.inf
                # g_{-i} = g + Ai^{-1} (eta - zg e/r)
                gm = 0.0
                for r in range(3):
                    gr = g[r]
                    for c in range(3):
                        gr += Am1[r, c] * (eta[c] - zg[c] * ee[q] / rr[q])
                    gm += zg[r] * gr
                pg = _q3(zg, Am1)
                o = 0.0
                pzx = np.zeros(4)
                for r in range(4):
                    for c in range(4):
                        pzx[r] += Px[b, r, c] * zx[c]
                    o += zx[r] * pzx[r]
                f = pg + o + V[b, t]
                m = xs[b, 0] + xs[b, 1] + xs[b, 2] + lam * xs[b, 3]
                v = s_obs[b, t] - gm - m
                for r in range(4):
                    xs[b, r] += pzx[r] / f * v
                for r in range(4):
                    for c in range(4):
                        Px[b, r, c] -= pzx[r] * pzx[c] / f
            for r in range(3):
                g[r] = gp[r]
                for c in range(3):
                    Pg[r, c] = Pgp[r, c]
        # ---- record fair spread
        if record:
            for b in range(B):
                if not act[b]:
                    continue
                nn = notch[b, t]
                if not np.isfinite(nn):
                    continue
                x1 = nn - n0
                zg[0] = 1.0
                zg[1] = x1
                zg[2] = x1 * x1
                mval = zg[0] * g[0] + zg[1] * g[1] + zg[2] * g[2] + xs[b, 0] + xs[b, 1] + xs[b, 2] + lam * xs[b, 3]
                vv = _q3(zg, Pg)
                for r in range(4):
                    for c in range(4):
                        vv += zx[r] * Px[b, r, c] * zx[c]
                fm_out[b, t] = mval
                fv_out[b, t] = vv
    return ll


@njit(cache=True)
def _d_start(OF, b, first, t, phD, qD):
    """Dealer-inventory prior when bond b becomes active on day t (decided 2026-10-07, option b'):
    D = 0 on the bond's first trading day; if the bond started before t (window start), D is accumulated from
    the customer order flow of days first..t-1 with the current phi_D: mean = sum phi^(t-1-s) (-OF_s),
    variance = q_D * sum phi^(2k) over the same days (the unobserved non-customer shocks)."""
    d = 0.0
    v = 0.0
    for s in range(first, t):
        of = OF[b, s] if np.isfinite(OF[b, s]) else 0.0
        d = phD * d - of
        v = phD * phD * v + qD
    return d, v


# ----------------------------------------------------------------------------------------------
# conditional-independence filter (used for estimation and evaluation)
# ----------------------------------------------------------------------------------------------
@njit(cache=True)
def _solve_spd(W, p, Winv):
    """Inverse and log-determinant of the leading p x p block of SPD W (Cholesky). Returns (ok, logdet)."""
    L = np.zeros((4, 4))
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
    # invert L then Winv = L^-T L^-1
    Li = np.zeros((4, 4))
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
def ci_filter(par_sets, par_idx, notch, s_obs, V, x, OF, first_day, t0, t1, n0, record, fm_out, fv_out,
              g_out, use_G=True, use_L=True):
    """Bond states are kept in conditional form x_i | g ~ N(m_i + B_i (g - gbar), S_i), i.e. bonds are
    conditionally independent given today's common state. The measurement update of a day is exact under that
    structure (star-shaped model g - x_i - y_i); the prediction step drops the cross-bond dependence that is not
    carried by the new common state. Same inputs / outputs as approx_filter.
    Ablations: use_G=False fixes the common curve at its mean (no daily common states); use_L=False removes L
    from the spread and ignores the liquidity measures. With record, g_out (D, 3) gets the filtered common states."""
    B = notch.shape[0]
    g = np.zeros(3)
    Pg = np.zeros((3, 3))
    m = np.zeros((B, 4))
    Bm = np.zeros((B, 4, 3))
    S = np.zeros((B, 4, 4))
    act = np.zeros(B, np.bool_)
    ll = 0.0
    Pinv = np.zeros((3, 3))
    Lam = np.zeros((3, 3))
    Lami = np.zeros((3, 3))
    eta = np.zeros(3)
    gnew = np.zeros(3)
    Pgn = np.zeros((3, 3))
    Pgni = np.zeros((3, 3))
    Hx = np.zeros((4, 4))
    Hg = np.zeros((4, 3))
    yv = np.zeros(4)
    dv = np.zeros(4)
    Rv = np.zeros(4)
    W = np.zeros((4, 4))
    Wi = np.zeros((4, 4))
    M = np.zeros((4, 3))
    HxS = np.zeros((4, 4))
    K = np.zeros((4, 4))
    e = np.zeros(4)
    Vx = np.zeros((4, 4))
    Cxg = np.zeros((4, 3))
    Cn = np.zeros((4, 3))
    Vn = np.zeros((4, 4))
    TV = np.zeros((4, 4))
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
        if not (abs(ph[0]) < 1.0 and abs(ph[1]) < 1.0 and abs(ph[2]) < 1.0 and abs(phI) < 1.0
                and abs(phL) < 1.0 and abs(phD) < 1.0):
            return -np.inf
        if not (qg[0] > 0.0 and qg[1] > 0.0 and qg[2] > 0.0 and qI > 0.0 and s2u >= 0.0 and qL > 0.0
                and qD >= 0.0 and hh[0] > 0.0 and hh[1] > 0.0 and hh[2] > 0.0):
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
                # Vx = S + B Pold B',  Cxg = B Pold
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
                # reclassification matrix D (row 0: d1 on b, d2 on c)
                for r in range(4):
                    for c in range(3):
                        Dm[r, c] = 0.0
                n_new = notch[b, t]
                n_old = notch[b, t - 1]
                if np.isfinite(n_new) and np.isfinite(n_old) and n_new != n_old:
                    Dm[0, 1] = n_new - n_old
                    Dm[0, 2] = (n_new - n0) ** 2 - (n_old - n0) ** 2
                # cov(x', g') = T Cxg Phi' - D Pg'
                for r in range(4):
                    for c in range(3):
                        s_ = tv[r] * Cxg[r, c] * ph[c]
                        for k in range(3):
                            s_ -= Dm[r, k] * Pg[k, c]
                        Cn[r, c] = s_
                # Var(x') = T Vx T' + Q + D Pg' D' - T cov(x,g') D' - D cov(g',x) T'
                # cov(x, g') = Cxg Phi'
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
                # mean
                of = OF[b, t - 1] if np.isfinite(OF[b, t - 1]) else 0.0
                m0 = tv[0] * m[b, 0] - (Dm[0, 1] * g[1] + Dm[0, 2] * g[2])
                m[b, 1] = m[b, 1]
                m[b, 2] = tv[2] * m[b, 2]
                m[b, 3] = tv[3] * m[b, 3] - of
                m[b, 0] = m0
                # B' = Cn Pg'^-1 ; S' = Vn - Cn Pg'^-1 Cn'
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
        # ---- activate new bonds
        for b in range(B):
            if t >= first_day[b] and not act[b]:
                act[b] = True
                for r in range(4):
                    m[b, r] = 0.0
                    for c in range(3):
                        Bm[b, r, c] = 0.0
                    for c in range(4):
                        S[b, r, c] = 0.0
                S[b, 0, 0] = qI / (1.0 - phI * phI)
                S[b, 1, 1] = s2u
                S[b, 2, 2] = qL / (1.0 - phL * phL)
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
            y = s_obs[b, t]
            nn = notch[b, t]
            if np.isfinite(y) and np.isfinite(nn):
                x1 = nn - n0
                for c in range(4):
                    Hx[pdim, c] = zx[c]
                Hg[pdim, 0] = 1.0
                Hg[pdim, 1] = x1
                Hg[pdim, 2] = x1 * x1
                yv[pdim] = y
                dv[pdim] = 0.0
                Rv[pdim] = V[b, t]
                pdim += 1
            if pdim == 0:
                continue
            # HxS, W, M, e
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
            # accumulate common information
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
            # K = S Hx' Wi  (4 x pdim)
            for r in range(4):
                for c in range(pdim):
                    s_ = 0.0
                    for k in range(pdim):
                        s_ += HxS[k, r] * Wi[k, c]
                    K[r, c] = s_
            # m += K e ; B -= K M ; S -= K HxS
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
            ll += -0.5 * (nobs * LOG2PI + slogw + math.log(detP) + math.log(detL) + squad - quad)
            # re-centre every active bond at the new common mean
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


# ----------------------------------------------------------------------------------------------
# exact joint filter (validation only)
# ----------------------------------------------------------------------------------------------
@njit(cache=True)
def exact_filter(p, notch, s_obs, V, x, OF, first_day, t0, t1, n0, record, fm_out, fv_out):
    """Full joint covariance over 3 + 4B states, single parameter set. Same model and conventions as
    approx_filter (bonds enter at their first active day with the stationary prior, independent of the rest)."""
    B = notch.shape[0]
    m = 3 + 4 * B
    a = np.zeros(m)
    P = np.zeros((m, m))
    act = np.zeros(B, np.bool_)
    mu = p[0:3]
    ph = p[3:6]
    qg = p[6:9]
    phI, qI, s2u = p[9], p[10], p[11]
    phL, qL = p[12], p[13]
    al = p[14:17]
    be = p[17:20]
    hh = p[20:23]
    phD, qD, lam = p[23], p[24], p[25]
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
            # mean
            gb, gc = 0.0, 0.0
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
            # covariance: diagonal part
            for r in range(m):
                for c in range(m):
                    P[r, c] *= tv[r] * tv[c]
            for r in range(3):
                P[r, r] += qg[r]
            # reclassification rows: I_b <- I_b - d1 * b - d2 * c   (applied to predicted moments)
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
            if t >= first_day[b] and not act[b]:
                act[b] = True
                o = 3 + 4 * b
                for r in range(4):
                    a[o + r] = 0.0
                P[o, o] = qI / (1.0 - phI * phI)
                P[o + 1, o + 1] = s2u
                P[o + 2, o + 2] = qL / (1.0 - phL * phL)
                dmean, dvar = _d_start(OF, b, first_day[b], t, phD, qD)
                a[o + 3] = dmean
                P[o + 3, o + 3] = dvar
        # sequential univariate updates: measures then spreads
        for b in range(B):
            if not act[b]:
                continue
            o = 3 + 4 * b
            for k in range(4):
                if k < 3:
                    y = x[k, b, t]
                    if not np.isfinite(y):
                        continue
                    nz = 1
                    idx[0] = o + 2
                    zv[0] = be[k]
                    d0 = al[k]
                    h = hh[k]
                else:
                    y = s_obs[b, t]
                    nn = notch[b, t]
                    if not (np.isfinite(y) and np.isfinite(nn)):
                        continue
                    x1 = nn - n0
                    nz = 7
                    idx[0], idx[1], idx[2] = 0, 1, 2
                    zv[0], zv[1], zv[2] = 1.0, x1, x1 * x1
                    idx[3], idx[4], idx[5], idx[6] = o, o + 1, o + 2, o + 3
                    zv[3], zv[4], zv[5], zv[6] = 1.0, 1.0, 1.0, lam
                    d0 = 0.0
                    h = V[b, t]
                for r in range(m):
                    s = 0.0
                    for q in range(nz):
                        s += P[r, idx[q]] * zv[q]
                    Pz[r] = s
                f = h
                pred = d0
                for q in range(nz):
                    f += zv[q] * Pz[idx[q]]
                    pred += zv[q] * a[idx[q]]
                if not (f > 0.0):
                    return -np.inf
                v = y - pred
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
