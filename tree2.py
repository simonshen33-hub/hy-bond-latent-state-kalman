"""tree2.py -- layer 2 of the call model: a two-factor trinomial tree (rate x credit). See README,
'Layer 2: the lattice'.

Clocks: variance accrues per business day (both factors are AR(1) per business day); interest accrues in the Z
engine's 30/360 years from settlement.
  rate    shadow short rate x, AR(1) per business day (phi_r, q_r) = Hull-White in business time; the rate used for
          discounting is max(alpha_i + x, 0) (lower bound 0, T1); alpha_i is fitted step by step so that the tree
          reproduces today's Treasury discount curve P(T) = (1 + z(T)/2)^(-2T) (curve.py, the Z engine) exactly
  credit  the bond's credit-only spread s_cr = m_i + y: m_i = the exact expected path of G + beta C + B + u of the
          multi-factor model (P dynamics, decision B), y = AR(1) per business day (phi_s, q_s) fitted to the
          multi-factor variance (T3)
  correlation rho of the daily shocks (T4): joint branch probability p_a q_b (1 + delta (a - Ea)(b - Eb)) with
          delta = rho / (sd_a sd_b) at each node pair (exact marginals, exact cross moment); where a joint
          probability would be negative, delta is shrunk at that node pair and the node pair is counted
Discount over step i at a node: exp(-max(alpha_i + x, 0) dT_i) x F_i(s), F_i(s) = the Z-engine discount ratio of
the step at spread s divided by the Treasury ratio P(T_{i+1}) / P(T_i). With zero volatility and no mean reversion
the tree is the layer-1 price (callprice.price) exactly (test_tree2.py). s_tot = s_cr + c_ld, c_ld = L + lam_D D
frozen at today's filtered value (T8).
Grid: every k-th business day after settlement and the last business day on or before maturity. A cash flow is
credited at the first grid date on or after its date with the factor D_s(T_cf) / D_s(T_grid) at the node's spread
(a flow after the last grid date -- maturity on a non-business day -- is credited there with a factor < 1).
Calls (T7, as callprice.price): possible on grid dates on or after the first call date; hazard over the step
lam x dv, dv = 30/360 years since the previous grid date (or since the business day before the first call date);
lam = exp(zeta0 + zeta1 M), M = W / H - 1, W = credit-only value at the node of the flows after the date (backward
pass 1), H = call price in force + accrued since the last flow credited before the date; itm_only: lam = 0 when
M < 0. Price (backward pass 2): X_i = q_i H_i + (1 - q_i)(CF_i + E[d_i X_{i+1}]); the dirty price is X_0.
Pruning: the node range of each factor grows by reachability alone (one node per
side per step), so with near-unit daily persistence a 400-step tree carries ~800 nodes per factor on the last date
(check: 283-997 rate x 49-857 credit nodes, 1.9 s a price). PRUNE_SD = K > 0 also caps the range at
+-(ceil(K sd_cum / dx) + 1) nodes, sd_cum = the factor's sd after the cumulative business days from settlement; a node
whose branch would leave the range branches inward (centre clamped, probabilities from the same mean and variance,
negatives clamped and renormalised: exact only to the mass beyond K sd). K = 0 (default) is the full tree, bit for
bit. Set by the environment variable P26A_TREE_PRUNE_SD (read at import, so spawned workers inherit it).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

import bonds as bondlib


G_MIN = 1e-3     # numerical device (2026-10-11): floor of 1 + (z + x)/2 in the tree's discounting
import os as _os
PRUNE_SD = float(_os.environ.get('P26A_TREE_PRUNE_SD', '0'))   # numerical device (2026-11-05): 0 = full tree


@njit(cache=True)
def _D(z, x, t):
    """Z-engine discount factor. Floored base (numerical device, found by run_tree_t5.py 2026-10-11): with a
    near-unit-root credit factor the daily tree has far nodes with spreads near -2 (decimal), where the Z-engine base
    1 + (z + x)/2 turns non-positive and the power is NaN; those nodes have probability ~1e-200, but 0 x NaN = NaN
    would spoil the whole price. Below G_MIN the base is floored, which keeps them finite and irrelevant."""
    g = 1.0 + (z + x) / 2.0
    if g < G_MIN:
        g = G_MIN
    return g ** (-2.0 * t)


@njit(cache=True)
def _bd_search(bd_ord, v, right):
    """first index with bd_ord > v (right) or >= v (not right)."""
    lo, hi = 0, bd_ord.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if (bd_ord[mid] <= v) if right else (bd_ord[mid] < v):
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def schedule(b, s_ord, sy, sm, sd, ptr, cf_ord, cf_ymd, cf_amt, iss_ymd, iss_ord, coupon, dc_eu, eom,
             cptr, c_ord, c_px, bd_ord, bd_ymd, k, idx, T, nbd, H, dv, cfa, cfT):
    """Grid of one bond-day (arrays sized >= number of business days to maturity + 2). Date 0 = settlement.
    Returns (N, after_amt, after_T); H[i] = nan when no call is possible on date i; cfa/cfT: the flow credited on
    date i (0 if none) and its own time."""
    g0, g1 = ptr[b], ptr[b + 1]
    mat_ord = cf_ord[g1 - 1]
    i0 = _bd_search(bd_ord, s_ord, True)
    i1 = _bd_search(bd_ord, mat_ord, True) - 1
    idx[0] = i0 - 1
    T[0] = 0.0
    nbd[0] = 0
    H[0] = np.nan
    dv[0] = 0.0
    cfa[0] = 0.0
    cfT[0] = 0.0
    N = 0
    if i1 >= i0:
        j = i0 + k - 1
        while j < i1:
            N += 1
            idx[N] = j
            j += k
        N += 1
        idx[N] = i1
    callable_bond = cptr[b + 1] > cptr[b] and c_ord[cptr[b]] < mat_ord
    c0, c1 = cptr[b], cptr[b + 1]
    fcb = _bd_search(bd_ord, c_ord[c0], False) if callable_bond else bd_ord.shape[0]
    for i in range(1, N + 1):
        v = bd_ord[idx[i]]
        T[i] = bondlib._dc(sy, sm, sd, bd_ymd[idx[i], 0], bd_ymd[idx[i], 1], bd_ymd[idx[i], 2], dc_eu[b], eom[b]) / 360.0
        nbd[i] = idx[i] - idx[i - 1]
        prev_ord = s_ord if i == 1 else bd_ord[idx[i - 1]]
        # flow credited on date i
        cfa[i] = 0.0
        cfT[i] = 0.0
        for jj in range(g0, g1):
            if cf_ord[jj] > prev_ord and cf_ord[jj] <= v:
                cfa[i] += cf_amt[jj]
                cfT[i] = bondlib._dc(sy, sm, sd, cf_ymd[jj, 0], cf_ymd[jj, 1], cf_ymd[jj, 2], dc_eu[b], eom[b]) / 360.0
        H[i] = np.nan
        dv[i] = 0.0
        if callable_bond and v >= c_ord[c0]:
            cidx = c0
            while cidx + 1 < c1 and c_ord[cidx + 1] <= v:
                cidx += 1
            last = -1
            for jj in range(g0, g1):
                if cf_ord[jj] <= prev_ord:
                    last = jj
            if last >= 0:
                ac = coupon[b] * bondlib._dc(cf_ymd[last, 0], cf_ymd[last, 1], cf_ymd[last, 2], bd_ymd[idx[i], 0],
                                             bd_ymd[idx[i], 1], bd_ymd[idx[i], 2], dc_eu[b], eom[b]) / 360.0
            else:
                ac = coupon[b] * bondlib._dc(iss_ymd[b, 0], iss_ymd[b, 1], iss_ymd[b, 2], bd_ymd[idx[i], 0],
                                             bd_ymd[idx[i], 1], bd_ymd[idx[i], 2], dc_eu[b], eom[b]) / 360.0
            H[i] = c_px[cidx] + ac
            # hazard start: the previous grid date, or the business day before the first callable business day
            if fcb - 1 > idx[i - 1]:
                ts = bondlib._dc(sy, sm, sd, bd_ymd[fcb - 1, 0], bd_ymd[fcb - 1, 1], bd_ymd[fcb - 1, 2], dc_eu[b],
                                 eom[b]) / 360.0
            else:
                ts = T[i - 1]
            dv[i] = T[i] - max(ts, 0.0)
    after_amt = 0.0
    after_T = 0.0
    last_ord = bd_ord[idx[N]] if N > 0 else s_ord
    for jj in range(g0, g1):
        if cf_ord[jj] > last_ord and cf_ord[jj] > s_ord:
            after_amt += cf_amt[jj]
            after_T = bondlib._dc(sy, sm, sd, cf_ymd[jj, 0], cf_ymd[jj, 1], cf_ymd[jj, 2], dc_eu[b], eom[b]) / 360.0
    return N, after_amt, after_T


@njit(cache=True)
def ar1_var(phi, q, n):
    if phi >= 1.0:
        return q * n
    return q * (1.0 - phi ** (2 * n)) / (1.0 - phi * phi)


@njit(cache=True)
def _branch(j, phin, v):
    """Trinomial branch of node j (value j dx) over one step with mean factor phin and variance v (in dx^2 units):
    centre node, pd, pm, pu."""
    E = j * phin
    kc = int(math.floor(E + 0.5))
    eta = E - kc
    pu = 0.5 * (v + eta * eta + eta)
    pd = 0.5 * (v + eta * eta - eta)
    pm = 1.0 - v - eta * eta
    return kc, pd, pm, pu


@njit(cache=True)
def _clean(p0, p1, p2):
    """clamp negative branch probabilities at 0 and renormalise; returns (p0, p1, p2, mean, sd, clamped)."""
    c = 1.0 if (p0 < 0.0 or p1 < 0.0 or p2 < 0.0) else 0.0
    p0 = max(p0, 0.0)
    p1 = max(p1, 0.0)
    p2 = max(p2, 0.0)
    t = p0 + p1 + p2
    p0 /= t
    p1 /= t
    p2 /= t
    E = p2 - p0
    return p0, p1, p2, E, math.sqrt(max(p0 + p2 - E * E, 0.0)), c


@njit(cache=True)
def _joint(pr0, pr1, pr2, Ea, sda, ps0, ps1, ps2, Eb, sdb, rho, out):
    """out[a, b] = pr_a ps_b (1 + delta (a - 1 - Ea)(b - 1 - Eb)), delta = rho / (sda sdb), shrunk if a joint
    probability would be negative. Returns 1.0 if shrunk."""
    pr = (pr0, pr1, pr2)
    ps = (ps0, ps1, ps2)
    delta = rho / (sda * sdb) if (sda > 0.0 and sdb > 0.0 and rho != 0.0) else 0.0
    shrunk = 0.0
    if delta != 0.0:
        mn = 0.0
        for a_ in range(3):
            for b_ in range(3):
                if pr[a_] > 0.0 and ps[b_] > 0.0:
                    c = (a_ - 1 - Ea) * (b_ - 1 - Eb) * delta
                    if c < mn:
                        mn = c
        if mn < -1.0:
            delta = delta / (-mn)
            shrunk = 1.0
    for a_ in range(3):
        for b_ in range(3):
            out[a_, b_] = pr[a_] * ps[b_] * (1.0 + delta * (a_ - 1 - Ea) * (b_ - 1 - Eb))
    return shrunk


@njit(cache=True)
def _callprob(zeta0, zeta1, M, itm_only, dv):
    """call probability over a step: 1 - exp(-lam dv), lam = exp(zeta0 + zeta1 M) (0 when itm_only and M < 0).
    Numerical guards (found by debug_tree_k1.py 2026-10-11): a step of zero 30/360 length (e.g. the 30th to the 31st)
    has probability 0, and the exponent is capped at 700, so that far nodes (huge W, lam overflowing to inf) never
    give inf x 0 = NaN."""
    if dv <= 0.0 or (itm_only and M < 0.0):
        return 0.0
    ex = zeta0 + zeta1 * M
    if ex > 700.0:
        ex = 700.0
    return 1.0 - math.exp(-math.exp(ex) * dv)


@njit(cache=True)
def _ranges(N, nbd, phi, q, dx, lo, hi, ksd=0.0):
    lo[0] = 0
    hi[0] = 0
    cum = 0
    for i in range(N):
        n = nbd[i + 1]
        cum += n
        if dx == 0.0:
            lo[i + 1] = 0
            hi[i + 1] = 0
            continue
        phin = phi ** n
        a = 1 << 30
        z = -(1 << 30)
        for j in range(lo[i], hi[i] + 1):
            kc = int(math.floor(j * phin + 0.5))
            if kc - 1 < a:
                a = kc - 1
            if kc + 1 > z:
                z = kc + 1
        if ksd > 0.0:
            J = int(math.ceil(ksd * math.sqrt(ar1_var(phi, q, cum)) / dx)) + 1
            if a < -J:
                a = -J
            if z > J:
                z = J
        lo[i + 1] = a
        hi[i + 1] = z


@njit(cache=True)
def _branch_in(j, phin, v, lo1, hi1):
    """_branch with the children kept inside [lo1, hi1] (pruned tree): returns centre, pd, pm, pu, clamped."""
    kc, pd, pm, pu = _branch(j, phin, v)
    if kc - 1 >= lo1 and kc + 1 <= hi1:
        return kc, pd, pm, pu, 0.0
    E = j * phin
    if kc - 1 < lo1:
        kc = lo1 + 1
    if kc + 1 > hi1:
        kc = hi1 - 1
    eta = E - kc
    pu = 0.5 * (v + eta * eta + eta)
    pd = 0.5 * (v + eta * eta - eta)
    pm = 1.0 - v - eta * eta
    return kc, pd, pm, pu, 1.0


@njit(cache=True)
def price(N, nbd, T, H, dv, cfa, cfT, after_amt, after_T, x, coef, zn, phr, qr, phs, qs, rho, mpath, c_ld,
          zeta0, zeta1, itm_only, kref, info, alpha_out, R, prune_sd=0.0):
    """Dirty price at settlement. mpath[i] = expected credit-only spread (decimal) on grid date i. R = recovery of
    face value (default branch, 2026-10-22; R = 0 is the tree without default, bit for bit): at a node with credit-only
    spread s the default intensity is lam = s / (1 - R) (credit triangle); over a step the holder discounts at
    rate + c_ld (survival path and default path alike), survives with S = F(lam + c_ld) / F(c_ld) and on default
    receives 100 R at the step end (no accrued); the credit-only value W (the call decision) is the same with
    c_ld = 0, S = F(lam). Flows credited between a cash-flow date and its grid date, and after the last grid date, use
    lam + c_ld (lam with c_ld = 0 for W). info (6,):
    [negative branch probabilities clamped, node pairs with delta shrunk, steps where the floored curve fit was
    impossible, rate nodes on the last date, credit nodes on the last date, largest relative curve-fit error
    |sum Q_i - P_i| / P_i]. alpha_out (N,): the fitted rate shifts."""
    for t in range(6):
        info[t] = 0.0
    Vr = ar1_var(phr, qr, kref)
    Vs = ar1_var(phs, qs, kref)
    dxr = math.sqrt(3.0 * Vr) if Vr > 0.0 else 0.0
    dxs = math.sqrt(3.0 * Vs) if Vs > 0.0 else 0.0
    lor = np.zeros(N + 1, np.int64)
    hir = np.zeros(N + 1, np.int64)
    los = np.zeros(N + 1, np.int64)
    his = np.zeros(N + 1, np.int64)
    _ranges(N, nbd, phr, qr, dxr, lor, hir, prune_sd)
    _ranges(N, nbd, phs, qs, dxs, los, his, prune_sd)
    info[3] = hir[N] - lor[N] + 1
    info[4] = his[N] - los[N] + 1
    zT = np.empty(N + 1)
    P = np.empty(N + 1)
    for i in range(N + 1):
        zT[i] = bondlib._zero_at(x, coef, zn, T[i])
        P[i] = _D(zT[i], 0.0, T[i])
    # ---- forward: fit alpha_i so the floored rate tree reproduces P
    alpha = np.zeros(N)
    W0 = int(np.max(hir - lor)) + 1
    Q = np.zeros(W0)
    Qn = np.zeros(W0)
    Q[0] = 1.0
    prev_alpha = 0.0
    for i in range(N):
        dT = T[i + 1] - T[i]
        nr = hir[i] - lor[i] + 1
        tot = 0.0
        for jj in range(nr):
            tot += Q[jj]
        target = P[i + 1]
        if dT <= 0.0:
            a = prev_alpha
        else:
            # f(a) = sum Q exp(-max(a + x, 0) dT) - target, decreasing in a
            xmax = hir[i] * dxr
            if tot - target < 0.0:
                a = -xmax - 1.0          # all nodes at the floor: the closest the floored tree can come
                info[2] += 1.0
            else:
                a = prev_alpha
                lo_a = -xmax - 1.0       # f(lo_a) = tot - target >= 0
                hi_a = 1.0
                while True:
                    f = -target
                    for jj in range(nr):
                        r = hi_a + (lor[i] + jj) * dxr
                        f += Q[jj] * math.exp(-(r if r > 0.0 else 0.0) * dT)
                    if f <= 0.0:
                        break
                    hi_a *= 2.0
                for _ in range(200):
                    f = -target
                    d = 0.0
                    for jj in range(nr):
                        r = a + (lor[i] + jj) * dxr
                        if r > 0.0:
                            e = Q[jj] * math.exp(-r * dT)
                            f += e
                            d -= dT * e
                        else:
                            f += Q[jj]
                    if f > 0.0:
                        lo_a = a
                    else:
                        hi_a = a
                    if abs(f) <= 1e-14 * target:
                        break
                    na = a - f / d if d < 0.0 else 0.5 * (lo_a + hi_a)
                    if not (lo_a < na < hi_a):
                        na = 0.5 * (lo_a + hi_a)
                    if na == a:
                        break
                    a = na
        alpha[i] = a
        alpha_out[i] = a
        prev_alpha = a
        # propagate Arrow-Debreu prices
        nr1 = hir[i + 1] - lor[i + 1] + 1
        for jj in range(nr1):
            Qn[jj] = 0.0
        phin = phr ** nbd[i + 1]
        vv = ar1_var(phr, qr, nbd[i + 1]) / (dxr * dxr) if dxr > 0.0 else 0.0
        for jj in range(nr):
            j = lor[i] + jj
            r = a + j * dxr
            disc = math.exp(-(r if r > 0.0 else 0.0) * dT) if dT > 0.0 else 1.0
            if dxr == 0.0:
                Qn[0] += Q[jj] * disc
                continue
            kc, pd, pm, pu, clp = _branch_in(j, phin, vv, lor[i + 1], hir[i + 1])
            if clp > 0.0:
                pd, pm, pu, _e, _s, _c = _clean(pd, pm, pu)
            Qn[kc - 1 - lor[i + 1]] += Q[jj] * disc * pd
            Qn[kc - lor[i + 1]] += Q[jj] * disc * pm
            Qn[kc + 1 - lor[i + 1]] += Q[jj] * disc * pu
        sq = 0.0
        for jj in range(nr1):
            Q[jj] = Qn[jj]
            sq += Qn[jj]
        if info[2] == 0.0:
            info[5] = max(info[5], abs(sq - P[i + 1]) / P[i + 1])
    # ---- backward: pass 1 (W, credit-only, flows after the date) and pass 2 (X, price) together
    Ws = int(np.max(his - los)) + 1
    Wn = np.zeros((W0, Ws))
    Xn = np.zeros((W0, Ws))
    Wc = np.zeros((W0, Ws))
    Xc = np.zeros((W0, Ws))
    # date N
    nr = hir[N] - lor[N] + 1
    ns = his[N] - los[N] + 1
    for js in range(ns):
        scr = mpath[N] + (los[N] + js) * dxs
        if R > 0.0:
            scr = scr / (1.0 - R)          # default intensity lam (credit triangle)
        stt = scr + c_ld
        dN_cr = _D(zT[N], scr, T[N])
        dN_tt = _D(zT[N], stt, T[N])
        a_cr = 0.0
        a_tt = 0.0
        if after_amt != 0.0:
            za = bondlib._zero_at(x, coef, zn, after_T)
            a_cr = after_amt * _D(za, scr, after_T) / dN_cr
            a_tt = after_amt * _D(za, stt, after_T) / dN_tt
        cf_tt = 0.0
        if cfa[N] != 0.0:
            zc = bondlib._zero_at(x, coef, zn, cfT[N])
            cf_tt = cfa[N] * _D(zc, stt, cfT[N]) / dN_tt
        q = 0.0
        if not np.isnan(H[N]):
            M = a_cr / H[N] - 1.0
            q = _callprob(zeta0, zeta1, M, itm_only, dv[N])
        for jr in range(nr):
            Wn[jr, js] = a_cr
            Xn[jr, js] = q * H[N] + (1.0 - q) * (cf_tt + a_tt) if not np.isnan(H[N]) else cf_tt + a_tt
    # per-step credit-node quantities
    Fcr = np.empty(Ws)
    Ftt = np.empty(Ws)
    cks = np.zeros(Ws, np.int64)
    cp0 = np.empty(Ws)
    cp1 = np.empty(Ws)
    cp2 = np.empty(Ws)
    cEb = np.empty(Ws)
    csd = np.empty(Ws)
    ccl = np.empty(Ws)
    J = np.empty((3, 3))
    cfcr = np.empty(Ws)       # flow credited at the child date, credit-only, per child credit node
    cftt = np.empty(Ws)
    for i in range(N - 1, -1, -1):
        dT = T[i + 1] - T[i]
        nr = hir[i] - lor[i] + 1
        ns = his[i] - los[i] + 1
        nr1 = hir[i + 1] - lor[i + 1] + 1
        ns1 = his[i + 1] - los[i + 1] + 1
        ratioP = P[i + 1] / P[i]
        Fl = (_D(zT[i + 1], c_ld, T[i + 1]) / _D(zT[i], c_ld, T[i])) / ratioP if R > 0.0 else 1.0
        for js in range(ns):
            scr = mpath[i] + (los[i] + js) * dxs
            if R > 0.0:
                scr = scr / (1.0 - R)
            stt = scr + c_ld
            Fcr[js] = (_D(zT[i + 1], scr, T[i + 1]) / _D(zT[i], scr, T[i])) / ratioP
            Ftt[js] = (_D(zT[i + 1], stt, T[i + 1]) / _D(zT[i], stt, T[i])) / ratioP
        zc = bondlib._zero_at(x, coef, zn, cfT[i + 1]) if cfa[i + 1] != 0.0 else 0.0
        for js in range(ns1):
            scr = mpath[i + 1] + (los[i + 1] + js) * dxs
            if R > 0.0:
                scr = scr / (1.0 - R)
            stt = scr + c_ld
            if cfa[i + 1] != 0.0:
                cfcr[js] = cfa[i + 1] * _D(zc, scr, cfT[i + 1]) / _D(zT[i + 1], scr, T[i + 1])
                cftt[js] = cfa[i + 1] * _D(zc, stt, cfT[i + 1]) / _D(zT[i + 1], stt, T[i + 1])
            else:
                cfcr[js] = 0.0
                cftt[js] = 0.0
        # flow credited on date i itself (used in X_i)
        cf_i = cfa[i] != 0.0
        zci = bondlib._zero_at(x, coef, zn, cfT[i]) if cf_i else 0.0
        phinr = phr ** nbd[i + 1]
        vvr = ar1_var(phr, qr, nbd[i + 1]) / (dxr * dxr) if dxr > 0.0 else 0.0
        phins = phs ** nbd[i + 1]
        vvs = ar1_var(phs, qs, nbd[i + 1]) / (dxs * dxs) if dxs > 0.0 else 0.0
        for js in range(ns):
            if dxs > 0.0:
                kk, a0, a1, a2, _cl = _branch_in(los[i] + js, phins, vvs, los[i + 1], his[i + 1])
            else:
                kk, a0, a1, a2 = 0, 0.0, 1.0, 0.0
            cks[js] = kk
            cp0[js], cp1[js], cp2[js], cEb[js], csd[js], ccl[js] = _clean(a0, a1, a2)
        for jr in range(nr):
            j_r = lor[i] + jr
            r = alpha[i] + j_r * dxr
            dr = math.exp(-(r if r > 0.0 else 0.0) * dT) if dT > 0.0 else 1.0
            if dxr > 0.0:
                kr, pr0, pr1, pr2, _cl = _branch_in(j_r, phinr, vvr, lor[i + 1], hir[i + 1])
            else:
                kr, pr0, pr1, pr2 = 0, 0.0, 1.0, 0.0
            pr0, pr1, pr2, Ea, sda, cl = _clean(pr0, pr1, pr2)
            info[0] += cl
            for js in range(ns):
                j_s = los[i] + js
                info[0] += ccl[js] if jr == 0 else 0.0
                info[1] += _joint(pr0, pr1, pr2, Ea, sda, cp0[js], cp1[js], cp2[js], cEb[js], csd[js], rho, J)
                ew = 0.0
                ex = 0.0
                for a_ in range(3):
                    cr = kr - 1 + a_ - lor[i + 1] if dxr > 0.0 else 0
                    for b_ in range(3):
                        p = J[a_, b_]
                        if p == 0.0:
                            continue
                        cs = cks[js] - 1 + b_ - los[i + 1] if dxs > 0.0 else 0
                        ew += p * (cfcr[cs] + Wn[cr, cs])
                        ex += p * Xn[cr, cs]
                Wv = dr * Fcr[js] * ew
                C = dr * Ftt[js] * ex
                if R > 0.0:                    # default within the step: 100 R at the step end
                    Wv += dr * (1.0 - Fcr[js]) * 100.0 * R
                    C += dr * (Fl - Ftt[js]) * 100.0 * R
                scr = mpath[i] + j_s * dxs
                if R > 0.0:
                    scr = scr / (1.0 - R)
                stt = scr + c_ld
                cfv = 0.0
                if cf_i:
                    cfv = cfa[i] * _D(zci, stt, cfT[i]) / _D(zT[i], stt, T[i])
                if np.isnan(H[i]):
                    Xv = cfv + C
                else:
                    M = Wv / H[i] - 1.0
                    q = _callprob(zeta0, zeta1, M, itm_only, dv[i])
                    Xv = q * H[i] + (1.0 - q) * (cfv + C)
                Wc[jr, js] = Wv
                Xc[jr, js] = Xv
        for jr in range(nr):
            for js in range(ns):
                Wn[jr, js] = Wc[jr, js]
                Xn[jr, js] = Xc[jr, js]
    return Xn[0, 0]


def fit_ar1(n_h, V_h):
    """(phi, q) of an AR(1) per business day whose variance at the horizons n_h (business days) is closest to V_h in
    log least squares; phi = exp(-th), th on [0, ln 2] (half-life >= 1 business day), log q in closed form given th.
    Search: th = 0 and a log-spaced grid of th in [1e-9, ln 2] (the objective is flat for large th, which defeats a
    plain golden-section search -- found by test_tree2.py 2026-10-11), then golden section between the grid
    neighbours of the best point. Returns (phi, q, th_at_bound)."""
    n_h = np.asarray(n_h, float)
    lv = np.log(np.asarray(V_h, float))

    def obj(th):
        S = n_h if th == 0.0 else np.expm1(-2.0 * th * n_h) / np.expm1(-2.0 * th)
        ls = np.log(S)
        lq = np.mean(lv - ls)
        return float(np.sum((lv - lq - ls) ** 2)), lq

    grid_th = np.concatenate([[0.0], np.exp(np.linspace(math.log(1e-9), math.log(math.log(2.0)), 400))])
    f = np.array([obj(t)[0] for t in grid_th])
    m = int(np.argmin(f))
    lo = grid_th[max(m - 1, 0)]
    hi = grid_th[min(m + 1, len(grid_th) - 1)]
    g = (math.sqrt(5.0) - 1.0) / 2.0
    a, b_ = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = obj(a)[0], obj(b_)[0]
    for _ in range(100):
        if fa <= fb:
            hi, b_, fb = b_, a, fa
            a = hi - g * (hi - lo)
            fa = obj(a)[0]
        else:
            lo, a, fa = a, b_, fb
            b_ = lo + g * (hi - lo)
            fb = obj(b_)[0]
    cand = [(f[m], grid_th[m]), (obj(0.5 * (lo + hi))[0], 0.5 * (lo + hi))]
    _, th = min(cand)
    lq = obj(th)[1]
    return math.exp(-th), math.exp(lq), bool(th == 0.0 or th >= grid_th[-1])


def afns_short_rate_var(p, h):
    """Q variance of the B-AFNS shadow short rate L + S at horizon h years (independent-factor AFNS, afns.py):
    s1^2 h + s2^2 (1 - e^{-2 lam h}) / (2 lam) + s3^2 [1/(4 lam) - e^{-2 lam h} (lam h^2/2 + h/2 + 1/(4 lam))]."""
    lam, s1, s2, s3 = p[0], p[7], p[8], p[9]
    h = np.asarray(h, float)
    e = np.exp(-2.0 * lam * h)
    return (s1 ** 2 * h + s2 ** 2 * (1.0 - e) / (2.0 * lam)
            + s3 ** 2 * (1.0 / (4.0 * lam) - e * (lam * h * h / 2.0 + h / 2.0 + 1.0 / (4.0 * lam))))


def credit_var(n, w, phis, qs):
    """Multi-factor variance of the bond's credit-only spread after n business days: sum_c w_c^2 Var_AR1(c)."""
    return sum(wc * wc * ar1_var(ph, q, n) for wc, ph, q in zip(w, phis, qs))


@njit(cache=True)
def joint_moments(N, nbd, phr, qr, phs, qs, rho, kref, out):
    """Forward propagation of the joint branch probabilities (no discounting) with the same branching as price():
    out[i] = (E x, E y, Var x, Var y, Cov) on date i, x and y in factor units (validation of the tree)."""
    Vr = ar1_var(phr, qr, kref)
    Vs = ar1_var(phs, qs, kref)
    dxr = math.sqrt(3.0 * Vr) if Vr > 0.0 else 0.0
    dxs = math.sqrt(3.0 * Vs) if Vs > 0.0 else 0.0
    lor = np.zeros(N + 1, np.int64)
    hir = np.zeros(N + 1, np.int64)
    los = np.zeros(N + 1, np.int64)
    his = np.zeros(N + 1, np.int64)
    _ranges(N, nbd, phr, qr, dxr, lor, hir)
    _ranges(N, nbd, phs, qs, dxs, los, his)
    W0 = int(np.max(hir - lor)) + 1
    Ws = int(np.max(his - los)) + 1
    Pm = np.zeros((W0, Ws))
    Pn = np.zeros((W0, Ws))
    Pm[0, 0] = 1.0
    J = np.empty((3, 3))
    for i in range(N + 1):
        ex = ey = exx = eyy = exy = 0.0
        for jr in range(hir[i] - lor[i] + 1):
            for js in range(his[i] - los[i] + 1):
                p = Pm[jr, js]
                if p == 0.0:
                    continue
                xv = (lor[i] + jr) * dxr
                yv = (los[i] + js) * dxs
                ex += p * xv
                ey += p * yv
                exx += p * xv * xv
                eyy += p * yv * yv
                exy += p * xv * yv
        out[i, 0] = ex
        out[i, 1] = ey
        out[i, 2] = exx - ex * ex
        out[i, 3] = eyy - ey * ey
        out[i, 4] = exy - ex * ey
        if i == N:
            break
        for jr in range(hir[i + 1] - lor[i + 1] + 1):
            for js in range(his[i + 1] - los[i + 1] + 1):
                Pn[jr, js] = 0.0
        n = nbd[i + 1]
        vvr = ar1_var(phr, qr, n) / (dxr * dxr) if dxr > 0.0 else 0.0
        vvs = ar1_var(phs, qs, n) / (dxs * dxs) if dxs > 0.0 else 0.0
        for jr in range(hir[i] - lor[i] + 1):
            j_r = lor[i] + jr
            if dxr > 0.0:
                kr, a0, a1, a2 = _branch(j_r, phr ** n, vvr)
            else:
                kr, a0, a1, a2 = 0, 0.0, 1.0, 0.0
            pr0, pr1, pr2, Ea, sda, _c = _clean(a0, a1, a2)
            for js in range(his[i] - los[i] + 1):
                p = Pm[jr, js]
                if p == 0.0:
                    continue
                j_s = los[i] + js
                if dxs > 0.0:
                    ks, b0, b1, b2 = _branch(j_s, phs ** n, vvs)
                else:
                    ks, b0, b1, b2 = 0, 0.0, 1.0, 0.0
                ps0, ps1, ps2, Eb, sdb, _c = _clean(b0, b1, b2)
                _joint(pr0, pr1, pr2, Ea, sda, ps0, ps1, ps2, Eb, sdb, rho, J)
                for a_ in range(3):
                    cr = kr - 1 + a_ - lor[i + 1] if dxr > 0.0 else 0
                    for b_ in range(3):
                        if J[a_, b_] == 0.0:
                            continue
                        cs = ks - 1 + b_ - los[i + 1] if dxs > 0.0 else 0
                        Pn[cr, cs] += p * J[a_, b_]
        for jr in range(hir[i + 1] - lor[i + 1] + 1):
            for js in range(his[i + 1] - los[i + 1] + 1):
                Pm[jr, js] = Pn[jr, js]


def grid(E, e, settle, k):
    """Schedule arrays of terms-bond e settling on `settle` (pandas Timestamp) with every k-th business day."""
    import pandas as pd
    B = E.B
    s_ord = int((pd.Timestamp(settle) - pd.Timestamp('1970-01-01')).days)
    mat_ord = int(B.cf_ord[B.ptr[e + 1] - 1])
    nmax = int(np.searchsorted(E.bd_ord, mat_ord, side='right') - np.searchsorted(E.bd_ord, s_ord, side='right')) + 2
    nmax = max(nmax, 2)
    idx = np.zeros(nmax, np.int64)
    T = np.zeros(nmax)
    nbd = np.zeros(nmax, np.int64)
    H = np.zeros(nmax)
    dv = np.zeros(nmax)
    cfa = np.zeros(nmax)
    cfT = np.zeros(nmax)
    d = pd.Timestamp(settle)
    N, aa, aT = schedule(e, s_ord, d.year, d.month, d.day, B.ptr, B.cf_ord, B.cf_ymd, B.cf_amt, B.iss_ymd, B.iss_ord,
                         B.coupon, B.dc_eu, B.eom, E.cptr, E.c_ord, E.c_px, E.bd_ord, E.bd_ymd, k, idx, T, nbd, H, dv,
                         cfa, cfT)
    return dict(N=N, idx=idx, T=T, nbd=nbd, H=H, dv=dv, cfa=cfa, cfT=cfT, after_amt=aa, after_T=aT)


def price_bondday(E, e, settle, crow, k, phr, qr, phs, qs, rho, mpath_fn, c_ld, zeta0, zeta1, itm_only, R):
    """Dirty price, info, alpha and the grid of one bond-day. mpath_fn(nbd_cum) -> expected credit-only spread
    (decimal) after that many business days (a vectorised function). R: recovery of face value of the default branch
    (0 = no default branch, the tree before 2026-10-22)."""
    g = grid(E, e, settle, k)
    N = g['N']
    cum = np.cumsum(g['nbd'][:N + 1])
    mpath = np.asarray(mpath_fn(cum), float)
    info = np.zeros(6)
    alpha = np.zeros(max(N, 1))
    P = price(N, g['nbd'], g['T'], g['H'], g['dv'], g['cfa'], g['cfT'], g['after_amt'], g['after_T'], E.zc.x,
              E.zc.coef[crow], E.zc.z_nodes[crow], phr, qr, phs, qs, rho, mpath, c_ld, zeta0, zeta1, itm_only, k,
              info, alpha, R, PRUNE_SD)
    return P, info, alpha, g
