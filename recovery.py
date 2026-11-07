"""recovery.py -- recovery rates by payment-rank group, estimated from the cross-section of an issuer's bonds;
used for the default branch of the layer-2 tree (tree2.py).

METHOD ONLY. The private project's data loader (every usable bond-day of the estimation panel) and its monthly
driver are not part of this repository: they read the licensed trade panel. What is here is the pricing kernel, the
profiled-likelihood objective and the window fit; `synth.py` builds the arrays they take.

Pricing (recovery of face value, RFV) of bond i on a day, default intensity lam (per year), recovery R of its group,
with the Z engine's conventions (curve.py zero rates z(t), semiannual compounding, 30/360 times t from settlement):
  survival   S(t) = D(t; lam) / D(t; 0),   D(t; x) = (1 + (z(t) + x)/2)^(-2t)      (so S(0) = 1)
  dirty      P(lam, R) = sum_k CF_k D(t_k; lam) + 100 R sum_j D(u_j; 0) [S(u_{j-1}) - S(u_j)]
             u_j = j h (j = 1, 2, ...), the last u at the final cash flow; default paid at the end of each interval
  R = 0 gives exactly the Z engine's price at spread lam (checked in test_lattice_synth.py), so lam is a
  spread-like intensity.
  h (years) is a numerical integration step and lam_max a numerical search bound. Both are numerical settings
  chosen by a convergence check, and the values in force are WITHHELD: pass them in (see params.template.json).
Data (in the private project): every bond-day with a visible trade (all trades, daily mean clean price; dirty =
clean + accrued), excluding a bond-day whose price is at or above the call price that applies (in force if callable,
else at the first call date): there the price is capped by the call, which RFV pricing does not model.
Issuer-days with at least two such bonds identify R: one common lam per issuer-day (profiled out by Gauss-Newton,
lam in [0, lam_max]), R per group common to the window; objective = sum of squared price errors (points).
Windows: the monthly 3-year windows of the credit model, so each month's R uses past data only, like every other
parameter.
Groups: secured (Secured, 1st lien), senior unsecured (Sr Unsecured, Unsecured), junior (2nd lien, Sr Subordinated,
Subordinated).
Search for R: a grid of common R values (numerical search device, step 0.05) then L-BFGS-B on the three R (logit
transform) from the best grid point, with the envelope gradient (window_grad). Every issuer-day is solved from both
ends of [0, lam_max] and the lower SSE is kept, so the profiled objective carries no history and the gradient is
analytic (an earlier version warm-started each issuer-day and its forward differences were numerical noise).
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

import bonds as bondlib

GROUPS = ('secured', 'senior unsecured', 'junior')
RANK_GROUP = {'Secured': 0, '1st lien': 0, 'Sr Unsecured': 1, 'Unsecured': 1, '2nd lien': 2, 'Sr Subordinated': 2,
              'Subordinated': 2}
GRID = np.round(np.arange(0.0, 0.951, 0.05), 2)


# ------------------------------------------------------------------------------------------ pricing kernels
@njit(cache=True)
def _disc(z, x, t):
    return (1.0 + (z + x) / 2.0) ** (-2.0 * t)


@njit(cache=True)
def rfv(t, a, rz, n, lam, R, h, cx, ccoef, czn):
    """(dirty price, d price / d lam) under RFV. t, a, rz: cash-flow times, amounts, zero rates (n); recovery grid
    u = h, 2h, ... and the final cash-flow time, zero rates from the curve row (cx, ccoef, czn)."""
    p = 0.0
    dp = 0.0
    for k in range(n):
        d = _disc(rz[k], lam, t[k])
        p += a[k] * d
        dp += -a[k] * t[k] * d / (1.0 + (rz[k] + lam) / 2.0)
    if R > 0.0:
        T = t[n - 1]
        s_prev = 1.0
        ds_prev = 0.0
        j = 1
        while True:
            u = j * h
            last = u >= T
            if last:
                u = T
            z = bondlib._zero_at(cx, ccoef, czn, u)
            d0 = _disc(z, 0.0, u)
            dl = _disc(z, lam, u)
            s = dl / d0
            ds = -u * dl / (1.0 + (z + lam) / 2.0) / d0
            p += 100.0 * R * d0 * (s_prev - s)
            dp += 100.0 * R * d0 * (ds_prev - ds)
            s_prev = s
            ds_prev = ds
            if last:
                break
            j += 1
    return p, dp


@njit(cache=True)
def _price_i(i, ptr, tt, aa, rr, nflow, crow, cx, ccoef, czn, lam, R, h):
    o = ptr[i]
    return rfv(tt[o:o + nflow[i]], aa[o:o + nflow[i]], rr[o:o + nflow[i]], nflow[i], lam, R, h, cx, ccoef[crow[i]],
               czn[crow[i]])


@njit(cache=True)
def issuer_lam(items, R_item, ptr, tt, aa, rr, nflow, crow, cx, ccoef, czn, target, lam0, lam_max, h):
    """Gauss-Newton for the common lam of one issuer-day. Returns (lam, sse, at_bound)."""
    lam = lam0
    for it in range(100):
        num = 0.0
        den = 0.0
        for q in range(items.shape[0]):
            p, dp = _price_i(items[q], ptr, tt, aa, rr, nflow, crow, cx, ccoef, czn, lam, R_item[q], h)
            e = p - target[items[q]]
            num += e * dp
            den += dp * dp
        if den <= 0.0:
            break
        new = lam - num / den
        if new < 0.0:
            new = 0.0
        if new > lam_max:
            new = lam_max
        if abs(new - lam) < 1e-12:
            lam = new
            break
        lam = new
    sse = 0.0
    for q in range(items.shape[0]):
        p, dp = _price_i(items[q], ptr, tt, aa, rr, nflow, crow, cx, ccoef, czn, lam, R_item[q], h)
        sse += (p - target[items[q]]) ** 2
    return lam, sse, lam >= lam_max


@njit(parallel=True, cache=True)
def window_sse(iptr, members, grp, ptr, tt, aa, rr, nflow, crow, cx, ccoef, czn, target, R, lam_max, h, lam_out):
    ng = iptr.shape[0] - 1
    sse = np.zeros(ng)
    atb = np.zeros(ng)
    for g in prange(ng):
        items = members[iptr[g]:iptr[g + 1]]
        Ri = np.empty(items.shape[0])
        for q in range(items.shape[0]):
            Ri[q] = R[grp[items[q]]]
        # both ends of [0, lam_max] as starts, the lower SSE kept (lam_out is output only: no warm start)
        lam, s, b = issuer_lam(items, Ri, ptr, tt, aa, rr, nflow, crow, cx, ccoef, czn, target, 0.0, lam_max, h)
        lam2, s2, b2 = issuer_lam(items, Ri, ptr, tt, aa, rr, nflow, crow, cx, ccoef, czn, target, lam_max,
                                  lam_max, h)
        if s2 < s:
            lam, s, b = lam2, s2, b2
        lam_out[g] = lam
        sse[g] = s
        atb[g] = 1.0 if b else 0.0
    return sse.sum(), int(atb.sum())


@njit(parallel=True, cache=True)
def window_grad(iptr, members, grp, ptr, tt, aa, rr, nflow, crow, cx, ccoef, czn, target, R, lam, h):
    """d SSE / d R (3,) at the profiled lam of every issuer-day (lam = the lam_out of window_sse at the same R): by the
    envelope theorem the profiled lam's own response drops out, and the price is linear in R, so
    d SSE / d R_g = sum over bond-days of group g of 2 (P - target) (P(lam, 1) - P(lam, 0))."""
    ng = iptr.shape[0] - 1
    gg = np.zeros((ng, 3))
    for g in prange(ng):
        for q in range(iptr[g], iptr[g + 1]):
            i = members[q]
            p, _ = _price_i(i, ptr, tt, aa, rr, nflow, crow, cx, ccoef, czn, lam[g], R[grp[i]], h)
            p1, _ = _price_i(i, ptr, tt, aa, rr, nflow, crow, cx, ccoef, czn, lam[g], 1.0, h)
            p0, _ = _price_i(i, ptr, tt, aa, rr, nflow, crow, cx, ccoef, czn, lam[g], 0.0, h)
            gg[g, grp[i]] += 2.0 * (p - target[i]) * (p1 - p0)
    return gg.sum(axis=0)


@njit(cache=True)
def _build(e_arr, s_ord, s_ymd, crow, ptr_b, cf_ord, cf_ymd, cf_amt, iss_ymd, iss_ord, coupon, dc_eu, eom,
           cx, ccoef, czn, maxn):
    """cash flows (times, amounts, zero rates) and accrued of every bond-day."""
    nb = e_arr.shape[0]
    nflow = np.zeros(nb, np.int64)
    acc = np.full(nb, np.nan)
    tb = np.empty(maxn)
    ab = np.empty(maxn)
    for i in range(nb):
        n, ac = bondlib._flows(e_arr[i], s_ord[i], s_ymd[i, 0], s_ymd[i, 1], s_ymd[i, 2], ptr_b, cf_ord, cf_ymd,
                               cf_amt, iss_ymd, iss_ord, coupon, dc_eu, eom, tb, ab)
        nflow[i] = n
        acc[i] = ac
    ptr = np.zeros(nb + 1, np.int64)
    for i in range(nb):
        ptr[i + 1] = ptr[i] + nflow[i]
    tt = np.empty(ptr[nb])
    aa = np.empty(ptr[nb])
    rr = np.empty(ptr[nb])
    for i in range(nb):
        n, ac = bondlib._flows(e_arr[i], s_ord[i], s_ymd[i, 0], s_ymd[i, 1], s_ymd[i, 2], ptr_b, cf_ord, cf_ymd,
                               cf_amt, iss_ymd, iss_ord, coupon, dc_eu, eom, tb, ab)
        for k in range(n):
            tt[ptr[i] + k] = tb[k]
            aa[ptr[i] + k] = ab[k]
            rr[ptr[i] + k] = bondlib._zero_at(cx, ccoef[crow[i]], czn[crow[i]], tb[k])
    return ptr[:nb], tt, aa, rr, nflow, acc


def price(A, i, lam, R, h):
    return _price_i(i, A['ptr'], A['tt'], A['aa'], A['rr'], A['nflow'], A['crow'], A['cx'], A['ccoef'], A['czn'],
                    lam, R, h)


def window_index(day, A, t0, t1):
    """issuer-day groups (>= 2 valid bond-days) inside [t0, t1]: (iptr, members)."""
    w = day[(day['t'] >= t0) & (day['t'] <= t1) & A['valid']]
    n = w.groupby(['issuer', 't'])['b'].transform('size')
    w = w[n >= 2].sort_values(['issuer', 't'])
    keys = w[['issuer', 't']].to_numpy()
    starts = np.flatnonzero(np.r_[True, (keys[1:] != keys[:-1]).any(axis=1)])
    iptr = np.append(starts, len(w)).astype(np.int64)
    return iptr, w.index.to_numpy(np.int64)


def fit_window(A, iptr, members, h, lam_max):
    from scipy.optimize import minimize
    lam = np.zeros(len(iptr) - 1)

    def sse(R):
        return window_sse(iptr, members, A['grp'], A['ptr'], A['tt'], A['aa'], A['rr'], A['nflow'], A['crow'],
                          A['cx'], A['ccoef'], A['czn'], A['target'], np.asarray(R, float), lam_max, h, lam)

    grid = [(float(g), float(sse(np.full(3, g))[0])) for g in GRID]
    g0 = min(grid, key=lambda x: x[1])[0]
    logit = lambda r: np.log(np.clip(r, 1e-6, 1 - 1e-6) / (1 - np.clip(r, 1e-6, 1 - 1e-6)))

    def f(u):
        R = 1.0 / (1.0 + np.exp(-u))
        v = sse(R)[0]
        gr = window_grad(iptr, members, A['grp'], A['ptr'], A['tt'], A['aa'], A['rr'], A['nflow'], A['crow'],
                         A['cx'], A['ccoef'], A['czn'], A['target'], R, lam, h)
        return v, gr * R * (1.0 - R)

    res = minimize(f, logit(np.full(3, g0)), method='L-BFGS-B', jac=True)
    R = 1.0 / (1.0 + np.exp(-res.x))
    v, nb = sse(R)
    return dict(R=R.tolist(), sse=float(v), lam_at_bound=int(nb), grid=grid, nit=int(res.nit),
                message=str(res.message), lam_median=float(np.median(lam)), lam_p99=float(np.quantile(lam, 0.99)))
