"""callprice.py -- price of a callable bond under the call-intensity model, layer 1 (today's state frozen;
see README, 'Layer 1: frozen-state call price').

For bond b settling on day s (cash flows CF_k at t_k, 30/360 years from s; curve.py zero rates z(t), semiannual):
  discounting (Z-engine convention)        D_x(t) = (1 + (z(t) + x)/2)^(-2t)
  x = s_tot  for the bond's cash flows / call payment (G + u + beta C + B + L + lam_D D, frozen)
  x = s_cr   for the call incentive (credit only: G + u + beta C + B; decision c2)
  call grid: every business day v after max(s, first call date) up to maturity (decision 1: callable every
     business day after the first call date; decision 2: no call before it); step dv = 30/360 years
  call payment H(v) = call price in force at v (call_schedule.csv step-downs) + accrued at v (30/360, the bond's
     day count; the coupon of a coupon date counts as accrued on that day and is not paid separately)
  incentive M(v) = V_str(v) / H(v) - 1,  V_str(v) = sum_{t_k > v} CF_k D_cr(t_k) / D_cr(v)   (dirty, at v)
  intensity  lam(v) = exp(zeta0 + zeta1 M(v))                                              (decision 7: g = exp)
     itm_only (diagnostic variant, 2026-10-11, user's option 2 as a measurement): lam(v) = 0 when M(v) < 0, i.e. no
     call while the bond is worth less than the call payment; the price is nearly continuous at M = 0 because a
     call there exchanges the bond for about its value
  daily call probability q(v) = 1 - exp(-lam(v) dv); survival S(v) = prod_{u <= v} (1 - q(u))
  contract payoff (main line, decision 5):
     P = sum_k CF_k D_tot(t_k) S(t_k) + sum_v H(v) D_tot(v) S(v-) q(v)
  RMV (Jarrow et al. 2010, comparison K): bondholders keep a fraction of market value at the call, (1 - k) absorbed
     in zeta0:  P = sum_k CF_k D_tot(t_k) exp(-sum_{u <= t_k} lam(u) du)
Bonds without a call schedule (and before the first call date) have lam = 0, so P is the Z-engine price at s_tot.
The model Z-spread is Z(P): the Z engine applied to P (bonds._solve), so the observation stays in Z space.

Business days of the grid: FRED curve days up to the last FRED day, then weekdays without US federal holidays
(pandas USFederalHolidayCalendar) -- the Z engine's calendar extended into the future.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

import bonds as bondlib

HERE = Path(__file__).parent


def business_days(fred_index, last='2060-12-31'):
    hist = pd.DatetimeIndex(fred_index)
    from pandas.tseries.holiday import USFederalHolidayCalendar
    fut = pd.bdate_range(hist[-1] + pd.Timedelta(days=1), last,
                         freq=pd.offsets.CustomBusinessDay(calendar=USFederalHolidayCalendar()))
    d = hist.append(fut)
    ymd = np.stack([d.year, d.month, d.day], axis=1).astype(np.int64)
    ordn = ((d - pd.Timestamp('1970-01-01')).days).to_numpy(dtype=np.int64)
    return ordn, ymd


def call_tables(cusips, path=HERE / 'data' / 'bloomberg' / 'call_schedule.csv'):
    """Per bond (aligned to cusips): cptr (B+1,), c_ord (call step dates, ordinal), c_px (call price per 100)."""
    cs = pd.read_csv(path, parse_dates=['call_date']).sort_values(['cusip', 'call_date'])
    g = {c: d for c, d in cs.groupby('cusip')}
    cptr = [0]
    c_ord, c_px = [], []
    for c in np.asarray(cusips).astype(str):
        d = g.get(c)
        if d is not None:
            c_ord.extend(((d['call_date'] - pd.Timestamp('1970-01-01')).dt.days).tolist())
            c_px.extend(d['call_price'].tolist())
        cptr.append(len(c_ord))
    return np.array(cptr, np.int64), np.array(c_ord, np.int64), np.array(c_px, float)


@njit(cache=True)
def _disc(z, x, t):
    g = 1.0 + (z + x) / 2.0
    if g <= 0.0:
        return np.nan
    return g ** (-2.0 * t)


@njit(cache=True)
def price(b, s_ord, sy, sm, sd, ptr, cf_ord, cf_ymd, cf_amt, iss_ymd, iss_ord, coupon, dc_eu, eom,
          x, coef, zn, cptr, c_ord, c_px, bd_ord, bd_ymd, s_tot, s_cr, zeta0, zeta1, rmv, t_buf, a_buf, r_buf,
          suf_buf, itm_only=False):
    """Dirty price per 100 (and accrued at s) under layer 1. s_tot, s_cr in decimal. NaN if no cash flows."""
    n, acc = bondlib._flows(b, s_ord, sy, sm, sd, ptr, cf_ord, cf_ymd, cf_amt, iss_ymd, iss_ord, coupon, dc_eu,
                            eom, t_buf, a_buf)
    if n == 0:
        return np.nan, np.nan
    for k in range(n):
        r_buf[k] = bondlib._zero_at(x, coef, zn, t_buf[k])
    c0, c1 = cptr[b], cptr[b + 1]
    g0 = ptr[b]
    mat_ord = cf_ord[ptr[b + 1] - 1]
    if c1 == c0 or c_ord[c0] >= mat_ord:
        pv = 0.0
        for k in range(n):
            pv += a_buf[k] * _disc(r_buf[k], s_tot, t_buf[k])
        return pv, acc
    # suffix sums of the credit-only discounted cash flows (for V_str)
    suf_buf[n] = 0.0
    for k in range(n - 1, -1, -1):
        suf_buf[k] = suf_buf[k + 1] + a_buf[k] * _disc(r_buf[k], s_cr, t_buf[k])
    # first grid day: the first business day strictly after s and on/after the first call date
    start = max(s_ord + 1, c_ord[c0])
    lo, hi = 0, bd_ord.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if bd_ord[mid] < start:
            lo = mid + 1
        else:
            hi = mid
    i = lo
    # index of the first remaining cash flow in the bond's table
    j0 = g0
    while j0 < ptr[b + 1] and cf_ord[j0] <= s_ord:
        j0 += 1
    kcf = 0              # next remaining cash flow (local index) not yet passed by the grid
    surv = 1.0
    cum_lam = 0.0
    pv = 0.0
    t_prev = 0.0
    # cash flows before the first grid day are paid with survival 1
    while kcf < n and cf_ord[j0 + kcf] < start:
        pv += a_buf[kcf] * _disc(r_buf[kcf], s_tot, t_buf[kcf])
        kcf += 1
    cidx = c0
    first = True
    while i < bd_ord.shape[0] and bd_ord[i] <= mat_ord:
        v = bd_ord[i]
        tv = bondlib._dc(sy, sm, sd, bd_ymd[i, 0], bd_ymd[i, 1], bd_ymd[i, 2], dc_eu[b], eom[b]) / 360.0
        if first:
            # the grid starts on the first call date (or the day after settlement); the step covers the time
            # since the previous business day
            tpv = bondlib._dc(sy, sm, sd, bd_ymd[i - 1, 0], bd_ymd[i - 1, 1], bd_ymd[i - 1, 2], dc_eu[b],
                              eom[b]) / 360.0 if i > 0 else 0.0
            t_prev = max(tpv, 0.0)
            first = False
        dv = tv - t_prev
        t_prev = tv
        while cidx + 1 < c1 and c_ord[cidx + 1] <= v:
            cidx += 1
        # accrued at v: since the last coupon date strictly before v (or the accrual start)
        jj = j0 + kcf - 1
        if jj >= g0:
            ac = coupon[b] * bondlib._dc(cf_ymd[jj, 0], cf_ymd[jj, 1], cf_ymd[jj, 2], bd_ymd[i, 0], bd_ymd[i, 1],
                                         bd_ymd[i, 2], dc_eu[b], eom[b]) / 360.0
        else:
            ac = coupon[b] * bondlib._dc(iss_ymd[b, 0], iss_ymd[b, 1], iss_ymd[b, 2], bd_ymd[i, 0], bd_ymd[i, 1],
                                         bd_ymd[i, 2], dc_eu[b], eom[b]) / 360.0
        H = c_px[cidx] + ac
        # value at v (dirty) of the cash flows after v, credit-only discounting
        kk = kcf
        while kk < n and cf_ord[j0 + kk] <= v:
            kk += 1
        zv = bondlib._zero_at(x, coef, zn, tv)
        dcr = _disc(zv, s_cr, tv)
        vstr = suf_buf[kk] / dcr if kk < n else 0.0
        lam = math.exp(zeta0 + zeta1 * (vstr / H - 1.0))
        if itm_only and vstr < H:
            lam = 0.0
        if rmv:
            cum_lam += lam * dv
        else:
            q = 1.0 - math.exp(-lam * dv)
            pv += H * _disc(zv, s_tot, tv) * surv * q
            surv *= 1.0 - q
        # cash flows on day v (coupon dates) are received if not called on or before v
        while kcf < n and cf_ord[j0 + kcf] <= v:
            if rmv:
                pv += a_buf[kcf] * _disc(r_buf[kcf], s_tot, t_buf[kcf]) * math.exp(-cum_lam)
            else:
                pv += a_buf[kcf] * _disc(r_buf[kcf], s_tot, t_buf[kcf]) * surv
            kcf += 1
        i += 1
    while kcf < n:          # maturity falls on a non-business day after the last grid day
        f = math.exp(-cum_lam) if rmv else surv
        pv += a_buf[kcf] * _disc(r_buf[kcf], s_tot, t_buf[kcf]) * f
        kcf += 1
    return pv, acc


@njit(cache=True)
def model_z(b, s_ord, sy, sm, sd, ptr, cf_ord, cf_ymd, cf_amt, iss_ymd, iss_ord, coupon, dc_eu, eom,
            x, coef, zn, cptr, c_ord, c_px, bd_ord, bd_ymd, s_tot, s_cr, zeta0, zeta1, rmv, t_buf, a_buf, r_buf,
            suf_buf, itm_only=False):
    """Z-spread (decimal) of the layer-1 model price (Z engine to maturity)."""
    pv, acc = price(b, s_ord, sy, sm, sd, ptr, cf_ord, cf_ymd, cf_amt, iss_ymd, iss_ord, coupon, dc_eu, eom,
                    x, coef, zn, cptr, c_ord, c_px, bd_ord, bd_ymd, s_tot, s_cr, zeta0, zeta1, rmv, t_buf, a_buf,
                    r_buf, suf_buf, itm_only)
    if not np.isfinite(pv):
        return np.nan
    n, _a = bondlib._flows(b, s_ord, sy, sm, sd, ptr, cf_ord, cf_ymd, cf_amt, iss_ymd, iss_ord, coupon, dc_eu,
                           eom, t_buf, a_buf)
    for k in range(n):
        r_buf[k] = bondlib._zero_at(x, coef, zn, t_buf[k])
    return bondlib._solve(r_buf, t_buf, a_buf, n, pv, 2.0)


@njit(cache=True)
def _disc_d(z, x, t):
    """D_x(t) and dD/dx."""
    g = 1.0 + (z + x) / 2.0
    if g <= 0.0:
        return np.nan, np.nan
    d = g ** (-2.0 * t)
    return d, -t * d / g


@njit(cache=True)
def price_d(b, s_ord, sy, sm, sd, ptr, cf_ord, cf_ymd, cf_amt, iss_ymd, iss_ord, coupon, dc_eu, eom,
            x, coef, zn, cptr, c_ord, c_px, bd_ord, bd_ymd, s_tot, s_cr, zeta0, zeta1, rmv, t_buf, a_buf, r_buf,
            suf_buf, itm_only=False):
    """price() with its exact derivatives (forward mode through the same loop): returns (dirty, accrued,
    dP/ds_tot, dP/ds_cr). suf_buf needs 2 (n + 1) entries (suffix sums and their s_cr derivatives)."""
    n, acc = bondlib._flows(b, s_ord, sy, sm, sd, ptr, cf_ord, cf_ymd, cf_amt, iss_ymd, iss_ord, coupon, dc_eu,
                            eom, t_buf, a_buf)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    for k in range(n):
        r_buf[k] = bondlib._zero_at(x, coef, zn, t_buf[k])
    c0, c1 = cptr[b], cptr[b + 1]
    g0 = ptr[b]
    mat_ord = cf_ord[ptr[b + 1] - 1]
    pv = 0.0
    dpt = 0.0
    dpc = 0.0
    if c1 == c0 or c_ord[c0] >= mat_ord:
        for k in range(n):
            d, dd = _disc_d(r_buf[k], s_tot, t_buf[k])
            pv += a_buf[k] * d
            dpt += a_buf[k] * dd
        return pv, acc, dpt, 0.0
    o = n + 1                                   # dsuf lives in suf_buf[o:o + n + 1]
    suf_buf[n] = 0.0
    suf_buf[o + n] = 0.0
    for k in range(n - 1, -1, -1):
        d, dd = _disc_d(r_buf[k], s_cr, t_buf[k])
        suf_buf[k] = suf_buf[k + 1] + a_buf[k] * d
        suf_buf[o + k] = suf_buf[o + k + 1] + a_buf[k] * dd
    start = max(s_ord + 1, c_ord[c0])
    lo, hi = 0, bd_ord.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if bd_ord[mid] < start:
            lo = mid + 1
        else:
            hi = mid
    i = lo
    j0 = g0
    while j0 < ptr[b + 1] and cf_ord[j0] <= s_ord:
        j0 += 1
    kcf = 0
    surv = 1.0
    dsurv = 0.0                                 # d surv / d s_cr
    cum_lam = 0.0
    dcum = 0.0                                  # d cum_lam / d s_cr
    t_prev = 0.0
    while kcf < n and cf_ord[j0 + kcf] < start:
        d, dd = _disc_d(r_buf[kcf], s_tot, t_buf[kcf])
        pv += a_buf[kcf] * d
        dpt += a_buf[kcf] * dd
        kcf += 1
    cidx = c0
    first = True
    while i < bd_ord.shape[0] and bd_ord[i] <= mat_ord:
        v = bd_ord[i]
        tv = bondlib._dc(sy, sm, sd, bd_ymd[i, 0], bd_ymd[i, 1], bd_ymd[i, 2], dc_eu[b], eom[b]) / 360.0
        if first:
            tpv = bondlib._dc(sy, sm, sd, bd_ymd[i - 1, 0], bd_ymd[i - 1, 1], bd_ymd[i - 1, 2], dc_eu[b],
                              eom[b]) / 360.0 if i > 0 else 0.0
            t_prev = max(tpv, 0.0)
            first = False
        dv = tv - t_prev
        t_prev = tv
        while cidx + 1 < c1 and c_ord[cidx + 1] <= v:
            cidx += 1
        jj = j0 + kcf - 1
        if jj >= g0:
            ac = coupon[b] * bondlib._dc(cf_ymd[jj, 0], cf_ymd[jj, 1], cf_ymd[jj, 2], bd_ymd[i, 0], bd_ymd[i, 1],
                                         bd_ymd[i, 2], dc_eu[b], eom[b]) / 360.0
        else:
            ac = coupon[b] * bondlib._dc(iss_ymd[b, 0], iss_ymd[b, 1], iss_ymd[b, 2], bd_ymd[i, 0], bd_ymd[i, 1],
                                         bd_ymd[i, 2], dc_eu[b], eom[b]) / 360.0
        H = c_px[cidx] + ac
        kk = kcf
        while kk < n and cf_ord[j0 + kk] <= v:
            kk += 1
        zv = bondlib._zero_at(x, coef, zn, tv)
        dcr, ddcr = _disc_d(zv, s_cr, tv)
        if kk < n:
            vstr = suf_buf[kk] / dcr
            dvstr = (suf_buf[o + kk] * dcr - suf_buf[kk] * ddcr) / (dcr * dcr)
        else:
            vstr = 0.0
            dvstr = 0.0
        lam = math.exp(zeta0 + zeta1 * (vstr / H - 1.0))
        dlam = lam * zeta1 * dvstr / H
        if itm_only and vstr < H:
            lam = 0.0
            dlam = 0.0
        if rmv:
            cum_lam += lam * dv
            dcum += dlam * dv
        else:
            e = math.exp(-lam * dv)
            q = 1.0 - e
            dq = e * dv * dlam
            dtv, ddtv = _disc_d(zv, s_tot, tv)
            pv += H * dtv * surv * q
            dpt += H * ddtv * surv * q
            dpc += H * dtv * (dsurv * q + surv * dq)
            dsurv = dsurv * (1.0 - q) - surv * dq
            surv *= 1.0 - q
        while kcf < n and cf_ord[j0 + kcf] <= v:
            d, dd = _disc_d(r_buf[kcf], s_tot, t_buf[kcf])
            if rmv:
                f = math.exp(-cum_lam)
                pv += a_buf[kcf] * d * f
                dpt += a_buf[kcf] * dd * f
                dpc -= a_buf[kcf] * d * f * dcum
            else:
                pv += a_buf[kcf] * d * surv
                dpt += a_buf[kcf] * dd * surv
                dpc += a_buf[kcf] * d * dsurv
            kcf += 1
        i += 1
    while kcf < n:
        d, dd = _disc_d(r_buf[kcf], s_tot, t_buf[kcf])
        if rmv:
            f = math.exp(-cum_lam)
            pv += a_buf[kcf] * d * f
            dpt += a_buf[kcf] * dd * f
            dpc -= a_buf[kcf] * d * f * dcum
        else:
            pv += a_buf[kcf] * d * surv
            dpt += a_buf[kcf] * dd * surv
            dpc += a_buf[kcf] * d * dsurv
        kcf += 1
    return pv, acc, dpt, dpc


@njit(cache=True)
def model_z_d(b, s_ord, sy, sm, sd, ptr, cf_ord, cf_ymd, cf_amt, iss_ymd, iss_ord, coupon, dc_eu, eom,
              x, coef, zn, cptr, c_ord, c_px, bd_ord, bd_ymd, s_tot, s_cr, zeta0, zeta1, rmv, t_buf, a_buf, r_buf,
              suf_buf, itm_only=False):
    """Model Z-spread (decimal) and its exact slopes dZ/ds_tot, dZ/ds_cr (implicit function: dZ/ds = (dP/ds) /
    (dPV_Z/dZ)). Z = bonds._solve followed by one Newton step, so Z is accurate to rounding and smooth in the
    inputs (numerical device: _solve alone stops at a 1e-12 relative price error)."""
    pv, acc, dpt, dpc = price_d(b, s_ord, sy, sm, sd, ptr, cf_ord, cf_ymd, cf_amt, iss_ymd, iss_ord, coupon, dc_eu,
                                eom, x, coef, zn, cptr, c_ord, c_px, bd_ord, bd_ymd, s_tot, s_cr, zeta0, zeta1, rmv,
                                t_buf, a_buf, r_buf, suf_buf, itm_only)
    if not np.isfinite(pv):
        return np.nan, np.nan, np.nan
    n, _a = bondlib._flows(b, s_ord, sy, sm, sd, ptr, cf_ord, cf_ymd, cf_amt, iss_ymd, iss_ord, coupon, dc_eu,
                           eom, t_buf, a_buf)
    for k in range(n):
        r_buf[k] = bondlib._zero_at(x, coef, zn, t_buf[k])
    z = bondlib._solve(r_buf, t_buf, a_buf, n, pv, 2.0)
    p, d = bondlib._pv(r_buf, t_buf, a_buf, n, z, 2.0)
    if not (np.isfinite(p) and d < 0.0):
        return np.nan, np.nan, np.nan
    z = z - (p - pv) / d
    p, d = bondlib._pv(r_buf, t_buf, a_buf, n, z, 2.0)
    return z, dpt / d, dpc / d


@njit(cache=True)
def model_obs_d(b, s_ord, sy, sm, sd, ptr, cf_ord, cf_ymd, cf_amt, iss_ymd, iss_ord, coupon, dc_eu, eom,
                x, coef, zn, cptr, c_ord, c_px, bd_ord, bd_ymd, s_tot, s_cr, zeta0, zeta1, rmv, t_buf, a_buf, r_buf,
                suf_buf, itm_only, obs_price, no_call):
    """The model value of an observation and its slopes in s_tot, s_cr (decimal): obs_price -> clean price per 100
    (price-space variant A, 2026-10-11), else the Z-spread (decimal, model_z_d). no_call -> the straight bond (Z
    observation: Z = s_tot exactly)."""
    if no_call:
        if not obs_price:
            return s_tot, 1.0, 0.0
        n, acc = bondlib._flows(b, s_ord, sy, sm, sd, ptr, cf_ord, cf_ymd, cf_amt, iss_ymd, iss_ord, coupon, dc_eu,
                                eom, t_buf, a_buf)
        if n == 0:
            return np.nan, np.nan, np.nan
        for k in range(n):
            r_buf[k] = bondlib._zero_at(x, coef, zn, t_buf[k])
        pv, d = bondlib._pv(r_buf, t_buf, a_buf, n, s_tot, 2.0)
        if not np.isfinite(pv):
            return np.nan, np.nan, np.nan
        return pv - acc, d, 0.0
    if obs_price:
        pv, acc, dpt, dpc = price_d(b, s_ord, sy, sm, sd, ptr, cf_ord, cf_ymd, cf_amt, iss_ymd, iss_ord, coupon,
                                    dc_eu, eom, x, coef, zn, cptr, c_ord, c_px, bd_ord, bd_ymd, s_tot, s_cr, zeta0,
                                    zeta1, rmv, t_buf, a_buf, r_buf, suf_buf, itm_only)
        return pv - acc, dpt, dpc
    return model_z_d(b, s_ord, sy, sm, sd, ptr, cf_ord, cf_ymd, cf_amt, iss_ymd, iss_ord, coupon, dc_eu, eom,
                     x, coef, zn, cptr, c_ord, c_px, bd_ord, bd_ymd, s_tot, s_cr, zeta0, zeta1, rmv, t_buf, a_buf,
                     r_buf, suf_buf, itm_only)


class Engine:
    """Python holder of the static arrays (bond terms, curves, call schedules, calendar)."""

    def __init__(self, pan_cusips=None):
        import curve
        terms = bondlib.load_terms(HERE / 'data' / 'bloomberg' / 'bloomberg_fields.csv')
        self.B = bondlib.build_bonds(terms)
        fred = curve.load_fred(HERE / 'data' / 'fred')
        self.zc = curve.build(fred, 'pchip')
        self.bd_ord, self.bd_ymd = business_days(fred.index)
        cus = self.B.terms['cusip'].astype(str).to_numpy()
        self.cptr, self.c_ord, self.c_px = call_tables(cus)
        self.maxn = int(np.max(np.diff(self.B.ptr)))
        if pan_cusips is not None:
            self.cus_idx = self.B.index(pan_cusips).astype(np.int64)

    def bufs(self):
        n = self.maxn + 1
        return np.empty(n), np.empty(n), np.empty(n), np.empty(2 * (n + 1))      # suf: 2 blocks for price_d

    def z(self, b, settle, crow, s_tot, s_cr, zeta0, zeta1, rmv=False):
        B = self.B
        d = pd.Timestamp(settle)
        s_ord = int((d - pd.Timestamp('1970-01-01')).days)
        tb, ab, rb, sb = self.bufs()
        return model_z(b, s_ord, d.year, d.month, d.day, B.ptr, B.cf_ord, B.cf_ymd, B.cf_amt, B.iss_ymd,
                       B.iss_ord, B.coupon, B.dc_eu, B.eom, self.zc.x, self.zc.coef[crow], self.zc.z_nodes[crow],
                       self.cptr, self.c_ord, self.c_px, self.bd_ord, self.bd_ymd, s_tot, s_cr, zeta0, zeta1, rmv,
                       tb, ab, rb, sb)

    def price(self, b, settle, crow, s_tot, s_cr, zeta0, zeta1, rmv=False):
        B = self.B
        d = pd.Timestamp(settle)
        s_ord = int((d - pd.Timestamp('1970-01-01')).days)
        tb, ab, rb, sb = self.bufs()
        return price(b, s_ord, d.year, d.month, d.day, B.ptr, B.cf_ord, B.cf_ymd, B.cf_amt, B.iss_ymd,
                     B.iss_ord, B.coupon, B.dc_eu, B.eom, self.zc.x, self.zc.coef[crow], self.zc.z_nodes[crow],
                     self.cptr, self.c_ord, self.c_px, self.bd_ord, self.bd_ymd, s_tot, s_cr, zeta0, zeta1, rmv,
                     tb, ab, rb, sb)
