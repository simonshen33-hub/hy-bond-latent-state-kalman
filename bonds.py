"""bonds.py -- contractual cash flows, accrued interest, yield to maturity and Z-spread.

Conventions (see README, "Bond math"):
  * Z-spread to maturity, no call/option adjustment: dirty = sum_k CF_k (1 + (z(t_k) + Z)/2)^(-2 t_k),
    z(.) = Treasury zero curve of the trade date (curve.py), t_k in years from settlement.
  * Street-convention yield (for the Bloomberg check): dirty = sum_k CF_k (1 + y/f)^(-f t_k).
  * Day count 30/360 for accrual and for t_k = DC(settle, date_k) / 360:
      '30/360', '30/360 NON-EOM'               -> US bond basis (SIA), EOM Feb rule only for EOM bonds
      'ISMA-30/360', 'ISMA-30/360 NONEOM'      -> 30E/360
  * Coupon dates: FIRST_CPN_DT, then every 12/f months on FIRST_CPN_DT's day of month (month end
    for EOM bonds whose first coupon is a month end), until maturity; 42 of the 1,236 supported bonds
    mature off that schedule (odd last period). Odd first coupon accrues from issue_dt; the last
    period pays coupon x DC(prev, maturity)/360. Unadjusted dates.
  * Accrued = coupon x DC(previous coupon date or issue_dt, settle)/360 (0 before issue_dt).
    Settlement on a coupon date: that coupon goes to the seller.
  * Settlement = T+2 business days before 2024-05-28, T+1 from 2024-05-28 (SEC rule); business days =
    days with a FRED Treasury curve.
Supported bonds: fixed coupon, bullet or plain callable (MTY_TYP 'CALLABLE' / 'AT MATURITY'), street
convention. Others (floating, PIK, defaulted, perpetual, sinking fund, step-up) get a status code and
NaN results -- their cash flows cannot be generated from the fields we hold.

The root solvers only target numerical precision (relative price error ~1e-12); they are not model
settings.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit, prange

SUPPORTED_CALC = {'STREET CONVENTION', 'STREET CONV FIXED'}
SUPPORTED_MTY = {'CALLABLE', 'AT MATURITY'}
DAYCOUNT = {'30/360': (0, True), '30/360 NON-EOM': (0, False),
            'ISMA-30/360': (1, True), 'ISMA-30/360 NONEOM': (1, False)}
T1_START = pd.Timestamp('2024-05-28')


# ----------------------------------------------------------------------------------------------
# terms and schedules
# ----------------------------------------------------------------------------------------------
def load_terms(fields_csv):
    """Bloomberg bloomberg_fields.csv -> one row per bond with parsed terms and a status code."""
    t = pd.read_csv(fields_csv, low_memory=False)
    out = pd.DataFrame({'cusip': t['cusip'].astype(str)})
    out['coupon'] = pd.to_numeric(t['coupon'], errors='coerce')
    out['freq'] = pd.to_numeric(t['CPN_FREQ'], errors='coerce')
    for c, src in (('issue', 'issue_dt'), ('maturity', 'maturity'), ('first_cpn', 'FIRST_CPN_DT')):
        out[c] = pd.to_datetime(t[src], errors='coerce')
    out['daycount'] = t['DAY_CNT_DES'].astype(str)
    out['cpn_typ'] = t['CPN_TYP'].astype(str)
    out['mty_typ'] = t['MTY_TYP'].astype(str)
    out['calc_typ'] = t['CALC_TYP_DES'].astype(str)
    status = np.full(len(out), 'ok', dtype=object)
    status[out['cpn_typ'] != 'FIXED'] = 'coupon_' + out['cpn_typ'][out['cpn_typ'] != 'FIXED']
    bad = (status == 'ok') & ~out['mty_typ'].isin(SUPPORTED_MTY)
    status[bad] = 'maturity_' + out['mty_typ'][bad]
    bad = (status == 'ok') & ~out['calc_typ'].isin(SUPPORTED_CALC)
    status[bad] = 'calc_' + out['calc_typ'][bad]
    bad = (status == 'ok') & ~out['daycount'].isin(DAYCOUNT)
    status[bad] = 'daycount_' + out['daycount'][bad]
    bad = (status == 'ok') & (out[['coupon', 'freq', 'issue', 'maturity', 'first_cpn']].isna().any(axis=1))
    status[bad] = 'missing_terms'
    out['status'] = status
    return out


def _month_end(y, m):
    return pd.Timestamp(year=y, month=m, day=1) + pd.offsets.MonthEnd(0)


def coupon_dates(first_cpn, maturity, freq, eom):
    """Unadjusted coupon dates: first_cpn, then every 12/f months on first_cpn's day of month (month
    end for EOM bonds whose first coupon falls on a month end), then maturity (an odd last period if
    maturity is off that schedule)."""
    step = 12 // int(freq)
    use_eom = eom and first_cpn == _month_end(first_cpn.year, first_cpn.month)
    dates = [first_cpn]
    k = 1
    while True:
        mm = first_cpn.month - 1 + k * step
        y, m = first_cpn.year + mm // 12, mm % 12 + 1
        last = _month_end(y, m)
        d = last if use_eom else pd.Timestamp(year=y, month=m, day=min(first_cpn.day, last.day))
        if d >= maturity:
            break
        dates.append(d)
        k += 1
    if dates[-1] != maturity:
        dates.append(maturity)
    return dates


@dataclass
class Bonds:
    terms: pd.DataFrame          # load_terms output, row b = bond b
    ptr: np.ndarray              # (n_bonds+1,) cash-flow rows of bond b: ptr[b]..ptr[b+1]-1
    cf_ord: np.ndarray           # day ordinal (days since 1970-01-01) of each cash-flow date
    cf_ymd: np.ndarray           # (n_cf, 3) int
    cf_amt: np.ndarray           # coupon (+100 principal at maturity) per 100 face
    iss_ymd: np.ndarray          # (n_bonds, 3) accrual start of the first coupon
    iss_ord: np.ndarray
    coupon: np.ndarray           # annual coupon, % of face
    freq: np.ndarray
    dc_eu: np.ndarray            # bool: 30E/360
    eom: np.ndarray              # bool
    ok: np.ndarray               # bool: supported

    def index(self, cusips):
        pos = pd.Series(np.arange(len(self.terms)), index=self.terms['cusip'])
        return pos.reindex(np.asarray(cusips, dtype=str)).to_numpy()


def _ord(ts):
    return int((pd.Timestamp(ts) - pd.Timestamp('1970-01-01')).days)


def build_bonds(terms):
    n = len(terms)
    ptr = np.zeros(n + 1, dtype=np.int64)
    ords, ymd, amt = [], [], []
    iss_ymd = np.zeros((n, 3), dtype=np.int64)
    iss_ord = np.zeros(n, dtype=np.int64)
    dc_eu = np.zeros(n, dtype=bool)
    eom = np.zeros(n, dtype=bool)
    ok = (terms['status'] == 'ok').to_numpy()
    for b, r in enumerate(terms.itertuples(index=False)):
        if ok[b]:
            dc_eu[b], eom[b] = DAYCOUNT[r.daycount][0] == 1, DAYCOUNT[r.daycount][1]
            dates = coupon_dates(r.first_cpn, r.maturity, r.freq, eom[b])
            iss_ymd[b] = (r.issue.year, r.issue.month, r.issue.day)
            iss_ord[b] = _ord(r.issue)
            prev = (r.issue.year, r.issue.month, r.issue.day)
            for i, d in enumerate(dates):
                cur = (d.year, d.month, d.day)
                frac = _dc(prev[0], prev[1], prev[2], cur[0], cur[1], cur[2], dc_eu[b], eom[b]) / 360.0
                regular = 0 < i < len(dates) - 1
                a = r.coupon / r.freq if regular else r.coupon * frac
                if i == len(dates) - 1:
                    a += 100.0
                ords.append(_ord(d))
                ymd.append(cur)
                amt.append(a)
                prev = cur
        ptr[b + 1] = len(ords)
    return Bonds(terms=terms.reset_index(drop=True), ptr=ptr, cf_ord=np.array(ords, dtype=np.int64),
                 cf_ymd=np.array(ymd, dtype=np.int64).reshape(-1, 3), cf_amt=np.array(amt, dtype=float),
                 iss_ymd=iss_ymd, iss_ord=iss_ord, coupon=terms['coupon'].to_numpy(dtype=float),
                 freq=terms['freq'].fillna(0).to_numpy(dtype=float), dc_eu=dc_eu, eom=eom, ok=ok)


# ----------------------------------------------------------------------------------------------
# settlement
# ----------------------------------------------------------------------------------------------
def settlement(trade_dates, business_days):
    """T+2 business days before 2024-05-28, T+1 from then (business days = curve days)."""
    td = np.asarray(pd.to_datetime(trade_dates), dtype='datetime64[ns]')
    bd = np.asarray(pd.to_datetime(business_days), dtype='datetime64[ns]')
    lag = np.where(td < np.datetime64(T1_START), 2, 1)
    i = np.searchsorted(bd, td, side='right') - 1 + lag     # k-th business day strictly after the trade
    if np.any(i >= bd.size):
        raise ValueError('settlement beyond the business-day calendar')
    return pd.DatetimeIndex(bd[i])


def _ymd_ord(dates):
    d = pd.DatetimeIndex(dates)
    ymd = np.stack([d.year, d.month, d.day], axis=1).astype(np.int64)
    ordn = ((d - pd.Timestamp('1970-01-01')).days).to_numpy(dtype=np.int64)
    return ymd, ordn


# ----------------------------------------------------------------------------------------------
# numba kernels
# ----------------------------------------------------------------------------------------------
@njit(cache=True)
def _is_leap(y):
    return (y % 4 == 0 and y % 100 != 0) or y % 400 == 0


@njit(cache=True)
def _last_feb(y, m, d):
    return m == 2 and d == (29 if _is_leap(y) else 28)


@njit(cache=True)
def _dc(y1, m1, d1, y2, m2, d2, eu, eom):
    """30/360 day count: 30E/360 if eu, else US bond basis (SIA) with the Feb rule for EOM bonds."""
    if eu:
        d1 = min(d1, 30)
        d2 = min(d2, 30)
    else:
        if eom and _last_feb(y1, m1, d1) and _last_feb(y2, m2, d2):
            d2 = 30
        if eom and _last_feb(y1, m1, d1):
            d1 = 30
        if d2 == 31 and d1 >= 30:
            d2 = 30
        if d1 == 31:
            d1 = 30
    return 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)


@njit(cache=True)
def _zero_at(x, coef, zn, t):
    if t <= x[0]:
        return zn[0]
    if t >= x[-1]:
        return zn[-1]
    lo, hi = 0, x.size - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x[mid] <= t:
            lo = mid
        else:
            hi = mid
    h = t - x[lo]
    return ((coef[0, lo] * h + coef[1, lo]) * h + coef[2, lo]) * h + coef[3, lo]


@njit(cache=True)
def _flows(b, s_ord, sy, sm, sd, ptr, cf_ord, cf_ymd, cf_amt, iss_ymd, iss_ord, coupon, dc_eu, eom,
           t_out, a_out):
    """Remaining cash flows after settlement s: fills t_out (years) / a_out, returns (n, accrued)."""
    g0, g1 = ptr[b], ptr[b + 1]
    j = g0
    while j < g1 and cf_ord[j] <= s_ord:
        j += 1
    if j == g1:
        return 0, np.nan
    if j > g0:
        py, pm, pd_ = cf_ymd[j - 1, 0], cf_ymd[j - 1, 1], cf_ymd[j - 1, 2]
        acc = coupon[b] * _dc(py, pm, pd_, sy, sm, sd, dc_eu[b], eom[b]) / 360.0
    elif s_ord > iss_ord[b]:
        acc = coupon[b] * _dc(iss_ymd[b, 0], iss_ymd[b, 1], iss_ymd[b, 2], sy, sm, sd, dc_eu[b], eom[b]) / 360.0
    else:
        acc = 0.0
    n = 0
    for k in range(j, g1):
        t_out[n] = _dc(sy, sm, sd, cf_ymd[k, 0], cf_ymd[k, 1], cf_ymd[k, 2], dc_eu[b], eom[b]) / 360.0
        a_out[n] = cf_amt[k]
        n += 1
    return n, acc


@njit(cache=True)
def _pv(r_base, t, a, n, x, comp):
    """PV and dPV/dx of sum a_k (1 + (r_base_k + x)/comp)^(-comp t_k). Returns (pv, d) or (inf, 0)."""
    pv = 0.0
    dv = 0.0
    for k in range(n):
        g = 1.0 + (r_base[k] + x) / comp
        if g <= 0.0:
            return np.inf, 0.0
        v = a[k] * g ** (-comp * t[k])
        pv += v
        dv += -t[k] * v / g
    return pv, dv


@njit(cache=True)
def _solve(r_base, t, a, n, target, comp):
    """x with PV(x) = target (PV decreasing in x): bracket + Newton with bisection fallback."""
    lo = -comp
    for k in range(n):
        lo = max(lo, -comp - r_base[k])
    lo = lo + 1e-9
    hi = 1.0
    for _ in range(200):
        p, _d = _pv(r_base, t, a, n, hi, comp)
        if p < target:
            break
        hi *= 2.0
    x = 0.5 * (lo + hi) if not (lo < 0.0 < hi) else 0.0
    for _ in range(200):
        p, d = _pv(r_base, t, a, n, x, comp)
        f = p - target
        if abs(f) <= 1e-12 * target:
            return x
        if f > 0:
            lo = x
        else:
            hi = x
        nx = x - f / d if d != 0.0 else 0.5 * (lo + hi)
        if not (lo < nx < hi):
            nx = 0.5 * (lo + hi)
        if nx == x:
            return x
        x = nx
    return x


@njit(parallel=True, cache=True)
def _kernel(mode, bidx, s_ord, s_ymd, value, crow,
            ptr, cf_ord, cf_ymd, cf_amt, iss_ymd, iss_ord, coupon, freq, dc_eu, eom, ok,
            x, coef, zn, maxn):
    """mode 0: clean -> Z;  1: Z -> clean;  2: clean -> yield;  3: yield -> clean.
    Returns (result, accrued)."""
    n_tr = bidx.shape[0]
    res = np.full(n_tr, np.nan)
    acc_out = np.full(n_tr, np.nan)
    for i in prange(n_tr):
        b = bidx[i]
        if b < 0 or not ok[b] or not np.isfinite(value[i]):
            continue
        t = np.empty(maxn)
        a = np.empty(maxn)
        n, acc = _flows(b, s_ord[i], s_ymd[i, 0], s_ymd[i, 1], s_ymd[i, 2], ptr, cf_ord, cf_ymd, cf_amt,
                        iss_ymd, iss_ord, coupon, dc_eu, eom, t, a)
        if n == 0:
            continue
        acc_out[i] = acc
        r = np.empty(n)
        if mode <= 1:
            comp = 2.0
            for k in range(n):
                r[k] = _zero_at(x, coef[crow[i]], zn[crow[i]], t[k])
        else:
            comp = freq[b]
            for k in range(n):
                r[k] = 0.0
        if mode == 0 or mode == 2:
            dirty = value[i] + acc
            if dirty <= 0.0:
                continue
            res[i] = _solve(r, t, a, n, dirty, comp)
        else:
            pv, _d = _pv(r, t, a, n, value[i], comp)
            res[i] = pv - acc
    return res, acc_out


# ----------------------------------------------------------------------------------------------
# python API
# ----------------------------------------------------------------------------------------------
def _run(mode, bonds, bond_idx, settle, value, curves=None, curve_row=None):
    bidx = np.asarray(bond_idx, dtype=float)
    bidx = np.where(np.isfinite(bidx), bidx, -1).astype(np.int64)
    s_ymd, s_ord = _ymd_ord(settle)
    value = np.asarray(value, dtype=float)
    maxn = int(np.max(np.diff(bonds.ptr))) if len(bonds.ptr) > 1 else 1
    if curves is None:
        x = np.zeros(2)
        coef = np.zeros((1, 4, 1))
        zn = np.zeros((1, 2))
        crow = np.zeros(bidx.size, dtype=np.int64)
    else:
        x, coef, zn = curves.x, curves.coef, curves.z_nodes
        crow = np.asarray(curve_row, dtype=np.int64)
    return _kernel(mode, bidx, s_ord, s_ymd, value, crow,
                   bonds.ptr, bonds.cf_ord, bonds.cf_ymd, bonds.cf_amt, bonds.iss_ymd, bonds.iss_ord,
                   bonds.coupon, bonds.freq, bonds.dc_eu, bonds.eom, bonds.ok, x, coef, zn, max(maxn, 1))


def zspread(bonds, bond_idx, settle, clean, curves, curve_row):
    """Z-spread (decimal) and accrued for clean prices (per 100 face)."""
    return _run(0, bonds, bond_idx, settle, clean, curves, curve_row)


def price_from_z(bonds, bond_idx, settle, z, curves, curve_row):
    """Clean price from a Z-spread (the spread -> price mapping of the report)."""
    return _run(1, bonds, bond_idx, settle, z, curves, curve_row)


def ytm(bonds, bond_idx, settle, clean):
    """Street-convention yield to maturity (decimal, compounding = coupon frequency)."""
    return _run(2, bonds, bond_idx, settle, clean)


def price_from_ytm(bonds, bond_idx, settle, y):
    return _run(3, bonds, bond_idx, settle, y)
