"""oas_obs.py -- the observation that enters the filter: a price inverted through the lattice to an
option-adjusted spread (OAS).

Why not the Z-spread to maturity
--------------------------------
The bonds are callable. A Z-spread to maturity charges the call twice: once in the market price the spread is
computed from, and again when the model prices the call. A filter fed that spread learns a "credit" state that
silently contains an option cost, and the option cost then moves with the rate curve, with time to the call date
and with the spread itself -- none of which is credit. Feeding the filter an OAS removes the double count: the
option is stripped off before the state is formed and put back when a price is needed.

Two trees, one set of inputs
----------------------------
tree 1 (inversion, this module)  strips the option off: given the day's price, find the spread at which the
    lattice reproduces it. That spread is the observation.
tree 2 (prediction, elsewhere)   puts the option back on the day a price is wanted, at that day's curve, that
    day's time to the call date and the filter's predicted credit spread.

Both trees must use the SAME credit volatility, mean reversion and credit-rate correlation, and those must be the
ones the filter itself estimated from the option-free spread -- otherwise the same OAS is stripped with one option
and restored with another, and the difference (an option-value difference, not a forecast) shows up as if it were
model error. In the private project this is reached by iteration to a fixed point: filter on the OAS set built
with the previous round's estimates, re-estimate, rebuild. The only difference allowed between the two trees is
that each may use the monthly estimate in force on its own day (point in time).

The inversion, per bond-day
---------------------------
  anchor   Z* = median Z-spread (bp) of the day's trades. Data only, no parameters. For the one issue
           observation of a bond, the issue Z on its first trading day.
  tree     tree2.price_bondday at a flat trial spread s, with the credit and rate factors and the correlation of
           the estimate in force (see params.template.json for the shape of that vector).
  root     q* with Z_tree(q*) = Z*, by secant from s0 = Z*, first step s1 = s0 - (Z_tree(s0) - Z*), tolerance
           ANCHOR_TOL, at most MAX_PRICES lattice prices. Two equal prices in a row mean the price is pinned by
           the call payment (flag 'flat') and the day is dropped.
  slope    c = (Z_tree(q* + H_SLOPE) - Z_tree(q*)) / H_SLOPE, one more price.
MAX_PRICES, H_SLOPE and ANCHOR_TOL are numerical devices (solver cap, difference step, root tolerance), not model
parameters.

Use in the filter
-----------------
A daily spread observation s with variance V becomes  q* + (s - Z*) / c  with variance  V / c^2  (oas_transform):
the first-order inverse of the tree map around the day's own anchor, so the trade-level noise and the customer
biases stay in bp of Z where they were estimated. No root, c <= 0, or a non-finite value -> the day is missing.
The iterated-EKF variant (fourstate4e.py) relinearises instead of using one slope: it needs the tree's Z on a
grid of offsets around q* (CURVE_OFFS, curve_nodes below), and re-values the call at the filter's updated spread.

Where liquidity enters
----------------------
The issuer's decision to call is a decision to refinance, so it looks at the cost of borrowing again -- credit --
not at how illiquid its bond happens to be. Three readings are implemented, chosen per run:
  'call'       c_ld = 0: the whole spread is the tree's credit spread, so the liquidity and inventory states also
               enter the call decision. (The simplest reading; what every set did first.)
  'discount'   the call decision uses the credit-only spread and the price discounts at credit + c_ld, with
               c_ld = L + lam_D D of the filter record for the day. q is then the credit-only OAS, c = dZ/dq and
               d_ld = dZ/dc_ld (one more price).
  'pressure'   L + lam_D D is a temporary pressure on the day's print, added to Z outside the tree: the tree runs
               on the credit spread alone and q solves Z_tree(q) = Z* - p_ld, so d_ld = 1 exactly, no extra price.

WITHHELD. The private project's drivers are not in this repository: the multiprocessing sweep over the panel, the
parquet/npz layout of the results, the fixed-point bookkeeping between rounds, and every estimated input the trees
take (credit and rate factors, correlation, call-intensity coefficients). Pass them in. `synth.py` builds a
synthetic bond and a placeholder parameter set so the inversion and the round trip can be exercised with no data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import bonds as bondlib
import tree2

MAX_PRICES = 10           # numerical device: cap of lattice prices in the root search
H_SLOPE = 1.0             # numerical device: forward-difference step of the slope (bp)
ANCHOR_TOL = 0.05         # numerical device: secant tolerance of the anchor shift (bp)
# Grid of the iterated-EKF curves (numerical device, fixed before any result): offsets from q* in bp. Most of the
# updated spreads land within a few bp of q*, so the nodes are dense near 0; 0 and +1 = H_SLOPE are the root and
# slope prices of the inversion and are not re-priced.
CURVE_OFFS = (-50.0, -20.0, -10.0, -5.0, -2.0, 0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)


def z_of(E, e, settle, crow, dirty):
    """Z-spread (decimal) of a dirty price: the Z engine of bonds.py applied to the lattice's price, so the
    observation stays in Z space."""
    B = E.B
    d = pd.Timestamp(settle)
    s_ord = int((d - pd.Timestamp('1970-01-01')).days)
    tb, ab, rb, sb = E.bufs()
    n, _ = bondlib._flows(e, s_ord, d.year, d.month, d.day, B.ptr, B.cf_ord, B.cf_ymd, B.cf_amt, B.iss_ymd,
                          B.iss_ord, B.coupon, B.dc_eu, B.eom, tb, ab)
    r = np.array([bondlib._zero_at(E.zc.x, E.zc.coef[crow], E.zc.z_nodes[crow], tb[j]) for j in range(n)])
    return bondlib._solve(r, tb[:n].copy(), ab[:n].copy(), n, dirty, 2.0)


def ztree(E, x, s_bp):
    """Z-spread (bp) of the lattice price of bond-day `x` at a flat credit spread of s_bp basis points.

    x: dict with the bond-day and the parameters in force --
        e, settle, crow   bond index in E.B, settlement date, curve row
        step              business days per lattice step
        phr, qr           rate factor (persistence, shock variance per business day)
        phs, qs           credit factor
        rho               correlation of the two daily shocks
        zeta0, zeta1      call-intensity coefficients, lam = exp(zeta0 + zeta1 M)
        c_ld              liquidity + inventory part of the spread (decimal), 0 under the 'call' reading
    """
    s = s_bp * 1e-4
    P, info, _, _ = tree2.price_bondday(E, x['e'], x['settle'], x['crow'], x['step'], x['phr'], x['qr'], x['phs'],
                                        x['qs'], x['rho'], lambda n: np.full(np.shape(n), s), x.get('c_ld', 0.0),
                                        x['zeta0'], x['zeta1'], True, 0.0)
    return z_of(E, x['e'], x['settle'], x['crow'], P) * 1e4 if np.isfinite(P) else np.nan


def solve(E, x, s0=None):
    """Invert one anchor: (q, c, resid, n_prices, flag, d_ld).

    q      the OAS that reproduces the day's anchor Z* through the lattice (bp)
    c      dZ/dq at q, the slope the filter linearises on
    resid  Z_tree(q) - Z*, within ANCHOR_TOL when flag is 'ok'
    flag   'ok' | 'flat' (the price is pinned by the call payment) | 'noconv' | 'nonfinite'
    d_ld   liquidity slope dZ/dc_ld; NaN under the 'call' reading

    x: as ztree, plus z_anchor (Z*, bp) and, for the liquidity readings, mode='pressure' with press (bp), or
    econ=True with c_ld. s0 starts the secant somewhere other than Z* -- the previous day's q, which saves
    lattice prices without moving the root beyond the tolerance.
    """
    zt = x['z_anchor'] - x.get('press', 0.0)       # 'pressure': the tree reproduces Z* less the day's pressure
    if s0 is None or not np.isfinite(s0):
        s0 = zt
    z0 = ztree(E, x, s0)
    n = 1
    if not np.isfinite(z0):
        return np.nan, np.nan, np.nan, n, 'nonfinite', np.nan
    s_prev, z_prev = s0, z0
    s = s0 - (z0 - zt)
    q, zq, flag = s0, z0, 'noconv'
    if abs(z0 - zt) <= ANCHOR_TOL:
        flag = 'ok'
    while flag == 'noconv' and n < MAX_PRICES:
        z = ztree(E, x, s)
        n += 1
        if not np.isfinite(z):
            flag = 'nonfinite'
            break
        q, zq = s, z
        if abs(z - zt) <= ANCHOR_TOL:
            flag = 'ok'
            break
        if z == z_prev:
            flag = 'flat'
            break
        s, s_prev, z_prev = s + (zt - z) * (s - s_prev) / (z - z_prev), s, z
    resid = zq - zt
    if flag != 'ok':
        return q, np.nan, resid, n, flag, np.nan
    zh = ztree(E, x, q + H_SLOPE)
    n += 1
    c = (zh - zq) / H_SLOPE if np.isfinite(zh) else np.nan
    d_ld = np.nan
    if x.get('mode') == 'pressure':
        d_ld = 1.0                     # the pressure is added to Z one for one
    elif x.get('econ'):
        zl = ztree(E, dict(x, c_ld=x.get('c_ld', 0.0) + H_SLOPE * 1e-4), q)
        n += 1
        d_ld = (zl - zq) / H_SLOPE if np.isfinite(zl) else np.nan
    return q, c, resid, n, flag, d_ld


def curve_nodes(E, x, q, resid=0.0, c=None, offs=CURVE_OFFS):
    """The lattice's Z (bp) at q + offs, for the iterated EKF (fourstate4e.curve). The nodes at 0 and +H_SLOPE are
    the inversion's own prices and are not re-priced: they are Z* + resid and that plus c * H_SLOPE."""
    out = np.empty(len(offs))
    for k, o in enumerate(offs):
        if o == 0.0:
            out[k] = x['z_anchor'] - x.get('press', 0.0) + resid
        elif o == H_SLOPE and c is not None:
            out[k] = x['z_anchor'] - x.get('press', 0.0) + resid + c * H_SLOPE
        else:
            out[k] = ztree(E, x, q + o)
    return out


def oas_transform(s, V, a):
    """Daily spread observations in OAS units: q* + (s - Z*) / c with variance V / c^2, the first-order inverse of
    the tree map around each day's own anchor. Missing (NaN) where there is no root or c <= 0.

    s, V: daily spread observation and its variance (bp, bp^2), any shape.
    a: dict of same-shaped arrays q (root), c (slope), z (anchor Z*)."""
    c = a['c']
    ok = np.isfinite(s) & np.isfinite(c) & (c > 0) & np.isfinite(a['q']) & np.isfinite(a['z'])
    cc = np.where(ok, c, 1.0)
    return np.where(ok, a['q'] + (s - a['z']) / cc, np.nan), np.where(ok, V / (cc * cc), np.nan)
