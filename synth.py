"""synth.py -- synthetic inputs, so every module in this repository runs with no real data.

Nothing here is estimated and nothing here is a result. The parameter values it produces are PLACEHOLDERS whose
only job is to make the code paths exercisable: a plausible order of magnitude, round numbers, documented as
placeholders. The fitted values of the private project are withheld (see params.template.json and the README).

    python -m synth              write data/ and params.synthetic.json
    python -m synth --help       options

What it writes
--------------
data/fred/DGS*.csv                  11 constant-maturity par yield series, in percent, in the layout the public
                                    FRED download has, so curve.load_fred reads them unchanged. Weekdays only,
                                    with a few days dropped to exercise the "use the previous curve" rule.
data/bloomberg/bloomberg_fields.csv one row per synthetic bond, in the column layout bonds.load_terms expects:
                                    fixed-coupon bullets and callables, plus a few unsupported kinds (floating,
                                    step-up, an unsupported day count) so the status codes get exercised.
data/bloomberg/call_schedule.csv    step-down call schedules for the callable bonds.
params.synthetic.json               placeholder parameter vectors, same keys as params.template.json.

What it offers as functions
---------------------------
panel(...)        a simulated four-state panel (states, trades, daily observations, liquidity measures, order
                  flow, notches, issuer groups) in the array layout the fourstate* filters take
flat_curve(...)   a one-day flat curve, for identities that need one
placeholder_params()  the dict written to params.synthetic.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import curve
import fourstate as fs
import fourstate3 as fs3
import fourstate4 as fs4

HERE = Path(__file__).parent
DATA = HERE / 'data'

# ----------------------------------------------------------------------------------------------
# PLACEHOLDER parameter values. Not estimates. Round numbers of a plausible order of magnitude for a US
# high-yield bond: spreads in basis points, variances in bp^2 per business day, rates in decimal per year.
# ----------------------------------------------------------------------------------------------
PLACEHOLDER = {
    # rating curve G_t(n) = a + b (n - n0) + c (n - n0)^2, each AR(1)
    'mu_a': 400.0, 'mu_b': 60.0, 'mu_c': 5.0,
    'phi_a': 0.99, 'phi_b': 0.99, 'phi_c': 0.99,
    'q_a': 25.0, 'q_b': 1.0, 'q_c': 0.1,
    # B: the bond's own part; u: permanent level
    'phi_I': 0.98, 'q_I': 50.0, 's2_u': 2500.0,
    # L: liquidity, and its three daily measures x_k = al_k + be_k L + e_k
    'phi_L': 0.90, 'q_L': 20.0,
    'al1': 1.5, 'al2': 0.004, 'al3': 0.5,
    'be1': -0.01, 'be2': -1e-5, 'be3': 0.01,
    'h1': 0.25, 'h2': 1e-5, 'h3': 0.25,
    # D: dealer inventory, driven by customer order flow; lam_D maps it into the spread
    'phi_D': 0.95, 'q_D': 0.0, 'lam_D': 50.0,
    # trade-type bias b_j and trade noise log R_j = d0 + d1 lv + d2 cust
    'g1': 5.0, 'g2': -1.0, 'g3': -5.0, 'g4': 1.0,
    'd0': 5.0, 'd1': -0.5, 'd2': 0.3,
    # issue-spread observation
    'iss_d': 10.0, 'iss_v': 400.0,
    # issuer factor
    'phi_C': 0.99, 'q_C': 25.0,
}
N0 = 13.0          # reference rating notch the curve is centred on (a design choice, not an estimate)

PLACEHOLDER_TREE = {
    # rate factor (per business day) and credit factor of the lattice, correlation of their daily shocks
    # rho is a placeholder small enough that the lattice represents it exactly at every node pair with these
    # variances; a larger |rho| makes it shrink the cross moment at some node pairs (tree2._joint)
    'phr': 0.999, 'qr': 1.0e-9, 'phs': 0.999, 'qs': 4.0e-9, 'rho': -0.05,
    # call intensity lam = exp(zeta0 + zeta1 M)
    'zeta0': -3.0, 'zeta1': 8.0,
    'step': 5,     # business days per lattice step
}
PLACEHOLDER_RECOVERY = {'h_step': 1.0 / 52.0, 'lam_max': 2.0}


def placeholder_params():
    return {
        '_note': 'PLACEHOLDERS, not estimates. The fitted values are withheld; see params.template.json.',
        'four_state': dict(PLACEHOLDER),
        'n0': N0,
        'lattice': dict(PLACEHOLDER_TREE),
        'recovery': dict(PLACEHOLDER_RECOVERY),
    }


def param_vector(names, overrides=None):
    """Placeholder parameter vector in the order of a fourstate module's NAMES."""
    d = dict(PLACEHOLDER)
    d.update(overrides or {})
    return np.array([d[n] for n in names], dtype=float)


# ----------------------------------------------------------------------------------------------
# synthetic Treasury curve
# ----------------------------------------------------------------------------------------------
def par_yields(dates, rng):
    """(n_days, 11) par yields in decimal from a random walk in level / slope / curvature. Kept positive and
    smooth so the semiannual bootstrap of curve.py is well posed."""
    n = len(dates)
    lam = 0.6
    L = 0.030 + np.cumsum(rng.normal(0.0, 0.0004, n))
    S = -0.015 + np.cumsum(rng.normal(0.0, 0.0004, n))
    C = 0.005 + np.cumsum(rng.normal(0.0, 0.0005, n))
    L = np.clip(L, 0.004, 0.09)
    t = curve.TENORS
    load_s = (1.0 - np.exp(-lam * t)) / (lam * t)
    load_c = load_s - np.exp(-lam * t)
    y = L[:, None] + S[:, None] * load_s[None, :] + C[:, None] * load_c[None, :]
    return np.clip(y, 0.0005, 0.12)


def write_curve(start='2016-01-04', end='2027-12-31', seed=0, folder=None):
    """Write the 11 FRED-style CSVs. A few weekdays are dropped (market closed), as in the real download."""
    folder = Path(folder or DATA / 'fred')
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    keep = rng.random(len(days)) > 0.02                      # ~2% of weekdays have no curve
    days = days[keep]
    y = par_yields(days, rng) * 100.0                        # FRED publishes percent
    for j, s in enumerate(curve.SERIES):
        pd.DataFrame({'DATE': days.strftime('%Y-%m-%d'), s: np.round(y[:, j], 2)}) \
            .to_csv(folder / f'{s}.csv', index=False)
    return folder, len(days)


def flat_curve(rate=0.04, date='2016-01-04'):
    """A one-day flat par curve as a ZeroCurves, for identities (flat curve => Z = ytm - rate)."""
    fred = pd.DataFrame(np.full((1, 11), rate), index=pd.DatetimeIndex([date]), columns=curve.SERIES)
    return curve.build(fred, 'pchip')


# ----------------------------------------------------------------------------------------------
# synthetic bond terms and call schedules
# ----------------------------------------------------------------------------------------------
_SECTORS = ['Communications', 'Consumer Cyclical', 'Consumer Non-cyclical', 'Energy', 'Industrial',
            'Technology', 'Basic Materials', 'Utilities', 'Financial Other', 'Transportation']
_RANKS = ['Secured', '1st lien', 'Sr Unsecured', 'Unsecured', '2nd lien', 'Sr Subordinated', 'Subordinated']


def write_bonds(n_bonds=40, n_issuers=12, seed=0, folder=None):
    """Write bloomberg_fields.csv and call_schedule.csv for n_bonds synthetic bonds.

    CUSIPs are made up: 'SYN' + a counter, zero-padded to 9 characters. The last few bonds are deliberately
    unsupported kinds so that bonds.load_terms' status codes are exercised."""
    folder = Path(folder or DATA / 'bloomberg')
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows, calls = [], []
    for b in range(n_bonds):
        issuer = b % n_issuers
        issue = pd.Timestamp('2016-02-01') + pd.Timedelta(days=int(rng.integers(0, 2000)))
        tenor = int(rng.choice([5, 7, 8, 10]))
        maturity = (issue + pd.DateOffset(years=tenor)).normalize()
        freq = 2
        first_cpn = issue + pd.DateOffset(months=6)
        coupon = float(np.round(4.0 + 6.0 * rng.random(), 3))
        callable_ = b % 4 != 3
        kind = 'FIXED'
        mty = 'CALLABLE' if callable_ else 'AT MATURITY'
        daycount = '30/360'
        calc = 'STREET CONVENTION'
        if b >= n_bonds - 3:                                  # unsupported kinds, on purpose
            kind, mty, daycount = [('FLOATING', 'AT MATURITY', '30/360'),
                                   ('STEP CPN', 'CALLABLE', '30/360'),
                                   ('FIXED', 'AT MATURITY', 'ACT/ACT')][b - (n_bonds - 3)]
        rows.append({
            'cusip': f'SYN{b:06d}', 'name': f'SYNISS{issuer} {coupon} {maturity:%Y}',
            'coupon': coupon, 'CPN_FREQ': freq, 'CPN_TYP': kind, 'MTY_TYP': mty,
            'CALC_TYP_DES': calc, 'DAY_CNT_DES': daycount,
            'issue_dt': issue.date(), 'maturity': maturity.date(), 'FIRST_CPN_DT': first_cpn.date(),
            'issue_price': float(np.round(98.0 + 2.0 * rng.random(), 3)),
            'amt_issued': float(int(rng.integers(3, 20)) * 1e8),
            'PAYMENT_RANK': _RANKS[int(rng.integers(0, len(_RANKS)))],
            'sector': _SECTORS[int(rng.integers(0, len(_SECTORS)))],
            'ticker': f'SYNISS{issuer}',
        })
        if mty == 'CALLABLE':
            first_call = (issue + pd.DateOffset(years=max(2, tenor - 5))).normalize()
            px = 100.0 + float(np.round(coupon / 2.0, 3))
            d = first_call
            while d < maturity:
                calls.append({'cusip': f'SYN{b:06d}', 'call_date': d.date(), 'call_price': round(px, 3)})
                d = (d + pd.DateOffset(years=1)).normalize()
                px = max(100.0, px - coupon / 4.0)
    pd.DataFrame(rows).to_csv(folder / 'bloomberg_fields.csv', index=False)
    pd.DataFrame(calls).to_csv(folder / 'call_schedule.csv', index=False)
    return folder, len(rows), len(calls)


# ----------------------------------------------------------------------------------------------
# synthetic four-state panel
# ----------------------------------------------------------------------------------------------
def panel(B=24, D=500, NG=6, seed=0, p_trade=0.35, par=None):
    """Simulate the four-state model and return everything the fourstate* filters take.

    States: common g = (a, b, c) and, per bond, (B_own, u, L, D) plus the issuer factor C_k. The simulation is
    the model's own transition and observation equations at the PLACEHOLDER parameters, so a filter run on the
    output must recover the states up to its own posterior variance -- that is what test_filters_synth.py checks.
    Returns a dict of arrays with the names the filters use.
    """
    rng = np.random.default_rng(seed)
    p = param_vector(fs4.NAMES) if par is None else np.asarray(par, float)
    d = {n: p[i] for i, n in enumerate(fs4.NAMES)}
    grp = np.arange(B) % NG                                    # issuer of each bond
    first_day = np.sort(rng.integers(0, max(1, D // 3), B))
    first_day[0] = 0
    mat_day = np.full(B, D + 10, np.int64)                     # none matures inside the window

    # common states
    g = np.zeros((3, D))
    mu = np.array([d['mu_a'], d['mu_b'], d['mu_c']])
    ph = np.array([d['phi_a'], d['phi_b'], d['phi_c']])
    qg = np.array([d['q_a'], d['q_b'], d['q_c']])
    g[:, 0] = mu + rng.normal(0, np.sqrt(qg / (1 - ph ** 2)))
    for t in range(1, D):
        g[:, t] = mu + ph * (g[:, t - 1] - mu) + rng.normal(0, np.sqrt(qg))

    # issuer factor
    C = np.zeros((NG, D))
    for k in range(NG):
        t0 = int(first_day[grp == k].min())
        for t in range(t0 + 1, D):
            C[k, t] = d['phi_C'] * C[k, t - 1] + rng.normal(0, math.sqrt(d['q_C']))

    # notch, order flow, per-bond states
    notch = np.full((B, D), np.nan)
    OF = np.zeros((B, D))
    Bown = np.zeros((B, D))
    u = np.zeros(B)
    L = np.zeros((B, D))
    Dst = np.zeros((B, D))
    beta = np.ones((B, D))
    wq = np.ones((B, D))
    u_prior = np.zeros(B)
    iss = np.full(B, np.nan)
    for b in range(B):
        n = float(rng.integers(10, 19))                        # Ba3-ish to Caa
        u[b] = rng.normal(0, math.sqrt(d['s2_u']))
        u_prior[b] = u[b] + rng.normal(0, 20.0)                # a noisy prior, as a regression prior would be
        for t in range(first_day[b], D):
            notch[b, t] = n
            if rng.random() < 0.002 and t > first_day[b]:      # a rating reclassification
                n = float(np.clip(n + rng.choice([-1.0, 1.0]), 1.0, 21.0))
            OF[b, t] = rng.normal(0.0, 0.004)
            if t > first_day[b]:
                Bown[b, t] = d['phi_I'] * Bown[b, t - 1] + rng.normal(0, math.sqrt(d['q_I']))
                L[b, t] = d['phi_L'] * L[b, t - 1] + rng.normal(0, math.sqrt(d['q_L']))
                Dst[b, t] = d['phi_D'] * Dst[b, t - 1] - OF[b, t - 1]
            else:
                L[b, t] = rng.normal(0, math.sqrt(d['q_L'] / (1 - d['phi_L'] ** 2)))
        # the rating-change offset of the model: G jumps, B offsets it. The simulation keeps notch constant
        # within a bond except for the rare reclassification, which the filter handles the same way.
        iss[b] = np.nan

    x1 = notch - N0
    s_true = (g[0][None, :] + g[1][None, :] * x1 + g[2][None, :] * x1 ** 2
              + beta * C[grp] + Bown + u[:, None] + L + d['lam_D'] * Dst)

    # trades: Poisson-ish number per bond-day, category 0 = dealer, 1 = customer buy, 2 = customer sell
    t_b, t_day, t_z, t_lvc, t_cat, t_n = [], [], [], [], [], np.zeros((B, D))
    t_vol = np.zeros((B, D))
    amt = np.full(B, 1.0e9)
    for b in range(B):
        for t in range(first_day[b], D):
            if rng.random() > p_trade:
                continue
            m = 1 + rng.poisson(2.0)
            t_n[b, t] = m
            for _ in range(m):
                lv = float(rng.normal(0.0, 1.0))
                cat = int(rng.choice([0, 1, 2], p=[0.35, 0.33, 0.32]))
                bias = (d['g1'] + d['g2'] * lv) if cat == 1 else (d['g3'] + d['g4'] * lv) if cat == 2 else 0.0
                cust = 1.0 if cat in (1, 2) else 0.0
                logR = d['d0'] + d['d1'] * lv + d['d2'] * cust
                z = s_true[b, t] + bias + rng.normal(0.0, math.exp(0.5 * logR))
                t_b.append(b); t_day.append(t); t_z.append(z); t_lvc.append(lv); t_cat.append(cat)
                t_vol[b, t] += math.exp(12.0 + 2.0 * lv)
    order = np.lexsort((np.array(t_day), np.array(t_b)))
    t_b = np.array(t_b)[order]
    t_day = np.array(t_day, np.int64)[order]
    t_z = np.array(t_z)[order]
    t_lvc = np.array(t_lvc)[order]
    t_cat = np.array(t_cat, np.int64)[order]
    ptr = np.r_[0, np.cumsum(np.bincount(t_b, minlength=B))].astype(np.int64)

    hide = np.zeros((B, D), np.bool_)
    s_obs, V, wll = fs.daily_obs(ptr, t_day, t_z, t_lvc, t_cat, hide,
                                 p[fs4.IDX['g1']:fs4.IDX['g1'] + 4].copy(),
                                 p[fs4.IDX['d0']:fs4.IDX['d0'] + 3].copy(), B, D, 0, D - 1)
    x = fs.measures(t_n, t_vol, amt, first_day)
    # the measures' observation rows are alpha_k + beta_k L + noise
    for k in range(3):
        al, be, h = p[fs4.IDX[f'al{k + 1}']], p[fs4.IDX[f'be{k + 1}']], p[fs4.IDX[f'h{k + 1}']]
        ok = np.isfinite(x[k])
        x[k][ok] = al + be * L[ok] + rng.normal(0.0, math.sqrt(h), int(ok.sum()))

    slot, gsize, smax = fs3.group_layout(grp)
    return dict(p=p, notch=notch, s_obs=s_obs, V=V, wll=wll, x=x, OF=OF, first_day=first_day.astype(np.int64),
                mat_day=mat_day, iss=iss, u_prior=u_prior, grp=grp.astype(np.int64), slot=slot, gsize=gsize,
                smax=smax, NG=NG, beta=beta, wq=wq, B=B, D=D, n0=N0, s_true=s_true, g_true=g, C_true=C,
                L_true=L, D_true=Dst, B_true=Bown, u_true=u, ptr=ptr, t_day=t_day, t_z=t_z, t_lvc=t_lvc,
                t_cat=t_cat, hide=hide)


def main():
    ap = argparse.ArgumentParser(description='write synthetic inputs (no real data is used or needed)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--bonds', type=int, default=40)
    ap.add_argument('--issuers', type=int, default=12)
    ap.add_argument('--start', default='2016-01-04')
    ap.add_argument('--end', default='2027-12-31')
    a = ap.parse_args()
    folder, n_days = write_curve(a.start, a.end, a.seed)
    print(f'curve   {folder}  {n_days} days, {len(curve.SERIES)} series')
    folder, n_b, n_c = write_bonds(a.bonds, a.issuers, a.seed)
    print(f'bonds   {folder}  {n_b} bonds, {n_c} call-schedule rows')
    out = HERE / 'params.synthetic.json'
    out.write_text(json.dumps(placeholder_params(), indent=1), encoding='utf-8')
    print(f'params  {out}  PLACEHOLDERS, not estimates')


if __name__ == '__main__':
    main()
