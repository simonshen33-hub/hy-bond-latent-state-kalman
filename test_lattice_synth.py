"""test_lattice_synth.py -- checks of the call price, the lattice, the OAS inversion and the recovery kernel on
synthetic bonds and a synthetic curve (run `python -m synth` first).

1. Nesting: with zero volatility and no mean reversion the two-factor lattice reproduces the frozen-state layer-1
   price (callprice.price) bit for bit, both payoffs. This is the check that the lattice adds volatility to a
   price it otherwise agrees with, rather than being a different model.
2. fit_ar1 recovers an AR(1) from its own term variances; joint_moments reproduces the exact variances of the two
   factors and the covariance implied by correlated daily shocks.
3. Curve fit with volatility: the Arrow-Debreu sums of the lattice reprice the Treasury discount curve.
4. OAS round trip: invert a price to an OAS through the lattice (oas_obs.solve), then re-price at that OAS and
   recover the anchor inside the solver tolerance. The gap between the anchor Z-spread and the OAS is the option
   cost the filter must not see -- it is reported, because it is the whole reason the OAS is used.
5. oas_transform is the identity at the anchor itself and has slope 1/c away from it.
6. Recovery kernel: RFV pricing at R = 0 is exactly the Z engine's price at a spread equal to the default
   intensity, the analytic derivative matches a central difference, and the price converges on 100 R as the
   intensity grows.

    python test_lattice_synth.py
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import bonds as bondlib
import callprice as cp
import oas_obs
import recovery as rc
import synth
import tree2

TP = synth.PLACEHOLDER_TREE


def _setup():
    E = cp.Engine()
    cands = np.flatnonzero((np.diff(E.B.ptr) > 0) & E.B.ok & (np.diff(E.cptr) > 0))
    return E, cands


def _args(E, e, day):
    d = pd.Timestamp(day)
    st = bondlib.settlement([d], E.zc.dates)[0]
    s_ord = int((st - pd.Timestamp('1970-01-01')).days)
    B = E.B
    return (e, s_ord, st.year, st.month, st.day, B.ptr, B.cf_ord, B.cf_ymd, B.cf_amt, B.iss_ymd, B.iss_ord,
            B.coupon, B.dc_eu, B.eom, E.zc.x, E.zc.coef[E.zc.index_for(pd.DatetimeIndex([d]))[0]],
            E.zc.z_nodes[E.zc.index_for(pd.DatetimeIndex([d]))[0]], E.cptr, E.c_ord, E.c_px, E.bd_ord, E.bd_ymd)


def test_lattice_nests_layer1(E, cands):
    tb, ab, rb, sb = E.bufs()
    rng = np.random.default_rng(0)
    days = pd.bdate_range('2019-01-02', '2024-12-31')
    diffs, n, nfloor = [], 0, 0
    for case in range(40):
        e = int(rng.choice(cands))
        day = days[rng.integers(len(days))]
        a = _args(E, e, day)
        nn, _ = bondlib._flows(*a[:14], tb, ab)
        if nn == 0:
            continue
        s_cr = float(rng.uniform(0.005, 0.08))
        s_tot = s_cr + float(rng.uniform(0.0, 0.01))
        z0, z1 = float(rng.uniform(-6.0, 1.0)), float(np.exp(rng.uniform(0.0, 3.0)))
        itm = bool(case % 2)
        p1, _ = cp.price(*a, s_tot, s_cr, z0, z1, False, tb, ab, rb, sb, itm)
        settle = pd.Timestamp('1970-01-01') + pd.Timedelta(days=int(a[1]))
        crow = int(E.zc.index_for(pd.DatetimeIndex([day]))[0])
        p2, info, _, _ = tree2.price_bondday(E, e, settle, crow, 1, 1.0, 0.0, 1.0, 0.0, 0.0,
                                             lambda c, s=s_cr: np.full(len(c), s), s_tot - s_cr, z0, z1, itm, 0.0)
        n += 1
        if info[2] > 0:                       # the floored discount could not fit the curve on that day
            nfloor += 1
            continue
        diffs.append(abs(p2 - p1))
    print(f'    {n} bond-days ({nfloor} skipped: floored discount), max |lattice - layer 1| '
          f'{max(diffs):.3e} price points, median {np.median(diffs):.3e}')
    assert len(diffs) > 20
    assert max(diffs) < 1e-8, max(diffs)


def test_ar1_and_moments():
    nh = np.arange(1, 11) * 252.0
    V = [tree2.ar1_var(0.998, 5e-7, k) for k in nh]
    ph, q, _ = tree2.fit_ar1(nh, V)
    print(f'    fit_ar1 recovery: phi {ph:.9f} (err {abs(ph - 0.998):.2e}), '
          f'q rel err {abs(q / 5e-7 - 1.0):.2e}')
    assert abs(ph - 0.998) < 1e-8 and abs(q / 5e-7 - 1.0) < 1e-6
    worst = 0.0
    for rho in (-0.3, 0.0, 0.3):
        for k in (1, 5):
            Nst = 400 // k
            nbd = np.full(Nst + 1, k, np.int64)
            out = np.zeros((Nst + 1, 5))
            tree2.joint_moments(Nst, nbd, TP['phr'], TP['qr'], TP['phs'], TP['qs'], rho, k, out)
            nn = Nst * k
            vx = tree2.ar1_var(TP['phr'], TP['qr'], nn)
            vy = tree2.ar1_var(TP['phs'], TP['qs'], nn)
            pp = TP['phr'] * TP['phs']
            e = [abs(out[-1, 2] / vx - 1.0), abs(out[-1, 3] / vy - 1.0)]
            if rho != 0.0:
                cxy = rho * math.sqrt(TP['qr'] * TP['qs']) * (1.0 - pp ** nn) / (1.0 - pp)
                e.append(abs(out[-1, 4] / cxy - 1.0))
            else:
                e.append(abs(out[-1, 4]) / max(math.sqrt(vx * vy), 1e-300))
            worst = max(worst, max(e))
    print(f'    joint_moments vs exact AR(1) variances and cross-covariance: worst relative error {worst:.2e}')
    assert worst < 1e-6, worst


def test_curve_fit_with_volatility(E, cands):
    rng = np.random.default_rng(1)
    days = pd.bdate_range('2019-01-02', '2024-12-31')
    errs, clamped, shrunk = [], 0, 0
    for case in range(10):
        e = int(rng.choice(cands))
        day = days[rng.integers(len(days))]
        crow = int(E.zc.index_for(pd.DatetimeIndex([day]))[0])
        settle = bondlib.settlement([pd.Timestamp(day)], E.zc.dates)[0]
        if E.B.ptr[e + 1] == E.B.ptr[e]:
            continue
        P, info, _, _ = tree2.price_bondday(E, e, settle, crow, TP['step'], TP['phr'], TP['qr'], TP['phs'],
                                            TP['qs'], (TP['rho'], -TP['rho'])[case % 2],
                                            lambda c: np.full(len(c), 0.03), 0.0, TP['zeta0'], TP['zeta1'],
                                            True, 0.0)
        if not np.isfinite(P):
            continue
        errs.append(info[5])
        clamped += int(info[0])
        shrunk += int(info[1])
    print(f'    {len(errs)} bond-days, worst relative error of the Arrow-Debreu sums against P(T): '
          f'{max(errs):.2e}  (clamped branch probabilities {clamped}, shrunk correlations {shrunk})')
    assert len(errs) > 5
    assert max(errs) < 1e-8, max(errs)


def test_oas_round_trip(E, cands):
    """A bond trading above its call price has a positive option cost, so its OAS is below its Z-spread to
    maturity. The inversion must find that OAS and the lattice must reproduce the anchor from it."""
    day = pd.Timestamp('2021-06-15')
    crow = int(E.zc.index_for(pd.DatetimeIndex([day]))[0])
    settle = bondlib.settlement([day], E.zc.dates)[0]
    rows, costs = [], []
    for e in cands:
        e = int(e)
        for z_anchor in (120.0, 250.0):
            x = dict(e=e, settle=settle, crow=crow, step=TP['step'], phr=TP['phr'], qr=TP['qr'], phs=TP['phs'],
                     qs=TP['qs'], rho=TP['rho'], zeta0=TP['zeta0'], zeta1=TP['zeta1'], c_ld=0.0,
                     z_anchor=z_anchor)
            q, c, resid, n, flag, _ = oas_obs.solve(E, x)
            if flag != 'ok':
                continue
            back = oas_obs.ztree(E, x, q)
            rows.append((abs(back - z_anchor), n, c))
            costs.append(z_anchor - q)
        if len(rows) >= 24:
            break
    worst = max(r[0] for r in rows)
    print(f'    {len(rows)} anchors inverted, {np.mean([r[1] for r in rows]):.1f} lattice prices each')
    print(f'    worst |Z_tree(q) - Z*| {worst:.4f} bp (solver tolerance {oas_obs.ANCHOR_TOL} bp)')
    print(f'    option cost Z* - OAS: median {np.median(costs):.2f} bp, max {max(costs):.2f} bp')
    print(f'    transform slope dZ/dq: min {min(r[2] for r in rows):.4f}, max {max(r[2] for r in rows):.4f}')
    assert len(rows) >= 20
    assert worst <= oas_obs.ANCHOR_TOL + 1e-9, worst
    assert max(costs) > 1.0, 'the synthetic set has no bond whose call is worth anything'
    assert all(r[2] > 0 for r in rows)


def test_oas_transform_identity():
    q = np.array([250.0, 300.0, np.nan, 180.0])
    c = np.array([0.9, 1.0, 0.8, -0.1])
    z = np.array([260.0, 300.0, 200.0, 190.0])
    s = np.array([260.0, 310.0, 210.0, 190.0])
    V = np.array([4.0, 4.0, 4.0, 4.0])
    o, Vo = oas_obs.oas_transform(s, V, dict(q=q, c=c, z=z))
    print(f'    at the anchor (s = Z*): observation {o[0]:.1f} = q {q[0]:.1f}, variance {Vo[0]:.4f} = '
          f'V / c^2 {V[0] / c[0] ** 2:.4f}')
    print(f'    away from it: s - Z* = {s[1] - z[1]:+.1f} bp with slope {c[1]} -> {o[1]:.1f}')
    print(f'    dropped: no root {np.isnan(o[2])}, non-positive slope {np.isnan(o[3])}')
    assert o[0] == q[0] and abs(Vo[0] - V[0] / c[0] ** 2) < 1e-12
    assert abs(o[1] - (q[1] + (s[1] - z[1]) / c[1])) < 1e-12
    assert np.isnan(o[2]) and np.isnan(o[3]) and np.isnan(Vo[2]) and np.isnan(Vo[3])


def _recovery_arrays(E, e_arr, day):
    d = pd.Timestamp(day)
    st = bondlib.settlement([d] * len(e_arr), E.zc.dates)
    s_ord = ((pd.DatetimeIndex(st) - pd.Timestamp('1970-01-01')).days).to_numpy(np.int64)
    s_ymd = np.stack([pd.DatetimeIndex(st).year, pd.DatetimeIndex(st).month,
                      pd.DatetimeIndex(st).day], axis=1).astype(np.int64)
    crow = E.zc.index_for(pd.DatetimeIndex([d] * len(e_arr))).astype(np.int64)
    B, zc = E.B, E.zc
    ptr, tt, aa, rr, nflow, acc = rc._build(np.asarray(e_arr, np.int64), s_ord, s_ymd, crow, B.ptr, B.cf_ord,
                                            B.cf_ymd, B.cf_amt, B.iss_ymd, B.iss_ord, B.coupon, B.dc_eu, B.eom,
                                            zc.x, zc.coef, zc.z_nodes, int(np.max(np.diff(B.ptr))))
    A = dict(ptr=ptr, tt=tt, aa=aa, rr=rr, nflow=nflow, crow=crow, cx=zc.x, ccoef=zc.coef, czn=zc.z_nodes)
    return A, acc, pd.DatetimeIndex(st), crow


def test_recovery_kernel(E):
    e_arr = np.flatnonzero(E.B.ok & (np.diff(E.B.ptr) > 0))[:12]
    day = pd.Timestamp('2021-06-15')
    A, acc, st, crow = _recovery_arrays(E, e_arr, day)
    lam = 0.035
    p0 = np.array([rc.price(A, i, lam, 0.0, 1 / 52)[0] - acc[i] for i in range(len(e_arr))])
    ze = bondlib.price_from_z(E.B, e_arr, st, np.full(len(e_arr), lam), E.zc, crow)[0]
    d = np.nanmax(np.abs(p0 - ze))
    print(f'    R = 0 vs the Z engine at the same spread, {len(e_arr)} bonds: max |difference| '
          f'{d:.2e} price points')
    rel = []
    for i in range(len(e_arr)):
        for R in (0.2, 0.5):
            _, dp = rc.price(A, i, lam, R, 1 / 52)
            fd = (rc.price(A, i, lam + 1e-6, R, 1 / 52)[0] - rc.price(A, i, lam - 1e-6, R, 1 / 52)[0]) / 2e-6
            rel.append(abs(dp - fd) / max(abs(fd), 1e-12))
    print(f'    analytic d price / d lam vs central difference: max relative error {max(rel):.2e}')
    print('    as the intensity grows the price converges on 100 R = 40 (deviation in price points):')
    meds = []
    for lam_big in (5.0, 20.0, 100.0, 500.0):
        dev = [abs(rc.price(A, i, lam_big, 0.4, 1 / 365)[0] - 40.0) for i in range(len(e_arr))]
        meds.append(float(np.median(dev)))
        print(f'        lam = {lam_big:5.0f}   median {np.median(dev):7.4f}   max {max(dev):7.4f}')
    assert d < 1e-8, d
    assert max(rel) < 1e-5, max(rel)
    assert meds == sorted(meds, reverse=True), meds
    assert meds[-1] < 0.5, meds[-1]


def main():
    print('building the engine from the synthetic data ...', flush=True)
    E, cands = _setup()
    print(f'    {int(E.B.ok.sum())} supported bonds, {len(cands)} of them callable, '
          f'{len(E.zc.dates):,} curve days')
    jobs = [('test_lattice_nests_layer1', lambda: test_lattice_nests_layer1(E, cands)),
            ('test_ar1_and_moments', test_ar1_and_moments),
            ('test_curve_fit_with_volatility', lambda: test_curve_fit_with_volatility(E, cands)),
            ('test_oas_round_trip', lambda: test_oas_round_trip(E, cands)),
            ('test_oas_transform_identity', test_oas_transform_identity),
            ('test_recovery_kernel', lambda: test_recovery_kernel(E))]
    for name, fn in jobs:
        print(name, flush=True)
        fn()
        print('  ok', flush=True)
    print(f'{len(jobs)} tests passed')


if __name__ == '__main__':
    main()
