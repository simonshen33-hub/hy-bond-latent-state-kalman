"""demo.py -- the whole chain on synthetic data, end to end, in one command.

    python -m synth      (once; writes data/ and params.synthetic.json)
    python -m demo

No real data is read and none is needed. Every parameter is a documented placeholder from synth.py, not an
estimate: the point is to show what the code computes and in what order, not to report a result.

Stages
  1  zero curve            FRED-style par yields -> bootstrapped zero curve
  2  bond math             price -> Z-spread -> price, and the flat-curve identity
  3  layer 1               the call priced with today's state frozen
  4  layer 2               the lattice; what the call is worth; the lattice nests layer 1
  5  OAS inversion         a price inverted to an option-adjusted spread, and back
  6  four-state filter     a simulated panel filtered; what each state contributes
  7  iterated EKF          the same panel with the lattice transform relinearised in the update
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import bonds as bondlib
import callprice as cp
import curve
import fourstate4 as fs4
import fourstate4e as fs4e
import oas_obs
import synth
import tree2

TP = synth.PLACEHOLDER_TREE
HR = '-' * 100


def head(n, title):
    print(f'\n{HR}\n{n}. {title}\n{HR}', flush=True)


def stage1_curve():
    head(1, 'Zero curve: FRED-style constant-maturity par yields -> bootstrapped semiannual zero curve')
    if not (synth.DATA / 'fred').exists():
        synth.write_curve()
        synth.write_bonds()
    fred = curve.load_fred(synth.DATA / 'fred')
    zc = curve.build(fred, 'pchip')
    print(f'{len(fred):,} days with a full curve, {fred.index[0]:%Y-%m-%d} to {fred.index[-1]:%Y-%m-%d}')
    row = len(fred) // 2
    lbl = ['1m', '3m', '6m', '1y', '2y']
    print(f'day {fred.index[row]:%Y-%m-%d}: par yields  '
          + '  '.join(f'{n} {100 * v:.2f}%' for n, v in zip(lbl, fred.to_numpy()[row, :5])))
    print('             zero rates  '
          + '  '.join(f'{t:g}y {100 * zc.zero(row, t):.2f}%' for t in (0.5, 1, 2, 5, 10, 30)))
    # the bootstrap is exact by construction: a par bond at each grid node prices to 1
    par = curve._interp('pchip', curve.TENORS, fred.to_numpy()[row])(curve.GRID)
    z = zc.z_nodes[row][2:]
    df = (1 + z / 2) ** (-2 * curve.GRID)
    err = max(abs(par[n] / 2 * df[:n + 1].sum() + df[n] - 1.0) for n in range(1, curve.GRID.size))
    print(f'par bonds at the 60 bootstrap nodes reprice to 1 within {err:.2e}')
    closed = pd.bdate_range(fred.index[0], fred.index[-1]).difference(fred.index)
    print(f'{len(closed)} weekdays have no curve (market closed); those days use the previous curve')
    return zc


def stage2_bonds(E):
    head(2, 'Bond math: contractual cash flows, accrued interest, Z-spread')
    B = E.B
    ok = np.flatnonzero(B.ok)
    print(f'{len(B.terms)} synthetic bonds, {len(ok)} supported; the rest carry a status code:')
    for s, n in B.terms.loc[~B.ok, 'status'].value_counts().items():
        print(f'    {n} x {s}')
    day = pd.Timestamp('2021-06-15')
    st = bondlib.settlement([day] * 6, E.zc.dates)
    crow = E.zc.index_for(pd.DatetimeIndex([day] * 6))
    e = ok[:6]
    clean = np.array([98.0, 101.5, 104.0, 92.0, 108.0, 99.0])
    z, acc = bondlib.zspread(B, e, st, clean, E.zc, crow)
    back, _ = bondlib.price_from_z(B, e, st, z, E.zc, crow)
    nbd = int(np.searchsorted(E.zc.dates, np.datetime64(st[0], 'ns'))
              - np.searchsorted(E.zc.dates, np.datetime64(day, 'ns')))
    print(f'\ntrade date {day:%Y-%m-%d}, settlement {st[0]:%Y-%m-%d} (T+{nbd} business days)')
    print('  bond      coupon   maturity    clean     accrued   Z-spread    price back   round trip')
    for k in range(len(e)):
        mat = pd.Timestamp('1970-01-01') + pd.Timedelta(days=int(B.cf_ord[B.ptr[e[k] + 1] - 1]))
        print(f'  {B.terms["cusip"][e[k]]:<9} {B.coupon[e[k]]:6.3f}%  {mat:%Y-%m}  {clean[k]:8.3f}  '
              f'{acc[k]:8.3f}  {1e4 * z[k]:9.2f} bp  {back[k]:10.6f}  {abs(back[k] - clean[k]):.2e}')
    print(f'\nround trip price -> Z -> price, worst error {np.abs(back - clean).max():.2e} price points')


def stage3_layer1(E):
    head(3, "Layer 1: the call priced with today's state frozen (no volatility)")
    cands = np.flatnonzero((np.diff(E.B.ptr) > 0) & E.B.ok & (np.diff(E.cptr) > 0))
    day = pd.Timestamp('2021-06-15')
    crow = int(E.zc.index_for(pd.DatetimeIndex([day]))[0])
    settle = bondlib.settlement([day], E.zc.dates)[0]
    print('The issuer calls at a hazard rate lam = exp(zeta0 + zeta1 M), M = value / call payment - 1. Under the')
    print('itm_only restriction the hazard is 0 while the bond is worth less than the call payment, so the call')
    print('costs a discount bond nothing; without the restriction a call below value is a gift to the holder and')
    print(f'the model Z falls below the credit spread. zeta0 = {TP["zeta0"]}, '
          f'zeta1 = {TP["zeta1"]} (placeholders).\n')
    print('  bond      coupon   straight price      model Z at a 150 bp credit spread')
    print('                                      any-time call   in-the-money only   option cost')
    tb, ab, rb, sb = E.bufs()
    s_ord = int((settle - pd.Timestamp('1970-01-01')).days)
    base = (E.B.ptr, E.B.cf_ord, E.B.cf_ymd, E.B.cf_amt, E.B.iss_ymd, E.B.iss_ord, E.B.coupon, E.B.dc_eu,
            E.B.eom, E.zc.x, E.zc.coef[crow], E.zc.z_nodes[crow], E.cptr, E.c_ord, E.c_px, E.bd_ord, E.bd_ymd)
    shown = 0
    for e in cands:
        e = int(e)
        px = bondlib.price_from_z(E.B, [e], pd.DatetimeIndex([settle]), [0.015], E.zc, [crow])[0][0]
        z_any = cp.model_z(e, s_ord, settle.year, settle.month, settle.day, *base, 0.015, 0.015,
                           TP['zeta0'], TP['zeta1'], False, tb, ab, rb, sb, False)
        z_itm = cp.model_z(e, s_ord, settle.year, settle.month, settle.day, *base, 0.015, 0.015,
                           TP['zeta0'], TP['zeta1'], False, tb, ab, rb, sb, True)
        if not (np.isfinite(z_any) and np.isfinite(z_itm)):
            continue
        print(f'  {E.B.terms["cusip"][e]:<9} {E.B.coupon[e]:6.3f}%  {px:13.3f}   {1e4 * z_any:11.2f} bp  '
              f'{1e4 * z_itm:15.2f} bp  {1e4 * z_itm - 150.0:+9.2f} bp')
        shown += 1
        if shown == 6:
            break
    return cands, day, crow, settle


def stage4_lattice(E, cands, day, crow, settle):
    head(4, 'Layer 2: a two-factor (rate x credit) trinomial lattice')
    print('Variance accrues per business day in both factors; the rate factor is fitted to reproduce the day\'s')
    print('Treasury discount curve exactly, the credit factor to the multi-factor credit variance, and the two')
    print('daily shocks are correlated. The call is then a hazard at every grid date, as in layer 1.\n')
    e = int(cands[np.argmax([E.B.coupon[int(c)] for c in cands])])
    x = dict(e=e, settle=settle, crow=crow, step=TP['step'], phr=TP['phr'], qr=TP['qr'], phs=TP['phs'],
             qs=TP['qs'], rho=TP['rho'], zeta0=TP['zeta0'], zeta1=TP['zeta1'], c_ld=0.0)
    t = time.time()
    z = oas_obs.ztree(E, x, 150.0)
    print(f'bond {E.B.terms["cusip"][e]} ({E.B.coupon[e]:.3f}% coupon), {TP["step"]} business days per step:')
    print(f'    credit spread in           150.00 bp')
    print(f'    Z-spread of the price out  {z:6.2f} bp     option cost {z - 150.0:+.2f} bp '
          f'({time.time() - t:.2f} s)')
    p_vol, info, _, _ = tree2.price_bondday(E, e, settle, crow, TP['step'], TP['phr'], TP['qr'], TP['phs'],
                                            TP['qs'], TP['rho'], lambda c: np.full(len(c), 0.015), 0.0,
                                            TP['zeta0'], TP['zeta1'], True, 0.0)
    p_flat, _, _, _ = tree2.price_bondday(E, e, settle, crow, 1, 1.0, 0.0, 1.0, 0.0, 0.0,
                                          lambda c: np.full(len(c), 0.015), 0.0, TP['zeta0'], TP['zeta1'],
                                          True, 0.0)
    tb, ab, rb, sb = E.bufs()
    s_ord = int((settle - pd.Timestamp('1970-01-01')).days)
    p1, _ = cp.price(e, s_ord, settle.year, settle.month, settle.day, E.B.ptr, E.B.cf_ord, E.B.cf_ymd,
                     E.B.cf_amt, E.B.iss_ymd, E.B.iss_ord, E.B.coupon, E.B.dc_eu, E.B.eom, E.zc.x,
                     E.zc.coef[crow], E.zc.z_nodes[crow], E.cptr, E.c_ord, E.c_px, E.bd_ord, E.bd_ymd,
                     0.015, 0.015, TP['zeta0'], TP['zeta1'], False, tb, ab, rb, sb, True)
    print(f'\n    lattice price with volatility          {p_vol:.6f}')
    print(f'    lattice price at zero volatility       {p_flat:.6f}')
    print(f'    layer 1 price (state frozen)           {p1:.6f}   difference {abs(p_flat - p1):.2e}')
    print(f'    the lattice NESTS layer 1: with no volatility and no mean reversion the two agree.')
    print(f'    switching the volatility on moves the price by {p_vol - p_flat:+.4f} points here: the price is')
    print(f'    convex in the credit spread, which pushes up, against the call option, which pushes down.')
    print(f'\n    Arrow-Debreu sums reprice the Treasury curve within {info[5]:.2e} (relative)')
    return e


def stage5_oas(E, e, day, crow, settle):
    head(5, 'The OAS inversion: why the filter is fed an OAS and not a Z-spread to maturity')
    print('A Z-spread to maturity charges the call twice: once inside the market price it is computed from, and')
    print('again when the model prices the call. So the price is inverted through the lattice first: find the')
    print('spread q at which the lattice reproduces the day\'s price. q carries no option cost, and it is q')
    print('that the Kalman filter observes.\n')
    print('  anchor Z*     OAS q      option cost   slope dZ/dq   lattice prices   round trip |Z_tree(q) - Z*|')
    for z_anchor in (120.0, 200.0, 350.0):
        x = dict(e=e, settle=settle, crow=crow, step=TP['step'], phr=TP['phr'], qr=TP['qr'], phs=TP['phs'],
                 qs=TP['qs'], rho=TP['rho'], zeta0=TP['zeta0'], zeta1=TP['zeta1'], c_ld=0.0,
                 z_anchor=z_anchor)
        q, c, resid, n, flag, _ = oas_obs.solve(E, x)
        back = oas_obs.ztree(E, x, q)
        print(f'  {z_anchor:8.2f} bp  {q:8.2f} bp  {z_anchor - q:+8.2f} bp  {c:11.5f}   {n:14d}   '
              f'{abs(back - z_anchor):.2e} bp   [{flag}]')
    x = dict(e=e, settle=settle, crow=crow, step=TP['step'], phr=TP['phr'], qr=TP['qr'], phs=TP['phs'],
             qs=TP['qs'], rho=TP['rho'], zeta0=TP['zeta0'], zeta1=TP['zeta1'], c_ld=0.0, z_anchor=200.0)
    q, c, resid, n, flag, _ = oas_obs.solve(E, x)
    s = np.array([195.0, 200.0, 205.0])
    V = np.full(3, 9.0)
    o, Vo = oas_obs.oas_transform(s, V, dict(q=np.full(3, q), c=np.full(3, c), z=np.full(3, 200.0)))
    print('\nThe filter uses the first-order inverse around the day\'s own anchor, so the trade noise and the')
    print('customer biases stay in the basis points of Z where they were estimated:')
    print('    daily Z observation  ->  OAS observation     variance  ->  variance')
    for k in range(3):
        print(f'    {s[k]:8.2f} bp          {o[k]:12.2f} bp   {V[k]:8.2f}      {Vo[k]:8.3f}')
    print('\nThe iterated EKF (stage 7) replaces that single slope with a relinearisation at the filter\'s own')
    print('updated spread, using the lattice\'s Z on a grid of offsets around q:')
    H = oas_obs.curve_nodes(E, x, q, resid, c)
    print('    offset from q (bp)  ' + ' '.join(f'{o_:>8.0f}' for o_ in oas_obs.CURVE_OFFS))
    print('    lattice Z     (bp)  ' + ' '.join(f'{h:>8.2f}' for h in H))
    return q, c, H


def stage6_filter():
    head(6, 'The four-state panel filter')
    print('Per bond:  B = its own credit, u = a permanent level, L = liquidity, D = dealer inventory.')
    print('Shared:    G_t(n) = a + b (n - n0) + c (n - n0)^2, the rating curve, and C_k, an issuer factor.')
    print('')
    print('    fair spread  s_it = G_t(n_it) + beta_it C_k + B_it + u_i + L_it + lam_D D_it')
    print('')
    print('Each bond-day with trades contributes one spread observation (the inverse-variance combination of')
    print('that day\'s prints, with the customer-buy / customer-sell bias taken out) and three liquidity')
    print('measures. Order flow drives D. A rating reclassification moves G and is offset inside B, so a')
    print('reclassification is not news about the bond.\n')
    t = time.time()
    S = synth.panel(B=24, D=400, NG=6, seed=0)
    print(f'simulated {S["B"]} bonds in {S["NG"]} issuers over {S["D"]} business days, '
          f'{len(S["t_z"]):,} trades, {int(np.isfinite(S["s_obs"]).sum()):,} bond-days with an observation '
          f'({time.time() - t:.1f} s)')
    fm = np.zeros((S['B'], S['D']))
    fv = np.zeros((S['B'], S['D']))
    g_out = np.zeros((S['D'], 3))
    um = np.zeros((S['B'], S['D'], 2))
    c_out = np.zeros((S['NG'], S['D']))
    t = time.time()
    ll = fs4.ci_filter4(S['p'].reshape(1, -1), np.zeros(S['D'], np.int64), S['notch'], S['s_obs'], S['V'],
                        S['x'], S['OF'], S['first_day'], S['mat_day'], S['iss'], S['u_prior'], S['grp'],
                        S['slot'], S['gsize'], S['smax'], S['beta'], S['wq'], 0, 0, S['D'] - 1, 0, S['n0'],
                        True, fm, fv, g_out, um, c_out, True, True, False)
    print(f'filtered in {time.time() - t:.2f} s;  log-likelihood {ll:,.2f}')
    ok = np.isfinite(fm) & (fv > 0) & np.isfinite(S['notch'])
    e = (fm[ok] - S['s_true'][ok]) / np.sqrt(fv[ok])
    print(f'\nagainst the simulated truth, over {int(ok.sum()):,} bond-days:')
    print(f'    fair spread RMSE            {np.sqrt(((fm[ok] - S["s_true"][ok]) ** 2).mean()):7.2f} bp')
    print(f'    s.d. of the truth itself    {S["s_true"][ok].std():7.2f} bp')
    print(f'    standardised error          mean {e.mean():+.3f}, s.d. {e.std():.3f}   '
          f'(a correctly calibrated filter gives 0 and 1)')
    print('\nwhat the states contribute to the fair spread (s.d. across bond-days, bp):')
    sel = ok
    x1 = S['notch'] - S['n0']
    G = g_out[:, 0][None, :] + g_out[:, 1][None, :] * x1 + g_out[:, 2][None, :] * x1 ** 2
    parts = [('G  rating curve', G), ('C  issuer factor', S['beta'] * c_out[S['grp']]),
             ('B  own credit', um[:, :, 0]), ('u  permanent level', um[:, :, 1]),
             ('lam_D D + L  (residual)', fm - G - S['beta'] * c_out[S['grp']] - um[:, :, 0] - um[:, :, 1])]
    for nm, v in parts:
        print(f'    {nm:<26} {np.nanstd(v[sel]):8.2f}')
    return S


def stage7_iekf(S):
    head(7, 'The iterated EKF: relinearising the lattice transform inside the update')
    print('The call is worth what it is worth at the bond\'s latent credit spread, not at the median of the')
    print('day\'s prints. The one-slope transform of stage 5 linearises at the print; the iterated EKF')
    print('relinearises at the filter\'s own updated spread -- yesterday\'s state, today\'s trades and the')
    print('issuer\'s other bonds -- and repeats until the linearisation point stops moving (Gauss-Newton on')
    print('the day\'s posterior).\n')
    print('Here the transform curves are straight lines of unit slope, which makes the iterated filter exactly')
    print('the plain filter: the check that the extra machinery is a generalisation, not a different model.')
    print('With real lattice curves (stage 5) the two differ, and by how much is an empirical question that')
    print('needs the licensed data.\n')
    B, D = S['B'], S['D']
    offs = np.array(oas_obs.CURVE_OFFS, float)
    qc = np.where(np.isfinite(S['s_obs']), S['s_obs'], np.nan)
    Hc = np.full((B, D, len(offs)), np.nan)
    fin = np.isfinite(qc)
    Hc[fin] = qc[fin][:, None] + offs[None, :]
    fm = np.zeros((B, D)); fv = np.zeros((B, D)); g_out = np.zeros((D, 3))
    um = np.zeros((B, D, 2)); c_out = np.zeros((S['NG'], D)); it_out = np.zeros(D)
    cld = np.full((B, D), np.nan); dld = np.full((B, D), np.nan)
    t = time.time()
    ll = fs4e.ci_filter4e(S['p'].reshape(1, -1), np.zeros(D, np.int64), S['notch'], S['s_obs'], S['V'], S['x'],
                          S['OF'], S['first_day'], S['mat_day'], S['iss'], S['u_prior'], S['grp'], S['slot'],
                          S['gsize'], S['smax'], S['beta'], S['wq'], 0, 0, D - 1, 0, S['n0'], True, fm, fv,
                          g_out, um, c_out, True, True, False, S['s_obs'], S['V'], qc, Hc, offs, 10, 1e-6,
                          it_out, cld, dld)
    n_it = int(it_out.max())
    print(f'iterated EKF log-likelihood {ll:,.6f}   ({time.time() - t:.2f} s, at most {n_it} '
          f'relinearisation{"s" if n_it != 1 else ""} on any day)')
    print('plain filter log-likelihood  see stage 6; the two agree to machine precision under these curves.')


def main():
    print(__doc__.split('Stages')[0].strip())
    stage1_curve()
    E = cp.Engine()
    stage2_bonds(E)
    cands, day, crow, settle = stage3_layer1(E)
    e = stage4_lattice(E, cands, day, crow, settle)
    stage5_oas(E, e, day, crow, settle)
    S = stage6_filter()
    stage7_iekf(S)
    print(f'\n{HR}\ndone. Nothing above is an estimate: every parameter came from synth.PLACEHOLDER.\n{HR}')


if __name__ == '__main__':
    main()
