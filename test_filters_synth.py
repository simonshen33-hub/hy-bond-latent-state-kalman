"""test_filters_synth.py -- checks of the four-state panel filters on a simulated panel (no real data).

1. The partitioned filter (ci_filter4: common 3x3 block + one block per issuer, cross-covariances dropped
   between days) stays close to the full joint filter (exact_filter4: 3 + NG + 4B states), and its posterior
   variance is close to it and centred slightly above it. This is the check that licenses using the
   partitioned filter, which is what makes the panel tractable.
2. With q_C -> 0 the issuer factor carries no shock and model v4 collapses to model v2 (fourstate2),
   compared through the two exact filters.
3. The filter recovers the simulated latent fair spread: the standardised error (filtered mean minus truth,
   over the filter's own posterior s.d.) has mean about 0 and s.d. about 1.
4. The iterated-EKF filter (fourstate4e) with straight-line transform curves of unit slope and no
   relinearisation is exactly the plain filter on the same observations -- i.e. the extra machinery is a
   generalisation, not a change of model.

    python test_filters_synth.py
"""
from __future__ import annotations

import numpy as np

import fourstate2 as fs2
import fourstate4 as fs4
import fourstate4e as fs4e
import oas_obs
import synth


def _outs(B, D, NG):
    return (np.zeros((B, D)), np.zeros((B, D)), np.zeros((D, 3)), np.zeros((B, D, 2)), np.zeros((NG, D)))


def _run4(S, p, record=True):
    fm, fv, g_out, um, c_out = _outs(S['B'], S['D'], S['NG'])
    par_sets = np.asarray(p, float).reshape(1, -1)
    par_idx = np.zeros(S['D'], np.int64)
    ll = fs4.ci_filter4(par_sets, par_idx, S['notch'], S['s_obs'], S['V'], S['x'], S['OF'], S['first_day'],
                        S['mat_day'], S['iss'], S['u_prior'], S['grp'], S['slot'], S['gsize'], S['smax'],
                        S['beta'], S['wq'], 0, 0, S['D'] - 1, 0, S['n0'], record, fm, fv, g_out, um, c_out,
                        True, True, False)
    return ll, fm, fv


def _run4_exact(S, p):
    fm = np.zeros((S['B'], S['D']))
    fv = np.zeros((S['B'], S['D']))
    ll = fs4.exact_filter4(np.asarray(p, float), S['notch'], S['s_obs'], S['V'], S['x'], S['OF'], S['first_day'],
                           S['mat_day'], S['iss'], S['u_prior'], S['grp'], S['NG'], S['beta'], S['wq'], 0, 0,
                           S['D'] - 1, 0, S['n0'], False, True, fm, fv)
    return ll, fm, fv


def _run2(S, p, exact):
    B, D = S['B'], S['D']
    clus = np.zeros((B, 1), np.int64)
    clus_of_day = np.zeros(D, np.int64)
    fm = np.zeros((B, D))
    fv = np.zeros((B, D))
    p = np.asarray(p, float)[:fs2.NP]
    if exact:
        return fs2.exact_filter2(p, S['notch'], S['s_obs'], S['V'], S['x'], S['OF'], S['first_day'], S['mat_day'],
                                 S['iss'], S['u_prior'], clus, clus_of_day, 0, 0, D - 1, 0, S['n0'], False, True,
                                 fm, fv), fm
    g_out = np.zeros((D, 3))
    um = np.zeros((B, D, 2))
    return fs2.ci_filter2(p.reshape(1, -1), np.zeros(D, np.int64), S['notch'], S['s_obs'], S['V'], S['x'],
                          S['OF'], S['first_day'], S['mat_day'], S['iss'], S['u_prior'], clus, clus_of_day, 0,
                          0, D - 1, 0, S['n0'], True, fm, fv, g_out, um, True, True, False), fm


def test_partitioned_close_to_joint(S):
    """The partitioned filter is an approximation, so this measures the approximation rather than asserting
    equality: the log-likelihood, the filtered fair spread and the posterior variance against the joint
    filter. The variance ratio is centred slightly above 1 (dropping cross-covariances usually removes
    information) but is not bounded below by 1 at every bond-day."""
    ll_a, fm_a, fv_a = _run4(S, S['p'])
    ll_e, fm_e, fv_e = _run4_exact(S, S['p'])
    rel = abs(ll_a - ll_e) / abs(ll_e)
    ok = np.isfinite(fm_a) & np.isfinite(fm_e) & np.isfinite(S['notch']) & (fv_e > 0)
    d = np.abs(fm_a[ok] - fm_e[ok])
    vr = fv_a[ok] / fv_e[ok]
    print(f'    loglik partitioned {ll_a:,.2f}   joint {ll_e:,.2f}   relative difference {rel:.3e}')
    print(f'    fair spread |difference| over {int(ok.sum()):,} bond-days (truth s.d. '
          f'{S["s_true"][ok].std():.0f} bp): median {np.median(d):.3f}, p99 {np.quantile(d, 0.99):.3f} bp')
    print(f'    posterior variance ratio approx/joint: median {np.median(vr):.4f}, '
          f'min {vr.min():.4f}, p99 {np.quantile(vr, 0.99):.4f}')
    assert np.isfinite(ll_a) and np.isfinite(ll_e)
    assert rel < 1e-2, rel
    assert np.median(d) < 2.0, np.median(d)
    assert np.quantile(vr, 0.01) > 0.8, np.quantile(vr, 0.01)
    assert 1.0 <= np.median(vr) < 1.1, np.median(vr)


def test_qC_zero_is_v2(S):
    """Compared through the two exact filters: the partitioned filters block by issuer (v4) and by bond (v2),
    so even for the same model their approximations differ, which is itself what test 1 measures."""
    p = S['p'].copy()
    p[fs4.IDX['q_C']] = 1e-12
    ll4, fm4, _ = _run4_exact(S, p)
    ll2, fm2 = _run2(S, p, exact=True)
    ok = np.isfinite(fm4) & np.isfinite(fm2) & (fm2 != 0.0)
    dmax = float(np.abs(fm4[ok] - fm2[ok]).max())
    print(f'    q_C -> 0:  v4 loglik {ll4:,.6f}   v2 loglik {ll2:,.6f}   diff {abs(ll4 - ll2):.3e}')
    print(f'    largest fair-spread difference: {dmax:.3e} bp')
    assert abs(ll4 - ll2) / abs(ll2) < 1e-6, abs(ll4 - ll2)
    assert dmax < 1e-4, dmax


def test_recovers_simulated_state(S):
    _, fm, fv = _run4(S, S['p'])
    ok = np.isfinite(fm) & (fv > 0) & np.isfinite(S['notch'])
    e = (fm[ok] - S['s_true'][ok]) / np.sqrt(fv[ok])
    print(f'    standardised error over {int(ok.sum()):,} bond-days: mean {e.mean():+.3f}, s.d. {e.std():.3f}')
    print(f'    RMSE {np.sqrt(((fm[ok] - S["s_true"][ok]) ** 2).mean()):.2f} bp against a fair spread with '
          f's.d. {S["s_true"][ok].std():.0f} bp')
    assert abs(e.mean()) < 0.25, e.mean()
    assert 0.7 < e.std() < 1.4, e.std()


def test_iekf_reduces_to_plain_filter(S):
    """Straight-line transform curves of unit slope through each day's own observation, and no relinearisation:
    the pseudo-observation is the observation itself, so the iterated-EKF filter must give the plain filter."""
    B, D = S['B'], S['D']
    offs = np.array(oas_obs.CURVE_OFFS, float)
    G = len(offs)
    qc = np.where(np.isfinite(S['s_obs']), S['s_obs'], np.nan)
    Hc = np.full((B, D, G), np.nan)
    fin = np.isfinite(qc)
    Hc[fin] = qc[fin][:, None] + offs[None, :]                    # Z = s exactly: slope 1, no curvature
    fm, fv, g_out, um, c_out = _outs(B, D, S['NG'])
    it_out = np.zeros(D)
    cld = np.full((B, D), np.nan)
    dld = np.full((B, D), np.nan)
    ll_e = fs4e.ci_filter4e(S['p'].reshape(1, -1), np.zeros(D, np.int64), S['notch'], S['s_obs'], S['V'], S['x'],
                            S['OF'], S['first_day'], S['mat_day'], S['iss'], S['u_prior'], S['grp'], S['slot'],
                            S['gsize'], S['smax'], S['beta'], S['wq'], 0, 0, D - 1, 0, S['n0'], True, fm, fv,
                            g_out, um, c_out, True, True, False, S['s_obs'], S['V'], qc, Hc, offs, 0, 1e-8,
                            it_out, cld, dld)
    ll_p, fm_p, _ = _run4(S, S['p'])
    ok = np.isfinite(fm) & np.isfinite(fm_p) & (fm_p != 0.0)
    dmax = float(np.abs(fm[ok] - fm_p[ok]).max())
    print(f'    iterated EKF {ll_e:,.6f}   plain {ll_p:,.6f}   diff {abs(ll_e - ll_p):.3e}')
    print(f'    largest fair-spread difference: {dmax:.3e} bp')
    assert abs(ll_e - ll_p) / abs(ll_p) < 1e-8
    assert dmax < 1e-8, dmax


def main():
    print('simulating a panel at the placeholder parameters ...', flush=True)
    S = synth.panel(B=24, D=400, NG=6, seed=0)
    n_obs = int(np.isfinite(S['s_obs']).sum())
    print(f'    {S["B"]} bonds, {S["NG"]} issuers, {S["D"]} business days, {len(S["t_z"]):,} trades, '
          f'{n_obs:,} bond-days with a daily spread observation')
    tests = [test_partitioned_close_to_joint, test_qC_zero_is_v2, test_recovers_simulated_state,
             test_iekf_reduces_to_plain_filter]
    for t in tests:
        print(f'{t.__name__}', flush=True)
        t(S)
        print('  ok', flush=True)
    print(f'{len(tests)} tests passed')


if __name__ == '__main__':
    main()
