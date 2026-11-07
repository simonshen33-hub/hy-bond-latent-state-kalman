"""test_ssm.py -- code-correctness checks for ssm.py / model_v1.py / simulate.py / estimate.py.

These tests check algebra and plumbing only. They draw arbitrary random systems from a seeded
generator and compare against statsmodels' independent Kalman filter/smoother; none of the random
values is a model parameter and nothing here is a simulation-recovery result (that needs the
parameter values of the estimated model, which are withheld).

Run:  python test_ssm.py
"""
from __future__ import annotations

import numpy as np
from statsmodels.tsa.statespace.kalman_smoother import KalmanSmoother

import estimate
import model_v1
import simulate
import ssm


def _rand_pd(rng, k):
    A = rng.standard_normal((k, k))
    return A @ A.T


def _rand_system(rng, n, m, p, tv_c):
    T = np.diag(np.tanh(rng.standard_normal(m)))
    T[np.triu_indices(m, 1)] = 0.0
    c = rng.standard_normal((n, m)) if tv_c else rng.standard_normal(m)
    return dict(Z=rng.standard_normal((p, m)), d=rng.standard_normal(p), H=_rand_pd(rng, p),
                T=T, c=c, RQR=_rand_pd(rng, m))


def _rand_y(rng, n, p):
    y = rng.standard_normal((n, p))
    y[rng.random((n, p)) < rng.random()] = np.nan      # partly missing rows
    y[rng.integers(0, n, size=n // 5)] = np.nan          # some fully missing rows
    return y


def _statsmodels(y, s, a1, P1):
    n, p = y.shape
    m = s['T'].shape[-1]
    kf = KalmanSmoother(k_endog=p, k_states=m, k_posdef=m)
    kf.bind(y.copy())
    kf['design'] = s['Z']
    kf['obs_intercept'] = s['d'][:, None] if s['d'].ndim == 1 else s['d'].T
    kf['obs_cov'] = s['H']
    kf['transition'] = s['T']
    kf['state_intercept'] = s['c'].T if s['c'].ndim == 2 else s['c'][:, None]
    kf['selection'] = np.eye(m)
    kf['state_cov'] = s['RQR']
    kf.initialize_known(a1, P1)
    return kf.smooth()


def test_against_statsmodels():
    rng = np.random.default_rng(20261007)
    for tv_c in (False, True):
        n, m, p = 300, 3, 4
        raw = _rand_system(rng, n, m, p, tv_c)
        sys = ssm.as_system(n_rows=n, **raw)
        a1, P1 = rng.standard_normal(m), _rand_pd(rng, m)
        y = _rand_y(rng, n, p)
        res = ssm.kfilter(y, [0, n], sys, a1, P1)
        sm = _statsmodels(y, raw, a1, P1)
        assert np.allclose(res['ll'][0], sm.llf_obs.sum()), (res['ll'][0], sm.llf_obs.sum())
        assert np.allclose(res['a_filt'], sm.filtered_state.T)
        assert np.allclose(res['P_filt'], np.moveaxis(sm.filtered_state_cov, -1, 0))
        assert np.allclose(res['a_pred'], sm.predicted_state[:, :n].T)
        assert np.allclose(res['P_pred'], np.moveaxis(sm.predicted_state_cov[:, :, :n], -1, 0))
        smo = ssm.smooth(res, [0, n], sys)
        assert np.allclose(smo['a_sm'], sm.smoothed_state.T)
        assert np.allclose(smo['P_sm'], np.moveaxis(sm.smoothed_state_cov, -1, 0))
        ll_only, _ = ssm.loglik(y, [0, n], sys, a1, P1)
        assert np.allclose(ll_only, res['ll'])


def test_panel_is_sum_of_series():
    rng = np.random.default_rng(1)
    m, p = 3, 4
    lens = [50, 0, 120, 7]
    n = sum(lens)
    raw = _rand_system(rng, n, m, p, tv_c=True)
    sys = ssm.as_system(n_rows=n, **raw)
    ptr = np.concatenate([[0], np.cumsum(lens)])
    a1 = rng.standard_normal((len(lens), m))
    P1 = np.stack([_rand_pd(rng, m) for _ in lens])
    y = _rand_y(rng, n, p)
    ll, nobs = ssm.loglik(y, ptr, sys, a1, P1)
    for b in range(len(lens)):
        g0, g1 = ptr[b], ptr[b + 1]
        sub = ssm.as_system(n_rows=g1 - g0, Z=raw['Z'], d=raw['d'], H=raw['H'], T=raw['T'],
                            c=raw['c'][g0:g1] if g1 > g0 else raw['c'][:1], RQR=raw['RQR']) if g1 > g0 else None
        if sub is None:
            assert ll[b] == 0.0 and nobs[b] == 0
            continue
        llb, nb = ssm.loglik(y[g0:g1], [0, g1 - g0], sub, a1[b], P1[b])
        assert np.allclose(llb[0], ll[b]) and nb[0] == nobs[b]
    assert nobs.sum() == np.isfinite(y).sum()


def test_missing_row_is_prediction_only():
    rng = np.random.default_rng(2)
    n, m, p = 40, 3, 4
    raw = _rand_system(rng, n, m, p, tv_c=False)
    sys = ssm.as_system(n_rows=n, **raw)
    y = _rand_y(rng, n, p)
    y[10] = np.nan
    res = ssm.kfilter(y, [0, n], sys, np.zeros(m), _rand_pd(rng, m))
    assert np.allclose(res['a_filt'][10], res['a_pred'][10])
    assert np.allclose(res['P_filt'][10], res['P_pred'][10])
    assert np.all(np.isnan(res['v'][10]))


def test_stationary():
    rng = np.random.default_rng(3)
    m = 3
    T = np.diag(np.tanh(rng.standard_normal(m)))
    c, RQR = rng.standard_normal(m), _rand_pd(rng, m)
    a, P = ssm.stationary(T, c, RQR)
    assert np.allclose(a, T @ a + c)
    assert np.allclose(P, T @ P @ T.T + RQR)
    try:
        ssm.stationary(np.eye(m), c, RQR)
        raise AssertionError('unit root accepted')
    except ValueError:
        pass


def _rand_nat(rng, spec):
    """Arbitrary admissible parameter dict for plumbing tests (not a model calibration)."""
    nat = {k: float(rng.standard_normal()) for k in ('mu_C', 'mu_L', 'mu_D', 'lam_D', 'gamma')}
    for s in model_v1.STATES:
        nat[f'phi_{s}'] = float(np.tanh(rng.standard_normal()))
    Q = _rand_pd(rng, 3)
    H = _rand_pd(rng, len(spec.rows))
    if not spec.q_full:
        Q = np.diag(np.diag(Q))
    if not spec.h_full:
        H = np.diag(np.diag(H))
    for i, s in enumerate(model_v1.STATES):
        nat[f'Q_{s}'] = Q[i, i]
    if spec.q_full:
        nat.update(Q_CL=Q[0, 1], Q_CD=Q[0, 2], Q_LD=Q[1, 2])
    for i, r in enumerate(spec.rows):
        nat[f'H_{r}'] = H[i, i]
        if spec.h_full:
            for j in range(i):
                nat[f'H_{r}{spec.rows[j]}'] = H[i, j]
    nat.update(spec.fixed)
    return nat


def test_transform_roundtrip():
    rng = np.random.default_rng(4)
    for q_full in (False, True):
        for h_full in (False, True):
            for rows in (('s', 'C', 'L', 'D'), ('s', 'C', 'L'), ('C', 's')):
                fixed = {} if h_full else {f'H_{rows[0]}': 1.5}
                spec = model_v1.Spec(rows=rows, q_full=q_full, h_full=h_full, init_of=0.3, fixed=fixed)
                nat = _rand_nat(rng, spec)
                model_v1.check_nat(nat, spec)
                back = model_v1.from_u(model_v1.to_u(nat, spec), spec)
                assert set(back) == set(nat)
                for k in nat:
                    assert np.isclose(back[k], nat[k]), (k, back[k], nat[k])
                assert np.allclose(model_v1.q_matrix(back), model_v1.q_matrix(nat))
                assert np.allclose(model_v1.h_matrix(back, rows), model_v1.h_matrix(nat, rows))


def test_model_system_matches_spec():
    """c_t = (I - Phi) mu - gamma OF_t e_D;  Z rows as in the report; stationary init with init_of."""
    rng = np.random.default_rng(5)
    spec = model_v1.Spec(rows=('s', 'C', 'L', 'D'), q_full=True, h_full=True, init_of=0.0)
    nat = _rand_nat(rng, spec)
    of = rng.standard_normal(6)
    sys = model_v1.system(nat, spec, of)
    mu = np.array([nat['mu_C'], nat['mu_L'], nat['mu_D']])
    Phi = np.diag([nat['phi_C'], nat['phi_L'], nat['phi_D']])
    for g in range(6):
        exp_c = (np.eye(3) - Phi) @ mu - nat['gamma'] * of[g] * np.array([0, 0, 1.0])
        assert np.allclose(sys['c'][g], exp_c)
    assert np.allclose(sys['Z'][0], [[1, 1, nat['lam_D']], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    a1, _ = model_v1.initial(nat, spec)
    assert np.allclose(a1, mu)              # init_of = 0 -> stationary mean is mu


def test_d_scale_identification():
    """Without an m_D row, rescaling D by k (mu_D, gamma -> k*; Q_D -> k^2; Q_CD, Q_LD -> k*;
    lam_D -> lam_D/k) leaves the likelihood unchanged: one normalisation is needed. With m_D it changes."""
    rng = np.random.default_rng(7)
    k = float(np.exp(rng.standard_normal()))
    for rows, same in ((('s', 'C', 'L'), True), (('s', 'C', 'L', 'D'), False)):
        spec = model_v1.Spec(rows=rows, q_full=True, h_full=False, init_of=float(rng.standard_normal()))
        nat = _rand_nat(rng, spec)
        ns, nt = 3, 80
        OF = rng.standard_normal((ns, nt))
        _, y = simulate.simulate(nat, spec, OF, rng.random((ns, nt, len(rows))) < rng.random(), rng)
        yf, of_rows, ptr = simulate.to_panel(y, OF)
        alt = dict(nat, mu_D=k * nat['mu_D'], gamma=k * nat['gamma'], Q_D=k * k * nat['Q_D'],
                   Q_CD=k * nat['Q_CD'], Q_LD=k * nat['Q_LD'], lam_D=nat['lam_D'] / k)
        ll0, _ = estimate.panel_loglik(nat, spec, yf, ptr, of_rows)
        ll1, _ = estimate.panel_loglik(alt, spec, yf, ptr, of_rows)
        assert np.isclose(ll0, ll1) == same, (rows, ll0, ll1)


def test_simulate_and_fit_plumbing():
    """simulate -> panel -> loglik -> fit runs end to end; the optimiser never ends below its start."""
    rng = np.random.default_rng(6)
    spec = model_v1.Spec(rows=('s', 'C', 'L', 'D'), q_full=False, h_full=False, init_of=0.0)
    nat = _rand_nat(rng, spec)
    ns, nt = 4, 60
    OF = rng.standard_normal((ns, nt))
    mask = rng.random((ns, nt, 4)) < rng.random()
    alpha, y = simulate.simulate(nat, spec, OF, mask, rng)
    assert alpha.shape == (ns, nt, 3) and y.shape == (ns, nt, 4)
    assert np.all(np.isnan(y[~mask])) and np.all(np.isfinite(y[mask]))
    yf, of_rows, ptr = simulate.to_panel(y, OF)
    ll0, nobs = estimate.panel_loglik(nat, spec, yf, ptr, of_rows)
    assert np.isfinite(ll0) and nobs == mask.sum()
    out = estimate.fit(spec, yf, ptr, of_rows, start_nat=nat, se=False)
    assert out['ll'] >= ll0


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for t in tests:
        t()
        print(f'ok  {t.__name__}')
    print(f'{len(tests)} tests passed')
