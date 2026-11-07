"""test_bonds.py -- checks for curve.py / bonds.py (definitions, identities, round trips).

Random prices / dates come from a seeded generator and only exercise the algebra.
Run:  python test_bonds.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import bonds
import curve

HERE = Path(__file__).parent
FIELDS = HERE / 'data' / 'bloomberg' / 'bloomberg_fields.csv'
FRED = HERE / 'data' / 'fred'


def test_daycount_definitions():
    dc = bonds._dc
    # US bond basis
    assert dc(2026, 1, 31, 2026, 3, 31, False, False) == 60
    assert dc(2026, 1, 30, 2026, 2, 28, False, False) == 28
    assert dc(2026, 2, 28, 2026, 8, 31, False, True) == 180      # EOM Feb rule
    assert dc(2026, 2, 28, 2026, 8, 31, False, False) == 183     # no EOM rule: d2=31 stays
    # 30E/360
    assert dc(2026, 2, 28, 2026, 8, 31, True, True) == 182
    assert dc(2026, 1, 31, 2026, 3, 31, True, False) == 60


def test_schedules():
    d = bonds.coupon_dates(pd.Timestamp('2024-07-15'), pd.Timestamp('2029-01-15'), 2, True)
    assert len(d) == 10 and d[0] == pd.Timestamp('2024-07-15') and d[-1] == pd.Timestamp('2029-01-15')
    assert all((b - a).days in (181, 182, 183, 184) for a, b in zip(d[:-1], d[1:]))
    d = bonds.coupon_dates(pd.Timestamp('2025-08-31'), pd.Timestamp('2030-02-28'), 2, True)
    assert all(x == x + pd.offsets.MonthEnd(0) for x in d)                     # month ends
    assert pd.Timestamp('2028-02-29') in d
    d = bonds.coupon_dates(pd.Timestamp('2025-08-28'), pd.Timestamp('2030-02-28'), 2, False)
    assert pd.Timestamp('2027-08-28') in d                                      # no EOM roll


def test_flat_par_curve_gives_flat_zero():
    for r in (0.01, 0.045, 0.08):
        x, z = curve.zero_nodes(np.full(11, r), 'pchip')
        assert np.allclose(z, r, atol=1e-12)


def test_bootstrap_reprices_par_bonds():
    fred = curve.load_fred(FRED)
    rng = np.random.default_rng(0)
    for row in fred.iloc[rng.integers(0, len(fred), 5)].to_numpy():
        for method in ('pchip', 'linear'):
            x, z = curve.zero_nodes(row, method)
            par = curve._interp(method, curve.TENORS, row)(curve.GRID)
            zg = z[2:]
            df = (1 + zg / 2) ** (-2 * curve.GRID)
            for n in range(1, curve.GRID.size):
                assert abs(par[n] / 2 * df[:n + 1].sum() + df[n] - 1.0) < 1e-12


def _sample_trades(B, rng, n):
    ok = np.flatnonzero(B.ok)
    b = rng.choice(ok, n)
    mat = B.terms['maturity'].to_numpy()[b]
    iss = B.terms['issue'].to_numpy()[b]
    lo = np.maximum(iss.astype('datetime64[D]').astype(np.int64), np.datetime64('2017-07-01', 'D').astype(np.int64))
    hi = np.minimum(mat.astype('datetime64[D]').astype(np.int64) - 30, np.datetime64('2026-09-30', 'D').astype(np.int64))
    keep = hi > lo
    b, lo, hi = b[keep], lo[keep], hi[keep]
    settle = pd.to_datetime((lo + (rng.random(b.size) * (hi - lo)).astype(np.int64)).astype('datetime64[D]'))
    clean = 40 + rng.random(b.size) * 75
    return b, settle, clean


def test_round_trips():
    B = bonds.build_bonds(bonds.load_terms(FIELDS))
    fred = curve.load_fred(FRED)
    rng = np.random.default_rng(1)
    b, settle, clean = _sample_trades(B, rng, 4000)
    y, acc = bonds.ytm(B, b, settle, clean)
    back, _ = bonds.price_from_ytm(B, b, settle, y)
    assert np.all(np.isfinite(y)) and np.max(np.abs(back - clean)) < 1e-8
    for method in ('pchip', 'linear'):
        zc = curve.build(fred, method)
        row = zc.index_for(settle)
        z, _ = bonds.zspread(B, b, settle, clean, zc, row)
        back, _ = bonds.price_from_z(B, b, settle, z, zc, row)
        assert np.all(np.isfinite(z)) and np.max(np.abs(back - clean)) < 1e-8


def test_flat_curve_z_equals_ytm_minus_rate():
    """Flat zero curve r, semiannual bonds: Z = ytm - r exactly (same discounting)."""
    B = bonds.build_bonds(bonds.load_terms(FIELDS))
    rng = np.random.default_rng(2)
    b, settle, clean = _sample_trades(B, rng, 2000)
    semi = B.freq[b] == 2
    b, settle, clean = b[semi], settle[semi], clean[semi]
    r = 0.04
    fred = pd.DataFrame(np.full((1, 11), r), index=pd.DatetimeIndex(['2017-01-03']), columns=curve.SERIES)
    zc = curve.build(fred, 'pchip')
    z, _ = bonds.zspread(B, b, settle, clean, zc, np.zeros(b.size, dtype=np.int64))
    y, _ = bonds.ytm(B, b, settle, clean)
    assert np.max(np.abs(z - (y - r))) < 1e-10


def test_accrued_on_and_after_coupon_date():
    B = bonds.build_bonds(bonds.load_terms(FIELDS))
    b = int(np.flatnonzero(B.ok)[0])
    g = B.ptr[b] + 2                                      # a regular coupon date
    cd = pd.Timestamp('1970-01-01') + pd.Timedelta(days=int(B.cf_ord[g]))
    _, acc = bonds.ytm(B, [b, b], pd.DatetimeIndex([cd, cd + pd.Timedelta(days=1)]), [100.0, 100.0])
    assert acc[0] == 0.0
    assert np.isclose(acc[1], B.coupon[b] / 360.0)


def test_settlement_cycle():
    bd = pd.bdate_range('2024-05-20', '2024-06-10')
    s = bonds.settlement(['2024-05-23', '2024-05-24', '2024-05-28'], bd)
    assert list(s) == [pd.Timestamp('2024-05-27'), pd.Timestamp('2024-05-28'), pd.Timestamp('2024-05-29')]


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for t in tests:
        t()
        print(f'ok  {t.__name__}')
    print(f'{len(tests)} tests passed')
