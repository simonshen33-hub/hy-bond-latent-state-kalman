"""curve.py -- daily Treasury zero curves from FRED constant-maturity par yields.

Method:
  1. par yields at the 11 FRED tenors are interpolated to a semiannual grid 0.5, 1.0, ..., 30 years
     (method 'pchip' = monotone cubic Hermite, or 'linear' for the robustness check);
  2. bills (1M, 3M, 6M) are taken as zero rates; from 1.0y the grid is bootstrapped as par bonds
     with semiannual coupons:  DF_n = (1 - c_n/2 * sum_{k<n} DF_k) / (1 + c_n/2);
  3. zero rates use semiannual compounding: z(t) = 2 (DF(t)^(-1/(2t)) - 1);
  4. between zero nodes (1/12, 1/4, 1/2, 1.0, 1.5, ..., 30) the same method interpolates; outside
     [1/12, 30] the nearest node is held flat.
Days without a curve (Treasury market closed) take the previous available curve (past information only).

Public API
----------
load_fred(folder) -> DataFrame  index = date, columns = TENORS (decimal yields), holidays dropped
zero_nodes(par_row, method) -> (x, z)                    zero curve nodes for one day
build(fred, method) -> ZeroCurves                        piecewise-cubic coefficients for every day
ZeroCurves.index_for(dates) -> int array                 row of the curve to use (previous if missing)
ZeroCurves.zero(row, t) -> z                             python evaluator (numba kernel in bonds.py)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

SERIES = ['DGS1MO', 'DGS3MO', 'DGS6MO', 'DGS1', 'DGS2', 'DGS3', 'DGS5', 'DGS7', 'DGS10', 'DGS20', 'DGS30']
TENORS = np.array([1 / 12, 3 / 12, 6 / 12, 1, 2, 3, 5, 7, 10, 20, 30], dtype=float)
GRID = np.arange(1, 61) * 0.5                       # 0.5 .. 30.0 years, semiannual bootstrap grid
NODES = np.concatenate([TENORS[:2], GRID])          # 1/12, 1/4, 0.5, 1.0, ..., 30.0


def load_fred(folder):
    folder = Path(folder)
    out = None
    for s in SERIES:
        d = pd.read_csv(folder / f'{s}.csv')
        d.columns = ['date', s]
        d['date'] = pd.to_datetime(d['date'])
        d[s] = pd.to_numeric(d[s], errors='coerce') / 100.0
        out = d if out is None else out.merge(d, on='date', how='outer')
    out = out.sort_values('date').set_index('date')
    full = out.notna().all(axis=1)
    part = out.notna().any(axis=1) & ~full
    if part.any():
        raise ValueError(f'{int(part.sum())} days with only some tenors -- not handled')
    return out[full]


def _interp(method, x, y):
    if method == 'pchip':
        return PchipInterpolator(x, y, extrapolate=False)
    if method == 'linear':
        return lambda t: np.interp(t, x, y)
    raise ValueError(f'unknown method {method!r}')


def zero_nodes(par_row, method):
    """Zero rates (semiannual compounding) at NODES from one day of par yields at TENORS."""
    par = np.asarray(par_row, dtype=float)
    c = _interp(method, TENORS, par)(GRID)
    df = np.empty(GRID.size)
    df[0] = 1.0 / (1.0 + par[2] / 2.0)               # 6M bill as a zero rate
    run = df[0]
    for n in range(1, GRID.size):
        df[n] = (1.0 - c[n] / 2.0 * run) / (1.0 + c[n] / 2.0)
        run += df[n]
    if np.any(df <= 0):
        raise ValueError('non-positive discount factor in bootstrap')
    zg = 2.0 * (df ** (-1.0 / (2.0 * GRID)) - 1.0)
    return NODES.copy(), np.concatenate([par[:2], zg])


@dataclass
class ZeroCurves:
    dates: np.ndarray        # datetime64[ns], ascending (days with a curve)
    x: np.ndarray            # (n_nodes,) node times in years
    coef: np.ndarray         # (n_days, 4, n_nodes-1) cubic coefficients per segment (descending powers)
    z_nodes: np.ndarray      # (n_days, n_nodes) zero rates at the nodes
    method: str

    def index_for(self, dates):
        """Row of the curve for each date: the same day, else the previous available day."""
        d = np.asarray(pd.to_datetime(dates), dtype='datetime64[ns]')
        i = np.searchsorted(self.dates, d, side='right') - 1
        if np.any(i < 0):
            raise ValueError('date before the first curve')
        return i

    def zero(self, row, t):
        return _eval_py(self.x, self.coef[row], self.z_nodes[row], np.asarray(t, dtype=float))


def _eval_py(x, coef, zn, t):
    t = np.clip(t, x[0], x[-1])
    j = np.clip(np.searchsorted(x, t, side='right') - 1, 0, x.size - 2)
    h = t - x[j]
    return ((coef[0, j] * h + coef[1, j]) * h + coef[2, j]) * h + coef[3, j]


def build(fred, method):
    n = len(fred)
    coef = np.empty((n, 4, NODES.size - 1))
    zn = np.empty((n, NODES.size))
    for r, row in enumerate(fred.to_numpy()):
        x, z = zero_nodes(row, method)
        zn[r] = z
        if method == 'pchip':
            coef[r] = PchipInterpolator(x, z).c
        else:
            slope = np.diff(z) / np.diff(x)
            coef[r] = np.stack([np.zeros_like(slope), np.zeros_like(slope), slope, z[:-1]])
    return ZeroCurves(dates=fred.index.to_numpy(dtype='datetime64[ns]'), x=NODES.copy(), coef=coef,
                      z_nodes=zn, method=method)
