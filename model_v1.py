"""model_v1.py -- the baseline three-state model as a system for ssm.py.

State alpha_t = (C_t, L_t, D_t)'   (credit, liquidity premium, dealer inventory).
Transition (the D equation is driven by customer order flow):
    alpha_{t+1} = mu + Phi (alpha_t - mu) - gamma OF_t e_D + eps_{t+1},   eps ~ N(0, Q)
 => T = Phi = diag(phi_C, phi_L, phi_D),   c_t = (I - Phi) mu - gamma OF_t e_D,   R = I.
Observation (joint system):
    y_t = (s_obs, m_C, m_L, m_D)' = Z alpha_t + eta_t,   eta ~ N(0, H),   d = 0
    Z = [[1, 1, lam_D],      s_obs = C + L + lam_D D + eta_P
         [1, 0, 0],          m_C   = C + eta_C
         [0, 1, 0],          m_L   = L + eta_L
         [0, 0, 1]]          m_D   = D + eta_D
Initial state: stationary distribution of the transition with OF_t held at spec.init_of
    a_inf = (I-T)^-1 c, P_inf = T P_inf T' + RQR'.

Spec (structure only; this module holds no parameter values):
    rows    : subset/order of ('s', 'C', 'L', 'D') present in the observation vector
    q_full  : Q full (correlated shocks) or diagonal
    h_full  : H full (correlated measurement errors) or diagonal
    fixed   : {scalar name: value} parameters held fixed (e.g. a scale normalisation)
    init_of : order-flow value used in c for the stationary initial mean

Parameter names (natural space):
    mu_C mu_L mu_D  phi_C phi_L phi_D  Q_C Q_L Q_D Q_CL Q_CD Q_LD  lam_D  gamma  H_<r> [H_<r1><r2>]
Unconstrained vector u used by the optimiser:
    mu, lam_D, gamma: identity;  phi = tanh(u) (|phi| < 1);
    Q, H: log variances (diagonal) or Cholesky factor with log diagonal (full).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import ssm

STATES = ('C', 'L', 'D')
ROW_Z = {'s': None, 'C': (1.0, 0.0, 0.0), 'L': (0.0, 1.0, 0.0), 'D': (0.0, 0.0, 1.0)}


@dataclass(frozen=True)
class Spec:
    rows: tuple
    q_full: bool
    h_full: bool
    init_of: float
    fixed: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.rows or len(set(self.rows)) != len(self.rows) or any(r not in ROW_Z for r in self.rows):
            raise ValueError(f'rows must be distinct entries of {tuple(ROW_Z)}')
        bad = set(self.fixed) - set(all_names(self))
        if bad:
            raise ValueError(f'fixed names not in the parameter list: {sorted(bad)}')
        if self.q_full and any(k.startswith('Q_') for k in self.fixed):
            raise ValueError('fixing Q entries is only supported with q_full=False')
        if self.h_full and any(k.startswith('H_') for k in self.fixed):
            raise ValueError('fixing H entries is only supported with h_full=False')


def _q_names(full):
    return ['Q_C', 'Q_L', 'Q_D'] + (['Q_CL', 'Q_CD', 'Q_LD'] if full else [])


def _h_names(rows, full):
    out = [f'H_{r}' for r in rows]
    if full:
        out += [f'H_{rows[i]}{rows[j]}' for i in range(len(rows)) for j in range(i)]
    return out


def all_names(spec):
    """Natural-space scalar names in a fixed order."""
    return (['mu_C', 'mu_L', 'mu_D', 'phi_C', 'phi_L', 'phi_D'] + _q_names(spec.q_full)
            + ['lam_D', 'gamma'] + _h_names(spec.rows, spec.h_full))


def free_names(spec):
    return [n for n in all_names(spec) if n not in spec.fixed]


# ----------------------------------------------------------------------------------------------
# natural <-> matrices
# ----------------------------------------------------------------------------------------------
def q_matrix(nat):
    Q = np.diag([nat['Q_C'], nat['Q_L'], nat['Q_D']]).astype(float)
    for (a, b), (i, j) in {('C', 'L'): (0, 1), ('C', 'D'): (0, 2), ('L', 'D'): (1, 2)}.items():
        k = f'Q_{a}{b}'
        if k in nat:
            Q[i, j] = Q[j, i] = nat[k]
    return Q


def h_matrix(nat, rows):
    p = len(rows)
    H = np.diag([nat[f'H_{r}'] for r in rows]).astype(float)
    for i in range(p):
        for j in range(i):
            k = f'H_{rows[i]}{rows[j]}'
            if k in nat:
                H[i, j] = H[j, i] = nat[k]
    return H


def z_matrix(nat, rows):
    Z = np.zeros((len(rows), 3))
    for i, r in enumerate(rows):
        Z[i] = (1.0, 1.0, nat['lam_D']) if r == 's' else ROW_Z[r]
    return Z


def system(nat, spec, of_rows):
    """ssm system for a panel whose rows carry order flow of_rows (n_rows,); OF on row g drives the
    transition from row g to g+1 (report: D_{t+1} = ... - gamma OF_t)."""
    of_rows = np.asarray(of_rows, dtype=float)
    n = of_rows.shape[0]
    mu = np.array([nat['mu_C'], nat['mu_L'], nat['mu_D']], dtype=float)
    Phi = np.diag([nat['phi_C'], nat['phi_L'], nat['phi_D']]).astype(float)
    c = np.tile((np.eye(3) - Phi) @ mu, (n, 1))
    c[:, 2] -= nat['gamma'] * of_rows
    p = len(spec.rows)
    return ssm.as_system(Z=z_matrix(nat, spec.rows), d=np.zeros(p), H=h_matrix(nat, spec.rows),
                         T=Phi, c=c, RQR=q_matrix(nat), n_rows=n)


def initial(nat, spec):
    """Stationary prior (a_{1|0}, P_{1|0}) with OF held at spec.init_of."""
    mu = np.array([nat['mu_C'], nat['mu_L'], nat['mu_D']], dtype=float)
    Phi = np.diag([nat['phi_C'], nat['phi_L'], nat['phi_D']]).astype(float)
    c = (np.eye(3) - Phi) @ mu
    c[2] -= nat['gamma'] * spec.init_of
    return ssm.stationary(Phi, c, q_matrix(nat))


# ----------------------------------------------------------------------------------------------
# natural <-> unconstrained
# ----------------------------------------------------------------------------------------------
def to_u(nat, spec):
    """Natural dict -> unconstrained vector over free parameters (order of free_names)."""
    out = {}
    for k in ('mu_C', 'mu_L', 'mu_D', 'lam_D', 'gamma'):
        out[k] = float(nat[k])
    for s in STATES:
        phi = float(nat[f'phi_{s}'])
        if not abs(phi) < 1.0:
            raise ValueError(f'phi_{s} must satisfy |phi| < 1')
        out[f'phi_{s}'] = float(np.arctanh(phi))
    out.update(_cov_to_u(q_matrix(nat), list(STATES), 'Q', spec.q_full))
    out.update(_cov_to_u(h_matrix(nat, spec.rows), list(spec.rows), 'H', spec.h_full))
    return np.array([out[k] for k in free_names(spec)])


def from_u(u, spec):
    """Unconstrained vector -> natural dict (fixed values filled in)."""
    names = free_names(spec)
    if len(u) != len(names):
        raise ValueError(f'u has {len(u)} entries, spec has {len(names)} free parameters')
    raw = dict(zip(names, map(float, u)))
    nat = dict(spec.fixed)
    for k in ('mu_C', 'mu_L', 'mu_D', 'lam_D', 'gamma'):
        if k in raw:
            nat[k] = raw[k]
    for s in STATES:
        k = f'phi_{s}'
        if k in raw:
            nat[k] = float(np.tanh(raw[k]))
    nat.update(_u_to_cov(raw, spec.fixed, list(STATES), 'Q', spec.q_full))
    nat.update(_u_to_cov(raw, spec.fixed, list(spec.rows), 'H', spec.h_full))
    return nat


def _off_name(labels, tag, i, j):
    """Name of off-diagonal (i, j), i > j. Q uses the report's names (Q_CL, Q_CD, Q_LD); H uses
    H_<row i><row j> in row order. In u-space the same name holds the Cholesky element L[i, j]."""
    if tag == 'Q':
        a, b = sorted((labels[i], labels[j]), key=STATES.index)
        return f'Q_{a}{b}'
    return f'H_{labels[i]}{labels[j]}'


def _cov_to_u(S, labels, tag, full):
    p = len(labels)
    if not full:
        if np.any(np.diag(S) <= 0):
            raise ValueError(f'{tag}: variances must be positive')
        return {f'{tag}_{labels[i]}': float(np.log(S[i, i])) for i in range(p)}
    L = np.linalg.cholesky(S)
    out = {f'{tag}_{labels[i]}': float(np.log(L[i, i])) for i in range(p)}
    for i in range(p):
        for j in range(i):
            out[_off_name(labels, tag, i, j)] = float(L[i, j])
    return out


def _u_to_cov(raw, fixed, labels, tag, full):
    p = len(labels)
    if not full:
        out = {}
        for i in range(p):
            k = f'{tag}_{labels[i]}'
            out[k] = fixed[k] if k in fixed else float(np.exp(raw[k]))
        return out
    L = np.zeros((p, p))
    for i in range(p):
        L[i, i] = np.exp(raw[f'{tag}_{labels[i]}'])
        for j in range(i):
            L[i, j] = raw[_off_name(labels, tag, i, j)]
    S = L @ L.T
    out = {f'{tag}_{labels[i]}': float(S[i, i]) for i in range(p)}
    for i in range(p):
        for j in range(i):
            out[_off_name(labels, tag, i, j)] = float(S[i, j])
    return out


def check_nat(nat, spec):
    """Raise unless nat holds every parameter of spec with admissible values."""
    missing = [k for k in all_names(spec) if k not in nat]
    if missing:
        raise ValueError(f'missing parameters: {missing}')
    extra = [k for k in nat if k not in all_names(spec)]
    if extra:
        raise ValueError(f'parameters not in this spec: {extra}')
    for s in STATES:
        if not abs(nat[f'phi_{s}']) < 1.0:
            raise ValueError(f'phi_{s} must satisfy |phi| < 1')
    for tag, S in (('Q', q_matrix(nat)), ('H', h_matrix(nat, spec.rows))):
        if np.min(np.linalg.eigvalsh(S)) <= 0:
            raise ValueError(f'{tag} is not positive definite')
    for k, v in spec.fixed.items():
        if nat[k] != v:
            raise ValueError(f'{k} is fixed at {v} but nat has {nat[k]}')
