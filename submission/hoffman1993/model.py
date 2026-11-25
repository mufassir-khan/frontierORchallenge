"""Exact set-partitioning MILP with optional weighted crew-base constraints.

Uses SciPy/HiGHS, included in the FrontierOR images. A feasible time-limit
incumbent is accepted; every returned vector is independently validated.
"""
import time
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, vstack


def solve(instance, time_limit=56.0, seed=0):
    deadline = time.monotonic() + time_limit
    m, n = (int(instance['dimensions'][k]) for k in ('num_rows', 'num_cols'))
    costs = np.asarray(instance['cost_vector'][:n], dtype=float)
    if len(costs) != n or n < 1 or m < 1 or not np.isfinite(costs).all():
        raise ValueError('Invalid dimensions or costs')
    rows, cols = [], []
    # Published checker clips stale rows/columns beyond the declared dimensions.
    for j, coverage in enumerate(instance['constraint_matrix_A']['columns'][:n]):
        for i in set(coverage):
            if 0 <= i < m:
                rows.append(i)
                cols.append(j)
    a = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(m, n)).tocsc()
    lower = np.ones(m)
    upper = np.ones(m)
    if instance.get('has_base_constraints') and instance.get('base_constraints'):
        base = instance['base_constraints']
        count = int(base['num_constraints'])
        d = np.asarray(base['D_matrix']['rows'][:count], dtype=float)[:, :n]
        if d.shape != (count, n):
            raise ValueError('Invalid base constraint dimensions')
        a = vstack([a, d], format='csc')
        lower = np.r_[lower, base['lower_bounds_d1'][:count]]
        upper = np.r_[upper, base['upper_bounds_d2'][:count]]
    result = milp(costs, integrality=np.ones(n), bounds=Bounds(0, 1),
                  constraints=LinearConstraint(a, lower, upper),
                  options={'time_limit': max(0.001, deadline - time.monotonic()),
                           'mip_rel_gap': 0.0, 'presolve': True})
    if result.x is None:
        raise RuntimeError('No feasible set partition found: ' + result.message)
    x = np.rint(result.x).astype(int)
    activity = a @ x
    if (np.any((x < 0) | (x > 1)) or np.any(activity < lower - 1e-6)
            or np.any(activity > upper + 1e-6)):
        raise RuntimeError('MILP incumbent failed independent feasibility validation')
    selected = np.flatnonzero(x).tolist()
    return {'objective_value': float(costs @ x), 'selected_rotations': selected,
            'variable_values': {str(j): float(x[j]) for j in range(n)}}
