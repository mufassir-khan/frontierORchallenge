"""Crew rotation selection with exact-cover search and a presolved binary model.

Only standard Python, NumPy and SciPy are required.  The returned decisions are
checked against every coverage and base row; a solver status is not treated as
proof of feasibility.  A time-limited feasible incumbent need not be optimal.
"""
from __future__ import annotations

import math
import time
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix, coo_matrix, vstack


class RotationProblem:
    def __init__(self, data):
        dimensions = data['dimensions']
        self.rows = int(dimensions['num_rows'])
        self.count = int(dimensions['num_cols'])
        self.cost = np.asarray(data['cost_vector'][:self.count], dtype=float)
        if self.rows < 1 or self.count < 1 or self.cost.shape != (self.count,):
            raise ValueError('Invalid rotation dimensions')
        if not np.isfinite(self.cost).all():
            raise ValueError('Rotation costs must be finite')
        raw = data['constraint_matrix_A']['columns']
        if len(raw) < self.count:
            raise ValueError('Coverage columns are missing')
        self.cover = [tuple(sorted({int(r) for r in col if 0 <= r < self.rows}))
                      for col in raw[:self.count]]
        self.base = np.empty((0, self.count))
        self.base_lower = np.empty(0)
        self.base_upper = np.empty(0)
        if data.get('has_base_constraints') and data.get('base_constraints'):
            spec = data['base_constraints']
            size = int(spec['num_constraints'])
            if size:
                self.base = np.asarray(spec['D_matrix']['rows'][:size], dtype=float)[:, :self.count]
                self.base_lower = np.asarray(spec['lower_bounds_d1'][:size], dtype=float)
                self.base_upper = np.asarray(spec['upper_bounds_d2'][:size], dtype=float)
                if self.base.shape != (size, self.count):
                    raise ValueError('Invalid base constraint matrix')
                if not np.isfinite(self.base).all() or not np.isfinite(self.base_lower).all() or not np.isfinite(self.base_upper).all():
                    raise ValueError('Nonfinite base constraint')
        # Identical coverage/base columns are interchangeable in a partition.
        # Keep the cheaper one, except empty columns which may both be useful.
        cheapest = {}
        keep_empty = []
        for j, coverage in enumerate(self.cover):
            if not coverage:
                keep_empty.append(j)
                continue
            signature = (coverage, tuple(self.base[:, j]))
            previous = cheapest.get(signature)
            if previous is None or self.cost[j] < self.cost[previous]:
                cheapest[signature] = j
        self.active = sorted(list(cheapest.values()) + keep_empty)
        self.masks = [sum(1 << r for r in self.cover[j]) for j in self.active]
        self.by_row = [[] for _ in range(self.rows)]
        for local, j in enumerate(self.active):
            for r in self.cover[j]:
                self.by_row[r].append(local)
        rr, cc = [], []
        for local, j in enumerate(self.active):
            rr.extend(self.cover[j]); cc.extend([local] * len(self.cover[j]))
        self.matrix = coo_matrix((np.ones(len(rr)), (rr, cc)),
                                 shape=(self.rows, len(self.active))).tocsc()
        self.lower = np.ones(self.rows)
        self.upper = np.ones(self.rows)
        if len(self.base_lower):
            self.matrix = vstack((self.matrix, csc_matrix(self.base[:, self.active])), format='csc')
            self.lower = np.r_[self.lower, self.base_lower]
            self.upper = np.r_[self.upper, self.base_upper]

    def valid(self, selected):
        x = np.zeros(len(self.active))
        x[selected] = 1
        lhs = self.matrix @ x
        return bool(np.all(lhs >= self.lower - 1e-7) and np.all(lhs <= self.upper + 1e-7))

    def seed_cover(self, deadline):
        """Bounded Algorithm X: branch on the most constrained uncovered leg."""
        all_rows = (1 << self.rows) - 1
        costs = self.cost[self.active]
        visits = 0
        answer = None
        has_base = bool(len(self.base_lower))
        base = self.base[:, self.active]

        def search(covered, chosen, base_total):
            nonlocal visits, answer
            visits += 1
            if visits % 32 == 0 and time.monotonic() >= deadline:
                return True
            if covered == all_rows:
                if self.valid(chosen):
                    answer = chosen[:]
                    return True
                return False
            remaining = all_rows ^ covered
            options = None
            while remaining:
                bit = remaining & -remaining
                row = bit.bit_length() - 1
                compatible = [j for j in self.by_row[row] if not (self.masks[j] & covered)]
                if not compatible:
                    return False
                if options is None or len(compatible) < len(options):
                    options = compatible
                    if len(options) == 1:
                        break
                remaining ^= bit
            options.sort(key=lambda j: (costs[j] / max(1, self.masks[j].bit_count()), costs[j]))
            for j in options:
                if time.monotonic() >= deadline:
                    return True
                updated = base_total + base[:, j] if has_base else base_total
                if has_base and np.all(base >= 0) and np.any(updated > self.base_upper + 1e-7):
                    continue
                if search(covered | self.masks[j], chosen + [j], updated):
                    return True
            return False

        # Python recursion is inappropriate for extremely many singleton legs.
        if self.rows < 700 and all(self.by_row):
            search(0, [], np.zeros(len(self.base_lower)))
        return answer

    def output(self, selected):
        if not self.valid(selected):
            raise RuntimeError('No valid exact cover was found within the time limit')
        chosen = sorted(self.active[j] for j in selected)
        chosen_set = set(chosen)
        return {'objective_value': float(sum(self.cost[j] for j in chosen)),
                'selected_rotations': chosen,
                'variable_values': {str(j): float(j in chosen_set) for j in range(self.count)}}


def solve(instance, time_limit=56.0, seed=0):
    start = time.monotonic()
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError('time_limit must be positive and finite')
    deadline = start + time_limit
    problem = RotationProblem(instance)
    chosen = problem.seed_cover(min(deadline, start + min(2.0, time_limit * 0.08)))
    objective = problem.cost[problem.active]
    best_cost = float(objective[chosen].sum()) if chosen is not None else math.inf
    matrix, lower, upper = problem.matrix, problem.lower, problem.upper
    if chosen is not None:
        # The cover is preserved outside HiGHS, so no incumbent is lost if the
        # optimization terminates before returning an integer solution.
        matrix = vstack((matrix, csc_matrix(objective.reshape(1, -1))), format='csc')
        lower = np.r_[lower, -np.inf]
        upper = np.r_[upper, best_cost]
    remaining = deadline - time.monotonic()
    if remaining > 0.02:
        result = milp(objective, integrality=np.ones(len(problem.active)), bounds=Bounds(0, 1),
                      constraints=LinearConstraint(matrix, lower, upper),
                      options={'time_limit': remaining * 0.985, 'mip_rel_gap': 0.0, 'presolve': True})
        if result.x is not None:
            candidate = np.flatnonzero(result.x > 0.5).tolist()
            value = float(objective[candidate].sum())
            if value <= best_cost + 1e-6 and problem.valid(candidate):
                chosen = candidate
    if chosen is None:
        raise RuntimeError('No feasible crew rotation selection found; infeasibility is not proven')
    return problem.output(chosen)
