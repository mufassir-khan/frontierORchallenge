"""Facility opening by Benders cuts, followed by complete scenario recourse.

The binary master contains facility decisions and scenario recourse estimates.
Transport-LP duals supply supporting cuts. For a large scenario set, weighted
strata provide a reduced planning sample; EVERY original scenario receives an
explicit feasible shipment plan and contributes to the reported objective.
Requires NumPy/SciPy, both present in the FrontierOR execution images.
"""
from __future__ import annotations

import math
import time
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix, eye, kron


class FacilityProblem:
    def __init__(self, data):
        self.f = int(data['num_facilities'])
        self.c = int(data['num_customers'])
        self.s = int(data['num_scenarios'])
        facilities = sorted(data['facilities'], key=lambda x: x['id'])
        scenarios = sorted(data['scenarios'], key=lambda x: x['id'])
        if self.f < 1 or self.c < 1 or self.s < 1:
            raise ValueError('Facility, customer and scenario counts must be positive')
        if [x['id'] for x in facilities] != list(range(self.f)) or [x['id'] for x in scenarios] != list(range(self.s)):
            raise ValueError('Expected consecutive zero-based facility and scenario IDs')
        self.capacity = np.asarray([x['capacity'] for x in facilities], dtype=float)
        self.fixed = np.asarray([x['opening_cost'] for x in facilities], dtype=float)
        self.cost = np.asarray(data['transportation_costs'], dtype=float)
        self.demand = np.asarray([x['demands'] for x in scenarios], dtype=float)
        self.probability = np.asarray([x['probability'] for x in scenarios], dtype=float)
        if self.cost.shape != (self.f, self.c) or self.demand.shape != (self.s, self.c):
            raise ValueError('Invalid transport/scenario dimensions')
        for array in (self.capacity, self.fixed, self.cost, self.demand, self.probability):
            if not np.isfinite(array).all():
                raise ValueError('Nonfinite facility-location data')
        if any(np.any(a < 0) for a in (self.capacity, self.cost, self.demand, self.probability)):
            raise ValueError('Negative capacity, demand, probability or transport cost')
        self.required = float(self.demand.sum(axis=1).max())
        if self.capacity.sum() < self.required - 1e-8:
            raise ValueError('Total capacity is insufficient for at least one scenario')
        self.supply_rows = kron(eye(self.f, format='csc'), np.ones((1, self.c)), format='csc')
        self.customer_rows = kron(np.ones((1, self.f)), eye(self.c, format='csc'), format='csc')
        # Cheapest arcs are shared by all scenarios, avoiding repeated sorting.
        self.arc_order = np.argsort(self.cost.ravel(), kind='stable').tolist()

    def greedy(self, opened, scenario):
        supply = self.capacity * opened
        unmet = self.demand[scenario].copy()
        y = np.zeros((self.f, self.c))
        for arc in self.arc_order:
            i, j = divmod(arc, self.c)
            quantity = min(supply[i], unmet[j])
            if quantity > 0:
                y[i, j] = quantity
                supply[i] -= quantity
                unmet[j] -= quantity
        if np.any(unmet > 1e-7):
            raise RuntimeError('Opening pattern does not admit feasible recourse')
        return y

    def transport(self, opened, scenario, seconds):
        result = linprog(self.cost.ravel(), A_ub=self.supply_rows,
                         b_ub=self.capacity * opened, A_eq=self.customer_rows,
                         b_eq=self.demand[scenario], bounds=(0, None), method='highs',
                         options={'time_limit': max(0.001, seconds)})
        if not result.success:
            return None
        allocation = result.x.reshape(self.f, self.c)
        if not self.valid_shipments(opened, scenario, allocation):
            return None
        # Q(x) >= beta*d + alpha*capacity*x, alpha <= 0.
        intercept = float(result.eqlin.marginals @ self.demand[scenario])
        coefficient = result.ineqlin.marginals * self.capacity
        return allocation, intercept, coefficient

    def valid_shipments(self, opened, k, allocation):
        return bool(np.isfinite(allocation).all() and np.all(allocation >= -1e-8)
                    and np.all(allocation.sum(axis=0) >= self.demand[k] - 1e-7)
                    and np.all(allocation.sum(axis=1) <= self.capacity * opened + 1e-7))

    def representatives(self, seed, sample_limit=None):
        """Weighted strata preserve probability mass, including zero-probability inputs."""
        limit = min(self.s, max(6, min(16, 24000 // (self.f * self.c))))
        if sample_limit is not None:
            limit = min(self.s, sample_limit)
        if self.s <= limit:
            return np.arange(self.s), self.probability.copy()
        rng = np.random.default_rng(seed)
        order = np.argsort(self.demand.sum(axis=1), kind='stable')
        groups = np.array_split(order, limit)
        selected, weights = [], []
        for group in groups:
            probability = self.probability[group]
            total = float(probability.sum())
            # Choose a real scenario close to its stratum's weighted mean.
            center = np.average(self.demand[group], axis=0, weights=probability) if total > 0 else self.demand[group].mean(axis=0)
            deviation = np.sum((self.demand[group] - center) ** 2, axis=1)
            best = np.flatnonzero(deviation <= deviation.min() + 1e-10)
            selected.append(int(group[int(rng.choice(best))]))
            weights.append(total)
        return np.asarray(selected), np.asarray(weights)

    def plan_openings(self, deadline, seed, sample_limit=None):
        sample, weights = self.representatives(seed, sample_limit)
        m = len(sample)
        current = np.ones(self.f, dtype=int)
        best, best_value = current.copy(), math.inf
        cuts = []
        seen = set()
        lower_bound = -math.inf
        for iteration in range(250):
            if time.monotonic() >= deadline:
                break
            signature = tuple(current.tolist())
            if signature in seen:
                break
            seen.add(signature)
            exact = True
            value = float(self.fixed @ current)
            for t, k in enumerate(sample):
                remaining = deadline - time.monotonic()
                if remaining < 0.02:
                    exact = False
                    break
                result = self.transport(current, int(k), min(remaining, 1.0))
                if result is None:
                    exact = False
                    break
                y, intercept, coefficient = result
                value += float(weights[t] * np.sum(self.cost * y))
                cuts.append((t, intercept, coefficient))
            if exact and value < best_value:
                best, best_value = current.copy(), value
            if exact and best_value - lower_bound <= 1e-6 * max(1, abs(best_value)):
                break
            remaining = deadline - time.monotonic()
            if remaining < 0.03 or not exact:
                break
            rr, cc, vv = [], [], []
            lower, upper = [self.required], [math.inf]
            for i, cap in enumerate(self.capacity):
                rr.append(0); cc.append(i); vv.append(cap)
            for t, intercept, coefficient in cuts:
                row = len(lower)
                lower.append(-math.inf); upper.append(-intercept)
                for i in np.flatnonzero(coefficient):
                    rr.append(row); cc.append(int(i)); vv.append(float(coefficient[i]))
                rr.append(row); cc.append(self.f+t); vv.append(-1)
            columns = self.f + m
            matrix = coo_matrix((vv, (rr, cc)), shape=(len(lower), columns)).tocsc()
            objective = np.r_[self.fixed, weights]
            answer = milp(objective, integrality=np.r_[np.ones(self.f), np.zeros(m)],
                          bounds=Bounds(0, np.r_[np.ones(self.f), np.full(m, np.inf)]),
                          constraints=LinearConstraint(matrix, lower, upper),
                          options={'time_limit': min(remaining * 0.95, 2.0), 'mip_rel_gap': 0.001})
            bound = getattr(answer, 'mip_dual_bound', None)
            if bound is not None:
                lower_bound = max(lower_bound, float(bound))
            if answer.x is None:
                break
            current = np.rint(answer.x[:self.f]).astype(int)
            if np.any((current != 0) & (current != 1)) or self.capacity @ current < self.required - 1e-7:
                break
        return best

    def serialize(self, opened, shipments):
        if self.capacity @ opened < self.required - 1e-7:
            raise RuntimeError('Final facility capacity validation failed')
        y = {}
        objective = float(self.fixed @ opened)
        for k, allocation in enumerate(shipments):
            if not self.valid_shipments(opened, k, allocation):
                raise RuntimeError(f'Invalid recourse in scenario {k}')
            objective += float(self.probability[k] * np.sum(self.cost * allocation))
            scenario = {}
            for i, row in enumerate(allocation):
                positive = np.flatnonzero(row > 0)
                if len(positive):
                    scenario[str(i)] = {str(int(j)): float(row[j]) for j in positive}
            y[str(k)] = scenario
        return {'objective_value': objective,
                'open_facilities': np.flatnonzero(opened).tolist(),
                'x': {str(i): int(v) for i, v in enumerate(opened)}, 'y': y}


def solve(instance, time_limit=56.0, seed=0):
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError('time_limit must be positive and finite')
    start = time.monotonic()
    deadline = start + time_limit
    problem = FacilityProblem(instance)
    # Reserve the larger share for complete recourse and output construction.
    opened = problem.plan_openings(start + 0.45*time_limit, seed)
    if not np.any(opened):
        # The published output contract requires a nonempty facility witness,
        # including the degenerate case where all scenario demands are zero.
        opened[int(np.argmin(problem.fixed))] = 1
    shipments = [problem.greedy(opened, k) for k in range(problem.s)]
    # Exact transportation is optional; every unoptimized scenario already has
    # a feasible shipment plan. High-probability scenarios get priority.
    for k in np.argsort(-problem.probability, kind='stable'):
        remaining = deadline - time.monotonic()
        if remaining < max(0.05, min(1.5, 0.04*time_limit)):
            break
        result = problem.transport(opened, int(k), min(remaining * 0.8, 0.5))
        if result is not None:
            allocation = np.maximum(result[0], 0)
            if (problem.valid_shipments(opened, k, allocation)
                    and np.sum(problem.cost * allocation) <= np.sum(problem.cost * shipments[k]) + 1e-7):
                shipments[k] = allocation
    incumbent = problem.serialize(opened, shipments)
    # Spend spare time checking a richer scenario approximation. Compare both
    # plans using the actual full scenario distribution, not sampled objectives.
    remaining = deadline - time.monotonic()
    if problem.s > 16 and remaining > 8:
        candidate = problem.plan_openings(time.monotonic() + min(remaining * 0.35, 12),
                                          seed, sample_limit=32)
        if not np.any(candidate):
            candidate[int(np.argmin(problem.fixed))] = 1
        if not np.array_equal(opened, candidate):
            alternative = [problem.greedy(candidate, k) for k in range(problem.s)]
            for k in np.argsort(-problem.probability, kind='stable'):
                remaining = deadline - time.monotonic()
                if remaining < 1.5:
                    break
                result = problem.transport(candidate, int(k), min(remaining * 0.8, 0.5))
                if result is not None:
                    allocation = np.maximum(result[0], 0)
                    if problem.valid_shipments(candidate, k, allocation):
                        alternative[k] = allocation
            proposed = problem.serialize(candidate, alternative)
            if proposed['objective_value'] < incumbent['objective_value']:
                incumbent = proposed
    return incumbent
