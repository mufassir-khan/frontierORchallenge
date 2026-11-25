"""Stochastic capacitated facility location with sampled MIP and full recourse.

A sample selects opening decisions, while a robust total-capacity constraint
covers the maximum demand across ALL scenarios. Every scenario is then served
explicitly. Greedy feasible shipments are retained when transport LPs time out.
No scenario is omitted from the output or expected-cost calculation.
"""
import time
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix, eye, kron, vstack


def solve(instance, time_limit=56.0, seed=0):
    started = time.monotonic()
    deadline = started + time_limit
    rng = np.random.default_rng(seed)
    facilities = sorted(instance['facilities'], key=lambda f: f['id'])
    scenarios = sorted(instance['scenarios'], key=lambda s: s['id'])
    f, c, s = (int(instance[k]) for k in ('num_facilities', 'num_customers', 'num_scenarios'))
    if [a['id'] for a in facilities] != list(range(f)) or [a['id'] for a in scenarios] != list(range(s)):
        raise ValueError('Expected zero-based consecutive facility and scenario IDs')
    capacity = np.asarray([a['capacity'] for a in facilities], dtype=float)
    fixed = np.asarray([a['opening_cost'] for a in facilities], dtype=float)
    transport = np.asarray(instance['transportation_costs'], dtype=float)
    demands = np.asarray([a['demands'] for a in scenarios], dtype=float)
    probability = np.asarray([a['probability'] for a in scenarios], dtype=float)
    if transport.shape != (f, c) or demands.shape != (s, c):
        raise ValueError('Invalid transport or scenario dimensions')
    if any(not np.isfinite(a).all() for a in (capacity, fixed, transport, demands, probability)):
        raise ValueError('Nonfinite instance values')
    if min(capacity.min(), transport.min(), demands.min(), probability.min()) < 0:
        raise ValueError('Capacities, costs, demands and probabilities must be nonnegative')
    maximum = float(demands.sum(axis=1).max())
    if capacity.sum() < maximum - 1e-7:
        raise ValueError('Total facility capacity cannot serve all scenarios')
    edge_order = sorted((float(transport[i, j]), i, j) for i in range(f) for j in range(c))

    def greedy(opened, demand):
        remaining = demand.copy()
        available = capacity * opened
        allocation = np.zeros((f, c))
        for _, i, j in edge_order:
            amount = min(available[i], remaining[j])
            if amount > 0:
                allocation[i, j] = amount
                available[i] -= amount
                remaining[j] -= amount
        if remaining.max() > 1e-6:
            raise RuntimeError('Could not construct feasible scenario shipments')
        return allocation

    sample_count = min(s, max(4, min(20, 20000 // max(1, f*c))))
    weights = probability / probability.sum() if probability.sum() > 0 else np.full(s, 1/s)
    sample = rng.choice(s, size=sample_count, replace=False, p=weights if np.count_nonzero(weights) >= sample_count else None)
    # Include an extreme scenario as well as representative random scenarios.
    sample = np.unique(np.r_[sample, int(np.argmax(demands.sum(axis=1)))])
    sample_weights = probability[sample].copy()
    if sample_weights.sum() == 0:
        sample_weights[:] = 1
    sample_weights *= probability.sum() / sample_weights.sum()

    def estimate(opened):
        return float(fixed @ opened + sum(w * np.sum(transport * greedy(opened, demands[k]))
                                         for k, w in zip(sample, sample_weights)))

    opened = np.ones(f, dtype=int)
    estimate_value = estimate(opened)
    # Fast capacity-preserving closure search supplies a useful incumbent even
    # when the sampled MILP cannot find an integer solution in its time budget.
    for _ in range(f):
        chosen = None
        for i in rng.permutation(f):
            if time.monotonic() > started + time_limit * 0.12:
                break
            if not opened[i] or capacity @ opened - capacity[i] < maximum - 1e-8:
                continue
            candidate = opened.copy(); candidate[i] = 0
            value = estimate(candidate)
            if value < estimate_value - 1e-7:
                chosen, estimate_value = candidate, value
        if chosen is None:
            break
        opened = chosen
    candidates = [np.ones(f, dtype=int), opened.copy()]
    z = len(sample)
    variable_count = f + z*f*c
    objective = np.r_[fixed, np.concatenate([w * transport.ravel() for w in sample_weights])]
    rr, cc, vv, low, high = [], [], [], [], []

    def row(terms, lb, ub):
        r = len(low); low.append(lb); high.append(ub)
        for col, value in terms:
            rr.append(r); cc.append(col); vv.append(value)

    row(enumerate(capacity), maximum, np.inf)
    for t, k in enumerate(sample):
        offset = f + t*f*c
        for j in range(c):
            row(((offset+i*c+j, 1) for i in range(f)), demands[k, j], demands[k, j])
        for i in range(f):
            row([(i, -capacity[i])] + [(offset+i*c+j, 1) for j in range(c)], -np.inf, 0)
    matrix = coo_matrix((vv, (rr, cc)), shape=(len(low), variable_count)).tocsc()
    mip_end = started + time_limit * 0.52
    if time.monotonic() < mip_end:
        result = milp(objective, integrality=np.r_[np.ones(f), np.zeros(z*f*c)],
                      bounds=Bounds(0, np.r_[np.ones(f), np.full(z*f*c, np.inf)]),
                      constraints=LinearConstraint(matrix, low, high),
                      options={'time_limit': max(0.001, mip_end-time.monotonic()), 'mip_rel_gap': 0.005})
        if result.x is not None:
            candidate = (result.x[:f] > 0.5).astype(int)
            if capacity @ candidate >= maximum - 1e-7:
                candidates.append(candidate)
    # Compare openings on every actual scenario, using the same allocation rule.
    best_value, allocations = float('inf'), None
    for candidate in candidates:
        trial = [greedy(candidate, demand) for demand in demands]
        value = float(fixed @ candidate + sum(probability[k] * np.sum(transport * y) for k, y in enumerate(trial)))
        if value < best_value:
            opened, allocations, best_value = candidate.copy(), trial, value
    # Transportation LP: each customer's demand is equality, each facility has
    # its own capacity. Sparse y below keeps even 1500-scenario outputs compact.
    supply = kron(eye(f, format='csc'), np.ones((1, c)), format='csc')
    demand_matrix = kron(np.ones((1, f)), eye(c, format='csc'), format='csc')
    for k in np.argsort(-probability):
        remaining = deadline - time.monotonic()
        if remaining < 0.1:
            break
        lp = linprog(transport.ravel(), A_ub=supply, b_ub=capacity * opened,
                     A_eq=demand_matrix, b_eq=demands[k], bounds=(0, None), method='highs',
                     options={'time_limit': min(remaining, max(0.05, time_limit*0.025))})
        if lp.x is not None:
            y = np.maximum(0, lp.x.reshape(f, c))
            if (np.all(y.sum(axis=0) >= demands[k]-1e-7) and
                    np.all(y.sum(axis=1) <= capacity*opened+1e-7) and
                    np.sum(y*transport) <= np.sum(allocations[k]*transport) + 1e-7):
                allocations[k] = y
    output = {}
    expected_cost = float(fixed @ opened)
    for k, y in enumerate(allocations):
        if np.any(y.sum(axis=0) < demands[k]-1e-6) or np.any(y.sum(axis=1) > capacity*opened+1e-6):
            raise RuntimeError('Final shipments failed validation')
        expected_cost += float(probability[k] * np.sum(transport*y))
        scenario_output = {}
        for i in range(f):
            positive = np.flatnonzero(y[i] > 0)
            if len(positive):
                scenario_output[str(i)] = {str(int(j)): float(y[i,j]) for j in positive}
        output[str(k)] = scenario_output
    return {'objective_value': expected_cost, 'open_facilities': np.flatnonzero(opened).tolist(),
            'x': {str(i): int(opened[i]) for i in range(f)}, 'y': output}
