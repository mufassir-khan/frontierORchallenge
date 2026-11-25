"""Unsplittable multicommodity routing using path generation and a binary MILP.

A feasible greedy incumbent is always retained. The path master selects at most
one physical path per commodity, with arc capacities; omitted commodities are
rejected at demand * artificial_arc_cost. Paths contain actual physical arc IDs.
"""
import heapq
import math
import random
import time
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


def solve(instance, time_limit=56.0, seed=0):
    started = time.monotonic()
    deadline = started + time_limit
    rng = random.Random(seed)
    arcs = instance['network']['arcs']
    commodities = instance['commodities']['commodity_list']
    nodes = instance['network']['nodes']
    n, k_count = len(arcs), len(commodities)
    capacity = [float(a['capacity']) for a in arcs]
    costs = [float(a.get('cost', 0)) for a in arcs]
    if any(c < 0 for c in capacity + costs):
        raise ValueError('Arc capacities and costs must be nonnegative')
    adjacency = {v: [] for v in nodes}
    for a, edge in enumerate(arcs):
        adjacency[edge['from_node']].append((edge['to_node'], a))
    demand = [float(c['demand']) for c in commodities]
    penalty = [d * float(c['artificial_arc_cost']) for d, c in zip(demand, commodities)]
    pools = [set() for _ in commodities]

    def path(k, available, weights):
        origin, target = commodities[k]['origin'], commodities[k]['destination']
        distance, parent = {origin: 0.0}, {}
        queue = [(0.0, origin)]
        while queue:
            value, u = heapq.heappop(queue)
            if value != distance[u]:
                continue
            if u == target:
                chosen = []
                while u != origin:
                    u, a = parent[u]
                    chosen.append(a)
                return tuple(reversed(chosen))
            for v, a in adjacency[u]:
                if available[a] + 1e-9 < demand[k]:
                    continue
                nxt = value + weights[a]
                if nxt < distance.get(v, math.inf):
                    distance[v], parent[v] = nxt, (u, a)
                    heapq.heappush(queue, (nxt, v))
        return None

    def path_cost(k, route):
        return demand[k] * sum(costs[a] for a in route)

    best = [None] * k_count
    best_cost = sum(penalty)
    iteration = 0
    generation_end = started + min(8.0, time_limit * 0.25)
    while time.monotonic() < generation_end and iteration < 100:
        available = capacity[:]
        selected = [None] * k_count
        jitter = [rng.uniform(0.5, 2) for _ in commodities]
        order = sorted(range(k_count), key=lambda k: -penalty[k] *
                       (1 if iteration == 0 else jitter[k]) / max(1, demand[k]) ** (iteration % 3 / 2))
        for k in order:
            if time.monotonic() >= generation_end:
                break
            weights = [costs[a] * demand[k] +
                       (1 + (0 if iteration == 0 else rng.random() * 2)) *
                       (1 + demand[k] / max(1e-9, available[a])) for a in range(n)]
            route = path(k, available, weights)
            if route is not None:
                pools[k].add(route)
                if path_cost(k, route) <= penalty[k]:
                    selected[k] = route
                    for a in route:
                        available[a] -= demand[k]
        value = sum(penalty[k] if r is None else path_cost(k, r) for k, r in enumerate(selected))
        if value < best_cost:
            best, best_cost = selected, value
        iteration += 1
    # Explore alternatives even for commodities blocked in greedy packings.
    for trial in range(8):
        if time.monotonic() >= started + time_limit * 0.35:
            break
        for k in range(k_count):
            if time.monotonic() >= started + time_limit * 0.35:
                break
            weights = [1 + demand[k] * costs[a] + rng.random() * (trial + 1)
                       + demand[k] / max(capacity[a], 1e-9) for a in range(n)]
            route = path(k, capacity, weights)
            if route is not None:
                pools[k].add(route)
    columns, objective, row, col, val = [], [], [], [], []
    for k, pool in enumerate(pools):
        for route in sorted(pool):
            benefit = path_cost(k, route) - penalty[k]
            if benefit > 0:
                continue
            j = len(columns)
            columns.append((k, route))
            objective.append(benefit)
            row.append(n + k); col.append(j); val.append(1.0)
            for a in route:
                row.append(a); col.append(j); val.append(demand[k])
    if columns and deadline - time.monotonic() > 0.05:
        a = coo_matrix((val, (row, col)), shape=(n + k_count, len(columns))).tocsc()
        result = milp(np.asarray(objective), integrality=np.ones(len(columns)),
                      bounds=Bounds(0, 1),
                      constraints=LinearConstraint(a, -np.inf, capacity + [1] * k_count),
                      options={'time_limit': max(0.001, deadline - time.monotonic()), 'mip_rel_gap': 0.0})
        if result.x is not None:
            candidate = [None] * k_count
            usage = [0.0] * n
            valid = True
            for j in np.flatnonzero(result.x > 0.5):
                k, route = columns[j]
                if candidate[k] is not None:
                    valid = False
                    break
                candidate[k] = route
                for edge in route:
                    usage[edge] += demand[k]
            valid &= all(u <= c + 1e-6 for u, c in zip(usage, capacity))
            value = sum(penalty[k] if r is None else path_cost(k, r) for k, r in enumerate(candidate))
            if valid and value < best_cost + 1e-8:
                best, best_cost = candidate, value
    outcomes = []
    for k, route in enumerate(best):
        outcomes.append({'commodity_id': commodities[k]['commodity_id'],
                         'rejected': route is None,
                         'path_arcs': [] if route is None else [
                             {'from': arcs[a]['from_node'], 'to': arcs[a]['to_node'],
                              'arc_id': arcs[a]['arc_id']} for a in route]})
    return {'objective_value': float(best_cost), 'commodities': outcomes}
