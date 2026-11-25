"""Orienteering: connected-tour MILP for small instances, adaptive insertion
and 2-opt/destroy-repair for large ones. Returns a single feasible cycle with
at least three distinct cities; preserves an incumbent throughout the search.
"""
import ast
import heapq
import math
import random
import time
import numpy as np


def solve(instance, time_limit=56.0, seed=0):
    deadline = time.monotonic() + time_limit
    rng = random.Random(seed)
    ids = sorted(map(int, instance['prizes']))
    n = len(ids)
    if n != instance['n'] or n < 3:
        raise ValueError('Orienteering needs at least three cities')
    index = {node: j for j, node in enumerate(ids)}
    depot = index[int(instance['depot'])]
    prizes = np.array([instance['prizes'][str(node)] for node in ids], dtype=float)
    distance = np.full((n, n), np.inf)
    np.fill_diagonal(distance, 0)
    for key, value in instance['travel_times'].items():
        i, j = ast.literal_eval(key)
        a, b = index[i], index[j]
        distance[a, b] = distance[b, a] = float(value)
    if not np.isfinite(distance).all() or np.any(distance < 0):
        raise ValueError('Expected a complete nonnegative symmetric travel-time matrix')
    budget = float(instance['t0'])
    other = [v for v in range(n) if v != depot]

    def length(route):
        return float(sum(distance[a, b] for a, b in zip(route, route[1:])))

    def reward(route):
        return float(sum(prizes[v] for v in route[1:-1]))

    triangles = [(distance[depot, a] + distance[a, b] + distance[b, depot], a, b)
                 for i, a in enumerate(other) for b in other[i+1:]]
    shortest, a, b = min(triangles)
    if shortest <= budget + 1e-8:
        best = [depot, a, b, depot]
    else:
        # Arbitrary symmetric matrices need not obey triangle inequality.
        # Find the shortest depot cycle through >=2 other vertices.
        best = None
        for source in other:
            if time.monotonic() >= deadline:
                break
            labels, parent = {source: 0.0}, {}
            queue = [(0.0, source)]
            while queue:
                value, u = heapq.heappop(queue)
                if value != labels[u]:
                    continue
                if u != source and distance[depot, source] + value + distance[u, depot] <= budget:
                    path, v = [u], u
                    while v != source:
                        v = parent[v]; path.append(v)
                    best = [depot] + list(reversed(path)) + [depot]
                    break
                for v in other:
                    if v == u:
                        continue
                    candidate = value + distance[u, v]
                    if candidate < labels.get(v, math.inf):
                        labels[v], parent[v] = candidate, u
                        heapq.heappush(queue, (candidate, v))
            if best:
                break
        if best is None:
            raise RuntimeError('No feasible cycle with three cities found')

    def two_opt(route, stop):
        route = route[:]
        for _ in range(30):
            if len(route) < 5 or time.monotonic() >= stop:
                break
            r = np.asarray(route)
            a, b = r[:-1], r[1:]
            delta = distance[a[:, None], a] + distance[b[:, None], b] - distance[a, b][:, None] - distance[a, b]
            delta[np.tril_indices(len(a), 1)] = np.inf
            delta[0, -1] = np.inf
            i, j = np.unravel_index(np.argmin(delta), delta.shape)
            if delta[i, j] >= -1e-8:
                break
            route[i+1:j+1] = reversed(route[i+1:j+1])
        return route

    def insert(route, alpha, stop):
        used = set(route)
        remaining = [v for v in other if v not in used]
        total = length(route)
        while remaining and time.monotonic() < stop:
            candidates = np.asarray(remaining)
            r = np.asarray(route)
            delta = distance[r[:-1, None], candidates] + distance[candidates, r[1:, None]] - distance[r[:-1], r[1:]][:, None]
            places = np.argmin(delta, axis=0)
            increment = delta[places, np.arange(len(candidates))]
            feasible = total + increment <= budget + 1e-8
            if not feasible.any():
                break
            score = (prizes[candidates] + 0.01) / np.maximum(0.05, increment) ** alpha
            score[~feasible] = -np.inf
            selected = int(np.argmax(score))
            v = remaining.pop(selected)
            route.insert(int(places[selected]) + 1, v)
            total += float(increment[selected])
        return route

    heuristic_end = deadline if n > 60 else min(deadline, time.monotonic() + time_limit * 0.25)
    current = best[:]
    best_value = reward(best)
    iteration = 0
    while time.monotonic() < heuristic_end:
        candidate = two_opt(current, heuristic_end)
        candidate = insert(candidate, rng.choice([0.7, 1.0, 1.3]), heuristic_end)
        candidate = two_opt(candidate, heuristic_end)
        value = reward(candidate)
        if value > best_value or (value == best_value and length(candidate) < length(best)):
            best, best_value = candidate[:], value
        if best_value >= float(sum(prizes)) - 1e-8:
            break
        current = (best if iteration % 5 == 0 else candidate)[:]
        if len(current) > 4:
            count = rng.randint(1, max(1, min(len(current)-4, len(current)//4)))
            remove = set(rng.sample(current[1:-1], count))
            current = [v for v in current if v not in remove]
        else:
            feasible = [(a, b) for cost, a, b in triangles if cost <= budget]
            if feasible:
                a, b = rng.choice(feasible)
                current = [depot, a, b, depot]
        if length(current) > budget + 1e-8:
            # Deletion can increase cost in a nonmetric or rounded matrix.
            current = best[:]
        iteration += 1
    if n <= 60 and deadline - time.monotonic() > 0.1 and best_value < sum(prizes):
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix
        edges = [(a, b) for a in range(n) for b in range(a+1, n)]
        e = len(edges)
        columns = e + n + 2 * e
        objective = np.zeros(columns)
        objective[e:e+n] = -prizes
        upper = np.r_[np.ones(e+n), np.full(2*e, n-1)]
        lower = np.zeros(columns); lower[e+depot] = 1
        integrality = np.r_[np.ones(e+n), np.zeros(2*e)]
        rr, cc, vv, lb, ub = [], [], [], [], []

        def constraint(terms, low, high):
            r = len(lb); lb.append(low); ub.append(high)
            for c, v in terms:
                rr.append(r); cc.append(c); vv.append(v)

        incident = [[] for _ in range(n)]
        balance = [[] for _ in range(n)]
        for j, (a, b) in enumerate(edges):
            incident[a].append((j, 1)); incident[b].append((j, 1))
            f, g = e+n+2*j, e+n+2*j+1
            balance[a].extend([(f, -1), (g, 1)])
            balance[b].extend([(f, 1), (g, -1)])
            constraint([(f, 1), (j, -(n-1))], -np.inf, 0)
            constraint([(g, 1), (j, -(n-1))], -np.inf, 0)
        for v in range(n):
            constraint(incident[v] + [(e+v, -2)], 0, 0)
            if v != depot:
                constraint(balance[v] + [(e+v, -1)], 0, 0)
        constraint([(j, distance[a, b]) for j, (a, b) in enumerate(edges)], -np.inf, budget)
        constraint([(e+v, 1) for v in range(n)], 3, np.inf)
        constraint([(e+v, prizes[v]) for v in range(n)], best_value, np.inf)
        matrix = coo_matrix((vv, (rr, cc)), shape=(len(lb), columns)).tocsc()
        result = milp(objective, integrality=integrality, bounds=Bounds(lower, upper),
                      constraints=LinearConstraint(matrix, lb, ub),
                      options={'time_limit': max(0.001, deadline-time.monotonic()), 'mip_rel_gap': 0.0})
        if result.x is not None:
            adjacency = {v: [] for v in range(n)}
            for j, (a, b) in enumerate(edges):
                if result.x[j] > 0.5:
                    adjacency[a].append(b); adjacency[b].append(a)
            route, previous, v = [depot], None, depot
            for _ in range(n):
                options = [u for u in adjacency[v] if u != previous]
                if not options:
                    break
                u = options[0]; route.append(u)
                previous, v = v, u
                if v == depot:
                    break
            if (route[-1] == depot and len(route) >= 4 and len(set(route[:-1])) == len(route)-1
                    and length(route) <= budget + 1e-6 and reward(route) >= best_value):
                best = route
    tour = [ids[v] for v in best]
    if length(best) > budget + 1e-6 or len(set(best[:-1])) != len(best)-1 or len(best) < 4:
        raise RuntimeError('Invalid final tour')
    return {'objective_value': reward(best), 'visited_nodes': tour[:-1],
            'edges': [[a, b] for a, b in zip(tour, tour[1:])], 'tour': tour}
