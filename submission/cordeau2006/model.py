#!/usr/bin/env python3
"""Dependency-free DARP heuristic. Usage: python solve.py instance.json -o solution.json.

Uses regret insertion and randomized destroy/repair. Feasible solutions are
validated before output; optimality and finding a feasible solution are not
guaranteed. Input time windows are used as supplied (including any tightening).
"""
from __future__ import annotations

import math
import random
import time


class SearchTimeout(Exception):
    pass


class DARP:
    def __init__(self, instance, deadline=float('inf')):
        self.deadline = deadline
        self.nodes = {int(v['node_id']): v for v in instance['nodes']}
        self.n = int(instance['num_users'])
        self.m = int(instance['num_vehicles'])
        self.cap = float(instance['vehicle_capacity'])
        self.ride_limit = float(instance['maximum_ride_time'])
        self.duration = float(instance['maximum_route_duration'])
        self.start, self.end = 0, 2 * self.n + 1
        if (len(self.nodes) != instance['num_nodes'] or
                set(self.nodes) != set(range(2 * self.n + 2))):
            raise ValueError('Expected unique node IDs 0 through 2 * num_users + 1.')
        if self.m < 1 or min(self.cap, self.ride_limit, self.duration) < 0:
            raise ValueError('Invalid vehicle count or limits.')
        self.pairs = {}
        self.users = {}
        for i, v in self.nodes.items():
            for field in ('x', 'y', 'earliest_time', 'latest_time', 'service_duration', 'load'):
                if not math.isfinite(float(v[field])):
                    raise ValueError(f'Nonfinite {field} at node {i}.')
            if v['earliest_time'] > v['latest_time'] or v['service_duration'] < 0:
                raise ValueError(f'Invalid time bounds at node {i}.')
            if v['node_type'] == 'pickup':
                d = int(v['paired_node'])
                drop = self.nodes[d]
                if (drop['node_type'] != 'dropoff' or drop['paired_node'] != i or
                        drop['user_id'] != v['user_id'] or v['load'] <= 0 or
                        drop['load'] != -v['load']):
                    raise ValueError(f'Invalid pickup/drop-off pairing at node {i}.')
                self.pairs[i] = d
                self.users[i] = str(v['user_id'])
        if (len(self.pairs) != self.n or len(set(self.users.values())) != self.n or
                set(self.pairs) | set(self.pairs.values()) != set(range(1, self.end))):
            raise ValueError('Invalid user/node coverage.')
        for i, kind in ((self.start, 'origin_depot'), (self.end, 'destination_depot')):
            if self.nodes[i]['node_type'] != kind or self.nodes[i]['load'] != 0 or self.nodes[i]['service_duration'] != 0:
                raise ValueError('Depots must have the correct type, zero load and zero service.')
        self.dist = [[math.hypot(self.nodes[i]['x'] - self.nodes[j]['x'],
                                 self.nodes[i]['y'] - self.nodes[j]['y'])
                      for j in range(self.end + 1)] for i in range(self.end + 1)]

    def check_time(self):
        if time.monotonic() >= self.deadline:
            raise SearchTimeout

    def cost(self, route):
        return sum(self.dist[a][b] for a, b in zip(route, route[1:]))

    def schedule(self, route):
        """Solve difference constraints by longest-path relaxation.

        Reverse edges impose ride-time and route-duration upper bounds. Thus
        waiting can be shifted before pickups or before depot departure.
        """
        self.check_time()
        if route[0] != self.start or route[-1] != self.end or len(set(route)) != len(route):
            return None
        positions = {v: i for i, v in enumerate(route)}
        load = 0
        for node in route:
            change = self.nodes[node]['load']
            load += change
            if load < max(0, change) - 1e-8 or load > min(self.cap, self.cap + change) + 1e-8:
                return None
        if abs(load) > 1e-8:
            return None
        edges = [(i, i + 1, self.nodes[a]['service_duration'] + self.dist[a][b])
                 for i, (a, b) in enumerate(zip(route, route[1:]))]
        for p, d in self.pairs.items():
            if (p in positions) != (d in positions):
                return None
            if p in positions:
                if positions[p] >= positions[d]:
                    return None
                service = self.nodes[p]['service_duration']
                edges.append((positions[d], positions[p], -service - self.ride_limit))
                edges.append((positions[p], positions[d], service + self.dist[p][d]))
        edges.append((len(route) - 1, 0, -self.duration))
        times = [float(self.nodes[v]['earliest_time']) for v in route]
        upper = [float(self.nodes[v]['latest_time']) for v in route]
        for _ in range(len(route)):
            changed = False
            for a, b, w in edges:
                candidate = times[a] + w
                if candidate > times[b] + 1e-9:
                    if candidate > upper[b] + 1e-8:
                        return None
                    times[b] = candidate
                    changed = True
            if not changed:
                return times
        return None  # Positive cycle: incompatible timing constraints.

    def options(self, routes, p, limit=2):
        """Best feasible insertion per vehicle; retain the best two vehicles."""
        d = self.pairs[p]
        options = []
        seen = set()
        for k, route in enumerate(routes):
            self.check_time()
            signature = tuple(route)
            if signature in seen:
                continue
            seen.add(signature)
            candidates = []
            for i in range(1, len(route)):
                with_pickup = route[:i] + [p] + route[i:]
                pickup_delta = self.dist[route[i-1]][p] + self.dist[p][route[i]] - self.dist[route[i-1]][route[i]]
                for j in range(i + 1, len(with_pickup)):
                    a, b = with_pickup[j-1:j+1]
                    delta = pickup_delta + self.dist[a][d] + self.dist[d][b] - self.dist[a][b]
                    candidates.append((delta, i, j))
            candidates.sort()
            for delta, i, j in candidates:
                self.check_time()
                candidate = route[:i] + [p] + route[i:]
                candidate.insert(j, d)
                if self.schedule(candidate) is not None:
                    options.append((delta, k, candidate))
                    break
        options.sort(key=lambda item: item[0])
        return options[:limit]

    def repair(self, routes, remaining, rng, noise=0.0):
        routes = [r[:] for r in routes]
        remaining = set(remaining)
        while remaining:
            chosen = None
            best_priority = -float('inf')
            for p in sorted(remaining):
                self.check_time()
                opts = self.options(routes, p)
                if not opts:
                    return None
                regret = opts[1][0] - opts[0][0] if len(opts) > 1 else 1e6
                priority = regret - 0.01 * opts[0][0] + noise * rng.random()
                if priority > best_priority:
                    best_priority, chosen = priority, (p, opts[0])
            p, (_, k, route) = chosen
            routes[k] = route
            remaining.remove(p)
        return routes

    def output(self, routes):
        visited = [v for r in routes for v in r[1:-1]]
        if len(routes) != self.m or sorted(visited) != list(range(1, self.end)):
            raise ValueError('Solution does not serve every user exactly once.')
        result = {'objective_value': sum(self.cost(r) for r in routes),
                  'routes': {}, 'service_times': {}, 'ride_times': {}}
        for k, route in enumerate(routes):
            times = self.schedule(route)
            if times is None:
                raise ValueError('Invalid route in final solution.')
            result['routes'][str(k)] = route
            result['service_times'][f'depot_start_{k}'] = times[0]
            result['service_times'][f'depot_end_{k}'] = times[-1]
            by_node = dict(zip(route, times))
            result['service_times'].update({str(v): by_node[v] for v in route[1:-1]})
            for p, d in self.pairs.items():
                if p in by_node:
                    # The challenge checker indexes requests by pickup node,
                    # not by the optional external user identifier.
                    result['ride_times'][str(p)] = by_node[d] - by_node[p] - self.nodes[p]['service_duration']
        return result


def solve(instance, time_limit=60.0, seed=0):
    """Return a schema-compatible solution, or raise if none was found in time."""
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError('time_limit must be positive and finite.')
    started = time.monotonic()
    problem = DARP(instance, started + time_limit * 0.98)
    rng = random.Random(seed)
    best = None
    best_cost = float('inf')
    current = None
    attempts = 0
    try:
        empty = [[problem.start, problem.end] for _ in range(problem.m)]
        if any(problem.schedule(r) is None for r in empty):
            raise ValueError('Depot-only route is infeasible under the supplied bounds.')
        while True:
            problem.check_time()
            attempts += 1
            if current is None:
                candidate = problem.repair(empty, problem.pairs, rng, 0 if attempts == 1 else 1e6)
            else:
                count = rng.randint(1, max(1, min(problem.n, math.ceil(problem.n * 0.35))))
                removed = set(rng.sample(list(problem.pairs), count))
                nodes = removed | {problem.pairs[p] for p in removed}
                partial = [[v for v in route if v not in nodes] for route in current]
                candidate = problem.repair(partial, removed, rng, rng.choice([0, 10, 100, 1e6]))
            if candidate is None:
                continue
            cost = sum(problem.cost(r) for r in candidate)
            if cost < best_cost:
                best, best_cost = [r[:] for r in candidate], cost
            if current is None or cost <= current_cost:
                current, current_cost = candidate, cost
            else:
                fraction = min(1.0, (time.monotonic() - started) / time_limit)
                temperature = max(1e-9, best_cost * 0.02 * (1 - fraction))
                if rng.random() < math.exp(-(cost - current_cost) / temperature):
                    current, current_cost = candidate, cost
            if problem.n <= 1:
                break
    except SearchTimeout:
        pass
    if best is None:
        raise RuntimeError('No feasible solution found within the time limit; infeasibility is not proven.')
    problem.deadline = float('inf')  # Reserve the final budget for validation/serialization.
    return problem.output(best)
