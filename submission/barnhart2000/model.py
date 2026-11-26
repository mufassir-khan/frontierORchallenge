"""Unsplittable origin/destination flow with capacity-priced path generation.

Small networks use an arc-based binary formulation. Larger networks alternate
LP dual pricing and shortest-path generation, then solve the resulting binary
path-packing model. Rejection is always available, so the search retains a
feasible complete assignment even if a mathematical solver times out.

Only NumPy and SciPy/HiGHS are required; both are in the challenge images.
"""
from dataclasses import dataclass
import heapq
import math
import random
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix


@dataclass(frozen=True)
class Request:
    identifier: int
    origin: int
    destination: int
    demand: float
    rejection: float


class Network:
    def __init__(self, instance):
        network = instance['network']
        self.node_ids = list(network['nodes'])
        lookup = {identifier: i for i, identifier in enumerate(self.node_ids)}
        self.arcs = list(network['arcs'])
        self.n = len(self.node_ids)
        self.m = len(self.arcs)
        if len(lookup) != self.n or len({arc['arc_id'] for arc in self.arcs}) != self.m:
            raise ValueError('Node and physical arc identifiers must be unique')
        self.tail = np.array([lookup[a['from_node']] for a in self.arcs], dtype=int)
        self.head = np.array([lookup[a['to_node']] for a in self.arcs], dtype=int)
        self.capacity = np.array([float(a['capacity']) for a in self.arcs])
        self.cost = np.array([float(a.get('cost', 0)) for a in self.arcs])
        self.outgoing = [[] for _ in self.node_ids]
        for edge, u in enumerate(self.tail):
            self.outgoing[u].append(edge)
        self.requests = []
        for item in instance['commodities']['commodity_list']:
            demand = float(item['demand'])
            self.requests.append(Request(item['commodity_id'], lookup[item['origin']],
                lookup[item['destination']], demand, demand * float(item['artificial_arc_cost'])))
        if len({r.identifier for r in self.requests}) != len(self.requests):
            raise ValueError('Commodity identifiers must be unique')
        numbers = list(self.capacity) + list(self.cost)
        numbers += [v for r in self.requests for v in (r.demand, r.rejection)]
        if any(not math.isfinite(v) or v < 0 for v in numbers):
            raise ValueError('Capacities, costs and demands must be finite and nonnegative')

    def shortest_path(self, request, available, unit_cost):
        """Dijkstra on eligible arcs; supports parallel arcs and arbitrary IDs."""
        if request.origin == request.destination:
            return ()
        labels = [math.inf] * self.n
        predecessor = [-1] * self.n
        labels[request.origin] = 0.0
        heap = [(0.0, request.origin)]
        while heap:
            distance, u = heapq.heappop(heap)
            if distance != labels[u]:
                continue
            if u == request.destination:
                route = []
                while u != request.origin:
                    edge = predecessor[u]
                    route.append(edge)
                    u = int(self.tail[edge])
                return tuple(reversed(route))
            for edge in self.outgoing[u]:
                if available[edge] + 1e-9 < request.demand:
                    continue
                v = int(self.head[edge])
                candidate = distance + float(unit_cost[edge])
                if candidate < labels[v] - 1e-12:
                    labels[v] = candidate
                    predecessor[v] = edge
                    heapq.heappush(heap, (candidate, v))
        return None

    def route_cost(self, k, route):
        request = self.requests[k]
        return request.rejection if route is None else request.demand * sum(self.cost[a] for a in route)

    def validate(self, assignment):
        if len(assignment) != len(self.requests):
            return False
        usage = np.zeros(self.m)
        for request, route in zip(self.requests, assignment):
            if route is None:
                continue
            current = request.origin
            seen = {current}
            for edge in route:
                if edge < 0 or edge >= self.m or self.tail[edge] != current:
                    return False
                current = int(self.head[edge])
                if current in seen:
                    return False
                seen.add(current)
                usage[edge] += request.demand
            if current != request.destination:
                return False
        return bool(np.all(usage <= self.capacity + 1e-7))

    def output(self, assignment):
        if not self.validate(assignment):
            raise RuntimeError('Internal routing validation failed')
        outcomes = []
        objective = 0.0
        for k, route in enumerate(assignment):
            objective += self.route_cost(k, route)
            outcomes.append({'commodity_id': self.requests[k].identifier,
                             'rejected': route is None,
                             'path_arcs': [] if route is None else [
                                 {'arc_id': self.arcs[a]['arc_id'],
                                  'from': self.arcs[a]['from_node'],
                                  'to': self.arcs[a]['to_node']} for a in route]})
        return {'objective_value': float(objective), 'commodities': outcomes}


def _arc_model(net, deadline):
    """Exact arc model, with binary acceptance and per-node degree bounds."""
    count = len(net.requests)
    variables = count * (net.m + 1)
    objective = np.zeros(variables)
    upper = np.ones(variables)
    rows, cols, values, lo, hi = [], [], [], [], []

    def add(terms, lower, upper_bound):
        row = len(lo)
        lo.append(lower); hi.append(upper_bound)
        for col, value in terms:
            rows.append(row); cols.append(col); values.append(value)

    for edge in range(net.m):
        add(((k*(net.m+1)+edge, r.demand) for k, r in enumerate(net.requests)), -np.inf, net.capacity[edge])
    for k, request in enumerate(net.requests):
        base = k * (net.m+1)
        accepted = base + net.m
        objective[base:accepted] = request.demand * net.cost
        objective[accepted] = -request.rejection
        balance = [[] for _ in range(net.n)]
        outgoing = [[] for _ in range(net.n)]
        for edge in range(net.m):
            u, v = net.tail[edge], net.head[edge]
            if net.capacity[edge] + 1e-9 < request.demand or u == v:
                upper[base+edge] = 0
            balance[u].append((base+edge, 1))
            balance[v].append((base+edge, -1))
            outgoing[u].append((base+edge, 1))
        if request.origin != request.destination:
            balance[request.origin].append((accepted, -1))
            balance[request.destination].append((accepted, 1))
        for node in range(net.n):
            add(balance[node], 0, 0)
            add(outgoing[node], -np.inf, 1)
    if time.monotonic() >= deadline or variables == 0:
        return None
    matrix = coo_matrix((values, (rows, cols)), shape=(len(lo), variables)).tocsc()
    result = milp(objective, integrality=np.ones(variables), bounds=Bounds(0, upper),
                  constraints=LinearConstraint(matrix, lo, hi),
                  options={'time_limit': max(0.001, deadline-time.monotonic()), 'mip_rel_gap': 0.0})
    if result.x is None:
        return None
    assignment = [None] * count
    for k, request in enumerate(net.requests):
        base = k*(net.m+1)
        if result.x[base+net.m] <= 0.5:
            continue
        allowed = np.where(result.x[base:base+net.m] > 0.5, net.capacity, -1)
        assignment[k] = net.shortest_path(request, allowed, net.cost + 1e-9)
    return assignment if net.validate(assignment) else None


class PathMaster:
    def __init__(self, net):
        self.net = net
        self.paths = []
        self.keys = set()

    def add(self, k, path):
        if path is None or (k, path) in self.keys:
            return False
        benefit = self.net.route_cost(k, path) - self.net.requests[k].rejection
        if benefit > 1e-8:
            return False
        self.keys.add((k, path))
        self.paths.append((k, path, benefit))
        return True

    def matrix(self):
        net = self.net
        rows, cols, values = [], [], []
        for j, (k, path, _) in enumerate(self.paths):
            rows.append(net.m+k); cols.append(j); values.append(1.0)
            for edge in path:
                rows.append(edge); cols.append(j); values.append(net.requests[k].demand)
        a = coo_matrix((values, (rows, cols)), shape=(net.m+len(net.requests), len(self.paths))).tocsc()
        return a, np.array([item[2] for item in self.paths]), np.r_[net.capacity, np.ones(len(net.requests))]


def solve(instance, time_limit=56.0, seed=0):
    started = time.monotonic()
    deadline = started + max(0.001, float(time_limit))
    net = Network(instance)
    rng = random.Random(seed)
    best = [None] * len(net.requests)
    best_cost = sum(r.rejection for r in net.requests)
    master = PathMaster(net)

    def retain(candidate):
        nonlocal best, best_cost
        if candidate is None or not net.validate(candidate):
            return
        value = sum(net.route_cost(k, r) for k, r in enumerate(candidate))
        if value < best_cost - 1e-8:
            best, best_cost = candidate[:], value
        for k, route in enumerate(candidate):
            master.add(k, route)

    # A capacity-aware randomized packing seeds feasible solutions and columns.
    seed_deadline = min(deadline, started + max(0.02, min(4.0, time_limit*0.12)))
    iteration = 0
    while time.monotonic() < seed_deadline and iteration < 30:
        available = net.capacity.copy()
        route_set = [None] * len(net.requests)
        order = list(range(len(net.requests)))
        if iteration:
            rng.shuffle(order)
        order.sort(key=lambda k: -net.requests[k].rejection /
                   max(1, net.requests[k].demand)**(iteration % 3 / 2) *
                   (1 if iteration == 0 else rng.uniform(0.6, 1.4)))
        for k in order:
            if time.monotonic() >= seed_deadline:
                break
            request = net.requests[k]
            scarcity = 1 / np.maximum(available, 0.25)
            weights = net.cost + 0.01 + scarcity * (0.1 + iteration/10)
            path = net.shortest_path(request, available, weights)
            master.add(k, path)
            if path is not None and net.route_cost(k, path) <= request.rejection + 1e-8:
                route_set[k] = path
                for edge in path:
                    available[edge] -= request.demand
        retain(route_set)
        iteration += 1
    if len(net.requests)*(net.m+1) <= 12000:
        retain(_arc_model(net, deadline))
        return net.output(best)

    # Pricing uses the LP's capacity shadow prices; generated paths can route a
    # commodity differently to avoid the capacity bottlenecks of greedy packing.
    pricing_deadline = started + time_limit*0.38
    shadow = np.zeros(net.m)
    for iteration in range(12):
        if time.monotonic() >= pricing_deadline:
            break
        added = False
        for k, request in enumerate(net.requests):
            if time.monotonic() >= pricing_deadline:
                break
            perturbation = 1e-7 if iteration % 2 == 0 else 1e-4
            weights = net.cost + shadow + perturbation * np.array([rng.random() for _ in range(net.m)])
            path = net.shortest_path(request, net.capacity, weights)
            added |= master.add(k, path)
        if not master.paths or not added or time.monotonic() >= pricing_deadline:
            break
        matrix, objective, limit = master.matrix()
        lp = linprog(objective, A_ub=matrix, b_ub=limit, bounds=(0, None), method='highs',
                     options={'time_limit': max(0.001, pricing_deadline-time.monotonic())})
        if not lp.success:
            break
        shadow = np.maximum(0, -lp.ineqlin.marginals[:net.m])
    if master.paths and time.monotonic() < deadline:
        matrix, objective, limit = master.matrix()
        result = milp(objective, integrality=np.ones(len(master.paths)), bounds=Bounds(0, 1),
                      constraints=LinearConstraint(matrix, -np.inf, limit),
                      options={'time_limit': max(0.001, deadline-time.monotonic()), 'mip_rel_gap': 0.0})
        if result.x is not None:
            candidate = [None] * len(net.requests)
            unique = True
            for j in np.flatnonzero(result.x > 0.5):
                k, route, _ = master.paths[j]
                if candidate[k] is not None:
                    unique = False
                    break
                candidate[k] = route
            if unique:
                retain(candidate)
    return net.output(best)
