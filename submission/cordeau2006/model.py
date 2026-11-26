"""DARP solver: floating-point feasibility plus OR-Tools paired-route search.

The routing search uses conservative integer travel times. Every candidate
returned to the caller is rescheduled with the original Euclidean distances.
Optional search nodes are only a construction aid: partial service is NEVER
accepted as a solution. Vehicle and request identifiers are not conflated.
"""
from __future__ import annotations

from collections import deque
import math
import random
import time


class DARP:
    """Instance data and an independent exact-distance route validator."""

    def __init__(self, instance):
        self.n = int(instance['num_users'])
        self.m = int(instance['num_vehicles'])
        self.end = 2*self.n+1
        self.capacity = int(instance['vehicle_capacity'])
        self.max_ride = float(instance['maximum_ride_time'])
        self.max_duration = float(instance['maximum_route_duration'])
        self.nodes = sorted(instance['nodes'], key=lambda node: node['node_id'])
        if (instance['num_nodes'] != self.end+1 or
                [v['node_id'] for v in self.nodes] != list(range(self.end+1)) or self.m < 1):
            raise ValueError('Invalid node or vehicle counts')
        self.early = [float(v['earliest_time']) for v in self.nodes]
        self.late = [float(v['latest_time']) for v in self.nodes]
        self.service = [float(v['service_duration']) for v in self.nodes]
        self.load = [int(v['load']) for v in self.nodes]
        for i in range(self.end+1):
            if (not all(math.isfinite(x) for x in (self.early[i], self.late[i], self.service[i]))
                    or self.early[i] > self.late[i] or self.service[i] < 0):
                raise ValueError('Invalid node time data')
        self.pairs = {v['node_id']: int(v['paired_node']) for v in self.nodes if v['node_type'] == 'pickup'}
        if len(self.pairs) != self.n or set(self.pairs) | set(self.pairs.values()) != set(range(1, self.end)):
            raise ValueError('Every request needs a distinct pickup and delivery')
        for p, d in self.pairs.items():
            if (self.nodes[d]['node_type'] != 'dropoff' or self.nodes[d]['paired_node'] != p
                    or self.nodes[p]['user_id'] != self.nodes[d]['user_id']
                    or self.load[p] <= 0 or self.load[d] != -self.load[p]):
                raise ValueError('Inconsistent pickup/dropoff pairing')
        self.distance = [[math.hypot(a['x']-b['x'], a['y']-b['y']) for b in self.nodes] for a in self.nodes]

    def cost(self, route):
        return sum(self.distance[a][b] for a, b in zip(route, route[1:]))

    def schedule(self, route):
        """Queue relaxation of time-window/ride/route-duration inequalities.

        A bound t[b] >= t[a] + w is an edge a->b of weight w. Upper
        ride and duration limits create backward edges, shifting waiting time
        before the pickup/depot instead of wrongly declaring a route infeasible.
        """
        if not route or route[0] != 0 or route[-1] != self.end or len(set(route)) != len(route):
            return None
        index = {node: position for position, node in enumerate(route)}
        onboard = 0
        graph = [[] for _ in route]
        for i, node in enumerate(route):
            onboard += self.load[node]
            if onboard < max(0, self.load[node]) or onboard > min(self.capacity, self.capacity+self.load[node]):
                return None
            if i:
                previous = route[i-1]
                graph[i-1].append((i, self.service[previous]+self.distance[previous][node]))
        if onboard:
            return None
        for p, d in self.pairs.items():
            if (p in index) != (d in index):
                return None
            if p in index:
                a, b = index[p], index[d]
                if a >= b:
                    return None
                graph[a].append((b, self.service[p]+self.distance[p][d]))
                graph[b].append((a, -self.service[p]-self.max_ride))
        graph[-1].append((0, -self.max_duration))
        values = [self.early[v] for v in route]
        queue = deque(range(len(route)))
        queued = [True]*len(route)
        depth = [0]*len(route)
        while queue:
            a = queue.popleft(); queued[a] = False
            for b, amount in graph[a]:
                candidate = values[a]+amount
                if candidate > values[b]+1e-10:
                    if candidate > self.late[route[b]]+1e-8:
                        return None
                    values[b] = candidate
                    depth[b] = depth[a]+1
                    if depth[b] >= len(route):
                        return None
                    if not queued[b]:
                        queue.append(b); queued[b] = True
        return values

    def result(self, routes):
        visits = [v for route in routes for v in route[1:-1]]
        if len(routes) != self.m or sorted(visits) != list(range(1, self.end)):
            raise ValueError('A complete solution must serve every request exactly once')
        service_times, ride_times = {}, {}
        for vehicle, route in enumerate(routes):
            times = self.schedule(route)
            if times is None:
                raise ValueError('Final route failed schedule/capacity validation')
            by_node = dict(zip(route, times))
            service_times[f'depot_start_{vehicle}'] = times[0]
            service_times[f'depot_end_{vehicle}'] = times[-1]
            service_times.update({str(v): by_node[v] for v in route[1:-1]})
            for p, d in self.pairs.items():
                if p in by_node:
                    # The published checker keys ride times by pickup node ID.
                    ride_times[str(p)] = by_node[d]-by_node[p]-self.service[p]
        return {'objective_value': float(sum(map(self.cost, routes))),
                'routes': {str(k): route for k, route in enumerate(routes)},
                'service_times': service_times, 'ride_times': ride_times}


def _construct(data, deadline, rng):
    """Deadline-bounded, exact-distance insertion seed for the routing engine."""
    attempt = 0
    while time.monotonic() < deadline:
        routes = [[0, data.end] for _ in range(data.m)]
        order = sorted(data.pairs, key=lambda p: (data.late[p] if attempt == 0 else
                       data.late[p]+rng.uniform(-80,80)))
        for p in order:
            choices = []
            seen = set()
            for k, route in enumerate(routes):
                if tuple(route) in seen:
                    continue
                seen.add(tuple(route))
                for i in range(1, len(route)):
                    picked = route[:i]+[p]+route[i:]
                    for j in range(i+1, len(picked)):
                        if time.monotonic() >= deadline:
                            return None
                        candidate = picked[:j]+[data.pairs[p]]+picked[j:]
                        increment = data.cost(candidate)-data.cost(route)
                        choices.append((increment, k, candidate))
            choices.sort(key=lambda item: item[0])
            chosen = None
            for _, k, route in choices:
                if time.monotonic() >= deadline:
                    return None
                if data.schedule(route) is not None:
                    chosen = k, route
                    break
            if chosen is None:
                break
            k, route = chosen
            routes[k] = route
        else:
            return routes
        attempt += 1
    return None


def _routing_search(data, incumbent, deadline):
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    scale = 1_000_000
    shift = min(data.early)
    ceil = lambda x: int(math.ceil(x*scale-1e-7))
    floor = lambda x: int(math.floor(x*scale+1e-7))
    horizon = ceil(max(data.late)-shift)+1
    manager = pywrapcp.RoutingIndexManager(data.end+1, data.m, [0]*data.m, [data.end]*data.m)
    routing = pywrapcp.RoutingModel(manager)
    index_nodes = [manager.IndexToNode(i) for i in range(manager.GetNumberOfIndices())]
    metric = [[int(round(v*10000)) for v in row] for row in data.distance]
    transit = [[ceil(data.service[a]+data.distance[a][b]) for b in range(data.end+1)] for a in range(data.end+1)]
    cost_callback = routing.RegisterTransitCallback(lambda a,b: metric[index_nodes[a]][index_nodes[b]])
    routing.SetArcCostEvaluatorOfAllVehicles(cost_callback)
    time_callback = routing.RegisterTransitCallback(lambda a,b: transit[index_nodes[a]][index_nodes[b]])
    routing.AddDimension(time_callback, horizon, horizon, False, 'Time')
    clock = routing.GetDimensionOrDie('Time')
    load_callback = routing.RegisterUnaryTransitCallback(lambda a: data.load[index_nodes[a]])
    routing.AddDimensionWithVehicleCapacity(load_callback, 0, [data.capacity]*data.m, True, 'Passengers')
    cp = routing.solver()
    penalty = max(1, sum(max(row) for row in metric)) * 100
    for node in range(1, data.end):
        index = manager.NodeToIndex(node)
        clock.CumulVar(index).SetRange(ceil(data.early[node]-shift), floor(data.late[node]-shift))
        routing.AddDisjunction([index], penalty)
    for vehicle in range(data.m):
        start, finish = routing.Start(vehicle), routing.End(vehicle)
        clock.CumulVar(start).SetRange(ceil(data.early[0]-shift), floor(data.late[0]-shift))
        clock.CumulVar(finish).SetRange(ceil(data.early[-1]-shift), floor(data.late[-1]-shift))
        clock.SetSpanUpperBoundForVehicle(floor(data.max_duration), vehicle)
    for p, d in data.pairs.items():
        a, b = manager.NodeToIndex(p), manager.NodeToIndex(d)
        routing.AddPickupAndDelivery(a,b)
        cp.Add(routing.VehicleVar(a) == routing.VehicleVar(b))
        cp.Add(routing.ActiveVar(a) == routing.ActiveVar(b))
        cp.Add(clock.CumulVar(b)-clock.CumulVar(a) >= ceil(data.service[p]+data.distance[p][d]))
        cp.Add(clock.CumulVar(b)-clock.CumulVar(a) <= floor(data.service[p]+data.max_ride))
    parameters = pywrapcp.DefaultRoutingSearchParameters()
    parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    parameters.time_limit.FromMilliseconds(max(1, int((deadline-time.monotonic())*1000)))
    parameters.log_search = False
    # The exact-distance seed can be infeasible after conservative rounding;
    # discard only the rounded assignment, retaining the valid external seed.
    assignment = routing.ReadAssignmentFromRoutes([r[1:-1] for r in incumbent], True) if incumbent else None
    if time.monotonic() >= deadline:
        return incumbent
    parameters.time_limit.FromMilliseconds(max(1, int((deadline-time.monotonic())*1000)))
    solution = (routing.SolveFromAssignmentWithParameters(assignment, parameters) if assignment
                else routing.SolveWithParameters(parameters))
    if solution is None:
        return incumbent
    routes = []
    for k in range(data.m):
        index = routing.Start(k)
        route = []
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route.append(data.end)
        routes.append(route)
    try:
        data.result(routes)
    except ValueError:
        return incumbent
    if incumbent is None or sum(map(data.cost,routes)) < sum(map(data.cost,incumbent)):
        return routes
    return incumbent


def solve(instance, time_limit=55.0, seed=0):
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError('time_limit must be finite and positive')
    began = time.monotonic()
    data = DARP(instance)
    deadline = began + time_limit*0.98
    rng = random.Random(seed)
    incumbent = _construct(data, min(deadline, time.monotonic()+min(3.0,time_limit*0.12)), rng)
    if data.n <= 1 and incumbent is not None:
        return data.result(incumbent)
    if time.monotonic() < deadline:
        incumbent = _routing_search(data, incumbent, deadline)
    if incumbent is None:
        raise RuntimeError('No complete feasible DARP solution found before the search deadline')
    return data.result(incumbent)
