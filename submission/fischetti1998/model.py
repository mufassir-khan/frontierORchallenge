"""Prize-collecting routing model for the FrontierOR orienteering problem.

All cities are optional except the depot; a hard route-length dimension enforces
its budget and a cardinality constraint requires two nondepot visits. Dropping a
city costs prize * (budget + 1), making collected prize the primary objective and
travel length the secondary objective. PyVRP improves large-instance seeds
when installed; OR-Tools remains the fallback. Every returned tour is validated.
"""
from __future__ import annotations

import ast
import heapq
import math
import time

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


def _integer(value, name):
    number = float(value)
    if not math.isfinite(number) or number < 0 or number != int(number):
        raise ValueError(f'{name} must be a nonnegative integer')
    return int(number)


class Orienteering:
    def __init__(self, instance):
        self.ids = sorted(int(key) for key in instance['prizes'])
        if len(self.ids) != instance['n'] or len(self.ids) < 3:
            raise ValueError('Expected at least three cities and one prize per city')
        self.index = {node: index for index, node in enumerate(self.ids)}
        self.depot = self.index[int(instance['depot'])]
        self.budget = _integer(instance['t0'], 't0')
        self.prize = [_integer(instance['prizes'][str(node)], 'prize') for node in self.ids]
        self.size = len(self.ids)
        self.distance = [[None] * self.size for _ in self.ids]
        for index in range(self.size):
            self.distance[index][index] = 0
        for key, value in instance['travel_times'].items():
            first, second = ast.literal_eval(key)
            a, b = self.index[first], self.index[second]
            cost = _integer(value, 'travel time')
            self.distance[a][b] = self.distance[b][a] = cost
        if any(value is None for row in self.distance for value in row):
            raise ValueError('Missing travel time in the complete graph')
        if self.prize[self.depot] != 0:
            raise ValueError('The depot prize must be zero')

    def length(self, tour):
        return sum(self.distance[a][b] for a, b in zip(tour, tour[1:]))

    def value(self, tour):
        return sum(self.prize[node] for node in tour[1:-1])

    def valid(self, tour):
        return (len(tour) >= 4 and tour[0] == tour[-1] == self.depot
                and len(set(tour[:-1])) == len(tour) - 1
                and self.length(tour) <= self.budget)

    def seed(self, deadline):
        """Choose a good triangle, or find a nonmetric cycle via shortest paths."""
        depot = self.depot
        others = [node for node in range(self.size) if node != depot]
        chosen, rank = None, None
        shortest, shortest_length = None, math.inf
        for offset, a in enumerate(others):
            for b in others[offset + 1:]:
                length = self.distance[depot][a] + self.distance[a][b] + self.distance[b][depot]
                if length <= self.budget:
                    if length < shortest_length:
                        shortest, shortest_length = [depot, a, b, depot], length
                    score = (self.prize[a] + self.prize[b], -length)
                    if rank is None or score > rank:
                        chosen, rank = [depot, a, b, depot], score
            if time.monotonic() >= deadline:
                break
        if chosen is not None:
            self.shortest_seed = shortest
            return chosen
        # The schema promises symmetry, but not triangle inequality. A feasible
        # longer cycle may exist even when every triangle exceeds the budget.
        for source in others:
            if time.monotonic() >= deadline:
                break
            labels = {source: 0}
            previous = {}
            queue = [(0, source)]
            while queue and time.monotonic() < deadline:
                length, node = heapq.heappop(queue)
                if labels[node] != length:
                    continue
                if node != source and self.distance[depot][source] + length + self.distance[node][depot] <= self.budget:
                    route = [node]
                    while route[-1] != source:
                        route.append(previous[route[-1]])
                    return [depot] + route[::-1] + [depot]
                for target in others:
                    if target == node:
                        continue
                    candidate = length + self.distance[node][target]
                    if candidate < labels.get(target, math.inf):
                        labels[target] = candidate
                        previous[target] = node
                        heapq.heappush(queue, (candidate, target))
        raise RuntimeError('No feasible tour with three cities was found')

    def expand_seed(self, initial, deadline):
        """Build several prize-aware starting tours before constrained search."""
        distances = np.asarray(self.distance, dtype=np.int64)
        prizes = np.asarray(self.prize, dtype=float)
        best = initial[:]

        def shorten(route):
            for _ in range(8):
                if time.monotonic() >= deadline or len(route) < 5:
                    break
                ends = np.asarray(route, dtype=int)
                first, second = ends[:-1], ends[1:]
                saving = (distances[first[:, None], first] + distances[second[:, None], second]
                          - distances[first, second][:, None] - distances[first, second])
                eligible = np.triu(np.ones(saving.shape, dtype=bool), 2)
                eligible[0, -1] = False
                eligible &= saving < 0
                if not np.any(eligible):
                    break
                row, column = np.unravel_index(np.argmin(np.where(eligible, saving, 0)), saving.shape)
                route[row + 1:column + 1] = reversed(route[row + 1:column + 1])
            return route

        for start in (getattr(self, 'shortest_seed', initial), initial):
            for exponent in (1.0, 0.7, 1.3):
                if time.monotonic() >= deadline:
                    return best
                route = start[:]
                for _ in range(3):
                    route = shorten(route)
                    total = self.length(route)
                    visited = set(route)
                    unused = [v for v in range(self.size) if v not in visited]
                    while unused and time.monotonic() < deadline:
                        candidates = np.asarray(unused, dtype=int)
                        endpoints = np.asarray(route, dtype=int)
                        extra = (distances[endpoints[:-1, None], candidates]
                                 + distances[candidates, endpoints[1:, None]]
                                 - distances[endpoints[:-1], endpoints[1:]][:, None])
                        positions = extra.argmin(axis=0)
                        increments = extra[positions, np.arange(len(unused))]
                        fits = total + increments <= self.budget
                        if not np.any(fits):
                            break
                        priority = (prizes[candidates] + 0.01) / np.maximum(0.1, increments) ** exponent
                        priority[~fits] = -np.inf
                        chosen = int(priority.argmax())
                        route.insert(int(positions[chosen]) + 1, unused.pop(chosen))
                        total += int(increments[chosen])
                    if self.valid(route) and (self.value(route), -self.length(route)) > (self.value(best), -self.length(best)):
                        best = route[:]
        return best

    def output(self, route):
        if not self.valid(route):
            raise RuntimeError('Invalid orienteering incumbent')
        nodes = [self.ids[node] for node in route]
        return {'objective_value': float(self.value(route)),
                'visited_nodes': nodes[:-1],
                'edges': [[a, b] for a, b in zip(nodes, nodes[1:])],
                'tour': nodes}

    def genetic_candidate(self, incumbent, deadline, seed):
        try:
            import pyvrp
            from pyvrp.stop import MaxRuntime
        except ImportError:
            return None
        model = pyvrp.Model()
        locations = [model.add_location(0, 0) for _ in self.distance]
        model.add_depot(locations[self.depot])
        clients = [node for node in range(self.size) if node != self.depot]
        client_index = {node: index for index, node in enumerate(clients)}
        for node in clients:
            model.add_client(locations[node], prize=self.prize[node] * (self.budget + 1), required=False)
        model.add_vehicle_type(num_available=1, max_distance=self.budget)
        for a, row in enumerate(self.distance):
            for b, value in enumerate(row):
                model.add_edge(locations[a], locations[b], distance=value)
        data = model.data()
        warm = pyvrp.Solution(data, [pyvrp.Route(data, [pyvrp.Activity(pyvrp.ActivityType.CLIENT, client_index[node])
                                     for node in incumbent[1:-1]], 0)])
        remaining = deadline - time.monotonic()
        if remaining <= 0.02:
            return incumbent
        result = model.solve(stop=MaxRuntime(remaining), seed=seed, display=False,
                             collect_stats=False, initial_solution=warm)
        if result.best.is_feasible() and len(result.best.routes()) == 1:
            candidate = [self.depot] + [clients[activity.idx] for activity in result.best.routes()[0]
                                        if activity.is_client()] + [self.depot]
            # PyVRP has no minimum-visit constraint. A too-short tour is rejected
            # here, leaving the verified three-city incumbent intact.
            if self.valid(candidate) and (self.value(candidate), -self.length(candidate)) > (self.value(incumbent), -self.length(incumbent)):
                return candidate
        return incumbent


def solve(instance, time_limit=56.0, seed=0):
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError('time_limit must be positive and finite')
    deadline = time.monotonic() + time_limit
    problem = Orienteering(instance)
    best = problem.seed(deadline)
    if problem.size > 60:
        preparation_end = min(deadline, time.monotonic() + min(7.0, time_limit * 0.15))
        best = problem.expand_seed(best, preparation_end)
    if time.monotonic() >= deadline or problem.value(best) == sum(problem.prize):
        return problem.output(best)
    if problem.size > 60 and deadline - time.monotonic() > 1:
        try:
            candidate = problem.genetic_candidate(best, deadline - 0.1, seed)
        except (ImportError, ValueError, RuntimeError, TypeError, AttributeError):
            candidate = None  # Optional acceleration must not discard a feasible tour.
        if candidate is not None:
            return problem.output(candidate)

    manager = pywrapcp.RoutingIndexManager(problem.size, 1, problem.depot)
    routing = pywrapcp.RoutingModel(manager)
    travel = routing.RegisterTransitMatrix(problem.distance)
    routing.SetArcCostEvaluatorOfAllVehicles(travel)
    routing.AddDimension(travel, 0, problem.budget, True, 'Travel')
    scale = problem.budget + 1
    if sum(problem.prize) * scale + problem.budget >= 2**62:
        raise ValueError('Objective exceeds safe integer range')
    active = []
    for node, prize in enumerate(problem.prize):
        if node == problem.depot:
            continue
        index = manager.NodeToIndex(node)
        routing.AddDisjunction([index], prize * scale)
        active.append(routing.ActiveVar(index))
    routing.solver().Add(routing.solver().Sum(active) >= 2)
    parameters = pywrapcp.DefaultRoutingSearchParameters()
    parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_INSERTION
    parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    parameters.log_search = False
    parameters.sat_parameters.num_search_workers = 1
    parameters.sat_parameters.random_seed = int(seed) & 0x7fffffff
    parameters.time_limit.FromMilliseconds(max(1, int((deadline - time.monotonic()) * 1000)))
    routing.CloseModelWithParameters(parameters)
    assignment = routing.ReadAssignmentFromRoutes([best[1:-1]], True)
    remaining = deadline - time.monotonic()
    if remaining > 0.01:
        parameters.time_limit.FromMilliseconds(max(1, int(remaining * 1000)))
        if assignment is not None:
            assignment = routing.SolveFromAssignmentWithParameters(assignment, parameters)
        else:
            assignment = routing.SolveWithParameters(parameters)
        if assignment is not None:
            index = routing.Start(0)
            candidate = [problem.depot]
            for _ in range(problem.size + 1):
                index = assignment.Value(routing.NextVar(index))
                candidate.append(manager.IndexToNode(index))
                if routing.IsEnd(index):
                    break
            if problem.valid(candidate) and (problem.value(candidate), -problem.length(candidate)) >= (problem.value(best), -problem.length(best)):
                best = candidate
    return problem.output(best)
