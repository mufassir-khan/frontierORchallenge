"""Expanded-node divisible-delivery routing using constrained local search.

Two cumulative dimensions jointly model the physical load. The amount loaded at
route departure equals all deliveries assigned to that route. The signed cargo
dimension then subtracts deliveries and adds pickups, staying within capacity.
This prevents collected goods from being used to fulfil deliveries.
"""
from __future__ import annotations

import math
import random
import time

from ortools.constraint_solver import pywrapcp, routing_enums_pb2


def _integer(value, label):
    number = float(value)
    if not math.isfinite(number) or number < 0 or number != int(number):
        raise ValueError(f'{label} must be a nonnegative integer')
    return int(number)


class DeliveryRouting:
    def __init__(self, instance):
        self.n = int(instance['num_customers'])
        if self.n < 1:
            raise ValueError('At least one customer is required')
        self.capacity = _integer(instance['vehicle_capacity'], 'vehicle capacity')
        customers = {int(customer['id']): customer for customer in instance['customers']}
        if set(customers) != set(range(1, self.n + 1)):
            raise ValueError('Expected customer IDs 1..num_customers')
        self.origin = [0] + list(range(1, self.n + 1)) * 2
        self.delivery = [0] + [_integer(customers[i]['delivery_demand'], 'delivery') for i in range(1, self.n + 1)] + [0] * self.n
        self.pickup = [0] * (self.n + 1) + [_integer(customers[i]['pickup_demand'], 'pickup') for i in range(1, self.n + 1)]
        if any(quantity > self.capacity for quantity in self.delivery + self.pickup):
            raise ValueError('An indivisible request exceeds vehicle capacity')
        base = instance['distance_matrix']
        if len(base) != self.n + 1 or any(len(row) != self.n + 1 for row in base):
            raise ValueError('Wrong distance matrix dimensions')
        raw = [[_integer(v, 'distance') for v in row] for row in base]
        self.distance = [[raw[a][b] for b in self.origin] for a in self.origin]

    def cost(self, route):
        return sum(self.distance[a][b] for a, b in zip([0] + route, route + [0]))

    def feasible(self, route):
        load = sum(self.delivery[node] for node in route)
        if load > self.capacity:
            return False
        previous = 0
        for node in route:
            if previous > self.n and previous - self.n == node:
                return False
            load += self.pickup[node] - self.delivery[node]
            if not 0 <= load <= self.capacity:
                return False
            previous = node
        return True

    def starting_routes(self, deadline, seed):
        """Best-fit insertion of whole customer pairs gives a feasible warm start."""
        rng = random.Random(seed)
        order = list(range(1, self.n + 1))
        rng.shuffle(order)
        order.sort(key=lambda node: (-max(self.delivery[node], self.pickup[node + self.n]),
                                     -self.distance[0][node]))
        routes = []
        for node in order:
            block = [node, node + self.n]
            selected = None
            best_increase = self.cost(block)
            if time.monotonic() < deadline:
                for vehicle, route in enumerate(routes):
                    if sum(self.delivery[v] for v in route) + self.delivery[node] > self.capacity:
                        continue
                    if sum(self.pickup[v] for v in route) + self.pickup[node + self.n] > self.capacity:
                        continue
                    # Initially every customer is served by two consecutive,
                    # colocated expanded nodes. Subsequent search can split them.
                    for position in range(0, len(route) + 1, 2):
                        before = route[position - 1] if position else 0
                        after = route[position] if position < len(route) else 0
                        increase = self.distance[before][node] + self.distance[node][after] - self.distance[before][after]
                        if increase <= best_increase:
                            candidate = route[:position] + block + route[position:]
                            if self.feasible(candidate):
                                selected, best_increase = (vehicle, candidate), increase
            if selected is None:
                routes.append(block)
            else:
                vehicle, candidate = selected
                routes[vehicle] = candidate
        return routes

    def output(self, routes):
        visits = [node for route in routes for node in route]
        if sorted(visits) != list(range(1, 2 * self.n + 1)) or not all(self.feasible(route) for route in routes):
            raise RuntimeError('Delivery/pickup route validation failed')
        full = [[0] + route + [0] for route in routes]
        details = []
        for route in full:
            details.append([{'node_id': node,
                             'type': 'depot' if node == 0 else ('linehaul' if node <= self.n else 'backhaul'),
                             'original_customer': self.origin[node],
                             'delivery': self.delivery[node],
                             'pickup': self.pickup[node],
                             'demand': self.delivery[node] + self.pickup[node]} for node in route])
        return {'objective_value': float(sum(self.cost(route) for route in routes)),
                'routes': full, 'routes_detailed': details}


def solve(instance, time_limit=56.0, seed=0):
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError('time_limit must be positive and finite')
    deadline = time.monotonic() + time_limit
    problem = DeliveryRouting(instance)
    best = problem.starting_routes(deadline, seed)
    if time.monotonic() >= deadline:
        return problem.output(best)
    # A customer-by-customer service is always feasible with at most n vehicles.
    manager = pywrapcp.RoutingIndexManager(2 * problem.n + 1, problem.n, 0)
    routing = pywrapcp.RoutingModel(manager)
    travel = routing.RegisterTransitMatrix(problem.distance)
    routing.SetArcCostEvaluatorOfAllVehicles(travel)
    delivery = routing.RegisterUnaryTransitVector(problem.delivery)
    cargo = routing.RegisterUnaryTransitVector([p - d for p, d in zip(problem.pickup, problem.delivery)])
    routing.AddDimension(delivery, 0, problem.capacity, True, 'Deliveries')
    routing.AddDimension(cargo, 0, problem.capacity, False, 'Cargo')
    delivery_dimension = routing.GetDimensionOrDie('Deliveries')
    cargo_dimension = routing.GetDimensionOrDie('Cargo')
    for vehicle in range(problem.n):
        routing.solver().Add(cargo_dimension.CumulVar(routing.Start(vehicle)) ==
                             delivery_dimension.CumulVar(routing.End(vehicle)))
    for customer in range(1, problem.n + 1):
        routing.NextVar(manager.NodeToIndex(customer + problem.n)).RemoveValue(manager.NodeToIndex(customer))
    parameters = pywrapcp.DefaultRoutingSearchParameters()
    parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    parameters.log_search = False
    parameters.sat_parameters.num_search_workers = 1
    parameters.sat_parameters.random_seed = int(seed) & 0x7fffffff
    parameters.time_limit.FromMilliseconds(max(1, int((deadline - time.monotonic()) * 1000)))
    routing.CloseModelWithParameters(parameters)
    assignment = routing.ReadAssignmentFromRoutes(best + [[] for _ in range(problem.n - len(best))], True)
    remaining = deadline - time.monotonic()
    if remaining > 0.01 and assignment is not None:
        parameters.time_limit.FromMilliseconds(max(1, int(remaining * 1000)))
        solution = routing.SolveFromAssignmentWithParameters(assignment, parameters)
        if solution is not None:
            routes = []
            for vehicle in range(problem.n):
                index = routing.Start(vehicle)
                route = []
                for _ in range(2 * problem.n + 1):
                    index = solution.Value(routing.NextVar(index))
                    if routing.IsEnd(index):
                        break
                    route.append(manager.IndexToNode(index))
                if route:
                    routes.append(route)
            if sum(map(problem.cost, routes)) <= sum(map(problem.cost, best)):
                # Independent physical-load validation also checks coverage.
                problem.output(routes)
                best = routes
    return problem.output(best)
