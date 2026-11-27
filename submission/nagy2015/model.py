"""Expanded-node delivery routing with offline VROOM/PyVRP candidates.

Optional engines warm-start and improve a validated route plan. OR-Tools is
the dependency-free fallback within the challenge image.

Two cumulative dimensions jointly model the physical load. The amount loaded at
route departure equals all deliveries assigned to that route. The signed cargo
dimension then subtracts deliveries and adds pickups, staying within capacity.
This prevents collected goods from being used to fulfil deliveries.
"""
from __future__ import annotations

import math
import random
import time
from datetime import timedelta

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

    def repair_colocated_order(self, routes):
        # Reversing an adjacent own-pickup/own-delivery pair leaves distance
        # unchanged and weakly decreases the intervening onboard load.
        for route in routes:
            for position in range(len(route) - 1):
                if route[position] > self.n and route[position] - self.n == route[position + 1]:
                    route[position], route[position + 1] = route[position + 1], route[position]
        return routes

    def vroom_candidate(self, seconds):
        """Optional PyPI VROOM binding: offline matrices, no routing server."""
        try:
            import vroom
        except ImportError:
            return None
        model = vroom.Input()
        model.set_durations_matrix(profile='car', matrix_input=self.distance)
        model.add_vehicle([vroom.Vehicle(vehicle, start=0, end=0, capacity=[self.capacity])
                           for vehicle in range(self.n)])
        model.add_job([vroom.Job(node, location=node, delivery=[self.delivery[node]],
                                pickup=[self.pickup[node]]) for node in range(1, 2 * self.n + 1)])
        result = model.solve(exploration_level=5, nb_threads=1 if __import__("os").environ.get("PORTFOLIO_WORKER") else 2,
                             timeout=timedelta(seconds=max(0.01, seconds))).to_dict()
        if result.get('unassigned'):
            return None
        routes = [[step['id'] for step in route['steps'] if step['type'] == 'job']
                  for route in result.get('routes', [])]
        routes = self.repair_colocated_order([route for route in routes if route])
        self.output(routes)
        return routes

    def genetic_candidate(self, incumbent, deadline, seed):
        """Optional PyVRP hybrid genetic search, seeded by a feasible plan."""
        try:
            import pyvrp
            from pyvrp.stop import MaxRuntime
        except ImportError:
            return None
        model = pyvrp.Model()
        locations = [model.add_location(0, 0) for _ in self.distance]
        model.add_depot(locations[0])
        for node in range(1, 2 * self.n + 1):
            model.add_client(locations[node], delivery=self.delivery[node], pickup=self.pickup[node])
        model.add_vehicle_type(num_available=self.n, capacity=self.capacity)
        for a, row in enumerate(self.distance):
            for b, cost in enumerate(row):
                model.add_edge(locations[a], locations[b], distance=cost)
        data = model.data()
        warm = pyvrp.Solution(data, [pyvrp.Route(data, [pyvrp.Activity(pyvrp.ActivityType.CLIENT, node - 1)
                                  for node in route], 0) for route in incumbent])
        remaining = deadline - time.monotonic()
        if remaining <= 0.02:
            return incumbent
        result = model.solve(stop=MaxRuntime(remaining), seed=seed, display=False,
                             collect_stats=False, initial_solution=warm)
        if not result.best.is_feasible():
            return incumbent
        routes = [[activity.idx + 1 for activity in route if activity.is_client()]
                  for route in result.best.routes()]
        routes = self.repair_colocated_order(routes)
        self.output(routes)
        return routes


def solve(instance, time_limit=56.0, seed=0, warm_start=None, search_method="guided"):
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError('time_limit must be positive and finite')
    deadline = time.monotonic() + time_limit
    problem = DeliveryRouting(instance)
    build_started=time.monotonic()
    best = problem.starting_routes(deadline, seed)
    build_seconds=time.monotonic()-build_started
    if warm_start is not None:
        routes = [r[1:-1] for r in warm_start['routes']]
        problem.output(routes)
        if sum(map(problem.cost, routes)) < sum(map(problem.cost, best)):
            best = routes
    def progress(stage):
        import os, sys
        if os.environ.get('FRONTIER_PROGRESS') == '1':
            print(f'[worker seed={seed}] {stage}; best objective={sum(map(problem.cost, best))}',file=sys.stderr,flush=True)
    import os, sys
    if os.environ.get('FRONTIER_PROGRESS') == '1':
        problem.output(best)
        worker=os.environ.get('FRONTIER_WORKER', '1')
        print(f'Worker {worker}: initial feasible objective={sum(map(problem.cost,best))}, routes={len(best)}, seed built in {build_seconds:.2f}s',file=sys.stderr,flush=True)
    progress('initial solution ready')
    remaining = deadline - time.monotonic()
    if remaining > 1 and search_method != 'annealing':
        try:
            progress('starting VROOM')
            candidate = problem.vroom_candidate(min(12.0, remaining * 0.25))
        except (ImportError, ValueError, RuntimeError, TypeError, AttributeError):
            candidate = None  # Keep the validated seed if an optional engine fails.
        if candidate is not None and sum(map(problem.cost, candidate)) < sum(map(problem.cost, best)):
            best = candidate
    progress('VROOM stage complete or skipped')
    if deadline - time.monotonic() > 1 and search_method != 'annealing':
        try:
            progress('starting PyVRP')
            candidate = problem.genetic_candidate(best, deadline - 0.1, seed)
        except (ImportError, ValueError, RuntimeError, TypeError, AttributeError):
            candidate = None
        if candidate is not None:
            if sum(map(problem.cost, candidate)) < sum(map(problem.cost, best)):
                best = candidate
            progress('PyVRP complete')
            return problem.output(best)
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
    parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING if search_method == 'annealing' else routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    parameters.log_search = False
    parameters.sat_parameters.num_search_workers = 1
    parameters.sat_parameters.random_seed = int(seed) & 0x7fffffff
    parameters.time_limit.FromMilliseconds(max(1, int((deadline - time.monotonic()) * 1000)))
    routing.CloseModelWithParameters(parameters)
    assignment = routing.ReadAssignmentFromRoutes(best + [[] for _ in range(problem.n - len(best))], True)
    remaining = deadline - time.monotonic()
    if remaining > 0.01 and assignment is not None:
        parameters.time_limit.FromMilliseconds(max(1, int(remaining * 1000)))
        progress('starting OR-Tools local search')
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
