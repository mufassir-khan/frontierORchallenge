"""Divisible delivery/pickup routing with capacity-checked destroy/repair.

Each request is an indivisible expanded node. Delivery and pickup of one
customer may be on different routes. Load counts all remaining deliveries
plus all collected pickups, so pickup goods cannot satisfy delivery demand.
"""
import math
import random
import time


def solve(instance, time_limit=56.0, seed=0):
    deadline = time.monotonic() + time_limit
    rng = random.Random(seed)
    n = int(instance['num_customers'])
    cap = float(instance['vehicle_capacity'])
    customers = {int(c['id']): c for c in instance['customers']}
    if set(customers) != set(range(1, n+1)):
        raise ValueError('Expected customer IDs 1..num_customers')
    raw = instance['distance_matrix']
    original = [0] + list(range(1, n+1)) * 2
    dist = [[float(raw[a][b]) for b in original] for a in original]
    quantity = [0.0] + [float(customers[i]['delivery_demand']) for i in range(1, n+1)] + [float(customers[i]['pickup_demand']) for i in range(1, n+1)]
    if any(q < 0 or q > cap for q in quantity):
        raise ValueError('An indivisible request exceeds capacity or is negative')

    def feasible(route):
        load = sum(quantity[v] for v in route if v <= n)
        if load > cap + 1e-8:
            return False
        previous = 0
        for v in route:
            if previous > n and previous - n == v:
                return False  # Published checker forbids this backward pair arc.
            load += quantity[v] if v > n else -quantity[v]
            if load > cap + 1e-8 or load < -1e-8:
                return False
            previous = v
        return True

    def cost(route):
        return sum(dist[a][b] for a, b in zip([0] + route, route + [0]))

    def normalize(route):
        # Removing nodes can expose a forbidden pickup->own-delivery arc.
        # Swapping this colocated pair reduces load between the two stops.
        for i in range(len(route)-1):
            if route[i] > n and route[i]-n == route[i+1]:
                route[i], route[i+1] = route[i+1], route[i]
        return route

    def improve(route, stop):
        route = route[:]
        for _ in range(3):
            changed = False
            for i in range(len(route)-1):
                if time.monotonic() >= stop:
                    return route
                for j in range(i+1, len(route)):
                    a = route[i-1] if i else 0
                    b = route[j+1] if j+1 < len(route) else 0
                    if dist[a][route[j]] + dist[route[i]][b] < dist[a][route[i]] + dist[route[j]][b] - 1e-8:
                        candidate = route[:i] + list(reversed(route[i:j+1])) + route[j+1:]
                        if feasible(candidate):
                            route, changed = candidate, True
            if not changed:
                break
        return route

    # Start with one feasible vehicle per customer, then merge by endpoint savings.
    routes = [[i, n+i] for i in range(1, n+1)]
    owner = {i: i-1 for i in range(1, n+1)}
    savings = sorted(((dist[0][a] + dist[0][b] - dist[a][b], a, b)
                      for a in range(1, n+1) for b in range(a+1, n+1)), reverse=True)
    for saving, a, b in savings:
        if time.monotonic() >= deadline or saving < -1e-8:
            break
        i, j = owner[a], owner[b]
        if i == j or not routes[i] or not routes[j]:
            continue
        possibilities = [routes[i] + routes[j], routes[j] + routes[i]]
        candidate = min((r for r in possibilities if feasible(r)), key=cost, default=None)
        if candidate is not None and cost(candidate) <= cost(routes[i]) + cost(routes[j]) + 1e-8:
            routes[i], routes[j] = candidate, []
            for v in candidate:
                owner[original[v]] = i
    routes = [improve(r, deadline) for r in routes if r]
    best = [r[:] for r in routes]
    best_cost = sum(map(cost, routes))
    current_cost = best_cost
    iteration = 0
    while time.monotonic() < deadline and n > 1:
        iteration += 1
        all_nodes = [v for r in routes for v in r]
        count = rng.randint(1, max(2, min(len(all_nodes), math.ceil(len(all_nodes)*0.18))))
        if iteration % 3 == 0:
            center = rng.choice(all_nodes)
            remove = set(sorted(all_nodes, key=lambda v: dist[center][v] + rng.random()*3)[:count])
        else:
            remove = set(rng.sample(all_nodes, min(count, len(all_nodes))))
        partial = [normalize([v for v in r if v not in remove]) for r in routes]
        partial = [r for r in partial if r]
        pending = list(remove)
        rng.shuffle(pending)
        if iteration % 2:
            pending.sort(key=lambda v: -quantity[v])
        noise = rng.choice([0, 0, 0.05, 0.15])
        complete = True
        for node in pending:
            if time.monotonic() >= deadline:
                complete = False
                break
            chosen = (2 * dist[0][node], len(partial), [node])
            chosen_score = chosen[0]
            for k, route in enumerate(partial):
                if time.monotonic() >= deadline:
                    complete = False
                    break
                for pos in range(len(route)+1):
                    a, b = (route[pos-1] if pos else 0), (route[pos] if pos < len(route) else 0)
                    delta = dist[a][node] + dist[node][b] - dist[a][b]
                    score = delta * (1 + noise * rng.random())
                    if score < chosen_score + 1e-9:
                        candidate = route[:pos] + [node] + route[pos:]
                        if feasible(candidate):
                            chosen, chosen_score = (delta, k, candidate), score
            if not complete:
                break
            _, k, candidate = chosen
            if k == len(partial):
                partial.append(candidate)
            else:
                partial[k] = candidate
        if not complete:
            break
        partial = [improve(r, deadline) for r in partial]
        value = sum(map(cost, partial))
        if value < best_cost - 1e-8:
            best, best_cost = [r[:] for r in partial], value
        fraction = max(0, (deadline-time.monotonic()) / max(time_limit, 1e-9))
        temperature = max(1e-8, best_cost * 0.005 * fraction)
        if value <= current_cost or rng.random() < math.exp(-(value-current_cost)/temperature):
            routes, current_cost = partial, value
        if iteration % 50 == 0:
            routes, current_cost = [r[:] for r in best], best_cost
    if sorted(v for r in best for v in r) != list(range(1, 2*n+1)) or not all(map(feasible, best)):
        raise RuntimeError('Final route validation failed')
    output_routes = [[0] + r + [0] for r in best]
    detail = [[{'node_id': v, 'type': 'depot' if v == 0 else ('linehaul' if v <= n else 'backhaul'),
                'original_customer': original[v], 'demand': quantity[v],
                'delivery': quantity[v] if 1 <= v <= n else 0,
                'pickup': quantity[v] if v > n else 0} for v in r] for r in output_routes]
    return {'objective_value': float(best_cost), 'routes': output_routes, 'routes_detailed': detail}
