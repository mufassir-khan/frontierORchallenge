# FrontierOR Solver Writeup

This submission contains six optimization algorithms, each in its own problem folder with `solve.py` as the entry point and `model.py` as the implementation. The common approach is to find a feasible solution, improve its objective within the time budget, and validate the final decisions before returning them. The methods combine constructive heuristics, mathematical programming, and open-source routing solvers; they do not guarantee global optimality within 60 seconds.

## Dial a Ride Problem — cordeau2006

**Problem.** Transport every passenger from pickup to drop-off while minimizing total distance. Each pair must use the same vehicle, with pickup first, while respecting capacity, service windows, maximum ride time, and route duration.

**Algorithm.** A deadline-bounded insertion heuristic constructs an initial route plan. OR-Tools then improves it using pickup-and-delivery constraints and guided local search. Integer-scaled travel times make the routing model conservative. An independent scheduling check uses the original Euclidean distances to verify time windows, ride times, and capacity, including cases requiring delayed depot departure. Only solutions serving every request are accepted.

[Implementation](submission/cordeau2006/model.py)

## Orienteering Problem — fischetti1998

**Problem.** Select a subset of prize-bearing cities and visit them in one depot-to-depot tour within a travel budget. Maximize collected prize while visiting at least three distinct cities, including the depot.

**Algorithm.** Construct a feasible starting cycle, then improve larger-instance seeds through prize-aware insertion and 2-opt route shortening. PyVRP uses iterated local search to improve tours when installed; OR-Tools provides the fallback and handles smaller instances. Scaled prize penalties prioritize collected reward over travel savings. The final validator checks connectivity through the tour sequence, unique visits, minimum tour size, and budget compliance. VROOM was tested but was not retained for this problem because its measured result was weaker.

[Implementation](submission/fischetti1998/model.py)

## Integer Multicommodity Flow — barnhart2000

**Problem.** Route each commodity along one directed path without splitting its demand or exceeding shared arc capacities. Commodities may instead be rejected. Minimize routing and rejection costs; rejection contributes demand multiplied by its artificial-arc cost.

**Algorithm.** Small instances use an arc-based binary optimization model. Larger instances start with capacity-aware greedy paths, then use linear-programming capacity prices to generate alternative shortest paths. A binary path-selection model chooses compatible routes. Randomized removal and rerouting explores residual capacity and supplies additional paths for a final optimization. SciPy/HiGHS solves the mathematical programs, while an independently validated feasible assignment is retained throughout.

[Implementation](submission/barnhart2000/model.py)

## Airline Crew Set Partitioning — hoffman1993

**Problem.** Choose crew rotations of minimum total cost so that every flight leg is covered exactly once, while satisfying any crew-base utilization bounds.

**Algorithm.** A bitset implementation of Algorithm X branches on the least flexible uncovered flight and tries input-order and cost-per-leg construction strategies. OR-Tools CP-SAT then searches for or improves a feasible cover. Base constraints and selected rotations are validated independently. Every improving feasible solution is checkpointed atomically. A separate supervisor enforces the wall-clock deadline and can return the saved solution even if the optimization worker crashes or overruns its budget.

[Implementation](submission/hoffman1993/model.py)

## Stochastic Facility Location — bodur2017

**Problem.** Choose facilities before uncertain customer demand is realized, then allocate shipments for each demand scenario. Minimize opening costs plus probability-weighted transportation costs while meeting demand and respecting facility capacities.

**Algorithm.** A sampled Benders approach selects facilities: a binary master proposes openings, and transportation linear programs generate cuts estimating recourse cost. A capacity constraint covers the largest total demand across all scenarios. Every original scenario receives explicit feasible shipments, with transportation LPs improving an initial greedy allocation. When time remains, a richer scenario sample generates another opening plan; the solver retains the better plan evaluated over the full scenario set.

[Implementation](submission/bodur2017/model.py)

## Divisible Delivery and Pickup Routing — nagy2015

**Problem.** Minimize vehicle travel while serving every customer's delivery and pickup demands. Delivery and pickup may use different vehicles, but each individual request must be served entirely on one visit. Remaining deliveries and collected pickups jointly consume capacity.

**Algorithm.** Represent each customer with separate delivery and pickup nodes and build an initial feasible plan. VROOM constructs an improved candidate using supplied distance matrices without a routing server. PyVRP then improves the best plan through iterated local search. OR-Tools provides a fallback using coupled delivery and cargo dimensions. Independent checks enforce complete service, capacity throughout each route, and the checker's restrictions on colocated visits.

[Implementation](submission/nagy2015/model.py)

## Execution and validation

All entry points accept `--problem`, `--instance`, `--output`, and `--time-limit`. Search budgets leave time to write the solution and exit. Each problem folder is self-contained, and runtime computation uses no network services.

For the optional routing improvements, enter these versions in the challenge's **Extra packages** field; including the requirements file alone does not install them:

```text
pyvroom==1.15.2
pyvrp==0.14.0
```

Local validation passed **16 public benchmark cases within 60 seconds**. These are feasibility and runtime checks, not confirmation of private leaderboard scores. Regression tests also cover timing, capacity, rational base constraints, optional-engine failures, and recovery from optimization-worker failures.

See the [validation report](tests/optimization_report.json), [dependency list](submission/requirements.txt), and [official submission contract](https://frontieror-challenge.com/docs/main/submission-format).

## Current execution modes

The five solvers `cordeau2006`, `fischetti1998`, `barnhart2000`, `hoffman1993`, and `bodur2017` have been restored to their pre-portfolio implementations from commit `10caccd` (matching the saved experimental baselines). Their original solver budgets and native parallelism are preserved, including the crew solver's checkpoint watchdog.

`nagy2015` retains the two-process, four-round portfolio, shared warm starts, and detailed terminal logging. Its entry point accepts `--portfolio` and `--no-portfolio`; portfolio mode remains the default. The benchmark runner applies those overrides only to `nagy2015` and continues to display runtime, objective, and checker results for all problems.

The downloaded `instances` folders are preserved. The historical [portfolio report](tests/portfolio/summary.json) records the earlier experiment, rather than the current configuration.
