"""Exact-cover construction, incumbent-preserving CP-SAT, and full validation.

All choices depend only on the supplied instance. Integer-bitset Algorithm X
tries two general branch orders; a feasibility-first CP-SAT search then improves
the best cover. Rational base rows are scaled exactly, never silently rounded.
The entry point supervises this worker independently of native solver limits.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import sys
import time


class CoverProblem:
    def __init__(self, instance):
        self.m = int(instance['dimensions']['num_rows'])
        self.n = int(instance['dimensions']['num_cols'])
        self.cost = list(map(float, instance['cost_vector'][:self.n]))
        if self.m < 1 or self.n < 1 or len(self.cost) != self.n or not all(map(math.isfinite,self.cost)):
            raise ValueError('Invalid rotation dimensions or costs')
        raw = instance['constraint_matrix_A']['columns']
        if len(raw) < self.n:
            raise ValueError('Missing coverage columns')
        # Match the published checker's handling of stale out-of-range entries.
        self.cover = [tuple(sorted({r for r in col if 0 <= r < self.m})) for col in raw[:self.n]]
        self.masks = [sum(1 << r for r in col) for col in self.cover]
        self.rowbits = [0]*self.m
        self.rows = [[] for _ in range(self.m)]
        for j,col in enumerate(self.cover):
            bit = 1 << j
            for r in col:
                self.rowbits[r] |= bit
                self.rows[r].append(j)
        self.bases = []
        spec = instance.get('base_constraints')
        if instance.get('has_base_constraints') and spec:
            for b in range(int(spec['num_constraints'])):
                coeff = list(map(float,spec['D_matrix']['rows'][b][:self.n]))
                lo,hi = float(spec['lower_bounds_d1'][b]),float(spec['upper_bounds_d2'][b])
                if len(coeff)!=self.n or not all(map(math.isfinite,coeff+[lo,hi])) or lo>hi:
                    raise ValueError('Invalid crew base row')
                self.bases.append((coeff,lo,hi))
        self.best = None
        self.best_cost = math.inf

    def valid(self, chosen):
        if len(set(chosen)) != len(chosen) or any(j<0 or j>=self.n for j in chosen):
            return False
        covered = 0
        for j in chosen:
            if covered & self.masks[j]:
                return False
            covered |= self.masks[j]
        if covered != (1<<self.m)-1:
            return False
        return all(lo-1e-7 <= math.fsum(coef[j] for j in chosen) <= hi+1e-7 for coef,lo,hi in self.bases)

    def accept(self, chosen, publish):
        chosen = sorted(chosen)
        value = math.fsum(self.cost[j] for j in chosen)
        if value < self.best_cost-1e-8 and self.valid(chosen):
            self.best, self.best_cost = chosen, value
            if publish:
                publish(self.output())
            return True
        return False

    def output(self):
        if self.best is None or not self.valid(self.best):
            raise RuntimeError('No feasible crew cover found within the time budget')
        selected = set(self.best)
        return dict(objective_value=float(self.best_cost), selected_rotations=self.best,
                    variable_values={str(j):float(j in selected) for j in range(self.n)})

    def seed(self, deadline, mode, publish):
        """Algorithm X with bit-parallel compatibility, bounded at every node.

        Branch first on the least flexible uncovered flight. Input-order and
        cost-per-leg orders are separate construction attempts, not assumptions
        about which rotations form a solution. A failed attempt proves nothing.
        """
        if self.m >= 700:
            return  # Avoid Python recursion limits; CP-SAT handles these cases.
        cache = set()
        nonnegative_base = all(all(v>=0 for v in row) for row,_,_ in self.bases)
        empty = [j for j,mask in enumerate(self.masks) if not mask and self.cost[j]<0] if not self.bases else []

        def search(todo, available, chosen, totals):
            if time.monotonic() >= deadline:
                return True
            if not todo:
                return self.accept(chosen+empty,publish)
            if not self.bases and todo in cache:
                return False
            options,count,left = 0,self.n+1,todo
            while left:
                bit = left & -left
                candidates = self.rowbits[bit.bit_length()-1] & available
                number = candidates.bit_count()
                if not number:
                    return False
                if number<count:
                    count,options = number,candidates
                    if count==1:
                        break
                left ^= bit
            choices = []
            while options:
                bit=options&-options; choices.append(bit.bit_length()-1); options^=bit
            if mode=='cost':
                choices.sort(key=lambda j:self.cost[j]/len(self.cover[j]))
            for j in choices:
                if time.monotonic() >= deadline:
                    return True
                update = [value+row[j] for value,(row,_,_) in zip(totals,self.bases)]
                if nonnegative_base and any(value>hi+1e-7 for value,(_,_,hi) in zip(update,self.bases)):
                    continue
                conflict=0
                for r in self.cover[j]:
                    conflict |= self.rowbits[r]
                if search(todo^self.masks[j],available&~conflict,chosen+[j],update):
                    return True
            if not self.bases and len(cache)<100000:
                cache.add(todo)
            return False
        search((1<<self.m)-1,(1<<self.n)-1,[],[0.0]*len(self.bases))


def integer_row(coeff, lower, upper):
    """Exact decimal-rational conversion with a checked int64 activity bound."""
    fractions = [Fraction(str(v)) for v in coeff]
    scale = 1
    for value in fractions:
        scale = math.lcm(scale,value.denominator)
    nums = [value.numerator*(scale//value.denominator) for value in fractions]
    divisor = 0
    for value in nums:
        divisor=math.gcd(divisor,abs(value))
    divisor=max(1,divisor)
    nums=[value//divisor for value in nums]
    lo=math.ceil(Fraction(str(lower))*scale/divisor)
    hi=math.floor(Fraction(str(upper))*scale/divisor)
    if max(sum(map(abs,nums)),abs(lo),abs(hi))>=2**60:
        raise OverflowError('Base row cannot be represented exactly by int64')
    return nums,lo,hi


def _float_search(problem, deadline, publish):
    """Fallback for exceptional rational rows that cannot fit CP-SAT integers."""
    import numpy as np
    from scipy.optimize import Bounds,LinearConstraint,milp
    from scipy.sparse import coo_matrix,vstack,csc_matrix
    rr=[];cc=[]
    for j,col in enumerate(problem.cover):
        rr.extend(col);cc.extend([j]*len(col))
    matrix=coo_matrix((np.ones(len(rr)),(rr,cc)),shape=(problem.m,problem.n)).tocsc()
    lo=np.ones(problem.m);hi=lo.copy()
    if problem.bases:
        matrix=vstack([matrix,csc_matrix([row for row,_,_ in problem.bases])],format='csc')
        lo=np.r_[lo,[lower for _,lower,_ in problem.bases]]
        hi=np.r_[hi,[upper for _,_,upper in problem.bases]]
    for feasibility in (True,False):
        remaining=deadline-time.monotonic()
        if remaining<0.05:break
        if feasibility and problem.best is not None:continue
        result=milp(np.zeros(problem.n) if feasibility else problem.cost,integrality=np.ones(problem.n),
                    bounds=Bounds(0,1),constraints=LinearConstraint(matrix,lo,hi),
                    options={'time_limit':remaining*(0.45 if feasibility else 0.95),'mip_rel_gap':0.001})
        if result.x is not None:problem.accept(np.flatnonzero(result.x>0.5).tolist(),publish)


def solve(instance,time_limit=50.0,seed=0,on_incumbent=None,deadline=None):
    start=time.monotonic()
    if not math.isfinite(time_limit) or time_limit<=0:
        raise ValueError('time_limit must be positive and finite')
    deadline=min(deadline,start+time_limit) if deadline is not None else start+time_limit
    problem=CoverProblem(instance)
    # Save feasible solutions BEFORE native presolve/search can consume the budget.
    for mode in ('input','cost'):
        if time.monotonic()>=deadline:break
        problem.seed(min(deadline,time.monotonic()+min(2.0,time_limit*0.06)),mode,on_incumbent)
    if deadline-time.monotonic()<0.05:return problem.output()
    from ortools.sat.python import cp_model
    model=cp_model.CpModel()
    x=[model.new_bool_var(f'x{j}') for j in range(problem.n)]
    for row in problem.rows:
        model.add_exactly_one([x[j] for j in row])
    try:
        for row,lo,hi in problem.bases:
            coeff,lower,upper=integer_row(row,lo,hi)
            model.add_linear_constraint(cp_model.LinearExpr.weighted_sum(x,coeff),lower,upper)
    except OverflowError:
        _float_search(problem,deadline,on_incumbent)
        return problem.output()

    class Incumbent(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            problem.accept([j for j,v in enumerate(x) if self.value(v)],on_incumbent)

    for feasibility in (True,False):
        remaining=deadline-time.monotonic()
        if remaining<0.05:break
        if feasibility and problem.best is not None:continue
        if not feasibility:
            # Costs are integer in the challenge schema; retain exact arithmetic.
            objective=cp_model.LinearExpr.weighted_sum(x,[int(v) if v.is_integer() else v for v in problem.cost])
            model.minimize(objective)
            if problem.best is not None:
                chosen=set(problem.best)
                for j,var in enumerate(x):model.add_hint(var,int(j in chosen))
        solver=cp_model.CpSolver()
        solver.parameters.num_search_workers=2
        solver.parameters.random_seed=int(seed)&0x7fffffff
        solver.parameters.max_time_in_seconds=remaining*(0.45 if feasibility else 0.96)
        solver.parameters.stop_after_first_solution=feasibility
        solver.parameters.log_search_progress=False
        status=solver.solve(model,Incumbent())
        if status in (cp_model.FEASIBLE,cp_model.OPTIMAL):
            problem.accept([j for j,v in enumerate(x) if solver.value(v)],on_incumbent)
        if status==cp_model.INFEASIBLE and problem.best is None:
            raise RuntimeError('The crew-cover model is infeasible')
        if status==cp_model.MODEL_INVALID:
            raise RuntimeError('CP-SAT rejected model: '+solver.solution_info())
    return problem.output()


def worker():
    parser=argparse.ArgumentParser()
    parser.add_argument('--instance',required=True,type=Path)
    parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--deadline',required=True,type=float)
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args()
    def publish(solution):
        payload=(json.dumps(solution,allow_nan=False,separators=(',',':'))+'\n').encode()
        if len(payload)>16*1024*1024:raise ValueError('Solution exceeds 16 MB')
        temporary=args.checkpoint.with_suffix('.tmp')
        temporary.write_bytes(payload)
        temporary.replace(args.checkpoint)
    try:
        instance=json.loads(args.instance.read_text())
        solution=solve(instance,max(0.001,args.deadline-time.monotonic()),args.seed,publish,args.deadline)
        publish(solution)
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr)
        return 1
    return 0


if __name__=='__main__':
    raise SystemExit(worker())
