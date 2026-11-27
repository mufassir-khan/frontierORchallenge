#!/usr/bin/env python3
"""Standalone FrontierOR runner for nagy2015."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

# The evaluation image gives each process two CPU cores.
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '2'
sys.dont_write_bytecode = True



def progress(message):
    if os.environ.get('FRONTIER_PROGRESS') == '1':
        print(message, file=sys.stderr, flush=True)


def portfolio_main(args, began):
    """Four cooperative rounds, two single-core processes, one best incumbent."""
    import subprocess
    import tempfile
    worker_code = '''import json,sys,time
started=time.monotonic()
from model import solve
source,warm,target,budget,seed,method=sys.argv[1:]
with open(source) as f: data=json.load(f)
with open(warm) as f: incumbent=json.load(f)
remaining=float(budget)-(time.monotonic()-started)
if remaining<=0: raise SystemExit(1)
if __import__('os').environ.get('FRONTIER_PROGRESS') == '1':
 print(f'[worker seed={seed} method={method}] searching; warm_start={incumbent.get("objective_value") if incumbent else None}',file=sys.stderr,flush=True)
result=solve(data,time_limit=remaining,seed=int(seed),warm_start=incumbent,search_method=method)
if __import__('os').environ.get('FRONTIER_PROGRESS') == '1':
 print(f'[worker seed={seed} method={method}] finished objective={result["objective_value"]}',file=sys.stderr,flush=True)
from pathlib import Path
p=Path(target); t=p.with_suffix('.tmp');t.write_text(json.dumps(result,allow_nan=False));t.replace(p)
'''
    if args.instance.resolve()==args.output.resolve():
        raise ValueError('Input and output must differ')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    stop=began+args.time_limit-min(5.0,args.time_limit*.15)
    best=None
    data=json.loads(args.instance.read_text())
    delivery=sum(c['delivery_demand'] for c in data['customers'])
    pickup=sum(c['pickup_demand'] for c in data['customers'])
    capacity=data['vehicle_capacity']
    progress('Starting VRPDDP optimization...')
    progress(f"\n========== INSTANCE ==========\nCustomers: {data['num_customers']}\nExpanded nodes: {2*data['num_customers']+1}\nCapacity: {capacity}\nTotal delivery: {delivery}\nTotal pickup: {pickup}\nLower-bound vehicles: {math.ceil(max(delivery,pickup)/capacity)}\nWorkers: 2\nRounds: 4\n==============================\n")
    with tempfile.TemporaryDirectory(prefix='.portfolio-',dir=args.output.parent) as directory:
        directory=Path(directory)
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PORTFOLIO_WORKER='1')
        for round_number in range(4):
            available=stop-time.monotonic()
            if available<.1: break
            duration=available/(4-round_number)
            round_stop=time.monotonic()+duration
            warm=directory/'warm.json';warm.write_text(json.dumps(best))
            progress(f'round {round_number+1}/4 started; budget={duration:.1f}s; shared best={best["objective_value"] if best else "pending"}')
            jobs=[]
            try:
                for lane,method in enumerate(('guided','annealing')):
                    target=directory/f'{round_number}-{lane}.json'
                    command=[sys.executable,'-B','-c',worker_code,str(args.instance.resolve()),str(warm),str(target),
                             str(max(.01,duration-.5)),str(args.seed+100*round_number+lane),method]
                    worker_env=dict(env,FRONTIER_WORKER=str(lane+1))
                    progress(f'Started worker {lane+1}/2 with strategy {method} (seed {args.seed+100*round_number+lane})')
                    process=subprocess.Popen(command,cwd=Path(__file__).resolve().parent,env=worker_env,
                        stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=None if env.get('FRONTIER_PROGRESS') == '1' else subprocess.DEVNULL)
                    jobs.append((process,target))
                next_update=time.monotonic()+5
                reported=set()
                while any(process.poll() is None for process,target in jobs) and time.monotonic()<round_stop:
                    for lane,(process,target) in enumerate(jobs):
                        if process.poll() is not None and lane not in reported:
                            progress(f'round {round_number+1}/4 worker {lane+1} exited code={process.returncode}')
                            reported.add(lane)
                    if time.monotonic()>=next_update:
                        active=[str(lane+1) for lane,(process,target) in enumerate(jobs) if process.poll() is None]
                        progress(f'round {round_number+1}/4 workers {",".join(active)} running; shared best={best["objective_value"] if best else "pending"}; remaining={max(0,round_stop-time.monotonic()):.1f}s')
                        next_update=time.monotonic()+5
                    time.sleep(min(.1,max(0,round_stop-time.monotonic())))
            finally:
                for process,target in jobs:
                    if process.poll() is None:
                        progress(f'round {round_number+1}/4 worker deadline reached; stopping pid={process.pid}')
                        process.kill()
                    process.wait(timeout=1)
            for lane,(process,target) in enumerate(jobs):
                if target.exists():
                    candidate=json.loads(target.read_text());value=candidate['objective_value']
                    if not math.isfinite(value):continue
                    progress(f'Worker {lane+1} objective: {value:g}, routes: {len(candidate["routes"])}')
                    if best is None or (value>best['objective_value'] if args.problem=='fischetti1998' else value<best['objective_value']):
                        best=candidate
                        progress(f'NEW BEST objective: {value:g} (worker {lane+1})')
            progress(f'round {round_number+1}/4 complete; best={best["objective_value"] if best else "pending"}')
        if best is None:raise RuntimeError('Portfolio found no feasible solution')
        payload=json.dumps(best,allow_nan=False,separators=(',',':'))+'\n'
        if len(payload.encode())>16*1024*1024:raise ValueError('Solution exceeds16 MB')
        checkpoint=directory/'final.json';checkpoint.write_text(payload);checkpoint.replace(args.output)
    progress(f'\n========== FINAL RESULT ==========\nFinal objective: {best["objective_value"]:g}\nNumber of routes: {len(best["routes"])}\nWorkers: 2\nTotal solve time: {time.monotonic()-began:.2f}s\nExternal validation: pending benchmark checker\n==================================')
    return 0


def main(argv=None):
    began = time.monotonic()
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument('--problem', choices=['nagy2015'], default='nagy2015')
    command.add_argument('--instance', required=True, type=Path)
    command.add_argument('--output', required=True, type=Path)
    command.add_argument('--time-limit', type=float, default=60.0)
    command.add_argument('--seed', type=int, default=0)
    command.add_argument('--portfolio', action=argparse.BooleanOptionalAction, default=True)
    args = command.parse_args(argv)
    if not math.isfinite(args.time_limit) or args.time_limit <= 0:
        command.error('--time-limit must be finite and positive')
    try:
        if args.portfolio:
            return portfolio_main(args, began)
        with args.instance.open(encoding='utf-8') as source:
            instance = json.load(source)
        from model import solve
        # Reading/imports count toward the wall clock. Allow time for independent
        # validation, JSON serialization and process shutdown before the kill.
        reserve = min(5.0, 0.15 * args.time_limit)
        available = args.time_limit - reserve - (time.monotonic() - began)
        if available <= 0:
            raise RuntimeError('No search time remains after input loading')
        solution = solve(instance, time_limit=available, seed=args.seed)
        objective = solution.get('objective_value')
        if isinstance(objective, bool) or not isinstance(objective, (int, float)) or not math.isfinite(objective):
            raise ValueError('The solver returned an invalid objective_value')
        payload = (json.dumps(solution, allow_nan=False, separators=(',', ':')) + '\n').encode('utf-8')
        if len(payload) > 16 * 1024 * 1024:
            raise ValueError('Solution exceeds the 16 MB output limit')
        if args.output.resolve() == args.instance.resolve():
            raise ValueError('The output path must differ from the read-only instance')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + '.tmp')
        temporary.write_bytes(payload)
        temporary.replace(args.output)
        return 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, ImportError) as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
