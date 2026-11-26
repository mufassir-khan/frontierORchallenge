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


def main(argv=None):
    began = time.monotonic()
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument('--problem', choices=['nagy2015'], default='nagy2015')
    command.add_argument('--instance', required=True, type=Path)
    command.add_argument('--output', required=True, type=Path)
    command.add_argument('--time-limit', type=float, default=60.0)
    command.add_argument('--seed', type=int, default=0)
    args = command.parse_args(argv)
    if not math.isfinite(args.time_limit) or args.time_limit <= 0:
        command.error('--time-limit must be finite and positive')
    try:
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
