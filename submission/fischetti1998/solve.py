#!/usr/bin/env python3
"""FrontierOR entry point for fischetti1998."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

# Bound numerical-library threads before importing the model or its dependencies.
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--problem', choices=['fischetti1998'], default='fischetti1998')
    parser.add_argument('--instance', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--time-limit', type=float, default=60.0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if not math.isfinite(args.time_limit) or args.time_limit <= 0:
        parser.error('--time-limit must be positive and finite')
    try:
        with open(args.instance, encoding='utf-8') as handle:
            instance = json.load(handle)
        from model import solve
        reserve = min(4.0, args.time_limit * 0.15)
        remaining = args.time_limit - reserve - (time.monotonic() - started)
        if remaining <= 0:
            raise RuntimeError('Time budget exhausted while loading input/libraries')
        solution = solve(instance, time_limit=remaining, seed=args.seed)
        payload = json.dumps(solution, allow_nan=False, separators=(',', ':'))
        if len(payload.encode('utf-8')) > 16 * 1024 * 1024:
            raise RuntimeError('Solution exceeds the challenge 16 MB output limit')
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('w', encoding='utf-8') as handle:
            handle.write(payload + '\n')
    except (ValueError, KeyError, TypeError, RuntimeError, OSError, ImportError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
