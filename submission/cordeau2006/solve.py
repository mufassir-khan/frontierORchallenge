#!/usr/bin/env python3
"""DARP entry point implementing the FrontierOR sandbox command-line contract."""
import argparse
import json
import math
import sys
import time

from model import solve


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--problem', choices=['cordeau2006'], default='cordeau2006')
    parser.add_argument('--instance', dest='instance_path', help='JSON instance path')
    parser.add_argument('instance', nargs='?', help='Legacy positional JSON path, or - for stdin')
    parser.add_argument('-o', '--output', default='-', help='JSON output path, or - for stdout')
    parser.add_argument('--time-limit', type=float, default=60.0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.instance_path is not None and args.instance is not None:
        parser.error('Use either --instance or a positional instance path, not both.')
    if not math.isfinite(args.time_limit) or args.time_limit <= 0:
        parser.error('--time-limit must be positive and finite.')
    instance_path = args.instance_path or args.instance or '-'
    try:
        if instance_path == '-':
            instance = json.load(sys.stdin)
        else:
            with open(instance_path, encoding='utf-8') as handle:
                instance = json.load(handle)
        # The sandbox kills the whole process at the limit, even with output.
        # Include input reading in the budget and reserve time to validate/write.
        reserve = min(3.0, args.time_limit * 0.1)
        budget = args.time_limit - reserve - (time.monotonic() - started)
        if budget <= 0:
            raise RuntimeError('Time budget exhausted while reading the instance.')
        solution = solve(instance, budget, args.seed)
        payload = json.dumps(solution, indent=2, allow_nan=False) + '\n'
        if args.output == '-':
            sys.stdout.write(payload)
        else:
            with open(args.output, 'w', encoding='utf-8') as handle:
                handle.write(payload)
    except (ValueError, KeyError, TypeError, RuntimeError, OSError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
