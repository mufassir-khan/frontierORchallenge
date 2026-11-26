#!/usr/bin/env python3
"""FrontierOR entry point with an independent wall-clock worker watchdog."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='2'
sys.dont_write_bytecode=True


def run_worker(command,checkpoint,stop):
    """The parent never enters a native optimizer and can always stop its worker."""
    process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
                             stderr=None,start_new_session=True)
    try:
        try:
            process.wait(timeout=max(0.001,stop-time.monotonic()))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL)
            process.wait(timeout=1)
    finally:
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGKILL)
            process.wait(timeout=1)
    # Atomic worker publication guarantees a fully written, validated incumbent,
    # even if optimization crashes or has to be killed after finding that cover.
    if not checkpoint.exists():
        raise RuntimeError('Search ended without a feasible cover; no invalid solution was written')


def main(argv=None):
    start=time.monotonic()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--problem',choices=['hoffman1993'],default='hoffman1993')
    parser.add_argument('--instance',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--time-limit',type=float,default=60)
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args(argv)
    if not math.isfinite(args.time_limit) or args.time_limit<=0:
        parser.error('--time-limit must be positive and finite')
    try:
        if args.instance.resolve()==args.output.resolve():
            raise ValueError('Input and output paths must differ')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        reserve=min(5.0,args.time_limit*0.15)
        stop=start+args.time_limit-reserve
        # Checkpoints live in the permitted output directory, not the read-only
        # algorithm folder; a unique directory prevents stale-solution reuse.
        with tempfile.TemporaryDirectory(prefix='.hoffman-',dir=args.output.parent) as directory:
            checkpoint=Path(directory)/'incumbent.json'
            command=[sys.executable,'-B',str(Path(__file__).resolve().with_name('model.py')),
                     '--instance',str(args.instance.resolve()),'--checkpoint',str(checkpoint),
                     '--deadline',str(stop-min(1.0,reserve/2)),'--seed',str(args.seed)]
            run_worker(command,checkpoint,stop)
            answer=json.loads(checkpoint.read_text())
            if not math.isfinite(answer['objective_value']):raise ValueError('Nonfinite objective')
            checkpoint.replace(args.output)
        return 0
    except (OSError,ValueError,KeyError,RuntimeError,subprocess.SubprocessError) as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
