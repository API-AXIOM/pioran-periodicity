#!/usr/bin/env python3
"""Stall watchdog for a run_workers.sh campaign.

Why: run_workers.sh's supervisor restarts a worker when run_sim.py *exits*
(e.g. a Julia GC segfault). It cannot see a worker that hangs without
exiting -- which is what killed the 2026-08-31 null campaign: 8 of 12
workers wedged inside ijl_gc_collect, burning ~90% CPU while writing
nothing to disk for 30+ hours.

How: each live run_sim.py holds its current fit's checkpoint files open. The
newest mtime among the campaign files a PID has open is its progress
signature. If that signature does not advance for --stall-min minutes, the
process is SIGKILLed (SIGTERM does not work: a GC-wedged process never
services it). The supervisor then restarts it and run_sim.py resumes that fit
from its points.hdf5, so a kill costs only the work since the last checkpoint.

Usage: nohup python3 watchdog.py <campaign-dir> [--stall-min 45] [--poll 300] &
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from typing import Dict, Optional, Tuple


def log(path: str, msg: str) -> None:
    with open(path, "a") as fh:
        fh.write(f"{time.strftime('%F %T')} {msg}\n")
        fh.flush()


def worker_pids(tag: str) -> list:
    """PIDs of live run_sim.py processes for this campaign (not the conda-run
    wrappers, which never hold the checkpoint files open)."""
    out = subprocess.run(["ps", "-eo", "pid=,command="], capture_output=True, text=True).stdout
    pids = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        pid, _, cmd = line.partition(" ")
        if "run_sim.py" in cmd and tag in cmd and "conda" not in cmd.split("run_sim.py")[0]:
            try:
                pids.append(int(pid))
            except ValueError:
                pass
    return pids


def progress_sig(pid: int, campaign_dir: str) -> Optional[float]:
    """Newest mtime among campaign files this PID holds open, or None if it
    holds none (between fits -- no evidence of a stall)."""
    out = subprocess.run(["lsof", "-p", str(pid)], capture_output=True, text=True).stdout
    newest = None
    for line in out.splitlines():
        path = line.rsplit(" ", 1)[-1].strip()
        if not path.startswith(campaign_dir):
            continue
        try:
            m = os.path.getmtime(path)
        except OSError:
            continue
        if newest is None or m > newest:
            newest = m
    return newest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("campaign_dir")
    ap.add_argument("--stall-min", type=float, default=45.0)
    ap.add_argument("--poll", type=float, default=300.0)
    args = ap.parse_args()

    campaign_dir = os.path.abspath(args.campaign_dir)
    tag = os.path.basename(campaign_dir)
    logfile = os.path.join(campaign_dir, "logs", "watchdog.log")
    log(logfile, f"watchdog start: dir={campaign_dir} stall_min={args.stall_min} poll_s={args.poll}")

    # pid -> (last progress signature, wall time we first saw that signature)
    seen: Dict[int, Tuple[float, float]] = {}
    while True:
        pids = worker_pids(tag)
        if not pids:
            log(logfile, f"no run_sim.py workers left for {tag} -- watchdog exiting")
            return 0
        for pid in pids:
            sig = progress_sig(pid, campaign_dir)
            if sig is None:
                continue
            prev = seen.get(pid)
            if prev is None or prev[0] != sig:
                seen[pid] = (sig, time.time())
                continue
            stalled_min = (time.time() - prev[1]) / 60.0
            if stalled_min >= args.stall_min:
                log(logfile, f"KILL pid={pid} -- no checkpoint progress for "
                             f"{stalled_min:.0f}min (supervisor will restart and resume)")
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError as exc:
                    log(logfile, f"  kill pid={pid} failed: {exc}")
                seen[pid] = (sig, time.time())
        for dead in set(seen) - set(pids):
            del seen[dead]
        time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())
