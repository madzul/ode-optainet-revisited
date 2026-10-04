"""Shared settings for all scripts. Run every script from the repository root, e.g.
    python src/run_exp.py
Environment variables:
    RESULTS_DIR  where results are read and written (default: results)
    N_JOBS       worker processes (default: all CPUs)
    INSTANCES    optional comma-separated instance indices (0-50) to restrict run_exp*.py
    RUNS         optional number of runs per (instance, algorithm) for run_exp*.py (default 20)
"""
import os

RES = os.environ.get("RESULTS_DIR", "results")
N_JOBS = int(os.environ.get("N_JOBS", os.cpu_count() or 2))
os.makedirs(RES, exist_ok=True)


def rpath(name):
    return os.path.join(RES, name)


def instance_filter(n_total):
    s = os.environ.get("INSTANCES", "").strip()
    return list(range(n_total)) if not s else [int(x) for x in s.split(",") if x.strip()]


def runs(default):
    return int(os.environ.get("RUNS", default))
