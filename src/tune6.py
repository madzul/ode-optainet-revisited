"""Equal-effort tuning for CLONALG (12 configurations, same held-out instances)."""
from paths import RES, N_JOBS, rpath, instance_filter, runs  # noqa: F401
import itertools, json, multiprocessing as mp, numpy as np
from tune import TUNE, BUDGET
from instance import random_dag
from decode import evaluator
from meta import clonalg
GRID = [(10, 20, 5.0), (10, 20, 7.5), (10, 20, 10.0), (5, 5, 7.5), (5, 5, 10.0), (10, 10, 7.5)]
def job(a):
    ci, ti, r = a
    n, m, ccr, b, seed = TUNE[ti]
    inst = random_dag(n, m, ccr, b, seed=seed); f = evaluator(inst)
    N, Nc, rho = GRID[ci]
    return ci, ti, clonalg(f, n, m, BUDGET, np.random.default_rng(30_000 * ti + 100 * ci + r), N=N, Nc=Nc, rho=rho)["cost"]
if __name__ == "__main__":
    jobs = [(ci, ti, r) for ci in range(len(GRID)) for ti in range(len(TUNE)) for r in range(5)]
    with mp.Pool(N_JOBS) as p: out = p.map(job, jobs, chunksize=6)
    best = {}
    for _, ti, c in out: best[ti] = min(best.get(ti, np.inf), c)
    sc = {}
    for ci, ti, c in out: sc.setdefault(ci, []).append((c - best[ti]) / best[ti])
    rows = sorted((np.mean(v) * 100, GRID[ci]) for ci, v in sc.items())
    print([(r[1], round(r[0], 2)) for r in rows])
    json.dump([[list(r[1]), r[0]] for r in rows], open(rpath("tuning_clonalg2.json"), "w"), indent=1)
