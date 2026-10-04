"""Equal-effort tuning for the reference opt-aiNet formulations (same 12 held-out instances,
5 runs, 20k evaluations). 9 configurations each; sigma/aff kept at the published default."""
from paths import RES, N_JOBS, rpath, instance_filter, runs  # noqa: F401
import itertools, json, multiprocessing as mp, numpy as np
from tune import TUNE, BUDGET
from instance import random_dag
from decode import evaluator
from meta import optainet_decastro, optainet_brownlee
GRID = list(itertools.product([5, 10, 20], [2, 5, 10]))
FAM = ["dCT", "Brownlee"]
def job(a):
    fam, ci, ti, r = a
    n, m, ccr, b, seed = TUNE[ti]
    inst = random_dag(n, m, ccr, b, seed=seed); f = evaluator(inst)
    rng = np.random.default_rng(20_000 * ti + 100 * ci + r + (7 if fam == "dCT" else 13))
    N0, Nc = GRID[ci]
    res = (optainet_decastro(f, n, m, BUDGET, rng, N0=N0, Nc=Nc) if fam == "dCT"
           else optainet_brownlee(f, n, m, BUDGET, rng, pop_size=N0, Nc=Nc))
    return fam, ci, ti, res["cost"]
if __name__ == "__main__":
    jobs = [(fam, ci, ti, r) for fam in FAM for ci in range(len(GRID)) for ti in range(len(TUNE)) for r in range(5)]
    with mp.Pool(N_JOBS) as p:
        out = p.map(job, jobs, chunksize=6)
    best = {}
    for _, _, ti, c in out: best[ti] = min(best.get(ti, np.inf), c)
    sc = {}
    for fam, ci, ti, c in out: sc.setdefault((fam, ci), []).append((c - best[ti]) / best[ti])
    summ = {}
    for fam in FAM:
        rows = sorted((np.mean(sc[(fam, ci)]) * 100, GRID[ci]) for ci in range(len(GRID)))
        summ[fam] = rows; print(fam, [(r[1], round(r[0], 2)) for r in rows])
    json.dump({k: [[v[1], v[0]] for v in vs] for k, vs in summ.items()}, open(rpath("tuning_round3.json"), "w"), indent=1)
