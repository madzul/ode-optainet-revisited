"""Equal-effort parameter tuning on held-out instances (seeds 1000+), 12 configurations per
algorithm family, 5 runs each, budget 20k evaluations. Score = mean RPD to the best value
found on each tuning instance."""
from paths import RES, N_JOBS, rpath, instance_filter, runs  # noqa: F401
import itertools, json, multiprocessing as mp, numpy as np
from instance import random_dag
from decode import evaluator
from meta import optainet, ga

BUDGET = 20000
TUNE = [(n, 8, ccr, b, 1000 + k) for k, (n, ccr, b) in
        enumerate(itertools.product([30, 80], [0.1, 1.0, 10.0], [0.5, 1.5]))]

CFG = {
    "ODE": [dict(N0=N0, Nc=Nc, sigma=s) for N0, Nc, s in
            itertools.product([5, 10], [2, 3, 5], [0.02, 0.05])],
    "GA": [dict(pop=p, pm=pm, pc=pc) for p, pm, pc in
           itertools.product([50, 100], [1.0, 2.0], [0.4, 0.6])],
}


def job(args):
    fam, ci, ti, r = args
    n, m, ccr, b, seed = TUNE[ti]
    inst = random_dag(n, m, ccr, b, seed=seed)
    f = evaluator(inst)
    rng = np.random.default_rng(10_000 * ti + 100 * ci + r)
    p = CFG[fam][ci]
    if fam == "ODE":
        res = optainet(f, n, m, BUDGET, rng, O=True, D=True, E=True, **p)
    else:
        res = ga(f, n, m, BUDGET, rng, pop=p["pop"], pm=p["pm"] / n, pc=p["pc"])
    return fam, ci, ti, res["cost"]


if __name__ == "__main__":
    jobs = [(fam, ci, ti, r) for fam in CFG for ci in range(len(CFG[fam]))
            for ti in range(len(TUNE)) for r in range(5)]
    with mp.Pool(N_JOBS) as pool:
        out = pool.map(job, jobs, chunksize=8)
    best = {}
    for _, _, ti, c in out:
        best[ti] = min(best.get(ti, np.inf), c)
    score = {}
    for fam, ci, ti, c in out:
        score.setdefault((fam, ci), []).append((c - best[ti]) / best[ti])
    summary = {}
    for fam in CFG:
        rows = sorted((np.mean(score[(fam, ci)]), ci) for ci in range(len(CFG[fam])))
        summary[fam] = {"chosen": CFG[fam][rows[0][1]],
                        "table": [(CFG[fam][ci], round(100 * s, 2)) for s, ci in rows]}
        print(fam, "chosen", CFG[fam][rows[0][1]])
        for cfg, s in summary[fam]["table"]:
            print("   ", cfg, f"meanRPD={s}%")
    json.dump(summary, open(rpath("tuning_round2.json"), "w"), indent=1)
