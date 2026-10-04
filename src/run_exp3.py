"""CLONALG (de Castro & Von Zuben 2002). Additional competitors: reference opt-aiNet formulations (de Castro & Timmis 2002;
Brownlee 2011), with published defaults and with equal-effort tuned settings
(results/tuning_round3.json, tuning_round4.json). Same protocol as run_exp.py."""
from paths import RES, N_JOBS, rpath, instance_filter, runs  # noqa: F401
import json, multiprocessing as mp, os, time
import numpy as np
from run_exp import instances, build, at_checkpoints, BUDGET, RUNS
from decode import evaluator, heft
from meta import clonalg

ALGS2 = ["CLONALG", "CLONALG-default", "CLONALG+H"]


def run2(alg, inst, f, rng, hm):
    n, m = inst.n, inst.m
    if alg == "CLONALG":
        return clonalg(f, n, m, BUDGET, rng, N=10, Nc=20, rho=5.0)
    if alg == "CLONALG+H":
        return clonalg(f, n, m, BUDGET, rng, N=10, Nc=20, rho=5.0, seeds=[hm])
    return clonalg(f, n, m, BUDGET, rng, N=100, Nc=10, rho=2.5, d=2)


def job(a):
    ii, alg, r = a
    inst = build(instances()[ii]); f = evaluator(inst); hm = heft(inst)[0]
    rng = np.random.default_rng(3_000_017 * ii + 130363 * ALGS2.index(alg) + r + 17)
    t = time.perf_counter(); res = run2(alg, inst, f, rng, hm); dt = time.perf_counter() - t
    return dict(inst=inst.name, ii=ii, alg=alg, run=r, cost=res["cost"], reported=res["reported"],
                evals=res["evals"], time=dt, cp=at_checkpoints(res["trace"]), mapping=res["best"].tolist())


if __name__ == "__main__":
    out = rpath("main3.jsonl"); done = set()
    if os.path.exists(out):
        for l in open(out):
            d = json.loads(l); done.add((d["ii"], d["alg"], d["run"]))
    jobs = [(ii, a, r) for ii in instance_filter(len(instances())) for a in ALGS2 for r in range(runs(RUNS)) if (ii, a, r) not in done]
    print(len(jobs), "jobs", flush=True)
    with mp.Pool(N_JOBS) as p, open(out, "a") as fh:
        for k, d in enumerate(p.imap_unordered(job, jobs, chunksize=4)):
            fh.write(json.dumps(d) + "\n")
            if k % 200 == 0: fh.flush(); print(k, flush=True)
