"""Experiment 2: replication of the dissertation's setting (200 generations, unlimited
evaluations, append decoder) to quantify (a) the effect of the two defects in Lampiran A
and (b) where the reported 5-50x speed-up of ODE over opt-aiNet comes from."""
from paths import RES, N_JOBS, rpath, instance_filter, runs  # noqa: F401
import json, multiprocessing as mp, os, time
import numpy as np
from instance import random_dag, dissertation_fft8, dissertation_random22, dissertation_input0002
from decode import evaluator
from meta import optainet, ode_dissertation

RUNS = 10
RUNS_OPT = 5
INF = 10**9
DISS = dict(N0=20, Nc=10, sigma=0.1)   # dissertation settings: popl 20, 10 clones


def insts():
    out = [dissertation_fft8(16), dissertation_random22(16), dissertation_input0002()]
    out += [random_dag(50, m, 1.0, 1.0, seed=11) for m in (4, 16)]
    return out


VARIANTS = ["opt-aiNet(200g)", "ODE-fixed(200g)", "ODE-diss", "ODE-diss+costfix",
            "ODE-diss+costfix+mutfix"]


def job(a):
    ii, v, r = a
    inst = insts()[ii]
    f = evaluator(inst, insertion=False)          # dissertation decoder (append)
    n, m = inst.n, inst.m
    rng = np.random.default_rng(97 * ii + 13 * VARIANTS.index(v) + 100_000 * r)
    t = time.perf_counter()
    if v == "opt-aiNet(200g)":
        res = optainet(f, n, m, INF, rng, max_gen=200, **DISS)
    elif v == "ODE-fixed(200g)":
        res = optainet(f, n, m, INF, rng, O=True, D=True, E=True, max_gen=200, **DISS)
    else:
        res = ode_dissertation(f, n, m, INF, rng, max_gen=200,
                               fix_cost="costfix" in v, fix_mutation="mutfix" in v)
    return dict(inst=inst.name, ii=ii, var=v, run=r, cost=res["cost"], reported=res["reported"],
                evals=res["evals"], time=time.perf_counter() - t)


if __name__ == "__main__":
    jobs = [(ii, v, r) for ii in range(len(insts())) for v in VARIANTS
            for r in range(RUNS_OPT if v == 'opt-aiNet(200g)' else RUNS)]
    jobs.sort(key=lambda j: j[1] == 'opt-aiNet(200g)')
    with mp.Pool(N_JOBS) as pool, open(rpath("bug.jsonl"), "w") as fh:
        for d in pool.imap_unordered(job, jobs, chunksize=2):
            fh.write(json.dumps(d) + "\n"); fh.flush()
    print("done")
