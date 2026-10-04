"""Main experiment. Writes one JSON line per (instance, algorithm, run) to results/main.jsonl
and deterministic baselines to results/baselines.jsonl."""
from paths import RES, N_JOBS, rpath, instance_filter, runs  # noqa: F401
import itertools, json, multiprocessing as mp, os, sys, time
import numpy as np
from instance import random_dag, fft_dag, gauss_dag, dissertation_fft8, dissertation_random22, dissertation_input0002
from decode import evaluator, heft, cpop, peft
from meta import optainet, ode_dissertation, ga, random_search

BUDGET = 20000
RUNS = 20
CHECK = [500, 1000, 2000, 5000, 10000, 20000]
AIS = dict(N0=5, Nc=2, sigma=0.05)          # tuned (results/tuning_round2.json)
GA = dict(pop=50, pc=0.6)                    # pm = 1/n


def instances():
    out = []
    for n, m, ccr, b in itertools.product([20, 50, 100], [4, 16], [0.1, 1.0, 10.0], [0.5, 1.5]):
        out.append(("random", (n, m, ccr, b)))
    for kind, m, ccr in itertools.product(["fft16", "gauss10"], [4, 16], [0.1, 1.0, 10.0]):
        out.append((kind, (m, ccr)))
    out += [("diss_fft8", ()), ("diss_rand22", ()), ("diss_in0002", ())]
    return out


def build(spec, seed=7):
    kind, a = spec
    if kind == "random":
        return random_dag(*a, seed=seed)
    if kind == "fft16":
        return fft_dag(16, a[0], a[1], 1.0, seed=seed)
    if kind == "gauss10":
        return gauss_dag(10, a[0], a[1], 1.0, seed=seed)
    return {"diss_fft8": dissertation_fft8, "diss_rand22": dissertation_random22,
            "diss_in0002": dissertation_input0002}[kind]()


ALGS = ["RS", "GA", "GA+H", "opt-aiNet", "+O", "+D", "+E", "+OD", "+OE", "+DE",
        "ODE-fixed", "ODE+H", "ODE-diss"]


def run_alg(alg, inst, f, rng, heft_map):
    n, m = inst.n, inst.m
    if alg == "RS":
        return random_search(f, n, m, BUDGET, rng)
    if alg == "GA":
        return ga(f, n, m, BUDGET, rng, pm=1.0 / n, **GA)
    if alg == "GA+H":
        return ga(f, n, m, BUDGET, rng, pm=1.0 / n, seeds=[heft_map], **GA)
    if alg == "ODE-diss":
        return ode_dissertation(f, n, m, BUDGET, rng)
    if alg == "ODE+H":
        return optainet(f, n, m, BUDGET, rng, O=True, D=True, E=True, seeds=[heft_map], **AIS)
    flags = {"opt-aiNet": "", "ODE-fixed": "ODE"}.get(alg, alg.lstrip("+"))
    return optainet(f, n, m, BUDGET, rng, O="O" in flags, D="D" in flags, E="E" in flags, **AIS)


def at_checkpoints(trace):
    out, j, best = [], 0, np.inf
    for c in CHECK:
        while j < len(trace) and trace[j][0] <= c:
            best = trace[j][1]; j += 1
        out.append(best)
    return out


def job(args):
    ii, alg, r = args
    inst = build(instances()[ii])
    f = evaluator(inst, insertion=True)
    heft_map = heft(inst)[0]
    rng = np.random.default_rng(1_000_003 * ii + 7919 * ALGS.index(alg) + r)
    t = time.perf_counter()
    res = run_alg(alg, inst, f, rng, heft_map)
    dt = time.perf_counter() - t
    return dict(inst=inst.name, ii=ii, alg=alg, run=r, cost=res["cost"], reported=res["reported"],
                evals=res["evals"], time=dt, cp=at_checkpoints(res["trace"]),
                mapping=res["best"].tolist())


def baselines():
    rows = []
    for ii, spec in enumerate(instances()):
        inst = build(spec)
        # lower bound: longest path with min execution cost and zero communication
        lb_cp = np.zeros(inst.n); wmin = inst.W.min(axis=1)
        for v in inst.topo_order():
            lb_cp[v] = wmin[v] + max([lb_cp[u] for u, _ in inst.preds[v]], default=0.0)
        rows.append(dict(inst=inst.name, ii=ii, kind=spec[0], n=inst.n, m=inst.m,
                         ccr=round(inst.ccr(), 3), beta=inst.meta.get("beta"),
                         heft=heft(inst)[1], cpop=cpop(inst)[1], peft=peft(inst)[1],
                         serial=float(inst.W.sum(axis=0).min()),
                         lb=float(max(lb_cp.max(), inst.W.min(axis=1).sum() / inst.m))))
    with open(rpath("baselines.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return rows


if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True)
    baselines()
    out = rpath("main.jsonl")
    done = set()
    if os.path.exists(out):
        for line in open(out):
            d = json.loads(line); done.add((d["ii"], d["alg"], d["run"]))
    jobs = [(ii, a, r) for ii in instance_filter(len(instances())) for a in ALGS for r in range(runs(RUNS))
            if (ii, a, r) not in done]
    print(f"{len(jobs)} jobs to run", flush=True)
    with mp.Pool(N_JOBS) as pool, open(out, "a") as fh:
        for k, d in enumerate(pool.imap_unordered(job, jobs, chunksize=4)):
            fh.write(json.dumps(d) + "\n")
            if k % 200 == 0:
                fh.flush(); print(f"{k}/{len(jobs)}", flush=True)
