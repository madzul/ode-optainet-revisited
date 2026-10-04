"""Merged analysis: main.jsonl + main2.jsonl (reference opt-aiNet formulations)."""
from paths import RES, N_JOBS, rpath, instance_filter, runs  # noqa: F401
import json
from collections import defaultdict
import numpy as np
from scipy import stats
from analyze import DET, holm, best_known

def load_all():
    base = {d["ii"]: d for d in map(json.loads, open(rpath("baselines.jsonl")))}
    runs, rep, times, cps = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for fn in [rpath("main.jsonl"), rpath("main2.jsonl"), rpath("main3.jsonl")]:
        for d in map(json.loads, open(fn)):
            runs[(d["ii"], d["alg"])].append(d["cost"]); rep[(d["ii"], d["alg"])].append(d["reported"])
            times[d["alg"]].append(d["time"]); cps[(d["ii"], d["alg"])].append(d["cp"])
    return base, runs, rep, times, cps

def val(base, runs, ii, a):
    return base[ii][DET[a]] if a in DET else float(np.mean(runs[(ii, a)]))

if __name__ == "__main__":
    base, runs, rep, times, cps = load_all()
    I = sorted(base)
    ALG = ["HEFT", "PEFT", "GA", "GA+H", "CLONALG", "CLONALG-default", "CLONALG+H", "opt-aiNet",
           "Brownlee-tuned", "dCT-tuned", "ODE-fixed", "ODE+H", "ODE-diss"]
    I = [ii for ii in I if all(a in DET or len(runs[(ii, a)]) >= 20 for a in ALG)]
    print("instances", len(I))
    M = np.array([[val(base, runs, ii, a) for a in ALG] for ii in I])
    rk = np.array([stats.rankdata(r) for r in M]).mean(0)
    chi, p = stats.friedmanchisquare(*M.T)
    print(f"Friedman chi2={chi:.1f} p={p:.2e}")
    bk = {ii: min(best_known(base, runs, ii), *[min(runs[(ii, a)]) for a in ALG if a not in DET]) for ii in I}
    out = {}
    for j, a in sorted(enumerate(ALG), key=lambda t: rk[t[0]]):
        rpd = np.mean([(M[k, j] - bk[ii]) / bk[ii] * 100 for k, ii in enumerate(I)])
        gm = float(np.exp(np.mean(np.log(M[:, j] / M[:, ALG.index("HEFT")]))))
        hb = int((M[:, j] < M[:, 0] - 1e-9).sum()); hw = int((M[:, j] > M[:, 0] + 1e-9).sum())
        t = np.mean(times[a]) if a not in DET else 0
        out[a] = dict(rank=float(rk[j]), rpd=float(rpd), gm=gm, better=hb, worse=hw, time=float(t))
        print(f"  {a:15s} rank {rk[j]:5.2f} RPD {rpd:6.1f} gm {gm:.3f} vs HEFT better/worse {hb}/{hw} time {t:.2f}")
    def pw(a, b):
        x, y = M[:, ALG.index(a)], M[:, ALG.index(b)]; d = x - y
        return int((d < -1e-9).sum()), int((d > 1e-9).sum()), int((abs(d) <= 1e-9).sum()), stats.wilcoxon(x, y, zero_method="zsplit").pvalue
    pairs = [("CLONALG", "ODE-fixed"), ("CLONALG", "dCT-tuned"), ("CLONALG", "GA"), ("CLONALG", "HEFT"),
             ("CLONALG", "PEFT"), ("CLONALG-default", "ODE-fixed"), ("CLONALG+H", "GA+H"), ("CLONALG+H", "ODE+H"),
             ("CLONALG", "CLONALG-default")]
    res = [pw(a, b) for a, b in pairs]; adj = holm(np.array([r[3] for r in res]))
    print("pairwise (first better / worse / ties, p, p_holm)")
    out["pairs"] = []
    for (a, b), r, q in zip(pairs, res, adj):
        print(f"  {a:15s} vs {b:15s} {r[0]:2d}/{r[1]:2d}/{r[2]:2d}  p={r[3]:.2e} holm={q:.2e}")
        out["pairs"].append([a, b, *r, float(q)])
    grp = defaultdict(list)
    for k, ii in enumerate(I):
        c = base[ii]["ccr"]; grp["low" if c < 0.5 else "mid" if c < 3 else "high"].append(k)
    print("gmean ratio to HEFT by CCR")
    out["ccr"] = {}
    for g in ["low", "mid", "high"]:
        ks = grp[g]
        row = {a: float(np.exp(np.mean(np.log(M[ks, ALG.index(a)] / M[ks, 0])))) for a in ALG}
        out["ccr"][g] = row
        print("  ", g, {a: round(v, 3) for a, v in row.items()})
    CH = [500, 1000, 2000, 5000, 10000, 20000]
    out["anytime"] = {}
    for a in ["GA", "ODE-fixed", "dCT-tuned", "CLONALG", "CLONALG-default", "CLONALG+H", "GA+H", "ODE+H"]:
        v = [float(np.exp(np.mean([np.log(np.mean([c[k] for c in cps[(ii, a)]]) / base[ii]["heft"]) for ii in I]))) for k in range(6)]
        out["anytime"][a] = v; print("  anytime", a, [round(x, 3) for x in v])
    json.dump(out, open(rpath("summary3.json"), "w"), indent=1)
