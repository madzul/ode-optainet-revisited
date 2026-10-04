"""Statistical analysis of results/main.jsonl and results/baselines.jsonl."""
from paths import RES, N_JOBS, rpath, instance_filter, runs  # noqa: F401
import json, sys
from collections import defaultdict
import numpy as np
from scipy import stats

MAIN = ["HEFT", "CPOP", "PEFT", "Serial", "RS", "GA", "GA+H", "opt-aiNet", "ODE-diss",
        "ODE-fixed", "ODE+H"]
ABL = ["opt-aiNet", "+O", "+D", "+E", "+OD", "+OE", "+DE", "ODE-fixed"]
DET = {"HEFT": "heft", "CPOP": "cpop", "PEFT": "peft", "Serial": "serial"}


def load(path=None):
    path = RES if path is None else path
    base = {d["ii"]: d for d in map(json.loads, open(f"{path}/baselines.jsonl"))}
    runs = defaultdict(list)
    rep = defaultdict(list)
    times = defaultdict(list)
    cps = defaultdict(list)
    for d in map(json.loads, open(f"{path}/main.jsonl")):
        runs[(d["ii"], d["alg"])].append(d["cost"])
        rep[(d["ii"], d["alg"])].append(d["reported"])
        times[d["alg"]].append(d["time"])
        cps[(d["ii"], d["alg"])].append(d["cp"])
    return base, runs, rep, times, cps


def complete_instances(base, runs, algs, nruns):
    stoch = [a for a in algs if a not in DET]
    return [ii for ii in sorted(base) if all(len(runs.get((ii, a), [])) >= nruns for a in stoch)]


def value(base, runs, ii, a, how=np.mean):
    if a in DET:
        return base[ii][DET[a]]
    return how(runs[(ii, a)])


def best_known(base, runs, ii):
    vals = [base[ii][k] for k in DET.values()]
    vals += [min(v) for (j, _), v in runs.items() if j == ii]
    return min(vals)


def holm(pvals):
    k = len(pvals); order = np.argsort(pvals); adj = np.empty(k); run = 0
    for r, i in enumerate(order):
        run = max(run, min(1, (k - r) * pvals[i])); adj[i] = run
    return adj


def friedman(base, runs, insts, algs):
    M = np.array([[value(base, runs, ii, a) for a in algs] for ii in insts])
    ranks = np.array([stats.rankdata(r) for r in M])
    chi, p = stats.friedmanchisquare(*M.T)
    return M, ranks.mean(axis=0), chi, p


def pairwise(M, algs, ref):
    j = algs.index(ref); out = []
    for k, a in enumerate(algs):
        if k == j:
            continue
        diff = M[:, k] - M[:, j]
        if np.allclose(diff, 0):
            out.append((a, 1.0, 0, 0, len(diff))); continue
        p = stats.wilcoxon(M[:, k], M[:, j], zero_method="zsplit").pvalue
        out.append((a, p, int((diff > 1e-9).sum()), int((diff < -1e-9).sum()),
                    int((abs(diff) <= 1e-9).sum())))
    adj = holm(np.array([o[1] for o in out]))
    return [(o[0], o[1], adj[i], o[2], o[3], o[4]) for i, o in enumerate(out)]


def gmean_ratio(base, runs, insts, a, ref="HEFT"):
    r = [value(base, runs, ii, a) / value(base, runs, ii, ref) for ii in insts]
    return float(np.exp(np.mean(np.log(r))))


def report(nruns=20):
    base, runs, rep, times, cps = load()
    insts = complete_instances(base, runs, MAIN + ABL, nruns)
    print(f"# complete instances: {len(insts)} / {len(base)}")
    out = {"n_instances": len(insts)}

    # --- Table: Friedman over main algorithms -------------------------------------------
    M, avg_rank, chi, p = friedman(base, runs, insts, MAIN)
    print(f"\nFriedman (main, k={len(MAIN)}): chi2={chi:.1f}, p={p:.2e}")
    rpd = {a: np.mean([(value(base, runs, ii, a) - best_known(base, runs, ii)) /
                       best_known(base, runs, ii) * 100 for ii in insts]) for a in MAIN}
    gm = {a: gmean_ratio(base, runs, insts, a) for a in MAIN}
    for a, r in sorted(zip(MAIN, avg_rank), key=lambda x: x[1]):
        print(f"  {a:10s} avg rank {r:5.2f}  mean RPD {rpd[a]:7.2f}%  gmean(ms/HEFT) {gm[a]:.3f}")
    out["main"] = dict(chi2=chi, p=p, rank=dict(zip(MAIN, avg_rank.tolist())), rpd=rpd, gm=gm)
    for ref in ["ODE-fixed", "HEFT", "GA"]:
        print(f"\n  Wilcoxon vs {ref} (Holm): alg  p  p_holm  worse better ties")
        pw = pairwise(M, MAIN, ref)
        for row in pw:
            print("   {:10s} {:.2e} {:.2e} {:3d} {:3d} {:3d}".format(*row))
        out[f"pairwise_{ref}"] = pw

    # --- groups by CCR / n / m / kind ---------------------------------------------------
    groups = defaultdict(list)
    for ii in insts:
        b = base[ii]
        groups[("ccr", "low" if b["ccr"] < 0.5 else "mid" if b["ccr"] < 3 else "high")].append(ii)
        groups[("m", b["m"])].append(ii)
        groups[("kind", b["kind"] if b["kind"] in ("random", "fft16", "gauss10") else "diss")].append(ii)
        if b["kind"] == "random":
            groups[("n", b["n"])].append(ii)
    out["groups"] = {}
    print("\nGeometric-mean makespan ratio to HEFT by group (lower is better)")
    hdr = ["PEFT", "Serial", "RS", "GA", "GA+H", "opt-aiNet", "ODE-diss", "ODE-fixed", "ODE+H"]
    print("  group        n  " + " ".join(f"{h:>9s}" for h in hdr))
    for g in sorted(groups, key=str):
        row = [gmean_ratio(base, runs, groups[g], a) for a in hdr]
        out["groups"][f"{g[0]}={g[1]}"] = dict(n=len(groups[g]), **dict(zip(hdr, row)))
        print(f"  {g[0]+'='+str(g[1]):11s} {len(groups[g]):2d}  " + " ".join(f"{v:9.3f}" for v in row))

    # --- beats HEFT / Serial counts (per run) ------------------------------------------
    print("\nShare of runs strictly better than HEFT / better than min(HEFT,Serial)")
    out["beat"] = {}
    for a in ["RS", "GA", "GA+H", "opt-aiNet", "ODE-diss", "ODE-fixed", "ODE+H"]:
        bh = np.mean([c < base[ii]["heft"] - 1e-9 for ii in insts for c in runs[(ii, a)]])
        bs = np.mean([c < min(base[ii]["heft"], base[ii]["serial"]) - 1e-9 for ii in insts for c in runs[(ii, a)]])
        out["beat"][a] = (bh, bs)
        print(f"  {a:10s} {100*bh:5.1f}%  {100*bs:5.1f}%")
    hs = np.mean([base[ii]["serial"] < base[ii]["heft"] for ii in insts])
    print(f"  instances where Serial < HEFT: {100*hs:.1f}%")
    out["serial_better_than_heft"] = hs

    # --- Ablation ------------------------------------------------------------------------
    MA, rk, chi, p = friedman(base, runs, insts, ABL)
    print(f"\nAblation Friedman: chi2={chi:.1f}, p={p:.2e}")
    pw = pairwise(MA, ABL, "opt-aiNet")
    rpdA = {a: np.mean([(value(base, runs, ii, a) - best_known(base, runs, ii)) /
                        best_known(base, runs, ii) * 100 for ii in insts]) for a in ABL}
    print("  variant    avg rank  mean RPD  p_holm(vs base)  worse better")
    out["ablation"] = {}
    for a, r in zip(ABL, rk):
        row = next((x for x in pw if x[0] == a), None)
        ph = f"{row[2]:.2e}" if row else "   -    "
        wb = f"{row[3]:3d} {row[4]:3d}" if row else ""
        print(f"  {a:10s} {r:6.2f}  {rpdA[a]:7.2f}%   {ph}   {wb}")
        out["ablation"][a] = dict(rank=r, rpd=rpdA[a], p_holm=row[2] if row else None,
                                  worse=row[3] if row else None, better=row[4] if row else None)

    # --- Reported vs true cost (ODE-diss) ------------------------------------------------
    diffs = [(r - c) / c for ii in insts for r, c in zip(rep[(ii, "ODE-diss")], runs[(ii, "ODE-diss")])]
    diffs = np.array(diffs)
    out["ode_diss_misreport"] = dict(frac=float((np.abs(diffs) > 1e-9).mean()),
                                     mean_under=float(diffs[diffs < -1e-9].mean()) if (diffs < -1e-9).any() else 0.0,
                                     worst=float(diffs.min()))
    print(f"\nODE-diss: reported != true in {100*(np.abs(diffs)>1e-9).mean():.1f}% of runs; "
          f"mean understatement {100*out['ode_diss_misreport']['mean_under']:.1f}%, worst {100*diffs.min():.1f}%")
    # --- anytime ---------------------------------------------------------------------------
    CHECK = [500, 1000, 2000, 5000, 10000, 20000]
    out["anytime"] = {}
    for a in ["RS", "GA", "opt-aiNet", "ODE-diss", "ODE-fixed", "ODE+H", "GA+H"]:
        vals = []
        for k in range(len(CHECK)):
            v = [np.mean([c[k] for c in cps[(ii, a)]]) / base[ii]["heft"] for ii in insts]
            vals.append(float(np.exp(np.mean(np.log(v)))))
        out["anytime"][a] = vals
    print("\nAnytime gmean(best/HEFT) at", CHECK)
    for a, v in out["anytime"].items():
        print(f"  {a:10s} " + " ".join(f"{x:.3f}" for x in v))
    out["time"] = {a: float(np.mean(t)) for a, t in times.items()}
    print("\nMean wall time per run (s):", {a: round(v, 2) for a, v in out["time"].items()})
    json.dump(out, open(rpath("summary.json"), "w"), indent=1, default=float)
    return out


if __name__ == "__main__":
    report(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
