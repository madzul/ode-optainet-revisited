"""Generate notebooks/colab_quickstart.ipynb and notebooks/kaggle_quickstart.ipynb from one
template, so the two versions differ only in their setup cell.

    python tools/build_notebooks.py
"""
import nbformat as nbf

REPO = "madzul/ode-optainet-revisited"
URL = f"https://github.com/{REPO}"
COLAB = f"https://colab.research.google.com/github/{REPO}/blob/main/notebooks/colab_quickstart.ipynb"
KAGGLE = f"https://kaggle.com/kernels/welcome?src={URL}/blob/main/notebooks/kaggle_quickstart.ipynb"

INTRO = f"""# Revisiting ODE opt-aiNet for heterogeneous DAG scheduling — {{platform}} quickstart

Code and data for the equal-budget re-evaluation of **ODE opt-aiNet** (Zulianto, ITB doctoral
dissertation, 2016) against HEFT, CPOP, PEFT, a genetic algorithm, CLONALG and the published
formulations of opt-aiNet. Repository: [{REPO}]({URL}).

This notebook:

1. reproduces the schedules printed in the dissertation,
2. runs the algorithms on one instance and draws the schedules,
3. checks that a re-run gives **bit-identical** results to the archived raw data,
4. recomputes the statistics and figures of the paper from the raw data,
5. shows the two defects found in the original implementation,
6. optionally re-runs the full experiment (about 4 h on 2 cores).

Sections 1–5 take a few minutes on a free {{platform}} instance."""

SETUP = {
    "Colab": [
        ("md", "## 0. Setup\nClones the repository and installs the dependencies "
               "(numpy, scipy, numba, matplotlib, pandas)."),
        ("code", f"""import os, sys
if not os.path.isdir("ode-optainet-revisited"):
    !git clone -q {URL}.git
%cd ode-optainet-revisited
!pip install -q -r requirements.txt
sys.path.insert(0, "src")
print("CPUs available:", os.cpu_count())"""),
    ],
    "Kaggle": [
        ("md", "## 0. Setup\n**Turn on internet access first:** *Settings → Internet → On* "
               "(requires a phone-verified Kaggle account). The cell clones the repository into "
               "`/kaggle/working` and installs the dependencies; most are already in the Kaggle image."),
        ("code", f"""import os, sys
os.chdir("/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd())
if not os.path.isdir("ode-optainet-revisited"):
    !git clone -q {URL}.git
%cd ode-optainet-revisited
!pip install -q -r requirements.txt
sys.path.insert(0, "src")
print("CPUs available:", os.cpu_count())"""),
    ],
}

BODY = [
    ("md", """## 1. Reproduce the dissertation's schedules
The shared decoder reproduces every schedule printed in the dissertation: HEFT on FFT-8 with 16
processors (Listing C.3, 11,528), the ODE opt-aiNet schedule (Listing C.4, 6,389) and the
input_00_02 solution (Gambar III.8, 2,763). It also shows a reference point the dissertation did
not report: running FFT-8 serially on its fastest processor gives 5,164."""),
    ("code", "!python src/reproduce_dissertation.py"),

    ("md", """## 2. Run the algorithms on one instance
Every metaheuristic gets the same budget of schedule evaluations. `+H` means the run is seeded
with HEFT's mapping, so it can never return anything worse than HEFT. Change `inst` to try another
instance, e.g. `random_dag(50, 16, ccr=10.0, beta=1.0, seed=1)`."""),
    ("code", """import numpy as np, pandas as pd, time
import matplotlib.pyplot as plt
from instance import dissertation_fft8, random_dag, fft_dag, gauss_dag
from decode import evaluator, heft, cpop, peft
from meta import optainet, clonalg, ga, random_search, ode_dissertation
from plotting import gantt

inst = dissertation_fft8(16)          # or: random_dag(50, 16, ccr=10.0, beta=1.0, seed=1)
BUDGET, SEEDS = 20000, range(5)

f = evaluator(inst); n, m = inst.n, inst.m
hmap, hms, _ = heft(inst)
algs = {
    "RS":        lambda r: random_search(f, n, m, BUDGET, r),
    "GA":        lambda r: ga(f, n, m, BUDGET, r, pop=50, pc=0.6, pm=1 / n),
    "opt-aiNet": lambda r: optainet(f, n, m, BUDGET, r, N0=5, Nc=2, sigma=0.05),
    "ODE-fixed": lambda r: optainet(f, n, m, BUDGET, r, O=True, D=True, E=True, N0=5, Nc=2, sigma=0.05),
    "CLONALG":   lambda r: clonalg(f, n, m, BUDGET, r, N=10, Nc=20, rho=5.0),
    "GA+H":      lambda r: ga(f, n, m, BUDGET, r, pop=50, pc=0.6, pm=1 / n, seeds=[hmap]),
    "ODE+H":     lambda r: optainet(f, n, m, BUDGET, r, O=True, D=True, E=True, N0=5, Nc=2, sigma=0.05, seeds=[hmap]),
}
rows = [dict(method=k, mean=v, best=v, worst=v, seconds=0.0) for k, v in
        [("HEFT", hms), ("CPOP", cpop(inst)[1]), ("PEFT", peft(inst)[1]), ("Serial", inst.W.sum(0).min())]]
best_map = {}
for name, run in algs.items():
    t = time.perf_counter()
    res = [run(np.random.default_rng(s)) for s in SEEDS]
    c = [x["cost"] for x in res]
    best_map[name] = res[int(np.argmin(c))]["best"]
    rows.append(dict(method=name, mean=np.mean(c), best=min(c), worst=max(c),
                     seconds=(time.perf_counter() - t) / len(SEEDS)))
df = pd.DataFrame(rows).set_index("method")
df["mean / HEFT"] = df["mean"] / hms
print(f"{inst.name}: n = {n} tasks, m = {m} processors, CCR = {inst.ccr():.2f}")
df.round(3)"""),
    ("code", """fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
gantt(inst, hmap, ax=axes[0], title=f"HEFT — makespan {hms:,.0f}")
gantt(inst, best_map["ODE+H"], ax=axes[1], title=f"ODE+H, best of {len(SEEDS)} runs — makespan {f(best_map['ODE+H']):,.0f}")
plt.tight_layout(); plt.show()"""),

    ("md", """## 3. Bit-exact reproducibility check
Every run is seeded by (instance, algorithm, run), so re-running a subset must give exactly the
archived makespans. This re-runs 3 instances × 21 algorithms × 2 runs into a scratch folder and
compares each result with `results/main*.jsonl`."""),
    ("code", "!python src/verify_repro.py 0,25,48 2"),

    ("md", """## 4. Statistics of the paper from the archived raw data
`results/` holds every run: 13,260 runs in the main experiment, 5,100 for the published opt-aiNet
formulations and 3,060 for CLONALG (51 instances × 20 runs each). The scripts below recompute the
Friedman ranks, Holm-corrected Wilcoxon tests, the ablation and the anytime curves."""),
    ("code", "!python src/analyze3.py"),
    ("code", """import json
s = json.load(open("results/summary3.json"))
algs = [a for a in s if isinstance(s[a], dict) and "rank" in s[a]]
t = pd.DataFrame({a: s[a] for a in algs}).T.sort_values("rank")
t.columns = ["avg rank", "mean RPD %", "makespan / HEFT", "instances better than HEFT",
             "instances worse than HEFT", "seconds per run"]
t.round(3)"""),
    ("code", "!python src/analyze.py | sed -n '/Ablation/,/ODE-diss:/p'"),

    ("md", "## 5. Figures"),
    ("code", """from plotting import fig_ccr, fig_ablation, fig_anytime
for fn, size in [(fig_ccr, (8, 5)), (fig_ablation, (7, 4)), (fig_anytime, (8, 5))]:
    fig, ax = plt.subplots(figsize=size); fn(ax); plt.tight_layout(); plt.show()"""),

    ("md", """## 6. The two defects in the original implementation
`ode_dissertation` is a line-by-line port of Listing A.1 of the dissertation. Two defects are
switchable: **(1)** after cloning, every clone is overwritten with the cost of one stale schedule
(lines 511–515), and **(2)** mutation swaps a gene when a *normal* draw is ≤ p (≈ 0.01), which
happens for about half of all genes. The first defect makes the reported makespan differ from the
true makespan of the returned schedule."""),
    ("code", """inst6 = dissertation_fft8(16); f6 = evaluator(inst6)
rows = []
for label, kw in [("original code", {}), ("+ cost fix", dict(fix_cost=True)),
                  ("+ cost and mutation fix", dict(fix_cost=True, fix_mutation=True))]:
    for s in range(5):
        r = ode_dissertation(f6, inst6.n, inst6.m, 20000, np.random.default_rng(s), **kw)
        rows.append(dict(variant=label, run=s, reported=r["reported"], true=r["cost"]))
d6 = pd.DataFrame(rows)
d6["share misreported"] = (d6.reported - d6.true).abs() > 1e-9
d6.groupby("variant", sort=False)[["reported", "true", "share misreported"]].mean().round(1)"""),

    ("md", """## 7. Full reproduction (optional)
Re-runs tuning, all experiments and the analysis into `results_repro/`. Expect about 4 hours on
2 cores (Colab) and less on Kaggle's 4 cores. The archived `results/` folder is not touched.
Set `RUN_FULL = True` to start."""),
    ("code", """RUN_FULL = False
if RUN_FULL:
    os.environ["RESULTS_DIR"] = "results_repro"
    for script in ["tune.py", "tune2.py", "tune3.py", "tune4.py", "tune5.py", "tune6.py",
                   "run_exp.py", "run_exp2.py", "run_exp3.py", "exp_bug.py", "exp_bug_opt.py",
                   "analyze.py", "analyze2.py", "analyze3.py"]:
        print("==>", script, flush=True)
        !python src/{script} > results_repro_{script}.log 2>&1
    print(open("results_repro_analyze3.py.log").read())
else:
    print("Skipped. Set RUN_FULL = True to re-run everything.")"""),
]


def build(platform, path):
    nb = nbf.v4.new_notebook()
    cells = [("md", INTRO.format(platform=platform))] + SETUP[platform] + BODY
    nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
                for kind, src in cells]
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"}}
    if platform == "Colab":
        nb.metadata["colab"] = {"provenance": [], "name": "colab_quickstart.ipynb"}
    else:
        nb.metadata["kaggle"] = {"accelerator": "none", "isInternetEnabled": True, "language": "python"}
    nbf.write(nb, path)
    print("wrote", path)


if __name__ == "__main__":
    build("Colab", "notebooks/colab_quickstart.ipynb")
    build("Kaggle", "notebooks/kaggle_quickstart.ipynb")
    print("Colab link: ", COLAB)
    print("Kaggle link:", KAGGLE)
