"""Figures of the paper (from the archived summaries) and a Gantt chart helper.

    python src/make_figures.py      # writes figures/*.png
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from paths import rpath

ACCENT, GREY, DARK = "#2a6fdb", "#b8b8b0", "#555555"


def _load(name):
    return json.load(open(rpath(name)))


def fig_ccr(ax=None):
    """Figure 1: geometric-mean makespan / HEFT by CCR level."""
    s1, s3 = _load("summary.json"), _load("summary3.json")
    algs = ["GA+H", "ODE+H", "CLONALG+H", "Serial", "PEFT", "GA", "ODE-fixed", "CLONALG",
            "dCT-tuned", "opt-aiNet", "ODE-diss"]
    levels = [("low", "Low CCR (0.015-0.1)", 0.25), ("mid", "Mid CCR (1)", 0.55), ("high", "High CCR (6.3-10)", 1.0)]

    def val(a, g):
        return s1["groups"][f"ccr={g}"]["Serial"] if a == "Serial" else s3["ccr"][g][a]
    ax = ax or plt.subplots(figsize=(8, 5))[1]
    for i, a in enumerate(algs):
        v = [val(a, g) for g, _, _ in levels]
        ax.plot([min(v), max(v)], [i, i], color=GREY, lw=2, zorder=1)
        for (g, lab, al), x in zip(levels, v):
            ax.scatter(x, i, s=60, color=ACCENT, alpha=al, edgecolor=ACCENT, zorder=2,
                       label=lab if i == 0 else None)
        ax.annotate(f"{v[2]:.2f}", (v[2], i), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=8)
    ax.axvline(1.0, color=DARK, ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xticks([0.8, 1, 1.5, 2, 3, 4, 5]); ax.set_xticklabels(["0.8", "1", "1.5", "2", "3", "4", "5"])
    ax.set_yticks(range(len(algs))); ax.set_yticklabels(algs); ax.invert_yaxis()
    ax.set_xlabel("Makespan / HEFT (geometric mean, log scale; lower is better)")
    ax.set_title("At high CCR only HEFT-seeded search and the serial schedule clearly beat HEFT", fontsize=10)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    return ax


def fig_ablation(ax=None):
    """Figure 2: ablation of the O, D and E switches (mean RPD)."""
    ab = _load("summary.json")["ablation"]
    order = ["opt-aiNet", "+O", "+D", "+OD", "+E", "+OE", "+DE", "ODE-fixed"]
    v = [ab[a]["rpd"] for a in order]
    ax = ax or plt.subplots(figsize=(7, 4))[1]
    ax.barh(order, v, color=[ACCENT if a == "ODE-fixed" else GREY for a in order])
    for i, x in enumerate(v):
        ax.text(x + 0.5, i, f"{x:.1f} %", va="center", fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel("Mean RPD to best known makespan (%)")
    ax.set_title("Each switch helps; D and E together do most of the work", fontsize=10)
    return ax


def fig_anytime(ax=None):
    """Figure 3: best makespan / HEFT against evaluations."""
    s1, s3 = _load("summary.json"), _load("summary3.json")
    x = [500, 1000, 2000, 5000, 10000, 20000]
    curves = {a: s1["anytime"][a] for a in ["RS", "opt-aiNet", "ODE-diss"]}
    curves.update({a: s3["anytime"][a] for a in ["GA", "ODE-fixed", "CLONALG", "GA+H", "ODE+H"]})
    style = {"ODE-fixed": dict(color=ACCENT, lw=3), "GA": dict(color=DARK, lw=2),
             "CLONALG": dict(color=DARK, lw=2, ls=":"), "GA+H": dict(color=GREY, lw=2, ls="--"),
             "ODE+H": dict(color=ACCENT, lw=1.5, ls="--")}
    ax = ax or plt.subplots(figsize=(8, 5))[1]
    for a, y in curves.items():
        ax.plot(x, y, marker="o", ms=3, label=f"{a} ({y[-1]:.2f})", **style.get(a, dict(color=GREY, lw=1.5)))
    ax.axhline(1.0, color=DARK, ls="--", lw=1)
    ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels(["500", "1k", "2k", "5k", "10k", "20k"])
    ax.set_xlabel("Schedule evaluations (log scale)"); ax.set_ylabel("Best makespan / HEFT")
    ax.set_title("ODE-fixed leads GA early, ties at 2,000 evaluations, then falls behind", fontsize=10)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    return ax


def gantt(inst, mapping, insertion=True, title=None, ax=None):
    """Gantt chart of the schedule decoded from a task->processor mapping."""
    from decode import schedule_detail
    mapping = np.asarray(mapping, dtype=np.int64)
    ast, aft = schedule_detail(inst.priority_order(), mapping, inst.W, inst.pptr, inst.pidx,
                               inst.pdat, inst.B, insertion)
    used = sorted(set(mapping.tolist()))
    row = {p: k for k, p in enumerate(used)}
    ax = ax or plt.subplots(figsize=(8, 0.45 * len(used) + 1.2))[1]
    for t in range(inst.n):
        y = row[int(mapping[t])]
        ax.barh(y, aft[t] - ast[t], left=ast[t], color=ACCENT, alpha=0.75, edgecolor="white")
        if aft[t] - ast[t] > 0.04 * aft.max():
            ax.text((ast[t] + aft[t]) / 2, y, f"t{t}", ha="center", va="center", fontsize=7, color="white")
    ax.set_yticks(range(len(used))); ax.set_yticklabels([f"p{p}" for p in used]); ax.invert_yaxis()
    ax.set_xlabel("time"); ax.set_title(title or f"makespan {aft.max():,.0f}", fontsize=10)
    return ax
