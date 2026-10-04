"""Metaheuristics for the task->processor mapping problem, all driven by a common
evaluation budget (number of schedule evaluations).

Encoding (as in the dissertation): a cell/chromosome is an integer vector x in {0..m-1}^n,
x[t] = processor of task t. Tasks are decoded in the fixed rank_u priority order.

Algorithms
- optainet(..., O, D, E): opt-aiNet in the form implemented in the dissertation lineage
  (Brownlee 2012, Listing III.1): every generation (re)evaluates the population, each cell
  produces clones, the best clone replaces its parent, the clonal step repeats until the
  progeny's average cost is better than the population's, cells are then suppressed by
  Hamming affinity (population sorted first, Timmis & Edmonds 2004) and d% random cells
  are added. Flags switch on the three ODE modifications:
    O  one-time evaluation  (cached fitness, parents never re-evaluated)
    D  dynamic number of clones, N_c,i = floor(omega * f*_i)  (f* = 1 for the best cell)
    E  elitism  (best-so-far cell re-inserted after suppression)
  optainet(O=D=E=True) is the corrected ODE opt-aiNet ("ODE-fixed").
- ode_dissertation(...): line-by-line port of Lampiran A (Listing A.1), including
  (a) the stale-cost assignment at lines 511-515 and (b) the swap mutation triggered by
  N(0,1) <= p. Flags allow repairing each defect separately.
- ga(...): generational GA, binary tournament, uniform crossover, per-gene reassignment.
- random_search(...).
All return dict(best=mapping, cost=true makespan, reported=cost reported by the algorithm,
evals=used evaluations, trace=[(evals, best_true_cost), ...]).
"""
import numpy as np


class Budget(Exception):
    pass


class Counter:
    """Wraps the evaluator, counts evaluations, records the anytime trace of the best
    *true* makespan seen, and raises Budget when exhausted."""

    def __init__(self, f, budget):
        self.f, self.budget, self.n = f, budget, 0
        self.best, self.best_x = np.inf, None
        self.trace = []

    def __call__(self, x):
        if self.n >= self.budget:
            raise Budget()
        self.n += 1
        c = self.f(x)
        if c < self.best:
            self.best, self.best_x = c, x.copy()
            self.trace.append((self.n, c))
        return c


def _mutate_reassign(x, alpha, m, rng):
    """Reassign each gene with probability alpha (at least one gene) to a different
    processor."""
    n = x.size
    y = x.copy()
    mask = rng.random(n) < alpha
    if not mask.any():
        mask[rng.integers(n)] = True
    k = int(mask.sum())
    y[mask] = (y[mask] + rng.integers(1, m, size=k)) % m if m > 1 else y[mask]
    return y


def _fstar(costs):
    """Normalised fitness in [0,1], 1 = best (lowest makespan)."""
    lo, hi = costs.min(), costs.max()
    if hi - lo <= 0:
        return np.ones_like(costs)
    return (hi - costs) / (hi - lo)


def _hamming_suppress(X, costs, thr):
    """Greedy suppression on the population sorted by cost: a cell survives if its Hamming
    distance to every already-kept (better) cell is >= thr."""
    idx = np.argsort(costs, kind="stable")
    kept = []
    for i in idx:
        if all(np.count_nonzero(X[i] != X[j]) >= thr for j in kept):
            kept.append(i)
    return kept


def _result(cnt, f, reported=None, x=None):
    x = cnt.best_x if x is None else x
    return dict(best=x, cost=float(f(x)), reported=float(cnt.best if reported is None else reported),
                evals=cnt.n, trace=cnt.trace)


# ---------------------------------------------------------------------------
def optainet(f, n, m, budget, rng, O=False, D=False, E=False, N0=20, Nc=10, d=0.4,
             sigma=0.1, inner_max=10, seeds=(), max_gen=10**9):
    cnt = Counter(f, budget)
    gen = 0
    thr = max(1, int(np.ceil(sigma * n)))
    X = rng.integers(m, size=(N0, n))
    for k, s in enumerate(seeds):
        X[k] = s
    C = np.full(N0, np.nan)
    elite_x, elite_c = None, np.inf
    try:
        while gen < max_gen:
            gen += 1
            # (1) fitness evaluation
            for i in range(len(X)):
                if (not O) or np.isnan(C[i]):
                    C[i] = cnt(X[i])
            b = int(np.argmin(C))
            if C[b] < elite_c:
                elite_c, elite_x = C[b], X[b].copy()
            avg = C.mean()
            # (2) clonal selection, repeated until progeny average < population average
            for _ in range(inner_max):
                fs = _fstar(C)
                PX, PC = [], []
                for i in range(len(X)):
                    nci = int(np.floor(Nc * fs[i])) if D else Nc
                    if nci <= 0:
                        continue  # cell produces no progeny (D): removed, as in Lampiran A
                    alpha = np.exp(1.0 - fs[i]) / n
                    best_c, best_x = np.inf, None
                    for _c in range(nci):
                        y = _mutate_reassign(X[i], alpha, m, rng)
                        cy = cnt(y)
                        if cy < best_c:
                            best_c, best_x = cy, y
                    PX.append(best_x); PC.append(best_c)
                PX, PC = np.array(PX), np.array(PC)
                if PC.mean() < avg:
                    break
            # (3) network interaction: suppression
            kept = _hamming_suppress(PX, PC, thr)
            X, C = PX[kept], PC[kept]
            b = int(np.argmin(C))
            if C[b] < elite_c:
                elite_c, elite_x = C[b], X[b].copy()
            # (E) elitism: re-insert best-so-far
            if E and not any((x == elite_x).all() for x in X):
                X = np.vstack([X, elite_x]); C = np.append(C, elite_c)
            # (4) add d% random cells
            R = rng.integers(m, size=(int(round(d * N0)), n))
            X = np.vstack([X, R]); C = np.append(C, np.full(len(R), np.nan))
    except Budget:
        pass
    res = _result(cnt, f)
    res["gens"] = gen
    return res


# ---------------------------------------------------------------------------
def ode_dissertation(f, n, m, budget, rng, aff_threshold=None, fix_cost=False,
                     fix_mutation=False, popl_size=20, max_gen=10**9):
    """Port of Listing A.1 (Lampiran A). aff_threshold is an absolute Hamming distance.
    fix_cost:      clones keep the cost computed in clone() (removes lines 511-515 bug).
    fix_mutation:  per-gene reassignment instead of the N(0,1)<=p swap."""
    cnt = Counter(f, budget)
    beta, clone_factor = 75.0, 0.095
    num_rand = int(0.4 * popl_size)
    if aff_threshold is None:
        aff_threshold = max(1, int(np.ceil(0.1 * n)))

    def cell(sched, cost, is_eval):
        return {"schedule": np.asarray(sched).copy(), "cost": cost, "norm": 0.0, "is_eval": is_eval}

    def calc_norm_cost(pop):
        pop.sort(key=lambda c: c["cost"])
        rng_ = pop[-1]["cost"] - pop[0]["cost"]
        for c in pop:
            c["norm"] = 1.0 if rng_ == 0 else 1.0 - ((pop[-1]["cost"] - c["cost"]) / rng_)

    def clone(pop):
        calc_norm_cost(pop)
        nc = int(round(clone_factor * len(pop)))
        out = []
        for p in pop:
            prob = (1.0 / beta) * np.exp(-p["norm"])
            dyn_nc = int(nc * (1 - p["norm"]))
            temp = []
            for _ in range(dyn_nc):
                c = cell(p["schedule"], p["cost"], p["is_eval"])
                if fix_mutation:
                    c["schedule"] = _mutate_reassign(c["schedule"], np.exp(-p["norm"]) * np.e / n, m, rng)
                else:
                    s = c["schedule"]
                    for j in range(n):
                        if rng.normal(0.0, 1.0) <= prob:
                            idx = rng.integers(n)
                            s[j], s[idx] = s[idx], s[j]
                temp.append(c)
            for c in temp:
                c["cost"] = cnt(c["schedule"])
                c["is_eval"] = False
            temp.sort(key=lambda c: c["cost"])
            if temp:
                out.append(temp[0])
        return out

    def affinity_supress(pop):
        res = []
        for c in pop:
            nb = [p for p in pop if np.count_nonzero(p["schedule"] != c["schedule"]) < aff_threshold]
            nb.sort(key=lambda p: p["cost"])
            if (not nb) or (c is nb[0]):
                res.append(c)
        return res

    best = {"cost": 1e12, "schedule": None}
    population = []
    try:
        for _ in range(popl_size):
            s = rng.integers(m, size=n)
            population.append(cell(s, cnt(s), True))
        p_ = population[-1]
        for gen in range(max_gen):
            for p in population:
                p_ = p
                if p["is_eval"] is False:
                    p["cost"] = cnt(p["schedule"])
                    p["is_eval"] = True
            calc_norm_cost(population)
            population.sort(key=lambda c: c["cost"])
            if population[0]["cost"] < best["cost"]:
                best = {"cost": population[0]["cost"], "schedule": population[0]["schedule"].copy()}
            average_cost_popl = np.mean([c["cost"] for c in population])
            done, i__ = False, 0
            while not done:
                i__ += 1
                cloned = clone(population)
                if not fix_cost:
                    z_cost = cnt(p_["schedule"])           # line 511: stale p_
                    for p_ in cloned:                      # lines 512-515
                        p_["cost"] = z_cost
                        p_["is_eval"] = True
                else:
                    for c in cloned:
                        c["is_eval"] = True
                average_cloned = np.mean([c["cost"] for c in cloned]) if cloned else np.inf
                if i__ > 50 or average_cloned < average_cost_popl:
                    done = True
            cloned.sort(key=lambda c: c["cost"])
            population = affinity_supress(cloned)
            population.append(cell(best["schedule"], best["cost"], False))
            for _ in range(num_rand - 1):
                population.append(cell(rng.integers(m, size=n), 1e12, False))
    except Budget:
        pass
    x = best["schedule"] if best["schedule"] is not None else cnt.best_x
    return dict(best=x, cost=float(f(x)), reported=float(best["cost"]), evals=cnt.n,
                trace=cnt.trace, true_best_seen=float(cnt.best))


# ---------------------------------------------------------------------------
def ga(f, n, m, budget, rng, pop=50, pc=0.9, pm=None, elite=2, seeds=()):
    cnt = Counter(f, budget)
    pm = 1.0 / n if pm is None else pm
    X = rng.integers(m, size=(pop, n))
    for k, s in enumerate(seeds):
        X[k] = s
    try:
        C = np.array([cnt(x) for x in X])
        while True:
            order = np.argsort(C)
            NX = [X[i].copy() for i in order[:elite]]
            NC = [C[i] for i in order[:elite]]
            while len(NX) < pop:
                a, b = rng.integers(pop, size=2); p1 = X[a] if C[a] < C[b] else X[b]
                a, b = rng.integers(pop, size=2); p2 = X[a] if C[a] < C[b] else X[b]
                child = np.where(rng.random(n) < 0.5, p1, p2) if rng.random() < pc else p1.copy()
                mask = rng.random(n) < pm
                if mask.any() and m > 1:
                    child[mask] = (child[mask] + rng.integers(1, m, size=int(mask.sum()))) % m
                NX.append(child); NC.append(cnt(child))
            X, C = np.array(NX), np.array(NC)
    except Budget:
        pass
    return _result(cnt, f)


def random_search(f, n, m, budget, rng):
    cnt = Counter(f, budget)
    try:
        while True:
            cnt(rng.integers(m, size=n))
    except Budget:
        pass
    return _result(cnt, f)


# ---------------------------------------------------------------------------
# Reference formulations of opt-aiNet (added for the comparison of variants)
# ---------------------------------------------------------------------------
def optainet_decastro(f, n, m, budget, rng, N0=20, Nc=10, d=0.4, sigma=0.1, eps=1e-3,
                      inner_max=1000, seeds=()):
    """opt-aiNet as specified by de Castro & Timmis (CEC 2002), discrete encoding:
    2.1 evaluate cells; 2.2-2.3 Nc clones each, mutated relative to the parent's normalised
    fitness, *parent kept*; 2.5 each cell is replaced by the best of {parent, clones};
    2.6 repeat until the population's average cost no longer changes (relative change < eps);
    3 suppression: among cells closer than ceil(sigma*n), keep only the best;
    4 add d% (of the current network size) random cells.
    Parents are (re-)evaluated once per clonal iteration (step 2.1)."""
    cnt = Counter(f, budget)
    thr = max(1, int(np.ceil(sigma * n)))
    X = rng.integers(m, size=(N0, n))
    for k, s in enumerate(seeds):
        X[k] = s
    try:
        while True:
            prev = None
            for _ in range(inner_max):
                C = np.array([cnt(x) for x in X])                  # step 2.1
                fs = _fstar(C)
                NX, NC = [], []
                for i in range(len(X)):
                    alpha = np.exp(1.0 - fs[i]) / n
                    bx, bc = X[i], C[i]                            # parent kept
                    for _c in range(Nc):
                        y = _mutate_reassign(X[i], alpha, m, rng)
                        cy = cnt(y)
                        if cy < bc:
                            bx, bc = y, cy
                    NX.append(bx); NC.append(bc)
                X, C = np.array(NX), np.array(NC)
                avg = C.mean()
                if prev is not None and abs(prev - avg) <= eps * prev:
                    break
                prev = avg
            kept = _hamming_suppress(X, C, thr)
            X = X[kept]
            R = rng.integers(m, size=(max(1, int(round(d * len(X)))), n))
            X = np.vstack([X, R])
    except Budget:
        pass
    return _result(cnt, f)


def optainet_brownlee(f, n, m, budget, rng, pop_size=20, Nc=10, beta=100.0, num_rand=None,
                      aff=0.05, seeds=()):
    """Port of Brownlee (2011), Listing 7.4, discrete encoding. Kept as published:
    normalised cost 1 - cost/range (not min-max), mutation rate (1/beta)*exp(-norm) used as a
    per-gene reassignment probability (capped at 1, at least one gene), only the best clone
    survives (parent discarded), the clonal step repeats until the progeny's mean cost is below
    the population's, neighbourhood suppression (a cell survives if it is the best within its
    own Hamming neighbourhood < ceil(aff*n)), num_rand random cells per generation, and the
    whole population is re-evaluated every generation (lines 87 and 90)."""
    cnt = Counter(f, budget)
    num_rand = int(round(0.4 * pop_size)) if num_rand is None else num_rand
    thr = max(1, int(np.ceil(aff * n)))
    X = rng.integers(m, size=(pop_size, n))
    for k, s in enumerate(seeds):
        X[k] = s
    try:
        for x in X:
            cnt(x)                                                  # line 87
        while True:
            C = np.array([cnt(x) for x in X])                       # line 90
            rg = C.max() - C.min()
            norm = np.ones_like(C) if rg == 0 else 1.0 - C / rg      # line 52
            avg = C.mean()
            while True:
                PX, PC = [], []
                for i in range(len(X)):
                    alpha = min(1.0, (1.0 / beta) * np.exp(-norm[i]))
                    bc, bx = np.inf, None
                    for _c in range(Nc):
                        y = _mutate_reassign(X[i], alpha, m, rng)
                        cy = cnt(y)
                        if cy < bc:
                            bx, bc = y, cy
                    PX.append(bx); PC.append(bc)
                PX, PC = np.array(PX), np.array(PC)
                if PC.mean() < avg:
                    break
            keep = []
            for i in range(len(PX)):
                nb = [j for j in range(len(PX)) if np.count_nonzero(PX[i] != PX[j]) < thr]
                nb.sort(key=lambda j: (PC[j], j))
                if nb[0] == i:
                    keep.append(i)
            X = np.vstack([PX[keep], rng.integers(m, size=(num_rand, n))])
    except Budget:
        pass
    return _result(cnt, f)


def clonalg(f, n, m, budget, rng, N=10, Nc=5, rho=2.5, d=None, seeds=()):
    """CLONALG, optimisation version (de Castro & Von Zuben, IEEE TEC 2002), discrete encoding.
    Every antibody is selected and produces Nc clones; each clone reassigns every gene with
    probability exp(-rho * f*) (f* = min-max normalised fitness, best = 1; at least one gene);
    the best member of each clone set (parent included) enters the next population; the d
    worst antibodies are replaced by random ones. Antibodies are evaluated once (cached)."""
    cnt = Counter(f, budget)
    d = max(1, int(round(0.1 * N))) if d is None else d
    X = rng.integers(m, size=(N, n))
    for k, s in enumerate(seeds):
        X[k] = s
    try:
        C = np.array([cnt(x) for x in X])
        while True:
            fs = _fstar(C)
            for i in range(N):
                alpha = float(np.exp(-rho * fs[i]))
                for _c in range(Nc):
                    y = _mutate_reassign(X[i], alpha, m, rng)
                    cy = cnt(y)
                    if cy < C[i]:
                        X[i], C[i] = y, cy
            worst = np.argsort(C)[N - d:]
            for i in worst:
                X[i] = rng.integers(m, size=n); C[i] = cnt(X[i])
    except Budget:
        pass
    return _result(cnt, f)
