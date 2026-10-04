"""Schedule decoder and list-scheduling baselines (numba-accelerated).

A metaheuristic chromosome is a task->processor mapping. Tasks are placed in a fixed
priority order (HEFT rank_u), each one at the earliest feasible start time on its assigned
processor. `insertion=True` uses the insertion policy of HEFT (fills idle gaps);
`insertion=False` appends after the last task on the processor (the policy implemented in
the dissertation code, Lampiran A).
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _ready_time(t, q, mapping, aft, pptr, pidx, pdat, B):
    r = 0.0
    for k in range(pptr[t], pptr[t + 1]):
        u = pidx[k]
        pu = mapping[u]
        c = 0.0 if pu == q else pdat[k] / B[pu, q]
        v = aft[u] + c
        if v > r:
            r = v
    return r


@njit(cache=True)
def _slot(ready, w, q, starts, ends, cnt, insertion):
    """Return (start, insert_position) on processor q."""
    c = cnt[q]
    if c == 0:
        return ready, 0
    if not insertion:
        s = ends[q, c - 1]
        return (ready if ready > s else s), c
    # gap before the first task
    if ready + w <= starts[q, 0]:
        return ready, 0
    for k in range(1, c):
        gs = ends[q, k - 1]
        st = ready if ready > gs else gs
        if st + w <= starts[q, k]:
            return st, k
    s = ends[q, c - 1]
    return (ready if ready > s else s), c


@njit(cache=True)
def _insert(q, pos, st, en, starts, ends, cnt):
    c = cnt[q]
    for k in range(c, pos, -1):
        starts[q, k] = starts[q, k - 1]
        ends[q, k] = ends[q, k - 1]
    starts[q, pos] = st
    ends[q, pos] = en
    cnt[q] = c + 1


@njit(cache=True)
def makespan(order, mapping, W, pptr, pidx, pdat, B, insertion):
    n, m = W.shape
    starts = np.empty((m, n))
    ends = np.empty((m, n))
    cnt = np.zeros(m, dtype=np.int64)
    aft = np.zeros(n)
    ms = 0.0
    for t in order:
        q = mapping[t]
        r = _ready_time(t, q, mapping, aft, pptr, pidx, pdat, B)
        w = W[t, q]
        st, pos = _slot(r, w, q, starts, ends, cnt, insertion)
        _insert(q, pos, st, st + w, starts, ends, cnt)
        aft[t] = st + w
        if aft[t] > ms:
            ms = aft[t]
    return ms


@njit(cache=True)
def schedule_detail(order, mapping, W, pptr, pidx, pdat, B, insertion):
    n, m = W.shape
    starts = np.empty((m, n))
    ends = np.empty((m, n))
    cnt = np.zeros(m, dtype=np.int64)
    ast = np.zeros(n)
    aft = np.zeros(n)
    for t in order:
        q = mapping[t]
        r = _ready_time(t, q, mapping, aft, pptr, pidx, pdat, B)
        w = W[t, q]
        st, pos = _slot(r, w, q, starts, ends, cnt, insertion)
        _insert(q, pos, st, st + w, starts, ends, cnt)
        ast[t] = st
        aft[t] = st + w
    return ast, aft


@njit(cache=True)
def heft_core(order, W, pptr, pidx, pdat, B, insertion, oct_tab, use_oct):
    """Greedy EFT processor selection along a given order. With use_oct it implements the
    PEFT selection rule (min EFT + OCT)."""
    n, m = W.shape
    starts = np.empty((m, n))
    ends = np.empty((m, n))
    cnt = np.zeros(m, dtype=np.int64)
    aft = np.zeros(n)
    mapping = np.full(n, -1, dtype=np.int64)
    for t in order:
        best_q, best_key, best_st, best_pos = -1, np.inf, 0.0, 0
        for q in range(m):
            r = _ready_time(t, q, mapping, aft, pptr, pidx, pdat, B)
            w = W[t, q]
            st, pos = _slot(r, w, q, starts, ends, cnt, insertion)
            key = st + w + (oct_tab[t, q] if use_oct else 0.0)
            if key < best_key:
                best_key, best_q, best_st, best_pos = key, q, st, pos
        mapping[t] = best_q
        _insert(best_q, best_pos, best_st, best_st + W[t, best_q], starts, ends, cnt)
        aft[t] = best_st + W[t, best_q]
    return mapping, aft.max()


def evaluator(inst, insertion=True, order=None):
    """Return a closure f(mapping)->makespan for the instance."""
    order = inst.priority_order() if order is None else order
    W, pptr, pidx, pdat, B = inst.W, inst.pptr, inst.pidx, inst.pdat, inst.B

    def f(mapping):
        return makespan(order, mapping, W, pptr, pidx, pdat, B, insertion)
    f.order = order
    return f


# ---------------------------------------------------------------------------
# List-scheduling baselines
# ---------------------------------------------------------------------------
_EMPTY = np.zeros((1, 1))


def heft(inst, insertion=True, saga_mode=False):
    order = inst.priority_order(saga_mode)
    mp, ms = heft_core(order, inst.W, inst.pptr, inst.pidx, inst.pdat, inst.B, insertion,
                       _EMPTY, False)
    return mp, ms, order


def _ready_list_schedule(inst, prio, select, insertion=True):
    """Generic ready-list scheduler: repeatedly take the ready task with highest priority;
    `select(t, eft_vector)` returns the processor."""
    n, m = inst.n, inst.m
    W, B = inst.W, inst.B
    starts = np.empty((m, n)); ends = np.empty((m, n)); cnt = np.zeros(m, dtype=np.int64)
    aft = np.zeros(n); mapping = np.full(n, -1, dtype=np.int64)
    indeg = np.array([len(p) for p in inst.preds])
    ready = [t for t in range(n) if indeg[t] == 0]
    order = []
    while ready:
        t = max(ready, key=lambda x: (prio[x], -x))
        ready.remove(t)
        cand = []
        for q in range(m):
            r = _ready_time(t, q, mapping, aft, inst.pptr, inst.pidx, inst.pdat, B)
            st, pos = _slot(r, W[t, q], q, starts, ends, cnt, insertion)
            cand.append((st, pos))
        eft = np.array([cand[q][0] + W[t, q] for q in range(m)])
        q = select(t, eft)
        st, pos = cand[q]
        _insert(q, pos, st, st + W[t, q], starts, ends, cnt)
        aft[t] = st + W[t, q]; mapping[t] = q; order.append(t)
        for v, _ in inst.succs[t]:
            indeg[v] -= 1
            if indeg[v] == 0:
                ready.append(v)
    return mapping, aft.max(), np.array(order)


def cpop(inst, insertion=True):
    """CPOP (Topcuoglu et al., 2002)."""
    ru, rd = inst.upward_rank(), inst.downward_rank()
    prio = ru + rd
    cp_len = prio.max()
    entry = [t for t in range(inst.n) if not inst.preds[t]]
    cp, t = set(), max(entry, key=lambda x: prio[x])
    cp.add(t)
    while inst.succs[t]:
        nxt = [v for v, _ in inst.succs[t] if abs(prio[v] - cp_len) < 1e-6 * max(1, cp_len)]
        if not nxt:
            break
        t = nxt[0]; cp.add(t)
    cp_list = sorted(cp)
    p_cp = int(np.argmin(inst.W[cp_list].sum(axis=0)))

    def select(t, eft):
        return p_cp if t in cp else int(np.argmin(eft))
    return _ready_list_schedule(inst, prio, select, insertion)[:2]


def peft(inst, insertion=True):
    """PEFT (Arabnejad & Barbosa, 2014): optimistic cost table lookahead."""
    n, m = inst.n, inst.m
    W = inst.W
    oct_tab = np.zeros((n, m))
    cbar = inst.rank_cfac
    for u in reversed(inst.topo_order()):
        for p in range(m):
            best = 0.0
            for v, d in inst.succs[u]:
                vals = oct_tab[v] + W[v] + np.where(np.arange(m) == p, 0.0, d * cbar)
                best = max(best, vals.min())
            oct_tab[u, p] = best
    rank_oct = oct_tab.mean(axis=1)

    def select(t, eft):
        return int(np.argmin(eft + oct_tab[t]))
    return _ready_list_schedule(inst, rank_oct, select, insertion)[:2]
