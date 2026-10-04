"""Problem instances for static DAG scheduling on heterogeneous (unrelated) processors.

Model (Topcuoglu et al., 2002): W[i, p] = computation cost of task i on processor p;
edge (i, j) carries data volume D_ij; communication time = D_ij / B[p, q] when p != q,
and 0 when both tasks run on the same processor.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Instance:
    name: str
    W: np.ndarray                 # (n, m) computation costs
    edges: list                   # list of (i, j, data)
    B: np.ndarray | None = None   # (m, m) bandwidth; default 1 off-diagonal
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.W = np.asarray(self.W, dtype=np.float64)
        self.n, self.m = self.W.shape
        if self.B is None:
            self.B = np.ones((self.m, self.m))
        self.B = np.asarray(self.B, dtype=np.float64).copy()
        np.fill_diagonal(self.B, np.inf)
        # CSR predecessor / successor structure
        preds = [[] for _ in range(self.n)]
        succs = [[] for _ in range(self.n)]
        for i, j, d in self.edges:
            preds[j].append((i, d))
            succs[i].append((j, d))
        self.preds, self.succs = preds, succs
        self.pptr = np.zeros(self.n + 1, dtype=np.int64)
        for j in range(self.n):
            self.pptr[j + 1] = self.pptr[j] + len(preds[j])
        self.pidx = np.array([i for j in range(self.n) for i, _ in preds[j]], dtype=np.int64)
        self.pdat = np.array([d for j in range(self.n) for _, d in preds[j]], dtype=np.float64)
        if self.pidx.size == 0:
            self.pidx = np.zeros(0, dtype=np.int64)
            self.pdat = np.zeros(0, dtype=np.float64)
        off = ~np.eye(self.m, dtype=bool)
        self.mean_inv_bw = float(np.mean(1.0 / self.B[off])) if self.m > 1 else 0.0
        self.rank_cfac = self.mean_inv_bw   # factor used in rank/OCT estimates
        self._check_acyclic()

    def _check_acyclic(self):
        indeg = np.array([len(p) for p in self.preds])
        stack = [i for i in range(self.n) if indeg[i] == 0]
        seen = 0
        while stack:
            u = stack.pop(); seen += 1
            for v, _ in self.succs[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    stack.append(v)
        if seen != self.n:
            raise ValueError(f"{self.name}: task graph has a cycle")

    def topo_order(self):
        indeg = np.array([len(p) for p in self.preds])
        q = [i for i in range(self.n) if indeg[i] == 0]
        out = []
        while q:
            u = q.pop(0); out.append(u)
            for v, _ in self.succs[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        return out

    def ccr(self):
        """Mean communication time / mean computation time (Topcuoglu definition)."""
        if not self.edges:
            return 0.0
        mc = np.mean([d for _, _, d in self.edges]) * self.mean_inv_bw
        return mc / self.W.mean()

    def upward_rank(self, saga_mode=False):
        """HEFT rank_u with mean computation and mean communication costs.
        saga_mode reproduces SAGA's averaging over all links incl. self-links (used only to
        validate our HEFT against SAGA)."""
        wbar = self.W.mean(axis=1)
        if saga_mode:
            iu = np.triu_indices(self.m)
            inv = np.where(np.isinf(self.B[iu]), 0.0, 1.0 / self.B[iu])
            cfac = inv.mean()
        else:
            cfac = self.rank_cfac
        r = np.zeros(self.n)
        for u in reversed(self.topo_order()):
            best = 0.0
            for v, d in self.succs[u]:
                best = max(best, d * cfac + r[v])
            r[u] = wbar[u] + best
        return r

    def downward_rank(self):
        wbar = self.W.mean(axis=1)
        r = np.zeros(self.n)
        for v in self.topo_order():
            best = 0.0
            for u, d in self.preds[v]:
                best = max(best, r[u] + wbar[u] + d * self.rank_cfac)
            r[v] = best
        return r

    def priority_order(self, saga_mode=False):
        """Tasks sorted by non-increasing rank_u; ties broken by topological position so the
        order is always a valid topological order."""
        r = self.upward_rank(saga_mode)
        pos = {t: k for k, t in enumerate(self.topo_order())}
        return np.array(sorted(range(self.n), key=lambda t: (-r[t], pos[t])), dtype=np.int64)


# ---------------------------------------------------------------------------
# Instances transcribed from the dissertation (Lampiran B, Tabel III.2)
# ---------------------------------------------------------------------------
_FFT8_W16 = [
    [1] * 16,
    [691, 571, 871, 921, 641, 671, 201, 521, 361, 541, 371, 611, 311, 461, 451, 731],
    [281, 661, 461, 691, 801, 501, 611, 771, 481, 331, 281, 261, 301, 451, 381, 701],
    [791, 321, 581, 571, 451, 921, 561, 211, 251, 551, 691, 651, 311, 231, 461, 541],
    [411, 791, 511, 541, 911, 371, 761, 221, 291, 181, 351, 641, 211, 391, 341, 631],
    [401, 651, 981, 811, 851, 591, 311, 531, 421, 241, 571, 551, 921, 441, 461, 331],
    [991, 451, 691, 881, 651, 561, 341, 581, 501, 861, 631, 791, 391, 701, 671, 381],
    [481, 751, 461, 651, 721, 731, 711, 841, 621, 521, 461, 681, 411, 871, 791, 541],
    [841, 821, 251, 651, 801, 261, 371, 631, 441, 751, 981, 891, 511, 841, 491, 421],
    [591, 221, 831, 841, 921, 341, 141, 721, 221, 301, 571, 271, 591, 771, 271, 631],
    [221, 561, 451, 651, 581, 401, 601, 521, 251, 251, 621, 581, 511, 381, 701, 721],
    [671, 261, 651, 521, 531, 491, 701, 761, 851, 561, 741, 821, 641, 401, 571, 371],
    [931, 761, 481, 701, 681, 611, 221, 821, 471, 481, 751, 371, 881, 751, 601, 401],
    [1] * 16,
]
_FFT8_E = [(0, 1, 1), (0, 2, 1), (0, 3, 1), (0, 4, 1),
           (1, 5, 4741), (1, 6, 3061), (2, 5, 4801), (2, 6, 6001),
           (3, 7, 9201), (3, 8, 9901), (4, 7, 8501), (4, 8, 7631),
           (5, 9, 8201), (5, 10, 1471), (6, 11, 1181), (6, 12, 1651),
           (7, 9, 1701), (7, 10, 1381), (8, 11, 1651), (8, 12, 1501),
           (9, 13, 1), (10, 13, 1), (11, 13, 1), (12, 13, 1)]

_RND22_W16 = [
    [498, 890, 437, 1064, 753, 572, 799, 934, 66, 779, 1801, 205, 1181, 699, 507, 317],
    [1269, 1507, 1887, 1927, 614, 1671, 1000, 1052, 1362, 1584, 1937, 613, 331, 926, 1435, 713],
    [1228, 646, 1746, 1869, 1803, 1509, 1671, 1797, 848, 1383, 1028, 1426, 1630, 1845, 1838, 1750],
    [179, 1322, 858, 1578, 1445, 992, 656, 1221, 1205, 1555, 969, 1165, 231, 1023, 1460, 1754],
    [431, 719, 1051, 1542, 913, 337, 776, 1121, 1259, 1708, 1352, 464, 1101, 719, 347, 1613],
    [1140, 965, 698, 891, 1285, 509, 301, 536, 452, 1244, 1857, 555, 392, 1414, 436, 1603],
    [909, 245, 769, 1882, 1635, 1566, 1344, 1518, 560, 896, 1863, 679, 319, 1720, 1367, 1638],
    [418, 175, 1406, 1651, 728, 763, 751, 840, 62, 1582, 486, 1068, 1041, 1187, 1009, 514],
    [484, 1082, 1215, 625, 280, 264, 307, 163, 744, 775, 918, 889, 121, 784, 349, 1224],
    [519, 122, 839, 1849, 192, 1341, 194, 742, 1252, 1370, 1575, 727, 509, 907, 1297, 563],
    [212, 756, 1445, 1625, 1058, 400, 620, 572, 265, 1725, 1822, 1258, 551, 338, 770, 372],
    [67, 126, 657, 1212, 553, 649, 70, 796, 856, 576, 704, 182, 264, 1040, 575, 317],
    [963, 376, 1348, 870, 698, 961, 212, 827, 479, 1648, 1475, 117, 888, 275, 1300, 400],
    [526, 903, 1530, 1189, 155, 1216, 327, 330, 338, 1523, 1404, 1382, 133, 285, 924, 1463],
    [425, 1114, 1063, 1586, 1075, 581, 367, 919, 916, 1565, 1649, 636, 357, 1108, 756, 1047],
    [1746, 1664, 1930, 1633, 1147, 1748, 1715, 1174, 487, 1884, 1051, 1739, 1190, 865, 1224, 1115],
    [727, 681, 1436, 1762, 1864, 1378, 1594, 1808, 981, 1669, 1878, 1434, 1576, 1570, 1812, 1595],
    [75, 981, 440, 544, 337, 1126, 271, 132, 1126, 1703, 1523, 373, 1020, 1072, 1050, 310],
    [658, 278, 1805, 1881, 1159, 461, 338, 828, 271, 1790, 979, 286, 412, 1649, 1121, 1659],
    [802, 1162, 1095, 1888, 1488, 380, 1221, 607, 126, 1229, 1168, 965, 895, 360, 1058, 1774],
    [165, 1227, 1680, 1045, 1736, 1166, 754, 418, 237, 928, 1800, 1187, 1121, 699, 258, 349],
    [478, 1889, 1649, 1384, 1858, 645, 1306, 435, 981, 439, 1892, 433, 809, 164, 1215, 1893],
]
_RND22_E = [(0, 2, 23), (2, 3, 30), (3, 4, 15), (4, 5, 19), (5, 6, 80), (0, 7, 10), (7, 8, 13),
            (8, 9, 15), (9, 10, 17), (10, 11, 16), (11, 1, 15), (6, 12, 32), (12, 13, 23),
            (13, 14, 23), (14, 15, 28), (15, 16, 20), (16, 21, 25), (6, 17, 20), (17, 18, 11),
            (18, 19, 19), (19, 20, 30), (20, 21, 27), (1, 21, 41), (8, 4, 21), (17, 16, 30),
            (19, 16, 16), (12, 15, 13), (19, 12, 11), (9, 4, 10), (3, 6, 20), (18, 1, 28),
            (19, 10, 20), (18, 16, 12), (7, 11, 54), (7, 4, 19)]

_IN0002_W = [[1269, 1507, 1887, 1927], [1228, 646, 1746, 1869], [179, 1322, 858, 1578],
             [431, 719, 1051, 1542], [1140, 965, 698, 891], [909, 245, 769, 1882],
             [418, 175, 1406, 1651], [484, 1082, 1215, 625], [519, 122, 839, 1849],
             [212, 756, 1445, 1625]]
_IN0002_E = [(0, 1, 18), (0, 2, 12), (0, 3, 9), (0, 4, 11), (0, 5, 14), (1, 7, 19), (1, 8, 16),
             (2, 6, 23), (3, 7, 27), (3, 8, 23), (4, 8, 13), (5, 7, 15), (6, 9, 17), (7, 9, 11),
             (8, 9, 13)]


def dissertation_fft8(m=16):
    W = np.array(_FFT8_W16, dtype=float)[:, :m]
    return Instance(f"diss_fft8_p{m}", W, list(_FFT8_E), meta={"source": "dissertation B.2"})


def dissertation_random22(m=16):
    W = np.array(_RND22_W16, dtype=float)[:, :m]
    return Instance(f"diss_rand22_p{m}", W, list(_RND22_E), meta={"source": "dissertation B.4"})


def dissertation_input0002():
    return Instance("diss_input_00_02_p4", np.array(_IN0002_W, float), list(_IN0002_E),
                    meta={"source": "dissertation Tabel III.2"})


# ---------------------------------------------------------------------------
# Synthetic generators
# ---------------------------------------------------------------------------
def _assign_costs(rng, n, m, edges_struct, ccr, beta, w_dag=50.0):
    """Topcuoglu et al. (2002) cost model: mean task cost w_i ~ U[0, 2 w_dag];
    w_ij ~ U[w_i(1-beta/2), w_i(1+beta/2)]; edge data ~ U[0, 2 ccr w_dag]
    (link bandwidth 1). Costs are clipped to a small positive value."""
    wbar = rng.uniform(0, 2 * w_dag, size=n)
    W = rng.uniform(wbar[:, None] * (1 - beta / 2), wbar[:, None] * (1 + beta / 2), size=(n, m))
    W = np.maximum(W, 1.0)
    edges = [(i, j, max(rng.uniform(0, 2 * ccr * w_dag), 0.1)) for i, j in edges_struct]
    return W, edges


def _rescale_ccr(W, edges, target):
    mc = np.mean([d for _, _, d in edges])
    f = target * W.mean() / mc
    return [(i, j, d * f) for i, j, d in edges]


def random_dag(n, m, ccr, beta, alpha=1.0, out_deg=3, seed=0, name=None):
    """Layered random DAG (Topcuoglu et al., 2002 generator): height ~ sqrt(n)/alpha levels,
    level widths random with mean alpha*sqrt(n); each task gets edges to tasks in later levels
    with average out-degree out_deg. A single entry and exit task are added."""
    rng = np.random.default_rng(seed)
    inner = n - 2
    height = max(1, int(round(np.sqrt(inner) / alpha)))
    width_mean = inner / height
    widths = np.maximum(1, rng.uniform(0, 2 * width_mean, size=height).round().astype(int))
    # adjust to exactly `inner` tasks
    while widths.sum() > inner:
        k = rng.integers(height)
        if widths[k] > 1:
            widths[k] -= 1
    while widths.sum() < inner:
        widths[rng.integers(height)] += 1
    levels, t = [], 1
    for w in widths:
        levels.append(list(range(t, t + w))); t += w
    exit_t = n - 1
    E = set()
    for li, lev in enumerate(levels[:-1]):
        later = [x for L in levels[li + 1:] for x in L]
        nxt = levels[li + 1]
        for u in lev:
            k = max(1, min(len(later), int(rng.poisson(out_deg)) or 1))
            # guarantee at least one edge to the next level to keep depth
            E.add((u, int(rng.choice(nxt))))
            for v in rng.choice(later, size=k, replace=False):
                E.add((u, int(v)))
    # every task with no predecessor gets one from previous level / entry
    has_pred = {j for _, j in E}
    for li, lev in enumerate(levels):
        for v in lev:
            if v not in has_pred:
                E.add((0, v) if li == 0 else (int(rng.choice(levels[li - 1])), v))
    has_succ = {i for i, _ in E}
    for lev in levels:
        for u in lev:
            if u not in has_succ:
                E.add((u, exit_t))
    for v in levels[0]:
        E.add((0, v))
    E = sorted(E)
    W, edges = _assign_costs(rng, n, m, E, ccr, beta)
    edges = _rescale_ccr(W, edges, ccr)
    nm = name or f"rand_n{n}_m{m}_ccr{ccr}_b{beta}_s{seed}"
    return Instance(nm, W, edges, meta=dict(kind="random", n=n, m=m, ccr=ccr, beta=beta, seed=seed))


def fft_dag(points, m, ccr, beta, seed=0):
    """FFT task graph (Topcuoglu et al., 2002): 2*points-1 recursive-call tasks followed by
    points*log2(points) butterfly tasks."""
    rng = np.random.default_rng(seed)
    E, idx = [], 0
    # recursive call tree (binary), root 0
    rec = list(range(2 * points - 1))
    for k in range(points - 1):
        E.append((k, 2 * k + 1)); E.append((k, 2 * k + 2))
    leaves = rec[points - 1:]
    nxt = 2 * points - 1
    prev = leaves
    stages = int(np.log2(points))
    for s in range(stages):
        cur = list(range(nxt, nxt + points)); nxt += points
        span = 2 ** s
        for i in range(points):
            partner = i ^ span
            E.append((prev[i], cur[i])); E.append((prev[partner], cur[i]))
        prev = cur
    n = nxt
    E = sorted(set(E))
    W, edges = _assign_costs(rng, n, m, E, ccr, beta)
    edges = _rescale_ccr(W, edges, ccr)
    return Instance(f"fft{points}_m{m}_ccr{ccr}_b{beta}_s{seed}", W, edges,
                    meta=dict(kind="fft", n=n, m=m, ccr=ccr, beta=beta, seed=seed))


def gauss_dag(size, m, ccr, beta, seed=0):
    """Gaussian elimination task graph (Topcuoglu et al., 2002), matrix size `size`:
    (size^2 + size - 2)/2 tasks."""
    rng = np.random.default_rng(seed)
    E, tid = [], {}
    c = 0
    for k in range(size - 1):
        tid[("p", k)] = c; c += 1
        for j in range(k + 1, size):
            tid[("u", k, j)] = c; c += 1
    for k in range(size - 1):
        piv = tid[("p", k)]
        for j in range(k + 1, size):
            E.append((piv, tid[("u", k, j)]))
            if k + 1 < size - 1:
                if j == k + 1:
                    E.append((tid[("u", k, j)], tid[("p", k + 1)]))
                else:
                    E.append((tid[("u", k, j)], tid[("u", k + 1, j)]))
    n = c
    W, edges = _assign_costs(rng, n, m, sorted(set(E)), ccr, beta)
    edges = _rescale_ccr(W, edges, ccr)
    return Instance(f"gauss{size}_m{m}_ccr{ccr}_b{beta}_s{seed}", W, edges,
                    meta=dict(kind="gauss", n=n, m=m, ccr=ccr, beta=beta, seed=seed))
