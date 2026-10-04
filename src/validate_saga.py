"""Cross-check our list schedulers against the SAGA reference implementations
(Coleman & Krishnamachari, arXiv:2403.07120). Requires: pip install -r requirements-saga.txt"""
import logging
logging.disable(logging.WARNING)
import numpy as np
from saga import Network, TaskGraph
from saga.schedulers.heft import HeftScheduler
from instance import Instance
from decode import heft


def saga_vs_ours(trials=200, seed=1):
    rng = np.random.default_rng(seed)
    diffs = []
    for k in range(trials):
        n, m = int(rng.integers(5, 40)), int(rng.integers(2, 8))
        speeds = rng.uniform(0.5, 2.0, m)
        bw = rng.uniform(0.5, 2.0, (m, m)); bw = (bw + bw.T) / 2
        cost = rng.uniform(1, 100, n)
        # random DAG on 0..n-1 (i<j)
        E = [(i, j, float(rng.uniform(1, 100))) for i in range(n) for j in range(i + 1, n)
             if rng.random() < 0.15]
        net = Network.create([(f"p{q}", float(speeds[q])) for q in range(m)],
                             [(f"p{a}", f"p{b}", float(bw[a, b])) for a in range(m) for b in range(a + 1, m)])
        tg = TaskGraph.create([(f"t{i}", float(cost[i])) for i in range(n)],
                              [(f"t{i}", f"t{j}", d) for i, j, d in E])
        sch = HeftScheduler().schedule(net, tg)
        ms_saga = max(t.end for tasks in sch.mapping.values() for t in tasks)
        W = cost[:, None] / speeds[None, :]
        inst = Instance(f"v{k}", W, E, B=bw)
        _, ms_ours, _ = heft(inst, insertion=True, saga_mode=True)
        diffs.append(abs(ms_ours - ms_saga) / ms_saga)
    diffs = np.array(diffs)
    return (diffs < 1e-9).mean(), diffs.max()



def cpop_peft_vs_saga():
    from saga.schedulers.cpop import CpopScheduler
    from saga.schedulers.peft import PEFTScheduler
    from decode import cpop, peft
    rng=np.random.default_rng(3); res={'cpop':[], 'peft':[]}
    for k in range(150):
        n,m=int(rng.integers(5,30)),int(rng.integers(2,6))
        sp=rng.uniform(.5,2,m); bw=np.ones((m,m))
        cost=rng.uniform(1,100,n)
        E=[(i,j,float(rng.uniform(1,100))) for i in range(n) for j in range(i+1,n) if rng.random()<.15]
        # single entry/exit to avoid SAGA super nodes
        E=[(0,j,1.0) for j in range(1,n-1)] + [(i,n-1,1.0) for i in range(1,n-1)] + [(i,j,d) for i,j,d in E if i>0 and j<n-1]
        net=Network.create([(f"p{q}",float(sp[q])) for q in range(m)],[(f"p{a}",f"p{b}",1.0) for a in range(m) for b in range(a+1,m)])
        tg=TaskGraph.create([(f"t{i}",float(cost[i])) for i in range(n)],[(f"t{i}",f"t{j}",d) for i,j,d in E])
        inst=Instance(f"v{k}",cost[:,None]/sp[None,:],E); inst.rank_cfac=(m-1)/(m+1)
        for nm,S,f in [('cpop',CpopScheduler,cpop),('peft',PEFTScheduler,peft)]:
            s=S().schedule(net,tg); ms=max(t.end for ts in s.mapping.values() for t in ts)
            res[nm].append(f(inst)[1]/ms)
    for nm,v in res.items():
        v=np.array(v); print(nm,'ours/saga: identical %.0f%%, mean %.4f, min %.3f max %.3f'%((abs(v-1)<1e-9).mean()*100,v.mean(),v.min(),v.max()))


if __name__ == "__main__":
    exact, worst = saga_vs_ours()
    print(f"HEFT vs SAGA: identical makespan on {exact*100:.1f}% of 200 instances; max rel. diff {worst:.2e}")
    cpop_peft_vs_saga()
