"""Reproduce the schedules printed in the dissertation (Zulianto, ITB, 2016) with the
shared decoder: Listing C.3 (HEFT, FFT-8, 16 processors, reported 11,528), Listing C.4
(ODE opt-aiNet, reported 6,389) and Gambar III.8 (input_00_02, reported 2,763).
No external dependencies beyond numpy/numba."""
import numpy as np
from instance import dissertation_fft8, dissertation_input0002, dissertation_random22
from decode import heft, makespan, cpop, peft


def mapping_from_listing(n, listing):
    mp = np.full(n, -1, dtype=np.int64)
    for p, tasks in listing.items():
        for t in tasks:
            mp[t] = p
    assert (mp >= 0).all()
    return mp


def check(label, got, expected):
    ok = abs(got - expected) < 1e-9
    print(f"  {'OK ' if ok else 'FAIL'} {label}: {got:,.0f} (dissertation: {expected:,})")
    return ok


if __name__ == "__main__":
    all_ok = True
    fft = dissertation_fft8(16)
    order = fft.priority_order()
    args = (fft.W, fft.pptr, fft.pidx, fft.pdat, fft.B)
    print(f"FFT-8 on 16 processors (CCR = {fft.ccr():.2f})")
    heft_diss = mapping_from_listing(14, {0: [0], 6: [1, 12], 7: [3, 8, 7, 11, 10],
                                          9: [4, 5, 9, 13], 11: [2, 6]})
    ode_diss = mapping_from_listing(14, {0: [0], 3: [2, 6, 12], 4: [11], 5: [1],
                                         8: [5, 9, 10], 9: [13], 13: [3, 4, 8, 7]})
    all_ok &= check("HEFT mapping of Listing C.3", makespan(order, heft_diss, *args, False), 11528)
    all_ok &= check("ODE mapping of Listing C.4", makespan(order, ode_diss, *args, False), 6389)
    all_ok &= check("our HEFT, computed independently", heft(fft)[1], 11528)
    print(f"  serial on the best single processor: {fft.W.sum(axis=0).min():,.0f}  (not reported in the dissertation)")
    print(f"  CPOP {cpop(fft)[1]:,.0f}   PEFT {peft(fft)[1]:,.0f}")

    i02 = dissertation_input0002()
    sol = np.array([0, 1, 0, 0, 3, 2, 1, 0, 1, 0])
    print("\ninput_00_02 on 4 processors")
    all_ok &= check("solution of Gambar III.8",
                    makespan(i02.priority_order(), sol, i02.W, i02.pptr, i02.pidx, i02.pdat, i02.B, False), 2763)

    r22 = dissertation_random22(16)
    print(f"\nrandom-22 on 16 processors (CCR = {r22.ccr():.3f}): HEFT {heft(r22)[1]:,.0f}, "
          f"PEFT {peft(r22)[1]:,.0f}, CPOP {cpop(r22)[1]:,.0f}")
    print("\nAll dissertation schedules reproduced." if all_ok else "\nSOME CHECKS FAILED")
