"""opt-aiNet (dissertation settings, 200 generations) on the three dissertation instances."""
from paths import RES, N_JOBS, rpath, instance_filter, runs  # noqa: F401
import json, multiprocessing as mp
from exp_bug import job
if __name__ == "__main__":
    jobs = [(ii, "opt-aiNet(200g)", r) for ii in range(3) for r in range(5)]
    with mp.Pool(N_JOBS) as pool, open(rpath("bug_opt.jsonl"), "w") as fh:
        for d in pool.imap_unordered(job, jobs):
            fh.write(json.dumps(d) + "\n"); fh.flush()
    print("done")
