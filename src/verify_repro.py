"""Bit-exact reproducibility check. Re-runs a subset of the main experiment into a scratch
directory and compares every makespan with the archived raw results.

    python src/verify_repro.py              # default: 3 instances x all 21 algorithms x 2 runs
    python src/verify_repro.py 0,25,48 3    # instances, runs
"""
import json, os, subprocess, sys, tempfile

SCRIPTS = [("run_exp.py", "main.jsonl"), ("run_exp2.py", "main2.jsonl"), ("run_exp3.py", "main3.jsonl")]


def main():
    insts = sys.argv[1] if len(sys.argv) > 1 else "0,25,48"
    nruns = sys.argv[2] if len(sys.argv) > 2 else "2"
    here = os.path.dirname(os.path.abspath(__file__))
    archive = os.environ.get("RESULTS_DIR", "results")
    scratch = tempfile.mkdtemp(prefix="repro_")
    env = dict(os.environ, RESULTS_DIR=scratch, INSTANCES=insts, RUNS=nruns)
    total = mismatch = 0
    for script, fname in SCRIPTS:
        subprocess.run([sys.executable, os.path.join(here, script)], env=env, check=True,
                       stdout=subprocess.DEVNULL)
        ref = {(d["ii"], d["alg"], d["run"]): d["cost"]
               for d in map(json.loads, open(os.path.join(archive, fname)))}
        for d in map(json.loads, open(os.path.join(scratch, fname))):
            total += 1
            r = ref.get((d["ii"], d["alg"], d["run"]))
            if r is None or abs(r - d["cost"]) > 1e-9 * max(1.0, abs(r)):
                mismatch += 1
                print("MISMATCH", d["inst"], d["alg"], d["run"], d["cost"], "archived:", r)
    print(f"Re-ran {total} runs (instances {insts}, {nruns} runs each): "
          f"{total - mismatch} identical to the archived results, {mismatch} different.")
    return mismatch == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
