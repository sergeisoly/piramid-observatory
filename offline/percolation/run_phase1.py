"""Run the whole phase-1 sequence: data (gate A), baselines (gate B), training (gate C), probes, analysis, export.

Usage: run_phase1.py --generator <checkout at 3dd47a8> --out <folder>
Each step writes its log to <out>/logs/<step>.log; finished steps are skipped on a rerun, and an interrupted
training resumes from its last checkpoint. Step times go to <out>/phase1_steps.json. The run stops if gate B or C
fails (pre-registered stopping rule) unless --continue-on-gate-fail is given.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--continue-on-gate-fail", action="store_true")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    tree, run = out / "tree0", out / "tree0-init0"
    logs = out / "logs"
    for d in (tree, run, logs):
        d.mkdir(parents=True, exist_ok=True)
    steps_file = out / "phase1_steps.json"
    record = C.load_json(steps_file) if steps_file.exists() else {}

    def step(name, script, *a, done=None):
        if done is not None and done.exists():
            print(f"[skip] {name}: {done.name} exists", flush=True)
            return
        cmd = [sys.executable, str(C.HERE / script), *map(str, a)]
        print(f"[run ] {name}: {' '.join(cmd)}", flush=True)
        t0 = time.time()
        with open(logs / f"{name}.log", "a") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        record[name] = {"seconds": round(time.time() - t0, 1), "returncode": rc,
                        "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        C.save_json(steps_file, record)
        if rc != 0:
            raise SystemExit(f"{name} failed with code {rc}; see {logs / (name + '.log')}")
        print(f"[done] {name} in {record[name]['seconds']} s", flush=True)

    def gate(ok, label):
        print(f"[gate] {label}: {'pass' if ok else 'FAIL'}", flush=True)
        if not ok and not args.continue_on_gate_fail:
            raise SystemExit(f"{label} failed: stopping as pre-registered")

    step("generate", "generate.py", "--generator", args.generator, "--out", tree, done=tree / "data_summary.json")
    gate(C.load_json(tree / "data_summary.json")["row_identity_and_structure_checks_pass"], "gate A checks")
    step("baselines", "baselines.py", "--tree", tree, done=tree / "baselines.json")
    gate(C.load_json(tree / "baselines.json")["gate_B"]["pass"], "gate B")
    resume = ["--resume"] if any(run.glob("ckpt_e*.pt")) else []
    step("train", "train.py", "--tree", tree, "--out", run, *resume, done=run / "metrics.json")
    r2 = C.load_json(run / "metrics.json")["test"]["r2"]
    gate(abs(r2 - 0.94) <= 0.03, f"gate C (test R2 {r2:.4f})")
    step("probe_raw", "probe.py", "--tree", tree, "--state", "raw", "--sklearn-check", "raw:50",
         done=tree / "probes_raw.npz")
    step("probe_untrained", "probe.py", "--tree", tree, "--run", run, "--state", "init", "--sklearn-check",
         "hidden0:50", done=run / "probes_untrained.npz")
    step("probe_trained", "probe.py", "--tree", tree, "--run", run, "--state", "final", "--sklearn-check",
         "hidden0:50", done=run / "probes_trained.npz")
    step("analyze", "analyze.py", "--tree", tree, "--run", run)
    step("export", "export.py", "--tree", tree, "--run", run, "--out", out / "export" / "percolation-offline.json")
    print("phase 1 complete", flush=True)


if __name__ == "__main__":
    main()
