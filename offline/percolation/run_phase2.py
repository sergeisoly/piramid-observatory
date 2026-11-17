"""Run phase 2 on top of a finished phase 1 (same --out folder).

Repeats (initializations 1 and 2 on the paper's tree; trees with graph seeds 1 and 2), the value-swap network, the
held-out and ridge-strength variants of the declared run, then diagnostics.py and the export. Independent jobs run in
parallel (up to --jobs at a time); logs go to <out>/logs/, step times to <out>/phase2_steps.json. Finished steps are
skipped on a rerun.
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import common as C

VSEED = 20001


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--export-to", help="export path (default <out>/export/percolation-offline.json)")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    logs = out / "logs"
    steps_file = out / "phase2_steps.json"
    record = C.load_json(steps_file) if steps_file.exists() else {}
    t0_dir, r0 = out / "tree0", out / "tree0-init0"
    values = t0_dir / f"values_v{VSEED}.npz"

    def cmd(script, *a):
        return [sys.executable, str(C.HERE / script), *map(str, a)]

    def run_many(jobs):
        """jobs: list of (name, command, done_marker). Runs up to --jobs at once; stops on the first failure."""
        todo = [j for j in jobs if not (j[2] and Path(j[2]).exists())]
        for name, _, _ in jobs:
            if all(name != t[0] for t in todo):
                print(f"[skip] {name}", flush=True)
        running = {}
        while todo or running:
            while todo and len(running) < args.jobs:
                name, c, _ = todo.pop(0)
                log = open(logs / f"{name}.log", "a")
                running[name] = (subprocess.Popen(c, stdout=log, stderr=subprocess.STDOUT), time.time(), log)
                print(f"[run ] {name}: {' '.join(c[1:])}", flush=True)
            time.sleep(2)
            for name in list(running):
                proc, t0, log = running[name]
                if proc.poll() is None:
                    continue
                log.close()
                del running[name]
                record[name] = {"seconds": round(time.time() - t0, 1), "returncode": proc.returncode,
                                "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
                C.save_json(steps_file, record)
                if proc.returncode != 0:
                    for p, _, _ in running.values():
                        p.wait()
                    raise SystemExit(f"{name} failed; see {logs / (name + '.log')}")
                print(f"[done] {name} in {record[name]['seconds']} s", flush=True)

    # Keep the phase-1 probe outputs of the declared run, to check that they reproduce.
    backup = out / "phase1_backup"
    if not backup.exists():
        for d in ("tree0", "tree0-init0"):
            (backup / d).mkdir(parents=True)
            for f in list((out / d).glob("probes_*")) + list((out / d).glob("analysis*.json")):
                shutil.copy2(f, backup / d / f.name)
            for f in (out / d).glob("probes_*"):
                f.unlink()
    T = time.time()

    # 1. Data: two more trees, value-swapped values on the paper's tree.
    trees = {1: out / "tree1", 2: out / "tree2"}
    run_many([(f"generate_tree{g}", cmd("generate.py", "--generator", args.generator, "--out", d, "--graph", g,
                                        "--embed", 10_000 + g, "--value", 20_000 + g), d / "data_summary.json")
              for g, d in trees.items()]
             + [("generate_values", cmd("generate.py", "--generator", args.generator, "--out", t0_dir, "--value-seed",
                                        VSEED), values)])
    run_many([(f"baselines_tree{g}", cmd("baselines.py", "--tree", d), d / "baselines.json") for g, d in trees.items()])

    # 2. Trainings in parallel, then the declared run's variants while they train.
    trains = [("train_tree0-init1", cmd("train.py", "--tree", t0_dir, "--out", out / "tree0-init1", "--seed", 43)),
              ("train_tree0-init2", cmd("train.py", "--tree", t0_dir, "--out", out / "tree0-init2", "--seed", 44)),
              ("train_tree1-init0", cmd("train.py", "--tree", trees[1], "--out", out / "tree1-init0")),
              ("train_tree2-init0", cmd("train.py", "--tree", trees[2], "--out", out / "tree2-init0")),
              ("train_tree0-vswap", cmd("train.py", "--tree", t0_dir, "--out", out / "tree0-vswap", "--values", values))]
    trains = [(n, c, Path(c[c.index("--out") + 1]) / "metrics.json") for n, c in trains]
    probes = []
    for state, tag in (("raw", "raw"), ("init", "untrained"), ("final", "trained")):
        base = ["--tree", t0_dir, "--state", state] + ([] if state == "raw" else ["--run", r0])
        where = t0_dir if state == "raw" else r0
        probes += [(f"probe_declared_{tag}", cmd("probe.py", *base), where / f"probes_{tag}.npz"),
                   (f"probe_declared_{tag}_hofit", cmd("probe.py", *base, "--fit-rows", "heldout"),
                    where / f"probes_{tag}_hofit.npz"),
                   (f"probe_declared_{tag}_a100", cmd("probe.py", *base, "--alpha", 100), where / f"probes_{tag}_a100.npz"),
                   (f"probe_declared_{tag}_a10000", cmd("probe.py", *base, "--alpha", 10000),
                    where / f"probes_{tag}_a10000.npz"),
                   (f"probe_declared_{tag}_zv{VSEED}", cmd("probe.py", *base, "--values", values),
                    where / f"probes_{tag}_zv{VSEED}.npz")]
    run_many(trains + probes)

    # 3. Probes of the new networks.
    jobs = []
    for rid, tree in (("tree0-init1", t0_dir), ("tree0-init2", t0_dir), ("tree1-init0", trees[1]),
                      ("tree2-init0", trees[2])):
        for state, tag in (("init", "untrained"), ("final", "trained")):
            jobs.append((f"probe_{rid}_{tag}", cmd("probe.py", "--tree", tree, "--run", out / rid, "--state", state),
                         out / rid / f"probes_{tag}.npz"))
    for g, d in trees.items():
        jobs.append((f"probe_tree{g}_raw", cmd("probe.py", "--tree", d, "--state", "raw"), d / "probes_raw.npz"))
    rv = out / "tree0-vswap"
    jobs += [("probe_vswap_trained_z", cmd("probe.py", "--tree", t0_dir, "--run", rv, "--state", "final"),
              rv / "probes_trained.npz"),
             (f"probe_vswap_trained_zv{VSEED}", cmd("probe.py", "--tree", t0_dir, "--run", rv, "--state", "final",
                                                    "--values", values), rv / f"probes_trained_zv{VSEED}.npz")]
    run_many(jobs)

    # 4. Analyses, diagnostics, export.
    an = [("analyze_declared", cmd("analyze.py", "--tree", t0_dir, "--run", r0), None),
          ("analyze_declared_ho", cmd("analyze.py", "--tree", t0_dir, "--run", r0, "--metric", "ho"), None),
          ("analyze_declared_hofit", cmd("analyze.py", "--tree", t0_dir, "--run", r0, "--suffix", "_hofit"), None),
          ("analyze_declared_a100", cmd("analyze.py", "--tree", t0_dir, "--run", r0, "--suffix", "_a100"), None),
          ("analyze_declared_a10000", cmd("analyze.py", "--tree", t0_dir, "--run", r0, "--suffix", "_a10000"), None)]
    for rid, tree in (("tree0-init1", t0_dir), ("tree0-init2", t0_dir), ("tree1-init0", trees[1]),
                      ("tree2-init0", trees[2])):
        an.append((f"analyze_{rid}", cmd("analyze.py", "--tree", tree, "--run", out / rid), None))
    run_many(an)
    run_many([("diagnostics", cmd("diagnostics.py", "--root", out), None)])
    dest = args.export_to or out / "export" / "percolation-offline.json"
    run_many([("export", cmd("export.py", "--tree", t0_dir, "--run", r0, "--summary", out / "phase2_summary.json",
                             "--out", dest), None)])
    record["total_seconds"] = round(time.time() - T, 1)
    C.save_json(steps_file, record)
    print(f"phase 2 complete in {record['total_seconds']} s", flush=True)


if __name__ == "__main__":
    main()
