"""Write the compact JSON for this presentation (schema percolation-offline/1) from finished runs.

Usage: export.py --tree <tree folder> --run <run folder> [--summary phase2_summary.json] [--out <json>]
The declared run gets every block in full; --summary (diagnostics.py) adds the other runs in compact form (5-bin
statistics, decision, gates, every tenth epoch of the training curve) and the phase-2 diagnostics.
"""
import argparse
import datetime
import json
from pathlib import Path

import numpy as np

import common as C

MAP_POINTS, MAP_FLOOR, MAP_SITE = 8000, 200, "hidden2"
DEFAULT_OUT = C.HERE.parent.parent / "public" / "data" / "percolation-offline.json"
LABEL = {"raw": "raw input x", "input_layer": "residual stream after the input layer",
         **{f"hidden{i}": f"hidden layer of MLP block {i}" for i in range(3)},
         **{f"block{i}": f"residual stream after MLP block {i}" for i in range(3)}}


def sig(v, n=4):
    if v is None or not np.isfinite(v):
        return None
    return float(f"{v:.{n}g}")


def sigs(a, n=4):
    return [sig(float(v), n) for v in a]


def rnd(a, d):
    return [None if v is None or not np.isfinite(v) else round(float(v), d) for v in a]


def q(block):
    return {k: sigs([np.nan if v is None else v for v in block[k]]) for k in ("median", "q25", "q75")}


def map_block(data, probes, run_id):
    raw, tr, un = probes["raw"], probes["trained"], probes["untrained"]
    depth = int(raw["map_depth"])
    rows, ntr = raw["map_rows"], int(raw["map_ntr"])
    assert np.array_equal(rows, tr["map_rows"]) and np.array_equal(rows, un["map_rows"])
    col = list(data["anc_depths"]).index(depth)
    fam_lat = data["anc"][rows, col]
    lats, sizes = np.unique(fam_lat, return_counts=True)
    total = sizes.sum()
    # Families above the floor share one sampling rate, chosen so the total is MAP_POINTS.
    rate = MAP_POINTS / total
    for _ in range(50):
        quota = np.minimum(sizes, np.maximum(MAP_FLOOR, np.round(rate * sizes))).astype(int)
        free = quota > np.minimum(sizes, MAP_FLOOR)
        if not free.any():
            break
        rate = (MAP_POINTS - quota[~free].sum()) / sizes[free].sum()
    rng = np.random.default_rng(0)
    pick = np.sort(np.concatenate([rng.choice(np.nonzero(fam_lat == l)[0], k, replace=False)
                                   for l, k in zip(lats, quota)]))
    X = data["X"]
    Xc = X - X.mean(0)
    evals, evecs = np.linalg.eigh(Xc.T @ Xc / len(X))
    axes = evecs[:, ::-1][:, :2]
    # PCA axes can be reflected without changing the projection. Fix the orientation to the
    # original plot: the largest-magnitude loading on each of its axes is negative.
    dominant = np.argmax(np.abs(axes), axis=0)
    axes = axes * -np.sign(axes[dominant, np.arange(axes.shape[1])])
    xy = Xc[rows[pick]] @ axes
    fam_index = np.searchsorted(lats, fam_lat[pick])
    return {
        "run": run_id, "depth": depth, "n_points": int(len(pick)), "site": MAP_SITE,
        "coords_note": "first two principal axes of x over all 200,000 points",
        "explained_variance": rnd(evals[::-1][:2] / evals.sum(), 4),
        "families": {"latent": lats.tolist(), "size": sizes.tolist(), "z": rnd(data["lat_z"][lats], 3),
                     "sampled": quota.tolist(), "weight": rnd(sizes / quota, 3)},
        "points": {"x": rnd(xy[:, 0], 2), "y": rnd(xy[:, 1], 2), "family": fam_index.tolist(),
                   "probe_val": (pick >= ntr).astype(int).tolist(), "net_split": data["split"][rows[pick]].tolist(),
                   "pred_raw": rnd(raw["map_pred_raw"][pick], 3),
                   "pred_trained": rnd(tr[f"map_pred_{MAP_SITE}"][pick], 3),
                   "pred_untrained": rnd(un[f"map_pred_{MAP_SITE}"][pick], 3)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--summary", help="phase2_summary.json from diagnostics.py: adds the other runs and diagnostics")
    args = ap.parse_args()
    tree = Path(args.tree)
    tree_id = tree.name
    data = np.load(tree / "dataset.npz")
    summ, base = C.load_json(tree / "data_summary.json"), C.load_json(tree / "baselines.json")
    raw = np.load(tree / "probes_raw.npz")

    ent_lat, ent_depth = raw["latent"], raw["depth"]
    row_of = {int(l): i for i, l in enumerate(ent_lat)}
    lat_parent = data["lat_parent"]
    parent_entry = []
    for l in ent_lat:
        p = int(lat_parent[l])
        while p >= 0 and p not in row_of:
            p = int(lat_parent[p])
        parent_entry.append(row_of.get(p, -1))
    first = C.load_json(Path(args.run[0]) / "analysis.json")
    kept = (raw["baseline_mse"] >= C.MIN_BASELINE_MSE).astype(int)
    elig = summ["eligible"]

    doc = {
        "schema": "percolation-offline/1", "status": "phase-1",
        "provenance": {
            "generator": summ["provenance"]["generator"],
            "protocol": "Brill & Carlson, arXiv:2606.20347, Sec. 5 and App. B, one-cluster dataset",
            "analysis_note": "Right-closed size bins exclude the observed minimum and include the maximum; both endpoints are set explicitly. See offline/percolation/README.md for the correction to the original run summaries.",
            "pipeline": {"path": "offline/percolation", **C.pipeline_revision()},
            "environment": C.load_json(Path(args.run[0]) / "metrics.json")["environment"],
            "training_pipeline": C.load_json(Path(args.run[0]) / "metrics.json")["pipeline"],
            "probe_pipeline": C.load_json(Path(args.run[0]) / "probes_trained.json")["pipeline"],
            "export_environment": C.environment(),
            "date": datetime.date.today().isoformat(),
            "decision_rule": "fixed before training; see offline/percolation/README.md and analyze.py",
            "runs": [],
        },
        "constants": {"n": C.N_POINTS, "d": C.DIM, "d_model": C.D_MODEL, "blocks": C.N_BLOCKS,
                      "expansion": C.EXPANSION, "params": None, "epochs": C.EPOCHS, "batch": C.BATCH,
                      "split": list(C.SPLIT), "probe_depths": C.PROBE_DEPTHS, "min_latent_points": C.MIN_LATENT_POINTS,
                      "probe_val_frac": C.PROBE_VAL_FRAC, "ridge_alpha": C.RIDGE_ALPHA,
                      "min_baseline_mse": C.MIN_BASELINE_MSE, "fom_bins": 5, "fom_bins_fig4": 8,
                      "thresholds": {"helps": 0.85, "no_gain": 0.95}},
        "sites": ["raw"] + C.SITES,
        "site_type": {"raw": "input", **{s: ("hidden" if s.startswith("hidden") else "residual") for s in C.SITES}},
        "site_label": LABEL,
        "trees": {tree_id: {
            "seeds": summ["seeds"], "latents": summ["latents"],
            "point_depth": {k: summ["point_depth"][k] for k in ("mean", "median", "max")},
            "eligible_by_depth": {"depth": [r["depth"] for r in elig], "points": [r["points"] for r in elig],
                                  "latents": [r["latents"] for r in elig], "largest": [r["max_size"] for r in elig]},
            "entries_probed": summ["entries_probed_10_200"],
            "largest_share": rnd(summ["largest_latent_share_of_eligible"], 4)}},
        "baselines": {"paper": base["paper_table1_one_cluster"],
                      tree_id: {"ridge": {"r2": round(base["ridge"]["r2"], 4), "mse": sig(base["ridge"]["mse"])},
                                "nn1": {"r2": round(base["nn1"]["r2"], 4), "mse": sig(base["nn1"]["mse"]),
                                        "r2_tie_averaged": round(base["nn1"]["r2_tie_averaged"], 4),
                                        "mse_tie_averaged": sig(base["nn1"]["mse_tie_averaged"])}}},
        "networks": {}, "entries": {tree_id: {
            "latent": ent_lat.tolist(), "depth": ent_depth.tolist(), "size": raw["size"].tolist(),
            "fom": sigs(raw["size"] / np.sqrt(C.N_POINTS)), "z": rnd(raw["z"], 3), "n_val": raw["n_val"].tolist(),
            "baseline_mse": sigs(raw["baseline_mse"]), "kept": kept.tolist(),
            "bin5": first["bins5"]["bin_of_entry"], "bin8": first["bins8"]["bin_of_entry"],
            "parent_entry": parent_entry}},
        "probe_mse": {}, "binned": {}, "depth_curves": {}, "gates": {}, "decision": {},
    }
    probes_first = None
    for i, rd in enumerate(args.run):
        run = Path(rd)
        rid = run.name
        met, an = C.load_json(run / "metrics.json"), C.load_json(run / "analysis.json")
        tr, un = np.load(run / "probes_trained.npz"), np.load(run / "probes_untrained.npz")
        if i == 0:
            probes_first = {"raw": raw, "trained": tr, "untrained": un}
            doc["constants"]["params"] = met["params"]
        doc["provenance"]["runs"].append({"id": rid, "role": "declared" if i == 0 else "extension", "tree": tree_id,
                                          "init": 0 if met["seed"] == C.TRAIN_SEED else met["seed"],
                                          "target": met["target"], "status": "complete"})
        h = met["history"]
        doc["networks"][rid] = {"test": {"r2": round(met["test"]["r2"], 4), "mse": sig(met["test"]["mse"])},
                                "test_after_250_epochs": {k: (round(v, 4) if k == "r2" else sig(v))
                                                          for k, v in met.get("test_after_250_epochs", {}).items()},
                                "curve": {"epoch": h["epoch"], "train_loss": sigs(h["train_loss"]),
                                          "val_r2": rnd(h["val_r2"], 4)},
                                "initial": {"train_loss": sig(met["initial_train_loss"])}}
        doc["probe_mse"][rid] = {"raw": sigs(raw["mse_raw"]),
                                 "trained": {s: sigs(tr[f"mse_{s}"]) for s in C.SITES},
                                 "untrained": {s: sigs(un[f"mse_{s}"]) for s in C.SITES}}
        bl = {}
        for nb in (5, 8):
            b = an[f"bins{nb}"]
            bl[f"bins{nb}"] = {"edges": sigs(b["edges"]), "centers": sigs(b["centers"]), "count": b["count"],
                               "mse": {"raw": q(b["mse"]["raw"]),
                                       **{c: {s: q(b["mse"][c][s]) for s in C.SITES} for c in ("trained", "untrained")}},
                               "ratio_to_raw": {c: {s: q(b["ratio_to_raw"][c][s]) for s in C.SITES}
                                                for c in ("trained", "untrained")},
                               "ratio_trained_untrained": {s: q(b["ratio_trained_untrained"][s]) for s in C.SITES}}
        doc["binned"][rid] = bl
        dc = an["depth_curves"]
        doc["depth_curves"][rid] = {
            "depth": dc["depth"], "baseline_mse": sigs(dc["baseline_mse"]), "raw": sigs(dc["raw"]),
            **{c: {s: sigs(dc[c][s]) for s in C.SITES} for c in ("trained", "untrained")},
            "r2": {"raw": rnd(raw["g_r2_raw"], 4),
                   **{c: {s: rnd(p[f"g_r2_{s}"], 4) for s in C.SITES} for c, p in (("trained", tr), ("untrained", un))}}}
        doc["gates"][rid] = {"A": an["gate_A"], "B": an["gate_B"], "C": an["gate_C"],
                             "D": {k: an["gate_D"][k] for k in ("pass", "ratio_below_1", "gap_shrinks")}}
        d = an["decision"]
        doc["decision"][rid] = {"label": d["label"], "bins_used": d["bins_used"],
                                "cells": {s: sigs(v) for s, v in d["cells"].items()}, "call": d["call"],
                                "sensitivity": {"-0.05": d["sensitivity"]["-0.05"], "+0.05": d["sensitivity"]["+0.05"],
                                                "residual_without_input_layer":
                                                    d["sensitivity"]["residual_without_input_layer"],
                                                "pooled": {t: {"cells": sigs(v["cells"]), "call": v["call"]}
                                                           for t, v in d["sensitivity"]["pooled"].items()}}}
    if args.summary:
        ex = C.load_json(args.summary)["export"]
        doc["status"] = "phase-2"
        doc["provenance"]["runs"] += ex["runs"]
        for k in ("trees", "baselines", "networks", "binned", "gates", "decision"):
            doc[k].update(ex[k])
        doc["decision_across_runs"] = ex["decision_across_runs"]
        doc["diagnostics"] = ex["diagnostics"]
    doc["map"] = map_block(data, probes_first, Path(args.run[0]).name)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, separators=(",", ":"), allow_nan=False)
    out.write_text(text)
    sizes = {k: len(json.dumps(v, separators=(",", ":"))) for k, v in doc.items()}
    print(f"wrote {out} ({len(text) / 1024:.0f} KB); by block (KB):",
          {k: round(v / 1024, 1) for k, v in sizes.items()})


if __name__ == "__main__":
    main()
