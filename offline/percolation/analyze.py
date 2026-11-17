"""Gates C and D, FOM-binned statistics, and the pre-registered trained-vs-untrained decision for one run.

Reads the tree's data_summary/baselines/probes_raw and the run's metrics/probes_trained/probes_untrained.
Writes <run>/analysis.json. Bins follow the authors' plotting script: log-spaced edges from the minimum to the maximum
FOM of the entries kept by the baseline filter, pandas.cut semantics (right-closed, the minimum falls out).
"""
import argparse
from pathlib import Path

import numpy as np
from scipy import stats

import common as C

PAPER_R2 = {"ridge": 0.51, "nn1": 0.94, "mlp": 0.94}
BAND = 0.03
MIN_BIN = 10
BLOCK_SITES = ["block0", "block1", "block2"]


def fom_bins(fom, kept, n_bins):
    f = fom[kept]
    edges = np.logspace(np.log10(f.min()), np.log10(f.max()), n_bins + 1)
    # Exponentiating log10 of an endpoint can move it across its own bin boundary.
    # Anchor both ends to the observations to keep (min, max] identical on every platform.
    edges[0], edges[-1] = f.min(), f.max()
    b = np.searchsorted(edges, fom, side="left") - 1  # (edges[i], edges[i+1]] -> i, as pandas.cut(right=True)
    b[(b < 0) | (b >= n_bins) | ~kept] = -1
    return b, edges


def quart(v):
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return None, None, None
    q = np.quantile(v, [0.25, 0.5, 0.75])
    return float(q[1]), float(q[0]), float(q[2])


def binned(values, b, n_bins):
    out = {"median": [], "q25": [], "q75": []}
    for i in range(n_bins):
        m, lo, hi = quart(values[b == i])
        out["median"].append(m), out["q25"].append(lo), out["q75"].append(hi)
    return out


def call(cells, lo, hi):
    """cells: {site: [r per eligible bin]}. Helps / no gain / mixed per amendment A1."""
    allv = np.concatenate([np.asarray(v, dtype=float) for v in cells.values()])
    if np.all(allv < lo):
        return "helps"
    if all(np.sum(np.asarray(v) >= hi) > len(v) / 2 for v in cells.values()):
        return "no gain"
    return "mixed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--suffix", default="", help="probe variant, e.g. _hofit, _a100, _zv20001")
    ap.add_argument("--untrained-run", help="run folder holding the untrained probes (default: --run)")
    ap.add_argument("--metric", choices=["all", "ho"], default="all",
                    help="ho: per-latent MSE over probe-validation rows held out from network training")
    args = ap.parse_args()
    tree, run = Path(args.tree), Path(args.run)
    urun = Path(args.untrained_run) if args.untrained_run else run
    summ, base = C.load_json(tree / "data_summary.json"), C.load_json(tree / "baselines.json")
    met = C.load_json(run / "metrics.json")
    sx = args.suffix
    P = {"raw": np.load(tree / f"probes_raw{sx}.npz"), "trained": np.load(run / f"probes_trained{sx}.npz"),
         "untrained": np.load(urun / f"probes_untrained{sx}.npz")}
    for k in ["trained", "untrained"]:
        assert np.array_equal(P[k]["latent"], P["raw"]["latent"]) and np.array_equal(P[k]["depth"], P["raw"]["depth"])
        assert np.allclose(P[k]["baseline_mse"], P["raw"]["baseline_mse"])
    ent = P["raw"]
    fom = ent["size"] / np.sqrt(C.N_POINTS)
    kept = ent["baseline_mse"] >= C.MIN_BASELINE_MSE
    key = "mse_ho_" if args.metric == "ho" else "mse_"
    raw = ent[f"{key}raw"]
    mse = {c: {s: P[c][f"{key}{s}"] for s in C.SITES} for c in ["trained", "untrained"]}
    finite = np.isfinite(raw) & np.all([np.isfinite(mse[c][s]) for c in mse for s in C.SITES], axis=0)
    res = {"variant": {"suffix": sx, "metric": args.metric, "untrained_run": urun.name},
           "entries": int(len(fom)), "kept": int(kept.sum()), "kept_and_scored": int((kept & finite).sum()),
           "fom_range_kept": [float(fom[kept].min()), float(fom[kept].max())]}

    for nb in (5, 8):
        b, edges = fom_bins(fom, kept, nb)
        centers = np.sqrt(edges[:-1] * edges[1:])
        b[~finite] = -1  # entries without a held-out validation point (metric ho) leave the bins
        blk = {"edges": edges.tolist(), "centers": centers.tolist(), "count": [int((b == i).sum()) for i in range(nb)],
               "bin_of_entry": b.tolist(), "mse": {"raw": binned(raw, b, nb)}, "ratio_to_raw": {},
               "ratio_trained_untrained": {}}
        for c in ["trained", "untrained"]:
            blk["mse"][c] = {s: binned(mse[c][s], b, nb) for s in C.SITES}
            blk["ratio_to_raw"][c] = {s: binned(mse[c][s] / raw, b, nb) for s in C.SITES}
        blk["ratio_trained_untrained"] = {s: binned(mse["trained"][s] / mse["untrained"][s], b, nb) for s in C.SITES}
        med = np.array(blk["mse"]["trained"]["block2"]["median"], dtype=float)
        if np.all(np.isfinite(med)) and np.all(med > 0):
            fit = stats.linregress(np.log10(centers), np.log10(med))
            blk["block2_powerlaw_slope_trained"] = {"slope": float(fit.slope), "r2": float(fit.rvalue ** 2)}
        res[f"bins{nb}"] = blk

    b5 = np.array(res["bins5"]["bin_of_entry"])
    use = [i for i in range(5) if res["bins5"]["count"][i] >= MIN_BIN]
    res["bins_used"] = use

    # Gate C.
    r2 = met["test"]["r2"]
    res["gate_C"] = {"pass": abs(r2 - PAPER_R2["mlp"]) <= BAND, "mlp_r2": r2, "mlp_mse": met["test"]["mse"],
                     "paper_r2": PAPER_R2["mlp"], "band": BAND}
    res["gate_B"] = {"pass": base["gate_B"]["pass"], "ridge_r2": base["ridge"]["r2"], "nn1_r2": base["nn1"]["r2"],
                     "ridge_mse": base["ridge"]["mse"], "nn1_mse": base["nn1"]["mse"],
                     "paper": {"ridge_r2": 0.51, "nn1_r2": 0.94}, "band": BAND}
    paper_tree = summ["seeds"] == C.DATA_SEEDS
    res["gate_A"] = {"pass": bool(summ["row_identity_and_structure_checks_pass"]
                                  and (summ["entries_depth_0_200"] == 317 or not paper_tree)),
                     "paper_tree": paper_tree,
                     "entries": summ["entries_depth_0_200"], "entries_probed": summ["entries_probed_10_200"],
                     "latents_per_depth": summ["latents_per_probed_depth"], "mean_depth": summ["point_depth"]["mean"],
                     "max_depth": summ["point_depth"]["max"],
                     "row_identity": bool(summ["checks"]["x_row_identity"] and summ["checks"]["y_row_identity"]
                                          and summ["checks"]["ancestor_table_matches_brute_walk_1000"])}

    # Gate D on the trained network.
    rr = res["bins5"]["ratio_to_raw"]["trained"]
    below = {s: all(rr[s]["median"][i] < 1 for i in use) for s in C.HIDDEN_SITES + BLOCK_SITES}
    gap = {}
    for typ, sites in (("hidden", C.HIDDEN_SITES), ("residual", BLOCK_SITES)):
        pooled = [np.median(np.concatenate([(mse["trained"][s] / raw)[b5 == i] for s in sites])) for i in use]
        gap[typ] = {"pooled_ratio_to_raw_by_bin": [float(v) for v in pooled], "shrinks": bool(pooled[-1] > pooled[0]),
                    "spearman_vs_bin": float(stats.spearmanr(range(len(pooled)), pooled).statistic)}
    hid_below_res = {f"hidden{i}<block{i}": [res["bins5"]["mse"]["trained"][f"hidden{i}"]["median"][j]
                                              < res["bins5"]["mse"]["trained"][f"block{i}"]["median"][j] for j in use]
                     for i in range(3)}
    res["gate_D"] = {"pass": all(below.values()) and all(g["shrinks"] for g in gap.values()),
                     "ratio_below_1": below, "gap_shrinks": {k: v["shrinks"] for k, v in gap.items()}, "gap": gap,
                     "input_layer_ratio_to_raw": rr["input_layer"]["median"], "hidden_below_residual": hid_below_res,
                     "block2_slope_5_bins": res["bins5"].get("block2_powerlaw_slope_trained"),
                     "block2_slope_8_bins": res["bins8"].get("block2_powerlaw_slope_trained"),
                     "paper_block2_slope_fig4a": -0.64}

    # Decision rule (amendment A1), per site type, with sensitivity.
    ratio = {s: mse["trained"][s] / mse["untrained"][s] for s in C.SITES}
    cells = {s: [float(np.median(ratio[s][b5 == i])) for i in use] for s in C.SITES}
    types = {"hidden": C.HIDDEN_SITES, "residual": C.RESIDUAL_SITES}
    dec = {"label": "one declared run", "bins_used": use, "cells": cells,
           "call": {t: call({s: cells[s] for s in ss}, 0.85, 0.95) for t, ss in types.items()}}
    sens = {}
    for name, (lo, hi) in {"-0.05": (0.80, 0.90), "+0.05": (0.90, 1.00), "lo-0.05": (0.80, 0.95),
                           "lo+0.05": (0.90, 0.95), "hi-0.05": (0.85, 0.90), "hi+0.05": (0.85, 1.00)}.items():
        sens[name] = {t: call({s: cells[s] for s in ss}, lo, hi) for t, ss in types.items()}
    sens["residual_without_input_layer"] = call({s: cells[s] for s in BLOCK_SITES}, 0.85, 0.95)
    pooled = {}
    for t, ss in types.items():
        pv = [float(np.median(np.concatenate([ratio[s][b5 == i] for s in ss]))) for i in use]
        pooled[t] = {"cells": pv, "call": call({"pooled": pv}, 0.85, 0.95)}
    sens["pooled"] = pooled
    dec["sensitivity"] = sens
    # Absolute errors beside the ratios: bin medians of per-entry MSE, trained and untrained.
    dec["absolute_bin_medians"] = {c: {s: [res["bins5"]["mse"][c][s]["median"][i] for i in use] for s in C.SITES}
                                   for c in ["trained", "untrained"]}
    dec["absolute_bin_medians"]["raw"] = [res["bins5"]["mse"]["raw"]["median"][i] for i in use]
    dec["share_of_entries_trained_below_untrained"] = {s: float(np.mean(ratio[s][kept & finite] < 1)) for s in C.SITES}
    res["decision"] = dec

    # Fig. 5 view: global probe-validation MSE per depth.
    res["depth_curves"] = {"depth": P["raw"]["g_depth"].tolist(), "baseline_mse": P["raw"]["g_mean_baseline_mse"].tolist(),
                           "raw": P["raw"]["g_mse_raw"].tolist(),
                           **{c: {s: P[c][f"g_mse_{s}"].tolist() for s in C.SITES} for c in ["trained", "untrained"]}}
    out_name = f"analysis{sx}{'_ho' if args.metric == 'ho' else ''}.json"
    C.save_json(run / out_name, res)
    print("gate A", res["gate_A"]["pass"], "| gate B", res["gate_B"]["pass"], "| gate C", res["gate_C"]["pass"],
          f"(test R2 {r2:.4f})", "| gate D", res["gate_D"]["pass"])
    print("bins5 counts", res["bins5"]["count"], "used", use)
    for s in C.SITES:
        print(f"  {s:12s} r(trained/untrained) by bin: " + " ".join(f"{v:.3f}" for v in cells[s]))
    print("call", dec["call"], "| sensitivity", {k: v for k, v in sens.items() if k != "pooled"},
          "| pooled", {t: v["call"] for t, v in pooled.items()})


if __name__ == "__main__":
    main()
