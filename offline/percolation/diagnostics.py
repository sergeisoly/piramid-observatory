"""Phase-2 summary over the runs written by run_phase2.py.

The decision across runs (amendment A3), the held-out and ridge-strength variants of the declared run, the value-swap
diagnostic, the distance of the residual stream from an affine map of x, and a check that the declared run's phase-1
probe numbers reproduce. Writes <root>/phase2_summary.json, whose "export" block export.py merges into the presentation’s data file.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

import common as C
from analyze import call, fom_bins
from export import q, rnd, sig, sigs
from probe import activations
from train import ResidualMLP

RUNS = [("tree0", "tree0-init0", "declared"), ("tree0", "tree0-init1", "repeat: initialization 1"),
        ("tree0", "tree0-init2", "repeat: initialization 2"), ("tree1", "tree1-init0", "repeat: tree 1"),
        ("tree2", "tree2-init0", "repeat: tree 2")]
TYPES = {"hidden": C.HIDDEN_SITES, "residual": C.RESIDUAL_SITES}
SENS = {"-0.05": (0.80, 0.90), "declared": (0.85, 0.95), "+0.05": (0.90, 1.00)}
BLOCKS = ["block0", "block1", "block2"]
VSITES = C.HIDDEN_SITES + BLOCKS + ["output"]
VSEED = "v20001"


def calls_of(cells):
    return {name: {t: call({s: cells[s] for s in ss}, lo, hi) for t, ss in TYPES.items()} for name, (lo, hi) in SENS.items()}


def across(per_run):
    out = {}
    for name in SENS:
        out[name] = {}
        for t in TYPES:
            cs = [c[name][t] for c in per_run.values()]
            out[name][t] = "helps" if all(c == "helps" for c in cs) else "no gain" if all(c == "no gain" for c in cs) \
                else "mixed"
    return out


def binned_median_ratio(num, den, b, use):
    r = num / den
    return [float(np.nanmedian(r[b == i])) if np.isfinite(r[b == i]).any() else None for i in use]


def value_swap(root):
    t0, r0, rv = root / "tree0", root / "tree0-init0", root / "tree0-vswap"
    files = {"z": {"raw": t0 / "probes_raw.npz", "untrained": r0 / "probes_untrained.npz",
                   "y_net": r0 / "probes_trained.npz", "yswap_net": rv / "probes_trained.npz"},
             "zswap": {"raw": t0 / f"probes_raw_z{VSEED}.npz", "untrained": r0 / f"probes_untrained_z{VSEED}.npz",
                       "y_net": r0 / f"probes_trained_z{VSEED}.npz", "yswap_net": rv / f"probes_trained_z{VSEED}.npz"}}
    res = {"values": C.load_json(t0 / f"values_{VSEED}.json"),
           "yswap_net_test": C.load_json(rv / "metrics.json")["test"],
           "same_initial_weights_as_declared": all(
               torch.equal(a, b) for a, b in zip(torch.load(r0 / "init.pt", map_location="cpu", weights_only=True).values(), torch.load(rv / "init.pt", map_location="cpu", weights_only=True).values()))}
    for target, f in files.items():
        P = {k: np.load(v) for k, v in f.items()}
        kept = P["raw"]["baseline_mse"] >= C.MIN_BASELINE_MSE
        b, edges = fom_bins(P["raw"]["size"] / np.sqrt(C.N_POINTS), kept, 5)
        use = [i for i in range(5) if (b == i).sum() >= 10]
        own, other = ("y_net", "yswap_net") if target == "z" else ("yswap_net", "y_net")
        blk = {"bins_used": use, "count": [int((b == i).sum()) for i in range(5)], "own_net": own,
               "raw_median": [float(np.median(P["raw"]["mse_raw"][b == i])) for i in use], "median": {}, "ratio": {}}
        for net in ("untrained", "y_net", "yswap_net"):
            blk["median"][net] = {s: [float(np.median(P[net][f"mse_{s}"][b == i])) for i in use] for s in VSITES}
        for s in VSITES:
            blk["ratio"][s] = {
                "other_net_over_own_net": binned_median_ratio(P[other][f"mse_{s}"], P[own][f"mse_{s}"], b, use),
                "own_net_over_untrained": binned_median_ratio(P[own][f"mse_{s}"], P["untrained"][f"mse_{s}"], b, use),
                "other_net_over_untrained": binned_median_ratio(P[other][f"mse_{s}"], P["untrained"][f"mse_{s}"], b, use)}
        blk["output_only_over_raw"] = {net: binned_median_ratio(P[net]["mse_output"], P["raw"]["mse_raw"], b, use)
                                       for net in ("untrained", "y_net", "yswap_net")}
        res[target] = blk
    # The pre-registered rule applied to the y'-network on its own values (diagnostic, not in the run set).
    P = {k: np.load(v) for k, v in files["zswap"].items()}
    kept = P["raw"]["baseline_mse"] >= C.MIN_BASELINE_MSE
    b, _ = fom_bins(P["raw"]["size"] / np.sqrt(C.N_POINTS), kept, 5)
    use = [i for i in range(5) if (b == i).sum() >= 10]
    cells = {s: binned_median_ratio(P["yswap_net"][f"mse_{s}"], P["untrained"][f"mse_{s}"], b, use) for s in C.SITES}
    res["yswap_net_call_on_zswap"] = {"cells": cells, "calls": calls_of(cells)}
    return res


def residual_distance(root):
    """Share of each site's activation variance that no affine map of x explains, over all points."""
    data = np.load(root / "tree0" / "dataset.npz")
    X = data["X"].astype(np.float32)
    dev = C.pick_device()
    rdev = C.pick_device(float64=True)
    Xd = torch.from_numpy(data["X"]).to(rdev)
    Xd = torch.cat([Xd, torch.ones(len(Xd), 1, dtype=torch.float64, device=rdev)], 1)
    G = Xd.T @ Xd
    out = {}
    for state, name in (("init", "untrained"), ("final", "trained")):
        model = ResidualMLP()
        model.load_state_dict(torch.load(root / "tree0-init0" / f"{state}.pt", map_location="cpu", weights_only=True))
        model.to(dev).eval()
        out[name] = {}
        for site in C.RESIDUAL_SITES + C.HIDDEN_SITES:
            A = torch.from_numpy(activations(model, X, site, dev)).to(rdev).double()
            B = torch.linalg.solve(G, Xd.T @ A)
            resid = ((A - Xd @ B) ** 2).sum()
            total = ((A - A.mean(0)) ** 2).sum()
            out[name][site] = float(resid / total)
            del A
    return out


def phase1_check(root):
    out = {}
    for f in ["tree0/probes_raw.npz", "tree0-init0/probes_untrained.npz", "tree0-init0/probes_trained.npz"]:
        old, new = root / "phase1_backup" / f, root / f
        if not old.exists():
            continue
        a, b = np.load(old), np.load(new)
        sites = [s for s in a["sites"]]
        out[f] = float(max(np.max(np.abs(b[f"mse_{s}"] / a[f"mse_{s}"] - 1)) for s in sites))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    S = {"runs": {}, "export": {"runs": [], "trees": {}, "baselines": {}, "networks": {}, "binned": {}, "gates": {},
                                "decision": {}}}
    per_run = {}
    for tree_id, rid, role in RUNS:
        an = C.load_json(root / rid / "analysis.json")
        met = C.load_json(root / rid / "metrics.json")
        base = C.load_json(root / tree_id / "baselines.json")
        cells = an["decision"]["cells"]
        per_run[rid] = calls_of(cells)
        S["runs"][rid] = {"role": role, "tree": tree_id, "seed": met["seed"], "test_r2": met["test"]["r2"],
                          "ridge_r2": base["ridge"]["r2"], "nn1_r2": base["nn1"]["r2"], "gates": {
                              g: an[f"gate_{g}"]["pass"] for g in "ABCD"}, "entries_kept": an["kept"],
                          "bins5_count": an["bins5"]["count"], "cells": cells, "calls": per_run[rid],
                          "share_trained_below_untrained": an["decision"]["share_of_entries_trained_below_untrained"],
                          "block2_slope_5_bins": an["gate_D"]["block2_slope_5_bins"]}
        if rid == "tree0-init0":
            continue
        S["export"]["runs"].append({"id": rid, "role": role, "tree": tree_id, "init": met["seed"] - C.TRAIN_SEED,
                                    "target": "y", "status": "complete"})
        h = met["history"]
        S["export"]["networks"][rid] = {"test": {"r2": round(met["test"]["r2"], 4), "mse": sig(met["test"]["mse"])},
                                        "curve": {"epoch": h["epoch"][9::10], "train_loss": sigs(h["train_loss"][9::10]),
                                                  "val_r2": rnd(h["val_r2"][9::10], 4)}}
        b5 = an["bins5"]
        S["export"]["binned"][rid] = {"bins5": {
            "edges": sigs(b5["edges"]), "centers": sigs(b5["centers"]), "count": b5["count"],
            "mse": {"raw": q(b5["mse"]["raw"]),
                    **{c: {s: {"median": sigs([np.nan if v is None else v for v in b5["mse"][c][s]["median"]])}
                           for s in C.SITES} for c in ("trained", "untrained")}},
            "ratio_trained_untrained": {s: q(b5["ratio_trained_untrained"][s]) for s in C.SITES}}}
        S["export"]["gates"][rid] = {g: {"pass": an[f"gate_{g}"]["pass"]} for g in "ABCD"}
        S["export"]["gates"][rid]["C"]["mlp_r2"] = round(met["test"]["r2"], 4)
        S["export"]["decision"][rid] = {"label": "one run", "bins_used": an["decision"]["bins_used"],
                                        "cells": {s: sigs(v) for s, v in cells.items()}, "call": per_run[rid]["declared"],
                                        "sensitivity": {k: per_run[rid][k] for k in ("-0.05", "+0.05")}}
        if tree_id != "tree0" and tree_id not in S["export"]["trees"]:
            summ = C.load_json(root / tree_id / "data_summary.json")
            el = summ["eligible"]
            S["export"]["trees"][tree_id] = {
                "seeds": summ["seeds"], "latents": summ["latents"],
                "point_depth": {k: summ["point_depth"][k] for k in ("mean", "median", "max")},
                "eligible_by_depth": {"depth": [r["depth"] for r in el], "points": [r["points"] for r in el],
                                      "latents": [r["latents"] for r in el], "largest": [r["max_size"] for r in el]},
                "entries_probed": summ["entries_probed_10_200"],
                "largest_share": rnd(summ["largest_latent_share_of_eligible"], 4)}
            S["export"]["baselines"][tree_id] = {"ridge": {"r2": round(base["ridge"]["r2"], 4), "mse": sig(base["ridge"]["mse"])},
                                                 "nn1": {"r2": round(base["nn1"]["r2"], 4), "mse": sig(base["nn1"]["mse"])}}
    S["across_runs"] = {"runs": [r for _, r, _ in RUNS], "call": across(per_run)}
    vmet = C.load_json(root / "tree0-vswap" / "metrics.json")
    S["export"]["runs"].append({"id": "tree0-vswap", "role": "diagnostic: value swap", "tree": "tree0", "init": 0,
                                "target": f"y' (value seed {VSEED[1:]})", "status": "complete"})
    S["export"]["networks"]["tree0-vswap"] = {"test": {"r2": round(vmet["test"]["r2"], 4), "mse": sig(vmet["test"]["mse"])}}

    variants = {}
    for name, fname in (("heldout_scoring", "analysis_ho.json"), ("heldout_fit", "analysis_hofit.json"),
                        ("alpha_100", "analysis_a100.json"), ("alpha_10000", "analysis_a10000.json")):
        an = C.load_json(root / "tree0-init0" / fname)
        cells = an["decision"]["cells"]
        variants[name] = {"bins_used": an["decision"]["bins_used"], "bins5_count": an["bins5"]["count"],
                          "kept_and_scored": an["kept_and_scored"], "cells": cells, "calls": calls_of(cells),
                          "absolute_bin_medians": an["decision"]["absolute_bin_medians"],
                          "share_trained_below_untrained": an["decision"]["share_of_entries_trained_below_untrained"],
                          "gate_D": {k: an["gate_D"][k] for k in ("pass", "ratio_below_1", "gap_shrinks")}}
    S["variants"] = variants
    S["value_swap"] = value_swap(root)
    S["residual_distance"] = residual_distance(root)
    S["phase1_reproduced_max_rel_diff"] = phase1_check(root)

    rnd_cells = lambda cells: {s: sigs(v) for s, v in cells.items()}
    vs = S["value_swap"]
    S["export"]["decision_across_runs"] = {"runs": S["across_runs"]["runs"], "call": S["across_runs"]["call"]["declared"],
                                           "sensitivity": {k: S["across_runs"]["call"][k] for k in ("-0.05", "+0.05")}}
    S["export"]["diagnostics"] = {
        "variants": {k: {"cells": rnd_cells(v["cells"]), "call": v["calls"]["declared"], "bins5_count": v["bins5_count"]}
                     for k, v in variants.items()},
        "value_swap": {
            "corr_latent_values": round(vs["values"]["corr_latent_values"], 4),
            "yswap_net_test_r2": round(vs["yswap_net_test"]["r2"], 4),
            **{t: {"bins_used": vs[t]["bins_used"], "own_net": vs[t]["own_net"], "raw_median": sigs(vs[t]["raw_median"]),
                   "median": {n: {s: sigs(v) for s, v in d.items()} for n, d in vs[t]["median"].items()},
                   "other_net_over_own_net": {s: sigs(vs[t]["ratio"][s]["other_net_over_own_net"]) for s in VSITES},
                   "output_only_over_raw": {n: sigs(v) for n, v in vs[t]["output_only_over_raw"].items()}}
               for t in ("z", "zswap")},
            "yswap_net_call_on_zswap": vs["yswap_net_call_on_zswap"]["calls"]["declared"]},
        "residual_distance": {k: {s: sig(v) for s, v in d.items()} for k, d in S["residual_distance"].items()}}
    C.save_json(root / "phase2_summary.json", S)
    print("across runs:", S["across_runs"]["call"])
    for rid, r in S["runs"].items():
        print(f"  {rid}: test R2 {r['test_r2']:.4f}, gates {r['gates']}, calls {r['calls']['declared']}")
    for k, v in variants.items():
        print(f"  variant {k}: {v['calls']['declared']} (bins {v['bins5_count']})")
    print("  value swap: y'-net call on z'", vs["yswap_net_call_on_zswap"]["calls"]["declared"])
    print("  residual distance:", S["residual_distance"])
    print("  phase-1 reproduced:", S["phase1_reproduced_max_rel_diff"])


if __name__ == "__main__":
    main()
