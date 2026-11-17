"""Linear probes for the depth-k ancestor's value z (paper Sec. 5), on the raw input or one network state.

Per depth k in 10, 15, ..., 200: rows are all points with depth > k whose depth-k ancestor owns >= 150 points, in
ascending row order, permuted by numpy default_rng(42); the last 30 % are probe-validation rows. Per site: features
standardized with the probe-train rows' mean and std (ddof 1, + 1e-6), ridge alpha = 1 with intercept, solved
exactly in float64 (on CUDA when available, else on the CPU). Score: per-latent MSE on probe-validation rows. Sites
are processed one at a time.

Writes probes_<tag><suffix>.{npz,csv,json} (tag = raw | trained | untrained; the suffix names a variant: _z<seed>
for a value-swapped target, _hofit for probes fit on network-held-out points only, _a<alpha> for another ridge
strength). Every file also holds per-latent MSE over the probe-validation rows held out from network training
(mse_ho_*). Network states add the output-only probe (site "output", the network's scalar prediction).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from scipy import linalg

import common as C
from train import ResidualMLP

CHUNK = 32_768


def probe_tasks(data, lat_z, heldout_only=False):
    """Rows, targets and the probe split per depth. Entries (latents) always come from the full eligible set, so every
    variant is paired with the declared protocol; heldout_only keeps only points held out from network training."""
    anc, depths, split = data["anc"], list(data["anc_depths"]), data["split"]
    lat_size = data["lat_size"]
    tasks = []
    for k in C.PROBE_DEPTHS:
        a = anc[:, depths.index(k)]
        ok = (a >= 0) & (lat_size[np.maximum(a, 0)] >= C.MIN_LATENT_POINTS)
        rows = np.nonzero(ok)[0]
        latents = np.unique(a[rows])
        if heldout_only:
            rows = rows[split[rows] != 0]
        perm = np.random.default_rng(C.PROBE_SEED).permutation(len(rows))
        rows = rows[perm]
        lid = a[rows]
        nv = int(C.PROBE_VAL_FRAC * len(rows))
        ntr = len(rows) - nv
        tasks.append(dict(depth=k, rows=rows, tr=rows[:ntr], va=rows[ntr:], ytr=lat_z[lid[:ntr]],
                          yva=lat_z[lid[ntr:]], lva=lid[ntr:], latents=latents, va_ho=split[rows[ntr:]] != 0))
    return tasks


def per_latent(err, lva, latents, mask=None):
    idx = np.searchsorted(latents, lva)
    if mask is not None:
        idx, err = idx[mask], err[mask]
    sums = np.bincount(idx, weights=err, minlength=len(latents))
    cnt = np.bincount(idx, minlength=len(latents))
    return np.divide(sums, cnt, out=np.full(len(latents), np.nan), where=cnt > 0), cnt


def fit_ridge(A, tr, ytr, alpha=C.RIDGE_ALPHA):
    """Ridge on standardized features, exactly as Ridge(alpha) after the authors' z-scoring, in float64.

    A is a float32 torch tensor on the float64 device (CUDA, else CPU); tr holds row indices.
    """
    dev, p = A.device, A.shape[1]
    tr = torch.from_numpy(tr).to(dev)
    mean = torch.zeros(p, dtype=torch.float64, device=dev)
    for s in range(0, len(tr), CHUNK):
        mean += A[tr[s:s + CHUNK]].double().sum(0)
    mean /= len(tr)
    G = torch.zeros((p, p), dtype=torch.float64, device=dev)
    ss = torch.zeros(p, dtype=torch.float64, device=dev)
    b = torch.zeros(p, dtype=torch.float64, device=dev)
    yc = torch.from_numpy(ytr.astype(np.float64)).to(dev)
    yc -= yc.mean()
    for s in range(0, len(tr), CHUNK):
        Z = A[tr[s:s + CHUNK]].double() - mean
        G += Z.T @ Z
        ss += (Z * Z).sum(0)
        b += Z.T @ yc[s:s + CHUNK]
    sd = torch.sqrt(ss / (len(tr) - 1)) + 1e-6
    G, b, sdn = G.cpu().numpy(), b.cpu().numpy(), sd.cpu().numpy()
    w = linalg.solve(G / np.outer(sdn, sdn) + alpha * np.eye(p), b / sdn, assume_a="pos")
    return mean, sd, torch.from_numpy(w).to(dev), float(ytr.astype(np.float64).mean())


def predict(A, rows, fit):
    mean, sd, w, ybar = fit
    rows = torch.from_numpy(rows).to(A.device)
    out = torch.empty(len(rows), dtype=torch.float64, device=A.device)
    for s in range(0, len(rows), CHUNK):
        out[s:s + CHUNK] = ((A[rows[s:s + CHUNK]].double() - mean) / sd) @ w + ybar
    return out.cpu().numpy()


def activations(model, X, site, dev, chunk=20_000):
    """One site's activations for all points, float32 on the CPU."""
    out = None
    with torch.no_grad():
        for s in range(0, len(X), chunk):
            acts = {}
            model(torch.from_numpy(X[s:s + chunk]).to(dev), sites=acts)
            a = acts[site].float().cpu().numpy()
            if out is None:
                out = np.empty((len(X), a.shape[1]), np.float32)
            out[s:s + chunk] = a
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--run", help="run folder with init.pt and final.pt (not needed for --state raw)")
    ap.add_argument("--state", choices=["raw", "init", "final"], required=True)
    ap.add_argument("--values", help="npz from generate.py --value-seed: probe its latent values (z') instead of z")
    ap.add_argument("--fit-rows", choices=["all", "heldout"], default="all",
                    help="heldout: fit and score only on points held out from network training")
    ap.add_argument("--alpha", type=float, default=C.RIDGE_ALPHA)
    ap.add_argument("--sklearn-check", help="SITE:DEPTH, refit that probe with sklearn Ridge on float32 input")
    args = ap.parse_args()
    tree = Path(args.tree)
    tag = {"raw": "raw", "init": "untrained", "final": "trained"}[args.state]
    suffix = (f"_z{Path(args.values).stem.split('_')[-1]}" if args.values else "") + \
             ("_hofit" if args.fit_rows == "heldout" else "") + (f"_a{args.alpha:g}" if args.alpha != C.RIDGE_ALPHA else "")
    out = tree if args.state == "raw" else Path(args.run)
    T = C.Timer()
    data = np.load(tree / "dataset.npz")
    X = data["X"].astype(np.float32)
    lat_size = data["lat_size"]
    lat_z = (np.load(args.values)["lat_z"] if args.values else data["lat_z"]).astype(np.float32)
    tasks = probe_tasks(data, lat_z, heldout_only=args.fit_rows == "heldout")
    n_lat = [len(t["latents"]) for t in tasks]
    map_depth = C.PROBE_DEPTHS[int(np.argmax(n_lat))]  # first maximum = shallower depth on ties
    T.mark("tasks")

    if args.state == "raw":
        sites, model, dev = ["raw"], None, None
    else:
        sites = C.SITES + ["output"]
        dev = C.pick_device()
        model = ResidualMLP()
        model.load_state_dict(torch.load(Path(args.run) / f"{args.state}.pt", map_location="cpu", weights_only=True))
        model.to(dev).eval()

    # Entry table (one row per probed latent), shared by every site.
    ent_depth = np.concatenate([[t["depth"]] * len(t["latents"]) for t in tasks]).astype(int)
    ent_lat = np.concatenate([t["latents"] for t in tasks])
    n_val, n_val_ho = np.zeros(len(ent_lat), np.int64), np.zeros(len(ent_lat), np.int64)
    base = np.zeros(len(ent_lat))
    mse = {s: np.zeros(len(ent_lat)) for s in sites}
    mse_ho = {s: np.zeros(len(ent_lat)) for s in sites}
    glob = {s: {"mse": [], "r2": []} for s in sites}
    mean_base, map_pred, unit_stats = [], {}, {}
    rdev = C.pick_device(float64=True)
    for site in sites:
        A_np = X if site == "raw" else activations(model, X, site, dev)
        A = torch.from_numpy(A_np).to(rdev)
        alive = (A_np != 0).mean(0)
        unit_stats[site] = {"units": A.shape[1], "never_active": int((alive == 0).sum()),
                            "active_below_0.1pct": int(((alive > 0) & (alive < 1e-3)).sum())}
        off = 0
        for t in tasks:
            fit = fit_ridge(A, t["tr"], t["ytr"], alpha=args.alpha)
            pv = predict(A, t["va"], fit)
            err = (pv - t["yva"]) ** 2
            glob[site]["mse"].append(float(err.mean()))
            var = float(np.var(t["yva"].astype(np.float64)))  # 0 when one latent covers the depth (constant target)
            glob[site]["r2"].append(1 - float(err.mean()) / var if var > 0 else float("nan"))
            sl = slice(off, off + len(t["latents"]))
            mse[site][sl], cnt = per_latent(err, t["lva"], t["latents"])
            mse_ho[site][sl], cnt_ho = per_latent(err, t["lva"], t["latents"], t["va_ho"])
            if site == sites[0]:
                n_val[sl], n_val_ho[sl] = cnt, cnt_ho
                base[sl] = (np.float32(t["ytr"].mean()) - lat_z[t["latents"]]).astype(np.float64) ** 2
                mean_base.append(float(np.mean((t["yva"] - t["ytr"].mean()) ** 2)))
            if t["depth"] == map_depth:
                map_pred[site] = predict(A, t["rows"], fit).astype(np.float32)
            if args.sklearn_check and args.sklearn_check == f"{site}:{t['depth']}":
                from sklearn.linear_model import Ridge
                Atr = A_np[t["tr"]]
                mu, sd = Atr.mean(0, keepdims=True), Atr.std(0, ddof=1, keepdims=True) + 1e-6
                pk = Ridge(alpha=args.alpha).fit((Atr - mu) / sd, t["ytr"]).predict((A_np[t["va"]] - mu) / sd)
                ek, _ = per_latent((pk - t["yva"]) ** 2, t["lva"], t["latents"])
                unit_stats["sklearn_check"] = {
                    "site": site, "depth": t["depth"],
                    "max_rel_diff_per_latent_mse": float(np.nanmax(np.abs(ek / mse[site][sl] - 1))),
                    "max_abs_diff_prediction": float(np.max(np.abs(pk - pv)))}
            off += len(t["latents"])
        del A, A_np
        T.mark(site)
        print(f"{tag}{suffix} {site}: median per-latent MSE {np.nanmedian(mse[site]):.4g} ({T.marks[site]} s)", flush=True)

    name = f"probes_{tag}{suffix}"
    map_task = tasks[C.PROBE_DEPTHS.index(map_depth)]
    np.savez(out / f"{name}.npz", depth=ent_depth, latent=ent_lat, size=lat_size[ent_lat], z=lat_z[ent_lat],
             n_val=n_val, n_val_ho=n_val_ho, baseline_mse=base, sites=np.array(sites), alpha=args.alpha,
             **{f"mse_{s}": mse[s] for s in sites}, **{f"mse_ho_{s}": mse_ho[s] for s in sites},
             g_depth=np.array(C.PROBE_DEPTHS), g_points=np.array([len(t["rows"]) for t in tasks]),
             g_latents=np.array(n_lat), g_mean_baseline_mse=np.array(mean_base),
             **{f"g_mse_{s}": np.array(glob[s]["mse"]) for s in sites},
             **{f"g_r2_{s}": np.array(glob[s]["r2"]) for s in sites},
             map_depth=map_depth, map_rows=map_task["rows"], map_ntr=len(map_task["tr"]),
             **{f"map_pred_{s}": map_pred[s] for s in sites})
    header = "depth,latent,size,fom,z,n_val,n_val_heldout,baseline_mse," + ",".join(f"mse_{s}" for s in sites)
    cols = [ent_depth, ent_lat, lat_size[ent_lat], lat_size[ent_lat] / np.sqrt(C.N_POINTS), lat_z[ent_lat], n_val,
            n_val_ho, base] + [mse[s] for s in sites]
    np.savetxt(out / f"{name}.csv", np.column_stack(cols), delimiter=",", header=header, comments="",
               fmt=["%d", "%d", "%d", "%.6g", "%.6g", "%d", "%d"] + ["%.6g"] * (1 + len(sites)))
    T.mark("total")
    C.save_json(out / f"{name}.json", {"tag": tag, "suffix": suffix, "values": args.values, "fit_rows": args.fit_rows,
                                      "alpha": args.alpha, "entries": len(ent_lat), "map_depth": map_depth,
                                      "units": unit_stats, "ridge_device": rdev.type,
                                      "cost": {"seconds": T.marks, "peak_rss_mb": round(C.peak_rss_mb()),
                                               "gpu_peak_mb": C.gpu_peak_mb(rdev)},
                                      "pipeline": C.pipeline_revision(), "environment": C.environment()})
    print(f"{name}: {len(ent_lat)} entries, map depth {map_depth}, total {T.marks['total']} s, "
          f"peak RSS {C.peak_rss_mb():.0f} MB", flush=True)


if __name__ == "__main__":
    main()
