"""Table 1 baselines: ridge regression on x and 1-nearest-neighbor over the full dataset.

Ridge: sklearn Ridge(alpha=1) with intercept on raw x, fit on the network's training split, scored on its test split.
1NN: for each test point, the y of the nearest other point among all 200,000 (the point itself excluded), as in the
generator's test_percolation_dataset.py. Candidates come from a float32 search on the GPU (CUDA or MPS) and are
re-ranked in float64.
Writes <tree>/baselines.json.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit

import common as C

K_CAND = 16


def nn_candidates(X, queries, device):
    """Indices of the K_CAND nearest points (float32 squared distances) for each query row."""
    Xd = torch.from_numpy(X.astype(np.float32)).to(device)
    sq = (Xd * Xd).sum(1)
    out = np.empty((len(queries), K_CAND), np.int64)
    chunk = 2048 if device.type == "cuda" else 512
    for s in range(0, len(queries), chunk):
        q = Xd[torch.from_numpy(queries[s:s + chunk]).to(device)]
        d2 = sq[None, :] - 2.0 * (q @ Xd.T) + (q * q).sum(1, keepdim=True)
        out[s:s + chunk] = torch.topk(d2, K_CAND, largest=False).indices.cpu().numpy()
    return out


def nn_predict(X, y, queries, cand, chunk=8192):
    """Exact float64 re-ranking in chunks of queries."""
    parts = [nn_predict_chunk(X, y, queries[s:s + chunk], cand[s:s + chunk]) for s in range(0, len(queries), chunk)]
    return tuple(np.concatenate(p) for p in zip(*parts))


def nn_predict_chunk(X, y, queries, cand):
    """Nearest other point, its distance, the tie-averaged squared error, the tie count, and a candidate-set check."""
    diff = X[cand] - X[queries][:, None, :]
    d = np.sqrt((diff * diff).sum(-1))
    d[cand == queries[:, None]] = np.inf
    order = np.argsort(d, axis=1, kind="stable")
    best = np.take_along_axis(cand, order[:, :1], 1)[:, 0]
    dmin = np.take_along_axis(d, order[:, :1], 1)
    tied = d <= dmin * (1 + 1e-9)
    sq_err = (y[cand] - y[queries][:, None]) ** 2
    tie_avg = (sq_err * tied).sum(1) / tied.sum(1)
    # The true nearest point lies among the candidates when the farthest candidate is clearly farther.
    safe = np.sort(d, axis=1)[:, -2] > dmin[:, 0] * 1.01
    return best, dmin[:, 0], tie_avg, tied.sum(1), safe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True, help="folder written by generate.py")
    args = ap.parse_args()
    tree = Path(args.tree)
    T = C.Timer()
    data = np.load(tree / "dataset.npz")
    X, y, split = data["X"], data["y"], data["split"]
    tr, te = np.nonzero(split == 0)[0], np.nonzero(split == 2)[0]
    res = {"paper_table1_one_cluster": {"ridge": {"r2": 0.51, "mse": 0.20}, "nn1": {"r2": 0.94, "mse": 0.024},
                                        "mlp": {"r2": 0.94, "mse": 0.026}}}

    ridge = Ridge(alpha=C.RIDGE_ALPHA).fit(X[tr], y[tr])
    res["ridge"] = C.r2_mse(y[te], ridge.predict(X[te]))
    tr2, te2 = next(ShuffleSplit(n_splits=1, test_size=0.1, random_state=42).split(X))
    res["ridge_generator_test_split"] = C.r2_mse(y[te2], Ridge(alpha=C.RIDGE_ALPHA).fit(X[tr2], y[tr2]).predict(X[te2]))
    T.mark("ridge")

    device = C.pick_device()
    cand = nn_candidates(X, te, device)
    best, dmin, tie_avg, n_tied, safe = nn_predict(X, y, te, cand)
    pred = y[best]
    res["nn1"] = C.r2_mse(y[te], pred)
    res["nn1"]["mse_tie_averaged"] = float(tie_avg.mean())
    res["nn1"]["r2_tie_averaged"] = 1 - float(tie_avg.mean()) / float(np.var(y[te]))
    res["nn1"]["queries_with_tied_nearest"] = int((n_tied > 1).sum())
    res["nn1"]["queries_with_unsafe_candidate_set"] = int((~safe).sum())
    T.mark("nn1_test")
    allq = np.arange(len(X))
    cand_all = nn_candidates(X, allq, device)
    best_all, _, tie_all, _, safe_all = nn_predict(X, y, allq, cand_all)
    res["nn1_all_points_as_queries"] = C.r2_mse(y, y[best_all])
    res["nn1_all_points_as_queries"]["mse_tie_averaged"] = float(tie_all.mean())
    res["nn1_all_points_as_queries"]["queries_with_unsafe_candidate_set"] = int((~safe_all).sum())
    T.mark("nn1_all")
    # sklearn cross-check on 2,000 test queries (brute force, float64): same nearest distance, same MSE up to ties.
    from sklearn.neighbors import NearestNeighbors
    m = 2000
    sub = te[:m]
    idx = NearestNeighbors(n_neighbors=2, algorithm="brute").fit(X).kneighbors(X[sub], return_distance=False)
    other = np.where(idx[:, 0] == sub, idx[:, 1], idx[:, 0])
    d_sk = np.linalg.norm(X[other] - X[sub], axis=1)
    res["nn1_sklearn_check_2000"] = dict(
        same_point=int((other == best[:m]).sum()), max_abs_distance_gap=float(np.abs(d_sk - dmin[:m]).max()),
        mse_sklearn=float(np.mean((y[other] - y[sub]) ** 2)), mse_ours=float(np.mean((y[best[:m]] - y[sub]) ** 2)))
    T.mark("total")
    res["y_test_var"] = float(np.var(y[te]))
    res["gate_B"] = {"ridge_r2_within_0.03": abs(res["ridge"]["r2"] - 0.51) <= 0.03,
                     "nn1_r2_within_0.03": abs(res["nn1"]["r2"] - 0.94) <= 0.03}
    res["gate_B"]["pass"] = all(res["gate_B"].values())
    res["cost"] = {"seconds": T.marks, "peak_rss_mb": round(C.peak_rss_mb()), "device": str(device),
                   "gpu_peak_mb": C.gpu_peak_mb(device)}
    res["environment"] = C.environment()
    C.save_json(tree / "baselines.json", res)
    for k in ["ridge", "ridge_generator_test_split", "nn1", "nn1_all_points_as_queries", "nn1_sklearn_check_2000",
              "gate_B", "cost"]:
        print(k, res[k])


if __name__ == "__main__":
    main()
