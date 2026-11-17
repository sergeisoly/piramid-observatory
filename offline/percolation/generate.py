"""Generate the one-cluster dataset with the authors' generator and store it with compact ground truth.

Writes <out>/dataset.npz and <out>/data_summary.json (gate A checks and statistics).

Row i of X, y and every per-point array is the generator's point with point_idx i. The generator's leaf
permutation (the random cyclic arrangement inside the coalescent) is used as is; nothing is relabeled.
Ground truth is stored as a parent pointer per node and a table of each point's ancestor at depths 0, 5, ..., 200,
instead of the generator's sparse matrix of all ancestors (about 9.4e7 entries).
"""
import argparse
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

import common as C


def compact_tree(points, latents):
    """Parent pointers, depths and values as arrays. Latent j is the j-th latent in the generator's dict order."""
    n = len(points)
    keys = np.fromiter(latents.keys(), np.int64, len(latents))
    key2j = {int(k): j for j, k in enumerate(keys)}
    lat_parent = np.full(len(keys), -1, np.int32)
    lat_depth = np.empty(len(keys), np.int32)
    lat_z = np.empty(len(keys), np.float64)
    lat_vsum = np.empty(len(keys), np.float64)
    for j, node in enumerate(latents.values()):
        if node.parents:
            lat_parent[j] = key2j[next(iter(node.parents)).point_idx]
        lat_depth[j] = node.depth
        lat_z[j] = node.value
        lat_vsum[j] = node.value_sum
    pt_idx = np.fromiter((p.point_idx for p in points), np.int64, n)
    pt_parent = np.fromiter((key2j[next(iter(p.parents)).point_idx] for p in points), np.int32, n)
    pt_depth = np.fromiter((p.depth for p in points), np.int32, n)
    pt_z = np.fromiter((p.value for p in points), np.float64, n)
    edges = np.array(sorted({(min(p.point_idx, q.point_idx), max(p.point_idx, q.point_idx))
                             for p in points for q in p.neighbors}), np.int64)
    return dict(lat_key=keys, lat_parent=lat_parent, lat_depth=lat_depth, lat_z=lat_z, lat_vsum=lat_vsum,
                pt_idx=pt_idx, pt_parent=pt_parent, pt_depth=pt_depth, pt_z=pt_z, edges=edges)


def subtree_sizes(pt_parent, lat_parent, lat_depth):
    size = np.bincount(pt_parent, minlength=len(lat_parent)).astype(np.int64)
    for d in range(int(lat_depth.max()), 0, -1):
        js = np.nonzero(lat_depth == d)[0]
        np.add.at(size, lat_parent[js], size[js])
    return size


def ancestor_table(pt_parent, lat_parent, lat_depth, lat_z, pt_z):
    """anc[i, c] = latent at depth ANC_DEPTHS[c] above point i (-1 if none); also the sum of z along each path."""
    col = np.full(int(lat_depth.max()) + 1, -1, np.int64)
    for c, d in enumerate(C.ANC_DEPTHS):
        if d < len(col):
            col[d] = c
    n = len(pt_parent)
    anc = np.full((n, len(C.ANC_DEPTHS)), -1, np.int32)
    zsum = pt_z.copy()
    cur = pt_parent.astype(np.int64)
    idx = np.arange(n)
    while len(idx):
        c = col[lat_depth[cur]]
        hit = c >= 0
        anc[idx[hit], c[hit]] = cur[hit]
        zsum[idx] += lat_z[cur]
        cur = lat_parent[cur].astype(np.int64)
        keep = cur >= 0
        idx, cur = idx[keep], cur[keep]
    return anc, zsum


def eligible_stats(anc, lat_size):
    rows = []
    for c, d in enumerate(C.ANC_DEPTHS):
        a = anc[:, c]
        a = a[a >= 0]
        lats, cnt = np.unique(a, return_counts=True)
        ok = cnt >= C.MIN_LATENT_POINTS
        assert np.array_equal(cnt, lat_size[lats])  # every point under a depth-d latent has depth > d
        rows.append(dict(depth=d, points=int(cnt[ok].sum()), latents=int(ok.sum()),
                         min_size=int(cnt[ok].min()) if ok.any() else 0,
                         max_size=int(cnt[ok].max()) if ok.any() else 0))
    return rows


def value_swap(gen, args, out):
    """New latent and point values (z', y') on the same tree; x is untouched because it does not use the value seed."""
    seeds = dict(graph=args.graph, embed=args.embed, value=args.value_seed)
    ds = gen.PercolationDataset("one_cluster", seeds=gen.Seeds(**seeds))
    points, latents = ds.construct(C.N_POINTS)
    y = ds.embed_labels(points, latents)
    t = compact_tree(points, latents)
    ref = np.load(out / "dataset.npz")
    same_tree = all(np.array_equal(t[k], ref[k]) for k in ("pt_parent", "pt_depth", "lat_parent", "lat_depth"))
    summary = dict(seeds=seeds, same_tree_as_dataset=bool(same_tree),
                   corr_latent_values=float(np.corrcoef(t["lat_z"], ref["lat_z"])[0, 1]),
                   corr_y=float(np.corrcoef(y, ref["y"])[0, 1]), y_mean_std=[float(y.mean()), float(y.std())],
                   array_sha256_16={"y": C.array_sha256(y), "lat_z": C.array_sha256(t["lat_z"])})
    assert same_tree, "value-swap tree differs from the dataset's tree"
    np.savez(out / f"values_v{args.value_seed}.npz", y=y, lat_z=t["lat_z"], pt_z=t["pt_z"])
    C.save_json(out / f"values_v{args.value_seed}.json", summary)
    print(summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", required=True, help="checkout of percolation-synthetic-data at the pinned commit")
    ap.add_argument("--out", required=True, help="output folder for this tree")
    ap.add_argument("--graph", type=int, default=C.DATA_SEEDS["graph"])
    ap.add_argument("--embed", type=int, default=C.DATA_SEEDS["embed"])
    ap.add_argument("--value", type=int, default=C.DATA_SEEDS["value"])
    ap.add_argument("--compare", help="optional JSON of expected per-depth counts (report 02's sim_stats.json)")
    ap.add_argument("--value-seed", type=int,
                    help="value-swap mode: same tree and x as <out>/dataset.npz, new values; writes values_v<seed>.npz")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    T = C.Timer()

    gen = C.load_generator(args.generator)
    if args.value_seed is not None:
        return value_swap(gen, args, out)
    seeds = dict(graph=args.graph, embed=args.embed, value=args.value)
    ds = gen.PercolationDataset("one_cluster", seeds=gen.Seeds(**seeds))
    points, latents = ds.construct(C.N_POINTS)
    T.mark("construct")
    X = ds.embed_features(points, latents, C.DIM)
    T.mark("embed_features")
    y = ds.embed_labels(points, latents)
    T.mark("embed_labels")
    rss_generator = C.peak_rss_mb()

    t = compact_tree(points, latents)
    n = len(points)
    checks = {}
    checks["one_cluster"] = len({p.cluster_idx for p in points}) == 1 and n == C.N_POINTS
    checks["rows_are_point_idx"] = bool(np.array_equal(t["pt_idx"], np.arange(n)))
    checks["latent_count_n_minus_1"] = len(t["lat_key"]) == n - 1

    # Brute-force ancestor walk on 1,000 random points with the generator's own node objects.
    rng = np.random.default_rng(0)
    sample = rng.choice(n, 1000, replace=False)
    brute = {}
    for i in sample:
        node, path = points[i], {}
        while node.parents:
            node = next(iter(node.parents))
            path[node.depth] = node.point_idx
        brute[int(i)] = path
    T.mark("compact_and_brute_walk")
    del points, latents, ds

    lat_size = subtree_sizes(t["pt_parent"], t["lat_parent"], t["lat_depth"])
    anc, zsum = ancestor_table(t["pt_parent"], t["lat_parent"], t["lat_depth"], t["lat_z"], t["pt_z"])
    T.mark("ancestor_table")

    ok = True
    for i, path in brute.items():
        for c, d in enumerate(C.ANC_DEPTHS):
            want = path.get(d)
            got = anc[i, c]
            ok &= (want is None and got == -1) or (want is not None and got >= 0 and t["lat_key"][got] == want)
        ok &= len(path) == t["pt_depth"][i]
    checks["ancestor_table_matches_brute_walk_1000"] = bool(ok)
    checks["root_unique_depth0"] = int((t["lat_depth"] == 0).sum()) == 1 and int(t["lat_parent"][0]) == -1
    checks["point_depth_is_parent_depth_plus_1"] = bool(np.all(t["pt_depth"] == t["lat_depth"][t["pt_parent"]] + 1))
    checks["root_size_is_n"] = int(lat_size[t["lat_depth"] == 0][0]) == n
    # y row identity: the generator's label equals the path sum of values over sqrt(1 + depth).
    y_tree = zsum / np.sqrt(t["pt_depth"] + 1.0)
    checks["y_equals_path_sum_max_abs_err"] = float(np.abs(y_tree - y).max())
    checks["y_row_identity"] = checks["y_equals_path_sum_max_abs_err"] < 1e-9
    # X row identity: every graph edge has length step * sqrt(d) (step = n^(-1/4) in one-cluster mode).
    e = t["edges"]
    edge_len = np.linalg.norm(X[e[:, 0]] - X[e[:, 1]], axis=1)
    step = n ** -0.25 * math.sqrt(C.DIM)
    checks["edges_n_minus_1"] = len(e) == n - 1
    checks["edge_length_max_rel_err"] = float(np.abs(edge_len / step - 1).max())
    checks["x_row_identity"] = checks["edge_length_max_rel_err"] < 1e-9
    parent_uf = np.arange(n)

    def find(a):
        while parent_uf[a] != a:
            parent_uf[a] = parent_uf[parent_uf[a]]
            a = parent_uf[a]
        return a
    for a, b in e:
        ra, rb = find(a), find(b)
        parent_uf[ra] = rb
    checks["graph_is_tree"] = checks["edges_n_minus_1"] and len({find(i) for i in range(n)}) == 1
    # Generator-test style checks.
    deg = np.bincount(e.ravel(), minlength=n)
    vals, cnt = np.unique(deg, return_counts=True)
    dom = np.arange(1, deg.max() + 1)
    pk = np.array([dict(zip(vals, cnt / n)).get(k, 0.0) for k in dom])
    qk = stats.poisson.pmf(dom, mu=1, loc=1)
    checks["degree_kl_vs_1_plus_poisson1"] = float(stats.entropy(pk, qk / qk.sum()))
    var = X.var(axis=0)
    checks["x_coord_var_min_max"] = [float(var.min()), float(var.max())]
    checks["x_coord_var_in_0.1_2"] = bool(var.min() > 0.1 and var.max() < 2.0)
    checks["y_mean_std"] = [float(y.mean()), float(y.std())]

    # Network split (torch random_split with Generator seed 42, as the authors' trainer).
    import torch
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(C.TRAIN_SEED)).numpy()
    split = np.empty(n, np.int8)
    a, b = C.SPLIT[0], C.SPLIT[0] + C.SPLIT[1]
    split[perm[:a]], split[perm[a:b]], split[perm[b:]] = 0, 1, 2
    checks["split_counts"] = np.bincount(split).tolist()

    elig = eligible_stats(anc, lat_size)
    entries_all = sum(r["latents"] for r in elig)
    entries_probed = sum(r["latents"] for r in elig if r["depth"] in C.PROBE_DEPTHS)
    largest_share = [r["max_size"] / r["points"] for r in elig if r["points"]]
    summary = dict(
        seeds=seeds, n=n, d=C.DIM, latents=len(t["lat_key"]),
        point_depth=dict(mean=float(t["pt_depth"].mean()), median=float(np.median(t["pt_depth"])),
                         max=int(t["pt_depth"].max()), min=int(t["pt_depth"].min())),
        eligible=elig, entries_depth_0_200=entries_all, entries_probed_10_200=entries_probed,
        largest_latent_share_of_eligible=[float(min(largest_share)), float(max(largest_share))],
        latents_per_probed_depth=[min(r["latents"] for r in elig if r["depth"] in C.PROBE_DEPTHS),
                                  max(r["latents"] for r in elig if r["depth"] in C.PROBE_DEPTHS)],
        checks=checks,
    )
    if args.compare:
        ref = C.load_json(args.compare)
        same = [all(r[k] == s[k] for k in r) for r, s in zip(elig, ref["one_cluster"])]
        summary["compare_with_report02_mirror"] = dict(
            per_depth_rows_equal=int(sum(same)), of=len(same),
            entries_equal=entries_all == ref["total_entries"],
            mean_depth_equal=abs(summary["point_depth"]["mean"] - ref["mean_point_depth"]) < 1e-9,
            max_depth_equal=summary["point_depth"]["max"] == ref["max_point_depth"])

    summary["array_sha256_16"] = {k: C.array_sha256(v) for k, v in dict(
        X=X, y=y, split=split, anc=anc, pt_parent=t["pt_parent"], lat_z=t["lat_z"], lat_size=lat_size).items()}
    summary["array_sums"] = {"X": float(X.sum()), "abs_X": float(np.abs(X).sum()), "y": float(y.sum())}
    np.savez(out / "dataset.npz", X=X, y=y, split=split, anc=anc, anc_depths=np.array(C.ANC_DEPTHS),
             pt_parent=t["pt_parent"], pt_depth=t["pt_depth"], pt_z=t["pt_z"], lat_key=t["lat_key"],
             lat_parent=t["lat_parent"], lat_depth=t["lat_depth"], lat_z=t["lat_z"], lat_size=lat_size,
             edges=e)
    T.mark("total")
    summary["cost"] = dict(seconds=T.marks, peak_rss_mb_after_generator=round(rss_generator),
                           peak_rss_mb=round(C.peak_rss_mb()))
    summary["provenance"] = dict(generator=dict(repo=C.GENERATOR_REPO, commit=C.GENERATOR_COMMIT,
                                                file_sha256=C.GENERATOR_SHA256, license="MIT"),
                                 pipeline=C.pipeline_revision(), environment=C.environment())
    passed = all(checks[k] for k in ["one_cluster", "rows_are_point_idx", "latent_count_n_minus_1",
                                     "ancestor_table_matches_brute_walk_1000", "root_unique_depth0",
                                     "point_depth_is_parent_depth_plus_1", "root_size_is_n", "y_row_identity",
                                     "x_row_identity", "graph_is_tree"])
    passed &= checks["degree_kl_vs_1_plus_poisson1"] < 1e-3
    summary["row_identity_and_structure_checks_pass"] = bool(passed)
    # The variance bound comes from the generator's 10k-point, d = 128 test fixture: informational here.
    summary["informational"] = ["x_coord_var_in_0.1_2", "y_mean_std"]
    C.save_json(out / "data_summary.json", summary)
    print(f"gate A checks pass: {passed}; entries 0-200: {entries_all}, probed 10-200: {entries_probed}; "
          f"mean depth {summary['point_depth']['mean']:.6f}, max {summary['point_depth']['max']}; "
          f"time {T.marks}; peak RSS {summary['cost']['peak_rss_mb']} MB")


if __name__ == "__main__":
    main()
