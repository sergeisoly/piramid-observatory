"""Train the residual MLP of paper Table 5 (one-cluster) on y from x, saving the initial weights first.

The seeding follows the call order of the authors' trainer up to model construction, so the initialization and
batch order follow the same global seed (42); bit-identity with their weights cannot be checked. Training runs on
CUDA, else MPS, else the CPU. Writes to <out>: init.pt, ckpt_eNNN.pt (every 50 epochs, with optimizer state for
resuming), final.pt, train.log (one line per epoch), metrics.json.
"""
import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import common as C


class Block(nn.Module):
    """x + W2 relu(W1 x + b1) + b2 (paper eq. 5); mlpblock[1] is the hidden-layer probe site."""

    def __init__(self, d, expansion):
        super().__init__()
        self.mlpblock = nn.Sequential(nn.Linear(d, expansion * d), nn.ReLU(), nn.Linear(expansion * d, d))

    def forward(self, x):
        return self.mlpblock(x) + x


class ResidualMLP(nn.Module):
    def __init__(self, d_in=C.DIM, d=C.D_MODEL, n_blocks=C.N_BLOCKS, expansion=C.EXPANSION):
        super().__init__()
        self.input_layer = nn.Linear(d_in, d)
        self.mlp_blocks = nn.ModuleList([Block(d, expansion) for _ in range(n_blocks)])
        self.output_layer = nn.Linear(d, 1)

    def forward(self, x, sites=None):
        """With sites (a dict), also stores the activation of every probe site in it."""
        h = self.input_layer(x)
        if sites is not None:
            sites["input_layer"] = h
        for i, b in enumerate(self.mlp_blocks):
            if sites is None:
                h = b(h)
            else:
                hid = b.mlpblock[1](b.mlpblock[0](h))
                h = b.mlpblock[2](hid) + h
                sites[f"hidden{i}"], sites[f"block{i}"] = hid, h
        out = self.output_layer(h)
        if sites is not None:
            sites["output"] = out
        return out


def mse_on(model, X, y, idx, chunk=20_000):
    model.eval()
    tot = 0.0
    with torch.no_grad():
        for s in range(0, len(idx), chunk):
            j = idx[s:s + chunk]
            tot += float(((model(X[j]) - y[j]) ** 2).sum())
    return tot / len(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True, help="folder written by generate.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS, help="epochs to run (the schedule always spans T_max)")
    ap.add_argument("--t-max", type=int, default=C.EPOCHS)
    ap.add_argument("--seed", type=int, default=C.TRAIN_SEED,
                    help="global seed for initialization and batch order (initialization k uses 42 + k)")
    ap.add_argument("--values", help="npz from generate.py --value-seed whose y replaces the dataset's y")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(Path(args.tree) / "dataset.npz")
    y_src = np.load(args.values)["y"] if args.values else data["y"]
    X_np, y_np, split = data["X"].astype(np.float32), y_src.astype(np.float32), data["split"]
    n = len(X_np)

    # Seeding and call order as the authors' trainer: seeds, split, loaders, one validation batch, model.
    # The network split always uses seed 42 (the dataset's split), so runs differ only in initialization and order.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    split_gen = torch.Generator().manual_seed(C.TRAIN_SEED)
    parts = torch.utils.data.random_split(range(n), list(C.SPLIT), generator=split_gen)
    tr, va, te = (np.asarray(p.indices) for p in parts)
    assert (split[tr] == 0).all() and (split[va] == 1).all() and (split[te] == 2).all(), "split differs from dataset"
    train_dl = DataLoader(range(len(tr)), batch_size=C.BATCH, shuffle=True)
    val_dl = DataLoader(range(len(va)), batch_size=C.BATCH, shuffle=False)
    next(iter(val_dl))
    model = ResidualMLP()
    n_params = sum(p.numel() for p in model.parameters())
    if not args.resume:
        torch.save(model.state_dict(), out / "init.pt")

    dev = C.pick_device()
    model.to(dev)
    X = torch.from_numpy(X_np).to(dev)
    y = torch.from_numpy(y_np).to(dev)[:, None]
    tr_d, va_d, te_d = (torch.from_numpy(a).to(dev) for a in (tr, va, te))
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    lossf = nn.MSELoss()
    var_va, var_te = float(np.var(y_np[va])), float(np.var(y_np[te]))

    log = open(out / "train.log", "a")
    t0 = time.perf_counter()
    model.eval()
    with torch.no_grad():  # initial loss over the training batches (the authors' trainer does this pass too)
        init_loss = float(np.mean([float(lossf(model(X[tr_d[p.to(dev)]]), y[tr_d[p.to(dev)]])) for p in train_dl]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.t_max, eta_min=C.ETA_MIN)
    hist = {"epoch": [], "train_loss": [], "val_mse": [], "val_r2": [], "lr": [], "seconds": []}
    start = 0
    if args.resume:
        ck = torch.load(sorted(out.glob("ckpt_e*.pt"))[-1], map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]), opt.load_state_dict(ck["opt"]), sched.load_state_dict(ck["sched"])
        torch.set_rng_state(ck["rng"])
        start, hist = ck["epoch"], ck["hist"]
    print(f"start epoch {start}; params {n_params}; initial train loss {init_loss:.6f}; device {dev}", file=log, flush=True)
    gpu_peak = 0
    for epoch in range(start, args.epochs):
        te0 = time.perf_counter()
        model.train()
        pos = torch.cat(list(train_dl)).to(dev)  # this epoch's shuffled order, drawn from the global seed
        tot = torch.zeros((), device=dev)
        nb = (len(pos) + C.BATCH - 1) // C.BATCH
        for b in range(nb):
            j = tr_d[pos[b * C.BATCH:(b + 1) * C.BATCH]]
            opt.zero_grad()
            loss = lossf(model(X[j]), y[j])
            loss.backward()
            opt.step()
            tot += loss.detach()
        list(val_dl)  # the authors' trainer iterates the validation loader every epoch (same seed consumption)
        val = mse_on(model, X, y, va_d)
        sched.step()
        gpu_peak = max(gpu_peak, C.gpu_peak_mb(dev))
        hist["epoch"].append(epoch + 1)
        hist["train_loss"].append(float(tot) / nb)
        hist["val_mse"].append(val)
        hist["val_r2"].append(1 - val / var_va)
        hist["lr"].append(opt.param_groups[0]["lr"])
        hist["seconds"].append(round(time.perf_counter() - te0, 3))
        print(f"epoch {epoch + 1} train_loss {hist['train_loss'][-1]:.6f} val_mse {val:.6f} "
              f"val_r2 {hist['val_r2'][-1]:.4f} lr {hist['lr'][-1]:.3e} sec {hist['seconds'][-1]:.2f} "
              f"rss_mb {C.peak_rss_mb():.0f}", file=log, flush=True)
        if (epoch + 1) % 50 == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "rng": torch.get_rng_state(), "epoch": epoch + 1, "hist": hist}, out / f"ckpt_e{epoch + 1:03d}.pt")

    wall = time.perf_counter() - t0
    torch.save(model.state_dict(), out / "final.pt")
    test_mse = mse_on(model, X, y, te_d)
    res = {"test": {"mse": test_mse, "r2": 1 - test_mse / var_te}, "val": {"mse": hist["val_mse"][-1],
           "r2": hist["val_r2"][-1]}, "initial_train_loss": init_loss, "params": n_params, "epochs": args.epochs,
           "t_max": args.t_max, "seed": args.seed, "target": "y" if not args.values else f"y from {Path(args.values).name}",
           "device": str(dev),
           "cost": {"wall_seconds": round(wall, 1), "mean_epoch_seconds": float(np.mean(hist["seconds"])),
                    "peak_rss_mb": round(C.peak_rss_mb()), "gpu_peak_mb": gpu_peak},
           "history": hist, "pipeline": C.pipeline_revision(), "environment": C.environment()}
    # Test R2 of the checkpoint after 250 epochs (the pinned public config stops there with the same schedule).
    ck250 = out / "ckpt_e250.pt"
    if ck250.exists():
        m2 = ResidualMLP().to(dev)
        m2.load_state_dict(torch.load(ck250, map_location="cpu", weights_only=False)["model"])
        e = mse_on(m2, X, y, te_d)
        res["test_after_250_epochs"] = {"mse": e, "r2": 1 - e / var_te}
    C.save_json(out / "metrics.json", res)
    print(f"done test_r2 {res['test']['r2']:.4f} test_mse {test_mse:.6f} wall {wall:.0f}s", file=log, flush=True)


if __name__ == "__main__":
    main()
