"""Part A: VAE vs CVAE on 2D toy data, sweeping latent size and architecture.

Batch A1 (latent sweep): {8gaussians, pinwheel} x {VAE, CVAE} x z in {1,2,4,8}
Batch A2 (architecture): pinwheel CVAE, z=2, hidden {32,128,512} x depth {1,2,3}
Each configuration uses seeds 0,1,2 on the same data split (seed 0).

Outputs
  results/toy_results.csv      one row per run (test metrics)
  results/toy_samples.npz      seed-0 samples / posterior means for figures
Run from the project root:  python src/run_toy.py [--quick]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import VAE, active_units, collect_posterior, mmd_rbf, set_seed  # noqa: E402
from data_toy import get_toy  # noqa: E402
from train import evaluate, make_batches, train_vae  # noqa: E402

N_CLASSES = {"8gaussians": 8, "pinwheel": 5}
DEV = torch.device("cpu")  # tiny models: CPU beats MPS kernel-launch overhead


def tensors(split):
    X, y = split
    return (torch.as_tensor(X, dtype=torch.float32, device=DEV),
            torch.as_tensor(y, dtype=torch.long, device=DEV))


def sample_like(model, y_ref, n, seed):
    """Samples from the model; for a CVAE, labels follow the test-label mix."""
    torch.manual_seed(seed)
    y = None
    if model.n_classes:
        y = y_ref[torch.randint(len(y_ref), (n,))]
    return model.sample(n, y=y, noise=True), y


def run_one(ds, data, cond, z_dim, hidden, depth, seed, epochs, batch, warmup=0):
    Xtr, ytr = tensors(data["train"]); Xva, yva = tensors(data["val"])
    Xte, yte = tensors(data["test"])
    set_seed(seed)
    nc = N_CLASSES[ds] if cond else 0
    model = VAE(2, z_dim, nc, hidden=hidden, depth=depth,
                likelihood="gaussian").to(DEV)
    t0 = time.time()
    hist = train_vae(model, make_batches(Xtr, ytr, batch, True, seed),
                     make_batches(Xva, yva, 2048), epochs=epochs, lr=1e-3,
                     patience=40, kl_warmup=warmup)
    test_b = make_batches(Xte, yte, 2048)
    ev = evaluate(model, test_b, iwae_k=500)
    tr = evaluate(model, make_batches(Xtr, ytr, 2048))
    mu, _, _ = collect_posterior(model, test_b)
    n_act, act = active_units(mu)

    xs, ys = sample_like(model, yte, 2000, seed=1000 + seed)
    mmd = mmd_rbf(xs, Xte[:2000])
    # Class-conditional fidelity: mean MMD between samples and data per class.
    mmd_cls = np.nan
    if cond:
        vals = []
        for c in range(nc):
            xc = model.sample(500, y=c)
            vals.append(mmd_rbf(xc, Xte[yte == c]))
        mmd_cls = float(np.mean(vals))
    row = dict(dataset=ds, model="CVAE" if cond else "VAE", z_dim=z_dim,
               hidden=hidden, depth=depth, seed=seed, kl_warmup=warmup,
               best_epoch=hist["best_epoch"], train_time=time.time() - t0,
               train_neg_elbo=tr["neg_elbo"], test_neg_elbo=ev["neg_elbo"],
               test_recon=ev["recon"], test_kl=ev["kl"],
               test_nll_iwae=ev["nll_iwae"], active_units=n_act,
               kl_dim=json.dumps([round(v, 4) for v in ev["kl_dim"]]),
               activity=json.dumps([round(float(v), 4) for v in act]),
               mmd=mmd, mmd_class=mmd_cls)
    extras = dict(samples=xs.numpy(),
                  sample_y=(ys.numpy() if ys is not None else np.zeros(0)),
                  post_mu=mu.numpy(), curves=np.array(
                      [hist["train_neg_elbo"], hist["val_neg_elbo"]]))
    return row, extras


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test")
    ap.add_argument("--batches", default="A1,A2", help="e.g. A3 (KL warm-up)")
    args = ap.parse_args()
    epochs = 3 if args.quick else 600
    seeds = [0] if args.quick else [0, 1, 2]
    os.makedirs("results", exist_ok=True)

    configs = []  # (batch, ds, cond, z, hidden, depth, warmup)
    for ds in ("8gaussians", "pinwheel"):
        for cond in (False, True):
            for z in (1, 2, 4, 8):
                configs.append(("A1", ds, cond, z, 128, 2, 0))
    for hidden in (32, 128, 512):
        for depth in (1, 2, 3):
            if (hidden, depth) != (128, 2):  # already covered by A1
                configs.append(("A2", "pinwheel", True, 2, hidden, depth, 0))
    # A3: unconditional VAE with a 100-epoch linear KL warm-up (collapse test).
    for ds in ("8gaussians", "pinwheel"):
        for z in (1, 2, 4, 8):
            configs.append(("A3", ds, False, z, 128, 2, 100))
    configs = [c for c in configs if c[0] in args.batches.split(",")]
    tag = "" if args.batches == "A1,A2" else "_" + args.batches.replace(",", "_")

    datasets = {ds: get_toy(ds, n=6000, seed=0) for ds in N_CLASSES}
    rows, extras = [], {}
    for i, (b, ds, cond, z, h, d, w) in enumerate(configs):
        for s in seeds:
            row, ex = run_one(ds, datasets[ds], cond, z, h, d, s, epochs, 256, w)
            row["batch"] = b
            rows.append(row)
            print(f"[{i + 1}/{len(configs)}] {b} {ds:10s} {row['model']:4s} z={z} "
                  f"h={h} d={d} s={s}  test -ELBO {row['test_neg_elbo']:.3f} "
                  f"IWAE {row['test_nll_iwae']:.3f} AU {row['active_units']} "
                  f"MMD {row['mmd']:.4f}  ({row['train_time']:.0f}s)", flush=True)
            if s == 0:
                key = f"{ds}|{row['model']}|z{z}|h{h}|d{d}" + (f"|w{w}" if w else "")
                for k, v in ex.items():
                    extras[f"{key}|{k}"] = v
            pd.DataFrame(rows).to_csv(f"results/toy_results{tag}.csv", index=False)
    # Ground-truth test data for the figures (standardized space).
    for ds, data in datasets.items():
        extras[f"{ds}|test_X"], extras[f"{ds}|test_y"] = data["test"]
    np.savez_compressed(f"results/toy_samples{tag}.npz", **extras)
    print("done:", len(rows), "runs")


if __name__ == "__main__":
    main()
