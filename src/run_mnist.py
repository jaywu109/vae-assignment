"""Part B: (C)VAEs on statically binarized MNIST.

Batches (all MLP h=512 d=2, Adam lr 1e-3, batch 128 unless varied):
  B1  CVAE latent sweep  z in {2,4,8,16,32,64}, seeds 0,1,2
  B1u VAE  latent sweep  z in {2,4,8,16,32,64}, seed 0 (unconditional baseline)
  B2  CVAE z=16 architecture: MLP h {256,512} x d {1,2,3} + conv, seeds 0,1
  B3  CVAE z=16 optimisation: lr {3e-4,3e-3} and batch {64,256}, seed 0

Each run writes results/mnist/<run_id>.json (metrics + curves) and
results/ckpt/<run_id>.pt (weights); finished runs are skipped, so the
script is resumable and can be sharded across processes:
    python src/run_mnist.py --shard 0 --nshards 3   (and 1, 2 in parallel)
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import VAE, active_units, collect_posterior, count_params, get_device, set_seed  # noqa: E402
from data_mnist import TensorBatcher, get_mnist_tensors  # noqa: E402
from train import evaluate, train_vae  # noqa: E402

OUT, CKPT = "results/mnist", "results/ckpt"


def configs():
    base = dict(model="CVAE", arch="mlp", z=16, h=512, d=2, lr=1e-3, bs=128)
    cs = []
    for z in (2, 4, 8, 16, 32, 64):
        for s in (0, 1, 2):
            cs.append(dict(base, batch="B1", z=z, seed=s))
        cs.append(dict(base, batch="B1u", model="VAE", z=z, seed=0))
    for h in (256, 512):
        for d in (1, 2, 3):
            for s in (0, 1):
                if (h, d) == (512, 2) and s in (0, 1):
                    continue  # identical to B1 z=16 seeds 0,1
                cs.append(dict(base, batch="B2", h=h, d=d, seed=s))
    for s in (0, 1):
        cs.append(dict(base, batch="B2", arch="conv", seed=s))
    for lr in (3e-4, 3e-3):
        cs.append(dict(base, batch="B3", lr=lr, seed=0))
    for bs in (64, 256):
        cs.append(dict(base, batch="B3", bs=bs, seed=0))
    for c in cs:
        c["run_id"] = (f"{c['model']}_{c['arch']}_z{c['z']}_h{c['h']}_d{c['d']}"
                       f"_lr{c['lr']:g}_bs{c['bs']}_s{c['seed']}")
    return cs


def run(cfg, data, dev, epochs, patience, iwae_k):
    set_seed(cfg["seed"])
    nc = 10 if cfg["model"] == "CVAE" else 0
    model = VAE(784, cfg["z"], nc, hidden=cfg["h"], depth=cfg["d"],
                likelihood="bernoulli", arch=cfg["arch"]).to(dev)
    B = {k: TensorBatcher(X, y, cfg["bs"] if k == "train" else 1000,
                          shuffle=(k == "train"), seed=cfg["seed"])
         for k, (X, y) in data.items()}
    t0 = time.time()
    hist = train_vae(model, B["train"], B["val"], epochs=epochs, lr=cfg["lr"],
                     patience=patience)
    train_time = time.time() - t0
    torch.manual_seed(123)
    te = evaluate(model, B["test"], iwae_k=iwae_k)
    tr = evaluate(model, TensorBatcher(*data["train"], 1000))
    mu, _, _ = collect_posterior(model, B["test"])
    n_act, act = active_units(mu)
    res = dict(cfg, n_params=count_params(model), train_time=train_time,
               epochs_run=len(hist["val_neg_elbo"]), best_epoch=hist["best_epoch"],
               train_neg_elbo=tr["neg_elbo"], train_recon=tr["recon"], train_kl=tr["kl"],
               test_neg_elbo=te["neg_elbo"], test_recon=te["recon"], test_kl=te["kl"],
               test_nll_iwae=te["nll_iwae"], active_units=n_act,
               activity=act.tolist(), kl_dim=te["kl_dim"],
               curves={k: hist[k] for k in ("train_neg_elbo", "val_neg_elbo",
                                            "train_kl", "val_kl")})
    torch.save(model.state_dict(), f"{CKPT}/{cfg['run_id']}.pt")
    with open(f"{OUT}/{cfg['run_id']}.json", "w") as f:
        json.dump(res, f)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--batches", default="B1,B1u,B2,B3")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--iwae_k", type=int, default=200)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--quick", action="store_true", help="1 epoch, 1 config")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); os.makedirs(CKPT, exist_ok=True)
    dev = get_device(a.device)
    data = {k: (X.to(dev), y.to(dev)) for k, (X, y) in get_mnist_tensors().items()}
    todo = [c for c in configs() if c["batch"] in a.batches.split(",")]
    todo = todo[a.shard::a.nshards]
    if a.quick:
        todo, a.epochs, a.iwae_k = todo[:1], 1, 10
    for i, c in enumerate(todo):
        if not a.quick and os.path.exists(f"{OUT}/{c['run_id']}.json"):
            continue
        r = run(c, data, dev, a.epochs, a.patience, a.iwae_k)
        print(f"[shard {a.shard} {i + 1}/{len(todo)}] {c['run_id']}  "
              f"test -ELBO {r['test_neg_elbo']:.2f}  IWAE {r['test_nll_iwae']:.2f}  "
              f"KL {r['test_kl']:.2f}  AU {r['active_units']}  "
              f"ep {r['epochs_run']}  {r['train_time'] / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
