"""Evaluate sample quality of all trained MNIST (C)VAEs with the fixed classifier.

Writes results/mnist_sample_metrics.csv (resumable; --force re-evaluates), results/fid_reference.csv,
and for seed-0 runs results/samples/<run_id>.npz (sample grid, style swap, reconstructions, z=2 manifold).
"""
import argparse
import csv
import glob
import json
import os
import sys
import numpy as np
import torch
from scipy import linalg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import VAE  # noqa: E402
from data_mnist import get_mnist_tensors  # noqa: E402
from classifier import load_classifier  # noqa: E402

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
CSV = os.path.join(R, 'mnist_sample_metrics.csv')
CFG = ['run_id', 'model', 'arch', 'z', 'h', 'd', 'seed', 'batch']
MET = ['acc_mean', 'acc_binary', 'conf_mean', 'pred_entropy', 'fid_binary', 'fid_mean']
dev = torch.device('mps')


@torch.no_grad()
def feats(clf, X, bs=1000):
    return torch.cat([clf.features(X[i:i + bs].to(dev)).cpu() for i in range(0, len(X), bs)]).double().numpy()


@torch.no_grad()
def probs(clf, X, bs=1000):
    return torch.cat([clf.logits(X[i:i + bs].to(dev)).softmax(-1).cpu() for i in range(0, len(X), bs)])


def frechet(f1, f2):
    """Frechet distance between Gaussians fitted to two feature sets."""
    m1, m2 = f1.mean(0), f2.mean(0)
    s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    cs = linalg.sqrtm(s1 @ s2)
    return float(((m1 - m2) ** 2).sum() + np.trace(s1 + s2 - 2 * np.real(cs)))


def build(cfg):
    m = VAE(784, cfg['z'], 10 if cfg['model'] == 'CVAE' else 0, hidden=cfg['h'], depth=cfg['d'],
            likelihood='bernoulli', arch=cfg['arch']).to(dev)
    m.load_state_dict(torch.load(os.path.join(R, 'ckpt', cfg['run_id'] + '.pt'), map_location=dev))
    return m.eval()


@torch.no_grad()
def eval_run(cfg, clf, ftest, Xte, yte):
    m, cond = build(cfg), cfg['model'] == 'CVAE'
    torch.manual_seed(0)
    labels = torch.arange(10).repeat_interleave(1000).to(dev)
    P = m.sample(10000, y=labels if cond else None)
    Bs = torch.bernoulli(P)
    out = {}
    pm, pb = probs(clf, P), probs(clf, Bs)
    pred = pm.argmax(1)
    out['acc_mean'] = (pred == labels.cpu()).float().mean().item() if cond else ''
    out['acc_binary'] = (pb.argmax(1) == labels.cpu()).float().mean().item() if cond else ''
    out['conf_mean'] = pm.max(1).values.mean().item()
    h = torch.bincount(pred, minlength=10).double() / len(pred)
    out['pred_entropy'] = -(h[h > 0] * h[h > 0].log()).sum().item()
    out['fid_binary'] = frechet(feats(clf, Bs), ftest)
    out['fid_mean'] = frechet(feats(clf, P), ftest)
    if cfg['seed'] == 0:
        save_extras(cfg, m, cond, Xte, yte)
    return out


@torch.no_grad()
def save_extras(cfg, m, cond, Xte, yte):
    """Fixed sample grid, style swap, recon (CVAE) and z=2 manifold."""
    g = torch.Generator().manual_seed(7)
    z10 = torch.randn(10, cfg['z'], generator=g).to(dev)
    d = {}
    if cond:
        zz = z10.repeat(10, 1)  # class-major: row c uses the same 10 z
        d['grid'] = m.sample(100, y=torch.arange(10).repeat_interleave(10).to(dev), z=zz)
        x, y = Xte[:8].to(dev), yte[:8].to(dev)
        mu, _ = m.encode(x, y)
        d['style_swap'] = torch.stack([torch.sigmoid(m.decode(mu, torch.full((8,), c, device=dev))) for c in range(10)], 1)
        d['recon'] = torch.sigmoid(m.decode(mu, y))
        d['orig'] = x
    else:
        d['grid'] = m.sample(100, z=torch.randn(100, cfg['z'], generator=g).to(dev))
    if cfg['z'] == 2:
        q = torch.distributions.Normal(0., 1.).icdf(torch.linspace(0.05, 0.95, 15))
        z = torch.stack([torch.stack([q[j], q[14 - i]]) for i in range(15) for j in range(15)]).to(dev)  # row i: z2 desc, col j: z1 asc
        if cond:
            d['manifold'] = torch.stack([torch.sigmoid(m.decode(z, torch.full((225,), c, device=dev))) for c in (3, 8)])
        else:
            d['manifold'] = torch.sigmoid(m.decode(z))
    os.makedirs(os.path.join(R, 'samples'), exist_ok=True)
    np.savez_compressed(os.path.join(R, 'samples', cfg['run_id'] + '.npz'), **{k: v.cpu().numpy() for k, v in d.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true')
    force = ap.parse_args().force
    clf = load_classifier(dev)
    data = get_mnist_tensors()
    Xte, yte = data['test']
    ftest = feats(clf, Xte)
    ref = os.path.join(R, 'fid_reference.csv')
    if not os.path.exists(ref) or force:
        with open(ref, 'w') as f:
            f.write(f'fid_test_vs_train\n{frechet(feats(clf, data["train"][0][:10000]), ftest)}\n')
    done = {}
    if os.path.exists(CSV) and not force:
        done = {r['run_id']: r for r in csv.DictReader(open(CSV))}
    for p in sorted(glob.glob(os.path.join(R, 'mnist', '*.json'))):
        cfg = json.load(open(p))
        rid = cfg['run_id']
        if rid in done or not os.path.exists(os.path.join(R, 'ckpt', rid + '.pt')):
            continue
        done[rid] = {**{k: cfg[k] for k in CFG}, **eval_run(cfg, clf, ftest, Xte, yte)}
        print(done[rid], flush=True)
        with open(CSV, 'w', newline='') as f:  # rewrite whole file so rows stay consistent
            w = csv.DictWriter(f, CFG + MET)
            w.writeheader(); w.writerows(done.values())


if __name__ == '__main__':
    main()
