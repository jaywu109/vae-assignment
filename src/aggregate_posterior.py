"""Compare the aggregate posterior q(z) on the MNIST test set with the prior N(0, I).

For every trained run, fit a Gaussian to q(z) = E_x q(z|x) by moment matching
(mean of mu; Cov_x(mu) + E_x diag(sigma^2)) and report KL(N(m, S) || N(0, I)),
together with the average per-example KL E_x KL(q(z|x) || p(z)).
Writes results/mnist_aggregate_posterior.csv.
"""
import csv
import glob
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import VAE  # noqa: E402
from data_mnist import get_mnist_tensors  # noqa: E402

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def parse(run_id):
    parts = run_id.split('_')
    get = lambda p: int(next(s for s in parts if s.startswith(p))[len(p):])
    return dict(cond=parts[0] == 'CVAE', arch=parts[1], z=get('z'), h=get('h'), d=get('d'))


@torch.no_grad()
def main():
    Xte, yte = get_mnist_tensors()['test']
    rows = []
    for f in sorted(glob.glob(os.path.join(R, 'mnist', '*.json'))):
        rid = os.path.basename(f)[:-5]
        c = parse(rid)
        m = VAE(784, c['z'], 10 if c['cond'] else 0, hidden=c['h'], depth=c['d'],
                likelihood='bernoulli', arch=c['arch']).to(dev)
        m.load_state_dict(torch.load(os.path.join(R, 'ckpt', rid + '.pt'), map_location=dev))
        m.eval()
        mus, vs = [], []
        for i in range(0, len(Xte), 1000):
            mu, lv = m.encode(Xte[i:i + 1000].to(dev), yte[i:i + 1000].to(dev) if c['cond'] else None)
            mus.append(mu.cpu().double()); vs.append(lv.exp().cpu().double())
        mu, v = torch.cat(mus), torch.cat(vs)
        z = c['z']
        mean = mu.mean(0)
        S = torch.cov(mu.T).reshape(z, z) + torch.diag(v.mean(0))
        kl_marg = 0.5 * (torch.trace(S) + mean @ mean - z - torch.logdet(S)).item()
        kl_avg = (0.5 * (v + mu ** 2 - 1 - v.log())).sum(1).mean().item()
        rows.append(dict(run_id=rid, z=z, kl_avg=round(kl_avg, 3), kl_marg_gauss=round(kl_marg, 4)))
        print(rows[-1], flush=True)
    with open(os.path.join(R, 'mnist_aggregate_posterior.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


if __name__ == '__main__':
    main()
