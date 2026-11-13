"""Statically binarized MNIST (fixed Bernoulli sample), cached to data/mnist_bin.pt."""
import os
import torch
from torchvision import datasets

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
CACHE = os.path.join(ROOT, 'mnist_bin.pt')


def _binarize(ds, seed):
    """Sample Bernoulli(pixel/255) once with a fixed generator; return (uint8 (N,784), labels)."""
    p = ds.data.float().div(255).reshape(len(ds.data), -1)
    g = torch.Generator().manual_seed(seed)
    return torch.bernoulli(p, generator=g).to(torch.uint8), ds.targets.clone()


def _build_cache():
    tr = datasets.MNIST(ROOT, train=True, download=True)
    te = datasets.MNIST(ROOT, train=False, download=True)
    Xtr, ytr = _binarize(tr, 0)
    Xte, yte = _binarize(te, 1)
    torch.save({'x_train': Xtr, 'y_train': ytr, 'x_test': Xte, 'y_test': yte}, CACHE)


def get_mnist_tensors():
    """Return {'train','val','test'} -> (X float32 (N,784) in {0,1}, y int64). 50k/10k/10k split."""
    if not os.path.exists(CACHE):
        _build_cache()
    d = torch.load(CACHE)
    perm = torch.randperm(60000, generator=torch.Generator().manual_seed(0))
    tr, va = perm[:50000], perm[50000:]
    f = lambda x: x.float()
    return {'train': (f(d['x_train'][tr]), d['y_train'][tr].long()),
            'val': (f(d['x_train'][va]), d['y_train'][va].long()),
            'test': (f(d['x_test']), d['y_test'].long())}


class TensorBatcher:
    """Iterate (xb, yb) minibatches of in-memory tensors; reshuffles each epoch if shuffle."""

    def __init__(self, X, y, batch_size, shuffle=False, seed=0):
        self.X, self.y, self.bs, self.shuffle = X, y, batch_size, shuffle
        self.gen = torch.Generator().manual_seed(seed)

    def __len__(self):
        return (len(self.X) + self.bs - 1) // self.bs

    def __iter__(self):
        n = len(self.X)
        idx = torch.randperm(n, generator=self.gen) if self.shuffle else torch.arange(n)
        idx = idx.to(self.X.device)
        for i in range(0, n, self.bs):
            j = idx[i:i + self.bs]
            yield self.X[j], self.y[j]


def get_mnist_loaders(batch_size=128, device=None):
    """Return {'train','val','test'} -> TensorBatcher with data placed on `device` (train shuffled)."""
    data = get_mnist_tensors()
    return {k: TensorBatcher(X.to(device), y.to(device), batch_size, shuffle=(k == 'train'))
            for k, (X, y) in data.items()}
