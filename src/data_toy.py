"""2D toy datasets (numpy only), deterministic given a seed."""
import numpy as np


def make_8gaussians(n, seed, radius=2.0, std=0.2):
    """8 Gaussian clusters on a circle; label k = cluster index. Returns (X float32 (n,2), y int64)."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 8, size=n)
    ang = 2 * np.pi * y / 8
    means = radius * np.stack([np.cos(ang), np.sin(ang)], 1)
    X = means + std * rng.standard_normal((n, 2))
    return X.astype(np.float32), y.astype(np.int64)


def make_two_moons(n, seed, noise=0.08):
    """Two interleaving half-circles; y in {0,1}. Returns (X float32 (n,2), y int64)."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    t = np.pi * rng.random(n)
    upper = np.stack([np.cos(t), np.sin(t)], 1)
    lower = np.stack([1 - np.cos(t), 0.5 - np.sin(t)], 1)
    X = np.where(y[:, None] == 0, upper, lower) + noise * rng.standard_normal((n, 2))
    return X.astype(np.float32), y.astype(np.int64)


def make_pinwheel(n, seed, n_arms=5, radial_std=0.3, tangential_std=0.05, rate=0.25):
    """Standard pinwheel (Johnson et al. 2016); y = arm index. Returns (X float32 (n,2), y int64)."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_arms, size=n)
    feats = rng.standard_normal((n, 2)) * np.array([radial_std, tangential_std])
    feats[:, 0] += 1.0  # radial coordinate ~ N(1, radial_std), tangential ~ N(0, tangential_std)
    ang = 2 * np.pi * y / n_arms + rate * np.exp(feats[:, 0])  # outer points twist more
    c, s = np.cos(ang), np.sin(ang)
    X = np.stack([feats[:, 0] * c - feats[:, 1] * s, feats[:, 0] * s + feats[:, 1] * c], 1)
    return X.astype(np.float32), y.astype(np.int64)


def split_and_standardize(X, y, seed, frac=(0.7, 0.15, 0.15)):
    """Shuffle, split into train/val/test; standardize with train mean/std. Returns dict."""
    n = len(X)
    perm = np.random.default_rng(seed).permutation(n)
    n_tr, n_va = int(frac[0] * n), int(frac[1] * n)
    idx = {'train': perm[:n_tr], 'val': perm[n_tr:n_tr + n_va], 'test': perm[n_tr + n_va:]}
    mean = X[idx['train']].mean(0)
    std = X[idx['train']].std(0)
    out = {k: (((X[i] - mean) / std).astype(np.float32), y[i]) for k, i in idx.items()}
    out['mean'], out['std'] = mean.astype(np.float32), std.astype(np.float32)
    return out


def get_toy(name, n=6000, seed=0):
    """Build toy dataset by name ('8gaussians'|'moons'|'pinwheel') and return the split dict."""
    makers = {'8gaussians': make_8gaussians, 'moons': make_two_moons, 'pinwheel': make_pinwheel}
    X, y = makers[name](n, seed)
    return split_and_standardize(X, y, seed)
