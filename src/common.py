"""Shared VAE / CVAE models, ELBO loss, and evaluation metrics.

Conventions
-----------
* Every model takes ``(x, y)`` where ``y`` is an int64 label tensor (or
  ``None`` for an unconditional model). A conditional model one-hot encodes
  ``y`` and concatenates it to the encoder input and to the decoder input.
* All per-example quantities are in nats and summed over data dimensions.
* ``likelihood='bernoulli'`` -> decoder returns logits (binarized MNIST).
  ``likelihood='gaussian'``  -> decoder returns (mean, std) per dimension
  with a learned, input-dependent diagonal variance (2D toy data).
"""
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_2PI = math.log(2 * math.pi)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(prefer="mps"):
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def mlp(in_dim, hidden, depth, out_dim, act=nn.ReLU):
    """``depth`` hidden layers of width ``hidden`` followed by a linear head."""
    layers, d = [], in_dim
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), act()]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class VAE(nn.Module):
    """MLP or convolutional (C)VAE with a diagonal-Gaussian posterior.

    Args:
        x_dim: data dimension (2 for toy data, 784 for MNIST).
        z_dim: latent dimension.
        n_classes: number of condition classes; 0 -> unconditional VAE.
        hidden, depth: width and number of hidden layers of each MLP.
        likelihood: 'bernoulli' or 'gaussian'.
        arch: 'mlp' or 'conv' (conv requires x_dim == 784).
        min_std: floor on the Gaussian decoder std (prevents the likelihood
            from diverging by shrinking the variance onto training points).
    """

    def __init__(self, x_dim, z_dim, n_classes=0, hidden=256, depth=2,
                 likelihood="bernoulli", arch="mlp", min_std=1e-2):
        super().__init__()
        assert likelihood in ("bernoulli", "gaussian")
        assert arch in ("mlp", "conv")
        self.x_dim, self.z_dim, self.n_classes = x_dim, z_dim, n_classes
        self.likelihood, self.arch, self.min_std = likelihood, arch, min_std
        c = n_classes
        out_mult = 2 if likelihood == "gaussian" else 1

        if arch == "mlp":
            self.enc = mlp(x_dim + c, hidden, depth, 2 * z_dim)
            self.dec = mlp(z_dim + c, hidden, depth, out_mult * x_dim)
        else:
            assert x_dim == 784 and likelihood == "bernoulli"
            self.enc_conv = nn.Sequential(
                nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(),   # 28 -> 14
                nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),  # 14 -> 7
                nn.Flatten(),
            )
            self.enc = mlp(64 * 7 * 7 + c, hidden, max(depth - 1, 0), 2 * z_dim)
            self.dec_fc = nn.Sequential(
                mlp(z_dim + c, hidden, max(depth - 1, 0), 64 * 7 * 7), nn.ReLU())
            self.dec_conv = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),  # 7 -> 14
                nn.ConvTranspose2d(32, 1, 4, 2, 1),              # 14 -> 28
            )

    # -- conditioning -------------------------------------------------------
    def _cond(self, h, y):
        if self.n_classes == 0:
            return h
        return torch.cat([h, F.one_hot(y, self.n_classes).to(h.dtype)], dim=-1)

    # -- encoder / decoder --------------------------------------------------
    def encode(self, x, y=None):
        """Returns posterior mean and log-variance, each (B, z_dim)."""
        h = x
        if self.arch == "conv":
            h = self.enc_conv(x.view(-1, 1, 28, 28))
        mu, logvar = self.enc(self._cond(h, y)).chunk(2, dim=-1)
        return mu, logvar.clamp(-12.0, 8.0)

    def decode(self, z, y=None):
        """Bernoulli: logits (B, x_dim). Gaussian: (mean, std)."""
        h = self._cond(z, y)
        if self.arch == "conv":
            out = self.dec_conv(self.dec_fc(h).view(-1, 64, 7, 7)).flatten(1)
        else:
            out = self.dec(h)
        if self.likelihood == "bernoulli":
            return out
        mean, raw = out.chunk(2, dim=-1)
        return mean, F.softplus(raw) + self.min_std

    def log_px_given_z(self, x, dec_out):
        """log p(x|z) summed over data dimensions -> (B,)."""
        if self.likelihood == "bernoulli":
            return -F.binary_cross_entropy_with_logits(
                dec_out, x, reduction="none").sum(-1)
        mean, std = dec_out
        return (-0.5 * ((x - mean) / std) ** 2 - torch.log(std)
                - 0.5 * LOG_2PI).sum(-1)

    def forward(self, x, y=None):
        mu, logvar = self.encode(x, y)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return z, mu, logvar, self.decode(z, y)

    # -- generation ---------------------------------------------------------
    @torch.no_grad()
    def sample(self, n, y=None, z=None, noise=True):
        """Draw ``n`` samples from p(z) p(x|z, y).

        Returns the decoder mean (Bernoulli probabilities or Gaussian means)
        when ``noise=False``; otherwise a draw from p(x|z) (Gaussian only;
        for Bernoulli, probabilities are returned since those are what is
        visualized).
        """
        dev = next(self.parameters()).device
        if z is None:
            z = torch.randn(n, self.z_dim, device=dev)
        if y is not None and not torch.is_tensor(y):
            y = torch.full((n,), int(y), dtype=torch.long, device=dev)
        out = self.decode(z, y)
        if self.likelihood == "bernoulli":
            return torch.sigmoid(out)
        mean, std = out
        return mean + std * torch.randn_like(mean) if noise else mean


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def kl_diag_gaussian(mu, logvar):
    """KL(q(z|x) || N(0, I)) per latent dimension -> (B, z_dim)."""
    return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)


def elbo_terms(model, x, y=None):
    """Single-sample ELBO pieces for a batch.

    Returns dict of per-example tensors: 'recon' = -log p(x|z) (B,),
    'kl' (B,), and 'kl_dim' (B, z_dim).
    """
    _, mu, logvar, dec_out = model(x, y)
    recon = -model.log_px_given_z(x, dec_out)
    kl_dim = kl_diag_gaussian(mu, logvar)
    return {"recon": recon, "kl": kl_dim.sum(-1), "kl_dim": kl_dim}


def vae_loss(model, x, y=None, beta=1.0):
    """Batch-mean of recon + beta * KL (= -ELBO when beta = 1)."""
    t = elbo_terms(model, x, y)
    loss = (t["recon"] + beta * t["kl"]).mean()
    return loss, t


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@torch.no_grad()
def iwae_nll(model, x, y=None, k=200, chunk=50):
    """Importance-weighted estimate of -log p(x) (Burda et al., 2016).

    log p(x) ~= logsumexp_i [log p(x|z_i) + log p(z_i) - log q(z_i|x)] - log k
    with z_i ~ q(z|x). Tighter than the ELBO and converges to log p(x) as
    k grows. Returns a (B,) tensor of per-example NLL in nats.
    """
    mu, logvar = model.encode(x, y)
    std = torch.exp(0.5 * logvar)
    B = x.shape[0]
    log_ws = []
    for start in range(0, k, chunk):
        kc = min(chunk, k - start)
        eps = torch.randn(kc, B, model.z_dim, device=x.device)
        z = mu + std * eps                                      # (kc, B, Z)
        zf = z.reshape(kc * B, -1)
        xf = x.unsqueeze(0).expand(kc, *x.shape).reshape(kc * B, -1)
        yf = None if y is None else y.repeat(kc)
        log_px = model.log_px_given_z(xf, model.decode(zf, yf)).view(kc, B)
        log_pz = (-0.5 * z.pow(2) - 0.5 * LOG_2PI).sum(-1)
        log_qz = (-0.5 * eps.pow(2) - 0.5 * logvar - 0.5 * LOG_2PI).sum(-1)
        log_ws.append(log_px + log_pz - log_qz)
    log_w = torch.cat(log_ws, 0)                                # (k, B)
    return -(torch.logsumexp(log_w, 0) - math.log(k))


@torch.no_grad()
def collect_posterior(model, batches):
    """Concatenate posterior means / log-variances / labels over a loader."""
    mus, lvs, ys = [], [], []
    for xb, yb in batches:
        mu, lv = model.encode(xb, yb if model.n_classes else None)
        mus.append(mu.cpu()); lvs.append(lv.cpu()); ys.append(yb.cpu())
    return torch.cat(mus), torch.cat(lvs), torch.cat(ys)


def active_units(mu, threshold=1e-2):
    """Burda et al. activity: Var_x(E_q[z_u|x]) > threshold.

    Args: mu (N, z_dim) posterior means over a dataset.
    Returns (number of active units, per-dim activity tensor).
    """
    act = mu.var(dim=0)
    return int((act > threshold).sum()), act


def mmd_rbf(a, b, bandwidths=(0.1, 0.2, 0.5, 1.0, 2.0)):
    """Unbiased squared MMD with a sum of RBF kernels (2D sample quality).

    a: (n, d), b: (m, d) tensors. Larger = sample distribution further from
    the data distribution; ~0 when they match.
    """
    a, b = a.double(), b.double()
    n, m = a.shape[0], b.shape[0]
    d_aa, d_bb, d_ab = torch.cdist(a, a) ** 2, torch.cdist(b, b) ** 2, torch.cdist(a, b) ** 2
    k_aa = k_bb = k_ab = 0.0
    for h in bandwidths:
        k_aa = k_aa + torch.exp(-d_aa / (2 * h * h))
        k_bb = k_bb + torch.exp(-d_bb / (2 * h * h))
        k_ab = k_ab + torch.exp(-d_ab / (2 * h * h))
    s_aa = (k_aa.sum() - k_aa.diagonal().sum()) / (n * (n - 1))
    s_bb = (k_bb.sum() - k_bb.diagonal().sum()) / (m * (m - 1))
    return float(s_aa + s_bb - 2 * k_ab.mean())


def count_params(model):
    return sum(p.numel() for p in model.parameters())
