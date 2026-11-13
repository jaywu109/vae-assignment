"""Training loop with validation-based early stopping for (C)VAEs.

``train_batches`` / ``val_batches`` are any re-iterable objects that yield
``(x, y)`` tensor pairs already on the model's device (e.g. the
``TensorBatcher`` in data_mnist.py or ``make_batches`` below).
"""
import copy
import time

import torch

from common import iwae_nll, vae_loss


class make_batches:
    """Minimal re-iterable batcher over in-memory tensors."""

    def __init__(self, X, y, batch_size, shuffle=False, seed=0):
        self.X, self.y, self.bs, self.shuffle = X, y, batch_size, shuffle
        self.g = torch.Generator().manual_seed(seed)

    def __len__(self):
        return (len(self.X) + self.bs - 1) // self.bs

    def __iter__(self):
        n = len(self.X)
        idx = (torch.randperm(n, generator=self.g) if self.shuffle
               else torch.arange(n)).to(self.X.device)
        for s in range(0, n, self.bs):
            j = idx[s:s + self.bs]
            yield self.X[j], self.y[j]


def _y(model, yb):
    return yb if model.n_classes else None


@torch.no_grad()
def evaluate(model, batches, iwae_k=0):
    """Mean per-example recon (-log p(x|z)), KL, -ELBO, and optional IWAE NLL."""
    model.eval()
    tot = {"recon": 0.0, "kl": 0.0, "nll_iwae": 0.0}
    kl_dim, n = 0.0, 0
    for xb, yb in batches:
        _, t = vae_loss(model, xb, _y(model, yb))
        tot["recon"] += float(t["recon"].sum())
        tot["kl"] += float(t["kl"].sum())
        kl_dim = kl_dim + t["kl_dim"].sum(0).cpu()
        if iwae_k:
            tot["nll_iwae"] += float(iwae_nll(model, xb, _y(model, yb), k=iwae_k).sum())
        n += xb.shape[0]
    out = {k: v / n for k, v in tot.items()}
    out["neg_elbo"] = out["recon"] + out["kl"]
    out["kl_dim"] = (kl_dim / n).tolist()
    if not iwae_k:
        out.pop("nll_iwae")
    return out


def train_vae(model, train_batches, val_batches, epochs=100, lr=1e-3,
              beta=1.0, kl_warmup=0, patience=20, verbose=0):
    """Adam training on recon + beta_t * KL, early-stopped on validation -ELBO.

    ``kl_warmup`` linearly anneals beta from 0 to ``beta`` over that many
    epochs (0 = off). Validation is always the true -ELBO (beta = 1), so
    checkpoints are comparable across settings. Restores the best weights
    and returns a history dict.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = {k: [] for k in ("train_neg_elbo", "train_recon", "train_kl",
                            "val_neg_elbo", "val_recon", "val_kl", "time")}
    best, best_state, best_epoch, bad = float("inf"), None, -1, 0
    for ep in range(epochs):
        t0 = time.time()
        b = beta * min(1.0, (ep + 1) / kl_warmup) if kl_warmup else beta
        model.train()
        s_rec = s_kl = n = 0.0
        for xb, yb in train_batches:
            loss, t = vae_loss(model, xb, _y(model, yb), beta=b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            s_rec += float(t["recon"].detach().sum()); s_kl += float(t["kl"].detach().sum())
            n += xb.shape[0]
        v = evaluate(model, val_batches)
        for k, val in (("train_recon", s_rec / n), ("train_kl", s_kl / n),
                       ("train_neg_elbo", (s_rec + s_kl) / n),
                       ("val_recon", v["recon"]), ("val_kl", v["kl"]),
                       ("val_neg_elbo", v["neg_elbo"]),
                       ("time", time.time() - t0)):
            hist[k].append(val)
        if verbose and (ep % verbose == 0 or ep == epochs - 1):
            print(f"ep {ep:3d}  beta {b:.2f}  train {hist['train_neg_elbo'][-1]:.3f}"
                  f"  val {v['neg_elbo']:.3f}  (rec {v['recon']:.3f} kl {v['kl']:.3f})")
        # Only start early stopping once the KL warm-up has finished.
        if ep + 1 >= kl_warmup:
            if v["neg_elbo"] < best - 1e-4:
                best, best_epoch, bad = v["neg_elbo"], ep, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
                if patience and bad >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    hist["best_epoch"] = best_epoch
    hist["best_val_neg_elbo"] = best
    return hist
