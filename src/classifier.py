"""Small CNN classifier on statically binarized MNIST, used to score VAE samples.

Run `python src/classifier.py` to train (seed 0, early stopping on val acc) and save results/classifier.pt.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_mnist import get_mnist_tensors  # noqa: E402

PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'classifier.pt')


class CNN(nn.Module):
    """conv32-conv64-FC128(features)-FC10; inputs are (N,784) in [0,1]."""

    def __init__(self):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(1, 32, 3, padding=1), nn.Conv2d(32, 64, 3, padding=1)
        self.fc1, self.fc2 = nn.Linear(64 * 7 * 7, 128), nn.Linear(128, 10)
        self.drop = nn.Dropout(0.4)

    def features(self, x):
        """128-d penultimate activations (post-ReLU), (N,128)."""
        h = F.max_pool2d(F.relu(self.c1(x.view(-1, 1, 28, 28))), 2)
        h = F.max_pool2d(F.relu(self.c2(h)), 2)
        return F.relu(self.fc1(h.flatten(1)))

    def logits(self, x):
        return self.fc2(self.drop(self.features(x)))

    forward = logits


def load_classifier(dev):
    """Load the trained classifier in eval mode on `dev`."""
    m = CNN().to(dev)
    m.load_state_dict(torch.load(PATH, map_location=dev))
    return m.eval()


@torch.no_grad()
def accuracy(m, X, y, bs=1000):
    return sum((m.logits(X[i:i + bs]).argmax(1) == y[i:i + bs]).sum().item() for i in range(0, len(X), bs)) / len(X)


def train(dev, epochs=25, patience=6):
    torch.manual_seed(0)
    d = {k: (X.to(dev), y.to(dev)) for k, (X, y) in get_mnist_tensors().items()}
    m = CNN().to(dev)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, bad = 0, 0
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(50000, device=dev)
        for i in range(0, 50000, 128):
            j = perm[i:i + 128]
            loss = F.cross_entropy(m.logits(d['train'][0][j]), d['train'][1][j])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        m.eval()
        va = accuracy(m, *d['val'])
        print(f'epoch {ep} val_acc {va:.4f}', flush=True)
        if va > best:
            best, bad = va, 0
            torch.save(m.state_dict(), PATH)
        else:
            bad += 1
            if bad >= patience:
                break
    m = load_classifier(dev)
    print(f'best val acc {best:.4f}  test acc {accuracy(m, *d["test"]):.4f}')


if __name__ == '__main__':
    train(torch.device('mps'))
