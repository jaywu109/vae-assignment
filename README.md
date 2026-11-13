# Variational Autoencoder experiments

CS 6956-001 Probabilistic Deep Learning — VAE / CVAE on 2D toy data and binarized MNIST.
The report is `report.tex` / `report.pdf`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Everything runs from the project root. MPS (Apple silicon) is used when available; no GPU is needed.

## Reproducing the results

```bash
# Part A: 2D toy data (8 Gaussians, pinwheel)
python src/run_toy.py                 # A1 + A2 sweeps -> results/toy_results.csv, toy_samples.npz
python src/run_toy.py --batches A3    # VAE with KL warm-up -> results/toy_results_A3.*
python src/make_toy_figures.py        # figures/report/toy_*, results/table_toy*.tex

# Part B: binarized MNIST (downloads MNIST to data/ on first use)
python src/run_mnist.py               # 40 runs -> results/mnist/*.json, results/ckpt/*.pt
#   (optionally in parallel: --shard {0,1,2} --nshards 3; finished runs are skipped)
python src/classifier.py              # CNN used for sample accuracy / FID -> results/classifier.pt
python src/eval_samples.py            # results/mnist_sample_metrics.csv, results/samples/*.npz
python src/aggregate_posterior.py     # results/mnist_aggregate_posterior.csv
python src/make_mnist_figures.py      # figures/report/mnist_*, results/table_mnist_*.tex

# Report
latexmk -pdf report.tex
```

Model checkpoints (`results/ckpt/`, `results/classifier.pt`) and the MNIST data are not tracked in git.

## Code

| File | Contents |
|---|---|
| `src/common.py` | `VAE` (MLP or conv, optional label conditioning, Bernoulli/Gaussian decoder), ELBO, IWAE, active units, MMD |
| `src/train.py` | Training loop (Adam, early stopping, optional KL warm-up) and evaluation |
| `src/data_toy.py`, `src/data_mnist.py` | Datasets |
| `src/run_toy.py`, `src/run_mnist.py` | Experiment sweeps |
| `src/classifier.py`, `src/eval_samples.py` | Sample-quality metrics (label accuracy, classifier-feature FID) |
| `src/aggregate_posterior.py` | Aggregate posterior vs. prior check |
| `src/make_toy_figures.py`, `src/make_mnist_figures.py` | Figures and LaTeX tables |
