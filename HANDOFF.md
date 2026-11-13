# Handoff: HW (VAE) — CS 6956-001 Probabilistic Deep Learning

Last updated: 2026-11-13. Project folder: `~/Random/vae_assignment` (git on
`main`, committed locally; no GitHub remote yet).

## Current status (latest)
- All 40 MNIST runs done; `eval_samples.py`, `aggregate_posterior.py`,
  `make_mnist_figures.py` rerun; figures checked (layout fixes applied).
- `report.tex` complete (MNIST section + Discussion written, no TODOs),
  compiles to 8 pages (longer than the ~6-page target).
- README has the reproduce commands. `results/ckpt/` is gitignored.
- Remaining: optional final review / trimming; create GitHub repo and push
  (ask the user for repo name and private/public).

## Assignment (summary)
Implement a VAE. Start with an easy-to-visualize 2D task, then scale to a
more interesting dataset (a conditional VAE is likely needed). Compare
architectures (width, depth, etc.) and standard hyperparameters, and
**explicitly compare different latent-space sizes and discuss the effects**.
Analysis must go beyond train/test error and **examine generated samples**.
Report format follows HW-1 (MDN). The final HW-1 report.tex is in
`~/Downloads/CS_6956_001Probabilistic_Deep_Learning.zip` (10pt article,
0.85in margins, ~4 pages, authors: Dai-Jie Wu (U1591617), Enyu Zhao
(U1591984); sections: Introduction / Methods and Experimental Setup /
Synthetic Experiment / Real-Data Experiment / Discussion and Conclusion /
Use of generative AI).

## Environment
- `.venv/` (Python 3.13, torch 2.15, torchvision 0.29, numpy, scipy, pandas,
  matplotlib). Run everything from the project root with `.venv/bin/python`.
- Apple M5, MPS works. Measured: MLP CVAE ≈ 1.2 s/epoch on MPS (1.7 s CPU);
  conv ≈ 4.4 s/epoch on MPS (22 s CPU). **No GPU needed.**
- MNIST is downloaded to `data/MNIST/`, statically binarized cache in
  `data/mnist_bin.pt` (both gitignored).

## Code (src/)
| File | What | Author |
|---|---|---|
| `common.py` | `VAE` class (MLP or conv; conditional via one-hot label on encoder+decoder input; Bernoulli or Gaussian decoder with learned std floor 1e-2), `elbo_terms`, `vae_loss` (β), `iwae_nll` (verified vs exact numeric integration, error < 0.002 nats), `active_units` (Burda: Var_x E[z|x] > 0.01), `mmd_rbf`, `collect_posterior` | Opus |
| `train.py` | `train_vae` (Adam, early stop on val −ELBO, optional linear KL warm-up), `evaluate` (recon, KL, −ELBO, KL per dim, optional IWAE) , `make_batches` | Opus |
| `data_toy.py` | `get_toy('8gaussians'|'moons'|'pinwheel', n=6000, seed=0)` → 70/15/15 split, standardized on train | Sonnet |
| `data_mnist.py` | `get_mnist_tensors()` → 50k/10k/10k binarized; `TensorBatcher` | Sonnet |
| `run_toy.py` | Part A sweeps → `results/toy_results.csv`, `results/toy_samples.npz`; `--batches A3` → `*_A3.*` | Opus |
| `run_mnist.py` | Part B sweeps (sharded, resumable) → `results/mnist/<run_id>.json`, `results/ckpt/<run_id>.pt` | Opus |
| `make_toy_figures.py` | Part A figures + tables (DONE) | Sonnet |
| `make_mnist_figures.py` | **in progress (Sonnet agent)** — MNIST figures/tables; rerun after sweep: `.venv/bin/python src/make_mnist_figures.py` | Sonnet |
| `classifier.py`, `eval_samples.py` | DONE (classifier test acc 98.87%; FID train-vs-test floor 3.4; FID uses 128-d classifier features, only comparable within this study). Rerun `eval_samples.py` after the sweep — it only adds new runs. — MNIST CNN classifier (`results/classifier.pt`) and sample metrics → `results/mnist_sample_metrics.csv`, `results/samples/*.npz`, `results/fid_reference.csv` | Sonnet |
| `make_toy_figures.py` | **in progress (Sonnet agent)** — Part A figures in `figures/report/` and `results/table_toy*.tex` | Sonnet |

## Status
### Part A — 2D toy (DONE: 96 runs, 3 seeds each)
- A1: {8gaussians, pinwheel} × {VAE, CVAE} × z ∈ {1,2,4,8}, MLP h128 d2.
- A2: pinwheel CVAE z=2, hidden {32,128,512} × depth {1,2,3}.
- A3: unconditional VAE with 100-epoch KL warm-up, both datasets, z ∈ {1,2,4,8}.

Key findings (test IWAE NLL in nats, standardized space):
1. **Posterior collapse in the plain VAE** (Gaussian decoder with learned
   variance): many runs have all KL ≈ 0 and become a single Gaussian
   (−ELBO ≈ 2.84). It gets *worse and more erratic as z grows*
   (8gaussians z=8: all 3 seeds collapse; pinwheel z≥2 collapses).
2. **KL warm-up fixes it**: with z ≥ 2, active units saturate at exactly
   **2 = data dimension**, for any z; NLL ≈ 1.3–1.5. z = 1 is too small
   (NLL ≈ 2.1, MMD much worse) → a latent bottleneck below intrinsic dim.
3. **CVAE**: on 8gaussians it uses **0 latent dims** (the label explains
   the cluster; residual is isotropic Gaussian the decoder handles alone),
   NLL ≈ −1.06 for every z. On pinwheel AU *grows* with z (1, 2, 2.67, 3
   for z = 1, 2, 4, 8) while NLL is flat beyond z = 2 (≈ −0.64) — the extra
   active dims carry little KL (≈0.2) and don't improve likelihood.
4. Architecture (A2, pinwheel CVAE z=2, test IWAE): width matters, depth
   barely (h32: −0.44…−0.52; h128/h512: −0.59…−0.66).
5. Caveat: the Burda activity metric (Var_x E[z|x] > 0.01) can flag a unit
   as "active" while its KL ≈ 0 (8gaussians VAE z=8 seed 0: AU=2, total KL
   0.02). Report KL per dim alongside AU.

Part A figures/tables are DONE: `src/make_toy_figures.py` →
`figures/report/toy_{samples,latent_sweep,kl_dims,latent_space}.{pdf,png}`,
`results/table_toy.tex`, `results/table_toy_arch.tex`.

### Part B — MNIST (RUNNING when this was written)
Started 3 background processes:
`.venv/bin/python src/run_mnist.py --shard {0,1,2} --nshards 3`
(logs: `results/mnist_log_{0,1,2}.txt`). 40 runs total, est. 20–40 min.
- B1: CVAE MLP h512 d2, z ∈ {2,4,8,16,32,64}, seeds 0–2.
- B1u: unconditional VAE, same z, seed 0.
- B2: CVAE z=16, MLP h {256,512} × d {1,2,3} + conv, seeds 0–1.
- B3: CVAE z=16, lr {3e-4, 3e-3}, batch {64, 256}, seed 0.
- Each JSON has: train/test −ELBO, recon, KL, test IWAE NLL (k=200),
  active units, per-dim activity and KL, learning curves, #params.
- Max 100 epochs, early stop patience 10 on val −ELBO.

If interrupted, just rerun the same command — finished runs are skipped.
Check progress: `ls results/mnist/*.json | wc -l` (target 40).

## Report draft (Opus) — `report.tex`, compiles cleanly (`latexmk -pdf report.tex`)
- Done: Introduction, Methods (model, ELBO, CVAE, warm-up, datasets,
  protocol, IWAE/AU/MMD/classifier-accuracy/FID metrics), full **2D
  Experiments** section with figs toy_samples / toy_latent_sweep /
  toy_kl_dims, Table (table_toy.tex), and four discussion paragraphs
  (collapse grows with d_z; warm-up → AU = 2; CVAE needs no latent on
  8 Gaussians; architecture). Bibliography (5 refs) and Gen-AI disclosure.
- TODO (red `\todo{}` markers): MNIST section, Discussion & Conclusion,
  decide whether to include `table_toy_arch.tex`. Target ~6 pages.

## MNIST figure script — DONE (works on any subset; ~3 s)
After the sweep finishes (40 JSONs), run in this order:
```
.venv/bin/python src/eval_samples.py        # adds sample metrics for new runs
.venv/bin/python src/make_mnist_figures.py  # regenerates all MNIST figs/tables
```
mnist_latent_sweep is 1x4: (a) IWAE NLL solid / −ELBO dashed, (b) recon
(left axis) & KL (right axis), (c) AU with y=x line, (d) test−train gap.
mnist_sample_quality: FID (log-y, floor 3.45 dotted), sample accuracy,
predicted-class entropy. Style swap uses CVAE z=16 (stacked with z=2).
- Original spec, for reference: from
  `results/mnist/*.json`, `results/mnist_sample_metrics.csv` and
  `results/samples/*.npz` make `figures/report/mnist_{latent_sweep,
  sample_quality,samples_by_z,manifold,style_swap,learning_curves}.{pdf,png}`
  and `results/table_mnist_{latent,arch,optim}.tex` + `results/mnist_summary.csv`.
  (Note: the B2/B3 baseline cell = B1 z=16 runs.) Colors: CVAE #009E73,
  VAE #0072B2.
- npz layouts: `grid` (100,784) class-major for CVAE (columns share z);
  `style_swap` (8,10,784); `recon`, `orig` (8,784); z=2 runs have
  `manifold` (CVAE (2,225,784) for labels 3 and 8; VAE (225,784)),
  row-major, row = z2 descending.

## Remaining steps
1. Wait for Part B to finish (40 JSONs), then run `eval_samples.py` and
   `make_mnist_figures.py` (commands above). Look at every PNG.
2. (done) MNIST figure/table script.
3. **(Opus)** Interpret results; write report.tex using the HW-1 template
   (copy its preamble). Discussion points: latent size trade-off
   (rate vs distortion, active units, sample quality vs likelihood),
   posterior collapse & warm-up, why CVAE uses fewer latent dims,
   overfitting gap, architecture effects.
4. **(Sonnet)** LaTeX layout + `latexmk -pdf report.tex`; update README with
   the reproduce commands. Fill in the Gen-AI disclosure (as in HW-1).
5. **(Opus)** Final review of report vs assignment requirements.
6. Commit, create a GitHub repo (ask user: name, private/public), push.

## Model split (user's preference)
Simple/mechanical work → Sonnet 5.5 subagents; core math, experiment design,
interpretation and final review → Opus 5.5.
