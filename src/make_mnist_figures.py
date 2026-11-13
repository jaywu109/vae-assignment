"""Figures, LaTeX tables and summary CSV for Part B (MNIST). Run from project root:
    .venv/bin/python src/make_mnist_figures.py
Works with any subset of finished runs; missing panels/rows are skipped with a warning."""
import os, json, glob, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

os.makedirs("figures/report", exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                     "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False})
COL = {"CVAE": "#009E73", "VAE": "#0072B2"}
ZS = [2, 4, 8, 16, 32, 64]
W = 6.8

def warn(msg):
    print("WARNING:", msg)

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/report/{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/report/%s.{pdf,png}" % name)

# ------------------------------------------------------------------ load
rows = []
for f in sorted(glob.glob("results/mnist/*.json")):
    d = json.load(open(f))
    rows.append(d)
runs = pd.DataFrame(rows)
print(f"loaded {len(runs)} runs")
if os.path.exists("results/mnist_sample_metrics.csv"):
    sm = pd.read_csv("results/mnist_sample_metrics.csv")[
        ["run_id", "acc_binary", "pred_entropy", "fid_binary", "fid_mean"]]
    runs = runs.merge(sm, on="run_id", how="left")
else:
    for c in ["acc_binary", "pred_entropy", "fid_binary", "fid_mean"]:
        runs[c] = np.nan
runs["gap"] = runs.test_neg_elbo - runs.train_neg_elbo
runs["best_ep1"] = runs.best_epoch + 1
runs["time_min"] = runs.train_time / 60
print("sample metrics available for %d runs" % runs.fid_binary.notna().sum())
try:
    FID_FLOOR = float(pd.read_csv("results/fid_reference.csv").iloc[0, 1])
except Exception:
    FID_FLOOR = None

METRICS = ["train_neg_elbo", "test_neg_elbo", "test_nll_iwae", "test_recon", "test_kl",
           "active_units", "gap", "acc_binary", "pred_entropy", "fid_binary", "fid_mean",
           "best_ep1", "time_min", "n_params"]

def agg(sub):
    """mean/std/n of METRICS over rows of sub."""
    out = {"n": len(sub)}
    for m in METRICS:
        v = sub[m].dropna()
        out[m] = v.mean() if len(v) else np.nan
        out[m + "_std"] = v.std(ddof=1) if len(v) > 1 else np.nan
    return out

def sel(model, batch=None, **kw):
    s = runs[runs.model == model]
    if batch is not None:
        s = s[s.batch.isin([batch] if isinstance(batch, str) else batch)]
    for k, v in kw.items():
        s = s[s[k] == v]
    return s

def by_z(model, batch):
    zs, A = [], []
    for z in ZS:
        s = sel(model, batch, z=z)
        if len(s):
            zs.append(z); A.append(agg(s))
    return np.array(zs), pd.DataFrame(A)

def line(ax, zs, A, m, color, ls="-", marker="o", label=None):
    if len(zs) == 0:
        return
    sd = A[m + "_std"].fillna(0).values
    ax.errorbar(zs, A[m].values, yerr=sd, color=color, ls=ls, marker=marker, ms=3.5,
                lw=1.3, capsize=2, label=label)

def zaxis(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ZS); ax.set_xticklabels([str(z) for z in ZS], fontsize=6.5)
    ax.minorticks_off()
    ax.set_xlabel("latent dim $z$")

cz, Ac = by_z("CVAE", "B1")
vz, Av = by_z("VAE", "B1u")
for name, zz in (("CVAE (B1)", cz), ("VAE (B1u)", vz)):
    miss = [z for z in ZS if z not in zz]
    if miss:
        warn(f"{name}: no runs for z={miss}")
    n = {z: len(sel(name.split()[0], "B1" if "B1)" in name else "B1u", z=z)) for z in zz}
    if name.startswith("CVAE") and any(v < 3 for v in n.values()):
        warn(f"CVAE B1 has fewer than 3 seeds at z={[z for z, v in n.items() if v < 3]}")

# ------------------------------------------------------------------ fig 1
fig, axs = plt.subplots(1, 4, figsize=(W, 2.2))
ax = axs[0]
for lab, zs, A in (("CVAE", cz, Ac), ("VAE", vz, Av)):
    line(ax, zs, A, "test_nll_iwae", COL[lab], "-", "o", f"{lab} IWAE NLL")
    line(ax, zs, A, "test_neg_elbo", COL[lab], "--", "s", f"{lab} $-$ELBO")
ax.set_ylabel("test (nats)"); ax.set_title("(a) likelihood"); zaxis(ax)
ax.legend(frameon=False, fontsize=6.5)
ax = axs[1]
ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
for lab, zs, A in (("CVAE", cz, Ac), ("VAE", vz, Av)):
    line(ax, zs, A, "test_recon", COL[lab], "-", "o", f"{lab} recon")
    line(ax2, zs, A, "test_kl", COL[lab], ":", "^", f"{lab} KL")
ax.set_ylabel("distortion: recon (solid)"); ax2.set_ylabel("rate: KL (dotted)")
ax.set_title("(b) rate-distortion"); zaxis(ax)
ax = axs[2]
for lab, zs, A in (("CVAE", cz, Ac), ("VAE", vz, Av)):
    line(ax, zs, A, "active_units", COL[lab], "-", "o", lab)
ax.plot(ZS, ZS, color="gray", ls=":", lw=1, label="$y=x$")
ax.set_yscale("log", base=2); ax.set_yticks(ZS); ax.set_yticklabels([str(z) for z in ZS])
ax.minorticks_off()
ax.set_ylabel("active units"); ax.set_title("(c) active units"); zaxis(ax)
ax.legend(frameon=False)
ax = axs[3]
for lab, zs, A in (("CVAE", cz, Ac), ("VAE", vz, Av)):
    line(ax, zs, A, "gap", COL[lab], "-", "o", lab)
ax.set_ylabel("test $-$ train $-$ELBO"); ax.set_title("(d) generalisation gap"); zaxis(ax)
ax.legend(frameon=False)
fig.tight_layout(w_pad=1.6)
save(fig, "mnist_latent_sweep")

# ------------------------------------------------------------------ fig 2
fig, axs = plt.subplots(1, 3, figsize=(W, 2.1))
ax = axs[0]
for lab, zs, A in (("CVAE", cz, Ac), ("VAE", vz, Av)):
    line(ax, zs, A, "fid_binary", COL[lab], "-", "o", lab)
if FID_FLOOR:
    ax.axhline(FID_FLOOR, color="gray", ls=":", lw=1, label="train vs test")
ax.set_yscale("log"); ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=[1, 2, 5]))
ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter()); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_ylabel("FID (binarised samples)")
ax.set_title("(a) FID, lower = better"); zaxis(ax); ax.legend(frameon=False)
ax = axs[1]
line(ax, cz, Ac, "acc_binary", COL["CVAE"], "-", "o", "CVAE")
ax.set_ylabel("label accuracy"); ax.set_title("(b) conditional accuracy"); zaxis(ax)
ax.set_ylim(top=1.005)
ax = axs[2]
line(ax, vz, Av, "pred_entropy", COL["VAE"], "-", "o", "VAE")
line(ax, cz, Ac, "pred_entropy", COL["CVAE"], "-", "o", "CVAE")
ax.axhline(np.log(10), color="gray", ls=":", lw=1, label="$\\ln 10$ (balanced)")
ax.set_ylabel("entropy of predicted classes"); ax.set_title("(c) class balance"); zaxis(ax)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.9), ncol=3, fontsize=6, handlelength=1.5, columnspacing=0.8)
fig.tight_layout(w_pad=0.8)
save(fig, "mnist_sample_quality")

# ------------------------------------------------------------------ tiles
def sample_npz(model, z, batch=None):
    s = sel(model, batch or ("B1" if model == "CVAE" else "B1u"), z=z, seed=0)
    if not len(s):
        return None
    p = f"results/samples/{s.iloc[0].run_id}.npz"
    return dict(np.load(p)) if os.path.exists(p) else None

def mosaic(imgs, nrows, ncols, sep=1, sepval=0.5):
    """imgs (nrows*ncols,784) -> image with thin separators."""
    t = imgs.reshape(nrows, ncols, 28, 28)
    H = nrows * 28 + (nrows - 1) * sep; Wd = ncols * 28 + (ncols - 1) * sep
    M = np.full((H, Wd), sepval, dtype=np.float32)
    for i in range(nrows):
        for j in range(ncols):
            M[i * (28 + sep): i * (28 + sep) + 28, j * (28 + sep): j * (28 + sep) + 28] = t[i, j]
    return M

def show(ax, M):
    ax.imshow(M, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

def blank(ax, msg):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(.5, .5, msg, ha="center", va="center", transform=ax.transAxes, color="gray", fontsize=8)

# ------------------------------------------------------------------ fig 3
zsel = [2, 8, 32, 64]
fig, axs = plt.subplots(2, 4, figsize=(W, 3.6))
for i, model in enumerate(("CVAE", "VAE")):
    for j, z in enumerate(zsel):
        ax = axs[i, j]
        S = sample_npz(model, z)
        if S is None:
            warn(f"samples_by_z: missing {model} z={z}")
            blank(ax, "missing")
        else:
            show(ax, mosaic(S["grid"], 10, 10))
        ax.set_title(f"{model}, $z$ = {z}", fontsize=9, color=COL[model])
fig.tight_layout(w_pad=0.5, h_pad=1.6)
save(fig, "mnist_samples_by_z")

# ------------------------------------------------------------------ fig 4
fig, axs = plt.subplots(1, 3, figsize=(W, 2.5))
Sv = sample_npz("VAE", 2); Sc = sample_npz("CVAE", 2)
panels = [("VAE (unconditional)", None if Sv is None or "manifold" not in Sv else Sv["manifold"], "VAE"),
          ("CVAE, label 3", None if Sc is None or "manifold" not in Sc else Sc["manifold"][0], "CVAE"),
          ("CVAE, label 8", None if Sc is None or "manifold" not in Sc else Sc["manifold"][1], "CVAE")]
for ax, (t, M, mod) in zip(axs, panels):
    if M is None:
        warn(f"manifold: missing '{t}'"); blank(ax, "missing")
    else:
        show(ax, mosaic(M, 15, 15, sep=0))
    ax.set_title(t, color=COL[mod])
    if M is not None:
        ax.set_xlabel("$z_1$ $\\rightarrow$"); ax.set_ylabel("$z_2$ $\\uparrow$")
fig.tight_layout(w_pad=0.5)
save(fig, "mnist_manifold")

# ------------------------------------------------------------------ fig 5
cands = [16, 2]
avail = sorted(z for z in ZS if sample_npz("CVAE", z) is not None)
if 16 not in avail and avail:
    cands[0] = max(avail)
    warn(f"style_swap: z=16 missing, falling back to z={cands[0]}")
cands = list(dict.fromkeys(cands))
fig, axs = plt.subplots(1, len(cands), figsize=(W, 3.0), squeeze=False)
for ax, z in zip(axs[0], cands):
    S = sample_npz("CVAE", z)
    if S is None or "style_swap" not in S:
        warn(f"style_swap: missing z={z}"); blank(ax, "missing"); continue
    gap = np.full((8, 1, 784), 0.0, dtype=np.float32)
    imgs = np.concatenate([S["orig"][:, None], S["recon"][:, None], S["style_swap"]], axis=1)  # 8,12,784
    M = mosaic(imgs.reshape(-1, 784), 8, 12, sep=1)
    show(ax, M)
    step = 29
    ax.set_xticks([14 + step * k for k in range(12)])
    ax.set_xticklabels(["orig", "rec"] + [str(k) for k in range(10)], fontsize=6.5)
    ax.tick_params(length=0, top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.set_title(f"CVAE $z$ = {z}: decoded with label $y$", fontsize=9, pad=14)
    ax.axvline(2 * step - 0.5, color="#D55E00", lw=0.8)
fig.tight_layout(w_pad=0.5)
save(fig, "mnist_style_swap")

# ------------------------------------------------------------------ fig 6
lc = [("CVAE", 2), ("CVAE", 16), ("CVAE", 64), ("VAE", 16)]
fig, axs = plt.subplots(1, 4, figsize=(W, 2.1))
for ax, (model, z) in zip(axs, lc):
    s = sel(model, "B1" if model == "CVAE" else "B1u", z=z, seed=0)
    ax.set_title(f"{model}, $z$ = {z}")
    if not len(s):
        warn(f"learning_curves: missing {model} z={z}"); blank(ax, "missing"); continue
    r = s.iloc[0]; c = r.curves
    ax.set_xlabel("epoch")
    tr, va = np.array(c["train_neg_elbo"]), np.array(c["val_neg_elbo"])
    ep = np.arange(1, len(tr) + 1)
    ax.plot(ep, tr, color=COL[model], lw=1.2, label="train")
    ax.plot(ep, va, color=COL[model], lw=1.2, ls="--", label="val")
    ax.axvline(r.best_epoch + 1, color="gray", ls=":", lw=0.8)
    tail = np.concatenate([tr[2:], va[2:]]) if len(tr) > 4 else np.concatenate([tr, va])
    pad = 0.15 * (tail.max() - tail.min() + 1e-6)
    ax.set_ylim(tail.min() - pad, tail.max() + pad)
    if model == "CVAE" and z == 2:
        ax.set_ylabel("$-$ELBO (nats)")
    ax.legend(frameon=False)
fig.tight_layout(w_pad=0.8)
save(fig, "mnist_learning_curves")

# ------------------------------------------------------------------ tables
def pm(m, s, dec=1, scale=1.0, bold=False):
    if pd.isna(m):
        return "--"
    t = f"{m*scale:.{dec}f}"
    if not pd.isna(s):
        t += f" $\\pm$ {s*scale:.{dec}f}"
    return f"\\textbf{{{t}}}" if bold else t

def write_tab(name, header, body, align):
    lines = ["\\begin{tabular}{" + align + "}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for b in body:
        lines.append("\\midrule" if b is None else " & ".join(b) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(f"results/{name}", "w").write("\n".join(lines) + "\n")
    print("wrote results/" + name)

# table 1
cfg = []
for model, batch in (("CVAE", "B1"), ("VAE", "B1u")):
    for z in ZS:
        s = sel(model, batch, z=z)
        cfg.append((model, z, agg(s) if len(s) else None))
best = min((a["test_nll_iwae"] for _, _, a in cfg if a), default=None)
body = []
for k, (model, z, a) in enumerate(cfg):
    if k and cfg[k - 1][0] != model:
        body.append(None)
    if a is None:
        body.append([model, str(z)] + ["--"] * 8); continue
    body.append([model, str(z),
                 pm(a["train_neg_elbo"], a["train_neg_elbo_std"]),
                 pm(a["test_neg_elbo"], a["test_neg_elbo_std"]),
                 pm(a["test_nll_iwae"], a["test_nll_iwae_std"], bold=a["test_nll_iwae"] == best),
                 pm(a["test_recon"], a["test_recon_std"]), pm(a["test_kl"], a["test_kl_std"]),
                 pm(a["active_units"], a["active_units_std"], dec=1),
                 pm(a["fid_binary"], a["fid_binary_std"]),
                 pm(a["acc_binary"], a["acc_binary_std"], scale=100) if model == "CVAE" else "--"])
write_tab("table_mnist_latent.tex",
          ["Model", "$z$", "Train $-$ELBO", "Test $-$ELBO", "Test IWAE", "Recon", "KL", "AU", "FID", "Acc (\\%)"],
          body, "llrrrrrrrr")

# table 2 (arch) -- baseline mlp h512 d2 z16 = B1 z=16 seeds 0,1
def arch_runs(arch, h, d):
    base = (arch == "mlp" and h == 512 and d == 2)
    s = sel("CVAE", "B1", z=16, seed=0).index.union(sel("CVAE", "B1", z=16, seed=1).index) if base else None
    return runs.loc[s] if base else sel("CVAE", "B2", arch=arch, h=h, d=d)

arch_cfgs = [("mlp", h, d) for h in (256, 512) for d in (1, 2, 3)] + [("conv", 512, 2)]
body, summ = [], []
for arch, h, d in arch_cfgs:
    s = arch_runs(arch, h, d)
    lab = f"MLP $h$={h}, $d$={d}" if arch == "mlp" else "Conv"
    if arch == "mlp" and h == 512 and d == 2:
        lab += " (base)"
    if not len(s):
        warn(f"arch table: missing {arch} h{h} d{d}"); body.append([lab] + ["--"] * 8); continue
    if len(s) < 2:
        warn(f"arch table: {arch} h{h} d{d} has only {len(s)} seed(s)")
    a = agg(s)
    summ.append(("B2", s, arch, h, d, 1e-3, 128, a))
    body.append([lab, f"{a['n_params']/1e6:.2f}M",
                 pm(a["test_nll_iwae"], a["test_nll_iwae_std"]), pm(a["test_neg_elbo"], a["test_neg_elbo_std"]),
                 pm(a["gap"], a["gap_std"]), pm(a["active_units"], a["active_units_std"], dec=1),
                 pm(a["fid_binary"], a["fid_binary_std"]), pm(a["best_ep1"], a["best_ep1_std"], dec=0),
                 pm(a["time_min"], a["time_min_std"])])
write_tab("table_mnist_arch.tex",
          ["Architecture", "\\#params", "Test IWAE", "Test $-$ELBO", "Gap", "AU", "FID", "Best ep.", "Time (min)"],
          body, "lrrrrrrrr")

# table 3 (optim)
base = sel("CVAE", "B1", z=16, seed=0)
opt = [("baseline", base)] + [(f"lr={lr}", sel("CVAE", "B3", lr=lr)) for lr in (3e-4, 3e-3)] + \
      [(f"bs={b}", sel("CVAE", "B3", bs=b)) for b in (64, 256)]
body = []
for lab, s in opt:
    if not len(s):
        warn(f"optim table: missing {lab}"); body.append(["--"] * 2 + ["--"] * 4); continue
    a = agg(s); r = s.iloc[0]
    summ.append(("B3", s, "mlp", 512, 2, r.lr, r.bs, a))
    body.append([f"{r.lr:g}", str(int(r.bs)), pm(a["test_nll_iwae"], None), pm(a["gap"], None),
                 f"{int(a['best_ep1'])}", pm(a["time_min"], None)])
write_tab("table_mnist_optim.tex", ["Learning rate", "Batch size", "Test IWAE", "Gap", "Best ep.", "Time (min)"],
          body, "rrrrrr")

# summary csv
out = []
def add(batch, s, arch, h, d, lr, bs, a):
    r = s.iloc[0]
    out.append({"batch": batch, "model": r.model, "arch": arch, "z": r.z, "h": h, "d": d, "lr": lr, "bs": bs, **a})
for model, batch in (("CVAE", "B1"), ("VAE", "B1u")):
    for z in ZS:
        s = sel(model, batch, z=z)
        if len(s):
            add(batch, s, "mlp", 512, 2, 1e-3, 128, agg(s))
for t in summ:
    add(*t)
pd.DataFrame(out).to_csv("results/mnist_summary.csv", index=False)
print("wrote results/mnist_summary.csv (%d rows)" % len(out))
