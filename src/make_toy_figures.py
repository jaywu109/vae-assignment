"""Figures and LaTeX tables for Part A (2D toy experiments)."""
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures/report", exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                     "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False})
COL = {"VAE": "#0072B2", "VAE+WU": "#E69F00", "CVAE": "#009E73"}  # Okabe-Ito
DS = ["8gaussians", "pinwheel"]
DSN = {"8gaussians": "8 Gaussians", "pinwheel": "Pinwheel"}

a = pd.read_csv("results/toy_results.csv")
if "kl_warmup" not in a.columns:
    a["kl_warmup"] = 0
a["kl_warmup"] = a["kl_warmup"].fillna(0)
b = pd.read_csv("results/toy_results_A3.csv")
df = pd.concat([a, b], ignore_index=True)
df["label"] = np.where((df.model == "VAE") & (df.kl_warmup > 0), "VAE+WU", df.model)
S = dict(np.load("results/toy_samples.npz", allow_pickle=True))
S3 = dict(np.load("results/toy_samples_A3.npz", allow_pickle=True))

def key(ds, lab, z, h=128, d=2, f="samples"):
    if lab == "VAE+WU":
        return S3[f"{ds}|VAE|z{z}|h{h}|d{d}|w100|{f}"]
    return S[f"{ds}|{lab}|z{z}|h{h}|d{d}|{f}"]

def row(ds, lab, z, seed=0, h=128, d=2):
    r = df[(df.dataset == ds) & (df.label == lab) & (df.z_dim == z) & (df.hidden == h) & (df.depth == d) & (df.seed == seed)]
    assert len(r) == 1, (ds, lab, z, len(r))
    return r.iloc[0]

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/report/{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

cls = plt.get_cmap("tab10")
# ---------------- 1. samples
cols = [("Test data", None, None), ("VAE, $z$=2", "VAE", 2), ("VAE+WU, $z$=2", "VAE+WU", 2),
        ("VAE+WU, $z$=8", "VAE+WU", 8), ("CVAE, $z$=2", "CVAE", 2)]
fig, axs = plt.subplots(2, 5, figsize=(6.8, 2.9))
for i, ds in enumerate(DS):
    tx, ty = S[f"{ds}|test_X"], S[f"{ds}|test_y"].astype(int)
    allp = [tx] + [key(ds, l, z) for _, l, z in cols[1:]]
    lim = max(np.abs(p).max() for p in allp) * 1.05
    lim = min(lim, 4.5)
    for j, (t, l, z) in enumerate(cols):
        ax = axs[i, j]
        if l is None:
            ax.scatter(*tx.T, c=[cls(k % 10) for k in ty], s=3, alpha=.6, lw=0)
        else:
            X = key(ds, l, z)
            if l == "CVAE":
                yy = key(ds, l, z, f="sample_y").astype(int)
                ax.scatter(*X.T, c=[cls(k % 10) for k in yy], s=3, alpha=.6, lw=0)
            else:
                ax.scatter(*X.T, c=COL[l], s=3, alpha=.5, lw=0)
            ax.text(.04, .04, f"MMD={row(ds, l, z).mmd:.3f}", transform=ax.transAxes, fontsize=7.5,
                    bbox=dict(fc="white", ec="none", alpha=.8, pad=1.2))
            ax.set_title(t, color=COL[l], fontsize=8.5) if i == 0 else None
        if l is None and i == 0: ax.set_title(t, fontsize=8.5)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(True)
        if j == 0: ax.set_ylabel(DSN[ds])
fig.subplots_adjust(wspace=.05, hspace=.08)
save(fig, "toy_samples")

# ---------------- aggregates
def agg(sub, cols_):
    g = sub.groupby(["dataset", "label", "z_dim"])[cols_]
    return g.mean(), g.std(ddof=1)

metrics = [("test_nll_iwae", "Test IWAE NLL"), ("active_units", "Active units"), ("mmd", "MMD$^2$")]
sw = df[(df.hidden == 128) & (df.depth == 2)]
m, s = agg(sw, [k for k, _ in metrics])
fig, axs = plt.subplots(2, 3, figsize=(6.8, 3.9), sharex=True)
off = {"VAE": -.04, "VAE+WU": 0, "CVAE": .04}
for i, ds in enumerate(DS):
    for j, (k, nm) in enumerate(metrics):
        ax = axs[i, j]
        for lab in COL:
            zs = sorted(sw[(sw.dataset == ds) & (sw.label == lab)].z_dim.unique())
            mu = np.array([m.loc[(ds, lab, z), k] for z in zs]); sd = np.array([s.loc[(ds, lab, z), k] for z in zs])
            x = np.array(zs) * 2 ** off[lab]
            ax.errorbar(x, mu, yerr=sd, color=COL[lab], marker="o", ms=3.5, lw=1.3, capsize=2, label=lab)
        ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels([1, 2, 4, 8])
        ax.minorticks_off()
        if k == "active_units": ax.axhline(2, color="gray", ls=":", lw=.8); ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8][:: 1]); ax.set_ylim(-1.3, 4.5); ax.set_yticks([0, 1, 2, 3, 4])
        if i == 0: ax.set_title(nm)
        if i == 1: ax.set_xlabel("latent dim $z$")
        if j == 0: ax.set_ylabel(DSN[ds])
        ax.grid(alpha=.25)
axs[0, 0].legend(frameon=False, loc="best")
fig.tight_layout(h_pad=.8, w_pad=.8)
save(fig, "toy_latent_sweep")

# ---------------- 3. KL dims
fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.2), sharey=False)
w = .27
for ax, ds in zip(axs, DS):
    for n, lab in enumerate(COL):
        kl = np.sort(np.array(json.loads(row(ds, lab, 8).kl_dim)))[::-1]
        ax.bar(np.arange(8) + (n - 1) * w, kl, w, color=COL[lab], label=lab)
    note = [f"{l}: all KL$\\approx$0" for l in COL if max(json.loads(row(ds, l, 8).kl_dim)) < 0.05]
    if note: ax.text(.97, .3, "\n".join(note), transform=ax.transAxes, ha="right", fontsize=8)
    ax.set_xticks(range(8)); ax.set_xticklabels(range(1, 9))
    ax.set_xlabel("latent dimension (sorted by KL)"); ax.set_title(DSN[ds] + ", $z$=8")
    ax.grid(axis="y", alpha=.25)
axs[0].set_ylabel("test KL (nats)"); axs[0].legend(frameon=False, loc="upper right")
fig.tight_layout(w_pad=1.2)
save(fig, "toy_kl_dims")

# ---------------- 4. latent space
ds = "pinwheel"; ty = S[f"{ds}|test_y"].astype(int)
fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.0))
for ax, (lab, t) in zip(axs, [("VAE+WU", "VAE+WU, $z$=2"), ("CVAE", "CVAE, $z$=2")]):
    mu = key(ds, lab, 2, f="post_mu")
    ax.scatter(*mu.T, c=[cls(k % 10) for k in ty], s=6, alpha=.7, lw=0)
    ax.set_title(t, color=COL[lab]); ax.set_xlabel("$\\mu_1$"); ax.set_ylabel("$\\mu_2$")
    ax.set_aspect("equal", adjustable="datalim")
fig.tight_layout(w_pad=1.2)
save(fig, "toy_latent_space")

# ---------------- 5. tables
def pm(mu, sd, nd): return f"${mu:.{nd}f}\\pm{sd:.{nd}f}$"
full_m, full_s = agg(sw, ["train_neg_elbo", "test_neg_elbo", "test_nll_iwae", "test_kl", "active_units", "mmd"])
lines = ["\\begin{tabular}{llrcccccc}", "\\toprule",
         "Dataset & Model & $z$ & Train $-$ELBO & Test $-$ELBO & Test IWAE NLL & KL & AU & MMD$^2$ \\\\", "\\midrule"]
for di, ds in enumerate(DS):
    best = min(full_m.loc[(ds, l, z), "test_nll_iwae"] for l in COL for z in [1, 2, 4, 8])
    first = True
    for lab in COL:
        for z in [1, 2, 4, 8]:
            mu, sd = full_m.loc[(ds, lab, z)], full_s.loc[(ds, lab, z)]
            c = [pm(mu[k], sd[k], 3) for k in ["train_neg_elbo", "test_neg_elbo"]]
            iw = pm(mu.test_nll_iwae, sd.test_nll_iwae, 3)
            if abs(mu.test_nll_iwae - best) < 1e-12: iw = f"$\\mathbf{{{mu.test_nll_iwae:.3f}\\pm{sd.test_nll_iwae:.3f}}}$"
            kl = pm(mu.test_kl, sd.test_kl, 2); au = pm(mu.active_units, sd.active_units, 1)
            mm = pm(mu.mmd, sd.mmd, 3)
            name = DSN[ds] if first else ""
            first = False
            lines.append(f"{name} & {lab.replace('+', '+')} & {z} & {c[0]} & {c[1]} & {iw} & {kl} & {au} & {mm} \\\\")
        if lab != "CVAE": lines.append("\\cmidrule{2-9}")
    if di == 0: lines.append("\\midrule")
lines += ["\\bottomrule", "\\end{tabular}"]
open("results/table_toy.tex", "w").write("\n".join(lines) + "\n")

ar = df[(df.dataset == "pinwheel") & (df.model == "CVAE") & (df.z_dim == 2)]
g = ar.groupby(["hidden", "depth"])[["test_nll_iwae", "active_units", "mmd"]]
am, asd = g.mean(), g.std(ddof=1)
L = ["\\begin{tabular}{rrccc}", "\\toprule", "Hidden & Depth & Test IWAE NLL & AU & MMD$^2$ \\\\", "\\midrule"]
for h in [32, 128, 512]:
    for d in [1, 2, 3]:
        mu, sd = am.loc[(h, d)], asd.loc[(h, d)]
        L.append(f"{h} & {d} & {pm(mu.test_nll_iwae, sd.test_nll_iwae, 3)} & {pm(mu.active_units, sd.active_units, 1)} & {pm(mu.mmd, sd.mmd, 3)} \\\\")
    if h != 512: L.append("\\midrule")
L += ["\\bottomrule", "\\end{tabular}"]
open("results/table_toy_arch.tex", "w").write("\n".join(L) + "\n")

pd.set_option("display.width", 200)
out = pd.concat([full_m.round(3).add_suffix("_m"), full_s.round(3).add_suffix("_sd")], axis=1)
print(out.to_string()); print(am.round(3).join(asd.round(3), rsuffix="_sd").to_string())
