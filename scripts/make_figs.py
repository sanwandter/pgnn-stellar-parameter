"""Generate the two result figures for the paper into final_submission/.

  fig_iacob_scatter.png : predicted vs literature (IACOB) for Teff, log g, R*,
                          Baseline vs PGNN-ft (seed 42) with 1:1 line. Shows the
                          OOD bias correction visually.
  fig_oor.png           : out-of-range rate (% of IACOB predictions outside FT26
                          ranges) per parameter, Baseline vs PGNN-ft (mean+-std,
                          3 seeds).

Run:
    python scripts/make_figs.py
"""
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pgnn.config import SEEDS, CHECKPOINT_DIR, PARAM_NAMES
from pgnn.data import load_all_data
from pgnn.model import PGNNModel

OUT = Path(__file__).resolve().parent.parent / "final_submission"
RANGES = {"teff": (30000.0, 50000.0), "logg": (2.9, 4.3), "rstar": (7.0, 70.0)}
C_BASE, C_PGNN = "#c0392b", "#2471a3"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = load_all_data(augment=False)
scaler = data["scaler"]
loader = data["iacob_loader"]


def denorm(val, k):
    return val * (scaler[k]["max"] - scaler[k]["min"]) + scaler[k]["min"]


def predict_iacob(run, seed):
    model = PGNNModel().to(device)
    model.load_state_dict(torch.load(
        CHECKPOINT_DIR / f"{run}_seed{seed}.pt", map_location=device, weights_only=True))
    model.eval()
    preds = {k: [] for k in PARAM_NAMES}
    with torch.no_grad():
        for batch in loader:
            y = model(batch["spectrum"].to(device))
            for k in PARAM_NAMES:
                preds[k].append(y[k].cpu().numpy())
    return {k: denorm(np.concatenate(preds[k]).squeeze(), k) for k in PARAM_NAMES}


# true labels (physical units) + R* mask
b = next(iter(loader))
true = {"teff": b["labels"]["teff"].numpy(),
        "logg": b["labels"]["logg"].numpy(),
        "rstar": b["labels"]["rstar"].numpy()}
mask_r = np.array(b["meta"]["has_rstar"])

# ---------------------------------------------------------------- scatter
pb = predict_iacob("baseline", 42)
pp = predict_iacob("pgnn_ft_lam025", 42)

panels = [("teff", r"$T_\mathrm{eff}$ [K]", None),
          ("logg", r"$\log g$", (2.9, 4.3)),
          ("rstar", r"$R_*$ [$R_\odot$]", (7, 70))]
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
for ax, (k, lab, rng) in zip(axes, panels):
    m = mask_r if k == "rstar" else np.ones_like(true[k], dtype=bool)
    t = true[k][m]
    ax.scatter(t, pb[k][m], s=14, c=C_BASE, alpha=0.55, label="Baseline", edgecolors="none")
    ax.scatter(t, pp[k][m], s=14, c=C_PGNN, alpha=0.75, label="PGNN (ft)", edgecolors="none")
    lo = min(t.min(), pb[k][m].min(), pp[k][m].min())
    hi = max(t.max(), pb[k][m].max(), pp[k][m].max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.7)
    if rng:
        ax.axhspan(rng[0], rng[1], color="green", alpha=0.06)
    if k == "rstar":
        ax.set_ylim(top=30)  # zoom in: predictions/labels live below 30 R_sun
    ax.set_xlabel(f"{lab} (literatura)")
    ax.set_ylabel(f"{lab} (predicho)")
    ax.set_title(lab)
axes[0].legend(loc="upper left", fontsize=8, framealpha=0.9)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_iacob_scatter.png", dpi=150, bbox_inches="tight")
print("wrote fig_iacob_scatter.png")

# ---------------------------------------------------------------- out-of-range bars
def pct_out(vals, lo, hi):
    return 100.0 * np.mean((vals < lo) | (vals > hi))

oor = {"baseline": {}, "pgnn_ft_lam025": {}}
for run in oor:
    acc = {k: [] for k in RANGES}
    for s in SEEDS:
        p = predict_iacob(run, s)
        for k, (lo, hi) in RANGES.items():
            acc[k].append(pct_out(p[k], lo, hi))
    oor[run] = {k: (np.mean(v), np.std(v)) for k, v in acc.items()}

ks = ["teff", "logg", "rstar"]
labels = [r"$T_\mathrm{eff}$", r"$\log g$", r"$R_*$"]
x = np.arange(len(ks)); w = 0.36
fig2, ax = plt.subplots(figsize=(5.2, 3.4))
bm = [oor["baseline"][k][0] for k in ks]; em = [oor["baseline"][k][1] for k in ks]
pm = [oor["pgnn_ft_lam025"][k][0] for k in ks]; ep = [oor["pgnn_ft_lam025"][k][1] for k in ks]
ax.bar(x - w/2, bm, w, yerr=em, capsize=3, color=C_BASE, label="Baseline")
ax.bar(x + w/2, pm, w, yerr=ep, capsize=3, color=C_PGNN, label="PGNN (ft)")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("\\% predicciones fuera de rango FT26")
ax.legend(fontsize=9)
fig2.tight_layout()
fig2.savefig(f"{OUT}/fig_oor.png", dpi=150, bbox_inches="tight")
print("wrote fig_oor.png")
