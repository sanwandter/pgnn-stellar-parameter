"""% of IACOB predictions outside FT26 validity ranges, per config.

Run: python scripts/out_of_range.py
"""
import numpy as np
import torch

from pgnn.config import SEEDS, CHECKPOINT_DIR, PARAM_NAMES
from pgnn.data import load_all_data
from pgnn.model import PGNNModel

RANGES = {"teff": (30000.0, 50000.0), "logg": (2.9, 4.3), "rstar": (7.0, 70.0)}

REPORTED = {
    "baseline": "baseline",
    "pgnn_ft": "pgnn_ft_lam025",
    "pgnn_scratch": "pgnn_scratch_lam025",
    "pgnn_domain": "pgnn_domain_lam025",
    "pgnn_ft_obs": "pgnn_ft_obs_lam025",
    "pgnn_scratch_obs": "pgnn_scratch_obs_lam025",
    "pgnn_domain_obs": "pgnn_domain_obs_lam025",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = load_all_data(augment=False)
scaler = data["scaler"]
loader = data["iacob_loader"]


def denorm(val, k):
    return val * (scaler[k]["max"] - scaler[k]["min"]) + scaler[k]["min"]


def predict_iacob(ckpt):
    model = PGNNModel().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    preds = {k: [] for k in PARAM_NAMES}
    with torch.no_grad():
        for batch in loader:
            x = batch["spectrum"].to(device)
            y = model(x)
            for k in PARAM_NAMES:
                preds[k].append(y[k].cpu().numpy())
    return {k: denorm(np.concatenate(preds[k]).squeeze(), k) for k in PARAM_NAMES}


def pct_out(vals, lo, hi):
    return 100.0 * np.mean((vals < lo) | (vals > hi))


print(f"{'config':22s} {'Teff %out':>14s} {'logg %out':>14s} {'R* %out':>14s}")
for name, run in REPORTED.items():
    rows = {k: [] for k in RANGES}
    for seed in SEEDS:
        ckpt = CHECKPOINT_DIR / f"{run}_seed{seed}.pt"
        if not ckpt.exists():
            print(f"  MISSING {ckpt}")
            continue
        p = predict_iacob(ckpt)
        for k, (lo, hi) in RANGES.items():
            rows[k].append(pct_out(p[k], lo, hi))
    cells = []
    for k in ["teff", "logg", "rstar"]:
        m, s = np.mean(rows[k]), np.std(rows[k])
        cells.append(f"{m:.1f}$\\pm${s:.1f}")
    print(f"{name:22s} " + " ".join(f"{c:>14s}" for c in cells))
