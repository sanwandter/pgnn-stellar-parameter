import numpy as np
import torch
from scipy.stats import wasserstein_distance

from pgnn.config import PARAM_NAMES, LOGG_GRID
from pgnn.model import PGNNModel, snap_logg
from pgnn.losses import PhysicsLoss, _split_labels_norm


def _denorm_np(val, smin, smax):
    return val * (smax - smin) + smin


def _predictions(model, loader, scaler, device) -> dict:
    model.eval()
    all_norm = {k: [] for k in PARAM_NAMES}

    with torch.no_grad():
        for batch in loader:
            x = batch["spectrum"].to(device, non_blocking=True)
            y_pred = model(x)
            for k in PARAM_NAMES:
                all_norm[k].append(y_pred[k].cpu().numpy())

    preds_norm = {k: np.concatenate(v).squeeze() for k, v in all_norm.items()}
    preds_phys = {}
    for k in PARAM_NAMES:
        preds_phys[k] = _denorm_np(
            preds_norm[k], scaler[k]["min"], scaler[k]["max"]
        )
    return preds_phys, preds_norm


def _snap_logg_np(logg_vals):
    grid = np.array(LOGG_GRID)
    diffs = np.abs(grid[None, :] - logg_vals[:, None])
    return grid[diffs.argmin(axis=1)]


def evaluate_isosceles(model, test_loader, labels_test, scaler, device) -> dict:
    preds, _ = _predictions(model, test_loader, scaler, device)
    results = {}
    param_order = ["teff", "logg", "logmdot", "rstar"]
    for i, k in enumerate(param_order):
        true = labels_test[:, i]
        pred = preds[k]
        if k == "logg":
            # ISOSCELES log g is discrete by design: snap the continuous prediction
            # to the nearest grid node for in-distribution eval. NOT applied to the
            # IACOB OOD path (real stars have continuous log g).
            pred = _snap_logg_np(pred)
        mae = float(np.mean(np.abs(pred - true)))
        rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
        results[k] = {"mae": mae, "rmse": rmse}
    return results


def evaluate_wasserstein(model, test_loader, uvespop_loader, labels_test, scaler, device) -> dict:
    preds_synth, _ = _predictions(model, test_loader, scaler, device)
    preds_obs, _ = _predictions(model, uvespop_loader, scaler, device)

    param_order = ["teff", "logg", "logmdot", "rstar"]
    true_synth = {k: labels_test[:, i] for i, k in enumerate(param_order)}

    results = {}
    for k in PARAM_NAMES:
        w1_pred = float(wasserstein_distance(preds_synth[k], preds_obs[k]))
        w1_true = float(wasserstein_distance(true_synth[k], preds_obs[k]))
        results[k] = {"w1": w1_pred, "w1_vs_true": w1_true}
    return results


def evaluate_iacob(model, iacob_loader, scaler, device) -> dict:
    preds, _ = _predictions(model, iacob_loader, scaler, device)

    batch = next(iter(iacob_loader))
    labels = batch["labels"]
    meta = batch["meta"]
    has_rstar = np.array(meta["has_rstar"])
    sources = np.array(meta["source"])

    teff_true = labels["teff"].numpy()
    logg_true = labels["logg"].numpy()
    rstar_true = labels["rstar"].numpy()

    results = {}

    # Teff — all 184
    pred_teff = preds["teff"]
    diff_teff = pred_teff - teff_true
    results["teff"] = {
        "mae": float(np.mean(np.abs(diff_teff))),
        "rmse": float(np.sqrt(np.mean(diff_teff ** 2))),
        "bias": float(np.mean(diff_teff)),
        "n": int(len(teff_true)),
    }

    # logg — all 184, no snap (raw continuous prediction)
    pred_logg = preds["logg"]
    diff_logg = pred_logg - logg_true
    results["logg"] = {
        "mae": float(np.mean(np.abs(diff_logg))),
        "rmse": float(np.sqrt(np.mean(diff_logg ** 2))),
        "bias": float(np.mean(diff_logg)),
        "n": int(len(logg_true)),
    }

    # R* — only 148 with R* available
    mask_r = has_rstar
    if mask_r.sum() > 0:
        pred_r = preds["rstar"][mask_r]
        true_r = rstar_true[mask_r]
        diff_r = pred_r - true_r
        results["rstar"] = {
            "mae": float(np.mean(np.abs(diff_r))),
            "rmse": float(np.sqrt(np.mean(diff_r ** 2))),
            "bias": float(np.mean(diff_r)),
            "n": int(mask_r.sum()),
        }

    return results


def compute_lphys_on_loader(model, loader, scaler, device) -> float:
    phys = PhysicsLoss(scaler).to(device)
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["spectrum"].to(device, non_blocking=True)
            y_pred = model(x)
            lp = phys(y_pred)
            bs = x.size(0)
            total += lp.item() * bs
            n += bs
    return total / max(n, 1)


def run_full_evaluation(model, data, scaler, device) -> dict:
    iso = evaluate_isosceles(
        model, data["test_loader"], data["labels_test"], scaler, device
    )
    w1 = evaluate_wasserstein(
        model, data["test_loader"], data["uvespop_loader"],
        data["labels_test"], scaler, device
    )
    iac = evaluate_iacob(model, data["iacob_loader"], scaler, device)

    lphys_test = compute_lphys_on_loader(
        model, data["test_loader"], scaler, device
    )
    lphys_uvespop = compute_lphys_on_loader(
        model, data["uvespop_loader"], scaler, device
    )
    lphys_iacob = compute_lphys_on_loader(
        model, data["iacob_loader"], scaler, device
    )

    gap = {}
    for k in ["teff", "logg", "rstar"]:
        if k in iac:
            gap[k] = iac[k]["mae"] - iso[k]["mae"]

    return {
        "isosceles_test": iso,
        "wasserstein": w1,
        "iacob": iac,
        "lphys_test": lphys_test,
        "lphys_uvespop": lphys_uvespop,
        "lphys_iacob": lphys_iacob,
        "gap": gap,
    }
