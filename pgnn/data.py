import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

from pgnn.config import DATA_DIR, PARAM_COLS, PARAM_NAMES, AUG_FACTOR, BATCH_SIZE


def compute_scaler(labels_train: np.ndarray) -> dict:
    scaler = {}
    for i, name in enumerate(PARAM_NAMES):
        col = labels_train[:, i]
        scaler[name] = {"min": float(col.min()), "max": float(col.max())}
    return scaler


def normalize(values, smin, smax):
    return (values - smin) / (smax - smin)


def denormalize_np(values, smin, smax):
    return values * (smax - smin) + smin


def normalize_labels(labels: np.ndarray, scaler: dict) -> np.ndarray:
    normed = np.zeros_like(labels, dtype=np.float32)
    for i, name in enumerate(PARAM_NAMES):
        normed[:, i] = normalize(
            labels[:, i], scaler[name]["min"], scaler[name]["max"]
        )
    return normed


def augment_spectrum(spectrum: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    s = spectrum.flatten().copy()
    n = len(s)

    # 1. Sub-pixel Doppler shift (interpolated, edge-clamped)
    shift_amount = rng.uniform(-5.0, 5.0)
    from scipy.ndimage import shift as ndimage_shift
    s = ndimage_shift(s, shift_amount, mode="nearest")

    # 2. Gaussian broadening (rotational + instrumental)
    sigma = rng.uniform(0.0, 2.5)
    s = gaussian_filter1d(s, sigma)

    # 3. Veiling: blend with flat continuum at level 1
    f_v = rng.uniform(0.0, 0.2)
    s = s * (1.0 - f_v) + f_v

    # 4. Continuum wobble (degree-2 polynomial, +-2-3 %)
    x = np.linspace(-1, 1, n)
    c1 = rng.uniform(-0.08, 0.08)
    c2 = rng.uniform(-0.08, 0.08)
    s = s * (1.0 + c1 * x + c2 * x ** 2)

    # 5. Vertical offset
    s = s + rng.uniform(-0.03, 0.03)

    # 6. Composite noise
    base_level = rng.uniform(0.01, 0.05)
    corr_sigma = rng.uniform(3.5, 7.0)
    low_freq = gaussian_filter1d(
        rng.normal(0, base_level, n), corr_sigma
    ) * 2.0

    hf_level = rng.uniform(0.005, 0.015)
    hf_sigma = rng.uniform(0.4, 0.9)
    high_freq = gaussian_filter1d(rng.normal(0, hf_level, n), hf_sigma)

    s = s + low_freq + high_freq

    return s.astype(np.float32)


def precompute_augmentation(flux: np.ndarray, labels_norm: np.ndarray,
                            aug_factor: int, seed: int = 0) -> tuple:
    n = len(flux)
    rng = np.random.default_rng(seed)
    flux_aug = np.empty((n * aug_factor, flux.shape[1]), dtype=np.float32)
    labels_aug = np.tile(labels_norm, (aug_factor, 1))

    for copy_idx in range(aug_factor):
        offset = copy_idx * n
        for i in range(n):
            flux_aug[offset + i] = augment_spectrum(flux[i], rng)
        print(f"  Augmentation copy {copy_idx+1}/{aug_factor} done")

    return flux_aug, labels_aug


class TensorDictDataset(Dataset):
    def __init__(self, spectra: torch.Tensor, labels: torch.Tensor):
        self.spectra = spectra
        self.labels = labels

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        return {
            "spectrum": self.spectra[idx],
            "labels_norm": self.labels[idx],
        }


class ObservedDataset(Dataset):
    def __init__(self, flux, labels=None, meta=None):
        self.flux = torch.from_numpy(flux.astype(np.float32)).unsqueeze(1)
        self.labels = labels
        self.meta = meta

    def __len__(self):
        return len(self.flux)

    def __getitem__(self, idx):
        out = {"spectrum": self.flux[idx]}
        if self.labels is not None:
            out["labels"] = {k: torch.tensor(v[idx], dtype=torch.float32)
                             for k, v in self.labels.items()}
        if self.meta is not None:
            out["meta"] = {k: v[idx] for k, v in self.meta.items()}
        return out


def load_all_data(data_dir=None, aug_seed=0, augment=True):
    if data_dir is None:
        data_dir = DATA_DIR
    data_dir = Path(data_dir)

    flux_all = np.load(data_dir / "isosceles_halpha.npy")
    params = pd.read_csv(data_dir / "isosceles_labels.csv")

    labels_all = np.column_stack([
        params[PARAM_COLS["teff"]].values,
        params[PARAM_COLS["logg"]].values,
        params[PARAM_COLS["logmdot"]].values,
        params[PARAM_COLS["rstar"]].values,
    ])

    n = len(flux_all)
    rng = np.random.default_rng(42)
    indices = rng.permutation(n)

    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    flux_train, labels_train = flux_all[train_idx], labels_all[train_idx]
    flux_val, labels_val = flux_all[val_idx], labels_all[val_idx]
    flux_test, labels_test = flux_all[test_idx], labels_all[test_idx]

    scaler = compute_scaler(labels_train)
    with open(data_dir / "min_max_scaler.json", "w") as f:
        json.dump(scaler, f, indent=2)

    labels_train_norm = normalize_labels(labels_train, scaler)
    labels_val_norm = normalize_labels(labels_val, scaler)
    labels_test_norm = normalize_labels(labels_test, scaler)

    if augment:
        print("Pre-computing augmentation...")
        flux_aug, labels_aug = precompute_augmentation(
            flux_train, labels_train_norm, AUG_FACTOR, seed=aug_seed
        )
    else:
        print("Augmentation disabled.")
        flux_aug = flux_train.astype(np.float32)
        labels_aug = labels_train_norm

    # Convert to tensors (spectra get channel dim)
    train_spec = torch.from_numpy(flux_aug).unsqueeze(1)          # (N or N*4, 1, 200)
    train_labels = torch.from_numpy(labels_aug)                    # (N or N*4, 4)
    val_spec = torch.from_numpy(flux_val.astype(np.float32)).unsqueeze(1)
    val_labels = torch.from_numpy(labels_val_norm)
    test_spec = torch.from_numpy(flux_test.astype(np.float32)).unsqueeze(1)
    test_labels = torch.from_numpy(labels_test_norm)

    train_ds = TensorDictDataset(train_spec, train_labels)
    val_ds = TensorDictDataset(val_spec, val_labels)
    test_ds = TensorDictDataset(test_spec, test_labels)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             pin_memory=True, num_workers=0)

    uvespop_flux = np.load(data_dir / "uvespop_o_halpha.npy")
    uvespop_ds = ObservedDataset(uvespop_flux)
    uvespop_loader = DataLoader(uvespop_ds, batch_size=len(uvespop_flux), shuffle=False)

    iacob_raw = np.load(data_dir / "iacob_halpha.npy", allow_pickle=True)

    teff_i = iacob_raw["Teff"].astype(np.float64)
    logg_i = iacob_raw["logg"].astype(np.float64)
    r_i = iacob_raw["R"].astype(np.float64)

    mask_teff = (teff_i >= 30_000) & (teff_i <= 50_000)
    mask_logg = (logg_i >= 2.9) & (logg_i <= 4.3)
    has_r = ~np.isnan(r_i) & (r_i > 0)
    mask_r = (r_i >= 7) & (r_i <= 70)
    mask_ft26 = mask_teff & mask_logg & (mask_r | ~has_r)

    iacob = iacob_raw[mask_ft26]
    spectra_iacob = iacob["H_alpha"].astype(np.float64)
    has_rstar = (~np.isnan(iacob["R"].astype(np.float64))
                 & (iacob["R"].astype(np.float64) > 0))

    labels_iacob = {
        "teff": iacob["Teff"].astype(np.float64),
        "logg": iacob["logg"].astype(np.float64),
        "rstar": np.where(has_rstar, iacob["R"].astype(np.float64), np.nan),
    }
    meta_iacob = {
        "source": iacob["source"],
        "has_rstar": has_rstar,
        "star_name": iacob["star_name"],
    }

    iacob_ds = ObservedDataset(spectra_iacob, labels_iacob, meta_iacob)
    iacob_loader = DataLoader(iacob_ds, batch_size=len(iacob), shuffle=False)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "uvespop_loader": uvespop_loader,
        "iacob_loader": iacob_loader,
        "scaler": scaler,
        "labels_train_norm": labels_train_norm,
        "labels_test": labels_test,
        "labels_test_norm": labels_test_norm,
    }
