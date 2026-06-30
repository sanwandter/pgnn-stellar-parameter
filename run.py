#!/usr/bin/env python3
"""PGNN+Halpha: full training & evaluation pipeline.

Usage:
    python run.py                              # all configs x 3 seeds
    python run.py --configs baseline           # single config
    python run.py --seeds 42                   # single seed
    python run.py --eval-only                  # skip training, evaluate checkpoints
    python run.py --lambda-max 0.20            # override lambda_max (grid search)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from pgnn.config import CONFIGS, SEEDS, CHECKPOINT_DIR, OUTPUT_DIR
from pgnn.data import load_all_data
from pgnn.model import PGNNModel
from pgnn.train import train_model
from pgnn.evaluate import run_full_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--lambda-max", type=float, default=None,
                        help="Override lambda_max for all non-baseline configs (grid search)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Suffix for results filename, e.g. 'lam020' → results_lam020.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    t0 = time.time()
    data = load_all_data()
    data_raw = None  # loaded lazily if a no_augment config is requested
    print(f"Data loaded in {time.time()-t0:.1f}s")

    scaler = data["scaler"]
    all_results = {}

    # Training order: baseline first (pgnn_ft depends on baseline_aug)
    config_order = []
    for c in args.configs:
        if c in ("baseline", "baseline_aug"):
            config_order.insert(0, c)
        else:
            config_order.append(c)

    for config_name in config_order:
        if config_name not in CONFIGS:
            print(f"Unknown config: {config_name}, skipping")
            continue

        config = dict(CONFIGS[config_name])  # copy so we can mutate

        # Select augmented vs raw data
        if config.get("no_augment"):
            if data_raw is None:
                print("Loading data (no augmentation)...")
                t0 = time.time()
                data_raw = load_all_data(augment=False)
                print(f"Raw data loaded in {time.time()-t0:.1f}s")
            active_data = data_raw
        else:
            active_data = data

        # Grid search: override lambda_max and suffix checkpoint name
        run_name = config_name
        if args.lambda_max is not None and config["lambda_max"] > 0:
            config["lambda_max"] = args.lambda_max
            run_name = f"{config_name}_lam{int(args.lambda_max * 100):03d}"

        all_results[run_name] = {}

        print(f"\n{'='*60}")
        print(f"Config: {run_name}  (lambda_max={config['lambda_max']})")
        print(f"{'='*60}")

        for seed in args.seeds:
            print(f"\n--- Seed {seed} ---")
            ckpt_path = str(CHECKPOINT_DIR / f"{run_name}_seed{seed}.pt")

            if not args.eval_only:
                baseline_ckpt = None
                if config["variant"].startswith("pgnn-ft"):
                    baseline_ckpt = str(CHECKPOINT_DIR / f"baseline_aug_seed{seed}.pt")
                    if not Path(baseline_ckpt).exists():
                        print(f"  ERROR: baseline checkpoint not found: {baseline_ckpt}")
                        print("  Train baseline first!")
                        continue

                ckpt_path = train_model(
                    run_name, config, seed, active_data, device,
                    baseline_ckpt=baseline_ckpt,
                )

            if not Path(ckpt_path).exists():
                print(f"  Checkpoint not found: {ckpt_path}, skipping eval")
                continue

            model = PGNNModel().to(device)
            model.load_state_dict(
                torch.load(ckpt_path, map_location=device, weights_only=True)
            )

            print("  Evaluating...")
            results = run_full_evaluation(model, active_data, scaler, device)
            all_results[run_name][f"seed_{seed}"] = results

            # Print summary
            iso = results["isosceles_test"]
            iac = results["iacob"]
            gap = results["gap"]
            print(f"  ISOSCELES test:")
            for k in ["teff", "logg", "logmdot", "rstar"]:
                print(f"    {k:8s}: MAE={iso[k]['mae']:.4f}  RMSE={iso[k]['rmse']:.4f}")
            print(f"  Wasserstein-1 (vs UVES POP O-type):")
            for k in ["teff", "logg", "logmdot", "rstar"]:
                print(f"    {k:8s}: W1={results['wasserstein'][k]['w1']:.4f}")
            print(f"  IACOB:")
            for k in ["teff", "logg", "rstar"]:
                if k in iac:
                    print(f"    {k:8s}: MAE={iac[k]['mae']:.4f}  "
                          f"RMSE={iac[k]['rmse']:.4f}  bias={iac[k]['bias']:+.4f}  "
                          f"n={iac[k]['n']}")
            print(f"  L_phys: test={results['lphys_test']:.6f}  "
                  f"uvespop={results['lphys_uvespop']:.6f}  "
                  f"iacob={results['lphys_iacob']:.6f}")
            print(f"  Gap (MAE_IACOB - MAE_synth):")
            for k, v in gap.items():
                print(f"    {k:8s}: {v:+.4f}")

    # Save all results
    fname = f"results_{args.tag}.json" if args.tag else "results.json"
    results_path = OUTPUT_DIR / fname
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
