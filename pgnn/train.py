import os
import random
import time
import json

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler

import pgnn.config as cfg
from pgnn.model import PGNNModel
from pgnn.losses import (
    PhysicsLoss, loss_sup, domain_constraints,
    LambdaPhysScheduler, compute_lphys_ref, _split_labels_norm, GradNorm,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def train_model(
    config_name: str,
    config: dict,
    seed: int,
    data: dict,
    device: torch.device,
    baseline_ckpt: str | None = None,
) -> str:
    set_seed(seed)

    scaler_data = data["scaler"]
    train_loader = data["train_loader"]
    val_loader = data["val_loader"]

    model = PGNNModel().to(device)

    if baseline_ckpt is not None:
        model.load_state_dict(torch.load(baseline_ckpt, map_location=device, weights_only=True))
        print(f"  Loaded baseline weights from {baseline_ckpt}")

    optimizer = Adam(model.parameters(), lr=cfg.LR, betas=(0.9, 0.999))
    lr_scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=config["patience_lr"]
    )
    amp_scaler = GradScaler()

    phys = PhysicsLoss(scaler_data).to(device)
    lphys_ref = compute_lphys_ref(data["labels_train_norm"], scaler_data, device)
    print(f"  L_phys_ref = {lphys_ref:.6f}")

    # GradNorm requires float32 — float16 overflows L_phys.
    use_gradnorm = config.get("use_gradnorm", False)
    gradnorm = None
    shared_params = None
    if use_gradnorm:
        gradnorm = GradNorm(device, alpha=config.get("gradnorm_alpha", 1.5))
        shared_params = [p for p in model.encoder.parameters() if p.requires_grad]
        print(f"  GradNorm ON (alpha={config.get('gradnorm_alpha', 1.5)}, "
              f"{len(shared_params)} shared tensors)")

    w_obs = config.get("w_obs", 0.0)
    uvespop_x = None
    iacob_x = None
    if w_obs > 0:
        uvespop_batch = next(iter(data["uvespop_loader"]))
        uvespop_x = uvespop_batch["spectrum"].to(device)
        iacob_batch = next(iter(data["iacob_loader"]))
        iacob_x = iacob_batch["spectrum"].to(device)

    lambda_sched = LambdaPhysScheduler(
        config["lambda_max"], config["e_delay"], config["e_ramp"]
    )

    ckpt_path = str(cfg.CHECKPOINT_DIR / f"{config_name}_seed{seed}.pt")
    best_val_lsup = float("inf")
    epochs_no_improve = 0
    es_warmup = config["e_delay"] + config["e_ramp"]
    use_reset = config.get("reset_after_ramp", False)
    log_rows = []

    t0 = time.time()
    for epoch in range(cfg.MAX_EPOCHS):
        lam = lambda_sched(epoch)

        # ---- TRAIN --------------------------------------------------------
        model.train()
        train_lsup_sum = 0.0
        train_lphys_sum = 0.0
        train_loss_sum = 0.0
        train_n = 0
        last_w_phys = None   # last GradNorm physics weight this epoch
        last_gn_ratio = None # last weighted L_phys/L_sup grad ratio this epoch

        for batch in train_loader:
            x = batch["spectrum"].to(device, non_blocking=True)
            y_true = _split_labels_norm(
                batch["labels_norm"].to(device, non_blocking=True)
            )

            optimizer.zero_grad(set_to_none=True)

            if use_gradnorm:
                # Float32 path: GradNorm sets the L_sup/L_phys balance adaptively.
                y_pred = model(x)
                lsup = loss_sup(y_pred, y_true)
                lphys = phys(y_pred)
                lphys_norm = lphys / max(lphys_ref, 1e-12)
                if epoch >= config["e_delay"]:
                    w, _, gn_ratio = gradnorm.step([lsup, lphys_norm], shared_params)
                    loss = w[0] * lsup + w[1] * lphys_norm
                    last_w_phys = w[1].item()
                    last_gn_ratio = gn_ratio
                else:
                    loss = lsup
                if config["w_domain"] > 0:
                    loss = loss + config["w_domain"] * domain_constraints(y_pred, phys)
                if uvespop_x is not None and epoch > es_warmup and es_warmup > 0:
                    y_uvespop = model(uvespop_x)
                    y_iacob   = model(iacob_x)
                    lphys_uvespop = phys(y_uvespop) / max(lphys_ref, 1e-12)
                    lphys_iacob   = phys(y_iacob)   / max(lphys_ref, 1e-12)
                    loss = loss + w_obs * (lphys_uvespop + lphys_iacob)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.GRAD_CLIP)
                optimizer.step()
            else:
                with autocast():
                    y_pred = model(x)
                    lsup = loss_sup(y_pred, y_true)

                # PhysicsLoss outside autocast in float32: float16 can overflow logg_phys
                # making -3.85*logg_phys → -inf, and 0*inf = nan poisons even lam=0 runs.
                y_pred_f32 = {k: v.float() for k, v in y_pred.items()}
                lphys = phys(y_pred_f32)
                lphys_norm = lphys / max(lphys_ref, 1e-12)
                loss = lsup
                if lam > 0:
                    loss = loss + lam * lphys_norm
                if config["w_domain"] > 0:
                    loss = loss + config["w_domain"] * domain_constraints(y_pred_f32, phys)
                # obs physics fires post-ramp only — avoids disrupting early supervised convergence
                if uvespop_x is not None and epoch > es_warmup and es_warmup > 0:
                    with autocast():
                        y_uvespop = model(uvespop_x)
                        y_iacob   = model(iacob_x)
                    y_uvespop_f32 = {k: v.float() for k, v in y_uvespop.items()}
                    y_iacob_f32   = {k: v.float() for k, v in y_iacob.items()}
                    lphys_uvespop = phys(y_uvespop_f32) / max(lphys_ref, 1e-12)
                    lphys_iacob   = phys(y_iacob_f32)   / max(lphys_ref, 1e-12)
                    loss = loss + w_obs * (lphys_uvespop + lphys_iacob)

                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.GRAD_CLIP)
                amp_scaler.step(optimizer)
                amp_scaler.update()

            bs = x.size(0)
            train_loss_sum += loss.item() * bs
            train_lsup_sum += lsup.item() * bs
            train_lphys_sum += lphys.item() * bs
            train_n += bs

        # ---- VALIDATE -----------------------------------------------------
        model.eval()
        val_loss_sum = 0.0
        val_lsup_sum = 0.0
        val_lphys_sum = 0.0
        val_n = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch["spectrum"].to(device, non_blocking=True)
                y_true = _split_labels_norm(
                    batch["labels_norm"].to(device, non_blocking=True)
                )

                with autocast():
                    y_pred = model(x)
                    lsup = loss_sup(y_pred, y_true)

                y_pred_f32 = {k: v.float() for k, v in y_pred.items()}
                lphys = phys(y_pred_f32)
                lphys_norm = lphys / max(lphys_ref, 1e-12)
                loss = lsup
                if lam > 0:
                    loss = loss + lam * lphys_norm

                bs = x.size(0)
                val_loss_sum += loss.item() * bs
                val_lsup_sum += lsup.item() * bs
                val_lphys_sum += lphys.item() * bs
                val_n += bs

        val_loss = val_loss_sum / val_n
        val_lsup = val_lsup_sum / val_n

        # LR scheduler and ES both track val_lsup (not total loss),
        # so λ ramp doesn't trigger premature early stopping.
        lr_scheduler.step(val_lsup)

        log_rows.append({
            "epoch": epoch,
            "lambda": lam,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss_sum / train_n,
            "train_lsup": train_lsup_sum / train_n,
            "train_lphys": train_lphys_sum / train_n,
            "val_loss": val_loss,
            "val_lsup": val_lsup,
            "val_lphys": val_lphys_sum / val_n,
            "gn_w_phys": last_w_phys,
            "gn_ratio": last_gn_ratio,
        })

        should_break = False
        if val_lsup < best_val_lsup:
            best_val_lsup = val_lsup
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        elif epoch >= es_warmup:
            epochs_no_improve += 1
            if epochs_no_improve >= config["patience_es"]:
                should_break = True

        if use_reset and epoch == es_warmup and es_warmup > 0:
            best_val_lsup = val_lsup
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
            should_break = False
            print(f"  [Post-ramp checkpoint ep {epoch}] val_lsup={val_lsup:.6f}")

        if should_break:
            print(f"  Early stopping at epoch {epoch} "
                  f"(best_val_lsup={best_val_lsup:.6f})")
            break

        if epoch % 10 == 0 or epoch == cfg.MAX_EPOCHS - 1:
            elapsed = time.time() - t0
            print(
                f"  [{config_name} s{seed}] ep {epoch:3d} | "
                f"train {train_loss_sum/train_n:.5f} | "
                f"val {val_loss:.5f} (sup={val_lsup:.5f} "
                f"phys={val_lphys_sum/val_n:.5f}) | "
                f"lam={lam:.3f} lr={optimizer.param_groups[0]['lr']:.1e} | "
                + (f"gn[w_phys={last_w_phys:.4f} ratio={last_gn_ratio:.2f}x] | "
                   if use_gradnorm and last_w_phys is not None else "")
                + f"{elapsed:.0f}s"
            )

    log_path = cfg.OUTPUT_DIR / f"log_{config_name}_seed{seed}.json"
    with open(log_path, "w") as f:
        json.dump(log_rows, f, indent=1)

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    print(f"  Best val_lsup = {best_val_lsup:.6f}  ({ckpt_path})")
    return ckpt_path
