"""Gradient-norm imbalance table for L_sup vs L_phys (Tabla tab:grad).

Run: python scripts/grad_balance.py
"""
import math
import torch

from pgnn.data import load_all_data
from pgnn.model import PGNNModel
from pgnn.losses import loss_sup, PhysicsLoss, compute_lphys_ref
from pgnn.config import CHECKPOINT_DIR

device = torch.device("cpu")
torch.manual_seed(0)

data = load_all_data(augment=False)
scaler = data["scaler"]
phys = PhysicsLoss(scaler).to(device)
lphys_ref = compute_lphys_ref(data["labels_train_norm"], scaler, device)

batch = next(iter(data["train_loader"]))
X = batch["spectrum"].to(device)
Y = batch["labels_norm"].to(device)
y_true = {"teff": Y[:, 0:1], "logg": Y[:, 1:2], "logmdot": Y[:, 2:3], "rstar": Y[:, 3:4]}
print(f"batch={X.shape[0]}  lphys_ref={lphys_ref:.4f}\n")


def gvec(loss, params):
    gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat([g.detach().reshape(-1) for g in gs if g is not None])


def measure(model, tag):
    model.eval()
    shared = list(model.encoder.parameters())
    y = model(X)
    Lsup = loss_sup(y, y_true)
    Lphys = phys(y)
    gs = gvec(Lsup, shared)
    gp = gvec(Lphys, shared)
    gsup, gphys = gs.norm().item(), gp.norm().item()
    gphys_n = gphys / lphys_ref
    cos = torch.nn.functional.cosine_similarity(gs, gp, dim=0).item()
    print(f"[{tag}]  L_sup={Lsup.item():.4e}  L_phys={Lphys.item():.4e}")
    print(f"   ||g_sup||      = {gsup:.4e}")
    print(f"   ||g_phys_raw|| = {gphys:.4e}   ratio raw  g_phys/g_sup        = {gphys/gsup:7.1f}x")
    print(f"   ||g_phys/ref|| = {gphys_n:.4e}   ratio norm (g_phys/ref)/g_sup = {gphys_n/gsup:7.2f}x")
    print(f"   cos(g_sup, g_phys) = {cos:+.4f}\n")


measure(PGNNModel().to(device), "RANDOM INIT")

ck = CHECKPOINT_DIR / "baseline_aug_seed42.pt"
if ck.exists():
    m = PGNNModel().to(device)
    m.load_state_dict(torch.load(ck, map_location=device, weights_only=True))
    measure(m, "BASELINE CKPT (lambda-ramp onset)")
else:
    print(f"no checkpoint at {ck}")
