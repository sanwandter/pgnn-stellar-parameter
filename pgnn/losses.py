import torch
import torch.nn as nn
import torch.nn.functional as F

from pgnn.config import PARAM_NAMES


class PhysicsLoss(nn.Module):
    """CNO-prescription residual evaluated in physical space."""

    def __init__(self, scaler: dict):
        super().__init__()
        for name in PARAM_NAMES:
            self.register_buffer(
                f"{name}_min", torch.tensor(scaler[name]["min"], dtype=torch.float32)
            )
            self.register_buffer(
                f"{name}_max", torch.tensor(scaler[name]["max"], dtype=torch.float32)
            )

    def denorm(self, x_norm, name):
        smin = getattr(self, f"{name}_min")
        smax = getattr(self, f"{name}_max")
        return x_norm * (smax - smin) + smin

    def forward(self, y_norm: dict) -> torch.Tensor:
        teff = self.denorm(y_norm["teff"], "teff")
        logg = self.denorm(y_norm["logg"], "logg")
        logmdot = self.denorm(y_norm["logmdot"], "logmdot")
        rstar = self.denorm(y_norm["rstar"], "rstar")

        rstar_safe = torch.clamp(rstar, min=1e-3)
        teff_safe = torch.clamp(teff, min=1.0)

        logmdot_cno = (
            -20.74
            + 1.44 * torch.log10(rstar_safe)
            - 3.85 * logg
            + 16.81 * torch.log10(teff_safe / 1000.0)
        )
        return F.mse_loss(logmdot, logmdot_cno)


def loss_sup(y_pred: dict, y_true: dict) -> torch.Tensor:
    total = 0.0
    for name in PARAM_NAMES:
        total = total + F.mse_loss(y_pred[name], y_true[name])
    return total / len(PARAM_NAMES)


def domain_constraints(y_pred_norm: dict, phys: PhysicsLoss) -> torch.Tensor:
    rstar = phys.denorm(y_pred_norm["rstar"], "rstar")
    logmdot = phys.denorm(y_pred_norm["logmdot"], "logmdot")
    return torch.mean(torch.relu(-rstar) ** 2) + torch.mean(torch.relu(logmdot) ** 2)


class LambdaPhysScheduler:
    def __init__(self, lambda_max: float, e_delay: int, e_ramp: int):
        self.lambda_max = lambda_max
        self.e_delay = e_delay
        self.e_ramp = e_ramp

    def __call__(self, epoch: int) -> float:
        if self.lambda_max == 0:
            return 0.0
        if epoch < self.e_delay:
            return 0.0
        if self.e_ramp == 0:
            return self.lambda_max
        progress = min(1.0, (epoch - self.e_delay) / self.e_ramp)
        return self.lambda_max * progress


class GradNorm:
    """Chen et al. 2018 adaptive task weighting for {L_sup, L_phys}."""

    def __init__(self, device, alpha=1.5, lr=0.025, n_tasks=2):
        self.alpha = alpha
        self.n = n_tasks
        self.theta = torch.zeros(n_tasks, device=device, requires_grad=True)
        self.opt = torch.optim.Adam([self.theta], lr=lr)
        self.L0 = None  # initial per-task losses, set on first step

    def weights(self) -> torch.Tensor:
        return torch.softmax(self.theta, dim=0) * self.n

    def step(self, losses, shared_params):
        w = self.weights()  # carries grad w.r.t. theta
        raw = []
        for Li in losses:
            grads = torch.autograd.grad(Li, shared_params, retain_graph=True,
                                        allow_unused=True)
            sq = sum((g.detach() ** 2).sum() for g in grads if g is not None)
            raw.append(torch.sqrt(sq + 1e-12))
        raw = torch.stack(raw)                 # detached raw grad norms
        gw = w * raw                           # weighted norms (grad via w)

        lt = torch.stack([Li.detach() for Li in losses]).clamp_min(1e-8)
        if self.L0 is None:
            self.L0 = lt.clone()
        rate = lt / self.L0
        rel = rate / rate.mean()               # inverse training rate
        target = (gw.mean() * rel ** self.alpha).detach()

        l_grad = (gw - target).abs().sum()
        self.opt.zero_grad()
        l_grad.backward()
        self.opt.step()

        with torch.no_grad():
            w_new = self.weights()
            ratio = ((w_new[1] * raw[1]) / (w_new[0] * raw[0] + 1e-12)).item()
        return w_new.detach(), raw.detach(), ratio


def compute_lphys_ref(labels_train_norm, scaler, device) -> float:
    phys = PhysicsLoss(scaler).to(device)
    t = torch.tensor(labels_train_norm, dtype=torch.float32, device=device)
    y = {
        "teff": t[:, 0:1],
        "logg": t[:, 1:2],
        "logmdot": t[:, 2:3],
        "rstar": t[:, 3:4],
    }
    with torch.no_grad():
        ref = phys(y)
    return ref.item()


def _split_labels_norm(labels_norm: torch.Tensor) -> dict:
    return {
        "teff": labels_norm[:, 0:1],
        "logg": labels_norm[:, 1:2],
        "logmdot": labels_norm[:, 2:3],
        "rstar": labels_norm[:, 3:4],
    }
