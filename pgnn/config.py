from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _REPO_ROOT / "data"
OUTPUT_DIR = _REPO_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

LOGG_GRID = [3.15, 3.30, 3.45, 3.60, 3.75, 3.90]

PARAM_NAMES = ["teff", "logg", "logmdot", "rstar"]
PARAM_COLS = {"teff": "TEFF", "logg": "LOGG", "logmdot": "LOGMDOT", "rstar": "RADIUS"}

BATCH_SIZE = 2048
MAX_EPOCHS = 200
LR = 5e-4
GRAD_CLIP = 1.0
AUG_FACTOR = 4

SEEDS = [42, 123, 2026]

CONFIGS = {
    "baseline": dict(
        lambda_max=0.0, e_delay=0, e_ramp=0, w_domain=0.0,
        patience_lr=5, patience_es=10, variant="baseline",
    ),
    # PGNN variants: checkpoint is force-saved at ramp completion so the final model
    # is always post-physics, not the pre-ramp minimum that ES would otherwise restore.
    "pgnn_ft": dict(
        lambda_max=0.10, e_delay=0, e_ramp=10, w_domain=0.0,
        patience_lr=4, patience_es=8, variant="pgnn-ft",
        reset_after_ramp=True,
    ),
    "pgnn_scratch": dict(
        lambda_max=0.10, e_delay=10, e_ramp=20, w_domain=0.0,
        patience_lr=5, patience_es=10, variant="pgnn-scratch",
        reset_after_ramp=True,
    ),
    "pgnn_domain": dict(
        lambda_max=0.10, e_delay=10, e_ramp=20, w_domain=0.1,
        patience_lr=5, patience_es=10, variant="pgnn-domain",
        reset_after_ramp=True,
    ),
    # PGNN+obs: same as PGNN but adds L_phys on IACOB (184 stars) + UVES POP (12 stars)
    # after ramp completion. w_obs=0.01 normalized by lphys_ref → ~10% of lsup at convergence.
    "pgnn_ft_obs": dict(
        lambda_max=0.10, e_delay=0, e_ramp=10, w_domain=0.0, w_obs=0.01,
        patience_lr=4, patience_es=8, variant="pgnn-ft-obs",
        reset_after_ramp=True,
    ),
    "pgnn_scratch_obs": dict(
        lambda_max=0.10, e_delay=10, e_ramp=20, w_domain=0.0, w_obs=0.01,
        patience_lr=5, patience_es=10, variant="pgnn-scratch-obs",
        reset_after_ramp=True,
    ),
    "pgnn_domain_obs": dict(
        lambda_max=0.10, e_delay=10, e_ramp=20, w_domain=0.1, w_obs=0.01,
        patience_lr=5, patience_es=10, variant="pgnn-domain-obs",
        reset_after_ramp=True,
    ),
    # GradNorm: physics weight set adaptively (no fixed lambda). Physics active
    # from epoch 0 (warm-started from baseline); es_warmup=e_delay+e_ramp=10
    # marks the reset-after-ramp checkpoint. lambda_max is ignored when
    # use_gradnorm=True.
    "pgnn_ft_gn": dict(
        lambda_max=0.10, e_delay=0, e_ramp=10, w_domain=0.0,
        patience_lr=4, patience_es=8, variant="pgnn-ft-gn",
        reset_after_ramp=True, use_gradnorm=True, gradnorm_alpha=1.5,
    ),
    "pgnn_scratch_gn": dict(
        lambda_max=0.10, e_delay=10, e_ramp=10, w_domain=0.0,
        patience_lr=5, patience_es=10, variant="pgnn-scratch-gn",
        reset_after_ramp=True, use_gradnorm=True, gradnorm_alpha=1.5,
    ),
    "pgnn_domain_gn": dict(
        lambda_max=0.10, e_delay=10, e_ramp=10, w_domain=0.1,
        patience_lr=5, patience_es=10, variant="pgnn-domain-gn",
        reset_after_ramp=True, use_gradnorm=True, gradnorm_alpha=1.5,
    ),
}
