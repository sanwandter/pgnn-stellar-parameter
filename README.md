# Physics-Guided Neural Networks for Stellar Parameter Estimation

Estimation of massive-star parameters ($T_\mathrm{eff}$, $\log g$, $\log \dot{M}$, $R_*$) from $H_\alpha$ spectral profiles using physics-guided neural networks (PGNN). A 1D-CNN trained on the synthetic ISOSCELES grid (~33k spectra) is regularized with an algebraic mass-loss prescription (CNO, Figueroa-Tapia et al. 2026) as a physics-consistency loss, reducing the synthetic-to-observed domain gap when evaluated on real spectra (IACOB, UVES POP).

**Key results** (mean normalized MAE over $T_\mathrm{eff}$, $\log g$, $R_*$; 5 seeds):

- The physics loss **halves the out-of-distribution error** on IACOB (0.52 → 0.27) at a modest in-distribution cost, and eliminates physically inconsistent predictions.
- The OOD gain is robust to the physics weight $\lambda_\mathrm{phys}$ (sweep 0.05–0.50, range < 0.02).
- Adaptive gradient balancing (**GradNorm**) matches the OOD error of the best fixed $\lambda_\mathrm{phys}$ with ~33% lower in-distribution error, Pareto-dominating the fixed-weight sweep.

## Method

**Model.** A ResNet-1D encoder over the $H_\alpha$ profile feeds four independent regression heads, one per parameter. Outputs are min–max normalized; no hard output constraints.

**Data.** ISOSCELES synthetic grid (33,444 O-type spectra with labels), split 80/10/10. The training split is expanded ×4 with an augmentation pipeline emulating observational effects: sub-pixel Doppler shift, Gaussian broadening (rotational + instrumental proxy), veiling, continuum wobble, vertical offset, and correlated + white noise. Observed spectra (IACOB with partial literature labels; UVES POP, unlabeled) are used for evaluation only — no labeled observed data enters training.

**Loss.** The training objective composes up to four terms, depending on the configuration:

$$
\mathcal{L} \;=\; \mathcal{L}_\mathrm{sup}
\;+\; \lambda(e)\,\frac{\mathcal{L}_\mathrm{phys}}{\mathcal{L}_\mathrm{phys}^\mathrm{ref}}
\;+\; w_\mathrm{dom}\,\mathcal{L}_\mathrm{dom}
\;+\; w_\mathrm{obs}\left(\tilde{\mathcal{L}}_\mathrm{phys}^\mathrm{UVES} + \tilde{\mathcal{L}}_\mathrm{phys}^\mathrm{IACOB}\right)
$$

- **Supervised term** — mean MSE over the four normalized parameters:

$$
\mathcal{L}_\mathrm{sup} = \frac{1}{4}\sum_{p \in \{T_\mathrm{eff},\, \log g,\, \log\dot{M},\, R_*\}} \mathrm{MSE}\big(\hat{y}_p,\, y_p\big)
$$

- **Physics term** — predictions are denormalized to physical units and the predicted $\log\dot{M}$ is penalized against the value implied by the CNO prescription evaluated on the *model's own* $T_\mathrm{eff}$, $\log g$, $R_*$ predictions:

$$
\mathcal{L}_\mathrm{phys} = \mathrm{MSE}\big(\log\dot{M},\; \log\dot{M}_\mathrm{CNO}\big),
\qquad
\log\dot{M}_\mathrm{CNO} = -20.74 + 1.44\,\log_{10} R_* - 3.85\,\log g + 16.81\,\log_{10}\!\frac{T_\mathrm{eff}}{10^3\,\mathrm{K}}
$$

  It is normalized by $\mathcal{L}_\mathrm{phys}^\mathrm{ref}$, the residual of the true training labels under the same prescription, making $\lambda$ dimensionless. The weight follows a delayed linear ramp, $\lambda(e) = \lambda_\mathrm{max}\cdot\min\big(1,\,(e - e_\mathrm{delay})/e_\mathrm{ramp}\big)$ for $e \ge e_\mathrm{delay}$, so supervised convergence is established before the constraint activates.

- **Domain constraints** (`domain` variants) — quadratic penalties on physically impossible outputs, $\mathcal{L}_\mathrm{dom} = \overline{\mathrm{ReLU}(-R_*)^2} + \overline{\mathrm{ReLU}(\log\dot{M})^2}$ (negative radii, non-negative $\log\dot{M}$).

- **Observed-spectra physics** (`obs` variants) — the same normalized physics residual evaluated on the model's predictions for *unlabeled* observed spectra (UVES POP + IACOB), active only after the ramp completes. This injects real-spectra signal without using any observed labels.

- **GradNorm** (`gn` variants) — replaces the fixed $\lambda$ with adaptive task weights that balance the gradient norms of $\mathcal{L}_\mathrm{sup}$ and $\mathcal{L}_\mathrm{phys}$ on the shared encoder (Chen et al. 2018, $\alpha = 1.5$).

**Configurations.** All share the same augmented training data and differ only along controlled axes:

| Config | Initialization | Physics loss | Extras |
|---|---|---|---|
| `baseline` | random | — | supervised reference |
| `pgnn_scratch` | random | fixed $\lambda$ | physics effect in isolation |
| `pgnn_ft` | warm start from `baseline` | fixed $\lambda$ | physics + pre-trained features |
| `pgnn_domain` | random | fixed $\lambda$ | + domain constraints |
| `*_obs` | as above | fixed $\lambda$ | + physics on unlabeled observed spectra |
| `*_gn` | as above | GradNorm | adaptive weighting |

**Training.** Adam (lr $5\times10^{-4}$), batch 2048, mixed precision (physics loss kept in float32 for numerical stability), gradient clipping. LR scheduling and early stopping track the *supervised* validation loss only, so the $\lambda$ ramp cannot trigger premature stopping; PGNN runs re-checkpoint at ramp completion to guarantee the saved model is post-physics.

**Evaluation.** Three levels: (1) in-distribution MAE/RMSE on the synthetic test split; (2) distributional coherence on unlabeled UVES POP via Wasserstein-1 against the true ISOSCELES label distribution; (3) pointwise OOD error (MAE, bias) against IACOB literature labels — plus a label-free physical-plausibility check via out-of-range prediction rates.

## Repository layout

```
pgnn/               model, losses (supervised + physics), training, evaluation
run.py              full pipeline: all configs × seeds, or single runs / eval-only
scripts/            paper figures and analysis
data/               H-alpha profiles (ISOSCELES, IACOB, UVES POP) + scaler
outputs/            per-seed metrics (JSON), training logs, checkpoints
final_submission/   report LaTeX source, figures and final PDF
```

## Installation

Requires Python ≥ 3.10. A CUDA GPU is recommended for training (CPU works for `--eval-only`).

```bash
pip install -r requirements.txt
```

## Usage

```bash
python run.py                      # train all configs × seeds
python run.py --configs pgnn_ft --seeds 42
python run.py --eval-only          # evaluate existing checkpoints
python scripts/make_figs.py        # paper figures
```

