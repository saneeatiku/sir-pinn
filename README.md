# Physics-Informed Neural Network for SIR Epidemic Parameter Recovery

A PINN that learns an SIR epidemic from **sparse, noisy, partially observed data**
(only the infectious fraction `I(t)` is observed) and recovers:

- the full trajectory, including the **never-observed** `S(t)` and `R(t)`,
- the transmission rate `β`, recovery rate `γ` and `R0 = β/γ`,
- a **forecast beyond the data window**.

It is compared against (a) the same network trained on data only and (b) classical
non-linear least squares with an ODE solver. Python, JAX, NumPy, SciPy, Matplotlib.

> **Data note:** all data are **synthetic**, generated from an SIR model with
> β = 0.5, γ = 0.2 per day (R0 = 2.5), I(0) = 1%, horizon 80 days. The same model class is used
> for fitting (a well-specified setting) and the initial condition is assumed known.
> Nothing here is a result on real surveillance data.

![Main fit](results/fig1_main_fit.png)

## Model

SIR equations (population fractions, S + I + R = 1):

    dS/dt = -β S I,     dI/dt = β S I - γ I,     dR/dt = γ I

- An MLP (3 hidden layers x 32 tanh units) maps scaled time τ = t/T to (S, I, R).
- The initial condition is a **hard constraint**: `u(τ) = u0 + τ · MLP(τ)`.
- β and γ are trainable scalars, parametrised as `exp(log β)`, `exp(log γ)` to stay positive.
- Loss = misfit on observed `I` + `w · mean squared ODE residual` at 256 collocation points over
  the **whole** horizon (including days with no data). Time derivatives use forward-mode
  autodiff (`jax.jvp`). The physics weight is ramped up over the first 15% of training.
- Adam (hand-written), cosine learning-rate decay, 40,000 steps; w = 100.
- Observation noise: multiplicative Gaussian, `I_obs = I (1 + noise · ε)`.

## Experiments (10 seeds per setting; median, IQR in `results/tables.md`)

Default setting: data on days 0-30 (25 random days), 10% noise. The epidemic peaks near day 16.
Hyper-parameters (w, steps) were chosen on separate tuning seeds (100-102), not the evaluation
seeds (0-9).

**1. Main example (seed 0):** β̂ = 0.483, γ̂ = 0.191, R̂0 = 2.53 (truth 0.5, 0.2, 2.5).

**2. How much of the epidemic is observed** (median relative error; "RMSE beyond" is the error of
`I(t)` after the last observation):

| Last observed day | PINN R0 err | ODE-LS R0 err | PINN growth err | ODE-LS growth err | PINN RMSE beyond | ODE-LS RMSE beyond | Data-only NN RMSE beyond |
|---|---|---|---|---|---|---|---|
| 10 | 0.628 | 0.192 | 0.043 | 0.022 | 0.118 | 0.039 | 0.877 |
| 15 | 0.035 | 0.035 | 0.014 | 0.012 | 0.007 | 0.007 | 0.888 |
| 20 | 0.007 | 0.010 | 0.014 | 0.013 | 0.005 | 0.004 | 0.305 |
| 30 | 0.018 | 0.015 | 0.025 | 0.014 | 0.000 | 0.000 | 0.272 |
| 50 | 0.028 | 0.022 | 0.020 | 0.017 | 0.000 | 0.000 | 0.017 |

**3. Noise level** (data to day 30): errors grow roughly in proportion to noise. At 5% / 10% /
20% / 30% noise the PINN's median R0 error is 0.9% / 1.8% / 3.7% / 5.5% (ODE-LS: 0.7% / 1.5% /
3.1% / 4.8%).

**4. Number of observations:** 8 / 15 / 25 / 50 points give PINN median R0 errors of
4.1% / 1.8% / 1.8% / 0.9% (ODE-LS: 3.6% / 1.8% / 1.5% / 0.7%).

Figures: `results/fig2_window.png`, `fig3_noise.png`, `fig4_nobs.png`; per-seed numbers in
`results/results.json`.

## What the results say

- A data-only network fits the observed window but **cannot forecast**: with data to day 30 its
  median error beyond the window is about 0.27 (as a fraction of the population) versus about
  0.000 for the PINN. The ODE constraint is what makes the forecast possible.
- The PINN **matches but does not beat** classical ODE fitting on this well-specified problem.
  The classical fit was equal or slightly better on growth-rate error at most settings.
  The PINN's advantage would be flexibility (unknown terms, time-varying β), not tested here.
- **Early data cannot identify β and γ separately.** With only 10 days observed, the growth rate
  β - γ is recovered well (4% error) but R0 is not (PINN median error 63%; ODE-LS 19% with a
  6%-159% interquartile range). Early-outbreak R0 estimates need prior information or
  uncertainty quantification.
- The PINN was under-trained at 12,000 steps in the hardest case and needed 40,000 steps to match
  the classical fit, so PINN results depend on the optimiser budget.

## Limitations

- Synthetic data from the same model class; known initial condition; deterministic dynamics.
- One true parameter set; no real surveillance data; no uncertainty quantification.
- Physics weight and training length tuned on only 3 seeds.

## Possible extensions

Time-varying β(t) to model interventions, a latent neural-ODE version, uncertainty via ensembles
or Bayesian methods, unknown initial conditions, and real data.

## Reproduce

    pip install -r requirements.txt
    python run_experiments.py --quick   # 3 seeds, a few minutes
    python run_experiments.py           # full run, about 30-40 minutes on one CPU core

`sir_pinn.py` contains the model, training, baselines and metrics; `run_experiments.py` produces
all figures and tables.

## Author and acknowledgement

Abdullahi Sani Atiku, MSc AI & ML, University of Birmingham. This project was developed with
the assistance of Claude (Anthropic) for code generation and experiment design; I reviewed and
am responsible for the content.
