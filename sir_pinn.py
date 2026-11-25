"""
sir_pinn.py -- Physics-informed neural network (PINN) for the SIR epidemic model.

Task (inverse problem): given sparse, noisy, PARTIALLY observed surveillance data
(only the infectious fraction I(t) is observed), recover
  * the full epidemic trajectory (S, I, R), including the unobserved compartments,
  * the transmission rate beta and recovery rate gamma (hence R0 = beta / gamma),
  * a forecast beyond the observation window.

SIR model (population fractions, S + I + R = 1, time in days):
    dS/dt = -beta S I
    dI/dt =  beta S I - gamma I
    dR/dt =  gamma I

Method: a small MLP maps (scaled) time to (S, I, R). The initial condition is enforced
exactly (hard constraint). beta and gamma are trainable scalars (log-parametrised so they
stay positive). The loss is
    data misfit on observed I  +  w * mean squared ODE residual at collocation points
where the residual is computed with forward-mode autodiff (jvp) over the WHOLE horizon,
including times where no data are available.

Baselines: (a) the same network trained on data only (no physics),
           (b) classical non-linear least squares with an ODE solver.

Dependencies: jax, numpy, scipy.   Units: days, population fractions.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import jvp, random, vmap
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# ----------------------------------------------------------------------
# 1. Ground truth and synthetic surveillance data
# ----------------------------------------------------------------------
TRUE = dict(beta=0.5, gamma=0.2, I0=0.01, T=80.0)      # R0 = 2.5


def sir_rhs(t, y, beta, gamma):
    S, I, R = y
    return [-beta * S * I, beta * S * I - gamma * I, gamma * I]


def simulate(beta, gamma, I0, T, t_eval):
    """Reference solution with a high-accuracy ODE solver. Returns array (len(t_eval), 3)."""
    sol = solve_ivp(sir_rhs, (0.0, T), [1.0 - I0, I0, 0.0], t_eval=t_eval,
                    args=(beta, gamma), rtol=1e-10, atol=1e-12)
    return sol.y.T


def make_observations(t_max_obs, n_obs, rel_noise, seed, truth=TRUE):
    """Noisy observations of I only, at n_obs random days in (0, t_max_obs].

    Noise is multiplicative Gaussian: I_obs = I * (1 + rel_noise * eps), a simple model
    for reporting error that scales with the number of cases.
    """
    rng = np.random.default_rng(seed)
    t_obs = np.sort(rng.uniform(0.0, t_max_obs, n_obs))
    I_true = simulate(truth["beta"], truth["gamma"], truth["I0"], truth["T"], t_obs)[:, 1]
    I_obs = np.clip(I_true * (1 + rel_noise * rng.standard_normal(n_obs)), 1e-6, None)
    return t_obs, I_obs


# ----------------------------------------------------------------------
# 2. Network and PINN definition
# ----------------------------------------------------------------------
def init_mlp(key, sizes):
    keys = random.split(key, len(sizes) - 1)
    return [(random.normal(k, (m, n)) * jnp.sqrt(2.0 / (m + n)), jnp.zeros(n))
            for k, m, n in zip(keys, sizes[:-1], sizes[1:])]


def mlp(layers, x):
    for W, b in layers[:-1]:
        x = jnp.tanh(x @ W + b)
    W, b = layers[-1]
    return x @ W + b


def state(net, tau, ic):
    """(S, I, R) at scaled time tau = t / T in [0, 1]; initial condition is exact."""
    return ic + tau * mlp(net, jnp.atleast_1d(tau))


def ode_residual(params, tau, ic, T):
    """Residual of the SIR equations (per day) at scaled time tau, via forward-mode autodiff."""
    f = lambda s: state(params["net"], s, ic)
    u, du = jvp(f, (tau,), (jnp.ones_like(tau),))
    S, I, _ = u
    beta, gamma = jnp.exp(params["log_beta"]), jnp.exp(params["log_gamma"])
    rhs = jnp.array([-beta * S * I, beta * S * I - gamma * I, gamma * I])
    return du / T - rhs


def loss_fn(params, tau_obs, I_obs, tau_col, ic, T, w_phys, scale):
    I_pred = vmap(lambda s: state(params["net"], s, ic)[1])(tau_obs)
    l_data = jnp.mean(((I_pred - I_obs) / scale) ** 2)
    r = vmap(lambda s: ode_residual(params, s, ic, T))(tau_col)
    l_phys = jnp.mean((r / (0.2 * scale)) ** 2)           # residual in units of 0.2*peak-I per day
    return l_data + w_phys * l_phys, (l_data, l_phys)


# ----------------------------------------------------------------------
# 3. Training (Adam, hand-written, compiled with lax.scan)
# ----------------------------------------------------------------------
def train(t_obs, I_obs, physics=True, seed=0, n_steps=40000, lr=2e-3, lr_param_mult=10.0,
          w_phys=100.0, n_col=256, hidden=(32, 32, 32), init_rates=(0.3, 0.3),
          ramp_frac=0.15, truth=TRUE):
    """Train a PINN (physics=True) or a data-only network (physics=False).

    The physics weight is ramped up linearly over the first `ramp_frac` of training so the
    network first learns the data, then is gradually pulled onto the ODE solution.
    Returns (params, history) with history arrays of length n_steps.
    """
    T, I0 = truth["T"], truth["I0"]
    ic = jnp.array([1.0 - I0, I0, 0.0])
    tau_obs = jnp.asarray(t_obs / T, dtype=jnp.float32)
    I_obs_j = jnp.asarray(I_obs, dtype=jnp.float32)
    tau_col = jnp.linspace(0.0, 1.0, n_col)                 # collocation points cover ALL of [0, T]
    scale = float(np.max(I_obs))

    key = random.PRNGKey(seed)
    params = {"net": init_mlp(key, (1,) + tuple(hidden) + (3,)),
              "log_beta": jnp.log(init_rates[0]), "log_gamma": jnp.log(init_rates[1])}
    lr_tree = {"net": jax.tree_util.tree_map(lambda _: 1.0, params["net"]),
               "log_beta": lr_param_mult, "log_gamma": lr_param_mult}

    m0 = jax.tree_util.tree_map(jnp.zeros_like, params)
    v0 = jax.tree_util.tree_map(jnp.zeros_like, params)
    b1, b2, eps = 0.9, 0.999, 1e-8
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    def step(carry, i):
        p, m, v = carry
        ramp = jnp.minimum(1.0, i / (ramp_frac * n_steps))
        w = (w_phys * ramp) if physics else 0.0
        (loss, (ld, lp)), g = grad_fn(p, tau_obs, I_obs_j, tau_col, ic, T, w, scale)
        t = i + 1.0
        sched = 0.5 * (1 + jnp.cos(jnp.pi * i / n_steps)) * 0.99 + 0.01   # cosine decay to 1%
        m = jax.tree_util.tree_map(lambda m_, g_: b1 * m_ + (1 - b1) * g_, m, g)
        v = jax.tree_util.tree_map(lambda v_, g_: b2 * v_ + (1 - b2) * g_ ** 2, v, g)
        upd = jax.tree_util.tree_map(
            lambda m_, v_, s_: s_ * lr * sched * (m_ / (1 - b1 ** t)) / (jnp.sqrt(v_ / (1 - b2 ** t)) + eps),
            m, v, lr_tree)
        p = jax.tree_util.tree_map(lambda p_, u_: p_ - u_, p, upd)
        return (p, m, v), (loss, ld, lp, jnp.exp(p["log_beta"]), jnp.exp(p["log_gamma"]))

    run = jax.jit(lambda c: jax.lax.scan(step, c, jnp.arange(n_steps, dtype=jnp.float32)))
    (params, _, _), (loss, ld, lp, beta_h, gamma_h) = run((params, m0, v0))
    hist = dict(loss=np.asarray(loss), data=np.asarray(ld), phys=np.asarray(lp),
                beta=np.asarray(beta_h), gamma=np.asarray(gamma_h))
    return params, hist


def predict(params, t, truth=TRUE):
    """Network prediction (S, I, R) at days t, shape (len(t), 3)."""
    T, I0 = truth["T"], truth["I0"]
    ic = jnp.array([1.0 - I0, I0, 0.0])
    tau = jnp.asarray(np.asarray(t) / T, dtype=jnp.float32)
    return np.asarray(vmap(lambda s: state(params["net"], s, ic))(tau))


# ----------------------------------------------------------------------
# 4. Classical baseline: non-linear least squares with an ODE solver
# ----------------------------------------------------------------------
def fit_ode_least_squares(t_obs, I_obs, init_rates=(0.3, 0.3), truth=TRUE):
    """Fit (beta, gamma) by minimising the misfit of I(t_obs) with an ODE solver in the loop."""
    I0 = truth["I0"]

    def resid(logp):
        beta, gamma = np.exp(logp)
        sol = solve_ivp(sir_rhs, (0.0, t_obs.max()), [1.0 - I0, I0, 0.0], t_eval=t_obs,
                        args=(beta, gamma), rtol=1e-8, atol=1e-10)
        if sol.y.shape[1] != len(t_obs):
            return np.full(len(t_obs), 1e3)
        return (sol.y[1] - I_obs) / I_obs.max()

    res = least_squares(resid, np.log(init_rates), method="trf")
    return tuple(np.exp(res.x))


# ----------------------------------------------------------------------
# 5. Metrics
# ----------------------------------------------------------------------
def trajectory_metrics(pred, t_eval, t_max_obs, truth=TRUE):
    """RMSE of I(t) inside and beyond the observation window, plus RMSE of hidden S and R."""
    ref = simulate(truth["beta"], truth["gamma"], truth["I0"], truth["T"], t_eval)
    pred = np.clip(pred, 0.0, 1.0)                      # fractions cannot leave [0, 1]
    inside, outside = t_eval <= t_max_obs, t_eval > t_max_obs
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2))) if len(a) else float("nan")
    return dict(
        rmse_I_in=rmse(pred[inside, 1], ref[inside, 1]),
        rmse_I_out=rmse(pred[outside, 1], ref[outside, 1]),
        rmse_S=rmse(pred[:, 0], ref[:, 0]),
        rmse_R=rmse(pred[:, 2], ref[:, 2]),
    )


def rate_metrics(beta, gamma, truth=TRUE):
    """Relative errors of beta, gamma, R0 = beta/gamma and growth rate beta - gamma."""
    tb, tg = truth["beta"], truth["gamma"]
    rel = lambda a, b: float(abs(a - b) / abs(b))
    return dict(err_beta=rel(beta, tb), err_gamma=rel(gamma, tg),
                err_R0=rel(beta / gamma, tb / tg), err_growth=rel(beta - gamma, tb - tg))
