"""
run_experiments.py -- reproduces every figure and table in the README.

    python run_experiments.py            # full run (about 15-20 min on one CPU core)
    python run_experiments.py --quick    # 3 seeds per setting, for a fast check

Hyper-parameters (physics weight 100; 40,000 steps for the PINN, 12,000 for the data-only NN) were chosen on separate tuning seeds (100-102),
never on the evaluation seeds (0..n_seeds-1) used below.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sir_pinn import (TRUE, fit_ode_least_squares, make_observations, predict, rate_metrics,
                      simulate, train, trajectory_metrics)

OUT = "results"
T = TRUE["T"]
T_EVAL = np.linspace(0, T, 321)
DEFAULT = dict(t_max=30.0, n_obs=25, noise=0.10)


def one_run(t_max, n_obs, noise, seed):
    """Run PINN, data-only NN and ODE least squares on one synthetic data set."""
    t_obs, I_obs = make_observations(t_max, n_obs, noise, seed)
    p, h = train(t_obs, I_obs, physics=True, seed=seed)
    q, _ = train(t_obs, I_obs, physics=False, seed=seed, n_steps=12000)
    b_ls, g_ls = fit_ode_least_squares(t_obs, I_obs)
    b_p, g_p = float(h["beta"][-1]), float(h["gamma"][-1])
    ls_traj = simulate(b_ls, g_ls, TRUE["I0"], T, T_EVAL)
    return {
        "pinn": {**rate_metrics(b_p, g_p), **trajectory_metrics(predict(p, T_EVAL), T_EVAL, t_max),
                 "beta": b_p, "gamma": g_p},
        "ode_ls": {**rate_metrics(b_ls, g_ls), **trajectory_metrics(ls_traj, T_EVAL, t_max),
                   "beta": float(b_ls), "gamma": float(g_ls)},
        "nn": trajectory_metrics(predict(q, T_EVAL), T_EVAL, t_max),
    }


def sweep(name, key, values, n_seeds):
    rows = []
    for v in values:
        cfg = {**DEFAULT, key: v}
        cache = f"{OUT}/cache_{name}_{v}_{n_seeds}.json"      # lets an interrupted run resume
        if os.path.exists(cache):
            runs = json.load(open(cache))
        else:
            runs = [one_run(cfg["t_max"], cfg["n_obs"], cfg["noise"], s) for s in range(n_seeds)]
            json.dump(runs, open(cache, "w"))
        rows.append({"value": v, "runs": runs})
        print(f"[{name}] {key}={v} done", flush=True)
    return rows


def agg(rows, method, metric):
    """Median and inter-quartile range across seeds (robust to the odd bad run)."""
    out = []
    for r in rows:
        x = np.array([run[method][metric] for run in r["runs"]])
        out.append((np.median(x), np.percentile(x, 25), np.percentile(x, 75)))
    return np.array(out)


def md_table(rows, label, methods_metrics):
    head = f"| {label} | " + " | ".join(f"{m} {k}" for m, k in methods_metrics) + " |\n"
    head += "|---|" + "---|" * len(methods_metrics) + "\n"
    body = ""
    for i, r in enumerate(rows):
        cells = []
        for m, k in methods_metrics:
            a = agg(rows, m, k)[i]
            cells.append(f"{a[0]:.3f} ({a[1]:.3f}-{a[2]:.3f})")
        body += f"| {r['value']} | " + " | ".join(cells) + " |\n"
    return head + body


def plot_sweep(rows, xlabel, fname, logx=False):
    xs = [r["value"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    spec = [("err_R0", "Relative error of $R_0=\\beta/\\gamma$", ["pinn", "ode_ls"]),
            ("err_growth", "Relative error of growth rate $\\beta-\\gamma$", ["pinn", "ode_ls"]),
            ("rmse_I_out", "RMSE of $I(t)$ beyond the data window", ["pinn", "ode_ls", "nn"])]
    names = {"pinn": "PINN", "ode_ls": "ODE least squares", "nn": "Data-only NN"}
    col = {"pinn": "C0", "ode_ls": "C1", "nn": "C3"}
    for a, (metric, title, methods) in zip(ax, spec):
        for m in methods:
            s = agg(rows, m, metric)
            a.plot(xs, s[:, 0], "o-", color=col[m], label=names[m])
            a.fill_between(xs, s[:, 1], s[:, 2], color=col[m], alpha=0.15)
        a.set_xlabel(xlabel); a.set_title(title, fontsize=10); a.grid(alpha=0.3)
        if metric == "rmse_I_out":
            a.set_yscale("log")
    ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/{fname}", dpi=150); plt.close(fig)


def main_figure(seed=0):
    t_obs, I_obs = make_observations(DEFAULT["t_max"], DEFAULT["n_obs"], DEFAULT["noise"], seed)
    p, h = train(t_obs, I_obs, physics=True, seed=seed)
    q, hq = train(t_obs, I_obs, physics=False, seed=seed)
    ref = simulate(TRUE["beta"], TRUE["gamma"], TRUE["I0"], T, T_EVAL)
    P, Q = predict(p, T_EVAL), predict(q, T_EVAL)
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    a = ax[0, 0]
    a.plot(T_EVAL, ref[:, 1], "k-", lw=2, label="Truth")
    a.plot(T_EVAL, P[:, 1], "C0--", lw=2, label="PINN")
    a.plot(T_EVAL, Q[:, 1], "C3:", lw=2, label="Data-only NN")
    a.scatter(t_obs, I_obs, c="k", s=18, zorder=5, label="Noisy observations of $I$")
    a.axvspan(DEFAULT["t_max"], T, color="grey", alpha=0.12, label="No data")
    a.set_ylim(-0.05, 0.4); a.set_xlabel("Day"); a.set_ylabel("Infectious fraction $I$")
    a.set_title("(a) Observed compartment and forecast"); a.legend(fontsize=8)
    a = ax[0, 1]
    for j, (nm, c) in enumerate([("S", "C2"), ("R", "C4")]):
        jj = 0 if nm == "S" else 2
        a.plot(T_EVAL, ref[:, jj], "-", color=c, lw=2, label=f"{nm} truth")
        a.plot(T_EVAL, P[:, jj], "--", color="k", lw=1.5, label="PINN estimate (dashed)" if j == 0 else None)
    a.set_xlabel("Day"); a.set_ylabel("Fraction"); a.legend(fontsize=8)
    a.set_title("(b) Never-observed $S$ and $R$ recovered by the PINN")
    a = ax[1, 0]
    a.plot(h["beta"], label=r"$\beta$ (PINN)", color="C0"); a.axhline(TRUE["beta"], color="C0", ls=":")
    a.plot(h["gamma"], label=r"$\gamma$ (PINN)", color="C1"); a.axhline(TRUE["gamma"], color="C1", ls=":")
    a.set_xlabel("Training step"); a.set_ylabel("Rate per day"); a.legend(fontsize=8)
    a.set_title("(c) Parameter learning (dotted = truth)")
    a = ax[1, 1]
    a.semilogy(h["data"], label="Data misfit"); a.semilogy(h["phys"], label="ODE residual")
    a.set_xlabel("Training step"); a.set_ylabel("Loss term"); a.legend(fontsize=8)
    a.set_title("(d) Loss terms")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig1_main_fit.png", dpi=150); plt.close(fig)
    return {"beta_hat": float(h["beta"][-1]), "gamma_hat": float(h["gamma"][-1]),
            "R0_hat": float(h["beta"][-1] / h["gamma"][-1])}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    n_seeds = 3 if ap.parse_args().quick else 10
    os.makedirs(OUT, exist_ok=True)
    main = main_figure()
    print("main figure:", main, flush=True)
    res = {
        "main": main,
        "window": sweep("window", "t_max", [10.0, 15.0, 20.0, 30.0, 50.0], n_seeds),
        "noise": sweep("noise", "noise", [0.05, 0.10, 0.20, 0.30], n_seeds),
        "nobs": sweep("nobs", "n_obs", [8, 15, 25, 50], n_seeds),
    }
    json.dump(res, open(f"{OUT}/results.json", "w"), indent=1)
    plot_sweep(res["window"], "Last observed day", "fig2_window.png")
    plot_sweep(res["noise"], "Relative noise level", "fig3_noise.png")
    plot_sweep(res["nobs"], "Number of observations", "fig4_nobs.png")
    mm = [("pinn", "err_R0"), ("ode_ls", "err_R0"), ("pinn", "err_growth"), ("ode_ls", "err_growth"),
          ("pinn", "rmse_I_out"), ("ode_ls", "rmse_I_out"), ("nn", "rmse_I_out")]
    with open(f"{OUT}/tables.md", "w") as f:
        for k, lab in [("window", "last observed day"), ("noise", "noise"), ("nobs", "n obs")]:
            f.write(f"### {k}\n\n" + md_table(res[k], lab, mm) + "\n")
    print("done")
