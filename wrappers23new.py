from __future__ import annotations
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any

from inference import poisson_logpdf, hmm_expected_states
from HMM_models import RampModelHMM, StepModelHMM
from models import RampModel as SimRamp, StepModel as SimStep


#   Likelihood + FB helpers

def make_ll(spikes: np.ndarray,
            rates:  np.ndarray,
            eps: float = 1e-8) -> np.ndarray:
    # Compute the log likelihood of the Poisson model
    rates = np.clip(rates, eps, None)
    return poisson_logpdf(spikes, rates)  # (T, K)


def run_fba(ll: np.ndarray,
            Tmat: np.ndarray,
            pi0: np.ndarray,
            *,
            filtering: bool = False) -> np.ndarray:
    # Posterior inference.hmm_expected_states
    post, _ = hmm_expected_states(pi0, Tmat, ll, filter=filtering)
    return post  # (T, K)



#   Posterior summaries & metrics

posterior_mean_x = lambda post, xgrid: (post * xgrid[None, :]).sum(1)
posterior_p_up   = lambda post, ups:   post[:, ups].sum(1)

mae_ramp = lambda x_true, x_hat: float(np.mean(np.abs(x_true - x_hat)))


def jump_time_error(p_up:     np.ndarray,
                    tau_true: int,
                    *,
                    thresh: float = 0.5) -> Tuple[float, int]:     
    cross    = np.where(p_up > thresh)[0]
    tau_hat  = int(cross[0]) if cross.size else len(p_up)
    return abs(tau_true - tau_hat), tau_hat


# Core evaluation loop

def run_one_setting(simulator,
                    Tmat: np.ndarray,
                    pi0: np.ndarray,
                    mapping,
                    rate_grid: np.ndarray,
                    metric_fun,
                    posterior_fun,
                    *,
                    N_trials: int = 25,
                    T: int = 100,
                    filtering: bool = False,
                    thresh: float = 0.5,
                    seed: int | None = None) -> Tuple[np.ndarray, List]:
    # Simulate *N_trials* and evaluate single‑trial inference error.
    if seed is not None:
        npr.seed(seed)

    spikes, latent, _rates = simulator.simulate(N_trials, T, get_rate=True)

    errs, examples = [], []
    for j in range(N_trials):
        ll   = make_ll(spikes[j], rate_grid)
        post = run_fba(ll, Tmat, pi0, filtering=filtering)
        est  = posterior_fun(post, mapping)

        if metric_fun is mae_ramp:
            err, extra = metric_fun(latent[j], est), None
        else:  # jump_time_error
            err, extra = metric_fun(est, latent[j], thresh=thresh)

        errs.append(err)

        if j < 3:                       # keep a few illustrative trials
            examples.append((latent[j], est, spikes[j], extra))

    return np.asarray(errs), examples



#   Parameter sweep convenience

def sweep_grid(grid: List[tuple],
               make_hmm,
               metric_fun,
               posterior_fun,
               *,
               K: int = 100,
               **kwargs):
    # Average error over a 2D parameter sweep
    xs, ys, zs = [], [], []
    for p in grid:
        sim, Tmat, pi0, mapping, rates = make_hmm(*p, K=K)
        errs, _ = run_one_setting(sim, Tmat, pi0, mapping, rates,
                                  metric_fun, posterior_fun, **kwargs)
        xs.append(p[0])
        ys.append(p[1] if len(p) > 1 else 0)
        zs.append(errs.mean())
    return np.asarray(xs), np.asarray(ys), np.asarray(zs)


#   Plotting helpers
def plot_example_ramp(x_true, x_hat, spikes, ax=None):
    ax = plt.gca() if ax is None else ax
    t  = np.arange(len(x_true))
    ax.plot(t, x_true, color="black", alpha=.3, lw=1, label=r"$x_{true}$")
    ax.plot(t, x_hat,  color="C0",   lw=2, label=r"$\hat{x}$")
    ax.vlines(np.where(spikes > 0)[0], *ax.get_ylim(),
              color="grey", alpha=.2)
    ax.set(xlabel="time", ylabel="x")
    ax.legend(frameon=False)


def plot_example_step(p_up, tau_true, tau_hat=None, ax=None):
    ax = plt.gca() if ax is None else ax
    t  = np.arange(len(p_up))
    ax.plot(t, p_up, color="C0", lw=2, label=r"$P(\mathrm{up})$")
    ax.axvline(tau_true, color="k", ls="--", alpha=.5,
               label=r"$\tau_{true}$")
    if tau_hat is not None:
        ax.axvline(tau_hat, color="C3", ls=":", label=r"$\hat{\tau}$")
    ax.set(xlabel="time", ylabel="probability")
    ax.legend(frameon=False)


#   Model builders

def build_ramp(beta: float,
               sigma: float,
               *,
               K: int = 101,
               dt: float = 1.0,
               Rh: float = 50.0,
               x0: float = 0.2):
    """Return (simulator, T, pi0, x‑grid, rate‑grid)."""
    sim  = SimRamp(beta=beta, sigma=sigma, x0=x0, Rh=Rh)
    hmm  = RampModelHMM(K=K, beta=beta, sigma=sigma, dt=dt)

    pi0           = np.zeros(K); pi0[0] = 1.0
    xgrid         = hmm.x_values
    rate_grid     = Rh * xgrid                    # Poisson mean counts
    return sim, hmm.T, pi0, xgrid, rate_grid


def build_step(m: int,
               r: int = 1,
               *,
               dt: float = 1.0,
               exact: bool = False,
               x0: float = 0.2,
               Rh: float = 50.0):
    sim = SimStep(m=m, r=r, x0=x0, Rh=Rh)
    hmm = StepModelHMM(m=m, r=r, dt=dt, exact=exact)

    K   = hmm.T.shape[0]
    pi0 = np.zeros(K); pi0[0] = 1.0

    if exact:
        rate_grid = np.concatenate((np.full(K-1, x0 * Rh), [Rh]))
        up_idx    = np.where(np.isclose(rate_grid, Rh))[0]  # all Rh states
    else:
        rate_grid = np.array([x0 * Rh, Rh])
        up_idx    = np.array([1])                           # state 1 is "up"

    return sim, hmm.T, pi0, up_idx, rate_grid


__all__ = [
    # Likelihood / FB
    "make_ll", "run_fba",
    # Posterior summaries
    "posterior_mean_x", "posterior_p_up",
    # Metrics
    "mae_ramp", "jump_time_error",
    # Core loops
    "run_one_setting", "sweep_grid",
    # Plotting helpers
    "plot_example_ramp", "plot_example_step",
    # Builders
    "build_ramp", "build_step",
]
