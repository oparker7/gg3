import functools, numpy as np
from HMM_models import RampModelHMM, StepModelHMM

@functools.lru_cache(maxsize=None)
def build_ramp_grid(M, K, T, R_h):
    dt = 1 / T
    beta_vals  = np.linspace(0, 4, M)
    sigma_vals = np.exp(np.linspace(np.log(0.04), np.log(4), M))
    x_grid     = np.arange(K) / (K - 1)

    T_grid = np.empty((M, M, K, K))
    for i, β in enumerate(beta_vals):
        for j, σ in enumerate(sigma_vals):
            T_grid[i, j] = RampModelHMM(K, β, σ, dt).T
    λ = R_h * x_grid * dt
    return T_grid, λ

@functools.lru_cache(maxsize=None)
def build_step_models(M, T, R_low, R_high):
    dt     = 1 / T
    m_vals = np.linspace(0.25*T, 0.75*T, M)
    r_vals = np.arange(1, 7)
    Ps     = np.empty((len(m_vals), len(r_vals), 2, 2))
    for i, m in enumerate(m_vals):
        for j, r in enumerate(r_vals):
            Ps[i, j] = StepModelHMM(m, r, dt, exact=False).T
    λ = np.array([R_low*dt, R_high*dt])
    return Ps, λ, m_vals, r_vals
