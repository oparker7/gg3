import numpy as np
from scipy.stats import norm, truncnorm

def construct_ramp_transition_matrix(K, beta, sigma, dt):
    x_values = np.arange(K) / (K - 1)
    T = np.zeros((K, K))

    for s in range(K):
        x_current = x_values[s]
        if s == K - 1:
            T[s, s] = 1.0
            continue

        dx = 1.0 / (K - 1)
        mean = x_current + beta * dt
        std = sigma * np.sqrt(dt)

        x_centres = x_values
        left_edges = np.clip(x_centres - dx / 2, 0.0, 1.0)
        right_edges = np.clip(x_centres + dx / 2, 0.0, 1.0)

        if std > 0:
            cdf_left = norm.cdf((left_edges - mean) / std)
            cdf_right = norm.cdf((right_edges - mean) / std)
            probs = cdf_right - cdf_left
            probs[0] += norm.cdf((0 - mean) / std)
            probs[-1] += 1 - norm.cdf((1 - mean) / std)
        else:
            probs = np.zeros(K)
            if mean < 0:
                probs[0] = 1.0
            elif mean > 1:
                probs[-1] = 1.0
            else:
                idx = int(round(mean * (K - 1)))
                probs[idx] = 1.0

        T[s, :] = probs / probs.sum()
    return T

def simulate_spikes_standalone(K, beta, sigma, dt, R_h, n_steps=100, initial_state=0.2):
    T = construct_ramp_transition_matrix(K, beta, sigma, dt)
    x_values = np.arange(K) / (K - 1)

    states = np.zeros(n_steps + 1, dtype=int)
    states[0] = round(initial_state*(K-1))

    for t in range(n_steps):
        probs = T[states[t], :]
        states[t + 1] = np.random.choice(K, p=probs)

    x_vals = x_values[states]
    rates = R_h * x_vals
    spikes = np.random.poisson(rates * dt)

    return states, x_vals, spikes

def ramp_sample_prior_and_simulate_spikes(
    N,                  # Number of spike trains
    K=100,
    R_h=50.0,
    n_steps=100,
    beta_range=(0.0, 4),
    sigma_range=(0.04, 4),
    M=30,
    prior_type='uniform',
    prior_sd_fraction=0.25,
    seed=None
):

    dt = 1/ n_steps
    np.random.seed(seed)
    
    # Grid
    beta_vals = np.linspace(*beta_range, M)
    lnsigma_vals = np.linspace(np.log(sigma_range[0]), np.log(sigma_range[1]), M)
    sigma_vals = np.exp(lnsigma_vals)

    # Prior
    if prior_type == 'uniform':
        prior = np.ones((M, M))
        prior /= prior.sum()

    elif prior_type == 'gaussian':
        beta_mu = np.mean(beta_range)
        sigma_mu = np.mean(np.log(sigma_range))

        beta_sd = prior_sd_fraction * (beta_range[1] - beta_range[0])
        sigma_sd = prior_sd_fraction * (np.log(sigma_range[1]) - np.log(sigma_range[0]))

        beta_prior = truncnorm.pdf(
            beta_vals,
            (beta_range[0] - beta_mu) / beta_sd,
            (beta_range[1] - beta_mu) / beta_sd,
            loc=beta_mu, scale=beta_sd
        )

        sigma_prior = truncnorm.pdf(
            lnsigma_vals,
            (np.log(sigma_range[0]) - sigma_mu) / sigma_sd,
            (np.log(sigma_range[1]) - sigma_mu) / sigma_sd,
            loc=sigma_mu, scale=sigma_sd
        )

        prior = np.outer(beta_prior, sigma_prior)
        prior /= prior.sum()
    else:
        raise ValueError(f"Unknown prior_type: {prior_type}")

    # Sample from prior grid
    prior_flat = prior.ravel()
    indices = np.random.choice(int(M * M), size=int(N), p=prior_flat)
    beta_idx, sigma_idx = np.unravel_index(indices, (M, M))
    betas = beta_vals[beta_idx]
    sigmas = sigma_vals[sigma_idx]

    # Simulate each trial
    all_states = []
    all_xvals = []
    all_spikes = []
    for b, s in zip(betas, sigmas):
        states, x_vals, spikes = simulate_spikes_standalone(
            K=K, beta=b, sigma=s, dt=dt, R_h=R_h, n_steps=n_steps
        )
        all_states.append(states)
        all_xvals.append(x_vals)
        all_spikes.append(spikes)

    return {
        "betas": betas,
        "sigmas": sigmas,
        "states": np.array(all_states),
        "x_vals": np.array(all_xvals),
        "spikes": np.array(all_spikes),
        "beta_vals": beta_vals,
        "sigma_vals": sigma_vals,
        "prior": prior
    }

def construct_step_transition_matrix(r, m, exact=True):
    """
    Transition matrix for StepModelHMM.
    p = r / (m + r)
    the 'exact' flag delineates betweent the 2 state and r+1 state models
    """
    p = r / (m + r)
    if exact:
        K = int(r) + 1
        T = np.zeros((K, K))
        for i in range(K - 1):
            T[i, i] = 1 - p
            T[i, i + 1] = p
        T[-1, -1] = 1.0
    else:
        K = 2
        T = np.zeros((K, K))
        T[0, 0] = 1 - p
        T[0, 1] = p
        T[1, 1] = 1.0
    return T, K

def simulate_step_markov_chain(T, K, n_steps, exact=True, r=None):
    states = np.zeros(n_steps + 1, dtype=int)
    tau_true = None
    for t in range(n_steps):
        states[t+1] = np.random.choice(K, p=T[states[t]])
        if tau_true is None:
            if (not exact and states[t+1] == 1) or (exact and states[t+1] == r):
                tau_true = t + 1
    if tau_true is None:
        tau_true = n_steps
    return states, tau_true

def simulate_spikes_from_states(states, n_steps, R_low=5.0, R_high=50.0, dt=0.1, exact=False, r=None):
    spikes = np.zeros(n_steps + 1, dtype=int)
    for t in range(n_steps + 1):
        if (not exact and states[t] == 1) or (exact and states[t] == r):
            rate_t = R_high
        else:
            rate_t = R_low
        spikes[t] = np.random.poisson(rate_t * dt)
    return spikes

def step_sample_prior_and_simulate_spikes(
    N,
    m_range=(0, 75),
    r_range=(1, 6),
    M_m=30,
    M_r=None,
    prior_type='uniform',
    prior_sd_fraction=0.25,
    n_steps=100,
    R_low=5.0,
    R_high=50.0,
    exact=True,
    seed=None,
):
    """
    Sample N parameter pairs (m, r) from prior (uniform or gaussian),
    simulate spike trains from the step model for each,
    and return dataset:
      - ms: shape (N,)
      - rs: shape (N,)
      - spikes: shape (N, n_steps+1)
      - states: shape (N, n_steps+1)
      - tau_trues: shape (N,)
    """
    dt = 1/ n_steps

    if seed is not None:
        np.random.seed(seed)

    if M_r is None:
        M_r = r_range[1] - r_range[0] + 1

    # Create grids
    m_vals = np.linspace(m_range[0], m_range[1], M_m)
    r_vals = np.arange(r_range[0], r_range[1] + 1)

    # Discretization steps
    d_m = (m_range[1] - m_range[0]) / (M_m - 1)
    d_r = 1  # discrete integer steps

    # Construct prior grid
    if prior_type == 'uniform':
        prior = np.ones((M_m, len(r_vals)))
        prior *= d_m * d_r
        prior /= prior.sum()
    elif prior_type == 'gaussian':
        m_mu = np.mean(m_range)
        r_mu = 1 # centre the gaussian at r=1 regardless of range
        # this is instructed in the coursework notebook

        m_sd = prior_sd_fraction * (m_range[1] - m_range[0])
        r_sd = prior_sd_fraction * (r_range[1] - r_range[0])

        m_prior = truncnorm.pdf(
            m_vals,
            (m_range[0] - m_mu) / m_sd,
            (m_range[1] - m_mu) / m_sd,
            loc=m_mu,
            scale=m_sd,
        )
        r_prior = truncnorm.pdf(
            r_vals,
            (r_range[0] - r_mu) / r_sd,
            (r_range[1] - r_mu) / r_sd,
            loc=r_mu,
            scale=r_sd,
        )

        prior = np.outer(m_prior, r_prior)
        prior /= prior.sum()
    else:
        raise ValueError(f"Unknown prior_type: {prior_type}")

    # Flatten prior to sample indices
    prior_flat = prior.ravel()
    indices = np.random.choice(len(prior_flat), size=N, p=prior_flat)
    m_idx, r_idx = np.unravel_index(indices, prior.shape)

    ms = m_vals[m_idx]
    rs = r_vals[r_idx]

    spikes = np.zeros((N, n_steps + 1), dtype=int)
    states = np.zeros((N, n_steps + 1), dtype=int)
    tau_trues = np.zeros(N, dtype=int)

    for i in range(N):
        T, K = construct_step_transition_matrix(rs[i], ms[i], exact=exact)
        st, tau = simulate_step_markov_chain(T, K, n_steps, exact=exact, r=rs[i])
        sp = simulate_spikes_from_states(st, n_steps, R_low, R_high, dt, exact=exact, r=rs[i])
        spikes[i, :] = sp
        states[i, :] = st
        tau_trues[i] = tau

    return {
        "ms": ms,
        "rs": rs,
        "spikes": spikes,
        "states": states,
        "tau_trues": tau_trues,
    }



## here is how to generate the data we need
'''
from spikedata import ramp_sample_prior_and_simulate_spikes as ramp_datagen
from spikedata import step_sample_prior_and_simulate_spikes as step_datagen

dataset_sizes = np.array([1, 5, 10, 100, 1e3, 1e4]).astype(int) # np random choice needs integer datatype
widths = np.array([0.25, 0.5, 0.75, 1])
fixed_N_for_vary_gaussian_width = 100

uniform_ramp_data = {}
gaussian_ramp_data_vary_dataset_size = {}
gaussian_ramp_data_vary_width = {}

uniform_step_data = {}
gaussian_step_data_vary_dataset_size = {}
gaussian_step_data_vary_width = {}


for N in dataset_sizes:
    uniform_ramp_data[N] = ramp_datagen(N=N, prior_type='uniform')
    gaussian_ramp_data_vary_dataset_size[N] = ramp_datagen(N=N, prior_type='gaussian')

    uniform_step_data[N] = step_datagen(N=N, prior_type='uniform')
    gaussian_step_data_vary_dataset_size[N] = step_datagen(N=N, prior_type='gaussian')

for sig in widths:
    gaussian_ramp_data_vary_width[sig] = ramp_datagen(N=fixed_N_for_vary_gaussian_width,
                                                      prior_type='gaussian',
                                                      prior_sd_fraction=sig)

    gaussian_step_data_vary_width[sig] = step_datagen(N=fixed_N_for_vary_gaussian_width,
                                                      prior_type='gaussian',
                                                      prior_sd_fraction=sig)

                                                      '''