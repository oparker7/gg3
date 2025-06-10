import numpy as np
import matplotlib.pyplot as plt
import inference
from inference import hmm_normalizer, poisson_logpdf
from HMM_models import RampModelHMM, StepModelHMM
from joblib import Parallel, delayed
from scipy.stats import truncnorm
from scipy.special import logsumexp

def perform_ramp_inference(
    K: int,
    beta: float,
    sigma: float,
    dt: float,
    T: int,
    R_h: float,
    N: int,
    pi0: np.ndarray = None,
):
    """
    Simulate N trials from the Ramp HMM, run HMM smoothing (or filtering), and return:
      - true_ramps    : shape (N, T+1), the ground-truth x_t trajectories
      - inferred_ramps: shape (N, T+1), E[x_t | spikes] for each trial
      - MAEs           : shape (N,), mean absolute error per trial

    Parameters
    ----------
    K       : number of discrete ramp levels (integer)
    beta    : drift parameter for the ramp-HMM (float)
    sigma   : diffusion parameter for the ramp-HMM (float)
    dt      : time-bin width (in seconds; float)
    T       : number of bins per trial (integer)
    R_h     : maximum (high) Poisson rate in Hz (float)
    N       : number of trials to simulate (integer)
    pi0     : length-K initial distribution; if None, assumes all mass on state 0 (np.ndarray of shape (K,))

    Returns
    -------
    true_ramps     : np.ndarray, shape (N, T+1)
    inferred_ramps : np.ndarray, shape (N, T+1)
    MAEs           : np.ndarray, shape (N,)
    """
    # 1) Build the Ramp HMM and transition matrix
    ramp_hmm = RampModelHMM(K=K, beta=beta, sigma=sigma, dt=dt)
    Ps = ramp_hmm.T       # (K × K) transition matrix

    # 2) Build pi0 if not provided
    if pi0 is None:
        pi0 = np.zeros(K)
        pi0[0] = 1.0       # always start in state 0

    # 3) Precompute x_grid = [0, 1/(K-1), 2/(K-1), ..., 1]
    x_grid = np.arange(K, dtype=float) / float(K - 1)

    # 4) Storage for ground truth and inference results
    true_ramps     = np.zeros((N, T + 1), dtype=float)
    inferred_ramps = np.zeros((N, T + 1), dtype=float)
    inferred_ramps_filtered = np.zeros((N, T + 1), dtype=float)
    MAEs           = np.zeros(N, dtype=float)
    MAEs_filtered  = np.zeros(N, dtype=float)
    # 5) Loop over trials
    for i in range(N):
        # 5a) Simulate one trial: (states_i, x_vals_i, spikes_i)
        states_i, x_vals_i, spikes_i = ramp_hmm.simulate_spikes(
            n_steps=T,
            initial_state=0,
            R_h=R_h,
            dt=dt
        )
        true_ramps[i, :] = x_vals_i   # store ground‐truth

        # 5b) Build Poisson log‐likelihood matrix ll_i: shape (T+1, K)
        #    If latent state is s, rate = R_h * (s / (K - 1))
        rates = R_h * x_grid       # shape (K,)
        lambdas = rates * dt  # convert to expected counts per bin
        ll_i = inference.poisson_logpdf(
            counts=spikes_i,       # shape (T+1,)
            lambdas=lambdas,         # shape (K,)
            mask=None              # yields shape (T+1, K)
        )

        # 5c) Run forward–backward: smoothing
        post_probs_i, logZ_i = inference.hmm_expected_states(
            pi0=pi0,     # shape (K,)
            Ps=Ps,       # shape (K, K)
            ll=ll_i,     # shape (T+1, K)
            filter=False
        )
        # post_probs_i[t, s] = P(s_t = s | data)

        # 5d) Compute E[x_t | data] = sum_s [ x_grid[s] * post_probs_i[t, s] ]
        Exp_x_i = (post_probs_i * x_grid[None, :]).sum(axis=1)   # shape (T+1,)
        inferred_ramps[i, :] = Exp_x_i

        # 5e) Compute MAE for this trial
        MAEs[i] = np.mean(np.abs(Exp_x_i - x_vals_i))

    # added filtered part irrispective of input so can compare for some trials
        post_probs_i_filtered, logZ_i = inference.hmm_expected_states(
            pi0=pi0,     # shape (K,)
            Ps=Ps,       # shape (K, K)
            ll=ll_i,     # shape (T+1, K)
            filter=True
        )

        # 5d) Compute E[x_t | data] = sum_s [ x_grid[s] * post_probs_i[t, s] ]
        Exp_x_i_filtered = (post_probs_i_filtered * x_grid[None, :]).sum(axis=1)   # shape (T+1,)
        inferred_ramps_filtered[i, :] = Exp_x_i_filtered

        # 5e) Compute MAE for this trial
        MAEs_filtered[i] = np.mean(np.abs(Exp_x_i_filtered - x_vals_i))

    return true_ramps, inferred_ramps, inferred_ramps_filtered, MAEs, MAEs_filtered

def plot_ramp_example(
    true_ramps: np.ndarray,
    inferred_ramps: np.ndarray,
    trial_index: int = 0,
    title: str = None
) -> None:
    """
    Plot true vs. inferred ramp for one example trial.

    Parameters
    ----------
    true_ramps     : np.ndarray, shape (N, T+1)
    inferred_ramps : np.ndarray, shape (N, T+1)
    trial_index    : integer index of trial to plot (0 <= trial_index < N)
    """
    x_true = true_ramps[trial_index, :]
    x_inf  = inferred_ramps[trial_index, :]

    plt.figure(figsize=(6, 3))
    plt.plot(x_true, label="True $x_t$", color="black")
    plt.plot(x_inf,  label="Inferred $E[x_t]$", color="red", alpha=0.7)
    plt.xlabel("Time bin $t$")
    plt.ylabel("$x(t)$")
    if title is None:
        plt.title(f"Ramp Inference: Trial {trial_index}")
    else:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def perform_step_inference(
    m: float,
    r: int,
    dt: float,
    T: int,
    R_low: float,
    R_high: float,
    N: int,
    exact: bool = True,
    pi0: np.ndarray = None,
):
    """
    Simulate N trials from the Step HMM, run HMM smoothing (or filtering), and return:
      - true_taus     : shape (N,), the ground-truth jump times τ_true for each trial
      - est_taus      : shape (N,), the inferred jump times τ_hat (via thresholding at 0.5)
      - MAEs          : shape (N,), |τ_hat - τ_true| for each trial

    Parameters
    ----------
    m       : mean of the NB distribution (float)
    r       : shape parameter of NB (integer). If exact=False, we use a 2-state chain; if exact=True, chain has r+1 states.
    dt      : time-bin width (float)
    T       : number of bins per trial (integer)
    R_low   : Poisson rate in low state (Hz; float)
    R_high  : Poisson rate in high state (Hz; float)
    N       : number of trials to simulate (integer)
    exact   : if False, use 2-state chain; if True, use (r+1)-state absorbing chain (boolean)
    pi0     : length-K initial distribution; if None, assume start in state 0 (np.ndarray)

    Returns
    -------
    true_taus : np.ndarray, shape (N,)
    est_taus  : np.ndarray, shape (N,)
    MAEs      : np.ndarray, shape (N,)
    """
    # 1) Build the Step HMM and transition matrix
    step_hmm = StepModelHMM(m=m, r=r, dt=dt, exact=exact)
    Ps = step_hmm.T       # (K × K) transition matrix
    K  = step_hmm.K       # number of discrete states

    # 2) Build pi0 if not provided
    if pi0 is None:
        pi0 = np.zeros(K)
        pi0[0] = 1.0       # always start in state 0

    # 3) Identify index/indices of “high” state(s)
    if not exact:
        high_states = [1]     # 2-state: state 1 is high
    else:
        high_states = [r]     # (r+1)-state: state r is absorbing high

    # 4) Precompute Poisson rates array: rates[s] = R_high if s in high_states else R_low
    rates = np.zeros(K, dtype=float)
    for s_idx in range(K):
        rates[s_idx] = R_high if (s_idx in high_states) else R_low

    # 5) Storage for true jump times, estimated jump times, and errors
    true_taus = np.zeros(N, dtype=int)
    est_taus  = np.zeros(N, dtype=int)
    est_taus_filtered  = np.zeros(N, dtype=int)
    MAEs      = np.zeros(N, dtype=float)
    MAEs_filtered      = np.zeros(N, dtype=float)
    P_high     = np.zeros((N, T + 1), dtype=float)
    P_high_f     = np.zeros((N, T + 1), dtype=float)

    # 6) Loop over trials
    for i in range(N):
        # 6a) Simulate one trial: (states_i, tau_true_i, spikes_i)
        states_i, tau_true_i, spikes_i = step_hmm.simulate_spikes(
            n_steps=T,
            R_low=R_low,
            R_high=R_high,
            dt=dt
        )
        true_taus[i] = tau_true_i

        # 6b) Build Poisson log-likelihood matrix ll_i: shape (T+1, K)
        #    If state = s, rate = rates[s]
        lambdas = rates * dt  # convert to expected counts per bin
        ll_i = inference.poisson_logpdf(
            counts=spikes_i,     # shape (T+1,)
            lambdas=lambdas,       # shape (K,)
            mask=None            # yields shape (T+1, K)
        )

        # 6c) Run forward-backward: smoothing or filtering
        post_probs_i, logZ_i = inference.hmm_expected_states(
            pi0=pi0,     # shape (K,)
            Ps=Ps,       # shape (K, K)
            ll=ll_i,     # shape (T+1, K)
            filter=False
        )
        # post_probs_i[t, s] = P(s_t = s | data)

        # 6d) Compute P_high(t) = sum_{s in high_states} post_probs_i[t, s]
        P_high_i = post_probs_i[:, high_states].sum(axis=1)  # shape (T+1,)
        P_high[i, :] = P_high_i

        # 6e) Estimate jump time: first t where P_high_i > 0.5; if none, use T
        crossing_times = np.where(P_high_i > 0.5)[0]
        tau_hat_i = int(crossing_times[0]) if crossing_times.size > 0 else T
        est_taus[i] = tau_hat_i

        # 6f) Compute MAE = |tau_hat_i - tau_true_i|
        MAEs[i] = abs(tau_hat_i - tau_true_i)

        # now with filtering

                # 6c) Run forward-backward: smoothing or filtering
        post_probs_i_filtered, logZ_i = inference.hmm_expected_states(
            pi0=pi0,     # shape (K,)
            Ps=Ps,       # shape (K, K)
            ll=ll_i,     # shape (T+1, K)
            filter=True
        )
        # post_probs_i[t, s] = P(s_t = s | data)

        # 6d) Compute P_high(t) = sum_{s in high_states} post_probs_i[t, s]
        P_high_i_filtered = post_probs_i_filtered[:, high_states].sum(axis=1)  # shape (T+1,)
        P_high_f[i, :] = P_high_i_filtered

        # 6e) Estimate jump time: first t where P_high_i > 0.5; if none, use T
        crossing_times_filtered = np.where(P_high_i_filtered > 0.5)[0]
        tau_hat_i_filtered = int(crossing_times_filtered[0]) if crossing_times_filtered.size > 0 else T
        est_taus_filtered[i] = tau_hat_i_filtered

        # 6f) Compute MAE = |tau_hat_i - tau_true_i|
        MAEs_filtered[i] = abs(tau_hat_i_filtered - tau_true_i)

    return true_taus, est_taus, est_taus_filtered, MAEs, MAEs_filtered, P_high, P_high_f

def plot_step_example(
    true_taus: np.ndarray,
    est_taus: np.ndarray,
    plot_P_high: np.ndarray,
    trial_index: int = 0,
    title: str = None
) -> None:
    """
    Plot posterior P(high at time t) for one example trial,
    and mark true_tau and est_tau as vertical lines.

    Parameters
    ----------
    true_taus  : np.ndarray, shape (N,)
    est_taus   : np.ndarray, shape (N,)
    plot_P_high: np.ndarray, shape (N, T+1), each row is P_high(t) for that trial
    trial_index: integer index of trial to plot
    """
    P_high_i = plot_P_high[trial_index]
    tau_true = true_taus[trial_index]
    tau_est  = est_taus[trial_index]
    T = len(P_high_i) - 1

    plt.figure(figsize=(6, 3))
    plt.plot(P_high_i, label=r"$P(\mathrm{high} \mid \mathrm{spikes})$", color="blue")
    plt.axvline(tau_true, color="black", linestyle="--", label="True $\\tau$")
    plt.axvline(tau_est,  color="red",   linestyle="-.", label="Est. $\\hat\\tau$")
    plt.xlabel("Time bin $t$")
    plt.ylabel(r"$P(\mathrm{high} \mid n)$")
    if title is None:
        plt.title(f"Step Inference: Trial {trial_index}")
    else:
        plt.title(title)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.show()

def ramp_inference_scan(
    true_beta=2,  # these are only used if
    true_sigma=1, # no spktrn_arg is used
    fixed_rh=50.0,
    T=500,
    N=100,
    K=50,
    beta_range=(0, 4),
    sigma_range=(0.04, 4),  # ranges as specified in handout
    M=30,
    seed=42,
    n_jobs=-1,
    prior_type='uniform',  # Choose between 'uniform' and 'gaussian'
    prior_sd_fraction=0.25,  # Used if prior_type is 'gaussian'
    ax=None,
    plot=True,
    spktrn_arg=None
):

    dt = 1/T

    # simulate a dataset using the true beta and sigma values for the ramp model.
    np.random.seed(seed)  # Set random seed for reproducibility

    # Create spatial grid for hidden states
    x_grid = np.arange(K) / (K - 1)

    # Initial distribution over hidden states: spike probability 1 at 20% of state space
    pi0 = np.zeros(K)
    s0 = round(0.2 * (K - 1))  # Starting state index
    pi0[s0] = 1.0

    if spktrn_arg is None:
        # Simulate N spike trains using the true parameters
        ramp_model_true = RampModelHMM(K=K, beta=true_beta, sigma=true_sigma, dt=dt)
        spike_trains = np.array([
            ramp_model_true.simulate_spikes(n_steps=T, initial_state=s0, R_h=fixed_rh, dt=dt)[2]
            for _ in range(N)
        ])
        # spike_trains shape: (N trials, T time steps)
    else:
        spike_trains = spktrn_arg

    # Generate a meshgrid of beta and sigma values, including transformation for log-sigma.
    beta_vals = np.linspace(*beta_range, M)
    lnsigma_vals = np.linspace(np.log(sigma_range[0]), np.log(sigma_range[1]), M)
    sigma_vals = np.exp(lnsigma_vals)

    # Grid cell size used in marginal likelihood approximation
    d_beta = (beta_range[1] - beta_range[0]) / (M - 1)
    d_sigma = (np.log(sigma_range[1]) - np.log(sigma_range[0])) / (M - 1)
    grid_area = d_beta * d_sigma

    # This section allows selection of prior type: uniform or Gaussian.
    if prior_type == 'uniform':
        # Uniform prior over grid: same weight at each grid point
        prior = np.ones((M, M))
        prior /= np.sum(prior)  # Normalize so sum = 1

    # gaussian prior is centred on the middle of the range
    # this is set out in 3.2.2
    # prior_sd_fraction tells us how wide we should look around the centre
    elif prior_type == 'gaussian':
        # Gaussian prior (truncated to bounds of grid)
        # Independent Gaussians over beta and log(sigma)

        # Set means of priors (center of the range for beta/log-sigma)
        beta_mu = np.mean(beta_range)
        sigma_mu = np.mean(np.log(sigma_range))  # log-domain mean for log-normal prior

        # Compute SDs as a fraction of range (e.g., 0.25 of full range)
        beta_sd = prior_sd_fraction * (beta_range[1] - beta_range[0])
        sigma_sd = prior_sd_fraction * (np.log(sigma_range[1]) - np.log(sigma_range[0]))

        # Use scipy's truncated normal PDF
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

        # Outer product to build 2D prior over grid
        prior = np.outer(beta_prior, sigma_prior)
        prior /= np.sum(prior)  # Normalize to sum = 1

    else:
        raise ValueError(f"Unknown prior_type: {prior_type}")

    # Reused code: creates transition matrices for all (β, σ) pairs in the grid.
    T_grid = np.empty((M, M, K, K))  # Shape: [beta, sigma, K, K]
    for i, beta in enumerate(beta_vals):
        for j, sigma in enumerate(sigma_vals):
            model = RampModelHMM(K=K, beta=beta, sigma=sigma, dt=dt)
            T_grid[i, j] = model.T  # Store transition matrix

    # Poisson rates used in likelihood evaluation
    lambdas = fixed_rh * x_grid * dt  # Shape: (K,)

    # Likelihood Computation Over Grid
    # Reused and parallelized as above

    def compute_log_likelihood(i, j):
        Ps = T_grid[i, j]
        ll_total = 0.0
        for y in spike_trains:
            ll = poisson_logpdf(y, lambdas)  # Observation likelihood
            #alphas = np.zeros_like(ll)
            #logZ = inference.forward_pass(pi0, Ps, ll, alphas)
            logZ = hmm_normalizer(pi0, Ps, ll)  # Log-normalizer (forward algorithm)
            ll_total += logZ
        return ll_total

    # Run in parallel across grid points
    log_likelihoods = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(compute_log_likelihood)(i, j)
        for i in range(M)
        for j in range(M)
    )

    # Reshape to 2D grid
    log_likelihoods = np.array(log_likelihoods).reshape(M, M)

    # combine prior and likelihood for full posterior
    # Stability trick: subtract max log-likelihood before exponentiating
    log_prior = np.log(prior)
    log_unnorm = log_likelihoods + log_prior
    marginal_log_likelihood = logsumexp(log_unnorm)  # log of marginal likelihood
    posterior = np.exp(log_unnorm - marginal_log_likelihood)  # Normalize posterior
    # marginal_likelihood = np.exp(marginal_log_likelihood)  

    # Marginal likelihood approximation via numerical integration (Riemann sum)
    
    if plot:
        # Reused plotting logic for visualizing posterior over parameters
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        # Contour plot of posterior
        B, S = np.meshgrid(beta_vals, sigma_vals, indexing='ij')
        contour = ax.contourf(S, B, posterior, levels=30, cmap='autumn')
        cbar = fig.colorbar(contour, ax=ax, label='Posterior Probability')

        # Mark true parameter location
        ax.scatter(true_sigma, true_beta, color='black', edgecolors='white', label='True Params', linewidth=2, s=200)
        ax.set_xlabel(r'$\sigma$', fontsize=20)
        ax.set_ylabel(r'$\beta$', fontsize=20)
        ax.set_title(f'Ramp Model Posterior (N={N}, Prior={prior_type})')
        ax.legend(fontsize=18)

    # --------------------
    # RETURN values:
    # - posterior: normalized grid posterior
    # - beta_vals, sigma_vals: grid axes
    # - marginal_likelihood: p(D | model), useful for Bayes factors
    # - ax: plotting axis (can be reused)
    return posterior, beta_vals, sigma_vals, marginal_log_likelihood, ax

def step_inference_scan(
    true_m=100.0,
    true_r=4,
    T=500,
    N=100,
    m_range=(0.25, 0.75),
    r_range=(1, 6),
    M=30,
    R_low=5.0,
    R_high=50.0,
    seed=42,
    n_jobs=-1,
    prior_type='uniform',  # or 'gaussian'
    prior_sd_fraction=0.25,
    ax=None,
    plot=True,
    spktrn_arg=None
):

    dt = 1/T
    m_range = tuple(np.array(m_range) * T)

    np.random.seed(seed)

    if spktrn_arg is None:
        # Simulate spike trains using true parameters
        step_model_true = StepModelHMM(m=true_m, r=true_r, dt=dt, exact=True)
        spike_trains = np.array([
            step_model_true.simulate_spikes(n_steps=T, R_low=R_low, R_high=R_high, dt=dt)[2]
            for _ in range(N)
        ])
    else:
        spike_trains = spktrn_arg

    # Grid values
    m_vals = np.linspace(*m_range, M)
    r_vals = np.arange(r_range[0], r_range[1] + 1)      
    M_m = len(m_vals)   # 30  ←  number of m-values  (continuous axis)
    M_r = len(r_vals)   #  6  ←  number of r-values  (discrete axis)                               


    d_m = (m_range[1] - m_range[0]) / (M_m - 1)
    # d_r = (r_range[1] - r_range[0]) / (M - 1)
    grid_area = d_m 

    if prior_type == 'uniform':
        prior = np.ones((M_m, M_r))
        prior *= d_m
        prior /= prior.sum()
    elif prior_type == 'gaussian':
        from scipy.stats import truncnorm

        m_mu = np.mean(m_range)
        r_mu = np.mean(r_range)

        m_sd = prior_sd_fraction * (m_range[1] - m_range[0])
        r_sd = prior_sd_fraction * (r_range[1] - r_range[0])

        m_prior = truncnorm.pdf(
            m_vals, (m_range[0]-m_mu)/m_sd, (m_range[1]-m_mu)/m_sd, loc=m_mu, scale=m_sd
        )
        r_prior = truncnorm.pdf(
            r_vals, (r_range[0]-r_mu)/r_sd, (r_range[1]-r_mu)/r_sd, loc=r_mu, scale=r_sd
        )

        prior = np.outer(m_prior, r_prior)
        prior /= np.sum(prior)
    else:
        raise ValueError(f"Unknown prior_type: {prior_type}")

    def compute_log_likelihood(i, j):
   
        # --- grid point -------------------------------------------------
        m = m_vals[i]
        r = int(r_vals[j])                    # make sure it’s an int
        model = StepModelHMM(m=m, r=r, dt=dt, exact=True)

        K     = model.K                      # K = r + 1
        Tmat  = model.T

        # π₀ shifted forward by the compulsory r transitions
        pi0 = np.zeros(K)
        pi0[0] = 1.0
        pi0_shift = pi0 @ np.linalg.matrix_power(Tmat, r)

        # emission rates (Hz × dt) — low everywhere except last state
        rates = np.full(K, R_low * dt)
        rates[-1] = R_high * dt

        # --- accumulate log evidence over trials -----------------------
        ll_total = 0.0
        for spikes in spike_trains:           # ‘spikes’ is 1-D, shape (T+1,)
            # poisson_logpdf(counts 1-D, rates 1-D)  →  (T+1, K) :contentReference[oaicite:1]{index=1}
            ll = poisson_logpdf(spikes, rates)          # exactly 2-D

            logZ = hmm_normalizer(pi0_shift, Tmat, ll)
            ll_total += logZ

        return ll_total



    from joblib import Parallel, delayed
    log_likelihoods = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(compute_log_likelihood)(i, j)
        for i in range(M_m)
        for j in range(M_r)
    )
    log_likelihoods = np.array(log_likelihoods).reshape(M_m, M_r)

    log_prior = np.log(prior)
    log_unnorm = log_likelihoods + log_prior
    marginal_log_likelihood = logsumexp(log_unnorm)  # log of marginal likelihood
    posterior = np.exp(log_unnorm - marginal_log_likelihood)  # Normalize posterior
    # marginal_likelihood = np.exp(marginal_log_likelihood)  # Marginal likelihood approximation


    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        # Use imshow: row = m, col = r
        im = ax.imshow(
            posterior,
            extent=[(r_vals[0] - 0.5), (r_vals[-1] + 1), m_vals[0], m_vals[-1]],
            aspect='auto',
            origin='lower',
            cmap='Blues'
        )
        cbar = fig.colorbar(im, ax=ax, label='Posterior Probability')

        # Make sure r axis is treated as discrete
        ax.set_xticks(r_vals)
        ax.set_xticklabels([str(r) for r in r_vals])
        ax.set_xlabel('r (discrete)', fontsize=16)
        ax.set_ylabel('m (continuous)', fontsize=16)

        ax.scatter(true_r, true_m, color='black', edgecolors='white',
                   label='True Params', linewidth=2, s=200)

        ax.set_title(f'Step Model Posterior (N={N}, Prior={prior_type})')
        ax.legend(fontsize=18)
    '''
    if plot:
        # Plot posterior
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        M_vals, R_vals = np.meshgrid(m_vals, r_vals, indexing='ij')
        contour = ax.contourf(R_vals, M_vals, posterior, levels=30, cmap='Blues')
        cbar = fig.colorbar(contour, ax=ax, label='Posterior Probability')
        ax.scatter(true_r, true_m, color='black', edgecolors='white', label='True Params', linewidth=2, s=200)
        ax.set_xlabel('r', fontsize=16)
        ax.set_ylabel('m', fontsize=16)
        ax.set_title(f'Step Model Posterior (N={N}, Prior={prior_type})')
        ax.legend(fontsize=18)
        '''
    return posterior, m_vals, r_vals, marginal_log_likelihood, ax


def bayes_model_selection_scan(spike_trains, T_grid, lambdas,
                               m_vals, r_vals, R_low, R_high, dt):
    """
    Compare Ramp and Step models for each spike train using marginal likelihood and Bayes factor.

    Parameters:
        spike_trains (np.ndarray): Shape (N, T)
        T_grid (np.ndarray): Grid of transition matrices for Ramp model [I, J, K, K]
        lambdas (np.ndarray): Firing rates for Ramp model [K]
        m_vals (np.ndarray): Step model change points
        r_vals (np.ndarray): Step model ramping rates
        R_low, R_high (float): Step model rate bounds
        dt (float): Time bin size

    Returns:
        results (list of dict): For each spike train, includes:
            - 'ml_ramp': best marginal likelihood under Ramp
            - 'ml_step': best marginal likelihood under Step
            - 'bayes_factor': ratio of ramp vs step
            - 'best_ramp_idx': (i, j) index in T_grid
            - 'best_step_idx': (i, j) index in m_vals x r_vals
    """

    N = spike_trains.shape[0]
    results = []
    pi0 = np.array([1.0, 0.0])  # Always start in state 0
    K = 2

    for trial_idx in range(N):
        y = spike_trains[trial_idx]

        # --- Ramp Model ---
        best_ramp_ll = -np.inf
        best_ramp_idx = None

        I, J = T_grid.shape[:2]
        for i in range(I):
            for j in range(J):
                Ps = T_grid[i, j]
                ll = poisson_logpdf(y, lambdas)  # (T, K)
                logZ = hmm_normalizer(pi0, Ps, ll)
                if logZ > best_ramp_ll:
                    best_ramp_ll = logZ
                    best_ramp_idx = (i, j)

        # --- Step Model ---
        best_step_ll = -np.inf
        best_step_idx = None

        for i, m in enumerate(m_vals):
            for j, r in enumerate(r_vals):
                model = StepModelHMM(m=m, r=r, dt=dt, exact=True)
                Ps = model.T
                rates = np.array([R_low * dt, R_high * dt])
                ll = poisson_logpdf(y, rates)  # (T, 2)
                logZ = hmm_normalizer(pi0, Ps, ll)
                if logZ > best_step_ll:
                    best_step_ll = logZ
                    best_step_idx = (i, j)

        # Bayes factor
        bf = np.exp(best_ramp_ll - best_step_ll)

        results.append({
            'ml_ramp': np.exp(best_ramp_ll),
            'ml_step': np.exp(best_step_ll),
            'bayes_factor': bf,
            'best_ramp_idx': best_ramp_idx,
            'best_step_idx': best_step_idx
        })

    return results
