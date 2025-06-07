import numpy as np
import matplotlib.pyplot as plt
import inference
from inference import hmm_normalizer, poisson_logpdf
from HMM_models import RampModelHMM, StepModelHMM
from joblib import Parallel, delayed

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
    exact: bool = False,
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

def ramp_grid_inference(
    true_beta=0.2,
    true_sigma=0.1,
    fixed_rh=50.0,      # fixed firing rate scale
    dt=0.002,
    T=100,
    N=50,
    K=50,
    beta_range=(0.05, 0.5),
    sigma_range=(0.05, 0.6),
    M=30,
    seed=42
):
    np.random.seed(seed)
    x_grid = np.arange(K) / (K - 1)
    pi0 = np.zeros(K)
    pi0[0] = 1.0

    # Simulate spike trains with true params
    ramp_model = RampModelHMM(K=K, beta=true_beta, sigma=true_sigma, dt=dt)
    spike_trains = []
    for _ in range(N):
        _, x_vals, spikes = ramp_model.simulate_spikes(n_steps=T, initial_state=0, R_h=fixed_rh, dt=dt)
        spike_trains.append(spikes)

    # Setup grids
    beta_vals = np.linspace(*beta_range, M)
    sigma_vals = np.linspace(*sigma_range, M)

    log_post = np.zeros((M, M))

    for i, beta in enumerate(beta_vals):
        for j, sigma in enumerate(sigma_vals):
            total_log_likelihood = 0.0
            model = RampModelHMM(K=K, beta=beta, sigma=sigma, dt=dt)
            Ps = model.T

            rates = fixed_rh * x_grid
            lambdas = rates * dt

            for y in spike_trains:
                ll = poisson_logpdf(y, lambdas)
                logZ = hmm_normalizer(pi0=pi0, Ps=Ps, ll=ll)
                total_log_likelihood += logZ

            log_post[i, j] = total_log_likelihood

    # Normalize log posterior
    log_post -= np.max(log_post)
    posterior = np.exp(log_post)
    posterior /= posterior.sum()

    # Plot posterior heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(
        posterior,
        extent=[sigma_vals[0], sigma_vals[-1], beta_vals[0], beta_vals[-1]],
        origin='lower',
        aspect='auto',
        cmap='viridis'
    )
    plt.colorbar(label='Posterior Probability')
    plt.scatter(true_sigma, true_beta, color='red', label='True Params')
    plt.xlabel('sigma')
    plt.ylabel('beta')
    plt.title(f'Ramp Model Posterior (N={N}, R_h={fixed_rh})')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return posterior, beta_vals, sigma_vals
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt

def ramp_grid_inference_optimized(
    true_beta=0.2,
    true_sigma=0.1,
    fixed_rh=50.0,
    dt=0.002,
    T=100,
    N=50,
    K=50,
    beta_range=(0.05, 0.5),
    sigma_range=(0.05, 0.6),
    M=30,
    seed=42,
    n_jobs=-1
):
    np.random.seed(seed)
    x_grid = np.arange(K) / (K - 1)
    pi0 = np.zeros(K)
    pi0[0] = 1.0

    # Simulate spike trains from true parameters
    ramp_model_true = RampModelHMM(K=K, beta=true_beta, sigma=true_sigma, dt=dt)
    spike_trains = np.array([
        ramp_model_true.simulate_spikes(n_steps=T, initial_state=0, R_h=fixed_rh, dt=dt)[2]
        for _ in range(N)
    ])  # shape: (N, T)

    # Parameter grid
    beta_vals = np.linspace(*beta_range, M)
    sigma_vals = np.linspace(*sigma_range, M)

    # Precompute transition matrices
    T_grid = np.empty((M, M, K, K))
    for i, beta in enumerate(beta_vals):
        for j, sigma in enumerate(sigma_vals):
            model = RampModelHMM(K=K, beta=beta, sigma=sigma, dt=dt)
            T_grid[i, j] = model.T

    # Precompute Poisson rates
    lambdas = fixed_rh * x_grid * dt  # shape: (K,)

    # Define log-likelihood function at each grid point
    def compute_log_likelihood(i, j):
        Ps = T_grid[i, j]
        ll_total = 0.0
        for y in spike_trains:
            ll = poisson_logpdf(y, lambdas)
            logZ = hmm_normalizer(pi0, Ps, ll)
            ll_total += logZ
        return ll_total

    # Run in parallel over the grid
    log_likelihoods = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(compute_log_likelihood)(i, j)
        for i in range(M)
        for j in range(M)
    )

    # Reshape results to grid
    log_post = np.array(log_likelihoods).reshape(M, M)

    # Normalize to get posterior
    log_post -= np.max(log_post)
    posterior = np.exp(log_post)
    posterior /= posterior.sum()

    # Plot posterior as a contour plot
    B, S = np.meshgrid(beta_vals, sigma_vals, indexing='ij')
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(S, B, posterior, levels=30, cmap='viridis')
    plt.colorbar(contour, label='Posterior Probability')
    plt.scatter(true_sigma, true_beta, color='red', edgecolors='white', label='True Params')
    plt.xlabel('σ (sigma)')
    plt.ylabel('β (beta)')
    plt.title(f'Ramp Model Posterior (N={N}, Rₕ={fixed_rh})')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return posterior, beta_vals, sigma_vals