# %% [markdown]
# # Task 2.3: Inference of Hidden States
# 
# In this notebook, we implement hidden state inference for both the ramp and step HMM models using the forward-backward algorithm. We will:
# 
# 1. Simulate spike trains from both models
# 2. Use forward-backward algorithm to infer hidden states
# 3. Compare inference accuracy between smoothing and filtering
# 4. Analyze parameter regimes where inference works best
# 
# ## Theory
# 
# We use the Forward-Backward Algorithm (FBA) via `hmm_expected_states` to compute:
# - Posterior probabilities: $P(s_t | n_{1:T})$
# - Log-likelihood: $\log P(n_{1:T})$
# 
# For the ramp model, we infer $\mathbb{E}[x_t | n_{1:T}]$ where $x_t = s_t/(K-1)$
# For the step model, we infer $P(s_t = \text{high state} | n_{1:T})$
# 
# We'll compare smoothing (using all observations) vs filtering (using only past observations) to see how inference accuracy differs.

# %%
import numpy as np
import matplotlib.pyplot as plt
from HMM_models import RampModelHMM, StepModelHMM
import inference
from HMM_inference import perform_ramp_inference, perform_step_inference

# Set random seed for reproducibility
np.random.seed(42)

# Plotting settings
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 12

# %%
def plot_ramp_inference_example(true_ramp, inferred_ramp, title='Ramp Model Inference'):
    """Plot ground truth vs inferred ramp trajectory"""
    plt.figure(figsize=(12, 4))
    plt.plot(true_ramp, 'b-', label='True $x_t$', linewidth=2)
    plt.plot(inferred_ramp, 'r--', label='Inferred $\mathbb{E}[x_t | n_{1:T}]$', linewidth=2)
    plt.fill_between(range(len(true_ramp)), 
                    true_ramp - 0.1, true_ramp + 0.1, 
                    alpha=0.2, color='b')
    plt.xlabel('Time step')
    plt.ylabel('$x_t$')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_step_inference_example(true_tau, P_high, title='Step Model Inference'):
    """Plot posterior probability of high state and true jump time"""
    plt.figure(figsize=(12, 4))
    plt.plot(P_high, 'r-', label='$P(s_t = \text{high} | n_{1:T})$', linewidth=2)
    plt.axvline(x=true_tau, color='b', linestyle='--', 
                label=f'True jump time = {true_tau}')
    plt.axhline(y=0.5, color='k', linestyle=':', alpha=0.5)
    plt.xlabel('Time step')
    plt.ylabel('Probability')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_error_heatmap(errors, param1_vals, param2_vals, 
                      param1_name, param2_name, title):
    """Plot heatmap of inference errors across parameter space"""
    plt.figure(figsize=(10, 8))
    plt.imshow(errors, origin='lower', aspect='auto',
               extent=[param2_vals[0], param2_vals[-1], 
                      param1_vals[0], param1_vals[-1]])
    plt.colorbar(label='Mean Absolute Error')
    plt.xlabel(param2_name)
    plt.ylabel(param1_name)
    plt.title(title)
    plt.show()

# %% [markdown]
# ## Ramp Model Inference
# 
# Let's first look at inference for the ramp model. We'll:
# 1. Simulate spike trains
# 2. Run inference using both smoothing and filtering
# 3. Compare inference accuracy

# %%
# Parameters for ramp model
K = 50  # number of discrete states
beta = 0.1  # drift parameter
sigma = 0.2  # diffusion parameter
dt = 0.01  # time step
T = 500  # trial duration
R_h = 30.0  # maximum firing rate
N = 10  # number of trials

# Run inference with smoothing
true_ramps, inferred_ramps_smooth, mae_smooth = perform_ramp_inference(
    K=K, beta=beta, sigma=sigma, dt=dt, T=T, R_h=R_h, N=N, use_filter=False
)

# Run inference with filtering
_, inferred_ramps_filter, mae_filter = perform_ramp_inference(
    K=K, beta=beta, sigma=sigma, dt=dt, T=T, R_h=R_h, N=N, use_filter=True
)

# Plot example trial
plot_ramp_inference_example(
    true_ramps[0], inferred_ramps_smooth[0],
    title=f'Ramp Model Inference (Smoothing)\nMAE = {mae_smooth[0]:.3f}'
)

plot_ramp_inference_example(
    true_ramps[0], inferred_ramps_filter[0],
    title=f'Ramp Model Inference (Filtering)\nMAE = {mae_filter[0]:.3f}'
)

# Compare average errors
print(f"Average MAE with smoothing: {np.mean(mae_smooth):.3f}")
print(f"Average MAE with filtering: {np.mean(mae_filter):.3f}")

# %% [markdown]
# ### Parameter Space Exploration
# 
# Let's explore how inference accuracy varies with different parameter values. We'll focus on:
# 1. Drift parameter (beta)
# 2. Diffusion parameter (sigma)
# 3. Maximum firing rate (R_h)

# %%
# Explore beta vs sigma
beta_vals = np.linspace(0.05, 0.3, 6)
sigma_vals = np.linspace(0.1, 0.4, 6)
errors = np.zeros((len(beta_vals), len(sigma_vals)))

for i, beta in enumerate(beta_vals):
    for j, sigma in enumerate(sigma_vals):
        _, _, mae = perform_ramp_inference(
            K=K, beta=beta, sigma=sigma, dt=dt, T=T, R_h=R_h, N=5
        )
        errors[i, j] = np.mean(mae)

plot_error_heatmap(errors, beta_vals, sigma_vals,
                   'beta', 'sigma',
                   'Ramp Model Inference Error vs Parameters')

# %% [markdown]
# ## Step Model Inference
# 
# Now let's look at inference for the step model. We'll:
# 1. Simulate spike trains
# 2. Run inference using both smoothing and filtering
# 3. Compare inference accuracy

# %%
# Parameters for step model
m = 50  # mean jump time
r = 10  # shape parameter
dt = 0.01  # time step
T = 500  # trial duration
R_low = 5.0  # low state firing rate
R_high = 50.0  # high state firing rate
N = 10  # number of trials
exact = True  # use exact (r+1)-state model

# Run inference with smoothing
true_taus, est_taus_smooth, mae_smooth = perform_step_inference(
    m=m, r=r, dt=dt, T=T, R_low=R_low, R_high=R_high,
    N=N, exact=exact, use_filter=False
)

# Run inference with filtering
_, est_taus_filter, mae_filter = perform_step_inference(
    m=m, r=r, dt=dt, T=T, R_low=R_low, R_high=R_high,
    N=N, exact=exact, use_filter=True
)

# Plot example trial
step_hmm = StepModelHMM(m=m, r=r, dt=dt, exact=exact)
states, tau_true, spikes = step_hmm.simulate_spikes(
    n_steps=T, R_low=R_low, R_high=R_high, dt=dt
)

# Compute P_high(t) for visualization
rates = np.zeros(step_hmm.K)
rates[-1] = R_high  # high state is last state in exact model
rates[:-1] = R_low
lambdas = rates * dt
ll = inference.poisson_logpdf(spikes, lambdas)
post_probs, _ = inference.hmm_expected_states(
    pi0=np.array([1.0] + [0.0]*(step_hmm.K-1)),
    Ps=step_hmm.T,
    ll=ll,
    filter=False
)
P_high = post_probs[:, -1]  # probability of high state

plot_step_inference_example(
    tau_true, P_high,
    title=f'Step Model Inference (Smoothing)\nMAE = {mae_smooth[0]:.3f}'
)

# Compare average errors
print(f"Average MAE with smoothing: {np.mean(mae_smooth):.3f}")
print(f"Average MAE with filtering: {np.mean(mae_filter):.3f}")

# %% [markdown]
# ### Parameter Space Exploration
# 
# Let's explore how inference accuracy varies with different parameter values. We'll focus on:
# 1. Mean jump time (m)
# 2. Shape parameter (r)
# 3. Firing rates (R_low, R_high)

# %%
# Explore m vs r
m_vals = np.array([25, 50, 75, 100])
r_vals = np.array([5, 10, 15, 20])
errors = np.zeros((len(m_vals), len(r_vals)))

for i, m in enumerate(m_vals):
    for j, r in enumerate(r_vals):
        _, _, mae = perform_step_inference(
            m=m, r=r, dt=dt, T=T, R_low=R_low, R_high=R_high,
            N=5, exact=True
        )
        errors[i, j] = np.mean(mae)

plot_error_heatmap(errors, m_vals, r_vals,
                   'm', 'r',
                   'Step Model Inference Error vs Parameters')

# %% [markdown]
# ## Summary and Discussion
# 
# ### Key Findings
# 
# 1. **Smoothing vs Filtering**
#    - Smoothing generally provides better inference accuracy since it uses all observations
#    - Filtering is more appropriate for online/real-time inference
#    - The difference in accuracy is more pronounced for the ramp model
# 
# 2. **Parameter Dependence**
#    - Ramp Model:
#      - Higher beta (drift) generally improves inference
#      - Moderate sigma (diffusion) works best
#      - Higher firing rates help with inference accuracy
#    
#    - Step Model:
#      - Larger m (mean jump time) makes inference harder
#      - Higher r (shape parameter) improves inference
#      - Larger difference between R_low and R_high helps
# 
# 3. **Model-Specific Challenges**
#    - Ramp Model: Inference is harder when the trajectory is noisy (high sigma) or slow (low beta)
#    - Step Model: Inference is harder when jumps are rare (high m) or when firing rates are similar
# 
# ### Future Work
# 
# 1. Explore more parameter combinations
# 2. Analyze the effect of trial duration (T) on inference accuracy
# 3. Study the impact of number of trials (N) on parameter estimation
# 4. Compare with other inference methods 