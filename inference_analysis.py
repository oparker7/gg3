# Contains functions used for task 2.3
# Combine into another file if needed, though I would keep it separate for clarity for now

import numpy as np
import matplotlib.pyplot as plt
from HMM_models import RampModelHMM, StepModelHMM
import inference
from HMM_inference import perform_ramp_inference, perform_step_inference


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

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

def plot_step_inference_example(true_tau, est_tau, P_high, title='Step Model Inference'):
    """Plot posterior probability of high state and true jump time"""
    plt.figure(figsize=(12, 4))
    plt.plot(P_high, 'r-', label='$P(s_t | n_{1:T})$', linewidth=2)
    plt.axvline(x=true_tau, color='b', linestyle='--', 
                label=f'True jump time = {true_tau}')
    plt.axvline(x=est_tau, color='r', linestyle='--', 
                label=f'Estimated jump time = {est_tau}')
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
    plt.imshow(errors, cmap='RdYlGn_r', origin='lower', aspect='auto',
               extent=[param2_vals[0], param2_vals[-1], 
                      param1_vals[0], param1_vals[-1]])
    plt.colorbar(label='Mean Absolute Error')
    plt.xlabel(param2_name)
    plt.ylabel(param1_name)
    plt.title(title)
    plt.show()

"""
Code snippets for improving Task 2.3 HMM inference analysis.
Each section contains code and explanations for specific improvements.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from HMM_inference import RampModelHMM, StepModelHMM, perform_ramp_inference, perform_step_inference

# ============================================================================
# 1. QUANTITATIVE EVALUATION METRICS
# ============================================================================

def evaluate_inference(true_states, inferred_states, smoothing=True):
    """
    Comprehensive evaluation of inference performance using multiple metrics.
    
    Metrics:
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error
    - Correlation: Pearson correlation coefficient
    - Accuracy: Classification accuracy (for step model)
    """
    # Mean absolute error
    mae = np.mean(np.abs(true_states - inferred_states))
    
    # Root mean squared error 
    rmse = np.sqrt(np.mean((true_states - inferred_states)**2))
    
    # Correlation coefficient
    corr = np.corrcoef(true_states, inferred_states)[0,1]
    
    # Classification accuracy for step model
    if len(np.unique(true_states)) == 2:  # Binary states
        acc = np.mean((inferred_states > 0.5) == (true_states > 0.5))
    else:
        acc = (true_states - inferred_states)/true_states
        
    return {
        'mae': mae,
        'rmse': rmse, 
        'correlation': corr,
        'accuracy': acc,
        'method': 'smoothing' if smoothing else 'filtering'
    }

# ============================================================================
# 2. PARAMETER REGIME ANALYSIS
# ============================================================================

def analyze_parameter_regime(model_type='ramp', param_name='beta', 
                           param_range=None, n_trials=10):
    """
    Analyze how inference accuracy depends on model parameters.
    
    Parameters:
    - model_type: 'ramp' or 'step'
    - param_name: parameter to analyze ('beta', 'sigma', 'm', 'r', etc.)
    - param_range: range of parameter values to test
    - n_trials: number of trials per parameter value
    """
    # Common parameters
    dt = 0.01  # time step
    T = 500  # trial duration
    N = n_trials
    
    if param_range is None:
        if model_type == 'ramp':
            if param_name == 'beta':
                param_range = np.linspace(0.05, 0.3, 10)  # beta values
            elif param_name == 'sigma':
                param_range = np.linspace(0.1, 0.4, 10)  # sigma values
        else:  # step model
            if param_name == 'm':
                param_range = np.linspace(25, 100, 10)  # mean jump time
            elif param_name == 'r':
                param_range = np.array([5, 10, 15, 20])  # shape parameter
            
    errors = []
    for param in param_range:
        trial_errors = []
        for _ in range(n_trials):
            if model_type == 'ramp':
                # Default ramp parameters
                K = 50
                beta = 0.1 if param_name != 'beta' else param
                sigma = 0.2 if param_name != 'sigma' else param
                R_h = 30.0 if param_name != 'sigma' else param
                pi0 = None if param_name != 'pi0' else param
                
                # Run inference
                _, _, mae = perform_ramp_inference(
                    K=K, beta=beta, sigma=sigma, dt=dt, T=T, R_h=R_h, N=1, pi0=pi0
                )
                trial_errors.append(mae[0])  # Get MAE for single trial
                
            else:  # step model
                # Default step parameters
                m = 50 if param_name != 'm' else param
                r = 10 if param_name != 'r' else param
                R_low = 5.0 if param_name != 'R_low' else param
                R_high = 50.0 if param_name != 'R_high' else param
                exact = True
                
                # Run inference
                _, _, mae = perform_step_inference(
                    m=m, r=r, dt=dt, T=T, R_low=R_low, R_high=R_high,
                    N=1, exact=exact
                )
                trial_errors.append(mae[0])  # Get MAE for single trial
            
        errors.append(np.mean(trial_errors))
        
    return param_range, errors

def plot_parameter_analysis(param_range, errors, param_name, model_type):
    """Plot parameter analysis results with confidence intervals."""
    plt.figure(figsize=(10,6))
    plt.plot(param_range, errors, 'o-', label='Mean Error')
    plt.fill_between(param_range, 
                    np.array(errors) - np.std(errors),
                    np.array(errors) + np.std(errors),
                    alpha=0.2)
    plt.xlabel(param_name)
    plt.ylabel('Mean Absolute Error')
    plt.title(f'Inference Accuracy vs {param_name} ({model_type.title()} Model)')
    plt.grid(True)
    plt.legend()
    plt.show()

def pi0_plot_parameter_analysis(param_range, errors, param_name, model_type):
    """Plot parameter analysis results with confidence intervals for p0"""
    state_indices = [np.argmax(pi0) for pi0 in param_range]
    plt.figure(figsize=(10,6))
    plt.plot(state_indices, errors, 'o-', label='Mean Error')
    plt.fill_between(state_indices, 
                    np.array(errors) - np.std(errors),
                    np.array(errors) + np.std(errors),
                    alpha=0.2)
    plt.xlabel(param_name)
    plt.ylabel('Mean Absolute Error')
    plt.title(f'Inference Accuracy vs {param_name} ({model_type.title()} Model)')
    plt.grid(True)
    plt.legend()
    plt.show()

# ============================================================================
# 3. SMOOTHING VS FILTERING COMPARISON
# ============================================================================

def compare_smoothing_filtering(model_type='ramp', n_trials=10):
    """
    Compare smoothing and filtering performance across multiple trials.
    
    Parameters:
    - model_type: 'ramp' or 'step'
    - n_trials: number of trials to run
    """
    # Common parameters
    dt = 0.01  # time step
    T = 500  # trial duration
    N = n_trials
    
    if model_type == 'ramp':
        # Ramp model parameters
        K = 50  # number of discrete states
        beta = 0.1  # drift parameter
        sigma = 0.2  # diffusion parameter
        R_h = 30.0  # maximum firing rate
        
        # Run inference with smoothing
        true_states, inferred_smooth, mae_smooth = perform_ramp_inference(
            K=K, beta=beta, sigma=sigma, dt=dt, T=T, R_h=R_h, N=N, use_filter=False
        )
        
        # Run inference with filtering
        _, inferred_filter, mae_filter = perform_ramp_inference(
            K=K, beta=beta, sigma=sigma, dt=dt, T=T, R_h=R_h, N=N, use_filter=True
        )
        
    else:  # step model
        # Step model parameters
        m = 50  # mean jump time
        r = 10  # shape parameter
        R_low = 5.0  # low state firing rate
        R_high = 50.0  # high state firing rate
        exact = True  # use exact (r+1)-state model
        
        # Run inference with smoothing
        true_states, inferred_smooth, mae_smooth = perform_step_inference(
            m=m, r=r, dt=dt, T=T, R_low=R_low, R_high=R_high,
            N=N, exact=exact, use_filter=False
        )
        
        # Run inference with filtering
        _, inferred_filter, mae_filter = perform_step_inference(
            m=m, r=r, dt=dt, T=T, R_low=R_low, R_high=R_high,
            N=N, exact=exact, use_filter=True
        )
    
    # Plot comparison
    metrics = ['mae', 'rmse', 'correlation']
    if model_type == 'step':
        metrics.append('accuracy')
        
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
    for i, metric in enumerate(metrics):
        if metric == 'mae':
            smooth_vals = mae_smooth
            filter_vals = mae_filter
        if metric == 'accuracy':
            smooth_vals = (true_states - inferred_smooth)/true_states
            filter_vals = (true_states - inferred_filter)/true_states
        else:
            # Compute other metrics
            smooth_vals = [evaluate_inference(true_states[j], inferred_smooth[j], smoothing=True)[metric] 
                         for j in range(n_trials)]
            filter_vals = [evaluate_inference(true_states[j], inferred_filter[j], smoothing=False)[metric] 
                         for j in range(n_trials)]
        
        axes[i].boxplot([smooth_vals, filter_vals], 
                       labels=['Smoothing', 'Filtering'])
        axes[i].set_title(f'{metric.upper()} Comparison')
        axes[i].grid(True)
    
    plt.suptitle(f'Smoothing vs Filtering Performance ({model_type.title()} Model)')
    plt.tight_layout()
    plt.show()

# ============================================================================
# 4. CONFIDENCE INTERVALS FOR INFERRED STATES
# ============================================================================

def plot_inference_with_confidence(true_states, inferred_states, 
                                 confidence_intervals, title=''):
    """
    Plot inferred states with confidence intervals.
    
    Parameters:
    - true_states: true hidden states
    - inferred_states: inferred states (smoothed or filtered)
    - confidence_intervals: tuple of (lower, upper) confidence bounds
    - title: plot title
    """
    plt.figure(figsize=(12, 6))
    
    # Plot true states
    plt.plot(true_states, 'k-', label='True states', alpha=0.5)
    
    # Plot inferred states with confidence intervals
    plt.plot(inferred_states, 'r--', label='Inferred states')
    plt.fill_between(range(len(inferred_states)),
                    confidence_intervals[0],
                    confidence_intervals[1],
                    color='r', alpha=0.2)
    
    plt.legend()
    plt.title(title)
    plt.grid(True)
    plt.show()

def compute_confidence_intervals(inferred_states, n_bootstrap=1000, 
                               confidence_level=0.95):
    """
    Compute confidence intervals using bootstrap resampling.
    
    Parameters:
    - inferred_states: inferred states
    - n_bootstrap: number of bootstrap samples
    - confidence_level: confidence level for intervals
    """
    n_states = len(inferred_states)
    bootstrap_samples = np.zeros((n_bootstrap, n_states))
    
    for i in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_states, n_states, replace=True)
        bootstrap_samples[i] = inferred_states[indices]
    
    # Compute confidence intervals
    alpha = (1 - confidence_level) / 2
    lower = np.percentile(bootstrap_samples, alpha * 100, axis=0)
    upper = np.percentile(bootstrap_samples, (1 - alpha) * 100, axis=0)
    
    return lower, upper

# ============================================================================
# Improved version of the confidence interval computation
# ============================================================================

def compute_simulation_confidence_intervals(
    simulate_func,
    n_simulations=100,
    confidence_level=0.95,
    extract_inferred=True,
    **kwargs
):
    """
    Runs a simulation function multiple times and computes the confidence interval
    of the average inferred signal at each time step.

    Parameters:
    - simulate_func: function to simulate, should return inferred signal as 2nd return value
    - n_simulations: number of times to run the simulation
    - confidence_level: confidence level for interval (e.g., 0.95)
    - extract_inferred: whether to use second return value of the function
    - **kwargs: keyword arguments to pass to simulate_func

    Returns:
    - mean_signal: mean inferred signal over all simulations
    - mean_over_trials: mean inferred signal over all trials
    - lower: lower bound of confidence interval
    - upper: upper bound of confidence interval
    """
    inferred_list = []

    for _ in range(n_simulations):
        result = simulate_func(**kwargs)
        inferred = result[1] if extract_inferred else result
        inferred_list.append(inferred)

    inferred_array = np.array(inferred_list)  # shape: (n_simulations, n_trials, T)

    # Compute mean over simulations
    mean_signal = inferred_array.mean(axis=0)  # shape: (n_trials, T)

    # Compute confidence intervals over simulations of the *mean* signal
    mean_over_trials = inferred_array.mean(axis=1)  # shape: (n_simulations, T)
    alpha = (1 - confidence_level) / 2
    lower = np.percentile(mean_over_trials, alpha * 100, axis=0)
    upper = np.percentile(mean_over_trials, (1 - alpha) * 100, axis=0)

    return mean_signal, mean_over_trials, lower, upper

# ============================================================================
# 5. OBSERVATION DURATION ANALYSIS
# ============================================================================

def analyze_observation_duration(model_type='ramp', durations=None, n_trials=10):
    """
    Analyze how inference accuracy depends on observation duration.
    
    Parameters:
    - model_type: 'ramp' or 'step'
    - durations: list of observation durations to test
    - n_trials: number of trials per duration
    """
    if durations is None:
        durations = [100, 200, 500, 1000, 2000]  # Default durations
        
    # Common parameters
    dt = 0.01  # time step
    
    errors = []
    for T in durations:
        trial_errors = []
        for _ in range(n_trials):
            if model_type == 'ramp':
                # Ramp model parameters
                K = 50
                beta = 0.1
                sigma = 0.2
                R_h = 30.0
                
                # Run inference
                _, _, mae = perform_ramp_inference(
                    K=K, beta=beta, sigma=sigma, dt=dt, T=T, R_h=R_h, N=1
                )
                trial_errors.append(mae[0])  # Get MAE for single trial
                
            else:  # step model
                # Step model parameters
                m = 50
                r = 10
                R_low = 5.0
                R_high = 50.0
                exact = True
                
                # Run inference
                _, _, mae = perform_step_inference(
                    m=m, r=r, dt=dt, T=T, R_low=R_low, R_high=R_high,
                    N=1, exact=exact
                )
                trial_errors.append(mae[0])  # Get MAE for single trial
            
        errors.append(np.mean(trial_errors))
        
    return durations, errors

def plot_duration_analysis(durations, errors, model_type):
    """Plot observation duration analysis results."""
    plt.figure(figsize=(10,6))
    plt.plot(durations, errors, 'o-')
    plt.xlabel('Observation Duration')
    plt.ylabel('Mean Absolute Error')
    plt.title(f'Inference Accuracy vs Observation Duration ({model_type.title()} Model)')
    plt.grid(True)
    plt.xscale('log')  # Log scale for duration
    plt.show()

# ============================================================================
# Example Usage
# ============================================================================

'''if __name__ == "__main__":
    # Example 1: Compare smoothing vs filtering
    compare_smoothing_filtering(model_type='ramp', n_trials=10)
    
    # Example 2: Parameter regime analysis
    param_range, errors = analyze_parameter_regime(
        model_type='ramp',
        param_name='beta',
        param_range=np.linspace(0.05, 0.3, 10)
    )
    plot_parameter_analysis(param_range, errors, 'Beta', 'Ramp')
    
    # Example 3: Observation duration analysis
    durations, errors = analyze_observation_duration(
        model_type='ramp',
        durations=[100, 200, 500, 1000, 2000]
    )
    plot_duration_analysis(durations, errors, 'Ramp')
    


    # Define ramp‐HMM parameters
    K       = 50          # number of discrete levels
    beta    = 0.1         # drift parameter
    sigma   = 0.2         # diffusion parameter
    dt      = 0.01        # bin width (seconds)
    T       = 500         # bins per trial
    R_h     = 30.0        # max Poisson rate (Hz)
    N       = 20          # number of trials
    use_filter = False    # False = smoothing; True = filtering
    # Example 4: Plot with confidence intervals
    model = RampModelHMM()
    true_states, spikes = model.simulate()
    true_ramps, inferred_ramps, MAEs = perform_ramp_inference(
        K       = K,
        beta    = beta,
        sigma   = sigma,
        dt      = dt,
        T       = T,
        R_h     = R_h,
        N       = N,
        pi0     = None,       # default: start in state 0
        use_filter = use_filter
    )
    lower, upper = compute_confidence_intervals(inferred_ramps[0])
    plot_inference_with_confidence(
        true_states, 
        inferred_ramps[0],
        (lower, upper),
        'Ramp Model Inference with 95% Confidence Intervals'
    ) 
'''