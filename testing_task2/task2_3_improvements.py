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
        acc = None
        
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
                R_h = 30.0
                
                # Run inference
                _, _, mae = perform_ramp_inference(
                    K=K, beta=beta, sigma=sigma, dt=dt, T=T, R_h=R_h, N=1
                )
                trial_errors.append(mae[0])  # Get MAE for single trial
                
            else:  # step model
                # Default step parameters
                m = 50 if param_name != 'm' else param
                r = 10 if param_name != 'r' else param
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
        
    errors = []
    for duration in durations:
        trial_errors = []
        for _ in range(n_trials):
            # Simulate data with specified duration
            if model_type == 'ramp':
                model = RampModelHMM()
                true_states, spikes = model.simulate(duration=duration)
                inferred = perform_ramp_inference(spikes, model)
            else:
                model = StepModelHMM()
                true_states, spikes = model.simulate(duration=duration)
                inferred = perform_step_inference(spikes, model)
                
            # Evaluate
            error = evaluate_inference(true_states, inferred['smoothed'])
            trial_errors.append(error['mae'])
            
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

if __name__ == "__main__":
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
    
    # Example 4: Plot with confidence intervals
    model = RampModelHMM()
    true_states, spikes = model.simulate()
    inferred = perform_ramp_inference(spikes, model)
    lower, upper = compute_confidence_intervals(inferred['smoothed'])
    plot_inference_with_confidence(
        true_states, 
        inferred['smoothed'],
        (lower, upper),
        'Ramp Model Inference with 95% Confidence Intervals'
    ) 
