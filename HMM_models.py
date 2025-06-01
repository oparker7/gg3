
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad

def _figure_grid(n_rows, n_cols, figsize=(12, 12)):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                             sharex=True, sharey=True, squeeze=False)
    for ax in axes.ravel():
        ax.set_xlim(0, 1000)
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("trial")
    return fig, axes

class RampModelHMM:
    """
    Hidden Markov Model implementation of the ramp model with boundary conditions.
    
    The continuous model follows:
    x_{t+1} = x_t + β*dt + σ*sqrt(dt)*ε_t
    
    With boundary conditions:
    - x_t cannot go below 0 (reflecting barrier)
    - x_t stays at 1 once it reaches it (absorbing barrier)
    """
    
    def __init__(self, K=50, beta=0.1, sigma=0.2, dt=0.01):
        """
        Initialize the HMM.
        
        Parameters:
        - K: Number of discrete states (grid points from 0 to 1)
        - beta: Drift parameter
        - sigma: Diffusion parameter
        - dt: Time step
        """
        self.K = K
        self.beta = beta
        self.sigma = sigma
        self.dt = dt
        
        # Create state grid: x_t = s_t / (K-1) where s_t ∈ {0, 1, ..., K-1}
        self.states = np.arange(K)
        self.x_values = self.states / (K - 1)
        
        # Construct transition matrix
        self.T = self._construct_transition_matrix()  # T is the class attribute for the transition matrix
        
    def _construct_transition_matrix(self):
        """Construct the transition matrix T where T[s,s'] = P(s_{t+1} = s' | s_t = s)"""
        T = np.zeros((self.K, self.K))
        
        for s in range(self.K):
            x_current = self.x_values[s]
            
            # Special case: absorbing state at x = 1 (s = K-1)
            if s == self.K - 1:
                T[s, s] = 1.0
                continue
            
            # For other states, calculate transition probabilities
            T[s, :] = self._calculate_transition_probs(x_current)
            
        return T

    def _calculate_transition_probs(self, x_current):
        probs   = np.zeros(self.K)
        dx      = 1.0 / (self.K - 1)
        mean    = x_current + self.beta * self.dt
        std     = self.sigma * np.sqrt(self.dt)

        # 1. mass that falls below 0  → bin 0   (works for any x_current)
        if std > 0:
            p_below0 = norm.cdf((0 - mean) / std)
        else:               # deterministic step
            p_below0 = 1.0 if mean < 0 else 0.0
        probs[0] += p_below0

        # 2. mass that falls above 1 → last bin
        if std > 0:
            p_above1 = 1 - norm.cdf((1 - mean) / std)
        else:
            p_above1 = 1.0 if mean > 1 else 0.0
        probs[-1] += p_above1

        # 3. interior bins (including 0 and K-1 again; harmless because non-overlapping)
        for s_prime in range(self.K):
            x_centre = self.x_values[s_prime]
            left  = x_centre - dx/2
            right = x_centre + dx/2
            # clip to [0,1] so we don’t re-add tails we already accounted for
            left  = max(left, 0.0)
            right = min(right, 1.0)
            if right > left:               # non-empty interval
                probs[s_prime] += self._integrate_gaussian(mean, std, left, right)

        # 4. renormalise
        probs /= probs.sum()
        return probs

    
    def _integrate_gaussian(self, mean, std, a, b):
        """Integrate Gaussian PDF from a to b"""
        if std == 0:
            return 1.0 if a <= mean <= b else 0.0
        
        return norm.cdf((b - mean) / std) - norm.cdf((a - mean) / std)
    
    def simulate(self, n_steps=500, initial_state=0):
        """
        Simulate the HMM for n_steps.
        
        Parameters:
        - n_steps: Number of time steps to simulate
        - initial_state: Initial state index (default: 0)
        
        Returns:
        - states: Array of state indices
        - x_values: Array of corresponding x_t values
        """
        states = np.zeros(n_steps + 1, dtype=int)
        states[0] = initial_state
        
        for t in range(n_steps):
            # Sample next state based on transition probabilities
            probs = self.T[states[t], :]
            states[t + 1] = np.random.choice(self.K, p=probs)
        

        x_values = self.x_values[states]
        return states, x_values
    
    def plot_transition_matrix(self, ax=None):
        """Plot the transition matrix as a heatmap"""
        if ax is None:
            ax = plt.gca()  # Get current axis if none provided

        im = ax.imshow(self.T, cmap='Blues', origin='lower')
        plt.colorbar(im, ax=ax, label='Transition Probability')
        ax.set_xlabel('Target State s\'')
        ax.set_ylabel('Current State s')
        ax.set_title(f'Transition Matrix (β={self.beta}, σ={self.sigma}, dt={self.dt})')

    
    def plot_stationary_distribution(self, ax=None, max_iter=1000, tol=1e-10):
        """Calculate and plot the stationary distribution"""
        # Start with uniform distribution

        pi = np.ones(self.K) / self.K
        
        # Power iteration to find stationary distribution
        for _ in range(max_iter):
            pi_new = pi @ self.T
            if np.linalg.norm(pi_new - pi) < tol:
                break
            pi = pi_new
        

        if ax is None:
            plt.figure(figsize=(10,6))
            ax = plt.gca()
            show_fig = True
        else:
            show_fig = False

        ax.plot(self.x_values, pi, 'b-', linewidth=2)
        ax.set_xlabel('x')
        ax.set_ylabel('Stationary Probability')
        ax.set_title(f'Stationary Distribution (β={self.beta}, σ={self.sigma})')
        ax.grid(True, alpha=0.3)

        if show_fig:
            plt.show()

        return pi

    def plot_trajectory_comparison(self, x_vals, label=None, ax=None):
        """ Plot the trajectory and its histogram.

        Parameters:
        - x_vals: Array of x(t) values (result from simulation)
        - label: Optional label for legend
        - ax: Optional tuple of matplotlib axes (trajectory_ax, hist_ax)
        """

        if ax is None:
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            show_fig = True
        else:
            show_fig = False

        trajectory_ax, hist_ax = ax

        # Plot trajectory
        trajectory_ax.plot(x_vals)
        trajectory_ax.set_title(f'Trajectory: β={self.beta}, σ={self.sigma}')
        trajectory_ax.set_ylabel('x(t)')
        trajectory_ax.set_xlabel('Time step')
        trajectory_ax.grid(True, alpha=0.3)

        # Plot histogram
        hist_ax.hist(x_vals, bins=20, alpha=0.7, density=True, label=label or f'β={self.beta}, σ={self.sigma}')
        hist_ax.set_xlabel('x')
        hist_ax.set_ylabel('Density')
        hist_ax.set_title('Distribution of States')
        hist_ax.legend()
        hist_ax.grid(True, alpha=0.3)

        if show_fig:
            plt.tight_layout()
            plt.show()

    def simulate_spikes(self, n_steps=500, initial_state=0, R_h=30.0, dt=None):
        """
        Simulate one trial of length n_steps for the ramp HMM, and return:
          - states:  array of length n_steps+1, each in {0,…,K-1}
          - x_vals:  array of length n_steps+1, x_vals[t] = states[t]/(K-1)
          - spikes:  array of length n_steps+1, where
                       rate[t] = R_h * x_vals[t],      or + baseline if desired
                       spikes[t] - Pois(rate[t]·dt)
        """
        if dt is None:
            dt = self.dt
        # 1) Draw the discrete chain exactly as simulate() does
        states = np.zeros(n_steps+1, dtype=int)
        states[0] = initial_state
        for t in range(n_steps):
            probs = self.T[states[t], :]
            states[t+1] = np.random.choice(self.K, p=probs)
        # 2) Convert to continuous x_t
        x_vals = states / float(self.K - 1)
        # 3) Turn into rate[t] = R_h * x_vals[t], then sample Poisson
        rates = R_h * x_vals
        spikes = np.random.poisson(lam = rates * dt)
        return states, x_vals, spikes


    def walk_plot(self, n_trials=5000, n_steps=1000, n_to_plot=5):

        trajectories = np.empty((n_trials, n_steps+1))
        
        for i in range(n_trials):
            _, trajectories[i, :]= self.simulate(n_steps=n_steps)

        for k in range(n_to_plot):
            plt.plot(trajectories[k], alpha=.6)
            plt.plot(trajectories.mean(0), color="k", lw=2, label="mean $x_t$")
            plt.ylim(0, 1.1)
            plt.xlim(0, n_steps)
            plt.title(rf"$\beta$={self.beta},  $\sigma$={self.sigma}", fontsize=20)

        plt.tight_layout()
        plt.show()

class StepModelHMM:
    def __init__(self, m, r=1,  dt=1.0, exact=False):
        """
        m: mean of the NB distribution
        r: shape parameter of NB (r=1 is geometric)
        T: transition matrix
        dt: timestep size
        exact: whether to use exact NB model (with r+1 states)
        """
        self.m = m
        self.r = r
        self.dt = dt
        self.exact = exact
        self.p = r / (m + r)
        self.K = r + 1 if exact else 2  # number of states
        self.T = self._construct_transition_matrix()

    def _construct_transition_matrix(self):
        T = np.zeros((self.K, self.K))
        if self.exact:
            for i in range(self.K - 1):
                T[i, i] = 1 - self.p
                T[i, i + 1] = self.p
            T[-1, -1] = 1.0  # absorbing state
        else:
            T[0, 0] = 1 - self.p
            T[0, 1] = self.p
            T[1, 1] = 1.0
        return T

    def simulate(self, n_steps=500, n_trials=10):
        all_trajectories = []
        all_jump_times = []

        for _ in range(n_trials):
            s = 0
            traj = [s]
            jump_time = None

            for t in range(n_steps):
                s = np.random.choice(self.K, p=self.T[s])
                traj.append(s)
                if not self.exact and s == 1 and jump_time is None:
                    jump_time = t
                elif self.exact and s == self.r and jump_time is None:
                    jump_time = t

            all_trajectories.append(traj)
            all_jump_times.append(jump_time if jump_time is not None else n_steps)

        return np.array(all_trajectories), np.array(all_jump_times)

    def plot_trajectories(self, trajectories, ax=None):

        if ax is None:
            fig, ax = plt.subplots(figsize=(10,6))
            show_fig = True
        else:
            show_fig = False


        for traj in trajectories:
            ax.plot(np.arange(len(traj)) * self.dt, traj)

        ax.set_xlabel("Time")
        ax.set_ylabel("State")
        ax.set_title("Step Model Trajectories")
        plt.grid()

        if show_fig:
            plt.tight_layout()
            plt.show()

    def plot_jump_histogram(self, jump_times, ax=None):

        if ax is None:
            fig, ax = plt.subplots(figsize=(10,6))
            show_fig = True
        else:
            show_fig = False


        ax.hist(jump_times * self.dt, bins=20, edgecolor='black')
        ax.set_xlabel("Jump Time")
        ax.set_ylabel("Count")
        ax.set_title("Histogram of Jump Times")
        plt.grid()

        if show_fig:
            plt.tight_layout()
            plt.show()

    def simulate_spikes(self, n_steps=500, R_low=5.0, R_high=50.0, dt=None):
        """
        Simulate one trial of the step HMM (2 state or r+1 state), and return:
          - states:   array of length n_steps+1, each in {0,…,K-1}
          - tau_true: integer index of first step into the 'high' state
          - spikes:   array of length n_steps+1, where
                        if states[t] < (high-state index) then rate = R_low
                        else then rate = R_high
                        spikes[t] - Pois(rate·dt)
        """
        if dt is None:
            dt = self.dt

        # Draw the discrete chain
        states   = np.zeros(n_steps+1, dtype=int)
        states[0] = 0
        tau_true = None
        for t in range(n_steps):
            states[t+1] = np.random.choice(self.K, p=self.T[states[t], :])
            # Detect first time we hit the “absorbing high”:
            if tau_true is None:
                if (not self.exact and states[t+1] == 1) \
                   or (self.exact and states[t+1] == self.r):
                    tau_true = t+1

        if tau_true is None:
            tau_true = n_steps  # if no jump, set it to final bin

        # 2) Build rate[t] and sample spikes[t]
        spikes = np.zeros(n_steps+1, dtype=int)
        for t in range(n_steps+1):
            if (not self.exact and states[t] == 1) or (self.exact and states[t] == self.r):
                rate_t = R_high
            else:
                rate_t = R_low
            spikes[t] = np.random.poisson(lam = rate_t * dt)

        return states, tau_true, spikes

