import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D

# Parameters
beta = 0.1  # drift parameter
sigma = 0.2  # diffusion parameter
dt = 0.01  # time step

def gaussian_pdf(x, mu, sigma2):
    """Gaussian PDF with mean mu and variance sigma2"""
    return norm.pdf(x, loc=mu, scale=np.sqrt(sigma2))

def gaussian_cdf(x, mu, sigma2):
    """Gaussian CDF with mean mu and variance sigma2"""
    return norm.cdf(x, loc=mu, scale=np.sqrt(sigma2))

def transition_prob(x_next, x_curr):
    """Compute transition probability P(x_{t+1}|x_t)"""
    mu = x_curr + beta * dt
    sigma2 = sigma**2 * dt
    
    if x_curr == 1:
        return 1.0 if x_next == 1 else 0.0
    elif x_curr == 0:
        if x_next == 0:
            return gaussian_cdf(0, mu, sigma2)
        elif x_next == 1:
            return 1 - gaussian_cdf(1, mu, sigma2)
        else:
            return gaussian_pdf(x_next, mu, sigma2) / (gaussian_cdf(1, mu, sigma2) - gaussian_cdf(0, mu, sigma2))
    else:
        if x_next == 0:
            return gaussian_cdf(0, mu, sigma2)
        elif x_next == 1:
            return 1 - gaussian_cdf(1, mu, sigma2)
        else:
            return gaussian_pdf(x_next, mu, sigma2) / (gaussian_cdf(1, mu, sigma2) - gaussian_cdf(0, mu, sigma2))

def plot_transition_from_zero():
    """Plot transition probability from x_t = 0"""
    x_next = np.linspace(0, 1, 1000)
    x_curr = 0
    
    # Compute probabilities
    probs = np.array([transition_prob(x, x_curr) for x in x_next])
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_next, probs, 'b-', label='P(x_{t+1}|x_t=0)')
    plt.xlabel('x_{t+1}')
    plt.ylabel('Probability')
    plt.title('Transition Probability from x_t = 0')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_transition_surface():
    """Create 3D surface plot of transition probabilities"""
    # Create meshgrid for 3D plot
    x_curr = np.linspace(0, 1, 100)
    x_next = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x_curr, x_next)
    
    # Compute transition probabilities
    Z = np.zeros_like(X)
    for i in range(len(x_curr)):
        for j in range(len(x_next)):
            Z[j,i] = transition_prob(x_next[j], x_curr[i])
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel('x_t')
    ax.set_ylabel('x_{t+1}')
    ax.set_zlabel('P(x_{t+1}|x_t)')
    ax.set_title('Transition Probability Surface')
    fig.colorbar(surf)
    plt.show()

def evolve_distribution(initial_dist, n_steps=10):
    """Evolve distribution over time starting from initial_dist"""
    x = np.linspace(0, 1, 100)
    distributions = [initial_dist]
    
    for _ in range(n_steps):
        next_dist = np.zeros_like(x)
        for i, x_curr in enumerate(x):
            for j, x_next in enumerate(x):
                next_dist[j] += distributions[-1][i] * transition_prob(x_next, x_curr)
        distributions.append(next_dist)
    
    return x, distributions

def plot_evolution_from_impulse():
    """Plot evolution of distribution from initial impulse"""
    # Create initial impulse at x=0
    x = np.linspace(0, 1, 100)
    initial_dist = np.zeros_like(x)
    initial_dist[0] = 1.0  # Impulse at x=0
    
    # Evolve distribution
    x, distributions = evolve_distribution(initial_dist, n_steps=5)
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    X, Y = np.meshgrid(x, range(len(distributions)))
    Z = np.array(distributions)
    
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('Time Step')
    ax.set_zlabel('Probability')
    ax.set_title('Evolution of Distribution from Initial Impulse')
    fig.colorbar(surf)
    plt.show()

if __name__ == "__main__":
    # Generate all three visualizations
    print("Generating transition probability from x_t = 0...")
    plot_transition_from_zero()
    
    print("Generating transition probability surface...")
    plot_transition_surface()
    
    print("Generating evolution from initial impulse...")
    plot_evolution_from_impulse() 