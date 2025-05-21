import numpy as np

def fanofac(spikes, bin_width, T=1000):
    edges = np.arange(0, T + 1, bin_width)
    
    binned = np.add.reduceat(spikes, edges[:-1], axis=1)
    
    mean = binned.mean(axis=0)
    var = binned.var(axis=0)
    
    epsilon = 1e-10
    safe_mean = np.where(mean < epsilon, 0, mean)
    fano = np.divide(var, safe_mean, out=np.zeros_like(var), where=safe_mean != 0)
    
    times = edges[:-1] + bin_width / 2
    
    return times, fano
