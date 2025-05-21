import numpy as np
from scipy.ndimage import gaussian_filter1d

def fanoFac(spikes, bin_width, T=1000):
    edges = np.arange(0, T + 1, bin_width)
    
    binned = np.add.reduceat(spikes, edges[:-1], axis=1)
    
    mean = binned.mean(axis=0)
    var = binned.var(axis=0)
    
    epsilon = 1e-10
    safe_mean = np.where(mean < epsilon, 0, mean)
    fano = np.divide(var, safe_mean, out=np.zeros_like(var), where=safe_mean != 0)
    
    times = edges[:-1] + bin_width / 2
    
    return times, fano

def genFanos(dataset, datasize, T=1000, bw=20):

    fanos = np.array([fanoFac(dataset[i],
                    bin_width=bw,
                    T=T)[1]
                    for i in range(datasize*2)]
                )

    n = len(fanos[0])
    clip_fac = 0.1 # fraction of array to remove from start and end
    fanos = np.array(fanos[:, int(np.floor(n*clip_fac)) : int(np.ceil(n*(1-clip_fac)))])
    fanos = gaussian_filter1d(fanos, sigma=1, axis=1)
    fanos_clean = np.nan_to_num(fanos, nan=0.0)

    return fanos_clean

def maxFanoFac(fanos, threshold=1.22):

    f_max = np.max(fanos, axis=1)
    classifier = (f_max > threshold).astype(int)

    return classifier

# notes: genFanoClassifyMax generates the fano factor arrays
# and then classifies based on the maximum for each model
# it would be nice to split into genFanos and maxFanoFac 
# so that the fano factors are generated and can then be used for different
# analyses but the function only seems to work when as one - have a look at ths later
def genFanoClassifyMax(dataset, datasize, T=1000, bw=20, threshold=1.22):

    fanos = np.array([fanoFac(dataset[i],
                    bin_width=bw,
                    T=T)[1]
                    for i in range(datasize*2)]
                )

    n = len(fanos[0])
    clip_fac = 0.1 # fraction of array to remove from start and end
    fanos = np.array(fanos[:, int(np.floor(n*clip_fac)) : int(np.ceil(n*(1-clip_fac)))])
    fanos = gaussian_filter1d(fanos, sigma=1, axis=1)
    f_max = np.max(fanos, axis=1)
    classifier = (f_max > threshold).astype(int)

    return classifier

def maxFanoDeriv(dataset, datasize,T=1000):
    return    


