import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import convolve
from scipy.ndimage import gaussian_laplace



def fanoFac(spikes, bin_width, smooth=None, T=1000):
    edges = np.arange(0, T + 1, bin_width)
    
    binned = np.add.reduceat(spikes, edges[:-1], axis=1)
    
    mean = binned.mean(axis=0)
    var = binned.var(axis=0)
    
    epsilon = 1e-10
    safe_mean = np.where(mean < epsilon, 0, mean)
    fano = np.divide(var, safe_mean, out=np.zeros_like(var), where=safe_mean != 0)

    win = smooth // bin_width if smooth else None
    if smooth is not None:
        kernel = np.ones(int(win)) / win
        fano = convolve(fano, kernel, mode='same')

    times = edges[:-1] + bin_width / 2
    
    return times, fano

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
    clip_fac = 0.15 # fraction of array to remove from start and end
    fanos = np.array(fanos[:, int(np.floor(n*clip_fac)) : int(np.ceil(n*(1-clip_fac)))])
    fanos = gaussian_filter1d(fanos, sigma=1, axis=1)
    f_max = np.max(fanos, axis=1)
    classifier = (f_max > threshold).astype(int)

    return classifier

def maxFanoDeriv(dataset, datasize, T=1000, bw=20, threshold=0.002):

    fanos = np.array([fanoFac(dataset[i],
                    bin_width=bw,
                    smooth=100,
                    T=T)[1]
                    for i in range(datasize*2)]
                )
    times = np.array(fanoFac(dataset[0],
                             bin_width=bw,
                             smooth=100,
                             T=T)[0])
                
    n = len(fanos[0])
    clip_fac = 0.1 # fraction of array to remove from start and end
    fanos = np.array(fanos[:, int(np.floor(n*clip_fac)) : int(np.ceil(n*(1-clip_fac)))])
    times = np.array(times[int(np.floor(n*clip_fac)) : int(np.ceil(n*(1-clip_fac)))])
    
    fanos_smoothed = np.array([gaussian_filter1d(fanos[i], sigma=3) for i in range(datasize*2)])
    gradients = [np.gradient(fanos_smoothed[i], times) for i in range(datasize*2)]

    g_max = np.max(np.abs(gradients), axis=1)

    classifier = (g_max > threshold).astype(int)

    return   g_max, classifier 


# output values don't line up with `task1_runlocal.ipynb`
def maxLapKern(dataset, datasize, T=1000, bw=20, threshold=0.002):

    fanos = np.array([fanoFac(dataset[i],
                    bin_width=bw,
                    smooth=100,
                    T=T)[1]
                    for i in range(datasize*2)]
    )


    fanos_smoothed = np.array([gaussian_filter1d(fanos[i], sigma=3) for i in range(datasize*2)])
    clip_frac = 0.10
    n = len(fanos[0])
    fanos = np.array(fanos[:, int(np.floor(n*clip_frac)) : int(np.ceil(n*(1-clip_frac)))])
    laplacian1d = np.array([-1, 16, -30, 16, -1]) / 12

    responses = np.array([convolve(row, laplacian1d, mode='same')
                          for row in fanos_smoothed
    ])


    n = len(responses[0])
    responses = responses[:, int(np.floor(n*clip_frac)) : int(np.ceil(n*(1-clip_frac)))]

    l_max = np.max(np.abs(responses), axis=1)

    classifier = (l_max > threshold).astype(int)

    return l_max, classifier

def rangeLoG(dataset, datasize, T=1000, bw=20, threshold=0.001):
    

    fanos = np.array([fanoFac(dataset[i],
                    bin_width=bw,
                    smooth=100,
                    T=T)[1]
                    for i in range(datasize*2)]
    )

    n = len(fanos[0])
    clip_frac = 0.1
    fanos = np.array(fanos[:, int(np.floor(n*clip_frac)) : int(np.ceil(n*(1-clip_frac)))])
    LoG_responses = [gaussian_laplace(fanos[i, :], sigma=8) for i in range(datasize*2)]

    lg_max = np.max(LoG_responses, axis=1)
    lg_min = np.min(LoG_responses, axis=1)
    rg = lg_max - lg_min

    classifier = (rg > threshold).astype(int)

    return rg, classifier

def accuracy(predictions, num):
    ground_truth = np.array([0]*num + [1]*num)
    correct = predictions == ground_truth
    num_correct = np.sum(correct)
    acc = num_correct / len(predictions)

    return acc, correct

def fanoClassify(dataset, datasize, T=1000, bw=20):
    mff = genFanoClassifyMax(dataset, datasize)
    _, mfd = maxFanoDeriv(dataset, datasize)
    _, lgc = rangeLoG(dataset, datasize)

    cls = np.vstack([mff, mfd, lgc])
    prds = np.where(cls == 0, -1, 1)
    sum_prds = np.sum(prds, axis=0)
    classifier = (sum_prds > 0).astype(int)
    performance, _ = accuracy(classifier, datasize)
    disagreements = np.sum((mff != mfd) | (mff != lgc) | (mfd != lgc))

    return classifier, performance, disagreements 