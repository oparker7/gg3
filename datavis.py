import numpy as np
import matplotlib.pyplot as plt
import models
from scipy.signal import convolve
from scipy.ndimage import gaussian_filter1d
import itertools
import seaborn as sns

def _find_bound_times_ramp(xs):
    T     = xs.shape[1]
    taus  = np.argmax(np.hstack((xs, np.ones((xs.shape[0], 1)))) >= 1., axis=1)
    taus[taus == 0] = -1
    taus[taus == T] = -1
    return taus

def _figure_grid(n_rows, n_cols, figsize=(12, 12)):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                             sharex=True, sharey=True, squeeze=False)
    for ax in axes.ravel():
        ax.set_xlim(0, 1000)
        ax.set_xlabel("Time (ms)", fontsize=20)
        ax.set_ylabel("x(t)", fontsize=20)
    return fig, axes

# Legend handles
_spike_bound_handles = [
    plt.Line2D([], [], marker="|", linestyle="", color="lightblue", label="spike"),
    plt.Line2D([], [], marker="|", linestyle="", color="red", label="bound reached")
]

def rampRasterPlot(beta_list, sigma_list,
                     n_trials=5000, T=1000, N_show=10):
    fig, ax = _figure_grid(len(beta_list), len(sigma_list))
    for i, b in enumerate(beta_list):
        for j, s in enumerate(sigma_list):
            ramp          = models.RampModel(beta=b, sigma=s)
            spikes, xs, _ = ramp.simulate(Ntrials=n_trials, T=T)

            events = [np.where(spikes[k] > 0)[0] for k in range(N_show)]
            taus   = _find_bound_times_ramp(xs)
            bound_events = [[t] if (k < N_show and taus[k] >= 0) else []
                            for k, t in enumerate(taus)]

            ax[i, j].eventplot(events, colors="lightcoral", lineoffsets=np.arange(N_show),
                               linelengths=.6)
            ax[i, j].eventplot(bound_events[:N_show], colors="black",
                               lineoffsets=np.arange(N_show), linelengths=.8, linewidths=2)

            ax[i, j].set_title(rf"$\beta$={b},  $\sigma$={s}", fontsize=20)

    ax[0, 0].legend(handles=_spike_bound_handles, loc="upper right", frameon=False)
    plt.tight_layout()
    return fig

def overall_ramp_walk_plot(beta_list, sigma_list, n_trials=5000, T=1000):
    n_to_plot = min(n_trials, 3)

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate a distinct color for each parameter combination
    color_cycle = sns.color_palette("husl", len(beta_list) * len(sigma_list))

    for idx, (b, s) in enumerate(itertools.product(beta_list, sigma_list)):
        ramp = models.RampModel(beta=b, sigma=s, x0=0)
        _, xs, _ = ramp.simulate(Ntrials=n_trials, T=T)

        # Plot a few individual trial paths
        for k in range(n_to_plot):
            ax.plot(xs[k], color=color_cycle[idx], alpha=0.75)

        # Plot the mean trajectory
        ax.plot(xs.mean(0), color=color_cycle[idx], lw=2, label=rf"$\beta$={b}, $\sigma$={s}")

    ax.set_ylim(0, 1.1)
    ax.set_xlim(0, T)
    ax.set_title("Ramp Walk Plot Across Parameter Space", fontsize=24)
    ax.set_xlabel("Time", fontsize=18)
    ax.set_ylabel("$x_t$", fontsize=18)
    ax.legend(frameon=False, fontsize=18, loc='upper left',
    bbox_to_anchor=(1.05, 0.75))  # X=1.05 moves it just outside the plot to the right)
    plt.tight_layout()
    plt.show()

def rampWalkPlot(beta_list, sigma_list,
                   n_trials=5000, T=1000):

    n_to_plot = min(n_trials, 5)

    fig, ax = _figure_grid(len(beta_list), len(sigma_list))
    for i, b in enumerate(beta_list):
        for j, s in enumerate(sigma_list):
            ramp          = models.RampModel(beta=b, sigma=s, x0=0)
            _, xs, _      = ramp.simulate(Ntrials=n_trials, T=T)

        
            for k in range(n_to_plot):
                ax[i, j].plot(xs[k], alpha=.6)
            ax[i, j].plot(xs.mean(0), color="k", lw=2, label="mean $x_t$")
            ax[i, j].set_ylim(0, 1.1)
            ax[i, j].set_xlim(0, T)
            ax[i, j].set_title(rf"$\beta$={b},  $\sigma$={s}", fontsize=20)

    ax[0, 0].legend(frameon=False)
    plt.suptitle('Continuous Ramp Model Walk Plot', fontsize=24)
    plt.tight_layout()
    plt.show()

def stepRasterPlot(m_list, r_list,
                     n_trials=5000, T=1000, N_show=10):
    fig, ax = _figure_grid(len(m_list), len(r_list))
    for i, m in enumerate(m_list):
        for j, r in enumerate(r_list):
            step                = models.StepModel(m=m, r=r)
            spikes, jumps, _    = step.simulate(Ntrials=n_trials, T=T)

            events       = [np.where(spikes[k] > 0)[0] for k in range(N_show)]
            jump_events  = [[jumps[k]] if k < N_show else [] for k in range(n_trials)]

            ax[i, j].eventplot(events, colors="lightblue", lineoffsets=np.arange(N_show), linelengths=.6)
            ax[i, j].eventplot(jump_events[:N_show], colors="black", lineoffsets=np.arange(N_show), linelengths=.8, linewidths=2)

            ax[i, j].set_title(rf"$m$={m},  $r$={r}", fontsize=20)

    ax[0, 0].legend(handles=_spike_bound_handles, loc="upper right", frameon=False)
    plt.tight_layout()
    plt.show()

def stepWalkPlot(m_list, r_list,
                   n_trials=5000, T=1000):
    fig, ax = _figure_grid(len(m_list), len(r_list))
    for i, m in enumerate(m_list):
        for j, r in enumerate(r_list):
            step                = models.StepModel(m=m, r=r)
            _, _, rates         = step.simulate(Ntrials=n_trials, T=T)

            for k in range(5):
                ax[i, j].plot(rates[k], alpha=.6)
            ax[i, j].plot(rates.mean(0), color="k", lw=2, label="mean rate")
            ax[i, j].set_ylim(0, 55)
            ax[i, j].set_title(rf"$m$={m},  $r$={r}", fontsize=9)

    ax[0, 0].legend(frameon=False)
    plt.tight_layout()
    plt.show()

def _pretty_grid(nr, nc, figsize=(12, 12), suptitle=None, ypad=.93):
    fig, ax = plt.subplots(nr, nc, figsize=figsize,
                           sharex=True, sharey=True, squeeze=False)
    for a in ax.ravel():
        a.grid(True, ls='--', lw=.4, color='#e5e5e5')
        a.set_facecolor("#fafafa")
    if suptitle:
        fig.suptitle(suptitle, y=ypad, fontsize=14)
    return fig, ax

def _bin_edges_and_count(bin_width, T=1000):
    edges = np.arange(0, T + 1, bin_width)   # inclusive 1000
    return edges, len(edges) - 1

def _smooth(arr, win):
    """simple boxcar smoothing"""
    if win is None or win < 2:
        return arr
    
    kernel = np.ones(int(win)) / win
    return convolve(arr, kernel, mode='same')

def overall_step_walk_plot(m_list, r_list, n_trials=5000, T=1000):
    n_to_plot = min(n_trials, 3)

    fig, ax = plt.subplots(figsize=(10, 6))

    color_cycle = sns.color_palette("husl", len(m_list) * len(r_list))

    for idx, (m, r) in enumerate(itertools.product(m_list, r_list)):
        step = models.StepModel(m=m, r=r)
        _, _, rates = step.simulate(Ntrials=n_trials, T=T)

        for k in range(n_to_plot):
            ax.plot(rates[k], color=color_cycle[idx], alpha=0.75)

        ax.plot(rates.mean(0), color=color_cycle[idx], lw=2,
                label=rf"$m$={m}, $\it{{r}}$={r}")

    ax.set_ylim(0, 55)
    ax.set_xlim(0, T)
    ax.set_title("Step Walk Plot Across Parameter Space", fontsize=24)
    ax.set_xlabel("Time", fontsize=18)
    ax.set_ylabel("Firing Rate", fontsize=18)
    ax.legend(frameon=False, fontsize=18, loc='upper left',
              bbox_to_anchor=(1.05, 0.75))
    plt.tight_layout()
    plt.show()
# convolution with 1d laplacian kernel
def laplacian(arr, plot=True):
    laplacian1d = np.array([-1, 16, -30, 16, -1]) / 12
    response = convolve(arr, laplacian1d, mode = 'same')


    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(response, label='Laplacian Response', linestyle='--')
        plt.legend()
        plt.title("Pulse Detection Using 1D Laplacian Kernel")
        plt.grid(True)
        plt.show()

    return response

# ️ PSTH grids for Ramp & Step
def psth_grid(model_cls, p1_list, p2_list,
              p1_name, p2_name,
              n_trials=100, T=1000,
              bin_width=50, smooth_ms=None,
              ymax=55):
    """
    Creates a grid of PSTH panels.
      model_cls : models.RampModel  or models.StepModel
      p1_list   : list of first parameter (β or m)
      p2_list   : list of second parameter (σ or r)
    """
    edges, n_bins = _bin_edges_and_count(bin_width, T)
    fig, ax = _pretty_grid(len(p1_list), len(p2_list),
                           suptitle=f"PSTH grid – {model_cls.__name__}")

    for i, p1 in enumerate(p1_list):
        for j, p2 in enumerate(p2_list):

            model = model_cls(p1, p2) if model_cls is models.StepModel \
                    else model_cls(beta=p1, sigma=p2)

            spikes, *_ = model.simulate(Ntrials=n_trials, T=T)
            # flatten spike times across trials
            spike_times = np.where(spikes)[1]  # column indices are times
            psth, _ = np.histogram(spike_times, bins=edges)
            psth = psth / n_trials / (bin_width / 1000)   # to Hz
            psth = _smooth(psth, smooth_ms // bin_width if smooth_ms else None)

            ax[i, j].bar(edges[:-1], psth, width=bin_width, align='edge',
                         color="#4a90e2")
            ax[i, j].set_ylim(0, ymax)
            ax[i, j].set_xlim(0, T)
            ax[i, j].set_title(fr"{p1_name}={p1} | {p2_name}={p2}", fontsize=9)

    ax[-1, 0].set_xlabel("time (ms)")
    for r in ax:
        r[0].set_ylabel("Hz")
    fig.tight_layout(rect=[0, 0, 1, .92])
    return fig


def overall_psth_plot(model_cls, p1_list, p2_list,
                      p1_name, p2_name,
                      n_trials=100, T=1000,
                      bin_width=50, smooth_ms=None,
                      ymax=55):
    """
    Overlayed PSTH for all (p1, p2) parameter combinations in one plot.
    model_cls : models.RampModel or models.StepModel
    p1_list   : list of first parameter (β or m)
    p2_list   : list of second parameter (σ or r)
    """
    edges, n_bins = _bin_edges_and_count(bin_width, T)
    fig, ax = plt.subplots(figsize=(10, 6))
    color_cycle = sns.color_palette("husl", len(p1_list) * len(p2_list))

    for idx, (p1, p2) in enumerate(itertools.product(p1_list, p2_list)):
        model = model_cls(p1, p2) if model_cls is models.StepModel \
                else model_cls(beta=p1, sigma=p2)

        spikes, *_ = model.simulate(Ntrials=n_trials, T=T)
        spike_times = np.where(spikes)[1]  # column indices are spike times
        psth, _ = np.histogram(spike_times, bins=edges)
        psth = psth / n_trials / (bin_width / 1000)  # convert to Hz
        psth = _smooth(psth, smooth_ms // bin_width if smooth_ms else None)

        ax.plot(edges[:-1], psth, lw=2, color=color_cycle[idx],
                label=fr"{p1_name}={p1}, {p2_name}={p2}")

    ax.set_ylim(0, ymax)
    ax.set_xlim(0, T)
    ax.set_xlabel("Time (ms)", fontsize=18)
    ax.set_ylabel("Hz", fontsize=18)
    ax.set_title(f"PSTH – {model_cls.__name__}", fontsize=24)
    ax.legend(frameon=False, fontsize=18, loc='upper left',
              bbox_to_anchor=(1.05, 0.75))
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)
    plt.grid(True, ls='--', lw=0.5, color='#e0e0e0')
    return fig




# Fano-factor time–series for a single setting
def fanoFactor(model, n_trials=5000, T=1000,
                bin_width=50, smooth_ms=None,
                ax=None, label=None, plot=False):

    if isinstance(model, models.RampModel):
        label = '\u03B2: ' + str(round(model.beta, 2)) + ', \u03C3: ' + str(round(model.sigma, 2))
        title = 'Ramp Model Fano Factor Plot'

    if isinstance(model, models.StepModel):
        label = 'm: '+str(round(model.m))+', r: '+str(round(model.r))
        title = 'Step Model Fano Factor Plot'
    
    edges, n_bins = _bin_edges_and_count(bin_width, T)
    spikes, *_    = model.simulate(Ntrials=n_trials, T=T)

    # Count spikes in each bin for every trial  ->  (n_trials, n_bins)
    binned = np.add.reduceat(spikes, edges[:-1], axis=1)
    mean   = binned.mean(axis=0)
    var    = binned.var(axis=0)
    fano   = var / mean
    fano   = _smooth(fano, smooth_ms // bin_width if smooth_ms else None)

    times = edges[:-1] + bin_width/2
    if plot:
        if ax is None: ax = plt.gca()
        ax.plot(times, fano, label=label)
        ax.set_xlabel("Time (ms)", fontsize=18); ax.set_ylabel("Fano factor", fontsize=18)
        ax.set_ylim(bottom=0, top=1.8)
        ax.grid(True, ls='--', lw=.4, color='#e5e5e5')
        ax.set_title(title, fontsize=24)
        ax.legend(ncol=2, fontsize=18)

    return times, fano

# ️ PSTH fluctuation vs #trials (quantitative Task 1.2)
def psth_variability(model_cls, params, trial_grid=(10,50,100,200,500,1000),
                     repeats=10, bin_width=50, T=1000):
    """
    Calculates the standard deviation of PSTH estimates across 'repeats'
    datasets for each N in trial_grid.
      params = dict(beta=…, sigma=…)  or  dict(m=…, r=…)
    Returns two arrays: Ns, std_err
    """
    edges, n_bins = _bin_edges_and_count(bin_width, T)
    stds = []
    for N in trial_grid:
        psths = []
        for _ in range(repeats):
            model = model_cls(**params)
            spikes, *_ = model.simulate(Ntrials=N, T=T)
            spike_times = np.where(spikes)[1]
            psth, _ = np.histogram(spike_times, bins=edges)
            psth = psth / N / (bin_width / 1000)
            psths.append(psth)
        psths = np.stack(psths)
        stds.append(psths.std(axis=0).mean())   # average over time bins

    return np.array(trial_grid), np.array(stds)

# Example plot:
# Ns, errs = psth_variability(models.RampModel, dict(beta=1, sigma=.15))
# plt.figure(); plt.loglog(Ns, errs, marker='o'); plt.xlabel("N trials");
# plt.ylabel("avg PSTH std dev"); plt.grid(True, which='both', ls='--')

#  Overlay comparison – find matched PSTHs
def overlay_psth_comparison(ramp_params, step_params,
                            n_trials=500, T=1000,
                            bin_width=20, smooth_ms=20):
    """
    Overlays Ramp vs Step PSTHs in one figure to visually judge similarity.
    """
    edges, _ = _bin_edges_and_count(bin_width, T)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for model_cls, pars, color, name in [
        (models.RampModel, ramp_params, "#4a90e2", "Ramp"),
        (models.StepModel, step_params, "#e26a4a", "Step")
    ]:
        model = model_cls(**pars)
        spikes, *_ = model.simulate(Ntrials=n_trials, T=T)
        spike_times = np.where(spikes)[1]
        psth, _ = np.histogram(spike_times, bins=edges)
        psth = psth / n_trials / (bin_width/1000)
        psth = _smooth(psth, smooth_ms // bin_width if smooth_ms else None)

        ax.plot(edges[:-1] + bin_width/2, psth, lw=2, label=name, color=color)

    ax.set_xlabel("time (ms)"); ax.set_ylabel("Hz")
    ax.set_xlim(0, T); ax.set_ylim(0)
    ax.legend(); ax.grid(True, ls='--', lw=.4)
    ax.set_title("Ramp vs Step PSTH overlay")
    fig.tight_layout()
    return fig


def analyse_psth(spike_trains, time, title):
    """Return smoothed psth, derivatives and the 2nd-derivative range."""
    t_lo = 0.1
    t_hi = 0.9
    SIGMA = 50
    dt = time[1] - time[0]
    psth = spike_trains.mean(axis=0)
    psth_smooth = gaussian_filter1d(psth, sigma=SIGMA)
    d1 = np.gradient(psth_smooth, dt)
    d2 = np.gradient(d1, dt)

    # range of 2nd derivative in the chosen window
    win = (time >= t_lo) & (time <= t_hi)
    d2_range = np.ptp(d2[win])
    print(f"{title:<35s}  Range = {d2_range:7.3f}")

    # plotting
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].plot(time, psth, label='raw')
    axs[0].plot(time, psth_smooth, '--', label='smoothed')
    axs[0].set_title(title); axs[0].legend()
    axs[0].set(xlabel='time (s)', ylabel='mean spike count')

    axs[1].plot(time, d1); axs[1].set_title('1st derivative')
    axs[1].set(xlabel='time (s)', ylabel='rate of change')

    axs[2].plot(time, d2); axs[2].set_title('2nd derivative')
    axs[2].set(xlabel='time (s)', ylabel='curvature')

    fig.tight_layout()
    plt.show()

    return d2_range
