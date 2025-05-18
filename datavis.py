import numpy as np
import matplotlib.pyplot as plt

def raster_plot(spikes, title='Raster Plot'):
    
    # convert the spike train into an array of spike times

    spike_times = [np.where(spikes !=0)[0] for i in range(spikes.shape[0])]

    # Create the plot, 'spikes' should be a list of arrays where each array contains event times for a different trial
    plt.eventplot(spike_times, linelengths=0.1, linestyles='solid')
    
    # Add labels and title
    plt.xlabel('Time Step Index')
    plt.ylabel('Trial Index')
    plt.title(title)
    
    plt.tight_layout()
    plt.show()

def jump_hist(jump_times):

    plt.hist(jump_times, bins=30)
    plt.xlabel('Jump Time')
    plt.ylabel('Frequency')
    plt.show()

def walk_plot(trajectories):
    N = len(trajectories)
    T = len(trajectories[0])
    for i in range(N):
        plt.plot(range(T), trajectories[i], label=f'Line {i+1}')

    plt.xlabel('Time')
    plt.ylabel('x')
    plt.title('Ramp Model Random Walk')
    plt.legend()
    plt.show()

# ---------- helpers ---------- #
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
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("trial")
    return fig, axes
# ----------------------------- #

# Legend handles
_spike_bound_handles = [
    plt.Line2D([], [], marker="|", linestyle="", color="lightblue", label="spike"),
    plt.Line2D([], [], marker="|", linestyle="", color="red", label="bound reached")
]

# ------------------------------------------------------------------ #
#                          RAMP  PLOTS                               #
# ------------------------------------------------------------------ #
def ramp_raster_plot(beta_list=[0.3, 1, 3], sigma_list=[0.05, .15, .3],
                     n_trials=500, T=1000, N_show=10):
    fig, ax = _figure_grid(len(beta_list), len(sigma_list))
    for i, b in enumerate(beta_list):
        for j, s in enumerate(sigma_list):
            ramp          = models.RampModel(beta=b, sigma=s)
            spikes, xs, _ = ramp.simulate(Ntrials=n_trials, T=T)

            events = [np.where(spikes[k] > 0)[0] for k in range(N_show)]
            taus   = _find_bound_times_ramp(xs)
            bound_events = [[t] if (k < N_show and taus[k] >= 0) else []
                            for k, t in enumerate(taus)]

            ax[i, j].eventplot(events, colors="lightblue", lineoffsets=np.arange(N_show),
                               linelengths=.6)
            ax[i, j].eventplot(bound_events[:N_show], colors="red",
                               lineoffsets=np.arange(N_show), linelengths=.8)

            ax[i, j].set_title(rf"$\beta$={b},  $\sigma$={s}", fontsize=9)

    ax[0, 0].legend(handles=_spike_bound_handles, loc="upper right", frameon=False)
    plt.tight_layout()
    return fig


def ramp_walk_plot(beta_list=[0.3, 1, 3], sigma_list=[0.05, .15, .3],
                   n_trials=500, T=1000):
    fig, ax = _figure_grid(len(beta_list), len(sigma_list))
    for i, b in enumerate(beta_list):
        for j, s in enumerate(sigma_list):
            ramp          = models.RampModel(beta=b, sigma=s)
            _, xs, _      = ramp.simulate(Ntrials=n_trials, T=T)

            for k in range(5):
                ax[i, j].plot(xs[k], alpha=.6)
            ax[i, j].plot(xs.mean(0), color="k", lw=2, label="mean $x_t$")
            ax[i, j].set_ylim(0, 1.1)
            ax[i, j].set_title(rf"$\beta$={b},  $\sigma$={s}", fontsize=9)

    ax[0, 0].legend(frameon=False)
    plt.tight_layout()
    return fig

# ------------------------------------------------------------------ #
#                          STEP  PLOTS                               #
# ------------------------------------------------------------------ #
def step_raster_plot(m_list=[200, 500, 800], r_list=[3, 30, 300],
                     n_trials=500, T=1000, N_show=10):
    fig, ax = _figure_grid(len(m_list), len(r_list))
    for i, m in enumerate(m_list):
        for j, r in enumerate(r_list):
            step                = models.StepModel(m=m, r=r)
            spikes, jumps, _    = step.simulate(Ntrials=n_trials, T=T)

            events       = [np.where(spikes[k] > 0)[0] for k in range(N_show)]
            jump_events  = [[jumps[k]] if k < N_show else [] for k in range(n_trials)]

            ax[i, j].eventplot(events, colors="lightblue", lineoffsets=np.arange(N_show), linelengths=.6)
            ax[i, j].eventplot(jump_events[:N_show], colors="red", lineoffsets=np.arange(N_show), linelengths=.8)

            ax[i, j].set_title(rf"$m$={m},  $r$={r}", fontsize=9)

    ax[0, 0].legend(handles=_spike_bound_handles, loc="upper right", frameon=False)
    plt.tight_layout()
    return fig


def step_walk_plot(m_list=[200, 500, 800], r_list=[3, 30, 300],
                   n_trials=500, T=1000):
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
    return fig
