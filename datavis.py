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