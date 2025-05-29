# GG3 Neural Data Analysis
GG3 -- Neural Data Analysis (Undergraduate IIA project, Department of Engineering, University of Cambridge)

## ToDo
 
 Week 2

 2.1. 
 - (Done) Work out distribution for ramp model
 - (Done) Create class for HMM ramp model
 - (Done) create trajectories of x_t
 - Based on x_t, calculate firing rate trajectory r_t
 - Compare rate trajectories with simulated trajectories from continuous model 
 - Work out why some trajectories get stuck at the initial state
 - Find threshold value of sigma so the trajectory doesn't get stuck

2.2.
- Time homogeneous markov chain for the step with two states
- Simulate this markov chain and plot x_t
- Histogram the jump times of this model
- Compare to the histograms in week 1
- Work out why this model is wrong
- (Done) Make new model with r+1 states
- (Done) simulate several trials of new markov chain
- Plot histograms of jump times (should add this as a class method)
- Compare histograms to week 1

2.3.


## Overview



## Venv and jupyter kernel setup

To create and enter venv: Open a PowerShell terminal in VS code and run `.\setup_venv.ps1`

To enter when the venv has already been created, run: `.\gg3_venv\Scripts\activate`

(Or in bash, run 'source setup.sh')

Select this Kernel in any jupyter notebooks to ensure dependencies.

## Credit

Source code for this repo was provided by Yashar Ahmadian and can be found at https://github.com/ahmadianlab/gg3_nda/tree/main
