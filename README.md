# GG3 Neural Data Analysis
GG3 -- Neural Data Analysis (Undergraduate IIA project, Department of Engineering, University of Cambridge)

## ToDo
 
 Week 2

 2.1. 
 - (Done) Work out distribution for ramp model
 - (Done) Create class for HMM ramp model
 - (Done) create trajectories of x_t
 - Based on x_t, calculate firing rate trajectory r_t (add this as a class method)
 - Compare rate trajectories with simulated trajectories from continuous model 
 - Work out why some trajectories get stuck at the initial state: when beta is zero, we need the noise movement i.e. sigma * sqrt dt to escape the bin that defines being at zero, i.e. the jump due to noise has to be bigger than 1/K
 - Find threshold value of sigma so the trajectory doesn't get stuck

2.2.
- (Done) Time homogeneous markov chain for the step with two states, acheived by using `exact=False`
- Simulate this markov chain and plot x_t
- Histogram the jump times of this model
- Compare to the histograms in week 1
- Work out why this model is wrong
- (Done) Make new model with r+1 states, use `exact=True`
- (Done) simulate several trials of new markov chain
- Plot histograms of jump times (should add this as a class method)
- Compare histograms to week 1

2.3.
- add class method to each HMM class to generate spike train, this could be done by creating a new class `HMM` with argument `step` or `ramp` and then inheriting the appropriate class from `HMM_models.py`
- Use `inference.py` to infer the posterior expectation of x_t from a number of trials
- Look at `GG3_project.ipynb` for which plots to generate
- Find where the parameter range where the inference is best
- Compare smoothing vs filterng for `hmm_expected_states`



## Overview



## Venv and jupyter kernel setup

To create and enter venv: Open a PowerShell terminal in VS code and run `.\setup_venv.ps1`

To enter when the venv has already been created, run: `.\gg3_venv\Scripts\activate`

(Or in bash, run 'source setup.sh')

Select this Kernel in any jupyter notebooks to ensure dependencies.

## Credit

Source code for this repo was provided by Yashar Ahmadian and can be found at https://github.com/ahmadianlab/gg3_nda/tree/main
