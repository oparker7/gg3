# GG3 Neural Data Analysis
GG3 -- Neural Data Analysis (Undergraduate IIA project, Department of Engineering, University of Cambridge)

## Overview

Activity in the lateral intra-parietal cortex (LIP) is involved in the control of saccadic eye movements, with the ‘activity’ relating to the firing of synapses in this area of the brain. It is possible to model the neural activity from two schools of thought. Firstly, the ‘ramping’ approach models the firing rate as following a drift-diffusion process akin to Brownian motion. More recently, evidence  has suggested that LIP neurons are better modelled with a ‘step’ firing rate, in that the firing rate jumps from some baseline level to a higher active level.

This is pertinent to study as it helps to place the neurons in the hierarchy of decision making, with the ramping activity resembling a likelihood approach with a decision being made once a threshold likelihood is met. Alternatively, the step model conveys that this area of the brain is simply having the information communicated to it and ‘switches on’ once the decision has been relayed.

Data studied in this experiment is largely in the form of ‘spike trains’, binary arrays indicating neural activity at specific time instances; these spike trains can be interpreted through the lens of firing rate in order to gain insight into the decision-making process that lays beneath.

The step model has one latent variable τ, the jump time, which is governed by two parameters m and r. The probability distribution of τ follows a negative binomial distribution; largely, m governs the mean jump time of an ensemble and r governs the certainty with which this happens.

The ramp model has a series of latent variables x_t which follows the drift-diffusion model as discussed, with β being the drift factor measuring the definite movement with each time step and σ scaling Gaussian noise that perturbs the movement with each time step.


## Venv and jupyter kernel setup

To create and enter venv: Open a PowerShell terminal in VS code and run `.\setup_venv.ps1`

To enter when the venv has already been created, run: `.\gg3_venv\Scripts\activate`

(Or in bash, run 'source setup.sh')

Select this Kernel in any jupyter notebooks to ensure dependencies.

## Credit

Source code for this repo was provided by Yashar Ahmadian and can be found at https://github.com/ahmadianlab/gg3_nda/tree/main
