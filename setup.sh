#!/bin/bash

# Exit if any command fails
set -e

# Create virtual environment
python -m venv .gg3_venv

# Activate the virtual environment
source .gg3_venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Add Jupyter kernel
python -m ipykernel install --user --name=.gg3_venv --display-name "Python (.gg3_venv)"
