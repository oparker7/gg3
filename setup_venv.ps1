# Exit if any command fails
$ErrorActionPreference = "Stop"

# Create virtual environment
python -m venv .gg3_venv
.gg3_venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt


# Add Jupyter kernel
python -m ipykernel install --user --name=.gg3_venv --display-name "Python (.gg3_venv)"


# configure git hooks
nbstripout --install
# (stops metadata from being committed to git)
