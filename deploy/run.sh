#!/bin/bash
# Activates the conda environment and starts the hands-free live classifier.
# Edit CONDA_SH and ENV_NAME to match your machine.

CONDA_SH="$HOME/miniconda/etc/profile.d/conda.sh"
ENV_NAME="cassi"

source "$CONDA_SH"
conda activate "$ENV_NAME"

cd "$(dirname "$0")/.."
python live/auto_capture.py
