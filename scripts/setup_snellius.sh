#!/bin/bash
# Setup script for Snellius supercomputer.
# Run once on the login node before submitting jobs.
#
# Usage:
#   bash scripts/setup_snellius.sh

set -euo pipefail

module load 2023
module load Python/3.12.3-GCCcore-13.2.0

echo "Creating virtual environment..."
python -m venv .venv
source .venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Creating output directory..."
mkdir -p output

echo "Setup complete. Submit jobs with:"
echo "  sbatch scripts/snellius_job.sh"
