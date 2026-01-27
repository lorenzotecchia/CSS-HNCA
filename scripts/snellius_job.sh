#!/bin/bash
#SBATCH --job-name=css-hnca-sweep
#SBATCH --array=0-53
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:10:00
#SBATCH --partition=rome
#SBATCH --output=output/slurm-sweep-%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lorenzo.tecchia@icloud.com

module load 2023
module load Python/3.12.3-GCCcore-13.2.0

cd $SLURM_SUBMIT_DIR
source .venv/bin/activate

python scripts/snellius_sweep.py --config-index $SLURM_ARRAY_TASK_ID
