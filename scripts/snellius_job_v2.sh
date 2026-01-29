#!/bin/bash
#SBATCH --job-name=css-hnca-sweep-v2
#SBATCH --array=0-1079
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=02:00:00
#SBATCH --partition=rome
#SBATCH --output=output/slurm-sweep-v2-%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lorenzo.tecchia@icloud.com

# Load modules (ignore errors if Python module name changed)
module load 2023 2>/dev/null || true
module load Python/3.12.3-GCCcore-13.2.0 2>/dev/null || true

cd $SLURM_SUBMIT_DIR
source .venv/bin/activate

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "Hostname: $(hostname)"
echo "Python: $(which python)"
echo "Start: $(date)"

python scripts/snellius_sweep_v2.py --config-index $SLURM_ARRAY_TASK_ID

echo "End: $(date)"
