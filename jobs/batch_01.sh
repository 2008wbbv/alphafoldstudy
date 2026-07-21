#!/bin/bash
#SBATCH --job-name=afstudy_batch_01
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/afstudy_batch_01_%j.log

# ==========================================================================
# EDIT THESE CLUSTER-SPECIFIC SETTINGS BEFORE SUBMITTING:
#   1. Account and partition (uncomment and fill in):
#        #SBATCH --account=YOUR_ACCOUNT
#        #SBATCH --partition=YOUR_PARTITION
#   2. Module / environment setup (uncomment and edit for your cluster):
#        module load python/3.11
#        source /path/to/your/venv/bin/activate
#   3. Confirm the working directory and paths below are correct.
# The #SBATCH headers above (job name, time, memory, cpus, output log) are
# safe defaults; tune them to your job and cluster limits.
# ==========================================================================

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p results/batch_01 logs

# module load python/3.11                 # <- edit for your cluster
# source /path/to/your/venv/bin/activate  # <- edit for your cluster

python src/af_study.py --registry jobs/batch_01.csv --results-dir results/batch_01
