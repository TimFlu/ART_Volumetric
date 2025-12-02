#!/bin/bash
#SBATCH --job-name=train
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=16
#SBATCH --partition=epyc2
#SBATCH --output=logs/slurm/logs_trainy_%j.out
#SBATCH --error=logs/slurm/logs_trainy_%j.err

module load Anaconda3

# Initialize conda for batch jobs
eval "$(conda shell.bash hook)"

conda activate ART_Volumetric


python scripts/training.py
