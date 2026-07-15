#!/bin/bash
# Campaign 1 / Stage 1a — FedDyn alpha screen (24 runs)
#
# MUST be run through SLURM: bare GPU processes on this node are SIGTERM'd after
# ~40-80s by SLURM's GPU policing daemon. See HPO_AGENT/results/00_preflight_blockers.md (B6).
#
# QOS `train`: 3 concurrent GPU jobs (GrpTRES gres/gpu=3, MaxJobsPU=3) and a HARD
# 6-hour MaxWall. A --time above 6h is rejected with QOSMaxWallDurationPerJobLimit
# and never starts. These queue behind the pending `fedgfe` jobs.
#
# 8 array tasks x `wandb agent --count 3` = 24 runs, 3 tasks at a time.
# Bounding each agent to 3 runs (~37 min each => ~1.9h) keeps every task well inside
# the 6h wall, so no run is ever guillotined mid-training.
#
#SBATCH --job-name=hpo-feddyn-1a
#SBATCH --partition=gpu
#SBATCH --qos=train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=05:45:00
#SBATCH --array=1-8%3
#SBATCH --output=/home/lpala/fedsky/ext/FedCMAPSS/HPO_AGENT/logs/stage1a_%A_%a.out
#SBATCH --error=/home/lpala/fedsky/ext/FedCMAPSS/HPO_AGENT/logs/stage1a_%A_%a.err

# NOTE: no `set -u` — conda's activation scripts reference unbound vars and abort under it.
set -eo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flvit
cd /home/lpala/fedsky/ext/FedCMAPSS

export OMP_NUM_THREADS=4
export WANDB__SERVICE_WAIT=300

echo "host=$(hostname) array_task=${SLURM_ARRAY_TASK_ID} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name --format=csv,noheader

# Bounded to 3 runs per agent so the task always finishes inside the 6h QOS wall.
wandb agent --count 3 ngslung/HPO_AGENT_FedCMAPSS/y21gj2xn
