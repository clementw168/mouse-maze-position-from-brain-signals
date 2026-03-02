#!/bin/bash
#SBATCH --job-name=runner
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --partition=audible,A100,L40S,A40
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00

set -euo pipefail

mkdir -p slurm_logs
echo "MOUSE_ID=$MOUSE_ID"


echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

uv run train.py --mouse_id $MOUSE_ID --stride 4 --window_size 108 --split_type temporal --loss_file "${MOUSE_ID}_temporal.png" --weights_file "${MOUSE_ID}_temporal.pth"
uv run train.py --mouse_id $MOUSE_ID --stride 4 --window_size 108 --split_type mid --loss_file "${MOUSE_ID}_mid.png" --weights_file "${MOUSE_ID}_mid.pth"
uv run train.py --mouse_id $MOUSE_ID --stride 4 --window_size 108 --split_type shuffled --loss_file "${MOUSE_ID}_shuffled.png" --weights_file "${MOUSE_ID}_shuffled.pth"

echo "Job finished at: $(date)"