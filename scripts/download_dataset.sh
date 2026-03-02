#!/bin/bash
#SBATCH --job-name=filesender_download
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --partition=audible,A100,L40S,A40
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00

set -euo pipefail

mkdir -p slurm_logs
OUTDIR="./dataset"
mkdir -p "$OUTDIR"

echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"
echo "Saving into: $OUTDIR"

TOKEN="3b689713-ecc0-4171-90ae-c9ca75d58962"

# List of file ids
IDS=(
  67270957 67270958 67270959 67270960 67270961 67270962 67270963 67270964 67270965 67270966
  67270967 67270968 67270969 67270970 67270971 67270972 67270973 67270974 67270975 67270976
  67270977 67270978 67270979 67270980 67270981 67270982 67270983 67270984 67270985 67270986
  67270987 67270988 67270989 67270990 67270991 67270993 67270994 67270997 67270999 67271001
  67271003 67271005 67271007 67271009 67271011 67271013 67271014 67271015 67271016 67271017
  67271018 67271019 67271020 67271021 67271022 67271023 67271024 67271025 67271026 67271027
  67271028 67271029 67271030 67271031 67271032 67271033 67271034 67271035 67271036 67271037
  67271038 67271039 67271040 67271041 67271042 67271043 67271044 67271045 67271046
)

# Download using server-provided names (Content-Disposition).
# NOTE: This cannot be combined with resume (-C -). If you need resume, we can implement a safer approach,
# but it requires determining the filename first and then using -o.
for id in "${IDS[@]}"; do
  url="https://filesender.renater.fr/download.php?token=${TOKEN}&files_ids=${id}"
  echo "Downloading files_ids=$id"
  curl -L --fail \
    --retry 10 --retry-all-errors --retry-delay 2 \
    -J -O --output-dir "$OUTDIR" \
    "$url"
done

echo "Job finished at: $(date)"