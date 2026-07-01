#!/usr/bin/env bash
#SBATCH --job-name=eye-cursor-train
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --gres=gpu:1

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs outputs

# Adjust these two lines for your cluster if it uses environment modules.
# module load python/3.11
# module load cuda

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-server.txt

export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

python -m eye_cursor.server_train \
  --calibration "calibration.npz" \
  --output "outputs/model.onnx" \
  --metadata "outputs/model_metadata.json" \
  --metrics "outputs/metrics.json" \
  --epochs 400 \
  --batch-size 256
