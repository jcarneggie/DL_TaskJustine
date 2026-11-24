#!/usr/bin/env bash
# Menjalankan seluruh pipeline secara berurutan (seed 42).
set -e
cd "$(dirname "$0")"
for m in ae sae vae gan dcgan ddpm; do
  python src/train.py --model $m
done
python src/mlp_experiment.py
python src/evaluate.py
python src/figures.py
