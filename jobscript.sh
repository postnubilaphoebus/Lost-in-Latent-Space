#!/bin/bash

#SBATCH --job-name=cnn_lip
#SBATCH --time=03:59:00
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=60G

module load Python/3.9.6-GCCcore-11.2.0
source /scratch/s4184416/peregrine/env/bin/activate
python3 main.py
