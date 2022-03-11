#!/bin/bash
#SBATCH --gres=gpu:4       # Request GPU "generic resources"
#SBATCH --cpus-per-task=8  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=3-12:00     # DD-HH:MM:SS

source ENV/bin/activate
python BART_BPE.py