#!/bin/bash
#SBATCH --job-name=af_meanstd_bihar318
#SBATCH --partition=emlab_nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/emlab/projects/current-projects/edge-autofloods/AutoFloods/scripts/logs/260817_meanstd_bihar318_%j.out
#SBATCH --error=/home/emlab/projects/current-projects/edge-autofloods/AutoFloods/scripts/logs/260817_meanstd_bihar318_%j.err

cd /home/emlab/projects/current-projects/edge-autofloods/AutoFloods/scripts
mamba run -n autofloods python 260817_MeanStdFloodMap_Bihar318.py
