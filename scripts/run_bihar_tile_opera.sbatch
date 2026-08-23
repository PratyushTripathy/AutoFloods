#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00

echo "===== BEGIN $(date -u '+%Y-%m-%d %H:%M:%S') UTC (host=$(hostname) job=${SLURM_JOB_ID} tile=${TILE_ID}) ====="
/home/ptripathy/miniforge-pypy3/envs/autofloods/bin/python /home/emlab/projects/current-projects/edge-autofloods/AutoFloods/scripts/bihar2024_tile_opera.py "${TILE_ID}"
rc=$?
echo "===== END   $(date -u '+%Y-%m-%d %H:%M:%S') UTC (exit=${rc}) ====="
exit $rc
