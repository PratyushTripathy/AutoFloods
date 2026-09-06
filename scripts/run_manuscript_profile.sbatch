#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

echo "===== BEGIN $(date -u '+%Y-%m-%d %H:%M:%S') UTC (host=$(hostname) job=${SLURM_JOB_ID} config=${CONFIG}) ====="
/home/ptripathy/miniforge-pypy3/envs/autofloods/bin/python -u \
  /home/emlab/projects/current-projects/edge-autofloods/AutoFloods/scripts/verification/manuscript_stage_profile.py \
  --config "${CONFIG}" --output_json "${OUTPUT_JSON}"
rc=$?
echo "===== END   $(date -u '+%Y-%m-%d %H:%M:%S') UTC (exit=${rc}) ====="
exit $rc
