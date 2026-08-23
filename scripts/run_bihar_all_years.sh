#!/bin/bash
# Orchestrator: process all of Bihar (22 tiles) via OPERA, one year at a
# time, in this exact order -- 2024 first (already soak-tested), then the
# rest of the requested 2017-2026 range. Waits for a year's 22 tile jobs
# to fully drain (all COMPLETED/FAILED, none PENDING/RUNNING) before
# submitting the next year, so years never overlap and compete for the
# same OPERA/ASF egress. Submitted as its own SLURM job (see
# run_bihar_all_years.sbatch) so it survives independent of any
# interactive session -- it just submits/polls, no heavy compute itself.
set -uo pipefail

BASE=/home/emlab/projects/current-projects/edge-autofloods/AutoFloods
CONFIG_DIR=$BASE/scripts/configs/bihar_opera
LOG_DIR=$BASE/output/bihar_opera_30m/logs
ORCH_LOG=$LOG_DIR/orchestrator.log

mkdir -p "$LOG_DIR"

TILES=(274 275 276 277 313 314 315 316 317 318 319 320 321 322 323 324 325 326 328 329 330 331)
YEARS=(2024 2017 2018 2019 2020 2021 2022 2023 2025 2026)

log() {
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S') UTC] $*" | tee -a "$ORCH_LOG"
}

log "===== Orchestrator starting (host=$(hostname)) ====="

for YEAR in "${YEARS[@]}"; do
    log "--- Year $YEAR: submitting ${#TILES[@]} tile jobs ---"
    JOB_IDS=()
    for TILE in "${TILES[@]}"; do
        CONFIG="$CONFIG_DIR/bihar_${YEAR}_tile${TILE}.yaml"
        JOB_ID=$(sbatch --parsable \
            --job-name="bihar_${YEAR}_t${TILE}" \
            --output="$LOG_DIR/${YEAR}_tile${TILE}_%j.out" \
            --error="$LOG_DIR/${YEAR}_tile${TILE}_%j.err" \
            --exclude=hpc-13.grit.ucsb.edu,hpc-15.grit.ucsb.edu \
            --mem=64G --cpus-per-task=8 --time=03:00:00 \
            --wrap="/home/ptripathy/miniforge-pypy3/envs/autofloods/bin/python $BASE/scripts/run_autofloods.py --config $CONFIG")
        JOB_IDS+=("$JOB_ID")
        log "  tile $TILE -> job $JOB_ID"
    done

    log "Year $YEAR: waiting for ${#JOB_IDS[@]} jobs to drain..."
    while true; do
        RUNNING=0
        for JID in "${JOB_IDS[@]}"; do
            STATE=$(sacct -j "$JID" --format=State --noheader --parsable2 2>/dev/null | head -1 | tr -d ' ')
            if [[ "$STATE" == "PENDING" || "$STATE" == "RUNNING" || "$STATE" == "" ]]; then
                RUNNING=$((RUNNING+1))
            fi
        done
        if [[ "$RUNNING" -eq 0 ]]; then
            break
        fi
        sleep 60
    done

    log "Year $YEAR: drained. Final states:"
    for JID in "${JOB_IDS[@]}"; do
        STATE=$(sacct -j "$JID" --format=State --noheader --parsable2 2>/dev/null | head -1 | tr -d ' ')
        log "  job $JID: $STATE"
    done
done

log "===== Orchestrator finished all years ====="
