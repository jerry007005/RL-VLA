#!/bin/bash
# Autonomous pipeline: SAE → goal expert (incl. sg_encoder_cache) → executor (incl. executor_feat_cache) → eval.
# Stages run sequentially. On stage failure, the pipeline stops.

set -e
cd /home/jerry007005/haochuan/githubRepo/RL-VLA

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a pipeline.log
}

# -----------------------------------------------------------------------------
# Stage 1: wait for SAE training (PID 3748192) to finish
# -----------------------------------------------------------------------------
SAE_PID=3748192
log "Pipeline launched. Waiting for SAE training (PID $SAE_PID) to finish ..."
while kill -0 $SAE_PID 2>/dev/null; do
    sleep 60
done
log "SAE training process exited."

if [ ! -f ./checkpoints/subgoal_encoder/checkpoint.pt ]; then
    log "ERROR: SAE checkpoint missing. Aborting pipeline."
    exit 1
fi
log "SAE checkpoint OK ($(du -h ./checkpoints/subgoal_encoder/checkpoint.pt | cut -f1))."

# -----------------------------------------------------------------------------
# Stage 2: goal expert training (also builds sg_encoder_cache via Phase 0.5)
# -----------------------------------------------------------------------------
log "Stage 2: launching goal expert training (train_subgoal_decoder.py)..."
python train_subgoal_decoder.py > train_goal_expert.log 2>&1
log "Stage 2 done: goal expert training exited."

if [ ! -f ./checkpoints/subgoal_decoder/checkpoint.pt ]; then
    log "ERROR: goal expert checkpoint missing. Aborting pipeline."
    exit 1
fi
log "Goal expert checkpoint OK ($(du -h ./checkpoints/subgoal_decoder/checkpoint.pt | cut -f1))."

# -----------------------------------------------------------------------------
# Stage 3: executor training (also builds executor_feat_cache)
# -----------------------------------------------------------------------------
log "Stage 3: launching executor training (train.py)..."
python train.py > train_executor.log 2>&1
log "Stage 3 done: executor training exited."

if [ ! -f ./checkpoints/executor/checkpoint.pt ]; then
    log "ERROR: executor checkpoint missing. Aborting pipeline."
    exit 1
fi
log "Executor checkpoint OK ($(du -h ./checkpoints/executor/checkpoint.pt | cut -f1))."

# -----------------------------------------------------------------------------
# Stage 4: eval (eval_executor.py with full set of tests)
# -----------------------------------------------------------------------------
log "Stage 4: launching eval (eval_executor.py)..."
python eval_executor.py > eval_executor.log 2>&1
log "Stage 4 done: eval finished."

log "Pipeline complete."
