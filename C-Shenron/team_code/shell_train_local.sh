#!/bin/bash
# ============================================================================
# C-Shenron Training Script — Single RTX 3060 (12GB VRAM)
# ============================================================================
# Usage:  bash shell_train_local.sh
# Run from:  C-Shenron/team_code/
# ============================================================================

# ---- Environment Setup ----
export OMP_NUM_THREADS=1          # Strictly limit thread pools to prevent Python SIGSEGV
export OPENBLAS_NUM_THREADS=1     # Prevent numpy thread explosion
export CUDA_VISIBLE_DEVICES=0     # Single GPU

# ---- Experiment Config ----
export TRAIN_ID=cshenron_town01_radar_v1

# ---- Paths (EDIT THESE FOR YOUR UBUNTU MACHINE) ----
ROOT_DIR="/storage/dataset"                      # Where data_collector_v2.py saved data
LOGDIR="/storage/training_logs"                  # Where model checkpoints + tensorboard logs go

# Create log directories
mkdir -p ${LOGDIR}/training_logs
touch ${LOGDIR}/training_logs/${TRAIN_ID}.log

echo "=============================================="
echo "  C-Shenron Training — RTX 3060 Edition"
echo "  Dataset:  ${ROOT_DIR}"
echo "  Logs:     ${LOGDIR}"
echo "  Train ID: ${TRAIN_ID}"
echo "=============================================="

# ---- Launch Training ----
# torchrun with nproc_per_node=1 for single GPU
# batch_size=2 is safe for 12GB VRAM with TransFuser-CR backbone
# radar_cat=1 = Front+Back radar concatenation (what we generated)
# radar_channels=1 = Use Shenron simulated radar (not CARLA native)
# setting=all = Train on all data, no validation split (we only have Town01)

torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --max_restarts=0 \
    --rdzv_id=12345 \
    --rdzv_backend=c10d \
    train.py \
    --id ${TRAIN_ID} \
    --epochs 30 \
    --batch_size 2 \
    --setting all \
    --root_dir ${ROOT_DIR} \
    --logdir ${LOGDIR} \
    --use_controller_input_prediction 1 \
    --use_wp_gru 0 \
    --use_discrete_command 1 \
    --use_tp 1 \
    --continue_epoch 1 \
    --load_file /storage/training_logs/cshenron_town01_radar_v1/model_0005.pth \
    --cpu_cores 8 \
    --num_repetitions 1 \
    --use_disk_cache 0 \
    --radar_channels 1 \
    --radar_cat 1 \
    2>&1 | tee ${LOGDIR}/training_logs/${TRAIN_ID}.log

echo ""
echo "=============================================="
echo "  Training Complete!"
echo "  Model saved to: ${LOGDIR}/${TRAIN_ID}/"
echo "=============================================="
