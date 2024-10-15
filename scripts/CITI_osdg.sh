#!/bin/bash

. "$HOME/miniconda3/etc/profile.d/conda.sh"
VIRTUAL_ENV_DISABLE_PROMPT=true conda activate DecomposeTransformer

# Default parameters
NAME="OSDG"
DEVICE="cuda:0"
CHECKPOINT=None
BATCH_SIZE=32
NUM_WORKERS=16
NUM_SAMPLES=64
CI_SPARSITY_RATIO=0.6
TI_RECOVERY_RATIO=0.1
INCLUDE_LAYERS="attention intermediate output"
EXCLUDE_LAYERS=None
LOG_DIR=""

# Parsing command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --name) NAME="$2"; shift ;;
        --device) DEVICE="$2"; shift ;;
        --checkpoint) CHECKPOINT="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --num_workers) NUM_WORKERS="$2"; shift ;;
        --num_samples) NUM_SAMPLES="$2"; shift ;;
        --ci_sparsity_ratio) CI_SPARSITY_RATIO="$2"; shift ;;
        --ti_recovery_ratio) TI_RECOVERY_RATIO="$2"; shift ;;
        --include_layers) INCLUDE_LAYERS="$2"; shift ;;
        --exclude_layers) EXCLUDE_LAYERS="$2"; shift ;;
        --log_dir) LOG_DIR="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

TORCHTEST_OUTPUT=$(python3 ../utils/torchtest.py $DEVICE)
if [[ $? -ne 0 ]]; then
    echo "torchtest.py encountered an error."
    echo "$TORCHTEST_OUTPUT"
    exit 1
fi
echo "torchtest.py output:"
echo "$TORCHTEST_OUTPUT"

cd ../experiments
echo "Running Python script"

for CONCERN in {0..9}; do
    python3 ./CITI.py \
        --name "$NAME" \
        --device "$DEVICE" \
        --checkpoint "$CHECKPOINT" \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        --num_samples "$NUM_SAMPLES" \
        --concern $CONCERN \
        --ci_ratio "$CI_SPARSITY_RATIO" \
        --ti_ratio "$TI_RECOVERY_RATIO" \
        --include_layers $INCLUDE_LAYERS \
        --exclude_layers $EXCLUDE_LAYERS \
        --log_dir $LOG_DIR
done
echo "Python script finished"
