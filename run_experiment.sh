#!/bin/bash

# Run GInRec simplified experiment
# Usage: bash run_experiment.sh [data_dir] [epochs] [batch_size]

DATA_DIR=${1:-"amazon-book"}
EPOCHS=${2:-100}
BATCH_SIZE=${3:-8}
LR=${4:-0.001}

echo "Running GInRec experiment..."
echo "Data directory: $DATA_DIR"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LR"

python train_simple.py \
    --data_dir "$DATA_DIR" \
    --n_epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR"

echo "Experiment completed!"
