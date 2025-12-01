#!/bin/bash
#
# Cross-Initialization Lottery Subspace Transfer Experiment
#
# Tests whether lottery subspaces are architecture-intrinsic or init-specific.
#
# Usage:
#   ./run_transfer_experiment.sh
#   ./run_transfer_experiment.sh --quick   # Fewer dimensions for fast test
#

set -e  # Exit on error

# Fix for cuDNN algorithm picker issues
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"

# Configuration
MODEL="TinyCNN"
DATASET="MNIST"
EPOCHS=3
DONOR_POINTS=1          # Just need 1 donor trajectory
RECIPIENT_POINTS=5      # Test on 5 different recipient seeds
DATA_DIR="../cross-init-lottery-subspace-data"

# Dimensions to explore
if [[ "$1" == "--quick" ]]; then
    echo "Running QUICK mode (fewer dimensions for testing)"
    DIMS="8 32 128 512"
else
    DIMS="8 32 128 512 1024 2048"
fi

echo "=============================================================="
echo "CROSS-INITIALIZATION LOTTERY SUBSPACE TRANSFER EXPERIMENT"
echo "=============================================================="
echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Dataset: $DATASET"
echo "  Epochs: $EPOCHS"
echo "  Donor points: $DONOR_POINTS"
echo "  Recipient points: $RECIPIENT_POINTS"
echo "  Dimensions: $DIMS"
echo ""

cd "$(dirname "$0")"

# Ensure output directory exists
mkdir -p "$DATA_DIR"

# Step 1: Generate donor lottery subspace
echo "=============================================================="
echo "STEP 1: Generating donor lottery subspace..."
echo "=============================================================="

python lottery_subspace.py \
    --model $MODEL \
    --dataset $DATASET \
    --epochs $EPOCHS \
    --points_to_collect $DONOR_POINTS \
    --ds_to_explore $DIMS \
    --opt_alg Adam \
    --lr 0.05 \
    --output_dir $DATA_DIR

# Find the donor file (most recent grad00.pkl)
DONOR_FILE=$(ls -t $DATA_DIR/artifact_lottery_subspace_${MODEL}_${DATASET}_*_grad00.pkl 2>/dev/null | head -1)

if [[ -z "$DONOR_FILE" ]]; then
    echo "ERROR: Could not find donor subspace file!"
    echo "Looking for: $DATA_DIR/artifact_lottery_subspace_${MODEL}_${DATASET}_*_grad00.pkl"
    exit 1
fi

echo ""
echo "Donor subspace file: $DONOR_FILE"
echo ""

# Step 2: Run transfer experiment (use donor's subspace)
echo "=============================================================="
echo "STEP 2: Running TRANSFER experiment..."
echo "        (Training with donor's lottery subspace)"
echo "=============================================================="

python lottery_subspace_transfer.py \
    --model $MODEL \
    --dataset $DATASET \
    --epochs $EPOCHS \
    --points_to_collect $RECIPIENT_POINTS \
    --ds_to_explore $DIMS \
    --opt_alg Adam \
    --lr 0.05 \
    --transfer_from "$DONOR_FILE" \
    --seed_offset 100

# Step 3: Run baseline experiment (each seed uses own subspace)
echo ""
echo "=============================================================="
echo "STEP 3: Running BASELINE experiment..."
echo "        (Each seed uses its own lottery subspace)"
echo "=============================================================="

python lottery_subspace_transfer.py \
    --model $MODEL \
    --dataset $DATASET \
    --epochs $EPOCHS \
    --points_to_collect $RECIPIENT_POINTS \
    --ds_to_explore $DIMS \
    --opt_alg Adam \
    --lr 0.05 \
    --seed_offset 100

# Step 4: Generate plots
echo ""
echo "=============================================================="
echo "STEP 4: Generating comparison plots..."
echo "=============================================================="

python plot_transfer_experiment.py \
    --data_dir $DATA_DIR \
    --save_dir figures/

echo ""
echo "=============================================================="
echo "EXPERIMENT COMPLETE!"
echo "=============================================================="
echo ""
echo "Results saved to: $DATA_DIR/"
echo "Figures saved to: figures/"
echo ""
echo "Key output files:"
ls -la $DATA_DIR/*transfer*_results.pkl 2>/dev/null | tail -2 || echo "  (results files)"
ls -la figures/fig_transfer*.png 2>/dev/null || echo "  (figure files)"
echo ""
echo "To view results:"
echo "  python plot_transfer_experiment.py --data_dir $DATA_DIR --show"
echo ""

