#!/bin/bash
#
# Minimal Reproduction Script for Conv-2 Results (MNIST only)
# "How Many Degrees of Freedom Do We Need to Train Deep Networks"
#

set -e

source $(conda info --base)/etc/profile.d/conda.sh
conda activate dof

cd /home/andrew/dev/dof/degrees-of-freedom

# GPU Memory Settings
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

MODEL="TinyCNN"
DATASET="MNIST"

# Reduced settings for faster runs
EPOCHS=5           # Down from 10
POINTS=2           # Down from 5 (still enough for error bars)

# Key dimensions only - skip tiny ones that obviously fail
DIMS="8 32 128 512 1024 2048 4096"

# Burn-in iterations
BURN_IN_ITERS="4 16"

echo "=============================================="
echo "Conv-2 FAST Reproduction (MNIST only)"
echo "Model: $MODEL | Dataset: $DATASET"
echo "Epochs: $EPOCHS | Points: $POINTS"
echo "Dims: $DIMS"
echo "=============================================="

# =============================================
# Experiment 1: Random Subspace (Figure 2, 4)
# =============================================
echo ""
echo "[1/4] Random Subspace Training..."
python burn_in_subspace.py \
    --model $MODEL \
    --dataset $DATASET \
    --epochs $EPOCHS \
    --points_to_collect $POINTS \
    --init_iters 0 \
    --ds_to_explore $DIMS \
    --lr 0.05 \
    --opt_alg Adam

# =============================================
# Experiment 2: Burn-in Subspace (Figure 3)
# Only 3 values: 0, 64, 256
# =============================================
echo ""
echo "[2/4] Burn-in Subspace Training..."
for INIT in $BURN_IN_ITERS; do
    echo "  Burn-in iterations: $INIT"
    python burn_in_subspace.py \
        --model $MODEL \
        --dataset $DATASET \
        --epochs $EPOCHS \
        --points_to_collect $POINTS \
        --init_iters $INIT \
        --ds_to_explore $DIMS \
        --lr 0.05 \
        --opt_alg Adam
done

# =============================================
# Experiment 3: Lottery Subspace (Figure 5)
# =============================================
echo ""
echo "[3/4] Lottery Subspace Training..."
python lottery_subspace.py \
    --model $MODEL \
    --dataset $DATASET \
    --epochs $EPOCHS \
    --points_to_collect $POINTS \
    --ds_to_explore $DIMS \
    --traj_steps 5 \
    --lr 0.05 \
    --opt_alg Adam

# =============================================
# Experiment 4: Lottery Ticket (Figure 5)
# =============================================
echo ""
echo "[4/4] Lottery Ticket Pruning..."
python lottery_ticket.py \
    --model $MODEL \
    --dataset $DATASET \
    --epochs $EPOCHS \
    --points_to_collect $POINTS

echo ""
echo "=============================================="
echo "Done! Results in: ../lottery-subspace-data/"
echo "Run: python plot_results.py --save_dir ./figures/"
echo "=============================================="
